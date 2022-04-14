import logging
import os
from collections import OrderedDict, Counter
import torch
import re

import pandas as pd
import numpy as np

import torch
from torch import nn

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.data import MetadataCatalog
from detectron2.config import configurable
from detectron2.evaluation import DatasetEvaluator

from pyxllib.xl import cache_file, TicToc, File, Dir, get_xllog, is_file, first_nonnone
from pyxllib.ai.torch import NvmDevice, XlPredictor
from pyxllib.data.coco import CocoMatch
from xlproject.kzconfig import register_d2dataset


class D2Launch:
    def eval_only(self, args=None):
        cfg = self.cfg
        args = args or self.args

        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    def train_eval(self, args=None):
        args = args or self.args
        self.train(args)
        self.eval(args)

    def _evalbase(self, cfg, model, model_path):
        """
        :return res: {dict: 11} {'AP': float, 'AP50': float, ...}
        """
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            str(model_path), resume=self.args.resume)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    def eval(self, args=None):
        # 1 参数
        cfg = self.cfg
        args = args or self.args
        model = Trainer.build_model(cfg)

        # 2 生成每个模型预测的统计结果，判断收敛性
        # 2.1 工具函数
        def make_result(f, d):
            print("Eval Model:", f)
            cfg.OUTPUT_DIR = str(d)
            return self._evalbase(cfg, model, f)

        # 2.2 生成待eval的model清单
        lines, eval_dirs = [], []
        if args.eval_model:
            modelfiles = [File(args.eval_model)]
        else:
            modelfiles = Dir().select('*.pth').subfiles()
            # 去重
            if len(modelfiles) > 1 and ('model_final' in str(modelfiles[-1])):
                if cfg.SOLVER.MAX_ITER % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                    # 不操作model_final的前一个重复模型
                    modelfiles = modelfiles[:-2] + modelfiles[-1:]

        # 2.3 生成每个模型在数据上的预测结果
        for f in modelfiles:
            d = Dir(' '.join([f.stem] + list(cfg.DATASETS.TEST)))
            eval_dirs.append(d)
            res = cache_file(File('coco_eval_result.pkl', d), lambda: make_result(f, d))
            d = {k: round(v, 2) for k, v in res['bbox'].items()}
            d['model_file'] = f.name
            lines.append(d)

        # 2.4 输出报告
        df = pd.DataFrame.from_records(lines)
        df = df[['model_file'] + list(df.columns)[:-1]]
        get_xllog().info('0 checkpoints')
        with pd.option_context('display.max_colwidth', -1, 'display.max_columns', 15,
                               'display.width', 200):  # 上下文控制格式
            print(df)

        # 3 统计多种指标结果
        if args.eval_degree > 1:
            valdata = list(cfg.DATASETS.TEST)[0]  # TODO 扩展支持多数据集
            gt = MetadataCatalog.get(valdata).json_file
            dt = eval_dirs[-1] / 'inference/coco_instances_results.json'
            cm = CocoMatch(gt, dt)
            cm.to_excel(File('cocomatch.xlsx', eval_dirs[-1]))
            cm.eval_all()

        # 4 生成可视化的数据
        if args.eval_degree > 2:
            cm.to_labelme_match(MetadataCatalog.get(valdata).image_root, Dir('cocomatch', eval_dirs[-1]))  # noqa

    @classmethod
    def cmdtools(cls):
        """ 通过命令行选择执行的功能
        """
        # with TicToc(__name__):
        #     for mode in ('train', 'eval-only', 'eval'):
        #         mode = mode.replace('-', '_')
        #         if getattr(self.args, mode):
        #             launch(
        #                 getattr(self, mode),
        #                 self.args.num_gpus,
        #                 num_machines=self.args.num_machines,
        #                 machine_rank=self.args.machine_rank,
        #                 dist_url=self.args.dist_url,
        #                 args=(self.args,),
        #             )

        raise NotImplementedError


def get_d2argcfg(cfg=None):
    """ 获得d2环境下的命令行参数args，和配置参数cfg """

    # 1 扩展的命令行参数功能
    def argument_parser():
        parser = default_argument_parser()

        # 自己扩展的 --train 功能
        parser.add_argument("--train", action="store_true", help="enable train")
        # 自己扩展的 --eval 功能
        parser.add_argument("--eval", action="store_true", help="enable eval")
        parser.add_argument("--eval_degree", type=int, default=3,
                            help="eval degree(default=3: maximum eval degree)")
        parser.add_argument("--eval_model", type=str, default='', help="main model for eval")

        return parser

    args = argument_parser().parse_args()

    # print("Command Line Args:", args)

    # 2 自定义的cfg解析规则
    def setup_cfg(args):
        # 获取一个默认配置的拷贝
        cfg_ = get_cfg()
        # 可以使用匿名数据集
        cfg_.MODEL.META_ARCHITECTURE = 'CUSTOM_MODEL'
        cfg_.DATASETS.TRAIN = ('CUSTOM_TRAIN_DATASETS',)
        cfg_.DATASETS.TEST = ('CUSTOM_TEST_DATASETS',)
        cfg_.MODEL.DEVICE = ''

        # 从文件读取配置并更新
        if args.config_file:
            cfg_.merge_from_file(args.config_file)
        elif is_file(cfg):  # 输入了配置文件路径
            cfg_.merge_from_file(cfg)
        elif isinstance(cfg, str):  # 输入了yaml格式的配置字符串
            f = File(..., Dir.TEMP).write(cfg, mode='.txt')
            cfg_.merge_from_file(str(f))
            f.delete()  # 临时文件，用完即删
        # 从命令行的opts读取并更新
        cfg_.merge_from_list(args.opts)
        # cfg_.freeze()  # 是否冻结参数，不可改动
        default_setup(cfg_, args)
        return cfg_

    cfg = setup_cfg(args)

    # 3 自动选cpu、gpu
    # 如果没有指定gpu，且特定参数情况下，则自动检索一个可用的gpu
    if args.num_gpus == 1 and cfg.MODEL.DEVICE == '':
        device = NvmDevice().get_most_free_torch_gpu_device()
        cfg.MODEL.DEVICE = f'cuda:{device.index}'  # 改配置文件没用，还是会用到0卡
        # torch.cuda.set_device(device)  # 用这个限定效果更好

    # 3 个性化配置
    # 工作目录切换到 OUTPUT_DIR
    # os.chdir(cfg.OUTPUT_DIR)

    # 4 需要什么数据，再注册对应的数据
    for name in set(cfg.DATASETS.TRAIN + cfg.DATASETS.TEST):
        register_d2dataset(name, error='ignore')

    return args, cfg


class D2Trainer(DefaultTrainer):
    def __init__(self, cfg=None):
        """
        :param str|filepath|CfgNode cfg: 扩展，支持从文件、字符串内容来初始化
            优先识别命令行--config-file输入的配置文件
            如果命令行未指定，则使用类初始化输入的cfg参数值（可以是配置文件路径，也可以是yaml格式的配置字符串）
            如果cfg也没输入，则使用内置的默认值
        """
        self.args, self.cfg = get_d2argcfg(cfg)
        super().__init__(self.cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def launch(self):
        self.resume_or_load(resume=self.args.resume)
        if self.cfg.TEST.AUG.ENABLED:
            self.register_hooks(
                [hooks.EvalHook(0, lambda: self.test_with_TTA(self.cfg, self.model))]
            )
        return self.train()

    def launch_eval(self):
        pass

    @classmethod
    def build_predictor(cls, cfg=None, state_file=None, device=None, *, batch_size=None):
        args, cfg = get_d2argcfg(cfg)
        device = device or cfg.MODEL.DEVICE
        batch_size = batch_size or cfg.SOLVER.IMS_PER_BATCH

        model = cls.build_model(cfg)

        if state_file is None:
            # 从cfg.OUTPUT_DIR里找checkpoint模型文件
            checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR)
            if checkpointer.has_checkpoint():
                state_file = checkpointer.get_checkpoint_file()
            else:
                raise ValueError('没有state_file文件')

        pred = XlPredictor(model, state_file, device, batch_size=batch_size)
        if hasattr(cls, 'build_transforms'):
            pred.transform = cls.build_transforms()

        return pred


class ClsEval(DatasetEvaluator):
    """ 分类问题的测评函数 """

    def reset(self):
        self.preds = []  # 所有的预测结果

    def process(self, inputs, outputs):
        """
        :param list[torch.tensor, torch.tensor] inputs: dataloader每次batch得到的内容结构
        :param torch.tensor outputs: model.inference对应的返回值
        :return:
        """
        for y, y_hat in zip(inputs[1].tolist(), outputs.tolist()):
            self.preds.append([y, y_hat])

    def evaluate(self):
        # 结果要以字典的形式返回，第一层是任务名，第二层是多个指标结果
        df = pd.DataFrame.from_records(self.preds, columns=['gt', 'dt'])
        print('各类别出现次数（行坐标gt，列坐标dt）：')
        print(pd.crosstab(df['gt'], df['dt']))

        acc = sum(df['gt'] == df['dt'])
        total = len(df)

        return {'Classification': {'Accuracy': acc / total, 'total': total}}


class D2ClsTrainer(D2Trainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return ClsEval()
