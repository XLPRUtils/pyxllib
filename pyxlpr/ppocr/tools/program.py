# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import platform
import yaml
import time
import paddle
import paddle.distributed as dist
from tqdm import tqdm
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from pyxlpr.ppocr.utils.stats import TrainingStats
from pyxlpr.ppocr.utils.save_load import save_model
from pyxlpr.ppocr.utils.utility import print_dict
from pyxlpr.ppocr.utils.logging import get_logger
from pyxlpr.ppocr.utils import profiler
from pyxlpr.ppocr.data import build_dataloader


class ArgsParser(ArgumentParser):
    def __init__(self):
        """ 这是pp自定义的一个命令行参数解释器 """

        ''' RawDescriptionHelpFormatter
        
        formatter_class：重置 help 信息输出的格式，可供选择的参数有：
        HelpFormatter、ArgumentDefaultsHelpFormatter、RawDescriptionHelpFormatter、RawTextHelpFormatter
        详见 Python 模块简介-argparse: https://mp.weixin.qq.com/s/s49awBykc7pFEV4XnFNO6g

        默认是HelpFormatter，应该是argparse提供的另一种使用提示吧。
        使用--help，获得的好像也是正常提示，没啥区别
        报错的情况我也测试了下，目前发现不了跟HelpFormatter有啥区别，先不管了。
        '''
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)

        self.add_argument("-c", "--config", help="configuration file to use")

        # argparse的nargs用法:https://docs.python.org/3/library/argparse.html?highlight=argparse%20nargs#nargs
        # +表示使用-o时，至少要提供1个参数值，也可以有多个值，但不能为空。进入内存后会组织为list对象。
        self.add_argument(
            "-o", "--opt", nargs='+', help="set configuration options")
        self.add_argument(
            '-p',
            '--profiler_options',
            type=str,
            default=None,
            help='The option of profiler, which should be in format \"key1=value1;key2=value2;key3=value3\".'
        )

    def parse_args(self, argv=None):
        """ 注意执行parse_args时，这里重载了 """
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        """ 把list格式的opt值，重新设计为字典格式
        """
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config


class AttrDict(dict):
    """Single level attribute dict, NOT recursive

    AttrDict就是个普通的字典类，没啥特别的
    """

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))


# 定义了一个全局配置字典
global_config = AttrDict()

default_config = {'Global': {'debug': False, }}


def load_config(file_path):
    """ 解析传入的yaml配置文件
    把配置文件的参数合并到全局配置，函数返回值也是全局配置

    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    merge_config(default_config)
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    merge_config(yaml.load(open(file_path, 'rb'), Loader=yaml.Loader))
    return global_config


def merge_config(config):
    """ 可以递归，把配置更新合并到全局配置中

    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in config.items():
        if "." not in key:
            if isinstance(value, dict) and key in global_config:
                global_config[key].update(value)
            else:
                global_config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                    sub_keys[0] in global_config
            ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                global_config.keys(), sub_keys[0])
            cur = global_config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]


def check_gpu(use_gpu):
    """
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu version.
    """
    err = "Config use_gpu cannot be set as true while you are " \
          "using paddlepaddle cpu version ! \nPlease try: \n" \
          "\t1. Install paddlepaddle-gpu to run model on GPU \n" \
          "\t2. Set use_gpu as false in config file to run " \
          "model on CPU"

    try:
        if use_gpu and not paddle.is_compiled_with_cuda():
            print(err)
            sys.exit(1)
    except Exception as e:
        pass


def train(config,
          train_dataloader,
          valid_dataloader,
          device,
          model,
          loss_class,
          optimizer,
          lr_scheduler,
          post_process_class,
          eval_class,
          pre_best_model_dict,
          logger,
          vdl_writer=None,
          scaler=None):
    cal_metric_during_train = config['Global'].get('cal_metric_during_train',
                                                   False)
    log_smooth_window = config['Global']['log_smooth_window']
    epoch_num = config['Global']['epoch_num']
    print_batch_step = config['Global']['print_batch_step']
    eval_batch_step = config['Global']['eval_batch_step']
    profiler_options = config['profiler_options']

    global_step = 0
    if 'global_step' in pre_best_model_dict:
        global_step = pre_best_model_dict['global_step']
    start_eval_step = 0
    if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
        start_eval_step = eval_batch_step[0]
        eval_batch_step = eval_batch_step[1]
        if len(valid_dataloader) == 0:
            logger.info(
                'No Images in eval dataset, evaluation during training will be disabled'
            )
            start_eval_step = 1e111
        logger.info(
            "During the training process, after the {}th iteration, an evaluation is run every {} iterations".
                format(start_eval_step, eval_batch_step))
    save_epoch_step = config['Global']['save_epoch_step']
    save_model_dir = config['Global']['save_model_dir']
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    main_indicator = eval_class.main_indicator
    best_model_dict = {main_indicator: 0}
    best_model_dict.update(pre_best_model_dict)
    train_stats = TrainingStats(log_smooth_window, ['lr'])
    model_average = False
    model.train()

    use_srn = config['Architecture']['algorithm'] == "SRN"
    extra_input = config['Architecture'][
                      'algorithm'] in ["SRN", "NRTR", "SAR", "SEED"]
    try:
        model_type = config['Architecture']['model_type']
    except:
        model_type = None
    algorithm = config['Architecture']['algorithm']

    if 'start_epoch' in best_model_dict:
        start_epoch = best_model_dict['start_epoch']
    else:
        start_epoch = 1

    for epoch in range(start_epoch, epoch_num + 1):
        # 每轮都会重新构建一次数据
        train_dataloader = build_dataloader(
            config, 'Train', device, logger, seed=epoch)
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        max_iter = len(train_dataloader) - 1 if platform.system(
        ) == "Windows" else len(train_dataloader)
        for idx, batch in enumerate(train_dataloader):
            profiler.add_profiler_step(profiler_options)
            train_reader_cost += time.time() - reader_start
            if idx >= max_iter:
                break
            lr = optimizer.get_lr()
            images = batch[0]
            if use_srn:
                model_average = True

            train_start = time.time()
            # use amp
            if scaler:
                with paddle.amp.auto_cast():
                    if model_type == 'table' or extra_input:
                        preds = model(images, data=batch[1:])
                    else:
                        preds = model(images)
            else:
                if model_type == 'table' or extra_input:
                    preds = model(images, data=batch[1:])
                elif model_type == "kie":
                    preds = model(batch)
                else:
                    preds = model(images)
            loss = loss_class(preds, batch)
            avg_loss = loss['loss']

            if scaler:
                scaled_avg_loss = scaler.scale(avg_loss)
                scaled_avg_loss.backward()
                scaler.minimize(optimizer, scaled_avg_loss)
            else:
                avg_loss.backward()
                optimizer.step()
            optimizer.clear_grad()

            train_run_cost += time.time() - train_start
            total_samples += len(images)

            if not isinstance(lr_scheduler, float):
                lr_scheduler.step()

            # logger and visualdl
            stats = {k: v.numpy().mean() for k, v in loss.items()}
            stats['lr'] = lr
            train_stats.update(stats)

            if cal_metric_during_train and (model_type != "det"):  # only rec and cls need
                batch = [item.numpy() for item in batch]
                if model_type in ['table', 'kie']:
                    eval_class(preds, batch)
                else:
                    post_result = post_process_class(preds, batch[1])
                    eval_class(post_result, batch)
                metric = eval_class.get_metric()
                train_stats.update(metric)

            if vdl_writer is not None and dist.get_rank() == 0:
                for k, v in train_stats.get().items():
                    vdl_writer.add_scalar('TRAIN/{}'.format(k), v, global_step)
                vdl_writer.add_scalar('TRAIN/lr', lr, global_step)

            if dist.get_rank() == 0 and (
                    (global_step > 0 and global_step % print_batch_step == 0) or
                    (idx >= len(train_dataloader) - 1)):
                logs = train_stats.log()
                strs = 'epoch: [{}/{}], iter: {}, {}, reader_cost: {:.5f} s, batch_cost: {:.5f} s, samples: {}, ips: {:.5f}'.format(
                    epoch, epoch_num, global_step, logs, train_reader_cost /
                                                         print_batch_step, (train_reader_cost + train_run_cost) /
                                                         print_batch_step, total_samples,
                                                         total_samples / (train_reader_cost + train_run_cost))
                logger.info(strs)
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            # eval
            if global_step > start_eval_step and \
                    (global_step - start_eval_step) % eval_batch_step == 0 and dist.get_rank() == 0:
                if model_average:
                    Model_Average = paddle.incubate.optimizer.ModelAverage(
                        0.15,
                        parameters=model.parameters(),
                        min_average_window=10000,
                        max_average_window=15625)
                    Model_Average.apply()
                cur_metric = eval(
                    model,
                    valid_dataloader,
                    post_process_class,
                    eval_class,
                    model_type,
                    extra_input=extra_input)
                cur_metric_str = 'cur metric, {}'.format(', '.join(
                    ['{}: {}'.format(k, v) for k, v in cur_metric.items()]))
                logger.info(cur_metric_str)

                # logger metric
                if vdl_writer is not None:
                    for k, v in cur_metric.items():
                        if isinstance(v, (float, int)):
                            vdl_writer.add_scalar('EVAL/{}'.format(k),
                                                  cur_metric[k], global_step)
                if cur_metric[main_indicator] >= best_model_dict[
                    main_indicator]:
                    best_model_dict.update(cur_metric)
                    best_model_dict['best_epoch'] = epoch
                    save_model(
                        model,
                        optimizer,
                        save_model_dir,
                        logger,
                        is_best=True,
                        prefix='best_accuracy',
                        best_model_dict=best_model_dict,
                        epoch=epoch,
                        global_step=global_step)
                best_str = 'best metric, {}'.format(', '.join([
                    '{}: {}'.format(k, v) for k, v in best_model_dict.items()
                ]))
                logger.info(best_str)
                # logger best metric
                if vdl_writer is not None:
                    vdl_writer.add_scalar('EVAL/best_{}'.format(main_indicator),
                                          best_model_dict[main_indicator],
                                          global_step)
            global_step += 1
            optimizer.clear_grad()
            reader_start = time.time()
        if dist.get_rank() == 0:
            save_model(
                model,
                optimizer,
                save_model_dir,
                logger,
                is_best=False,
                prefix='latest',
                best_model_dict=best_model_dict,
                epoch=epoch,
                global_step=global_step)
        if dist.get_rank() == 0 and epoch > 0 and epoch % save_epoch_step == 0:
            save_model(
                model,
                optimizer,
                save_model_dir,
                logger,
                is_best=False,
                prefix='iter_epoch_{}'.format(epoch),
                best_model_dict=best_model_dict,
                epoch=epoch,
                global_step=global_step)
    best_str = 'best metric, {}'.format(', '.join(
        ['{}: {}'.format(k, v) for k, v in best_model_dict.items()]))
    logger.info(best_str)
    if dist.get_rank() == 0 and vdl_writer is not None:
        vdl_writer.close()
    return


def eval(model,
         valid_dataloader,
         post_process_class,
         eval_class,
         model_type=None,
         extra_input=False):
    model.eval()
    with paddle.no_grad():
        total_frame = 0.0
        total_time = 0.0
        pbar = tqdm(
            total=len(valid_dataloader),
            desc='eval model:',
            position=0,
            leave=True)
        max_iter = len(valid_dataloader) - 1 if platform.system(
        ) == "Windows" else len(valid_dataloader)
        for idx, batch in enumerate(valid_dataloader):
            # if idx >= max_iter:
            #     break
            images = batch[0]
            start = time.time()
            if model_type == 'table' or extra_input:
                preds = model(images, data=batch[1:])
            elif model_type == "kie":
                preds = model(batch)
            else:
                preds = model(images)
            batch = [item.numpy() for item in batch]
            # Obtain usable results from post-processing methods
            total_time += time.time() - start
            # Evaluate the results of the current batch
            if model_type in ['table', 'kie']:
                eval_class(preds, batch)
            else:
                post_result = post_process_class(preds, batch[1])
                # print(post_result)
                eval_class(post_result, batch)

            pbar.update(1)
            total_frame += len(images)
        # Get final metric，eg. acc or hmean
        metric = eval_class.get_metric()

    pbar.close()
    model.train()
    metric['total_frame'] = int(total_frame)
    metric['fps'] = total_frame / total_time
    return metric


def update_center(char_center, post_result, preds):
    result, label = post_result
    feats, logits = preds
    logits = paddle.argmax(logits, axis=-1)
    feats = feats.numpy()
    logits = logits.numpy()

    for idx_sample in range(len(label)):
        if result[idx_sample][0] == label[idx_sample][0]:
            feat = feats[idx_sample]
            logit = logits[idx_sample]
            for idx_time in range(len(logit)):
                index = logit[idx_time]
                if index in char_center.keys():
                    char_center[index][0] = (
                                                    char_center[index][0] * char_center[index][1] +
                                                    feat[idx_time]) / (char_center[index][1] + 1)
                    char_center[index][1] += 1
                else:
                    char_center[index] = [feat[idx_time], 1]
    return char_center


def get_center(model, eval_dataloader, post_process_class):
    pbar = tqdm(total=len(eval_dataloader), desc='get center:')
    max_iter = len(eval_dataloader) - 1 if platform.system(
    ) == "Windows" else len(eval_dataloader)
    char_center = dict()
    for idx, batch in enumerate(eval_dataloader):
        if idx >= max_iter:
            break
        images = batch[0]
        start = time.time()
        preds = model(images)

        batch = [item.numpy() for item in batch]
        # Obtain usable results from post-processing methods
        post_result = post_process_class(preds, batch[1])

        # update char_center
        char_center = update_center(char_center, post_result, preds)
        pbar.update(1)

    pbar.close()
    for key in char_center.keys():
        char_center[key] = char_center[key][0]
    return char_center


def preprocess(is_train=False, *, use_visualdl=True, from_dict=None):
    """ 用于获取配置、设备、日志、visualdl相关对象工具

    :param use_visualdl: 除了检查配置文件是否开启vdl，这个参数同时也为True时才会开启
        在有时候需要preprocess获得前三者，但并不需要重复开一个vdl时使用
    """

    # 1 config
    if from_dict:
        config = global_config
        merge_config(default_config)
        merge_config(from_dict)
        profile_dic = {"profiler_options": None}
    else:
        # global_config/config <-- default_config + 配置文件 FLAGS.config + 命令行参数 FLAGS.opt
        FLAGS = ArgsParser().parse_args()
        profiler_options = FLAGS.profiler_options
        config = load_config(FLAGS.config)  # 返回的是全局变量global_config
        # 可以递归，把配置（这里是命令行参数）更新合并到全局配置中
        merge_config(FLAGS.opt)  # 该函数会修改全局变量，所以会修改config的值
        profile_dic = {"profiler_options": FLAGS.profiler_options}
    merge_config(profile_dic)

    ''' pp处理跟d2有点区别。d2底层默认配置了很复杂的一套默认参数值。
    pp则几乎什么都没有，只有很简洁的一个默认配置，然后叠加配置文件里的参数，再更新命令行设置的参数。
    相比d2的好处，是pp的yaml是纯粹的yaml配置文件，没有任何特殊的依赖要求。
    所以框架里有些必须要获取的结构内容，但很容易自定义扩展各种其他配置参数值。
    
    因为该种设计模式，后面的接口会有对应很多默认值的设置，确保没有传递对应配置时，能run。
    '''

    # 2 logger
    if is_train:
        # 跟is_train有关，如果开启，会在save_model_dir目录下备份一个config.yml配置文件，
        # 并且会把日志记录到train.log文件中。
        # save_config
        save_model_dir = config['Global']['save_model_dir']
        os.makedirs(save_model_dir, exist_ok=True)
        with open(os.path.join(save_model_dir, 'config.yml'), 'w') as f:
            yaml.dump(
                dict(config), f, default_flow_style=False, sort_keys=False)
        log_file = '{}/train.log'.format(save_model_dir)
    else:  # 否则虽然有日志类，但不会把运行记录到文件中
        log_file = None
    logger = get_logger(name='root', log_file=log_file)

    # 3 device
    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global']['use_gpu']
    check_gpu(use_gpu)  # 在使用gpu时会检查cuda是否可用

    # 检查是否在所支持的算法组件里，自己应该可以通过后续框架的学习，扩展自己的算法组件。
    # 需要的话，自己可以把这些算法论文都搜出来，学习一遍。
    alg = config['Architecture']['algorithm']
    assert alg in [
        'EAST', 'DB', 'SAST', 'Rosetta', 'CRNN', 'STARNet', 'RARE', 'SRN',
        'CLS', 'PGNet', 'Distillation', 'NRTR', 'TableAttn', 'SAR', 'PSE',
        'SEED', 'SDMGR'
    ]
    windows_not_support_list = ['PSE']
    if platform.system() == "Windows" and alg in windows_not_support_list:
        logger.warning('{} is not support in Windows now'.format(
            windows_not_support_list))
        sys.exit()

    # dist.ParallelEnv().dev_id不太清楚作用，看文档也推荐不直接使用这个接口。
    #   我测试了下，虽然0卡有在用，默认还是返回0，总之不是啥智能判断获得空余显卡这种功能~
    #   简单来说，就是设置了device，细节我也先不用太纠结。
    # 应该是跟分布式有关，在分布式的时候，这里才会有些区别。
    #   默认单卡情况，第14行获得的dist.get_world_size()也是1。
    #   第14行会自动标记一个是否使用分布式训练的参数distributed。
    device = 'gpu:{}'.format(dist.ParallelEnv().dev_id) if use_gpu else 'cpu'
    device = paddle.set_device(device)

    config['Global']['distributed'] = dist.get_world_size() != 1

    # 4 vdl_write，如果开启了可视化功能
    # 在save_model_dir目录下，会再建立一个vdl目录，返回一个vdl_writer对象
    if config['Global']['use_visualdl'] and use_visualdl:
        from visualdl import LogWriter
        save_model_dir = config['Global']['save_model_dir']
        vdl_writer_path = '{}/vdl/'.format(save_model_dir)
        os.makedirs(vdl_writer_path, exist_ok=True)
        vdl_writer = LogWriter(logdir=vdl_writer_path)
    else:
        vdl_writer = None

    # 用logger输出config的内容
    print_dict(config, logger)
    logger.info('train with paddle {} and device {}'.format(paddle.__version__,
                                                            device))
    return config, device, logger, vdl_writer
