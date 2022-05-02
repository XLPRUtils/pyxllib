# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import os
import random
import traceback
from paddle.io import Dataset
from .imaug import transform, create_operators

import json

from pyxllib.xl import run_once, XlPath

__all__ = ['SimpleDataSet', 'XlSimpleDataSet']


class SimpleDataSet(Dataset):
    """ paddleocr 源生的基础数据格式
    每张图的标注压缩在一个总的txt文件里
    """

    def __init__(self, config, mode, logger, seed=None):
        super(SimpleDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()

        # 这里能取到全局的配置信息
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        # label文件里图片路径和json之间的分隔符
        self.delimiter = dataset_config.get('delimiter', '\t')

        label_file_list = dataset_config.pop('label_file_list')
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get("ratio_list", [1.0])
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert len(
            ratio_list
        ) == data_source_num, "The length of ratio_list should be the same as the file_list."
        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']

        self.seed = seed
        logger.info("Initialize indexs of datasets:%s" % label_file_list)
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        # shuffle好像在dataset和dataloader会重复操作，虽然其实也没什么事~~
        if self.mode == "train" and self.do_shuffle:
            self.shuffle_data_random()
        # （读取图片）数据增广操作器
        self.ops = create_operators(dataset_config['transforms'], global_config)

    def get_image_info_list(self, file_list, ratio_list):
        """ 从多个文件按比例随机抽样获取样本 """
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                if self.mode == "train" or ratio_list[idx] < 1.0:
                    random.seed(self.seed)
                    lines = random.sample(lines,
                                          round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines

    def shuffle_data_random(self):
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    def get_ext_data(self):
        """ 是否要添加其他图片及数量

        猜测是用于mixup、cropmix等场合的数据增广，除了当前图，能随机获取其他来源图片，做综合处理
        其它来源的图，会调用前2个opts，读取图片，解析标签
        """
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, 'ext_data_num'):
                ext_data_num = getattr(op, 'ext_data_num')
                break
        load_data_ops = self.ops[:2]
        ext_data = []

        while len(ext_data) < ext_data_num:
            # 随机从中抽一个样本
            file_idx = self.data_idx_order_list[np.random.randint(self.__len__(
            ))]
            data_line = self.data_lines[file_idx]
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                continue
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            data = transform(data, load_data_ops)

            if data is None or data['polys'].shape[1]!=4:
                continue
            ext_data.append(data)
        return ext_data

    def __getitem__(self, idx):
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        # print(data_line.split()[0])
        try:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            # 无论是图片还是任何文件，都是统一先读成bytes了，然后交由transform的配置实现。
            #	其中DecodeImage又可以解析读取图片数据。
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            data['ext_data'] = self.get_ext_data()
            if data['label'] == '[]':  # 没有文本的图片
                data['label'] = '[{"transcription": "###", "points": [[0, 0], [1, 0], [1, 1], [0, 1]]}]'
            outs = transform(data, self.ops)
        except Exception as e:
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, e))
            outs = None
        if outs is None:
            # 如果遇到解析出错的数据，训练阶段会随机取另一个图片代替。
            #	eval阶段则直接取下一张有效图片代替。
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = np.random.randint(self.__len__(
            )) if self.mode == "train" else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)
        return outs

    def __len__(self):
        return len(self.data_idx_order_list)


class SimpleDataSetExt(SimpleDataSet):
    """ 自定义的数据结构类，支持输入标注文件所在目录来初始化

    这里的 __init__、get_image_info_list 设计了一套特殊的输入范式
    """

    def __init__(self, config, mode, logger, seed=None):
        self.logger = logger
        self.mode = mode.lower()

        # 这里能取到全局的配置信息
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        # label文件里图片路径和json之间的分隔符
        self.delimiter = dataset_config.get('delimiter', '\t')

        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']

        self.seed = seed
        data_list = dataset_config.get('data_list', [])
        logger.info("Initialize indexs of datasets:%s" % data_list)
        self.data_lines = self.get_image_info_list(data_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        # shuffle好像在dataset和dataloader会重复操作，虽然其实也没什么事~~
        if self.mode == "train" and self.do_shuffle:
            self.shuffle_data_random()
        # （读取图片）数据增广操作器
        self.ops = create_operators(dataset_config['transforms'], global_config)

    def get_image_info_list(self, data_list):
        """ 从标注文件所在目录获取每张图的标注信息

        :return: list data_lines，每一行有两列，第1列是图片数据相对data_dir的路径，\t隔开，第2列是json标注数据
        """
        if isinstance(data_list, dict):
            data_list = [data_list]

        data_lines = []

        for idx, cfg in enumerate(data_list):
            add_lines = [x.encode('utf8') for x in self.get_image_info(cfg)]
            data_lines.extend(add_lines)

        return data_lines

    def get_image_info(self, cfg):
        raise NotImplementedError


class XlSimpleDataSet(SimpleDataSetExt):
    """ 支持直接配置原始的icdar2015数据格式
    """

    def get_image_info(self, cfg):
        # 我的XlSimpleDataSet，type是必填字段
        t = cfg.pop('type') if ('type' in cfg) else ''
        func = getattr(self, 'from_' + t, None)
        if func:
            return func(**cfg)
        else:
            raise TypeError('指定数据集格式不存在')

    def __repr__(self):
        """ 跟runonce有关，需要用这个构造类字符串，判断参数是否重复 """
        args = [self.data_dir, self.seed]
        return 'XlSimpleDataSet(' + ','.join(map(str, args)) + ')'

    def _select_ratio(self, data, ratio):
        """ 随机筛选给定的data数组数据

        方便有些数据没有物理地址划分训练、验证集的，可以通过ratio直接设置，目前只支持一个数值ratio，后续也可以考虑支持list，更灵活的截选策略
        """
        random.seed(4101)  # 我这里是要用固定策略拆分数据，不用self.seed
        random.shuffle(data)
        n = len(data)
        if isinstance(ratio, float):
            if ratio > 0:
                data = data[:int(ratio * n)]
            elif ratio < 0:
                data = data[int(ratio * n):]
        elif isinstance(ratio, list):
            left, right = ratio  # 这个ratio是每个类别的模板分开处理的
            data = data[int(left * n):int(right * n)]
        return data

    @run_once('str')  # 这个标注格式是固定的，不用每次重复生成，可以使用run_once限定
    def from_icdar2015(self, subdir, label_dir, ratio=None):
        data_dir = XlPath(self.data_dir)
        subdir = data_dir / subdir
        label_dir = data_dir / label_dir

        data_lines = []

        txt_files = list(label_dir.glob('*.txt'))
        if ratio is not None:
            txt_files = self._select_ratio(txt_files, ratio)

        def label2json(content):
            """ 单个图的content标注内容转为json格式 """
            label = []
            for line in content.splitlines():
                tmp = line.split(',')
                points = tmp[:8]
                s = []
                for i in range(0, len(points), 2):
                    b = points[i:i + 2]
                    b = [int(t) for t in b]
                    s.append(b)
                result = {"transcription": tmp[8], "points": s}
                label.append(result)
            return label

        for f in txt_files:
            # stem[3:]是去掉标注文件名多出的'gt_'的前缀
            impath = (subdir / (f.stem[3:] + '.jpg')).relative_to(data_dir).as_posix()
            # icdar的标注文件，有的是utf8，有的是utf-8-sig，这里使用我的自动识别功能
            json_label = label2json(f.read_text(encoding=None))
            label = json.dumps(json_label, ensure_ascii=False)
            data_lines.append(('\t'.join([impath, label])))

        return data_lines

    @run_once('str')
    def from_refineAgree(self, subdir, json_dir, label_file):
        """ 只需要输入根目录 """
        from pyxlpr.data.labelme import LabelmeDict

        data_dir = XlPath(self.data_dir)
        subdir = data_dir / subdir
        json_dir = data_dir / json_dir
        label_file = data_dir / label_file

        def labelme2json(d):
            """ labelme的json转为paddle的json标注 """
            label = []
            shapes = d['shapes']
            for sp in shapes:
                msg = json.loads(sp['label'])
                if msg['type'] != '印刷体':
                    continue
                result = {"transcription": msg['text'],
                          "points": LabelmeDict.to_quad_pts(sp)}
                label.append(result)
            return label

        data_lines = []
        sample_list = label_file.read_text().splitlines()
        for x in sample_list:
            if not x: continue  # 忽略空行
            impath = (subdir / (x + '.jpg')).relative_to(data_dir).as_posix()
            f = json_dir / (x + '.json')
            json_label = labelme2json(f.read_json())
            label = json.dumps(json_label, ensure_ascii=False)
            data_lines.append(('\t'.join([impath, label])))

        return data_lines

    @run_once('str')
    def from_labelme_det(self, subdir='.', ratio=None, transcription_field='text'):
        """ 读取sub_data_dir目录（含子目录）下所有的json文件为标注文件

        :param transcription_field: 对于检测任务这个值一般没什么用，主要是一些特殊数据，标记"#"的会记为难样本，跳过不检测
            None, 不设置则取sp['label']为文本值
            若设置，则按字典解析label并取对应名称的键值

        json1: labelme的json标注文件
        json2: paddle的SimpleDataSet要传入的json格式
        """
        from pyxlpr.data.labelme import LabelmeDict

        data_dir = XlPath(self.data_dir)
        subdir = data_dir / subdir
        data_lines = []

        def json1_to_json2(d):
            res = []
            shapes = d['shapes']
            for sp in shapes:
                label = sp['label']
                if transcription_field:
                    msg = json.loads(sp['label'])
                    label = msg[transcription_field]
                result = {"transcription": label,
                          "points": LabelmeDict.to_quad_pts(sp)}
                res.append(result)
            return res

        json1_files = list(subdir.rglob('*.json'))
        if ratio is not None:
            json1_files = self._select_ratio(json1_files, ratio)

        for json1_file in json1_files:
            data = json1_file.read_json()
            # 比较简单的检查是否为合法labelme的规则
            if 'imagePath' not in data:
                continue
            img_file = json1_file.parent / data['imagePath']  # 不确定关联的图片格式，所以直接从labelme里取比较准
            json2_data = json1_to_json2(data)
            json2_str = json.dumps(json2_data, ensure_ascii=False)
            data_lines.append(('\t'.join([img_file.relative_to(data_dir).as_posix(), json2_str])))
        return data_lines
