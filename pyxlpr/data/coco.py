#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/01/20 14:51

"""
对pycocotools的一些功能整合、封装

代码中，gt指ground truth，真实标注
    dt指detection，模型检测出来的结果

除了 label.py 中定义的
    CocoGtData 专门处理 gt 格式数据
    CocoData 同时处理 gt dt 格式数据
这里对外有两个类
    CocoEval 计算coco指标
    CocoMatch 进行一些高级的结果分析
        生成的结果可以用 xllabelme 打开 （pip install xllabelme）
"""

from pyxllib.prog.pupil import check_install_package

check_install_package('xlcocotools')

from collections import ChainMap, defaultdict, Counter
import copy
import json
import os
import pathlib
import random
import sys

import pandas as pd
import PIL
from tqdm import tqdm

from pyxllib.file.packlib.zipfile import ZipFile
from pyxllib.prog.newbie import round_int
from pyxllib.prog.pupil import DictTool
from pyxllib.prog.specialist import mtqdm
from pyxllib.algo.pupil import Groups, make_index_function, matchpairs
from pyxllib.algo.geo import rect_bounds, rect2polygon, reshape_coords, ltrb2xywh, xywh2ltrb, ComputeIou
from pyxllib.algo.stat import dataframes_to_excel
from pyxllib.file.specialist import File, Dir, PathGroups
from pyxllib.debug.specialist import get_xllog
from pyxlpr.data.icdar import IcdarEval
from pyxlpr.data.labelme import LABEL_COLORMAP7, ToLabelmeJson, LabelmeDataset, LabelmeDict
from xlcocotools.coco import COCO
from xlcocotools.cocoeval import COCOeval


class CocoGtData:
    """ 类coco格式的json数据处理

    不一定要跟coco gt结构完全相同，只要相似就行，
        比如images、annotaions、categories都可以扩展自定义字段
    """

    def __init__(self, gt):
        self.gt_dict = gt if isinstance(gt, dict) else File(gt).read()

    @classmethod
    def gen_image(cls, image_id, file_name, height=None, width=None, **kwargs):
        """ 初始化一个图片标注，使用位置参数，复用的时候可以节省代码量 """

        # 没输入height、width时会自动从file_name读取计算
        # 但千万注意，这里coco的file_name输入的是相对路径，并不一定在工作目录下能work，一般还是推荐自己输入height、width
        if height is None or width is None:
            width, height = PIL.Image.open(str(file_name)).size

        im = {'id': int(image_id), 'file_name': file_name,
              'height': int(height), 'width': int(width)}
        if kwargs:
            im.update(kwargs)
        return im

    @classmethod
    def gen_images(cls, imdir, start_idx=1):
        """ 自动生成标准的images字段

        :param imdir: 图片目录
        :param start_idx: 图片起始下标
        :return: list[dict(id, file_name, width, height)]
        """
        files = Dir(imdir).select_files(['*.jpg', '*.png'])
        images = []
        for i, f in enumerate(files, start=start_idx):
            w, h = Image.open(str(f)).size
            images.append({'id': i, 'file_name': f.name, 'width': w, 'height': h})
        return images

    @classmethod
    def points2segmentation(cls, pts):
        """ labelme的points结构转segmentation分割结构
        """
        # 1 两个点要转4个点
        if len(pts) == 2:
            pts = rect2polygon(pts)
        else:
            pts = list(pts)

        # 2 点集要封闭，末尾要加上第0个点
        pts.append(pts[0])

        # 多边形因为要画出所有的点，还要封闭，数据有点多，还是只存整数节省空间
        pts = [round_int(v) for v in reshape_coords(pts, 1)]

        return pts

    @classmethod
    def gen_annotation(cls, **kwargs):
        """ 智能地生成一个annotation字典

        这个略微有点过度封装了
        但没事，先放着，可以不拿出来用~~

        :param points: 必须是n*2的结构
        """
        a = kwargs.copy()

        # a = {'id': 0, 'area': 0, 'bbox': [0, 0, 0, 0],
        #       'category_id': 1, 'image_id': 0, 'iscrowd': 0, 'segmentation': []}

        if 'points' in a:  # points是一个特殊参数，使用“一个”多边形来标注（注意区别segmentation是多个多边形）
            if 'segmentation' not in a:
                a['segmentation'] = [cls.points2segmentation(a['points'])]
            del a['points']
        if 'bbox' not in a:
            pts = []
            for seg in a['segmentation']:
                pts += seg
            a['bbox'] = ltrb2xywh(rect_bounds(pts))
        if 'area' not in a:  # 自动计算面积
            a['area'] = int(a['bbox'][2] * a['bbox'][3])
        for k in ['id', 'image_id']:
            if k not in a:
                a[k] = 0
        if 'category_id' not in a:
            a['category_id'] = 1
        if 'iscrowd' not in a:
            a['iscrowd'] = 0

        return a

    @classmethod
    def gen_quad_annotations(cls, file, *, image_id, start_box_id, category_id=1, **kwargs):
        """ 解析一张图片对应的txt标注文件

        :param file: 标注文件，有多行标注
            每行是x1,y1,x2,y2,x3,y3,x4,y4[,label] （label可以不存在）
        :param image_id: 该图片id
        :param start_box_id: box_id起始编号
        :param category_id: 归属类别
        """
        lines = File(file).read()
        box_id = start_box_id
        annotations = []
        for line in lines.splitlines():
            vals = line.split(',', maxsplit=8)
            if len(vals) < 2: continue
            attrs = {'id': box_id, 'image_id': image_id, 'category_id': category_id}
            if len(vals) == 9:
                attrs['label'] = vals[8]
            # print(vals)
            seg = [int(v) for v in vals[:8]]
            attrs['segmentation'] = [seg]
            attrs['bbox'] = ltrb2xywh(rect_bounds(seg))
            if kwargs:
                attrs.update(kwargs)
            annotations.append(cls.gen_annotation(**attrs))
            box_id += 1
        return annotations

    @classmethod
    def gen_categories(cls, cats):
        if isinstance(cats, list):
            # 如果输入是一个类别列表清单，则按1、2、3的顺序给其编号
            return [{'id': i, 'name': x, 'supercategory': ''} for i, x in enumerate(cats, start=1)]
        else:
            raise TypeError

        # TODO 扩展支持其他构造方法

    @classmethod
    def gen_gt_dict(cls, images, annotations, categories, outfile=None):
        data = {'images': images, 'annotations': annotations, 'categories': categories}
        if outfile is not None:
            File(outfile).write(data)
        return data

    @classmethod
    def is_gt_dict(cls, gt_dict):
        if isinstance(gt_dict, (tuple, list)):
            return False
        has_keys = set('images annotations categories'.split())
        return not (has_keys - gt_dict.keys())

    def clear_gt_segmentation(self, *, inplace=False):
        """ 有的coco json文件太大，如果只做普通的bbox检测任务，可以把segmentation的值删掉
        """
        gt_dict = self.gt_dict if inplace else copy.deepcopy(self.gt_dict)
        for an in gt_dict['annotations']:
            an['segmentation'] = []
        return gt_dict

    def get_catname_func(self):
        id2name = {x['id']: x['name'] for x in self.gt_dict['categories']}

        def warpper(cat_id, default=...):
            """
            :param cat_id:
            :param default: 没匹配到的默认值
                ... 不是默认值，而是代表匹配不到直接报错
            :return:
            """
            if cat_id in id2name:
                return id2name[cat_id]
            else:
                if default is ...:
                    raise IndexError(f'{cat_id}')
                else:
                    return default

        return warpper

    def _group_base(self, group_anns, reserve_empty=False):
        if reserve_empty:
            for im in self.gt_dict['images']:
                yield im, group_anns.get(im['id'], [])
        else:
            id2im = {im['id']: im for im in self.gt_dict['images']}
            for k, v in group_anns.items():
                yield id2im[k], v

    def group_gt(self, *, reserve_empty=False):
        """ 遍历gt的每一张图片的标注

        这个是用字典的方式来实现分组，没用 df.groupby 的功能

        :param reserve_empty: 是否保留空im对应的结果

        :return: [(im, annos), ...] 每一组是im标注和对应的一组annos标注
        """
        group_anns = defaultdict(list)
        [group_anns[an['image_id']].append(an) for an in self.gt_dict['annotations']]
        return self._group_base(group_anns, reserve_empty)

    def select_gt(self, ids, *, inplace=False):
        """ 删除一些images标注（会删除对应的annotations），挑选数据，或者减小json大小

        :param ids: int类型表示保留的图片id，str类型表示保留的图片名，可以混合使用
            [341427, 'PMC4055390_00006.jpg', ...]
        :return: 筛选出的新字典
        """
        gt_dict = self.gt_dict
        # 1 ids 统一为int类型的id值
        if not isinstance(ids, (list, tuple, set)):
            ids = [ids]
        map_name2id = {item['file_name']: item['id'] for item in gt_dict['images']}
        ids = set([(map_name2id[x] if isinstance(x, str) else x) for x in ids])

        # 2 简化images和annotations
        dst = {'images': [x for x in gt_dict['images'] if (x['id'] in ids)],
               'annotations': [x for x in gt_dict['annotations'] if (x['image_id'] in ids)],
               'categories': gt_dict['categories']}
        if inplace: self.gt_dict = dst
        return dst

    def random_select_gt(self, number=20, *, inplace=False):
        """ 从gt中随机抽出number个数据 """
        ids = [x['id'] for x in self.gt_dict['images']]
        random.shuffle(ids)
        gt_dict = self.select_gt(ids[:number])
        if inplace: self.gt_dict = gt_dict
        return gt_dict

    def select_gt_by_imdir(self, imdir, *, inplace=False):
        """ 基于imdir目录下的图片来过滤src_json """
        # 1 对比下差异
        json_images = set([x['file_name'] for x in self.gt_dict['images']])
        dir_images = set(os.listdir(str(imdir)))

        # df = SetCmper({'json_images': json_images, 'dir_images': dir_images}).intersection()
        # print('json_images intersection dir_images:')
        # print(df)

        # 2 精简json
        gt_dict = self.select_gt(json_images & dir_images)
        if inplace: self.gt_dict = gt_dict
        return gt_dict

    def reset_image_id(self, start=1, *, inplace=False):
        """ 按images顺序对图片重编号 """
        gt_dict = self.gt_dict if inplace else copy.deepcopy(self.gt_dict)
        # 1 重置 images 的 id
        old2new = {}
        for i, im in enumerate(gt_dict['images'], start=start):
            old2new[im['id']] = i
            im['id'] = i

        # 2 重置 annotations 的 id
        for anno in gt_dict['annotations']:
            anno['image_id'] = old2new[anno['image_id']]

        return gt_dict

    def reset_box_id(self, start=1, *, inplace=False):
        anns = self.gt_dict['annotations']
        if not inplace:
            anns = copy.deepcopy(anns)

        for i, anno in enumerate(anns, start=start):
            anno['id'] = i
        return anns

    def to_labelme_cls(self, root, *, bbox=True, seg=False, info=False):
        """
        :param root: 图片根目录
        :return:
            extdata，存储了一些匹配异常信息
        """
        root, data = Dir(root), {}
        catid2name = {x['id']: x['name'] for x in self.gt_dict['categories']}

        # 1 准备工作，构建文件名索引字典
        gs = PathGroups.groupby(root.select_files('**/*'))

        # 2 遍历生成labelme数据
        not_finds = set()  # coco里有的图片，root里没有找到
        multimatch = dict()  # coco里的某张图片，在root找到多个匹配文件
        for img, anns in tqdm(self.group_gt(reserve_empty=True), disable=not info):
            # 2.1 文件匹配
            imfiles = gs.find_files(img['file_name'])
            if not imfiles:  # 没有匹配图片的，不处理
                not_finds.add(img['file_name'])
                continue
            elif len(imfiles) > 1:
                multimatch[img['file_name']] = imfiles
                imfile = imfiles[0]
            else:
                imfile = imfiles[0]

            # 2.2 数据内容转换
            lmdict = LabelmeDict.gen_data(imfile)
            img = DictTool.or_(img, {'xltype': 'image'})
            lmdict['shapes'].append(LabelmeDict.gen_shape(json.dumps(img, ensure_ascii=False), [[-10, 0], [-5, 0]]))
            for ann in anns:
                if bbox:
                    ann = DictTool.or_(ann, {'category_name': catid2name[ann['category_id']]})
                    label = json.dumps(ann, ensure_ascii=False)
                    shape = LabelmeDict.gen_shape(label, xywh2ltrb(ann['bbox']))
                    lmdict['shapes'].append(shape)

                if seg:
                    # 把分割也显示出来（用灰色）
                    for x in ann['segmentation']:
                        an = {'box_id': ann['id'], 'xltype': 'seg', 'shape_color': [191, 191, 191]}
                        label = json.dumps(an, ensure_ascii=False)
                        lmdict['shapes'].append(LabelmeDict.gen_shape(label, x))

            f = imfile.with_suffix('.json')

            data[f.relpath(root)] = lmdict

        return LabelmeDataset(root, data,
                              extdata={'categories': self.gt_dict['categories'],
                                       'not_finds': not_finds,
                                       'multimatch': Groups(multimatch)})

    def to_labelme(self, root, *, bbox=True, seg=False, info=False):
        self.to_labelme_cls(root, bbox=bbox, seg=seg, info=info).writes()

    def split_data(self, parts, *, shuffle=True):
        """ 数据拆分器

        :param dict parts: 每个部分要拆分、写入的文件名，以及数据比例
            py≥3.6的版本中，dict的key是有序的，会按顺序处理开发者输入的清单
            这里比例求和可以不满1，但不能超过1
        :param bool shuffle: 是否打乱原有images顺序
        :return: 同parts的字典，但值变成了拆分后的coco数据
        """
        # 1 读入data
        assert sum(parts.values()) <= 1, '比例和不能超过1'
        data = self.gt_dict
        if shuffle:
            data = data.copy()
            data['images'] = data['images'].copy()
            random.shuffle(data['images'])

        # 2 生成每一个部分的文件
        def select_annotations(annotations, image_ids):
            # 简单的for循环和if操作，可以用“列表推导式”写
            return [an for an in annotations if (an['image_id'] in image_ids)]

        res = {}
        total_num, used_rate = len(data['images']), 0
        for k, v in parts.items():
            # 2.1 选择子集图片
            images = data['images'][int(used_rate * total_num):int((used_rate + v) * total_num)]
            image_ids = {im['id'] for im in images}

            # 2.2 生成新的字典
            res[k] = {'images': images,
                      'annotations': select_annotations(data['annotations'], image_ids),
                      'categories': data['categories']}

            # 2.4 更新使用率
            used_rate += v
        return res


class CocoData(CocoGtData):
    """ 这个类可以封装一些需要gt和dt衔接的功能 """

    def __init__(self, gt, dt=None, *, min_score=0):
        """
        :param gt: gt的dict或文件
            gt是必须传入的，可以只传入gt
            有些任务理论上可以只有dt，但把配套的gt传入，能做更多事
        :param dt: dt的list或文件
        :param min_score: CocoMatch这个系列的类，初始化增加min_score参数，支持直接滤除dt低置信度的框
        """
        super().__init__(gt)

        def get_dt_list(dt, min_score=0):
            # dt
            default_dt = []
            # default_dt = [{'image_id': self.gt_dict['images'][0]['id'],
            #                'category_id': self.gt_dict['categories'][0]['id'],
            #                'bbox': [0, 0, 1, 1],
            #                'score': 1}]
            # 这样直接填id有很大的风险，可能会报错。但是要正确填就需要gt的信息，传参麻烦~~
            # default_dt = [{'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 1, 1], 'score': 1}]

            if not dt:
                dt_list = default_dt
            else:
                dt_list = dt if isinstance(dt, (list, tuple)) else File(dt).read()
                if min_score:
                    dt_list = [b for b in dt_list if (b['score'] >= min_score)]
                if not dt_list:
                    dt_list = default_dt
            return dt_list

        self.dt_list = get_dt_list(dt, min_score)

    @classmethod
    def is_dt_list(cls, dt_list):
        if not isinstance(dt_list, (tuple, list)):
            return False
        item = dt_list[0]
        has_keys = set('score image_id category_id bbox'.split())
        return not (has_keys - item.keys())

    def select_dt(self, ids, *, inplace=False):
        gt_dict, dt_list = self.gt_dict, self.dt_list
        # 1 ids 统一为int类型的id值
        if not isinstance(ids, (list, tuple, set)):
            ids = [ids]
        if gt_dict:
            map_name2id = {item['file_name']: item['id'] for item in gt_dict['images']}
            ids = [(map_name2id[x] if isinstance(x, str) else x) for x in ids]
        ids = set(ids)

        # 2 简化images
        dst = [x for x in dt_list if (x['image_id'] in ids)]
        if inplace: self.dt_list = dst
        return dst

    def group_dt(self, *, reserve_empty=False):
        """ 对annos按image_id分组，返回 [(im1, dt_anns1), (im2, dt_anns2), ...] """
        group_anns = defaultdict(list)
        [group_anns[an['image_id']].append(an) for an in self.dt_list]
        return self._group_base(group_anns, reserve_empty)

    def group_gt_dt(self, *, reserve_empty=False):
        """ 获得一张图片上gt和dt的标注结果

        [(im, gt_anns, dt_anns), ...]
        """
        raise NotImplementedError

    def to_icdar_label_quad(self, outfile, *, min_score=0):
        """ 将coco的dt结果转为icdar的标注格式

        存成一个zip文件，zip里面每张图对应一个txt标注文件
        每个txt文件用quad八个数值代表一个标注框

        适用于 sroie 检测格式
        """
        # 1 获取dt_list
        if min_score:
            dt_list = [b for b in self.dt_list if (b['score'] >= min_score)]
        else:
            dt_list = self.dt_list

        # 2 转df，按图片分组处理
        df = pd.DataFrame.from_dict(dt_list)  # noqa from_dict可以传入List[Dict]
        df = df.groupby('image_id')

        # 3 建立一个zip文件
        myzip = ZipFile(str(outfile), 'w')

        # 4 遍历每一组数据，生成一个文件放到zip里面
        id2name = {im['id']: pathlib.Path(im['file_name']).stem for im in self.gt_dict['images']}
        for image_id, items in df:
            label_file = id2name[image_id] + '.txt'
            quads = [rect2polygon(xywh2ltrb(x), dtype=int).reshape(-1) for x in items['bbox']]
            quads = [','.join(map(str, x)) for x in quads]
            myzip.writestr(label_file, '\n'.join(quads))
        myzip.close()


class Coco2Labelme(ToLabelmeJson):
    """ coco格式的可视化

    TODO segmentation 分割 效果的可视化
    """

    def add_segmentation(self, row):
        """ 分割默认先都用灰色标注 """
        r = dict()
        r['gt_box_id'] = row['gt_box_id']
        r['label'] = 'seg'
        r['points'] = row['gt_ltrb']
        r['shape_color'] = [191, 191, 191]

        # 5 保存
        self.add_shape2(**r)

    # def _sort_anns(self, anns):
    #     if anns and 'score' in anns[0]:
    #         anns = sorted(anns, key=lambda x: -x['score'])  # 权重从大到小排序
    #     return anns

    def add_gt_shape(self, row, attrs=None):
        """
        :param row: df的一行数据series
        :param attrs: 其他扩展字段值
        """
        # 1 基本字段
        r = dict()
        for name in ['gt_box_id', 'gt_category_id', 'gt_area']:
            r[name] = row[name]
        r['gt_ltrb'] = ','.join(map(str, row['gt_ltrb']))

        # 2 主要字段
        r['label'] = row['gt_category_name']  # 这个需要上层的anns_match2, labelme_match传入的df实现提供这个字段
        r['points'] = row['gt_ltrb']
        if row['gt_supercategory'] != '':
            r['group_id'] = row['gt_supercategory']

        # 3 row中其他自定义字段
        # 这些是已经处理过的标准字段，进入黑名单，不显示；其他字段默认白名单都显示
        std_an_keys = set('gt_box_id gt_category_id gt_ltrb gt_area iscrowd file_name '
                          'gt_category_name gt_supercategory gt_segmentation dt_segmentation'.split())

        # 如果跟labelme的标准字段重名了，需要区分下：比如 label
        std_lm_keys = set('label points group_id shape_type flags'.split())  # labelme的标准字段
        ks = set(row.index) - std_an_keys
        for k in ks:
            if k in std_lm_keys:
                r['_' + k] = row[k]
            else:
                r[k] = row[k]
        if 'dt_ltrb' in r:
            r['dt_ltrb'] = ','.join(map(str, r['dt_ltrb']))

        # 4 精简字段：聚合以dt、gt为前缀的所有字段
        group_keys = defaultdict(list)
        res = dict()
        for k, v in r.items():
            for part in ('dt', 'gt'):
                if k.startswith(part + '_'):
                    group_keys[part].append(k)
                    break
            else:
                res[k] = v

        # 聚合后的属性排序准则
        order = ['category_id', 'category_name', 'score', 'ltrb', 'area', 'box_id']
        idxfunc = make_index_function(order)
        for part in ('dt', 'gt'):
            keys = group_keys[part]
            m = len(part) + 1
            keys.sort(key=lambda k: idxfunc(k[m:]))
            res[part] = '/'.join([str(r[k]) for k in keys])  # 数值拼接
            res['~' + part] = '/'.join([str(k[m:]) for k in keys])  # 解释key，如果很熟悉了可以选择关闭

        # 5 扩展字段
        if attrs:
            res.update(attrs)

        # 6 保存
        self.add_shape2(**res)

    def add_dt_shape(self, row, attrs=None):
        # 1 基本字段
        r = dict()
        for name in ['iou', 'dt_category_id', 'dt_score']:
            r[name] = row[name]
        r['dt_ltrb'] = ','.join(map(str, row['dt_ltrb']))

        # 2 主要字段
        r['label'] = row['dt_category_name']
        if 'dt_segmentation' in row:
            r['points'] = row['dt_segmentation'][0]
        else:
            r['points'] = row['dt_ltrb']

        # 3 扩展字段
        if attrs:
            r.update(attrs)

        # 4 保存
        self.add_shape2(**r)

    def _anns_init(self, df, segmentation=False):
        df = df.copy()
        df.drop(['image_id'], axis=1, inplace=True)

        columns = df.columns
        if segmentation:
            pass
        else:
            if 'gt_segmentation' in columns:
                df.drop('gt_segmentation', axis=1, inplace=True)
            if 'dt_segmentation' in columns:
                df.drop('dt_segmentation', axis=1, inplace=True)

        return df

    def anns_gt(self, df, *, segmentation=False, shape_attrs=None):
        """ Coco2Df.gt的可视化

        :param df: Coco2Df生成的df后，输入特定的某一组image_id、file_name
        :param segmentation: 是否显示segmentation分割效果
        :param shape_attrs: 人工额外强制设置的字段值
        """
        df = self._anns_init(df, segmentation)
        for idx, row in df.iterrows():
            if segmentation:
                self.add_segmentation(row)
            self.add_gt_shape(row, shape_attrs)

    def anns_match(self, df, *, hide_match_dt=False, segmentation=False, shape_attrs=None):
        """ Coco2Df.match的可视化

        正确的gt用绿框，位置匹配到但类别错误的用黄框，绿黄根据iou设置颜色深浅，此时dt统一用灰色框
        漏检的gt用红框，多余的dt用蓝框

        :param hide_match_dt: 不显示灰色的dt框

        TODO 研究labelme shape的flags参数含义，支持shape的过滤显示？
        """
        df = self._anns_init(df, segmentation)
        if not shape_attrs:
            shape_attrs = {}

        def get_attrs(d):
            return dict(ChainMap(shape_attrs, d))

        for idx, row in df.iterrows():
            r = row
            if r['gt_category_id'] == -1:  # 多余的dt
                self.add_dt_shape(r, get_attrs({'shape_color': [0, 0, 255]}))
            elif r['dt_category_id'] == -1:  # 没有被匹配到的gt
                self.add_gt_shape(r, get_attrs({'shape_color': [255, 0, 0]}))
            else:  # 匹配到的gt和dt
                if not hide_match_dt:
                    self.add_dt_shape(r, get_attrs({'shape_color': [191, 191, 191]}))
                color_value = int(255 * r['iou'])

                if r['gt_category_id'] == r['dt_category_id']:
                    self.add_gt_shape(r, get_attrs({'shape_color': [0, color_value, 0]}))
                else:
                    self.add_gt_shape(r, get_attrs({'shape_color': [color_value, color_value, 0]}))

    def anns_match2(self, df, *, hide_match_dt=False, segmentation=False, shape_attrs=None, colormap=None):
        """ 按类别区分框颜色
        """
        import imgviz

        df = self._anns_init(df, segmentation)
        if not shape_attrs:
            shape_attrs = {}

        def get_attrs(d):
            return dict(ChainMap(shape_attrs, d))

        if not colormap:
            colormap = imgviz.label_colormap(value=200)
        m = len(colormap)

        for idx, row in df.iterrows():
            r = row
            attrs = {'shape_color': colormap[r['gt_category_id'] % m],
                     'vertex_fill_color': colormap[r['dt_category_id'] % m]}

            if r['gt_category_id'] == -1:  # 多余的dt
                self.add_dt_shape(r, get_attrs(attrs))
            elif r['dt_category_id'] == -1:  # 没有被匹配到的gt
                self.add_gt_shape(r, get_attrs(attrs))
            else:  # 匹配到的gt和dt
                if not hide_match_dt:
                    self.add_dt_shape(r, get_attrs({'shape_color': [191, 191, 191]}))
                attrs['vertex_fill_color'] = [int(r['iou'] * v) for v in attrs['vertex_fill_color']]
                self.add_gt_shape(r, get_attrs(attrs))


class CocoEval(CocoData):
    def __init__(self, gt, dt, iou_type='bbox', *, min_score=0, print_mode=False):
        """
        TODO coco_gt、coco_dt本来已存储了很多标注信息，有些冗余了，是否可以跟gt_dict、dt_list等整合，去掉些没必要的组件？
        """
        super().__init__(gt, dt, min_score=min_score)

        # type
        self.iou_type = iou_type

        # evaluater
        self.coco_gt = COCO(gt, print_mode=print_mode)  # 这不需要按图片、类型分类处理
        self.coco_dt, self.evaluater = None, None
        if self.dt_list:
            self.coco_dt = self.coco_gt.loadRes(self.dt_list)  # 这个返回也是coco对象
            self.evaluater = COCOeval(self.coco_gt, self.coco_dt, iou_type, print_mode=print_mode)

    @classmethod
    def evaluater_eval(cls, et, img_ids=None, *, print_mode=False):
        """ coco官方目标检测测评方法
        https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb

        :param img_ids:
        :param print_mode: 注意这里的print_mode不同于初始化的print_mode，指的是不同的东西
        :return:
        """
        # 1 coco是有方法支持过滤id，只计算部分图的分值结果
        # 没有输入img_ids，也要显式重置为全部数据
        if not img_ids:
            img_ids = et.cocoGt.imgIds.values()
        et.params.imgIds = list(img_ids)

        # 2 每张图片、每个类别的iou等核心数据的计算
        et.evaluate()
        # 在不同参数测评指标下的分数
        et.accumulate()

        # 3 显示结果
        if print_mode:  # 如果要显示结果则使用标准计算策略
            et.summarize(print_mode=print_mode)
            return round(et.stats[0], 4)
        else:  # 否则简化计算过程
            return round(et.step_summarize(), 4)

    def eval(self, img_ids=None, *, print_mode=False):
        return self.evaluater_eval(self.evaluater, img_ids=img_ids, print_mode=print_mode)

    def eval_dt_score(self, step=0.1):
        """ 计算按一定阈值滤除框后，对coco指标产生的影响 """
        dt_list = copy.copy(self.dt_list)

        i = 0
        records = []
        columns = ['≥dt_score', 'n_dt_box', 'coco_score']
        while i < 1:
            dt_list = [x for x in dt_list if x['score'] >= i]
            if not dt_list: break
            coco_dt = self.coco_gt.loadRes(dt_list)
            evaluater = COCOeval(self.coco_gt, coco_dt, self.iou_type)
            records.append([i, len(dt_list), self.evaluater_eval(evaluater)])
            i += step
        df = pd.DataFrame.from_records(records, columns=columns)
        return df

    def parse_dt_score(self, step=0.1, *, print_mode=False):
        """ dt按不同score过滤后效果

        注意如果数据集很大，这个功能运算特别慢，目前测试仅20张图都要10秒
        可以把print_mode=True打开观察中间结果

        注意这个方法，需要调用后面的 CocoMatch
        """
        gt_dict, dt_list = self.gt_dict, self.dt_list

        i = 0
        records = []
        columns = ['≥dt_score', 'n_dt_box', 'n_match_box', 'n_matchcat_box',
                   'coco_score',
                   'icdar2013', 'ic13_precision', 'ic13_recall',
                   'f1_score']
        if print_mode: print(columns)
        while i < 1:
            dt_list = [x for x in dt_list if x['score'] >= i]
            if not dt_list: break
            cm = CocoMatch(gt_dict, dt_list, eval_im=False)

            ie = IcdarEval(*cm.to_icdareval_data())
            ic13 = ie.icdar2013()

            row = [i, cm.n_dt_box(), cm.n_match_box(), cm.n_matchcat_box(),
                   cm.eval(), ic13['hmean'], ic13['precision'], ic13['recall'], cm.f1_score()]

            if print_mode: print(row)
            records.append(row)
            i += step
        df = pd.DataFrame.from_records(records, columns=columns)

        if print_mode:
            with pd.option_context('display.max_colwidth', -1, 'display.max_columns', 20,
                                   'display.width', 200):  # 上下文控制格式
                print(df)

        return df


class CocoParser(CocoEval):
    def __init__(self, gt, dt=None, iou_type='bbox', *, min_score=0, print_mode=False):
        """ coco格式相关分析工具，dt不输入也行，当做没有任何识别结果处理~~
            相比CocoMatch比较轻量级，不会初始化太久，但提供了一些常用的基础功能
        """
        super().__init__(gt, dt, iou_type, min_score=min_score, print_mode=print_mode)
        # gt里的images、categories数据，已转成df表格格式
        self.images, self.categories = self._get_images_df(), self._get_categories_df()
        # gt、dt的统计表
        self.gt_anns, self.dt_anns = self._get_gt_anns_df(), self._get_dt_anns_df()

    @classmethod
    def bbox2ltrb(cls, b):
        return [int(round(v, 0)) for v in xywh2ltrb(b)]

    def _get_images_df(self):
        """ 从gt['images']转df
        """
        df = pd.DataFrame.from_dict(self.gt_dict['images'])
        df.rename(columns={'id': 'image_id'}, inplace=True)
        df.set_index('image_id', inplace=True)
        return df

    def _get_categories_df(self):
        """ 把gt['categories']转df
        """
        df = pd.DataFrame.from_dict(self.gt_dict['categories'])
        df.rename(columns={'id': 'category_id'}, inplace=True)
        df.set_index('category_id', inplace=True)
        return df

    def _get_gt_anns_df(self):
        """ 输入gt的json文件或字典，转df格式

        # TODO 暂时没考虑iscrowd=1的情况，先不处理这个字段
        """

        # 1 读取数据，转字典
        df = pd.DataFrame.from_dict(self.gt_dict['annotations'])

        # 2 构建完整表格信息
        df['gt_ltrb'] = [self.bbox2ltrb(b) for b in df['bbox']]
        df['area'] = [int(round(v, 0)) for v in df['area']]
        df.rename(columns={'id': 'gt_box_id', 'category_id': 'gt_category_id',
                           'area': 'gt_area', 'segmentation': 'gt_segmentation'}, inplace=True)

        # 3 筛选最终使用的表格及顺序
        columns = ['image_id', 'gt_box_id', 'gt_category_id', 'gt_ltrb', 'gt_area', 'gt_segmentation']
        ext = set(df.columns) - set(columns + ['bbox'])  # 扩展字段
        columns += list(ext)
        return df[columns]

    def _get_dt_anns_df(self):
        # 1 读取数据，转列表
        df = pd.DataFrame.from_dict(self.dt_list)  # noqa

        # 2 构建完整表格信息
        columns = ['image_id', 'dt_category_id', 'dt_ltrb', 'dt_score']
        if len(df) > 0:
            df['dt_ltrb'] = [self.bbox2ltrb(b) for b in df['bbox']]
            df['dt_score'] = [round(v, 4) for v in df['score']]
            df['dt_segmentation'] = df['segmentation']  # 就算构建的时候没有segmentation字段，xlcocotools也会自动添加生成的
            df.rename(columns={'category_id': 'dt_category_id'}, inplace=True)
            # 3 筛选最终使用的表格及顺序
            ext = set(df.columns) - set(columns + ['bbox', 'score', 'category_id', 'segmentation'])  # 扩展字段
            columns += list(ext)
            return df[columns]
        else:
            return pd.DataFrame(columns=columns)

    def to_icdareval_data(self, *, min_score=0.):
        """ 转成可供IcdarEval测评的数据格式

        :param min_score: dt框至少需要的score置信度，可以按0.5过滤掉低置信度的框再计算
        :return: gt, dt
            两个数据，一个gt, 一个dt
        """
        # 1 gt的格式转换
        res = defaultdict(list)
        for item in self.gt_dict['annotations']:
            ltrb = self.bbox2ltrb(item['bbox'])
            # gt的label需要加上一个文字内容标注，这里用iscrowd代替
            label = ','.join(map(lambda x: str(round(x)), ltrb + [item['iscrowd']]))
            # 除了"图片"，还要区分"类别"
            res[f"{item['image_id']},{item['category_id']}"].append(label)
        gt = {k: '\n'.join(v).encode() for k, v in res.items()}

        # 2 dt的格式转换
        res = defaultdict(list)
        for item in self.dt_list:
            if item['score'] < min_score: continue
            ltrb = self.bbox2ltrb(item['bbox'])
            label = ','.join(map(lambda x: str(round(x)), ltrb))
            res[f"{item['image_id']},{item['category_id']}"].append(label)
        dt = {k: '\n'.join(v).encode() for k, v in res.items()}

        return gt, dt

    def icdar2013(self):
        ie = IcdarEval(*self.to_icdareval_data())
        return ie.icdar2013()['hmean']

    def to_excel(self, savepath, segmentation=False):
        """ 将所有统计表导入到一个excel文件

        :param savepath: 保存的文件名
        :param segmentation: 是否保存segmentation的值
        """
        with pd.ExcelWriter(str(savepath)) as writer:
            self.images.to_excel(writer, sheet_name='images', freeze_panes=(1, 0))
            self.categories.to_excel(writer, sheet_name='categories', freeze_panes=(1, 0))
            gt_anns = self.gt_anns
            if not segmentation: gt_anns = gt_anns.drop('gt_segmentation', axis=1)
            gt_anns.to_excel(writer, sheet_name='gt_anns', freeze_panes=(1, 0))
            self.dt_anns.to_excel(writer, sheet_name='dt_anns', freeze_panes=(1, 0))

    def to_labelme_gt(self, imdir, dst_dir=None, *, segmentation=False, max_workers=4):
        """ 在图片目录里生成图片的可视化json配置文件

        :param segmentation: 是否显示分割效果
        """

        def func(g):
            # 1 获得图片id和文件
            image_id, df = g
            imfile = File(df.iloc[0]['file_name'], imdir)
            if not imfile:
                return  # 如果没有图片不处理

            # 2 生成这张图片对应的json标注
            if dst_dir:
                imfile = imfile.copy(dst_dir, if_exists='skip')
            lm = Coco2Labelme(imfile)
            height, width = lm.img.size  # 也可以用image['height'], image['width']获取
            # 注意df取出来的image_id默认是int64类型，要转成int，否则json会保存不了int64类型
            lm.add_shape('', [0, 0, 10, 0], shape_type='line', shape_color=[0, 0, 0],
                         n_gt_box=len(df), image_id=int(image_id),
                         size=f'{height}x{width}')
            lm.anns_gt(df, segmentation=segmentation)
            lm.write()  # 保存json文件到img对应目录下

        if dst_dir:
            dst_dir = Dir(dst_dir)
            dst_dir.ensure_dir()
        gt_anns = self.gt_anns.copy()
        # 为了方便labelme操作，需要扩展几列内容
        gt_anns['file_name'] = [self.images.loc[x, 'file_name'] for x in gt_anns['image_id']]
        gt_anns['gt_category_name'] = [self.categories.loc[x, 'name'] for x in gt_anns['gt_category_id']]
        gt_anns['gt_supercategory'] = [self.categories.loc[x, 'supercategory'] for x in gt_anns['gt_category_id']]
        mtqdm(func, list(gt_anns.groupby('image_id').__iter__()), 'create labelme gt jsons', max_workers=max_workers)


class CocoMatchBase:
    def __init__(self, match_df):
        """ match_df匹配表格相关算法

        这个类是算法本质，而CocoMatch做了一层封装，默认对整个match_df进行处理。
        这个底层类可以用来计算每张图的具体情况
        """
        self.match_anns = match_df

    def n_gt_box(self):
        return sum([x != -1 for x in self.match_anns['gt_category_id']])

    def n_dt_box(self):
        return sum([x != -1 for x in self.match_anns['dt_category_id']])

    def n_match_box(self, iou=0.5):
        """ 不小于iou的框匹配到的数量 """
        return sum(self.match_anns['iou'] >= iou)

    def n_matchcat_box(self, iou=0.5):
        """ 不仅框匹配到，类别也对应的数量 """
        df = self.match_anns
        return sum(((df['iou'] >= iou) & (df['gt_category_id'].eq(df['dt_category_id']))))

    def get_clsmatch_arr(self, iou=0.5):
        """ 返回不小于iou下，框匹配的gt、dt对应的类别编号矩阵arr1, arr2 """
        df = self.match_anns
        df = df[df['iou'] >= iou]
        return list(df['gt_category_id']), list(df['dt_category_id'])

    def f1_score(self, average='weighted', iou=sys.float_info.epsilon):
        """ coco本来是同时做检测、分类的，所以有coco自己的评价指标

        一般单独做多分类任务是用F1分值
        这里尝试将结果dt框和原始的gt框做匹配，然后强行算出一个f1值

        :param average:
            weighted：每一类都算出f1，然后加权平均
            macro：每一类都算出f1，然后求平均值（因为合同样本不均衡问题，而且有的类甚至只出现1次，结果会大大偏）
            micro：按二分类形式直接计算全样本的f1 (就相当于直接 正确数/总数)
        :param iou: 不小于iou的匹配框才计算分类准确率
        :return:
        """
        from sklearn.metrics import f1_score
        gt_arr, dt_arr = self.get_clsmatch_arr(iou)
        if gt_arr:
            return round(f1_score(gt_arr, dt_arr, average=average), 4)
        else:
            return -1

    def multi_iou_f1_df(self, step=0.1):
        """ 计算多种iou下匹配的框数量，和分类质量 """
        records = []
        columns = ['iou', 'n_boxmatch', 'n_clsmatch',
                   'f1_weighted', 'f1_macro', 'f1_micro']
        i = 0
        while i <= 1:
            r = [i, self.n_match_box(i), self.n_matchcat_box(i)]
            if r[1]:
                r.append(self.f1_score('weighted', i))
                r.append(self.f1_score('macro', i))
                r.append(self.f1_score('micro', i))
                records.append(r)
            else:
                records.append(r + [0, 0, 0])
                break
            i += step

        df = pd.DataFrame.from_records(records, columns=columns)
        return df


class CocoMatch(CocoParser, CocoMatchBase):
    def __init__(self, gt, dt=None, *, min_score=0, eval_im=True, print_mode=False):
        """ coco格式相关分析工具，dt不输入也行，当做没有任何识别结果处理~~

        :param min_score: 滤除dt中score小余min_score的框
        :param eval_im: 是否对每张图片计算coco分数
        """
        # 因为这里 CocoEval、_CocoMatchBase 都没有父级，不会出现初始化顺序混乱问题
        #   所以我直接指定类初始化顺序了，没用super
        CocoParser.__init__(self, gt, dt, min_score=min_score)
        match_anns = self._get_match_anns_df(print_mode=print_mode)
        CocoMatchBase.__init__(self, match_anns)
        self.images = self._get_match_images_df(eval_im=eval_im, print_mode=print_mode)
        self.categories = self._get_match_categories_df()

    def _get_match_anns_df(self, *, print_mode=False):
        """ 将结果的dt框跟gt的框做匹配，注意iou非常低的情况也会匹配上

        TODO 有些框虽然没匹配到，但并不是没有iou，只是被其他iou更高的框抢掉了而已，可以考虑新增一个实际最大iou值列
        TODO 这里有个隐患，我找不到的框是用-1的类id来标记。但如果coco数据里恰好有个-1标记的类，就暴雷了~~
        TODO 210512周三11:27，目前新增扩展了label，这个是采用白名单机制加的，后续是可以考虑用黑名单机制来设定
        """
        from tqdm import tqdm

        # 1 读取数据
        gt_df, dt_df = self.gt_anns.groupby('image_id'), self.dt_anns.groupby('image_id')

        # 2 初始化
        records = []
        gt_columns = ['gt_box_id', 'gt_category_id', 'gt_ltrb', 'gt_area']
        ext = set(self.gt_anns.keys()) - set(gt_columns + ['image_id'])
        gt_columns += list(ext)
        gt_default = [-1, -1, '', 0] + [None] * len(ext)  # 没有配对项时填充的默认值
        if 'label' in self.gt_anns.columns:
            gt_columns.append('label')
            gt_default.append('')

        dt_columns = ['dt_category_id', 'dt_ltrb', 'dt_score', 'dt_segmentation']
        ext = set(self.dt_anns.keys()) - set(dt_columns + ['image_id', 'iscrowd', 'area', 'id'])
        dt_columns += list(ext)
        dt_default = [-1, '', 0] + [None] * len(ext)

        columns = ['image_id'] + gt_columns + ['iou'] + dt_columns

        def gt_msg(x=None):
            if x is None:
                return [image_id] + gt_default
            else:
                return [image_id] + [x[k] for k in gt_columns]

        def dt_msg(y=None, iou_score=0):
            if y is None:
                return [0] + dt_default
            else:
                return [round(iou_score, 4)] + [y[k] for k in dt_columns]

        # 3 遍历匹配
        for image_id, image in tqdm(self.images.iterrows(),
                                    f'_get_match_anns_df, groups={len(self.images)}', disable=not print_mode):
            # 3.1 计算匹配项
            # gt和dt关于某张图都有可能没有框
            # 比如合同检测，有的图可能什么类别对象都没有，gt中这张图本来就没有box；dt检测也是，某张图不一定有结果
            gt_group_df = gt_df.get_group(image_id) if image_id in gt_df.groups else []
            dt_group_df = dt_df.get_group(image_id) if image_id in dt_df.groups else []
            n, m = len(gt_group_df), len(dt_group_df)

            pairs = []
            if n and m:
                # 任意多边形相交面积算法速度太慢
                # gt_bboxes = [ShapelyPolygon.gen(b) for b in gt_group_df['gt_ltrb']]  # noqa 已经用if做了判断过滤
                # dt_bboxes = [ShapelyPolygon.gen(b) for b in dt_group_df['dt_ltrb']]  # noqa
                # pairs = matchpairs(gt_bboxes, dt_bboxes, ComputeIou.polygon2, index=True)

                # 改成ltrb的相交面积算法会快一点
                # gt_bboxes = [ShapelyPolygon.gen(b) for b in gt_group_df['gt_ltrb']]  # noqa 已经用if做了判断过滤
                # dt_bboxes = [ShapelyPolygon.gen(b) for b in dt_group_df['dt_ltrb']]  # noqa
                pairs = matchpairs(gt_group_df['gt_ltrb'].to_list(), dt_group_df['dt_ltrb'].to_list(),
                                   ComputeIou.ltrb, index=True)

            # 3.2 按gt顺序存入每条信息
            dt_ids = set(range(m))
            match_ids = {p[0]: (p[1], p[2]) for p in pairs}
            for i in range(n):
                x = gt_group_df.iloc[i]
                if i in match_ids:
                    # 3.2.1 gt与dt匹配的box
                    j, iou_score = match_ids[i]
                    dt_ids.remove(j)
                    records.append(gt_msg(x) + dt_msg(dt_group_df.iloc[j], iou_score))
                else:
                    # 3.2.2 有gt没有对应dt的box
                    records.append(gt_msg(x) + dt_msg())

            # 3.2.3 还有剩余未匹配到的dt也要记录
            for j in dt_ids:
                records.append(gt_msg() + dt_msg(dt_group_df.iloc[j]))

        # 4 保存结果
        return pd.DataFrame.from_records(records, columns=columns)

    def _get_match_images_df(self, *, eval_im=True, print_mode=False):
        """ 在原有images基础上，扩展一些图像级别的识别结果情况数据 """
        # 1 初始化，新增字段
        images, match_anns = self.images.copy(), self.match_anns.groupby('image_id')
        columns = ['coco_score', 'n_gt_box', 'n_dt_box', 'n_match0.5_box', 'n_matchcat0.5_box', 'f1_micro0.5',
                   'ic13_score']
        for c in columns:
            images[c] = -1.0

        # 2 填写扩展字段的值
        for image_id in tqdm(images.index, '_get_match_images_df', disable=not print_mode):
            # 2.1 跳过不存在的图
            if image_id not in match_anns.groups:
                continue
            df = match_anns.get_group(image_id)

            # 2.2 增加每张图片的coco分数
            if eval_im:
                images.loc[image_id, 'coco_score'] = self.eval([image_id])

            # 2.3 增加每张图片的多分类分数
            m = CocoMatchBase(df)
            images.loc[image_id, 'n_gt_box'] = m.n_gt_box()
            images.loc[image_id, 'n_dt_box'] = m.n_dt_box()
            images.loc[image_id, 'n_match0.5_box'] = m.n_match_box(0.5)
            images.loc[image_id, 'n_matchcat0.5_box'] = m.n_matchcat_box(0.5)
            images.loc[image_id, 'f1_micro0.5'] = m.f1_score('micro', 0.5)

            # 2.4 增加每张图片的ic13分数
            # df要先按category_id分组，多个ltrb值存成list
            if eval_im:
                gt, dt = dict(), dict()
                for key, items in df.groupby('gt_category_id'):
                    if key != -1:
                        gt[key] = list(items['gt_ltrb'])
                for key, items in df.groupby('dt_category_id'):
                    if key != -1:
                        dt[key] = list(items['dt_ltrb'])
                images.loc[image_id, 'ic13_score'] = IcdarEval(gt, dt).icdar2013()['hmean']

        return images

    def _get_match_categories_df(self):
        """ 在原有categories基础上，扩展一些每个类别上整体情况的数据

        match_support: iou为0.5时实际匹配上的框数量，match从"匹配"而来
            而support则是出自f1指标，称为gt提供了多少框
        """
        from sklearn.metrics import classification_report

        # 1 初始化，新增字段
        categories, match_anns = self.categories.copy(), self.match_anns
        columns = ['n_gt_box', 'n_dt_box', 'match_support', 'f1_score', 'precision', 'recall']
        for c in columns:
            categories[c] = -1.0
        categories.loc[-1] = ['', 'non_match'] + [-1] * 6  # noqa

        # 2 填写扩展字段的值
        for k, v in Counter(match_anns['gt_category_id']).items():
            categories.loc[k, 'n_gt_box'] = v

        for k, v in Counter(match_anns['dt_category_id']).items():
            categories.loc[k, 'n_dt_box'] = v

        # 要调换一下-1这个类的情况，这样才是对应的没找到的gt，和多余的dt
        categories.loc[-1, 'n_gt_box'], categories.loc[-1, 'n_dt_box'] = \
            categories.loc[-1, 'n_dt_box'], categories.loc[-1, 'n_gt_box']

        gt_arr, dt_arr = self.get_clsmatch_arr(0.5)
        if gt_arr:
            d = classification_report(gt_arr, dt_arr, output_dict=True)
            for k, v in d.items():
                if k not in ('accuracy', 'macro avg', 'weighted avg'):
                    k = int(k)
                    categories.loc[k, 'match_support'] = v['support']
                    categories.loc[k, 'f1_score'] = round(v['f1-score'], 4)
                    categories.loc[k, 'precision'] = round(v['precision'], 4)
                    categories.loc[k, 'recall'] = round(v['recall'], 4)

        return categories

    def eval_all(self):
        """ 把目前支持的所有coco格式的测评全部跑一遍
        """
        xllog = get_xllog()
        xllog.info('1 coco官方评测指标（综合性指标）')
        self.eval(print_mode=True)

        xllog.info('2 icdar官方三种评测方法')
        ie = IcdarEval(*self.to_icdareval_data())
        print('icdar2013  ', ie.icdar2013())
        print('deteval    ', ie.deteval())
        print('iou        ', ie.iou())
        ie = IcdarEval(*self.to_icdareval_data(min_score=0.5))
        print('如果滤除dt中score<0.5的低置信度框：')
        print('icdar2013  ', ie.icdar2013())
        print('deteval    ', ie.deteval())
        print('iou        ', ie.iou())
        sys.stdout.flush()

        xllog.info('3 框匹配情况，多分类F1值')
        # TODO 这个结果补充画个图表？
        print(f'gt共有{self.n_gt_box()}，dt共有{self.n_dt_box()}')
        print(self.multi_iou_f1_df(0.1))

        xllog.info('4 dt按不同score过滤后效果')
        print(self.parse_dt_score())

    def to_excel(self, savepath, *, segmentation=False):
        dataframes_to_excel(savepath,
                            {'images': self.images,
                             'categories': self.categories,
                             'match_anns': self.match_anns})

    def _to_labelme_match(self, match_func_name, imdir, dst_dir=None, *, segmentation=False, hide_match_dt=False,
                          **kwargs):
        """ 可视化目标检测效果

        :param imdir: 默认会把结果存储到imdir
        :param dst_dir: 但如果写了dst_dir参数，则会有选择地从imdir筛选出图片到dst_dir
        """

        def func(g):
            # 1 获得图片id和文件
            image_id, df = g
            imfile = File(df.iloc[0]['file_name'], imdir)
            if not imfile:
                return  # 如果没有图片不处理
            image = self.images.loc[image_id]
            image = image.drop(['file_name', 'height', 'width'])

            # 2 生成这张图片对应的json标注
            if dst_dir and dst_dir.exists():
                imfile = imfile.copy(dst_dir, if_exists='skip')
            lm = Coco2Labelme(imfile)

            height, width = lm.data['imageHeight'], lm.data['imageWidth']
            # 注意df取出来的image_id默认是int64类型，要转成int，否则json会保存不了int64类型
            lm.add_shape('', [0, 0, 10, 0], shape_type='line', shape_color=[0, 0, 0],
                         size=f'{height}x{width}', **(image.to_dict()))
            getattr(lm, match_func_name)(df, segmentation=segmentation, hide_match_dt=hide_match_dt, **kwargs)
            lm.write(if_exists=None)  # 保存json文件到img对应目录下

        if dst_dir is not None:
            dst_dir = Dir(dst_dir)
            dst_dir.ensure_dir()
        match_anns = self.match_anns.copy()
        # 为了方便labelme操作，需要扩展几列内容
        match_anns['file_name'] = [self.images.loc[x, 'file_name'] for x in match_anns['image_id']]
        match_anns['gt_category_name'] = [self.categories.loc[x, 'name'] for x in match_anns['gt_category_id']]
        match_anns['dt_category_name'] = [self.categories.loc[x, 'name'] for x in match_anns['dt_category_id']]
        match_anns['gt_supercategory'] = [self.categories.loc[x, 'supercategory'] for x in match_anns['gt_category_id']]
        mtqdm(func, list(iter(match_anns.groupby('image_id'))), max_workers=8, desc='make labelme json:')

    def to_labelme_match(self, imdir, dst_dir=None, *, segmentation=False, hide_match_dt=False):
        self._to_labelme_match('anns_match', imdir, dst_dir, segmentation=segmentation, hide_match_dt=hide_match_dt)

    def to_labelme_match2(self, imdir, dst_dir=None, *, segmentation=False, hide_match_dt=False,
                          colormap=LABEL_COLORMAP7):
        self._to_labelme_match('anns_match2', imdir, dst_dir, segmentation=segmentation, hide_match_dt=hide_match_dt,
                               colormap=colormap)
