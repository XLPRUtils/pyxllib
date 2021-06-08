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

from pyxllib.xlcv import *

from collections import ChainMap

# 使用该模块需要安装 xlcocotools
try:
    import xlcocotools
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'xlcocotools'])
from xlcocotools.coco import COCO
from xlcocotools.cocoeval import COCOeval

from pyxllib.data.label import LABEL_COLORMAP7, ToLabelmeJson, CocoGtData, CocoData
from pyxllib.data.icdar import IcdarEval


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
    def __init__(self, gt, dt, iou_type='bbox', *, min_score=0, printf=False):
        """
        TODO coco_gt、coco_dt本来已存储了很多标注信息，有些冗余了，是否可以跟gt_dict、dt_list等整合，去掉些没必要的组件？
        """
        super().__init__(gt, dt, min_score=min_score)

        # type
        self.iou_type = iou_type

        # evaluater
        self.coco_gt = COCO(gt, printf=printf)  # 这不需要按图片、类型分类处理
        self.coco_dt, self.evaluater = None, None
        if self.dt_list:
            self.coco_dt = self.coco_gt.loadRes(self.dt_list)  # 这个返回也是coco对象
            self.evaluater = COCOeval(self.coco_gt, self.coco_dt, iou_type, printf=printf)

    @classmethod
    def evaluater_eval(cls, et, img_ids=None, *, printf=False):
        """ coco官方目标检测测评方法
        https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb

        :param img_ids:
        :param printf: 注意这里的printf不同于初始化的printf，指的是不同的东西
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
        if printf:  # 如果要显示结果则使用标准计算策略
            et.summarize(printf=printf)
            return round(et.stats[0], 4)
        else:  # 否则简化计算过程
            return round(et.step_summarize(), 4)

    def eval(self, img_ids=None, *, printf=False):
        return self.evaluater_eval(self.evaluater, img_ids=img_ids, printf=printf)

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

    def parse_dt_score(self, step=0.1, *, printf=False):
        """ dt按不同score过滤后效果

        注意如果数据集很大，这个功能运算特别慢，目前测试仅20张图都要10秒
        可以把printf=True打开观察中间结果

        注意这个方法，需要调用后面的 CocoMatch
        """
        gt_dict, dt_list = self.gt_dict, self.dt_list

        i = 0
        records = []
        columns = ['≥dt_score', 'n_dt_box', 'n_match_box', 'n_matchcat_box',
                   'coco_score',
                   'icdar2013', 'ic13_precision', 'ic13_recall',
                   'f1_score']
        if printf: print(columns)
        while i < 1:
            dt_list = [x for x in dt_list if x['score'] >= i]
            if not dt_list: break
            cm = CocoMatch(gt_dict, dt_list, eval_im=False)

            ie = IcdarEval(*cm.to_icdareval_data())
            ic13 = ie.icdar2013()

            row = [i, cm.n_dt_box(), cm.n_match_box(), cm.n_matchcat_box(),
                   cm.eval(), ic13['hmean'], ic13['precision'], ic13['recall'], cm.f1_score()]

            if printf: print(row)
            records.append(row)
            i += step
        df = pd.DataFrame.from_records(records, columns=columns)

        if printf:
            with pd.option_context('display.max_colwidth', -1, 'display.max_columns', 20,
                                   'display.width', 200):  # 上下文控制格式
                print(df)

        return df


class _CocoParser(CocoEval):
    def __init__(self, gt, dt=None, iou_type='bbox', *, min_score=0, printf=False):
        """ coco格式相关分析工具，dt不输入也行，当做没有任何识别结果处理~~
            相比CocoMatch比较轻量级，不会初始化太久，但提供了一些常用的基础功能
        """
        super().__init__(gt, dt, iou_type, min_score=min_score, printf=printf)
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
            df.rename(columns={'category_id': 'dt_category_id'}, inplace=True)
            # 3 筛选最终使用的表格及顺序
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

    def labelme_gt(self, imdir, dst_dir=None, *, segmentation=False, max_workers=4):
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
            height, width = lm.img.size()  # 也可以用image['height'], image['width']获取
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


class _CocoMatchBase:
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


class CocoMatch(_CocoParser, _CocoMatchBase):
    def __init__(self, gt, dt=None, *, min_score=0, eval_im=True, printf=False):
        """ coco格式相关分析工具，dt不输入也行，当做没有任何识别结果处理~~

        :param min_score: 滤除dt中score小余min_score的框
        :param eval_im: 是否对每张图片计算coco分数
        """
        # 因为这里 CocoEval、_CocoMatchBase 都没有父级，不会出现初始化顺序混乱问题
        #   所以我直接指定类初始化顺序了，没用super
        _CocoParser.__init__(self, gt, dt, min_score=min_score)
        match_anns = self._get_match_anns_df(printf=printf)
        _CocoMatchBase.__init__(self, match_anns)
        self.images = self._get_match_images_df(eval_im=eval_im, printf=printf)
        self.categories = self._get_match_categories_df()

    def _get_match_anns_df(self, *, printf=False):
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
        gt_default = [-1, -1, '', 0]  # 没有配对项时填充的默认值
        if 'label' in self.gt_anns.columns:
            gt_columns.append('label')
            gt_default.append('')

        dt_columns = ['dt_category_id', 'dt_ltrb', 'dt_score']
        dt_default = [-1, '', 0]

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
                                    f'_get_match_anns_df, groups={len(self.images)}', disable=not printf):
            # 3.1 计算匹配项
            # gt和dt关于某张图都有可能没有框
            # 比如合同检测，有的图可能什么类别对象都没有，gt中这张图本来就没有box；dt检测也是，某张图不一定有结果
            gt_group_df = gt_df.get_group(image_id) if image_id in gt_df.groups else []
            dt_group_df = dt_df.get_group(image_id) if image_id in dt_df.groups else []
            n, m = len(gt_group_df), len(dt_group_df)

            pairs = []
            if n and m:
                # 任意多边形相交面积算法速度太慢
                # gt_bboxes = [shapely_polygon(b) for b in gt_group_df['gt_ltrb']]  # noqa 已经用if做了判断过滤
                # dt_bboxes = [shapely_polygon(b) for b in dt_group_df['dt_ltrb']]  # noqa
                # pairs = matchpairs(gt_bboxes, dt_bboxes, ComputeIou.polygon2, index=True)

                # 改成ltrb的相交面积算法会快一点
                # gt_bboxes = [shapely_polygon(b) for b in gt_group_df['gt_ltrb']]  # noqa 已经用if做了判断过滤
                # dt_bboxes = [shapely_polygon(b) for b in dt_group_df['dt_ltrb']]  # noqa
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

    def _get_match_images_df(self, *, eval_im=True, printf=False):
        """ 在原有images基础上，扩展一些图像级别的识别结果情况数据 """
        # 1 初始化，新增字段
        images, match_anns = self.images.copy(), self.match_anns.groupby('image_id')
        columns = ['coco_score', 'n_gt_box', 'n_dt_box', 'n_match0.5_box', 'n_matchcat0.5_box', 'f1_micro0.5',
                   'ic13_score']
        for c in columns:
            images[c] = -1.0

        # 2 填写扩展字段的值
        for image_id in tqdm(images.index, '_get_match_images_df', disable=not printf):
            # 2.1 跳过不存在的图
            if image_id not in match_anns.groups:
                continue
            df = match_anns.get_group(image_id)

            # 2.2 增加每张图片的coco分数
            if eval_im:
                images.loc[image_id, 'coco_score'] = self.eval([image_id])

            # 2.3 增加每张图片的多分类分数
            m = _CocoMatchBase(df)
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
        self.eval(printf=True)

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

    def _labelme_match(self, match_func_name, imdir, dst_dir=None, *, segmentation=False, hide_match_dt=False,
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
            if dst_dir:
                imfile = imfile.copy(dst_dir, if_exists='skip')
            lm = Coco2Labelme(imfile)

            height, width = lm.data['imageHeight'], lm.data['imageWidth']
            # 注意df取出来的image_id默认是int64类型，要转成int，否则json会保存不了int64类型
            lm.add_shape('', [0, 0, 10, 0], shape_type='line', shape_color=[0, 0, 0],
                         size=f'{height}x{width}', **(image.to_dict()))
            getattr(lm, match_func_name)(df, segmentation=segmentation, hide_match_dt=hide_match_dt, **kwargs)
            lm.write()  # 保存json文件到img对应目录下

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

    def labelme_match(self, imdir, dst_dir=None, *, segmentation=False, hide_match_dt=False):
        self._labelme_match('anns_match', imdir, dst_dir, segmentation=segmentation, hide_match_dt=hide_match_dt)

    def labelme_match2(self, imdir, dst_dir=None, *, segmentation=False, hide_match_dt=False, colormap=LABEL_COLORMAP7):
        self._labelme_match('anns_match2', imdir, dst_dir, segmentation=segmentation, hide_match_dt=hide_match_dt,
                            colormap=colormap)
