#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/08/15 00:59

import os
from tqdm import tqdm
import json
import ujson
import copy
from collections import Counter

import numpy as np

from pyxllib.prog.newbie import round_int
from pyxllib.prog.pupil import DictTool
from pyxllib.prog.specialist import get_xllog, Iterate
from pyxllib.file.specialist import PathGroups, get_encoding, XlPath
from pyxllib.prog.specialist import mtqdm
from pyxllib.cv.expert import xlpil
from pyxllib.algo.geo import ltrb2xywh, rect_bounds, warp_points, resort_quad_points, rect2polygon, get_warp_mat


def __0_basic():
    """ 这里可以写每个模块注释 """


class BasicLabelDataset:
    """ 一张图一份标注文件的一些基本操作功能 """

    def __init__(self, root, relpath2data=None, *, reads=True, prt=False, fltr=None, slt=None, extdata=None):
        """
        :param root: 数据所在根目录
        :param dict[str, readed_data] relpath2data: {relpath: data1, 'a/1.txt': data2, ...}
            如果未传入data具体值，则根据目录里的情况自动初始化获得data的值

            relpath是对应的XlPath标注文件的相对路径字符串
            data1, data2 是读取的标注数据，根据不同情况，会存成不同格式
                如果是json则直接保存json内存对象结构
                如果是txt可能会进行一定的结构化解析存储
        :param extdata: 可以存储一些扩展信息内容
        :param fltr: filter的缩写，PathGroups 的过滤规则。一般用来进行图片匹配。
            None，没有过滤规则，就算不存在slt格式的情况下，也会保留分组
            'json'等字符串规则, 使用 select_group_which_hassuffix，必须含有特定后缀的分组
            judge(k, v)，自定义函数规则
        :param slt: select的缩写，要选中的标注文件后缀格式
            如果传入slt参数，该 Basic 基础类只会预设好 file 参数，数据部分会置 None，需要后续主动读取

        >> BasicLabelData('textGroup/aabb', {'a.json': ..., 'a/1.json': ...})
        >> BasicLabelData('textGroup/aabb', slt='json')
        >> BasicLabelData('textGroup/aabb', fltr='jpg', slt='json')  # 只获取有对应jpg图片的json文件
        >> BasicLabelData('textGroup/aabb', fltr='jpg|png', slt='json')
        """

        # 1 基础操作
        root = XlPath(root)
        self.root, self.rp2data, self.extdata = root, relpath2data or {}, extdata or {}
        self.pathgs = None

        if relpath2data is not None or slt is None:
            return

        # 2 如果没有默认data数据，以及传入slt参数，则需要使用默认文件关联方式读取标注
        relpath2data = {}
        gs = PathGroups.groupby(XlPath(root).rglob_files())
        if isinstance(fltr, str):
            gs = gs.select_group_which_hassuffix(fltr)
        elif callable(fltr):
            gs = gs.select_group(fltr)
        self.pathgs = gs

        # 3 读取数据
        for stem, suffixs in tqdm(gs.data.items(), f'{self.__class__.__name__}读取数据', disable=not prt):
            f = XlPath(stem + f'.{slt}')
            if reads and f.exists():
                # dprint(f)  # 空json会报错：json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
                relpath2data[f.relpath(self.root)] = f.read_auto()
            else:
                relpath2data[f.relpath(self.root)] = None

        self.rp2data = relpath2data

    def __len__(self):
        return len(self.rp2data)

    def read(self, relpath, **kwargs):
        """
        :param relpath: 必须是斜杠表示的相对路径 'a/1.txt'、'b/2.json'
        """
        self.rp2data[relpath] = (self.root / relpath).read_auto(**kwargs)

    def reads(self, prt=False, **kwargs):
        """ 为了性能效率，初始化默认不会读取数据，需要调用reads才会开始读取数据 """
        for k in tqdm(self.rp2data.keys(), f'读取{self.__class__.__name__}数据', disable=not prt):
            self.rp2data[k] = (self.root / k).read_auto(**kwargs)

    def write(self, relpath, **kwargs):
        """
        :param relpath: 必须是斜杠表示的相对路径 'a/1.txt'、'b/2.json'
        """
        data = self.rp2data[relpath]
        file = self.root / relpath
        if file.is_file():  # 如果文件存在，要遵循原有的编码规则
            with open(str(file), 'rb') as f:
                bstr = f.read()
            encoding = get_encoding(bstr)
            kwargs['encoding'] = encoding
            kwargs['if_exists'] = 'replace'
            file.write_auto(data, **kwargs)
        else:  # 否则直接写入
            file.write_auto(data, **kwargs)

    def writes(self, *, max_workers=8, print_mode=False, **kwargs):
        """ 重新写入每份标注文件

        可能是内存里修改了数据，需要重新覆盖
        也可能是从coco等其他格式初始化，转换而来的内存数据，需要生成对应的新标注文件
        """
        mtqdm(lambda x: self.write(x, **kwargs), self.rp2data.keys(), desc=f'{self.__class__.__name__}写入标注数据',
              max_workers=max_workers, disable=not print_mode)


def __1_labelme():
    """ """


# 我自己按照“红橙黄绿蓝靛紫”的顺序展示
LABEL_COLORMAP7 = [(0, 0, 0), (255, 0, 0), (255, 125, 0), (255, 255, 0),
                   (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]


def is_labelme_json_data(data):
    """ 是labelme的标注格式
    :param data: dict
    :return: True or False
    """
    if not isinstance(data, dict):
        return False
    has_keys = set('version flags shapes imagePath imageData imageHeight imageWidth'.split())
    return not (has_keys - data.keys())


def reduce_labelme_jsonfile(jsonpath):
    """ 删除imageData """
    p = str(jsonpath)

    with open(p, 'rb') as f:
        bstr = f.read()
    encoding = get_encoding(bstr)
    data = ujson.loads(bstr.decode(encoding=encoding))

    if is_labelme_json_data(data) and data['imageData']:
        data['imageData'] = None
        XlPath(p).write_json(data, encoding=encoding, if_exists='replace')


def reduce_labelme_dir(d, print_mode=False):
    """ 精简一个目录里的所有labelme json文件 """

    def printf(*args, **kwargs):
        if print_mode:
            print(*args, **kwargs)

    i = 0
    for f in XlPath(d).rglob_files('*.json'):
        data = f.read_json()
        if data.get('imageData'):
            data['imageData'] = None
            f.write_json(data)
            i += 1
            printf(i, f)


class ToLabelmeJson:
    """ 标注格式转label形式

    初始化最好带有图片路径，能获得一些相关有用的信息
    然后自定义实现一个 get_data 接口，实现self.data的初始化，运行完可以从self.data取到字典数据
        根据需要可以定制自己的shape，修改get_shape函数
    可以调用write写入文件

    document: https://www.yuque.com/xlpr/pyxllib/ks5h4o
    """

    # 可能有其他人会用我库的高级接口，不应该莫名其妙报警告。除非我先实现自己库内该功能的剥离
    # @deprecated(reason='建议使用LabelmeData实现')
    def __init__(self, imgpath):
        """
        :param imgpath: 可选参数图片路径，强烈建议要输入，否则建立的label json会少掉图片宽高信息
        """
        self.imgpath = XlPath(imgpath)
        # 读取图片数据，在一些转换规则比较复杂，有可能要用到原图数据
        if self.imgpath:
            # 一般都只需要获得尺寸，用pil读取即可，速度更快，不需要读取图片rgb数据
            self.img = xlpil.read(self.imgpath)
        else:
            self.img = None
        self.data = self.get_data_base()  # 存储json的字典数据

    def get_data(self, infile):
        """ 格式转换接口函数，继承的类需要自己实现这个方法

        :param infile: 待解析的标注数据
        """
        raise NotImplementedError('get_data方法必须在子类中实现')

    def get_data_base(self, name='', height=0, width=0):
        """ 获得一个labelme标注文件的框架 （这是标准结构，也可以自己修改定制）

        如果初始化时没有输入图片，也可以这里传入name等的值
        """
        # 1 默认属性，和图片名、尺寸
        if self.imgpath:
            name = self.imgpath.name
            height, width = self.img.height, self.img.width
        # 2 构建结构框架
        data = {'version': '4.5.6',
                'flags': {},
                'shapes': [],
                'imagePath': name,
                'imageData': None,
                'imageWidth': width,
                'imageHeight': height,
                }
        return data

    def get_shape(self, label, points, shape_type=None, dtype=None, group_id=None, **kwargs):
        """ 最基本的添加形状功能

        :param shape_type: 会根据points的点数量，智能判断类型，默认一般是polygon
            其他需要自己指定的格式：line、circle
        :param dtype: 可以重置points的存储数值类型，一般是浮点数，可以转成整数更精简
        :param group_id: 本来是用来分组的，但其值会以括号的形式添加在label后面，可以在可视化中做一些特殊操作
        """
        # 1 优化点集数据格式
        points = np.array(points, dtype=dtype).reshape(-1, 2).tolist()
        # 2 判断形状类型
        if shape_type is None:
            m = len(points)
            if m == 1:
                shape_type = 'point'
            elif m == 2:
                shape_type = 'rectangle'
            elif m >= 3:
                shape_type = 'polygon'
            else:
                raise ValueError
        # 3 创建标注
        shape = {'flags': {},
                 'group_id': group_id,
                 'label': str(label),
                 'points': points,
                 'shape_type': shape_type}
        shape.update(kwargs)
        return shape

    def get_shape2(self, **kwargs):
        """ 完全使用字典的接口形式 """
        label = kwargs.get('label', '')
        points = kwargs['points']  # 这个是必须要有的字段
        kw = copy.deepcopy(kwargs)
        del kw['label']
        del kw['points']
        return self.get_shape(label, points, **kw)

    def add_shape(self, *args, **kwargs):
        self.data['shapes'].append(self.get_shape(*args, **kwargs))

    def add_shape2(self, **kwargs):
        self.data['shapes'].append(self.get_shape2(**kwargs))

    def write(self, dst=None, if_exists='replace'):
        """
        :param dst: 往dst目标路径存入json文件，默认名称在self.imgpath同目录的同名json文件
        :return: 写入后的文件路径
        """
        if dst is None and self.imgpath:
            dst = self.imgpath.with_suffix('.json')
        # 官方json支持indent=None的写法，但是ujson必须要显式写indent=0
        return XlPath(dst).write_auto(self.data, if_exists=if_exists, indent=0)

    @classmethod
    def create_json(cls, imgpath, annotation):
        """ 输入图片路径p，和对应的annotation标注数据（一般是对应目录下txt文件） """
        try:
            obj = cls(imgpath)
        except TypeError as e:  # 解析不了的输出错误日志
            get_xllog().exception(e)
            return
        obj.get_data(annotation)
        obj.write()  # 保存json文件到img对应目录下

    @classmethod
    def main_normal(cls, imdir, labeldir=None, label_file_suffix='.txt'):
        """ 封装更高层的接口，输入目录，直接标注目录下所有图片

        :param imdir: 图片路径
        :param labeldir: 标注数据路径，默认跟imdir同目录
        :return:
        """
        ims = XlPath(imdir).rglob_images()
        if not labeldir: labeldir = imdir
        txts = [(XlPath(labeldir) / (f.stem + label_file_suffix)) for f in ims]
        cls.main_pair(ims, txts)

    @classmethod
    def main_pair(cls, images, labels):
        """ 一一配对匹配处理 """
        Iterate(zip(images, labels)).run(lambda x: cls.create_json(x[0], x[1]),
                                         pinterval='20%', max_workers=8)


class Quad2Labelme(ToLabelmeJson):
    """ 四边形类标注转labelme """

    def get_data(self, infile):
        lines = XlPath(infile).read_text().splitlines()
        for line in lines:
            # 一般是要改这里，每行数据的解析规则
            vals = line.split(',', maxsplit=8)
            if len(vals) < 9: continue
            pts = [int(v) for v in vals[:8]]  # 点集
            label = vals[-1]  # 标注的文本
            # get_shape还有shape_type形状参数可以设置
            #  如果是2个点的矩形，或者3个点以上的多边形，会自动判断，不用指定shape_type
            self.add_shape(label, pts)


class LabelmeDict:
    """ Labelme格式的字典数据

    这里的成员函数基本都是原地操作
    """

    @classmethod
    def gen_data(cls, imfile=None, **kwargs):
        """ 主要框架结构
        :param imfile: 可以传入一张图片路径
        """
        # 1 传入图片路径的初始化
        if imfile:
            file = XlPath(imfile)
            name = file.name
            img = xlpil.read(file)
            height, width = img.height, img.width
        else:
            name, height, width = '', 0, 0

        # 2 字段值
        data = {'version': '5.0.1',
                'flags': {},
                'shapes': [],
                'imagePath': name,
                'imageData': None,
                'imageWidth': width,
                'imageHeight': height,
                }
        if kwargs:
            data.update(kwargs)
        return data

    @classmethod
    def gen_ocr_data(cls, imfile=None, **kwargs):
        """" 支持调用PaddleOCR进行预识别

        该接口是为了方便性保留，更推荐使用 PaddleOCR.labelme_ocr 的功能进行批量识别
        """
        from paddleocr import PaddleOCR
        ppocr = PaddleOCR.get_paddleocr()

        data = cls.gen_data(imfile, **kwargs)
        lines = ppocr.ocr(str(imfile))
        for line in lines:
            pts, [text, score] = line
            pts = [[int(p[0]), int(p[1])] for p in pts]  # 转整数
            sp = cls.gen_shape({'text': text, 'score': round(float(score), 4)}, pts)
            data['shapes'].append(sp)
        return data

    @classmethod
    def gen_shape(cls, label, points, shape_type=None, dtype=None, group_id=None, **kwargs):
        """ 最基本的添加形状功能

        :param label: 支持输入dict类型，会编码为json格式的字符串
        :param shape_type: 会根据points的点数量，智能判断类型，默认一般是polygon
            其他需要自己指定的格式：line、circle
        :param dtype: 可以重置points的存储数值类型，一般是浮点数，可以转成整数更精简
        :param group_id: 本来是用来分组的，但其值会以括号的形式添加在label后面，可以在可视化中做一些特殊操作
        """
        # 1 优化点集数据格式
        points = np.array(points, dtype=dtype).reshape(-1, 2).tolist()
        # 2 判断形状类型
        if shape_type is None:
            m = len(points)
            if m == 1:
                shape_type = 'point'
            elif m == 2:
                shape_type = 'rectangle'
            elif m >= 3:
                shape_type = 'polygon'
            else:
                raise ValueError
        # 3 创建标注
        if isinstance(label, dict):
            label = json.dumps(label, ensure_ascii=False)
        shape = {'flags': {},
                 'group_id': group_id,
                 'label': str(label),
                 'points': points,
                 'shape_type': shape_type}
        shape.update(kwargs)
        return shape

    @classmethod
    def gen_shape2(cls, **kwargs):
        """ 完全使用字典的接口形式 """
        label = kwargs.get('label', '')
        points = kwargs['points']  # 这个是必须要有的字段
        kw = copy.deepcopy(kwargs)
        if 'label' in kw:
            del kw['label']
        if 'points' in kw:
            del kw['points']
        return cls.gen_shape(label, points, **kw)

    @classmethod
    def reduce(cls, lmdict, *, inplace=True):
        if not inplace:
            lmdict = copy.deepcopy(lmdict)

        lmdict['imageData'] = None
        return lmdict

    @classmethod
    def refine_structure(cls, old_json_path, *, old_img_path=None,
                         new_stem_name=None, new_img_suffix=None):
        """ 重置labelme标注文件，这是一个比较综合的调整优化接口

        :param old_json_path: 原json路径
        :param old_img_path: 原图路径，可以不填，从old_json_path推算出来
        :param new_stem_name: 新的stem昵称，没写的时候，以json的stem为准
        :param new_img_suffix: 是否要调整图片后缀格式，常用图图片格式统一操作
            没写的时候，以找到的图片为准，如果图片没找到，则以imagePath的后缀为准
        """
        from pyxllib.cv.expert import xlcv

        # 1 参数解析
        old_json_path = XlPath(old_json_path)
        parent = old_json_path.parent
        lmdict = old_json_path.read_json()

        if old_img_path is None:
            old_img_path = parent / lmdict['imagePath']
            if not old_img_path.is_file():
                # 如果imagePath的图片并不存在，需要用json的名称去推导，如果也还是不存在，就按照imagePath的后缀设置
                try:
                    old_img_path = next(parent.glob_images(f'{old_json_path.stem}.*'))
                except StopIteration:
                    old_img_path = parent / (old_json_path.stem + XlPath(lmdict['imagePath']).suffix)

        if new_stem_name is None:
            new_stem_name = old_json_path.stem

        if new_img_suffix is None:
            new_img_suffix = old_img_path.suffix

        # 2 重命名、重置
        new_json_path = parent / (new_stem_name + '.json')
        new_img_path = parent / (new_stem_name + new_img_suffix)

        # 优化json数据
        cls.reduce(lmdict)
        lmdict['imagePath'] = new_img_path.name
        new_json_path.write_json(lmdict)
        if new_json_path.as_posix() != old_json_path.as_posix():
            old_json_path.delete()

        # TODO points浮点过长的优化？xllabelme默认优化了？

        # 优化图片
        if old_img_path.is_file():
            xlcv.write(xlcv.read(old_img_path), new_img_path)
            if new_img_path.as_posix() != old_img_path.as_posix():
                old_img_path.delete()

    @classmethod
    def flip_points(cls, lmdict, direction, *, inplace=True):
        """
        :param direction: points的翻转方向
            1表示顺时针转90度，2表示顺时针转180度...
            -1表示逆时针转90度，...
        :return:
        """
        if not inplace:
            lmdict = copy.deepcopy(lmdict)

        w, h = lmdict['imageWidth'], lmdict['imageHeight']
        pts = [[[0, 0], [w, 0], [w, h], [0, h]],
               [[h, 0], [h, w], [0, w], [0, 0]],
               [[w, h], [0, h], [0, 0], [w, 0]],
               [[0, w], [0, 0], [h, 0], [h, w]]]
        warp_mat = get_warp_mat(pts[0], pts[direction % 4])

        if direction % 2:
            lmdict['imageWidth'], lmdict['imageHeight'] = lmdict['imageHeight'], lmdict['imageWidth']
        shapes = lmdict['shapes']
        for i, shape in enumerate(shapes):
            pts = [warp_points(x, warp_mat)[0].tolist() for x in shape['points']]
            if shape['shape_type'] == 'rectangle':
                pts = resort_quad_points(rect2polygon(pts))
                shape['points'] = [pts[0], pts[2]]
            elif shape['shape_type'] == 'polygon' and len(pts) == 4:
                shape['points'] = resort_quad_points(pts)
            else:  # 其他形状暂不处理，也不报错
                pass
        return lmdict

    @classmethod
    def flip_img_and_json(cls, impath, direction):
        """  旋转impath，如果有对应的json也会自动处理
        demo_flip_labelme，演示如何使用翻转图片、labelme标注功能

        :param XlPath impath: 图片路径
        :param direction: 标记现在图片是哪个方向：0是正常，1是向右翻转，2是向下翻转，3是向左翻转
            顺时针0123表示当前图片方向
            甚至该参数可以设成None，没有输入的时候调用模型识别结果，不过那个模型不是很准确，先不搞这种功能
        """
        # 图片旋转
        im = xlpil.read(impath)
        im = xlpil.flip_direction(im, direction)
        xlpil.write(im, impath)

        # json格式
        jsonpath = impath.with_suffix('.json')
        if jsonpath.exists():
            lmdict = jsonpath.read_json('utf8')  # 必须是labelme的格式，其他格式不支持处理哦
            cls.flip_points(lmdict, -direction)  # 默认是inplace操作
            jsonpath.write_json(lmdict, 'utf8')

    @classmethod
    def update_labelattr(cls, lmdict, *, points=False, inplace=True):
        """

        :param points: 是否更新labelattr中的points、bbox等几何信息
            并且在无任何几何信息的情况下，增设points
        """
        if not inplace:
            lmdict = copy.deepcopy(lmdict)

        for shape in lmdict['shapes']:
            # 1 属性字典，至少先初始化一个label属性
            labelattr = DictTool.json_loads(shape['label'], 'label')
            # 2 填充其他扩展属性值
            keys = set(shape.keys())
            stdkeys = set('label,points,group_id,shape_type,flags'.split(','))
            for k in (keys - stdkeys):
                labelattr[k] = shape[k]
                del shape[k]  # 要删除原有的扩展字段值

            # 3 处理points等几何信息
            if points:
                if 'bbox' in labelattr:
                    labelattr['bbox'] = ltrb2xywh(rect_bounds(shape['points']))
                else:
                    labelattr['points'] = shape['points']

            # + 写回shape
            shape['label'] = json.dumps(labelattr, ensure_ascii=False)
        return lmdict

    @classmethod
    def to_quad_pts(cls, shape):
        """ 将一个形状标注变成4个点标注的四边形 """
        pts = shape['points']
        t = shape['shape_type']
        if t == 'rectangle':
            return rect2polygon(pts)
        elif t == 'polygon':
            if len(pts) != 4:
                # 暂用外接矩形代替
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                r = [(min(xs), min(ys)), (max(xs), max(ys))]
                pts = rect2polygon(r)
            return pts
        else:
            raise NotImplementedError(f'{t}')


class LabelmeDataset(BasicLabelDataset):
    def __init__(self, root, relpath2data=None, *, reads=True, prt=False, fltr='json', slt='json', extdata=None):
        """
        :param root: 文件根目录
        :param relpath2data: {jsonfile: lmdict, ...}，其中 lmdict 为一个labelme文件格式的标准内容
            如果未传入data具体值，则根据目录里的情况自动初始化获得data的值

            210602周三16:26，为了工程等一些考虑，删除了 is_labelme_json_data 的检查
                尽量通过 fltr、slt 的机制选出正确的 json 文件
        """
        super().__init__(root, relpath2data, reads=reads, prt=prt, fltr=fltr, slt=slt, extdata=extdata)

        # 已有的数据已经读取了，这里要补充空labelme标注
        if self.pathgs:
            for stem, suffixs in tqdm(self.pathgs.data.items(), f'{self.__class__.__name__}优化数据', disable=not prt):
                f = XlPath(stem + f'.{slt}')
                if reads and not f.exists():
                    self.rp2data[f.relpath(self.root)] = LabelmeDict.gen_data(XlPath.init(stem, suffix=suffixs[0]))

        # 优化rp2data，去掉一些并不是labelme的字典
        rp2data = {}
        for k, v in self.rp2data.items():
            if is_labelme_json_data(v):
                rp2data[k] = v
        self.rp2data = rp2data

    def reduces(self):
        """ 移除imageData字段值 """
        for lmdict in self.rp2data.values():
            LabelmeDict.reduce(lmdict)

    def refine_structures(self, *, img_suffix=None):
        """ 整套labelme数据的重置

        :param img_suffix: 是否统一图片的后缀格式，比如.jpg

        不过不同的情景问题不同，请了解清楚这个函数的算法逻辑，能解决什么性质的问题后，再调用
        """
        # 有些字典的imagePath可能有错误，可以调用该方法修正
        for jsonfile in tqdm(self.rp2data.keys(), desc='labelme字典优化'):
            LabelmeDict.refine_structure(self.root / jsonfile, new_img_suffix=img_suffix)

    def update_labelattrs(self, *, points=False):
        """ 将shape['label'] 升级为字典类型

        可以处理旧版不动产标注 content_class 等问题
        """
        for jsonfile, lmdict in self.rp2data.items():
            LabelmeDict.update_labelattr(lmdict, points=points)

    def to_excel(self, savepath):
        """ 转成dataframe表格查看

        这个细节太多，可以 labelme 先转 coco 后，借助 coco 转 excel
            coco 里会给 image、box 编号，能显示一些补充属性
        """
        from pyxlpr.data.coco import CocoParser
        gt_dict = self.to_coco_gt_dict()
        CocoParser(gt_dict).to_excel(savepath)

    @classmethod
    def plot(self, img, lmdict):
        """ 将标注画成静态图 """
        raise NotImplementedError

    def to_coco_gt_dict(self, categories=None):
        """ 将labelme转成 coco gt 标注的格式

        分两种大情况
        1、一种是raw原始数据转labelme标注后，首次转coco格式，这种编号等相关数据都可以重新生成
            raw_data --可视化--> labelme --转存--> coco
        2、还有种原来就是coco，转labelme修改标注后，又要再转回coco，这种应该尽量保存原始值
            coco --> labelme --手动修改--> labelme' --> coco'
            这种在coco转labelme时，会做一些特殊标记，方便后续转回coco
        3、 1, 2两种情况是可以连在一起，然后形成 labelme 和 coco 之间的多次互转的

        :param categories: 类别
            默认只设一个类别 {'id': 0, 'name': 'text', 'supercategory'}
            支持自定义，所有annotations的category_id
        :return: gt_dict
            注意，如果对文件顺序、ann顺序有需求的，请先自行操作self.data数据后，再调用该to_coco函数
            对image_id、annotation_id有需求的，需要使用CocoData进一步操作
        """
        from pyxlpr.data.coco import CocoGtData

        if not categories:
            if 'categories' in self.extdata:
                # coco 转过来的labelme，存储有原始的 categories
                categories = self.extdata['categories']
            else:
                categories = [{'id': 0, 'name': 'text', 'supercategory': ''}]

        # 1 第一轮遍历：结构处理 jsonfile, lmdict --> data（image, shapes）
        img_id, ann_id, data = 0, 0, []
        for jsonfile, lmdict in self.rp2data.items():
            # 1.0 升级为字典类型
            lmdict = LabelmeDict.update_labelattr(lmdict, points=True)

            for sp in lmdict['shapes']:  # label转成字典
                sp['label'] = json.loads(sp['label'])

            # 1.1 找shapes里的image
            image = None
            # 1.1.1 xltype='image'
            for sp in filter(lambda x: x.get('xltype', None) == 'image', lmdict['shapes']):
                image = DictTool.json_loads(sp['label'])
                if not image:
                    raise ValueError(sp['label'])
                # TODO 删除 coco_eval 等字段？
                del image['xltype']
                break
            # 1.1.2 shapes里没有图像级标注则生成一个
            if image is None:
                # TODO file_name 加上相对路径？
                image = CocoGtData.gen_image(-1, lmdict['imagePath'],
                                             lmdict['imageHeight'], lmdict['imageWidth'])
            img_id = max(img_id, image.get('id', -1))

            # 1.2 遍历shapes
            shapes = []
            for sp in lmdict['shapes']:
                label = sp['label']
                if 'xltype' not in label:
                    # 普通的标注框
                    d = sp['label'].copy()
                    # DictTool.isub_(d, '')
                    ann_id = max(ann_id, d.get('id', -1))
                    shapes.append(d)
                elif label['xltype'] == 'image':
                    # image，图像级标注数据；之前已经处理了，这里可以跳过
                    pass
                elif label['xltype'] == 'seg':
                    # seg，衍生的分割标注框，在转回coco时可以丢弃
                    pass
                else:
                    raise ValueError
            data.append([image, shapes])

        # 2 第二轮遍历：处理id等问题
        images, annotations = [], []
        for image, shapes in data:
            # 2.1 image
            if image.get('id', -1) == -1:
                img_id += 1
                image['id'] = img_id
            images.append(image)

            # 2.2 annotations
            for sp in shapes:
                sp['image_id'] = img_id
                if sp.get('id', -1) == -1:
                    ann_id += 1
                    sp['id'] = ann_id
                # 如果没有框类别，会默认设置一个。 （强烈建议外部业务功能代码自行设置好category_id）
                if 'category_id' not in sp:
                    sp['category_id'] = categories[0]['id']
                DictTool.isub(sp, ['category_name'])
                ann = CocoGtData.gen_annotation(**sp)
                annotations.append(ann)

        # 3 result
        gt_dict = CocoGtData.gen_gt_dict(images, annotations, categories)
        return gt_dict

    def to_ppdet(self, outfile=None, print_mode=True):
        """ 转paddle的文本检测格式

        图片要存相对目录，默认就按self的root参数设置
        """
        lines = []

        # 1 转成一行行标注数据
        for jsonfile, lmdict in tqdm(self.rp2data.items(), disable=not print_mode):
            shapes = []  # pp格式的标注清单

            for sp in lmdict['shapes']:
                attrs = DictTool.json_loads(sp['label'], 'text')
                d = {'transcription': attrs['text'],
                     'points': round_int(LabelmeDict.to_quad_pts(sp), ndim=2)}
                shapes.append(d)
            imfile = os.path.split(jsonfile)[0] + f'/{lmdict["imagePath"]}'
            lines.append(f'{imfile}\t{json.dumps(shapes, ensure_ascii=False)}')

        # 2 输出
        content = '\n'.join(lines)
        if outfile:
            XlPath(outfile).write_text(content)
        return content

    def get_char_count_dict(self):
        """ 文本识别需要用到的功能，检查字符集出现情况

        return dict: 返回一个字典，k是出现的字符，v是各字符出现的次数，按顺序从多到少排序
            差不多是Counter的结构
        """
        texts = []
        for lmdict in self.rp2data.values():
            for sp in lmdict['shapes']:
                text = DictTool.json_loads(sp['label'], 'text')['text']
                texts.append(text)
        ct = Counter(''.join(texts))
        return {k: v for k, v in ct.most_common()}

    def check_char_set(self, refdict=None):
        """ 检查本套labelme数据集里，text字符集出现情况，和paddleocr的识别字典是否有额外新增字符
        """
        from pyxllib.algo.specialist import DictCmper
        from pyxlpr.ppocr.utils import get_dict_content

        # 0 计算字典
        if refdict is None:
            d1 = get_dict_content('ppocr_keys_v1.txt')
            refdict = {k: 1 for k in d1.split('\n')}

        d2 = self.get_char_count_dict()

        print('1 整体统计信息')
        dc = DictCmper({'refdict': refdict, 'chars': d2})
        print(dc.pair_summary())

        print('2 新增字符及出现数量（如果只是多出空白字符，可以统一转空格处理）')
        keys = set(list(d2.keys())) - set(list(refdict.keys()))
        sorted(keys, key=lambda k: -d2[k])
        for k in keys:
            print(repr(k), d2[k])

        # 3 返回所有新增的非空字符
        return {k for k in keys if k.strip()}

    def to_pprec(self, image_dir, txt_path, *, reset=False):
        """ 转paddle的文本识别格式

        :param image_dir: 要导出文本行数据所在的目录
        :param txt_path: 标注文件的路径
        :param reset: 目标目录存在则重置
        """
        pass


class ItemFormula:
    """ 为m2302中科院题库准备的labelme数据相关处理算法
    如果写到labelme中，其他脚本就复用不了了，所以把核心算法功能写到这里

    line_id的功能是动态计算的，参考了与上一个shape的区别，这里暂时不提供静态算法。
    """

    @classmethod
    def check_label(cls, shapes):
        """ 检查数据异常 """
        # "删除" 是否都为手写，并且text为空

    @classmethod
    def joint_label(cls, shapes):
        """ 将shapes的label拼接成一篇文章

        :return: [line1, line2, ...]
        """
        last_line_id = 1
        paper_text = []
        line_text = []
        for sp in shapes:
            label = json.loads(sp['label'])
            if label['line_id'] != last_line_id:
                last_line_id = label['line_id']
                paper_text.append(' '.join(line_text))
                line_text = []

            if label['content_class'] == '公式':
                t = '$' + label['text'] + '$'
            else:
                t = label['text']
            line_text.append(t)

        paper_text.append(' '.join(line_text))

        return paper_text
