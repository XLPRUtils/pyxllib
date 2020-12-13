#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/08/15 00:59


from pyxllib.basic import *
from pyxllib.debug import pd
from pyxllib.cv import imread, np_array, np


def is_labelme_json_data(data):
    """ 是labelme的标注格式
    :param data: dict
    :return: True or False
    """
    has_keys = set('version flags shapes imagePath imageData imageHeight imageWidth'.split())
    return not (has_keys - data.keys())


def reduce_labelme_jsonfile(jsonpath):
    """ 删除imageData """
    p = File(jsonpath)
    data = p.read(mode='.json')
    if is_labelme_json_data(data) and data['imageData']:
        data['imageData'] = None
        p.write(data, encoding=p.encoding, if_exists='delete')


class ToLabelmeJson:
    """ 标注格式转label形式

    初始化最好带有图片路径，能获得一些相关有用的信息
    然后自定义实现一个 get_data 接口，实现self.data的初始化，运行完可以从self.data取到字典数据
        根据需要可以定制自己的shape，修改get_shape函数
    可以调用write写入文件

    具体用法可以参考 handzuowen https://www.yuque.com/xlpr/datalabel/wzu73p 的数据处理代码
    """

    def __init__(self, imgpath=None):
        """
        :param imgpath: 可选参数图片路径，强烈建议要输入，否则建立的label json会少掉图片宽高信息
        """
        self.imgpath = File(imgpath)
        # 读取图片数据，在一些转换规则比较复杂，有可能要用到原图数据
        if self.imgpath:
            self.img = imread(str(self.imgpath))
        else:
            self.img = None
        self.data = None  # 存储json的字典数据

    def get_data(self, infile):
        """ 格式转换接口函数，继承的类需要自己实现这个方法

        :param infile: 待解析的标注数据
        """
        # s = Path(infile).read()
        # self.data = self.get_data_base()
        raise NotImplementedError('get_data方法必须在子类中实现')

    def get_data_base(self, name='', height=0, width=0):
        """ 获得一个labelme标注文件的框架

        如果初始化时没有输入图片，也可以这里传入name等的值
        """
        # 1 默认属性，和图片名、尺寸
        if self.imgpath:
            name = self.imgpath.name
            height, width = self.img.shape[:2]
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

    def get_shape(self, label, points, shape_type=None, dtype=None):
        """最基本的添加形状功能

        :param shape_type: 会根据points的点数量，智能判断类型，默认一般是polygon
        :param dtype: 可以重置points的存储数值类型，一般是浮点数，可以转成整数更精简
        """
        # 1 优化点集数据格式
        points = np_array(points, dtype).reshape(-1, 2).tolist()
        # 2 判断形状类型
        if shape_type is None:
            m = len(points)
            if m == 2:
                shape_type = 'rectangle'
            elif m >= 3:
                shape_type = 'polygon'
            else:
                raise ValueError
        # 3 创建标注
        shape = {'flags': {},
                 'group_id': None,
                 'label': str(label),
                 'points': points,
                 'shape_type': shape_type}
        return shape

    def write(self, dst=None, if_exists='delete'):
        """
        :param dst: 往dst目标路径存入json文件，默认名称在self.imgpath同目录的同名json文件
        :return: 写入后的文件路径
        """
        if dst is None and self.imgpath:
            dst = self.imgpath.with_suffix('.json')
        return File(dst).write(self.data, if_exists=if_exists)


def get_labelme_shapes_df(dir, pattern='**/*.json', max_workers=None, pinterval=None, **kwargs):
    """ 获得labelme文件的shapes清单列表

    :param max_workers: 这个运算还不是很快，默认需要开多线程
    """

    def func(p):
        data = p.read()
        if not data['shapes']: return
        df = pd.DataFrame.from_records(data['shapes'])
        df['filename'] = p.relpath(dir)
        # 坐标转成整数，看起来比较精简点
        df['points'] = [np.array(v, dtype=int).tolist() for v in df['points']]
        li.append(df)

    li = []
    Dir(dir).select(pattern).procpaths(func, max_workers=max_workers, pinterval=pinterval, **kwargs)
    shapes_df = pd.concat(li)
    # TODO flags和group_id字段可以放到最后面
    shapes_df.reset_index(inplace=True, drop=True)

    return shapes_df
