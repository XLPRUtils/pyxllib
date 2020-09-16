#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/09/16 11:08

"""
jsondata是轻量级的功能库

而本jsondata2则是依赖pyxllib.cv的一些高级功能
"""

from pyxllib.cv import *


class ToLabelmeJson:
    """ 标注格式转label形式 """

    def __init__(self, imgpath=None):
        """
        :param imgpath: 可选参数图片路径，强烈建议要输入，否则建立的label json会少掉图片宽高信息
        """
        self.imgpath = Path(imgpath)
        # 读取图片数据，在一些转换规则比较复杂，有可能要用到原图数据
        if self.imgpath.is_file():
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
        if self.imgpath.is_file():
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
        points = ensure_nparr(points, dtype).reshape(-1, 2).tolist()
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

    def write(self, dst=None, if_exists='replace'):
        """
        :param dst: 往dst目标路径存入json文件，默认名称在self.imgpath同目录的同名json文件
        :return: 写入后的文件路径
        """
        if dst is None and self.imgpath.is_file():
            dst = self.imgpath.with_suffix('.json')
        return Path(dst).write(self.data, if_exists=if_exists)
