#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/25 09:34

import pathlib

from pyxllib.xl import *
from fvcore.common.registry import Registry

____basic = """
基础组件
"""


class CommonPathBase:
    def __init__(self, prefix=None):
        if prefix is None:
            # 只要在这里设置服务器数据目录；本地目录则会将/根目录映射到D:/盘对应目录
            if os.getenv('PYXLPR_COMMONDIR'):  # 可以使用PYXLPR_COMMONDIR='D:/'、'/'来自定义数据根目录
                prefix = os.getenv('PYXLPR_COMMONDIR')
            else:
                prefix = 'D:/' if sys.platform == 'win32' else '/'
            prefix = XlPath(prefix)  # 默认是当前操作系统的文件类型；也可以显示输入PosixPath格式的prefix

        self.datasets = prefix / 'home/datasets'
        self.huangzhicai = prefix / 'home/huangzhicai'
        self.chenkunze = prefix / 'home/chenkunze'
        self.slns = prefix / 'home/chenkunze/slns'

        # slns 相关
        self.d2configs = self.slns / 'detectron2/configs'
        self.xlproject = self.slns / 'pyxlpr/xlproject'

        # datasets 相关
        self.realestate2020 = self.datasets / 'RealEstate2020'
        self.realestate_coco = self.datasets / 'RealEstate2020/coco_fmt'

        # textGroup 相关
        self.textGroup = self.datasets / 'textGroup'
        self.icdar2013 = self.textGroup / 'ICDAR2013'
        self.ic13loc = self.textGroup / 'ICDAR2013/Task2.1 Text Localization'
        self.publaynet = self.textGroup / 'PubLayNet/publaynet'
        self.AHCDB = self.textGroup / 'AHCDB'
        self.sroie = self.textGroup / 'SROIE2019'  # 智财,占秋原来整理的数据
        self.sroie2 = self.textGroup / 'SROIE2019+'  # 我重新整理过的数据，并且子目录data里有新的数据
        self.cipdf = self.textGroup / 'CIPDF'  # 从cninfo下载的pdf文件数据
        self.cord = self.textGroup / 'CORD'  # 从cninfo下载的pdf文件数据
        self.xeocr1 = self.textGroup / 'Xeon1OCR'  # 从cninfo下载的pdf文件数据

        # chenkunze
        # 项目中一些太大的目录迁移到refdir存储；之前没有想过按chenkunze的目录同步；现在不太需要refdir了
        self.refdir = self.chenkunze / 'refdir'

        # huangzhicai
        self.zclogs = self.huangzhicai / 'workshop/ocrwork/uniocr/logs'
        self.voc2007 = self.huangzhicai / 'data/detec/voc2007/VOCdevkit/VOC2007'


if sys.platform == 'win32':
    common_path = CommonPathBase()
    tp10_common_path = CommonPathBase(pathlib.PurePosixPath('/'))  # 十卡服务器的常用目录
else:
    common_path = CommonPathBase(XlPath('/'))
    tp10_common_path = common_path

____coco = """
普通的coco格式数据

目前需求这样的设计模式够了
1、但其实局限还不少，还有很多不便自定义的（不使用register_coco_instances，更灵活地修改底层）
2、以及在非d2场景的数据引用

不过现在也难想清楚，等后面切实需要的时候再扩展修改
"""

COCO_INSTANCES_REGISTRY = Registry('COCO_INSTANCES')
COCO_INSTANCES_REGISTRY.__doc__ = """
从数据集字符串名，映射到对应的初始化函数
"""


class RegisterData:
    """ 旧版的数据集注册器，暂时不去修改优化了 """

    @classmethod
    def register_all(cls):
        with TicToc('RegisterData'):
            cls.register_by_annotations_dir(common_path.realestate_coco / 'agreement',
                                            common_path.realestate_coco / 'annotations')

            cls.register_by_annotations_dir(common_path.realestate_coco / 'agreement',
                                            common_path.realestate_coco / 'annotations_det')

            cls.register_by_annotations_dir(common_path.realestate_coco / 'agreement6_shade',
                                            common_path.realestate_coco / 'annotations',
                                            'agreement_train6_shade.json')

            cls.register_by_annotations_dir(common_path.sroie2 / 'images',
                                            common_path.sroie2 / 'annotations')

            cls.register_by_annotations_dir(common_path.voc2007 / 'JPEGImages',
                                            common_path.voc2007 / 'coco_annotations')

            # 裁剪后的sroie数据
            cls.register_by_annotations_dir(common_path.sroie2 / 'data/task3_crop/images',
                                            common_path.sroie2 / 'data/task3_crop')

            cls.register_by_annotations_dir(common_path.cipdf / 'images',
                                            common_path.cipdf / 'annotations')

    @classmethod
    def register_by_annotations_dir(cls, imdir, andir,
                                    patter=re.compile(r'.+_(train|val|test|minival)\d{0,}\.json'),
                                    classes=None):
        r""" 注册coco类型的数据格式

        :param imdir: 图片所在目录
        :param andir: 标注所在目录
            会注册目录下所有以 _[train|val|test]\d{0,}.json 为后缀的文件
        :param patter: 在andir下，要匹配分析的json文件
        :type patter: str | re.compile
        :param classes: 类别清单，例如 ['text', 'title', 'list', 'table', 'figure']
            如果输入该参数，则这批patter匹配的所有文件都以这个classes为准
            否则每个json读取自己的 categories 作为类清单

        本函数用来简化detectron2中coco类型数据格式的注册过程，基本通用
        但是由于要读取josn获取类别信息，在一些特别大的json读取
        """
        from detectron2.data.datasets import register_coco_instances

        # 1 标注文件
        files = Dir(andir).select(patter).subfiles()

        # 2 注册每一个json文件
        for f in files:
            if classes:
                cats = classes
            else:
                cats = f.read(encoding='utf8')['categories']
                cats = [x['name'] for x in cats]
            register_coco_instances(f.stem, {'thing_classes': cats}, str(f), str(imdir))


class _DatasetRegister:
    ROOT = pathlib.Path('.')
    CLASSES = ('text',)
    META_DATA = {}

    @classmethod
    def coco_instances(cls, name, json, imdir):
        def func():
            return cls.META_DATA, cls.ROOT / json, cls.ROOT / imdir

        COCO_INSTANCES_REGISTRY._do_register(name, func)  # noqa 不用装饰器就只能使用_do_register来注册了


class Publaynet(_DatasetRegister):
    """ 论文版本分析的数据集 """
    ROOT = common_path.publaynet
    CLASSES = ['text', 'title', 'list', 'table', 'figure']
    META_DATA = {'thing_classes': CLASSES}


Publaynet.coco_instances('publaynet_train', 'train_brief.json', 'train')
Publaynet.coco_instances('publaynet_val', 'val.json', 'val')
Publaynet.coco_instances('publaynet_val_mini', 'val_mini.json', 'val_mini')
Publaynet.coco_instances('publaynet_test', 'test_ids.json', 'test')

# # 也可以这样自定义函数注册数据，函数名就是数据名，然后返回 META_DATA, json, imdir 即可
# @COCO_INSTANCES_Registry.register()
# def publaynet_train():
#     return Publaynet.META_DATA, Publaynet.ROOT / 'train_brief.json', Publaynet.ROOT / 'train'


____register = """
数据集注册器
"""


def register_d2dataset(name, *, error=None):
    """ 注册到 detectron2 的MetadataCatalog、DatasetCatalog中

    :param name: 数据集名称
    :return:
    """
    from detectron2.data import MetadataCatalog
    from detectron2.data.datasets import register_coco_instances

    if name in MetadataCatalog.keys():
        # 已有的数据，就不用注册了
        pass
    elif name in COCO_INSTANCES_REGISTRY:
        register_coco_instances(name, *(COCO_INSTANCES_REGISTRY.get(name)()))
    else:
        if error == 'ignore':
            pass
        else:
            raise ValueError(f'未预设的数据集名称 {name}')
