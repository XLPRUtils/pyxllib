#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/08/15 00:59

from collections import defaultdict

from tqdm import tqdm

from pyxllib.basic import *
from pyxllib.debug import pd
from pyxllib.cv import np_array, np, PilImg

# 我自己按照“红橙黄绿蓝靛紫”的顺序展示
LABEL_COLORMAP7 = [(0, 0, 0), (255, 0, 0), (255, 125, 0), (255, 255, 0),
                   (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]


def is_labelme_json_data(data):
    """ 是labelme的标注格式
    :param data: dict
    :return: True or False
    """
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
        File(p).write(data, encoding=encoding, if_exists='delete')


class GenLabelme:
    @classmethod
    def data(cls, imfile=None, **kwargs):
        """ 主要框架结构
        :param imfile: 可以传入一张图片路径
        """
        # 1 传入图片路径的初始化
        if imfile:
            file = File(imfile)
            name = file.name
            img = PilImg(str(file))
            height, width = img.size()
        else:
            name, height, width = '', 0, 0

        # 2 字段值
        data = {'version': '4.5.7',
                'flags': {},
                'shapes': [],
                'imagePath': name,
                'imageData': None,
                'imageWidth': width,
                'imageHeight': height,
                }
        if kwargs: data.update(kwargs)
        return data

    @classmethod
    def shape(cls, label, points, shape_type=None, dtype=None, group_id=None, **kwargs):
        """ 最基本的添加形状功能

        :param shape_type: 会根据points的点数量，智能判断类型，默认一般是polygon
            其他需要自己指定的格式：line、circle
        :param dtype: 可以重置points的存储数值类型，一般是浮点数，可以转成整数更精简
        :param group_id: 本来是用来分组的，但其值会以括号的形式添加在label后面，可以在可视化中做一些特殊操作
        """
        # 1 优化点集数据格式
        points = np_array(points, dtype).reshape(-1, 2).tolist()
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

    @classmethod
    def shape2(cls, **kwargs):
        """ 完全使用字典的接口形式 """
        label = kwargs.get('label', '')
        points = kwargs['points']  # 这个是必须要有的字段
        kw = copy.deepcopy(kwargs)
        del kw['label']
        del kw['points']
        return cls.shape(label, points, **kw)


class ToLabelmeJson:
    """ 标注格式转label形式

    初始化最好带有图片路径，能获得一些相关有用的信息
    然后自定义实现一个 get_data 接口，实现self.data的初始化，运行完可以从self.data取到字典数据
        根据需要可以定制自己的shape，修改get_shape函数
    可以调用write写入文件

    document: https://www.yuque.com/xlpr/pyxllib/ks5h4o
    """

    @deprecated(version=VERSION, reason='建议使用GenLabelme实现')
    def __init__(self, imgpath):
        """
        :param imgpath: 可选参数图片路径，强烈建议要输入，否则建立的label json会少掉图片宽高信息
        """
        self.imgpath = File(imgpath)
        # 读取图片数据，在一些转换规则比较复杂，有可能要用到原图数据
        if self.imgpath:
            # 一般都只需要获得尺寸，用pil读取即可，速度更快，不需要读取图片rgb数据
            self.img = PilImg(self.imgpath)
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
            height, width = self.img.size()
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
        """最基本的添加形状功能

        :param shape_type: 会根据points的点数量，智能判断类型，默认一般是polygon
            其他需要自己指定的格式：line、circle
        :param dtype: 可以重置points的存储数值类型，一般是浮点数，可以转成整数更精简
        :param group_id: 本来是用来分组的，但其值会以括号的形式添加在label后面，可以在可视化中做一些特殊操作
        """
        # 1 优化点集数据格式
        points = np_array(points, dtype).reshape(-1, 2).tolist()
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

    def write(self, dst=None, if_exists='delete'):
        """
        :param dst: 往dst目标路径存入json文件，默认名称在self.imgpath同目录的同名json文件
        :return: 写入后的文件路径
        """
        if dst is None and self.imgpath:
            dst = self.imgpath.with_suffix('.json')
        # 官方json支持indent=None的写法，但是ujson必须要显式写indent=0
        return File(dst).write(self.data, if_exists=if_exists, indent=0)

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
        ims = Dir(imdir).select(['*.jpg', '*.png']).subfiles()
        if not labeldir: labeldir = imdir
        txts = [File(f.stem, labeldir, suffix=label_file_suffix) for f in ims]
        cls.main_pair(ims, txts)

    @classmethod
    def main_pair(cls, images, labels):
        """ 一一配对匹配处理 """
        Iterate(zip(images, labels)).run(lambda x: cls.create_json(x[0], x[1]),
                                         pinterval='20%', max_workers=8)


class Quad2Labelme(ToLabelmeJson):
    """ 四边形类标注转labelme """

    def get_data(self, infile):
        lines = File(infile).read().splitlines()
        for line in lines:
            # 一般是要改这里，每行数据的解析规则
            vals = line.split(',', maxsplit=8)
            if len(vals) < 9: continue
            pts = [int(v) for v in vals[:8]]  # 点集
            label = vals[-1]  # 标注的文本
            # get_shape还有shape_type形状参数可以设置
            #  如果是2个点的矩形，或者3个点以上的多边形，会自动判断，不用指定shape_type
            self.add_shape(label, pts)


class LabelmeData:
    def __init__(self, root, data):
        """
        :param root: 文件根目录
        :param data: {'PMC4055390_00006': [file, lmdata], ...}
            其中 lmdata 为一个labelme文件格式的标准内容
        """
        self.root, self.data = Dir(root), data

    @classmethod
    def init(cls, root, *, disable=True, filter_json=None):
        """ 标准init方法，init_from_labelme, 已经有labelme的json标注文件，可以用该方法初始化

        :param root: 数据根目录
        :param disable: True, 不显示读取进度条； False，显示读取进度条
        :param filter_json: 初始化模式
            None，遍历所有json来初始化
            'fmt', filter by json format, 根据json是否为labelme格式进行过滤
            'img', filter by image, 根据是否有配对的图片文件过滤json文件
        """
        root = Dir(root)
        # 1 筛选过滤json文件
        if filter_json == 'img':
            pattern = re.compile(r'.+\.(jpe?g|png|bmp)$', flags=re.IGNORECASE)
            imfiles = root.select(pattern).subfiles()
            files = []
            for imf in imfiles:
                f = imf.with_suffix('.json')
                if f:
                    files.append(f)
        else:
            files = root.select('**/*.json').subfiles()

        # 2 读取内存数据
        data = defaultdict(list)  # List[[File, lmdata]]
        for file in tqdm(files, desc='读取labelme json数据', disable=disable):
            lmdata = file.read()
            if filter_json == 'fmt' and not is_labelme_json_data(lmdata):
                continue
            data[file.stem] = [file, lmdata]

        return cls(root, data)

    def __len__(self):
        return len(self.data)

    def reduce(self):
        """ 移除imageData字段值 """
        for _, data in self.data.values():
            data['imageData'] = None

    def writes(self, *, max_workers=8, disable=True):
        """ 重新写入json文件

        可能是内存里修改了数据，需要重新覆盖
        也可能是从coco初始化来的内存数据，需要生成对应的json文件
        """

        def write(x):
            file, lmdata = x
            if file:  # 如果文件存在，要遵循原有的编码规则
                with open(str(file), 'rb') as f:
                    bstr = f.read()
                encoding = get_encoding(bstr)
                file.write(lmdata, encoding=encoding, if_exists='delete')
            else:  # 否则直接写入
                file.write(lmdata)

        mtqdm(write, self.data.values(), desc='写入labelme json数据', max_workers=max_workers, disable=disable)

    def to_df(self, *, disable=True):
        """ 转成dataframe表格查看 """

        def read(x):
            file, lmdata = x
            if not lmdata['shapes']:
                return
            df = pd.DataFrame.from_records(lmdata['shapes'])
            df['filename'] = file.relpath(self.root)
            # 坐标转成整数，看起来比较精简点
            df['points'] = [np.array(v, dtype=int).tolist() for v in df['points']]
            ls.append(df)

        ls = []
        # 这个不建议开多线程，顺序会乱
        mtqdm(read, self.data.values(), desc='labelme转df', disable=disable)
        shapes_df = pd.concat(ls)
        # TODO flags 和 group_id 字段可以放到最后面
        shapes_df.reset_index(inplace=True, drop=True)

        return shapes_df
