#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/11/17

""" 图片文本行标注相关处理
"""

from pyxllib.xlcv import *

from functools import reduce

from shapely.geometry import MultiPolygon

from pyxllib.algo.geo import split_vector_interval
from pyxllib.algo.disjoint import disjoint_set
from pyxllib.algo.shapelylib import ShapelyPolygon


class TextlineShape:
    """ 一个文本行标注对象

    这里的基础功能主要是几何关系计算，可以继承类后扩展其他功能
    """

    def __init__(self, box, *, shrink_bound=False):
        """
        :param box: 可以转成Polygon的数据类型
        :param shrink_bound: 倾斜度过大的文本框，需要特殊处理，把外接矩形缩小会更准确些。
        """
        self.polygon = ShapelyPolygon.gen(box)
        self.bounds = self.polygon.bounds
        if shrink_bound:
            b = self.bounds
            total_area = (b[2] - b[0]) * (b[3] - b[1])
            # 缩放比例
            self.bounds = bound_scale(self.bounds, self.polygon.area / total_area)
            # self.bounds = bound_scale(self.bounds, 0.5)

        self.minx, self.maxx = self.bounds[0], self.bounds[2]
        self.width = self.maxx - self.minx
        self.miny, self.maxy = self.bounds[1], self.bounds[3]
        self.height = self.maxy - self.miny
        self.centroid = self.polygon.centroid

    def in_the_same_line(self, other):
        """ 两个框在同一个文本行 """
        if other.miny < self.centroid.y < other.maxy:
            return True
        elif self.miny < other.centroid.y < self.maxy:
            return True
        else:
            return False

    def is_lr_intersect(self, other, gap=5):
        """ 左右相交
        """
        if other.minx - gap <= self.minx <= other.maxx + gap:
            return True
        elif other.minx - gap <= self.maxx <= other.maxx + gap:
            return True
        else:
            return False

    def is_tb_intersect(self, other, gap=5):
        """ 上下相交
        """
        # 这个 gap 规则是不动产的，不能放在通用规则里
        # gap = min(50, self.height / 2, other.height / 2)  # 允许的最大间距，默认按照最小的高，但还要再设置一个50的上限
        if other.miny - gap <= self.miny <= other.maxy + gap:
            return True
        elif other.miny - gap <= self.maxy <= other.maxy + gap:
            return True
        else:
            return False

    def is_intersect(self, other):
        return self.polygon.intersects(other)

    def __add__(self, other):
        """ 合并两个文本行 """
        box = rect2polygon(MultiPolygon([self.polygon, other.polygon]).bounds)
        return TextlineShape(box)

    def __lt__(self, other):
        """ 框的排序准则 """
        if self.in_the_same_line(other):
            return self.centroid.x < other.centroid.x
        else:
            return self.centroid.y < other.centroid.y

    @classmethod
    def merge(cls, shapes):
        """ 将同张图片里的多个shape进行合并 """
        # 1 对文本框分组
        shape_groups = disjoint_set(shapes, lambda x, y: x.is_intersect(y))

        # 2 合并文本内容
        new_shapes = []
        for group in shape_groups:
            shape = reduce(lambda x, y: x + y, sorted(group))
            new_shapes.append(shape)
        return new_shapes


def im_textline_split(im, maxsplit=None, minwidth=3):
    """ 这是最基础版本的示例：比较干净，白底黑字，没有太大倾斜的处理情况

    一般各种特殊任务的数据，需要根据具体任务定制、修改该函数
    """
    img = xlcv.read(im, 0)
    m = np.mean(img)
    # 比较干净的图可以这样，直接做二值化，并且对二值化中的图要求比较高，基本不能出现一个文字的像素
    bi = img < m
    vec = bi.sum(axis=0)
    return split_vector_interval(vec, maxsplit=maxsplit, minwidth=minwidth)


def merge_labels_by_widths(labels, widths, sep=' '):
    """ 一组数量不少于len(widths)的labels，参照widths给的每一部分权重，合并文本内容

    算是和图片分割配套的相关功能，往往文本内容要跟着图片的切割情况进行拆分

    这个算法其实也可以用来做拆分，比如要把'abcdefg'拆成[20, 30]的两段，
        可以用list先把前者变成单字符的list就行了 ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    :param labels: 一组字符串
    :param widths: 一组参考宽度
    :param sep: 拼接的时候使用的间隔符
    :return: 尽可能拼接出符合参考宽度的一组字符串

    >>> merge_labels_by_widths(['aa', 'bbb', 'c', 'ccc'], [10,10,20])
    ['aa', 'bbb', 'c ccc']
    >>> merge_labels_by_widths(['a', 'a', 'b', 'b'], [13, 10, 10])
    ['a a', 'b', 'b']
    >>> merge_labels_by_widths(['a', 'a', 'b', 'b'], [10, 10, 10])
    ['a', 'a', 'b b']
    >>> merge_labels_by_widths(['a', 'b', 'c'], [11, 12, 13])
    ['a', 'b', 'c']
    >>> merge_labels_by_widths(['a'], [10, 12])  # labels比widths少时，后面的统一用''填充
    ['a', '']
    >>> merge_labels_by_widths([''], [10, 12])
    ['', '']

    TODO 感觉实现的代码还有点凌乱，可能还有改进空间
    """
    # 1 统一量纲
    if len(labels) < len(widths):
        labels += [''] * (len(widths) - len(labels))
    label_widths = [strwidth(x) for x in labels]
    n_label = len(labels)
    assert sum(widths), 'widths必须要有权重值'
    r = sum(label_widths) / sum(widths)
    widths = [r * w for w in widths]

    # 2 用贪心算法合并
    need_merge = n_label - len(widths)
    i, k, new_labels = 0, 0, []
    for w in widths:
        if k < need_merge:
            label_width = label_widths[i]
            j = i + 1
            while j < n_label and k < need_merge and abs(label_width + label_widths[j] - w) < abs(label_width - w):
                label_width += label_widths[j]
                j += 1
                k += 1
            new_labels.append(sep.join(labels[i:j]))
            i = j
        elif k == need_merge:
            new_labels += labels[i:]
            i = n_label
            break
    # 还有未匹配使用的，全部拼接到末尾
    if i + 1 <= n_label:
        new_labels[-1] = ' '.join([new_labels[-1]] + labels[i:])

    return new_labels


class TextlineAnnotation(TextlineShape):
    """ coco格式的标注 """

    def __init__(self, anno):
        super().__init__(xywh2ltrb(anno['bbox']))
        self.anno = anno

    def __add__(self, other):
        """ 两个coco标注的合并 """
        # 以 self 框的属性为基准
        anno, anno2 = self.anno.copy(), other.anno

        # 合并后的 bbox
        anno['bbox'] = ltrb2xywh(MultiPolygon([self.polygon, other.polygon]).bounds)

        # 合并分割属性
        if anno2['segmentation']:
            anno['segmentation'] += anno2['segmentation']

        # 合并 label
        if 'label' in anno or 'label' in anno2:
            text = anno2.get('label', '')
            if text: text = ' ' + text
            anno['label'] = anno.get('label', '') + text

        return TextlineAnnotation(anno)

    @classmethod
    def merge(cls, annotations):
        """ 合并同一文本行上相近、相交的文本标注 """
        # 1 转 shape 格式
        shapes = [cls(x) for x in annotations]

        # 2 对文本框分组
        shape_groups = disjoint_set(shapes, lambda x, y: x.in_the_same_line(y) and x.is_lr_intersect(y))

        # 3 合并文本内容
        new_shapes = []
        for group in shape_groups:
            shape = reduce(lambda x, y: x + y, sorted(group))
            new_shapes.append(shape)

        # 4 转回 annotations 格式
        return [x.anno for x in new_shapes]

    @classmethod
    def split(cls, im, annotations, split_func=im_textline_split):
        """ coco标注格式的处理，将图片im对应的文本行标注结果 annos，按照空白背景切分开

        :param im: 图片数据
        :param annotations: coco 格式的 annotations
        :param split_func: 分析图片数据时所用投影分析函数，需要返回带有文本内容的列区间

        如果有label文本，会跟着一起切割处理

        :return:
            新的annotations数组
            注意，有的图片处理起来会有问题，此时会返回 []，建议丢弃这些图片
        """
        new_annos = []
        for anno in annotations:
            # 仅测试某个特定的 anno
            # if anno['id'] != 2345:
            #     continue

            x, y, w, h = anno['bbox']
            _, t, _, b = xywh2ltrb(anno['bbox'])
            subim = xlcv.get_sub(im, xywh2ltrb(anno['bbox']))
            spans = split_func(subim)
            # print(anno['label'], spans)
            # 左右放宽一些，并且计算基于全图的绝对坐标
            spans = [[x + max(span[0] - 3, 0), x + min(span[1] + 3, w)] for span in spans]

            if len(spans) == 0:
                # 一些特殊情况，很可能是框标的位置偏了，质量不行
                return []  # 整张图的标注都不要了，直接返回空值
            elif len(spans) == 1:
                l, r = spans[0]
                a = copy.copy(anno)
                a['bbox'] = ltrb2xywh([l, t, r, b])
                new_annos.append(a)
            else:  # 拆分出了多段
                # 这里 label 最好也要拆一下
                labels = anno['label'].split()
                if len(labels) > len(spans):
                    labels = merge_labels_by_widths(labels, [(span[1] - span[0]) for span in spans])
                elif len(labels) < len(spans):
                    # imwrite(subim, 'subim.jpg')
                    # print(x, y, w, h)
                    # 要检查出现这些情况的所有数据：labels的少于spans
                    get_xllog().warning(DPrint.format({'$异常': 'len(labels)<len(spans)',
                                                       'labels': labels, 'spans': spans}))
                    # 这种情况先保留原始框
                    new_annos.append(anno)
                    continue

                for span, label in zip(spans, labels):
                    l, r = span
                    a = copy.copy(anno)
                    a['bbox'] = ltrb2xywh([l, t, r, b])
                    a['label'] = label
                    new_annos.append(a)

        return new_annos


class TextlineSpliter:
    """
    TextString2016、Casia 基本都可以直接用
    """

    @classmethod
    def spliter(cls, im, maxsplit=None, minwidth=3):
        """ （核心处理接口功能）比较干净，白底黑字，没有太大倾斜的处理情况
        如果有其他特殊情况，记得要重置这个处理方式，见EnglishWord

        :param im: 输入图片路径，或者np.ndarray矩阵
        :param maxsplit: 最大切分数量，即最多得到几个子区间
            没设置的时候，会对所有满足条件的情况进行切割
        :param minwidth: 每个切分位置最小具有的宽度
        :return: [(l, r), (l, r), ...]  每一段文本的左右区间

        详细文档：https://www.yuque.com/xlpr/data/cx6xm5
        """
        img = xlcv.read(im, 0)
        m = np.mean(img)
        # 比较干净的图可以这样，直接做二值化，并且对二值化中的图要求比较高，基本不能出现一个文字的像素
        bi = img < m
        vec = bi.sum(axis=0) - 2
        return split_vector_interval(vec, maxsplit=maxsplit, minwidth=minwidth)

    @classmethod
    def split_img(cls, file, maxsplit=None, minwidth=3):
        """
        :param file: 输入np.ndarray图片，或者pil图片，或者图片路径
        :param maxsplit:
        :param minwidth:
        :return: 返回切分后的np.ndarray格式的图片清单
        """
        img = xlcv.read(file)
        vec = cls.spliter(img, maxsplit, minwidth)
        imgs = [img[:, l:r + 1] for l, r in vec]
        return imgs

    @classmethod
    def spliter_img(cls, file, maxsplit=None, minwidth=3):
        """ 可视化，测试一张图的切分效果
        如果不是测试self.root里的图片，可以直接输入一个绝对路径的图片file
        """
        im = xlcv.read(file, 0)
        cols = cls.spliter(im, maxsplit=maxsplit, minwidth=minwidth)

        lines = [[c, 0, c, im.shape[0] - 1] for c in np.array(cols, dtype=int).reshape(-1)]
        # 偶数区间划为为红色
        im2 = xlcv.lines(im, lines[::4], [0, 0, 255])
        im2 = xlcv.lines(im2, lines[1::4], [0, 0, 255])
        # 奇数区间划分为蓝色
        im2 = xlcv.lines(im2, lines[2::4], [255, 0, 0])
        im2 = xlcv.lines(im2, lines[3::4], [255, 0, 0])

        return im2

    @classmethod
    def show_spliter_imgs(cls, dir_state, *, save=None, show=True):
        ImagesDir.debug_func(dir_state,  # 随机抽取10张图片
                             lambda img_file: cls.spliter_img(img_file, maxsplit=None, minwidth=3),  # 执行功能
                             save=save,  # 结果保存位置
                             show=show)  # 是否imshow结果图

    @classmethod
    def relabel_labelfile(cls, p, maxsplit=None, minwidth=3, imgdir='images'):
        """ 对一份文件里标注的所有图片，批量进行转换，并加入一列新的坐标数据 """
        lines = p.read().splitlines()
        res = []
        for line in lines:
            line = line.split(maxsplit=1)
            im = xlcv.read(p.parent / f'{imgdir}/{line[0]}', 0)
            cols = cls.spliter(im, maxsplit, minwidth)
            line.append(' '.join(map(str, np.array(cols, dtype=int).reshape(-1))))
            res.append('\t'.join(line))
        content = '\n'.join(res)
        p.with_stem(p.stem + f'+text_interval-minw={minwidth}').write(content, if_exists='replace')

    @classmethod
    def relabel_labelfiles(cls, root, maxsplit=None, minwidth=3, imgdir='images'):
        """ 切分所有的文件
        :param root: 根目录
        :param imgdir: 图片所在子目录名称
        :return:
        """
        root = Dir(root)
        cls.relabel_labelfile(root / 'val.txt', maxsplit, minwidth, imgdir)
        cls.relabel_labelfile(root / 'test.txt', maxsplit, minwidth, imgdir)
        cls.relabel_labelfile(root / 'train.txt', maxsplit, minwidth, imgdir)

    @classmethod
    def split_labelfiles(cls, src, dst, minwidth=3, imgdir='images'):
        def func(name):
            """ 对一份文件里标注的所有图片，批量进行转换，并加入一列新的坐标数据

            p  原来的.txt标注文件路径
            p_im  原来的图片路径
            q   切割后的.txt标注文件路径
            q_im  切割后的图片路径

            """
            p, q = File(name, src), File(name, dst)
            if not p: return
            lines = p.read().splitlines()
            res = []
            for line in lines:
                # 获得图片文件，切分的单词
                line = line.split(maxsplit=1)
                if len(line) < 2: continue

                p_im = File(p.parent / f'{imgdir}/{line[0]}')
                # print(p_im)
                words = line[1].split()

                if len(words) < 2:
                    q_im = File(f'{imgdir}/{p_im.name}', dst)
                    p_im.copy(q_im)
                    res.append(f'{q_im.name}\t{words[0]}')
                else:
                    # 切分图片
                    imgs = cls.split_img(p_im, len(words), minwidth)
                    # 重新生成标注
                    for k, im in enumerate(imgs):
                        q_im = File(f'{imgdir}/{p_im.stem}_{k}', dst, suffix=p_im.suffix)
                        xlcv.write(im, q_im, if_exists='replace')
                        res.append(f'{q_im.name}\t{words[k]}')
            content = '\n'.join(res)
            q.write(content, if_exists='replace')

        src, dst = Dir(src), Dir(dst)
        for name in ['val.txt', 'test.txt', 'train.txt']:
            # for name in ['append.txt']:
            # for name in ['val.txt']:
            func(name)


class EnglishWordTLS(TextlineSpliter):
    @classmethod
    def spliter(cls, img, maxsplit=None, minwidth=3):
        """ 同 TextLineSpliter.spliter
            这个功能针对处理 带噪声干扰的白底黑字图片
        """
        img = xlcv.read(img, 0)
        h, w = img.shape
        vec = img[int(h / 3):int(2 * h / 3)].mean(axis=0)  # 只用上下中间的三分之一
        vec = vec.mean() - vec + 5  # 文字变正，背景变负；因为背景有很多黑点噪声，还要多减一
        return split_vector_interval(vec, maxsplit=maxsplit, minwidth=minwidth)


class TLSMain:
    def textstring2016(self):
        # d = TextLineSpliter('/home/datasets/textGroup/TextString2016/')
        d = r'D:\datasets\TextString2016'
        # ob.test('images/T0000-03.jpg', minwidth=3)
        TextlineSpliter.relabel_labelfiles(d, minwidth=3)

    def casia(self):
        os.chdir('/home/datasets/textGroup/casia/offlinehw/CASIA-HWDB2.x_pngImg_line')
        TextlineSpliter.relabel_labelfiles('CASIA-HWDB2.0_savePTTSImg_line', minwidth=3)
        TextlineSpliter.relabel_labelfiles('CASIA-HWDB2.1_savePTTSImg_line', minwidth=3)
        TextlineSpliter.relabel_labelfiles('CASIA-HWDB2.2_savePTTSImg_line', minwidth=3)

    def english_word(self):
        # ob.test('total/1.jpg', 4, 3)
        EnglishWordTLS.relabel_labelfiles(r'D:\datasets\english-word', minwidth=10, imgdir='total')

    def sroie(self):
        path = Dir('SROIE2019/task1train_626p_repo/task1train_626p_patch/')
        root = Dir(path, '/home/datasets/textGroup')
        TextlineSpliter.show_spliter_imgs(root.select('images/*.png').sample(10),
                                          save=File(path / 'temp', '/home/datasets/textGroup'),
                                          show=False)


if __name__ == '__main__':
    with TicToc(__name__):
        pass
