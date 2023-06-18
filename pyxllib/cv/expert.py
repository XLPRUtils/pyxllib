#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/08/25 15:57

from collections import defaultdict
import concurrent.futures

import cv2
import pandas as pd
from tqdm import tqdm

import PIL.Image

from pyxllib.algo.stat import update_dataframes_to_excel
from pyxllib.file.specialist import get_etag, XlPath
from pyxllib.prog.specialist import Iterate
from pyxllib.cv.xlcvlib import CvImg, xlcv
from pyxllib.cv.xlpillib import PilImg, xlpil


def __1_目录级处理图片的功能():
    pass


class ImagesDir(XlPath):
    """ 这个函数功能，默认都是原地操作，如果怕以防万一出问题，最好对原始数据有另外的备份，而在新的目录里操作 """

    def debug_image_func(self, func, pattern='*', *, save=None, show=False):
        """
        :param func: 对每张图片执行的功能，函数应该只有一个图片路径参数  new_img = func(img)
            当函数有多个参数时，可以用lambda函数技巧： lambda im: func(im, arg1=..., arg2=...)
        :param save: 如果输入一个目录，会将debug结果图存储到对应的目录里
        :param show: 如果该参数为True，则每处理一张会imshow显示处理效果
            此时弹出的窗口里，每按任意键则显示下一张，按ESC退出
        :return:

        TODO 显示原图、处理后图的对比效果
        TODO 支持同时显示多张图处理效果
        """
        if save:
            save = XlPath(save)

        for f in self.glob_images(pattern):
            im1 = xlcv.read(f)
            im2 = func(im1)

            if save:
                xlcv.write(im2, self / save / f.name)

            if show:
                xlcv.imshow2(im2)
                key = cv2.waitKey()
                if key == '0x1B':  # ESC 键
                    break

    def fix_suffixs(self, pattern='**/*', log_file='_图片统计.xlsx', max_workers=None, pinterval=None):
        """ 修正错误的后缀名

        :param pinterval: 支持智能地判断进度间隔
        """

        # 1 修改后缀
        # 定义并行处理子函数
        def process_image_file(args):
            """ 处理单个图片文件，修正后缀名 """
            file, ext = args
            xlcv.write(xlcv.read(file), file)  # 读取图片，并按照原本文件名期望的格式存储
            ls.append([file.relpath(self).as_posix(), ext])

        ls = []
        files_with_exts = list(self.xglob_faker_suffix_images(pattern))
        if pinterval is None and files_with_exts:
            p = max(1000 * 100 // len(files_with_exts), 1)  # 最小也按1%进度展示
            if p < 50:  # 间隔只有小余50%，才比较有显示的意义
                pinterval = f'{p}%'  # 每1千张显示进度
        Iterate(files_with_exts).run(process_image_file, max_workers=max_workers, pinterval=pinterval)

        # 2 记录修改情况
        df = pd.DataFrame.from_records(ls, columns=['图片名', '原图片类型'])
        if log_file:
            update_dataframes_to_excel(XlPath.init(log_file, self), {'修改后缀名': df})
        return df

    def reduce_image_filesize(self, pattern='**/*',
                              limit_size=4 * 1024 ** 2, *,
                              read_flags=None,
                              change_length=False,
                              suffix=None,
                              log_file='_图片统计.xlsx',
                              max_workers=None, pinterval=None):
        """ 减小图片尺寸，可以限制目录里尺寸最大的图片不超过多少

        :param limit_size: 限制的尺寸
            一般自己的相册图片，亲测300kb其实就够了~~，即 300 * 1024
            百度API那边，好像不同接口不太一样，4M、6M、10M等好像都有
                但百度那是base64后的尺寸，会大出1/3
                为了够用，一般要限定在4M等比例的3/4比例内
        :param read_flags: 读取图片时的参数，设为1，可以把各种RGBA等奇怪的格式，统一为RGB
        :param change_length: 默认是要减小图片的边长，尺寸，来压缩图片的
            可以设为False，不调整尺寸，纯粹读取后再重写，可能也能压缩不少尺寸
        :param suffix: 可以统一图片后缀格式，默认保留原图片名称
            要带前缀'.'，例如'.jpg'
            注意其他格式的原图会被删除

        因为所有图片都会读入后再重新写入，速度可能会稍慢
        """

        # 1 调试信息
        print('原始大小', self.size(human_readable=True))

        # 2 精简图片尺寸
        # 定义并行处理子函数
        def process_image_file(f):
            """处理单个图片文件，减小图片尺寸"""
            size1 = f.size()
            im = xlpil.read(f, read_flags)
            _suffix = suffix or f.suffix
            if change_length:
                im = xlpil.reduce_filesize(im, limit_size, _suffix)
            size2 = xlpil.evaluate_image_file_size(im, _suffix)
            dst_f = f.with_suffix(_suffix)
            if size2 < size1:  # 只有文件尺寸确实变小的才更新
                xlpil.write(im, dst_f)
            if f.suffix != _suffix:
                f.delete()
            ls.append([f.relpath(self).as_posix(), dst_f.relpath(self).as_posix(), size1, size2])

        ls = []
        files = list(self.glob_images(pattern))
        if pinterval is None and files:
            p = max(100 * 100 // len(files), 1)  # 最小也按1%进度展示
            if p < 50:  # 间隔只有小余50%，才比较有显示的意义
                pinterval = f'{p}%'  # 每1千张显示进度
        Iterate(files).run(process_image_file, max_workers=max_workers, pinterval=pinterval)

        print('新目录大小', self.size(human_readable=True))

        # 3 记录修改细节
        # 注意，如果不使用suffix参数，'新图片'的值应该跟'原图片'是一样的
        # 以及当尝试精简的'新文件大小'大于'原文件大小'时，图片其实是不会被覆盖更新的
        df = pd.DataFrame.from_records(ls, columns=['原图片', '新图片', '原文件大小', '新文件大小'])
        if log_file:
            update_dataframes_to_excel(XlPath.init(log_file, self), {'图片瘦身': df})
        return df

    def adjust_image_shape(self, pattern='*', min_length=None, max_length=None, print_mode=True):
        """ 调整图片尺寸 """

        def printf(*args, **kwargs):
            if print_mode:
                print(*args, **kwargs)

        j = 1
        for f in self.glob_images(pattern):
            # 用pil库判断图片尺寸更快，但处理过程用的是cv2库
            h, w = xlpil.read(f).size[::-1]
            x, y = min(h, w), max(h, w)

            if (min_length and x < min_length) or (max_length and y > max_length):
                im = xlcv.read(f)
                im2 = xlcv.adjust_shape(im, min_length, max_length)
                if im2.shape != im.shape:
                    printf(f'{j}、{f} {im.shape} -> {im2.shape}')
                    xlcv.write(im2, f)
                    j += 1

    def check_repeat_phash_images(self, pattern='**/*', **kwargs):
        from pyxllib.cv.imhash import phash
        if 'files' not in kwargs:
            kwargs['files'] = self.glob_images(pattern)
        if 'hash_func' not in kwargs:
            kwargs['hash_func'] = lambda p: phash(p)
        self.check_repeat_files(pattern, **kwargs)

    def check_repeat_dhash_images(self, pattern='**/*', **kwargs):
        from pyxllib.cv.imhash import dhash
        if 'files' not in kwargs:
            kwargs['files'] = self.glob_images(pattern)
        if 'hash_func' not in kwargs:
            kwargs['hash_func'] = lambda p: dhash(p)
        self.check_repeat_files(pattern, **kwargs)


def find_modified_images(dirs, print_mode=False):
    """ 查找可能被修改过的图片

    一般用在数据标注工作中，对收回来的数据目录，和原本数据目录做个对比，
    以name作为对应关联，看前后图片是否内容发生变换，比如旋转。

    :param list[str] dirs: 图片所在目录列表
    :param bool print_mode: 是否打印进度提示，默认为 False
    :return dict[str, list[str]]: 包含图片名字和可能被修改过的图片路径列表的字典

    示例用法：
    import os
    from pprint import pprint
    from pyxllib.cv.expert import find_modified_images

    os.chdir('/home/chenkunze/data')
    res = find_modified_images([r'm2305latex2lgx/train_images_sub',
                                r'm2305latex2lg/1、做完的数据'])
    pprint(res)
    """
    from pyxllib.file.specialist import get_etag  # 发现不能用相似，还是得用etag

    # 1 将图片按名字分组
    def group_by_name(dirs):
        """ 将图片按名字分组

        :param list[str] dirs: 图片所在目录列表
        :return dict[str, list[str]]: 包含图片名字和对应图片路径列表的字典

        >>> group_by_name(['path/to/dir1', 'path/to/dir2'])
        {'image1.jpg': ['path/to/dir1/image1.jpg'], 'image2.png': ['path/to/dir2/image2.png']}
        """
        image_groups = {}
        for dir in dirs:
            for path in XlPath(dir).rglob_images():
                image_name = path.name
                if image_name not in image_groups:
                    image_groups[image_name] = []
                image_groups[image_name].append(path)
        return image_groups

    image_groups = group_by_name(dirs)

    # 2 存储有哪些变化的分组
    modified_images = {}
    progress_counter = 0

    if print_mode:
        total_files = sum(len(paths) for paths in image_groups.values())
        print(f"Total files: {total_files}")

    for image_name, paths in image_groups.items():
        if len(paths) <= 1:
            continue

        hash_values = [get_etag(str(path)) for path in paths]
        sizes = [PIL.Image.open(path).size for path in paths]

        # 这里可以增强，更加详细展示差异，比如是不是被旋转了90度、180度、270度，但会大大提升运算量，暂时不添加
        if len(set(hash_values)) > 1 or len(set(sizes)) > 1:
            # 获取posix风格路径
            modified_images[image_name] = [XlPath(path).as_posix() for path in paths]

        if print_mode:
            progress_counter += len(paths)
            print(f"Progress: {progress_counter}/{total_files}")

    return modified_images
