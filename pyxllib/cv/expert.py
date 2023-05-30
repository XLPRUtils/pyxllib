#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/08/25 15:57

from collections import defaultdict

import cv2
from tqdm import tqdm

import PIL.Image

from pyxllib.file.specialist import get_etag, XlPath
from pyxllib.cv.xlcvlib import CvImg, xlcv
from pyxllib.cv.xlpillib import PilImg, xlpil


def __1_目录级处理图片的功能():
    pass


class ImagesDir(XlPath):
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

    def reduce_image_filesize(self, pattern='*',
                              limit_size=4 * 1024 ** 2, *,
                              read_flags=None,
                              change_length=True,
                              suffix=None,
                              print_mode=True):
        """ 减小图片尺寸，可以限制目录里尺寸最大的图片不超过多少

        :param limit_size: 限制的尺寸
            一般自己的相册图片，亲测300kb其实就够了~~，即 300 * 1024
            百度API那边，好像不同接口不太一样，4M、6M、10M等好像都有
                但百度那是base64后的尺寸，会大出1/3
                为了够用，
        :param read_flags: 读取图片时的参数，设为1，可以把各种RGBA等奇怪的格式，统一为RGB
        :param change_length: 默认是要减小图片的边长，尺寸，来压缩图片的
            可以设为False，不调整尺寸，纯粹读取后再重写，可能也能压缩不少尺寸
        :param suffix: 可以统一图片后缀格式，默认保留原图片名称
            要带前缀'.'，例如'.jpg'
            注意其他格式的原图会被删除

        因为所有图片都会读入后再重新写入，速度可能会稍慢
        """

        def printf(*args, **kwargs):
            if print_mode:
                print(*args, **kwargs)

        printf('原始大小', self.size(human_readable=True))

        for f in tqdm(self.glob_images(pattern), disable=not print_mode):
            im = xlpil.read(f, read_flags)
            _suffix = suffix or f.suffix
            if change_length:
                im = xlpil.reduce_filesize(im, limit_size, _suffix)
            xlpil.write(im, f.with_suffix(_suffix))
            if f.suffix != _suffix:
                f.delete()

        printf('新目录大小', self.size(human_readable=True))

    def adjust_image_shape(self, pattern='*', min_length=None, max_length=None, print_mode=True):
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


def find_modified_images(dirs):
    """ 查找可能被修改过的图片

    一般用在数据标注工作中，对收回来的数据目录，和原本数据目录做个对比，
    以name作为对应关联，看前后图片是否内容发生变换，比如旋转。

    :param list[str] dirs: 图片所在目录列表
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
    from pyxllib.cv.imhash import dhash

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
    for image_name, paths in image_groups.items():
        if len(paths) <= 1:
            continue

        hash_values = [dhash(path) for path in paths]
        sizes = [PIL.Image.open(path).size for path in paths]

        # 这里可以增强，更加详细展示差异，比如是不是被旋转了90度、180度、270度，但会大大提升运算量，暂时不添加
        if len(set(hash_values)) > 1 or len(set(sizes)) > 1:
            # 获取posix风格路径
            modified_images[image_name] = [XlPath(path).as_posix() for path in paths]

    return modified_images
