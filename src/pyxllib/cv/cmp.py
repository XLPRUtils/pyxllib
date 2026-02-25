#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/08/25 15:57

from pyxllib.prog.lazyimport import lazy_import

try:
    import PIL.Image
except ModuleNotFoundError:
    PIL = lazy_import('PIL', 'Pillow')

from pyxllib.file.specialist import get_etag, XlPath


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
