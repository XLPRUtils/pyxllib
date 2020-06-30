#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/06/02 16:00


from collections import defaultdict
import concurrent.futures
import io
import os
import subprocess
import re


import requests

try:
    import PIL
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'pillow'])
    import PIL

from PIL import Image

try:
    from get_image_size import get_image_size
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'opsdroid-get-image-size'])
    from get_image_size import get_image_size


from pyxllib.debug.dprint import dprint
from pyxllib.debug.pathlib_ import Path


def get_img_content(in_):
    """获取in_代表的图片的二进制数据
    :param in_: 可以是本地文件，也可以是图片url地址，也可以是Image对象
    """
    from pyxllib.debug.judge import is_url, is_file

    # 1 取不同来源的数据
    if is_url(in_):
        content = requests.get(in_).content
        img = Image.open(io.BytesIO(content))
    elif is_file(in_):
        with open(in_, 'rb') as f:
            content = f.read()
        img = Image.open(in_)
    elif isinstance(in_, Image.Image):
        img = in_
    else:
        raise ValueError

    img = image_rgba2rgb(img)  # 如果是RGBA类型，要把透明底变成白色
    file = io.BytesIO()
    img.save(file, 'JPEG')
    content = file.getvalue()

    return content


def magick(infile, *, outfile=None, if_exists='error', transparent=None, trim=False, density=None, other_args=None):
    """调用iamge magick的magick.exe工具
    :param infile: 处理对象文件
    :param outfile: 输出文件，可以不写，默认原地操作（只设置透明度、裁剪时可能会原地操作）
    :param if_exists: 如果目标文件已存在要怎么处理
    :param transparent:
        True: 将白底设置为透明底
        可以传入颜色参数，控制要设置的为透明底的背景色，默认为'white'
        注意也可以设置rgb：'rgb(164,192,167)'
    :param trim: 裁剪掉四周空白
    :param density: 设置图片的dpi值
    :param other_args: 其他参数，输入格式如：['-quality', 100]
    :return:
        False：infile不存在或者不支持的文件扩展名
        返回生成的文件名（outfile）
    """
    # 1 条件判断，有些情况下不用处理
    if not outfile: outfile = infile

    def func(infile, outfile):
        # 判断是否是支持的输入文件类型
        ext = os.path.splitext(infile)[1].lower()
        if not Path(infile).is_file() or not ext in ('.png', '.eps', '.pdf', '.jpg', '.jpeg', '.wmf', '.emf'): return False

        # 2 生成需要执行的参数
        cmd = ['magick.exe']
        # 透明、裁剪、density都是可以重复操作的
        if density: cmd.extend(['-density', str(density)])
        cmd.append(infile)
        if transparent:
            if not isinstance(transparent, str): transparent_ = 'white'
            else: transparent_ = transparent
            cmd.extend(['-transparent', transparent_])
        if trim: cmd.append('-trim')
        if other_args: cmd.extend(other_args)
        cmd.append(outfile)

        # 3 生成目标png图片
        print(' '.join(cmd))
        subprocess.run(cmd, stderr=subprocess.PIPE, shell=True)  # 看不懂magick出错写的是啥，关了

    return Path(infile).process(outfile, func, if_exists, arg1=infile, arg2=outfile).fullpath


def ensure_pngs(folder, *, if_exists='ignore',
                transparent=None, trim=False,
                density=None, epsdensity=None,
                max_workers=None):
    """确保一个目录下的所有图片都有一个png版本格式的文件
    :param folder: 目录名，会遍历直接目录下所有没png的stem名称生成png
    :param if_exists: 如果文件已存在，要进行的操作
        'replace'，直接替换
        'backup'，备份后生成新文件
        'ignore'，不写入

    :param transparent: 设置转换后的图片是否要变透明
    :param trim: 是否裁剪边缘空白
    :param density: 缩放尺寸
        TODO magick 和 inkscape 的dpi参考值是不同的，inkscape是真的dpi，magick有个比例差，我还没搞明白
    :param epsdensity: eps转png时默认放大的比例，注意默认100%是72，写144等于长宽各放大一倍
    :param max_workers: 并行的最大线程数
    """

    # 1 提取字典d，key<str>: 文件stem名称， value<set>：含有的扩展名
    d = defaultdict(set)
    for file in os.listdir(folder):
        if file.endswith('-eps-converted-to.pdf'): continue
        name, ext = os.path.splitext(file)
        ext = ext.lower()
        if ext in ('.png', '.eps', '.pdf', '.jpg', '.jpeg', '.wmf', '.emf', '.svg'):
            d[name].add(ext)

    # 2 遍历处理每个stem的图片
    executor = concurrent.futures.ThreadPoolExecutor(max_workers)
    for name, exts in d.items():
        # 已经存在png格式的图片时，看if_exists参数
        if '.png' in exts:
            if if_exists == 'ignore':
                continue
            elif if_exists == 'backup':
                Path(name, '.png', folder).backup(move=True)
            elif if_exists == 'replace':
                pass
            else:
                raise ValueError

        # 注意这里必须按照指定的类型优先级顺序，找到母图后替换，不能用找到的文件类型顺序
        for t in ('.eps', '.pdf', '.jpg', '.jpeg', '.wmf', '.emf', '.svg'):
            if t in exts:
                filename = os.path.join(folder, name)
                if t == '.svg':  # svg用inkscape软件单独处理
                    cmd = ['inkscape.exe', '-f', filename + t]  # 使用inkscape把svg转png
                    if trim: cmd.append('-D')  # 裁剪参数
                    if density: cmd.extend(['-d', str(density)])  # 设置dpi参数
                    cmd.extend(['-e', filename + '.png'])
                    executor.submit(subprocess.run, cmd)
                elif t == '.eps':
                    executor.submit(magick, filename + t, outfile=filename + '.png',
                                    transparent=transparent, trim=trim, density=epsdensity)
                else:
                    executor.submit(magick, filename + t, outfile=filename + '.png',
                                    transparent=transparent, trim=trim, density=density)
                break
    executor.shutdown()


def zoomsvg(file, scale=1):
    """
    :param file:
        如果输入一个目录，会处理目录下所有的svg图片
        否则只处理指定的文件
        如果是文本文件，则处理完文本后返回
    :param scale: 缩放的比例，默认100%不调整
    :return:
    """
    if scale == 1: return

    def func(m):
        def g(m): return m.group(1) + str(float(m.group(2)) * scale)

        return re.sub(r'((?:height|width)=")(\d+(?:\.\d+)?)', g, m.group())

    if os.path.isfile(file):
        s = re.sub(r'<svg .+?>', func, Path(file).read(), flags=re.DOTALL)
        Path(file).write(s, if_exists='replace')
    elif os.path.isdir(file):
        for f in os.listdir(file):
            if not f.endswith('.svg'): continue
            f = os.path.join(file, f)
            s = re.sub(r'<svg\s+.+?>', func, Path(f).read(), flags=re.DOTALL)
            Path(file).write(s, if_exists='replace')
    elif isinstance(file, str) and '<svg ' in file:  # 输入svg的代码文本
        return re.sub(r'<svg .+?>', func, file, flags=re.DOTALL)


def reduce_image_filesize(path, filesize):
    """
    :param path: 图片路径，支持png、jpg等多种格式
    :param filesize: 单位Bytes
        可以用 300*1024 来表示 300KB
    :return:

    >> reduce_image_filesize('a.jpg', 300*1024)
    """
    from PIL import Image

    path = Path(path)
    # 1 无论什么情况，都先做个100%的resize处理，很可能会去掉一些没用的冗余信息
    im = Image.open(f'{path}')
    im.resize(im.size).save(f'{path}')

    # 2 然后开始循环处理
    while True:
        r = path.size / filesize
        if r <= 1: break
        # 假设图片面积和文件大小成正比，如果r=4，表示长宽要各减小至1/(r**0.5)才能到目标文件大小
        rate = min(1 / (r**0.5), 0.95)  # 并且限制每轮至少要缩小至95%，避免可能会迭代太多轮
        im = Image.open(f'{path}')
        im.resize((int(im.size[0]*rate), int(im.size[1]*rate))).save(f'{path}')


def image_rgba2rgb(im):
    if im.mode in ('RGBA', 'P'):
        # 判断图片mode模式，如果是RGBA或P等可能有透明底，则和一个白底图片合成去除透明底
        background = Image.new('RGBA', im.size, (255, 255, 255))
        # composite是合成的意思。将右图的alpha替换为左图内容
        im = Image.alpha_composite(background, im.convert('RGBA')).convert('RGB')
    return im


____section_temp = """
临时添加的新功能
"""
