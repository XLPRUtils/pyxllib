#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/06/02 16:00


from collections import defaultdict
import concurrent.futures
import os
import re
import subprocess
from pathlib import Path


def backup_file(path: Path):
    if not path.exists():
        return path
    for i in range(1, 10000):
        b = path.with_name(path.name + f'.bak{i}')
        if not b.exists():
            path.rename(b)
            return b
    raise RuntimeError(str(path))


def magick(infile, *, outfile=None, if_exists='error', transparent=None, trim=False, density=None, other_args=None):
    """ 调用iamge magick的magick.exe工具

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
    if not outfile:
        outfile = infile

    # 2
    # 200914周一20:40，这有个相对路径的bug，修复了下，否则 test/a.png 会变成 test/test/a.png
    outp = Path(outfile)
    if outp.exists():
        if if_exists in ('ignore', 'skip'):
            return outfile
        elif if_exists == 'backup':
            backup_file(outp)
        elif if_exists == 'replace':
            pass
        else:
            raise FileExistsError(str(outp))

    ext = os.path.splitext(infile)[1].lower()
    if not Path(infile).is_file() or not ext in ('.png', '.eps', '.pdf', '.jpg', '.jpeg', '.wmf', '.emf'):
        return False

    cmd = ['magick.exe']
    if density:
        cmd.extend(['-density', str(density)])
    cmd.append(infile)
    if transparent:
        transparent_ = transparent if isinstance(transparent, str) else 'white'
        cmd.extend(['-transparent', transparent_])
    if trim:
        cmd.append('-trim')
    if other_args:
        cmd.extend(other_args)
    cmd.append(outfile)

    cmd = [x.replace('\\', '/') for x in cmd]
    print(' '.join(cmd))
    subprocess.run(cmd, stderr=subprocess.PIPE, shell=True)

    return outfile


def ensure_pngs(folder, *, if_exists='skip',
                transparent=None, trim=False,
                density=None, epsdensity=None,
                max_workers=None):
    """ 确保一个目录下的所有图片都有一个png版本格式的文件

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
                backup_file(Path(folder) / f'{name}.png')
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
    """ 缩放svg文件

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
        p = Path(file)
        s = re.sub(r'<svg .+?>', func, p.read_text(encoding='utf-8', errors='ignore'), flags=re.DOTALL)
        p.write_text(s, encoding='utf-8', errors='ignore')
    elif os.path.isdir(file):
        for f in os.listdir(file):
            if not f.endswith('.svg'): continue
            f = os.path.join(file, f)
            p = Path(f)
            s = re.sub(r'<svg\s+.+?>', func, p.read_text(encoding='utf-8', errors='ignore'), flags=re.DOTALL)
            p.write_text(s, encoding='utf-8', errors='ignore')
    elif isinstance(file, str) and '<svg ' in file:  # 输入svg的代码文本
        return re.sub(r'<svg .+?>', func, file, flags=re.DOTALL)
