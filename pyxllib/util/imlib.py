#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
任何模块代码的第一个字符串用来写文档注释
"""

__author__ = '陈坤泽'
__email__ = '877362867@qq.com'
__date__ = '2018/07/11 19:14'

import struct

from code4101py.util.filelib import *

try:
    import PIL
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'pillow'])
    import PIL

try:
    import fitz
except ModuleNotFoundError:
    # subprocess.run(['pip3', 'install', 'PyMuPdf'])
    pass

# import fitz

try:
    from get_image_size import get_image_size
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'opsdroid-get-image-size'])
    from get_image_size import get_image_size


# import PIL.ExifTags
# from PIL import Image


def magick(infile, *, outfile=None, force=True, transparent=None, trim=False, density=None, other_args=None):
    """调用iamge magick的magick.exe工具
    :param infile: 处理对象文件
    :param outfile: 输出文件，可以不写，默认原地操作（只设置透明度、裁剪时可能会原地操作）
    :param force:
        True: 即使outfile已存在，也重新覆盖生成
        False: 如果outfile已存在，不生成
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
    # 1、条件判断，有些情况下不用处理
    if not outfile: outfile = infile
    # 所有转换参数都没有设置，则不处理
    if not force and Path(outfile).is_file(): return outfile
    # 判断是否是支持的输入文件类型
    ext = os.path.splitext(infile)[1].lower()
    if not Path(infile).is_file() or not ext in ('.png', '.eps', '.pdf', '.jpg', '.jpeg', '.wmf', '.emf'): return False

    # 2、生成需要执行的参数
    cmd = ['magick.exe']
    # 透明、裁剪、density都是可以重复操作的
    if density: cmd.extend(['-density', str(density)])
    cmd.append(infile)
    if transparent:
        if not isinstance(transparent, str): transparent = 'white'
        cmd.extend(['-transparent', transparent])
    if trim: cmd.append('-trim')
    if other_args: cmd.extend(other_args)
    cmd.append(outfile)

    # 3、生成目标png图片
    print(' '.join(cmd))
    subprocess.run(cmd, stderr=subprocess.PIPE, shell=True)  # 看不懂magick出错写的是啥，关了
    return outfile


def pdf2svg_oldversion(pdffile, target=None, *, trim=False):
    """新版的，pymupdf生成的svg无法配合inkscape进行trim，所以这个旧版暂时还是要保留

    :param pdffile: 一份pdf文件
    :param target: 目标目录
        None：
            如果只有一页，转换到对应目录下同名svg文件
            如果有多页，转换到同名目录下多份svg文件
    :param trim:
        True: 去除边缘空白
    :return:

    需要第三方工具：pdf2svg（用于文件格式转换），inkscape（用于svg编辑优化）
        注意pdf2svg的参数不支持中文名，因为这个功能限制，搞得我这个函数实现好麻烦！
        还要在临时文件夹建立文件，因为重名文件+多线程问题，还曾引发一个bug搞了一下午。
        （这些软件都以绿色版形式整理在win3/imgtools里）

    注意！！！ 这个版本的代码先不要删！先不要删！先不要删！包括pdf2svg.exe那个蠢货软件也先别删！
        后续研究inkscape这个蠢货的-D参数在处理pymupdf生成的svg为什么没用的时候可以进行对比
    """
    import fitz
    pages = fitz.open(pdffile).pageCount

    basename = tempfile.mktemp()
    f1 = basename + '.pdf'
    filescopy(pdffile, f1)  # 复制到临时文件，防止中文名pdf2svg工具处理不了

    if pages == 1:
        if target is None: target = pdffile[:-3] + 'svg'
        f2 = basename + '.svg'
        # print(['pdf2svg.exe', f1, f2])
        subprocess.run(['pdf2svg.exe', f1, f2])

        if trim: subprocess.run(['inkscape.exe', '-f', f2, '-D', '-l', f2])
        filescopy(f2, target)
    else:
        if target is None: target = pdffile[:-4] + '_svg\\'
        executor = concurrent.futures.ThreadPoolExecutor()
        Path(basename + '/').ensure_dir()

        def func(f1, f2, i):
            subprocess.run(['pdf2svg.exe', f1, f2, str(i)])
            if trim: subprocess.run(['inkscape.exe', '-f', f2, '-D', '-l', f2])
            filescopy(f2, target + f'{i}.svg')

        for i in range(1, pages + 1):
            f2 = basename + f'\\{i}.svg'
            executor.submit(func, f1, f2, i)
        executor.shutdown()
        filescopy(basename, target[:-1])
        filesdel(basename + '/')

    filesdel(f1)


def pdf2imagebase(pdffile, target=None, scale=None, ext='.png'):
    """使用python的PyMuPdf模块，不需要额外插件
    导出的图片从1开始编号
    TODO 要加多线程？效率影响大吗？

    :param pdffile: pdf原文件
    :type target: 相对于原文件所在目录的目标目录名，也可以写文件名，表示重命名
        None：
            当该pdf只有1页时，才默认把图片转换到当前目录。
            否则默认新建一个文件夹来存储图片。（目录名默认为文件名）
    :param scale: 缩放尺寸
        1：原尺寸
        1.5：放大为原来的1.5倍
    :param ext: 导出的图片格式
    :param return: 返回生成的图片列表
    """
    import fitz
    # 1、基本参数计算
    pdf = fitz.open(pdffile)
    num_pages = pdf.pageCount

    # 大于1页的时候，默认新建一个文件夹来存储图片
    if target is None and num_pages > 1: target = Path(pdffile).stem + '/'

    newfile = Path(pdffile).abs_dstpath(target)
    if newfile.endswith('.pdf'): newfile = os.path.splitext(newfile)[0] + ext
    Path(newfile).ensure_dir()

    # 2、图像数据的获取
    def get_svg_image(n):
        page = pdf.loadPage(n)
        txt = page.getSVGimage()
        if scale: txt = zoomsvg(txt, scale)
        return txt

    def get_png_image(n):
        """获得第n页的图片数据"""
        page = pdf.loadPage(n)
        if scale:
            pix = page.getPixmap(fitz.Matrix(scale, scale))  # 长宽放大到scale倍
        else:
            pix = page.getPixmap()
        return pix.getPNGData()

    # 3、分析导出的图片文件名
    files = []
    if num_pages == 1:
        image = get_svg_image(0) if ext == '.svg' else get_png_image(0)
        files.append(newfile)
        writefile(image, newfile, if_exists='replace')
    else:  # 有多页
        number_width = math.ceil(math.log10(num_pages + 1))  # 根据总页数计算需要的对齐域宽
        stem, ext = os.path.splitext(newfile)
        for i in range(num_pages):
            image = get_svg_image(i) if ext == '.svg' else get_png_image(i)
            name = ('-{:0' + str(number_width) + 'd}').format(i + 1)
            files.append(stem + name + ext)
            writefile(image, stem + name + ext, if_exists='replace')
    return files


def pdf2png(pdffile, target=None, scale=None):
    pdf2imagebase(pdffile, target=target, scale=scale, ext='.png')


def pdf2svg(pdffile, target=None, scale=None, trim=False):
    """
    :param pdffile: 见pdf2imagebase
    :param target: 见pdf2imagebase
    :param scale: 见pdf2imagebase
    :param trim: 如果使用裁剪功能，会调用pdf-crop-margins第三方工具
        https://pypi.org/project/pdfCropMargins/
    :return:
    """
    if trim:  # 先对pdf文件进行裁剪再转换
        pdf = Path(pdffile)
        newfile = pdf.abs_dstpath('origin.pdf')
        pdf.copy(newfile)
        # subprocess.run(['pdf-crop-margins.exe', '-p', '0', newfile, '-o', pdffile], stderr=subprocess.PIPE) # 本少： 会裁过头！
        # 本少： 对于上下边处的 [] 分数等，会裁过头，先按百分比 -p 0 不留边，再按绝对点数收缩/扩张 -a -1  负数为扩张，单位为bp
        # 本少被自己坑了，RamDisk 与 pdf-crop-margins.exe 配合，只能取 SCSI 硬盘，如果 Direct-IO 就不行，还不报错，还以为是泽少写的代码连报错都不会
        subprocess.run(['pdf-crop-margins.exe', '-p', '0', '-a', '-1', newfile, '-o', pdffile],
                       stderr=subprocess.PIPE)
    # TODO 有时丢图
    pdf2imagebase(pdffile, target=target, scale=scale, ext='.svg')


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

    # 1、提取字典d，key<str>: 文件stem名称， value<set>：含有的扩展名
    d = defaultdict(set)
    for file in os.listdir(folder):
        if file.endswith('-eps-converted-to.pdf'): continue
        name, ext = os.path.splitext(file)
        ext = ext.lower()
        if ext in ('.png', '.eps', '.pdf', '.jpg', '.jpeg', '.wmf', '.emf', '.svg'):
            d[name].add(ext)

    # 2、遍历处理每个stem的图片
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
        writefile(s, file, if_exists='replace')
    elif os.path.isdir(file):
        for f in os.listdir(file):
            if not f.endswith('.svg'): continue
            f = os.path.join(file, f)
            s = re.sub(r'<svg\s+.+?>', func, Path(f).read(), flags=re.DOTALL)
            writefile(s, f, if_exists='replace')
    elif isinstance(file, str) and '<svg ' in file:  # 输入svg的代码文本
        return re.sub(r'<svg .+?>', func, file, flags=re.DOTALL)


def pdfs2pngs(path, scale=None):
    """pdf教材批量转图片
    :param path: 要处理的目录
    :param scale: 控制pdf2png的尺寸，一般要设1.2会清晰些
    :return:

    这个函数转换中，不要去删除原文件！不要去删除原文件！删除原文件请另外写功能。
    警告：该函数针对pdf课本转图片上传有些定制功能，并不是纯粹的通用功能函数
    """
    cwd = os.getcwd()

    # 1、第1轮遍历，生成所有png
    for dirpath, dirnames, filenames in os.walk(path):
        os.chdir(dirpath)
        dprint(dirpath)
        executor = concurrent.futures.ThreadPoolExecutor(4)
        for file in filenames:
            if file.endswith('.pdf'):
                executor.submit(pdf2png, file, scale=scale)
            if file.endswith('.jpg'):  # 大王物理中有jpg图片
                executor.submit(subprocess.run, ['magick.exe', file, file[:-4] + '.png'])
        executor.shutdown()

    # 2、第2轮遍历，找出宽与高比例在1.3~1.6的png图片，只裁剪出右半部分
    dprint('2、第2轮遍历，找出宽与高比例在1.3~1.6的png图片，只裁剪出右半部分')
    for dirpath, dirnames, filenames in os.walk(path):
        os.chdir(dirpath)
        executor = concurrent.futures.ThreadPoolExecutor(4)
        for file in filenames:
            if not file.endswith('.png'): continue
            w, h = get_image_size(file)
            if 1.3 <= w / h <= 1.6:  # 有的文件太大PIL处理不了，所以还是让magick来搞~~
                half_w = w // 2
                executor.submit(subprocess.run, ['mogrify.exe', '-crop', f'{half_w}x{h}+{w - half_w}+0', file])
        executor.shutdown()

    # 3、第3轮遍历，宽超过1000的，压缩到1000内
    dprint('3、第3轮遍历，宽超过1000的，压缩到1000内')
    for dirpath, dirnames, filenames in os.walk(path):
        os.chdir(dirpath)
        executor = concurrent.futures.ThreadPoolExecutor(4)
        for file in filenames:
            if not file.endswith('.png'): continue
            w, h = get_image_size(file)
            if w > 1000: executor.submit(subprocess.run, ['mogrify.exe', '-resize', '1000x', file])
        executor.shutdown()

    os.chdir(cwd)  # 恢复原工作目录


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
    # 1、无论什么情况，都先做个100%的resize处理，很可能会去掉一些没用的冗余信息
    im = Image.open(f'{path}')
    im.resize(im.size).save(f'{path}')

    # 2、然后开始循环处理
    while True:
        r = path.size / filesize
        if r <= 1: break
        # 假设图片面积和文件大小成正比，如果r=4，表示长宽要各减小至1/(r**0.5)才能到目标文件大小
        rate = min(1 / (r**0.5), 0.95)  # 并且限制每轮至少要缩小至95%，避免可能会迭代太多轮
        im = Image.open(f'{path}')
        im.resize((int(im.size[0]*rate), int(im.size[1]*rate))).save(f'{path}')


____section_temp = """
临时添加的新功能
"""
