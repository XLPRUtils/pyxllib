#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/04/04 17:03

""" 专门给utools的快捷命令扩展的一系列python工具库
"""
from pyxllib.prog.pupil import check_install_package

check_install_package('fire')
check_install_package('humanfriendly')
check_install_package('pandas')
check_install_package('pyautogui', 'PyAutoGui')  # 其实pip install不区分大小写，不过官方这里安装是驼峰名

import datetime
import json
import os
import pathlib
import pyperclip
import re

import fire
from humanfriendly import format_timespan
import pandas as pd
import pyautogui

from pyxllib.file.specialist import File, Dir
from pyxllib.debug.specialist import browser, TicToc, parse_datetime
from pyxllib.robot.autogui import type_text, clipboard_decorator


def _print_df_result(df, outfmt='text'):
    # subinput可以强制重置输出类型
    # TODO argparser可以对str类型进行参数解析？
    # if 'browser' in kwargs['subinput']:
    #     outfmt = 'browser'
    # elif 'html' in kwargs['subinput']:
    #     outfmt = 'html'
    # elif 'text' in kwargs['subinput']:
    #     outfmt = 'text'

    if outfmt == 'html':
        # content = df.to_html().replace('–', '-')  # 很多app会出现一个utf8特殊的-，转gbk编码会出错+
        try:  # utools里按gbk会有很多编码问题，如果不能直接显示，就打开浏览器看
            print(df.to_html())
        except UnicodeEncodeError:
            browser(df)
    elif outfmt == 'browser':
        browser(df)
    else:
        with pd.option_context('display.max_colwidth', -1, 'display.max_columns', 20,
                               'display.width', 200):  # 上下文控制格式
            print(df)


class UtoolsBase:
    def __init__(self, cmds, *, outfmt='text'):
        """
        :param cmds: "快捷命令"响应的命令集，即调用出可传入的输入文件、所在窗口、选中文件等信息
        :param outfmt: 输出格式
            text，纯文本
            html，html格式
            browser，用浏览器打开
        """
        # 1 删除没有值的键
        self.cmds = dict()
        for k, v in cmds.items():
            if cmds[k] not in ('{{' + k + '}}', ''):
                self.cmds[k] = v

        # 2 「快捷命令」的ClipText宏展开，如果存在连续3个单引号，会跟py字符串定义冲突，产生bug
        #   所以这段不建议用宏展开，这里该类会自动调用pyperclip获取
        #   这个可以作为可选功能关闭，但实验证明这个对性能其实没有太大影响
        if 'ClipText' not in self.cmds:
            self.cmds['ClipText'] = pyperclip.paste().replace('\r\n', '\n')

        # 3 解析json数据
        # 解析window模式的WindowInfo
        if 'WindowInfo' in self.cmds:
            self.cmds['WindowInfo'] = json.loads(self.cmds['WindowInfo'].encode('utf8'))

        # 解析files模式的MatchedFiles
        if 'MatchedFiles' in self.cmds:
            # 右边引用没写错。因为MatchedFiles本身获取内容不稳定，可能会有bug，取更稳定的payload来初始化MatchedFiles
            self.cmds['MatchedFiles'] = json.loads(self.cmds['payload'].encode('utf8'))
            # 在有些软件中，比如everything，是可以选中多个不同目录的文件的。。。所以严格来说不能直接按第一个文件算pwd
            # 但这个讯息是可以写一个工具分析的
            # self.cmds['pwd'] = os.path.dirname(self.cmds['MatchedFiles'][0]['path'])

            # TODO 如果选中的文件全部是同一个目录下的，要存储pwd
            dirs = {os.path.dirname(f['path']) for f in self.cmds['MatchedFiles']}
            if len(dirs) == 1:
                self.cmds['pwd'] = list(dirs)[0]

        # 4 当前工作目录
        # 要根据场景切换pwd
        #   files默认工作目录是app: C:\Users\chen\AppData\Local\Programs\utools
        #   window默认工作目录是exe
        # window模式存储的pwd跟各种软件有关
        #   explorer就是当前窗口，没问题
        #   onenote存的是桌面
        if 'pwd' in self.cmds:
            os.chdir(self.cmds['pwd'])

        # 5 subinput可以用字典结构扩展参数
        if 'subinput' in self.cmds:
            if re.match(r'\{.+\}$', self.cmds['subinput']):
                ext_kwargs = eval(self.cmds['subinput'])
                self.cmds.update(ext_kwargs)
        else:
            self.cmds['subinput'] = ''

        self.outfmt = self.cmds['outfmt'] if 'outfmt' in self.cmds else outfmt

    def check_cmds(self):
        """ 显示所有参数值 """
        df = pd.DataFrame.from_records([(k, v) for k, v in self.cmds.items()],
                                       columns=['key', 'content'])
        _print_df_result(df, self.outfmt)

    def bcompare(self):
        """ 可以通过subinput设置文件后缀类型 """
        from pyxllib.debug.specialist import bcompare

        suffix = self.cmds.get('subinput', None)
        try:
            f1 = File('left', Dir.TEMP, suffix=suffix)
            f2 = File('right', Dir.TEMP, suffix=suffix)
        except OSError:  # 忽略错误的扩展名
            f1 = File('left', Dir.TEMP)
            f2 = File('right', Dir.TEMP)
        f1.write(suffix)
        f2.write(self.cmds['ClipText'])
        bcompare(f1, f2, wait=False)


class UtoolsFile(UtoolsBase):
    """ 文件相关操作工具 """

    def __init__(self, cmds, *, outfmt='text'):
        super().__init__(cmds, outfmt=outfmt)

        # 如果是window等模式，补充 MatchedFiles
        if 'MatchedFiles' in self.cmds:
            self.paths = [pathlib.Path(f['path']) for f in self.cmds['MatchedFiles']]
        else:
            self.paths = [pathlib.Path(os.path.abspath(f)).resolve() for f in os.listdir('.')]

    @classmethod
    def is_image(cls, f):
        return bool(re.match('jpe?g|png|gif|bmp', f.suffix.lower()[1:]))

    def codefile(self):
        """ 多用途文件处理器

        核心理念是不设定具体功能，而是让用户在 subinput 自己写需要执行的py代码功能
        而在subinput，可以用一些特殊的标识符来传参

        以file为例，有4类参数：
            file，处理当前目录 文件
            rfile，递归处理所有目录下 文件
            files，当前目录 所有文件
            rfiles，递归获取所有目录下 所有文件

        类型上，file可以改为
            dir，只分析目录
            path，所有文件和目录

        扩展了一些特殊的类型：
            imfile，图片文件
        """
        from functools import reduce
        try:
            from xlproject.kzconfig import KzDataSync
            from pyxllib.data.labelme import reduce_labelme_jsonfile
        except ModuleNotFoundError:
            pass

        tt = TicToc()

        # 1 获得所有标识符
        # 每个单词前后都有空格，方便定界
        keywords = filter(lambda x: re.match(r'[a-zA-Z_]+$', x),
                          set(re.findall(r'\.?[a-zA-Z_]+\(?', self.cmds['subinput'])))
        keywords = ' ' + ' '.join(keywords) + ' '
        # print('keywords:', keywords)

        # 2 生成一些备用的智能参数
        # 一级目录下选中的文件
        paths = self.paths
        # 用正则判断一些智能参数是否要计算，注意不能只判断files，如果出现file，也是要引用files的
        files = [File(p) for p in paths if p.is_file()] if re.search(r'files? ', keywords) else []
        imfiles = [f for f in files if self.is_image(f)] if re.search(r' imfiles? ', keywords) else []
        # 出现r系列的递归操作，都是要计算出dirs的
        dirs = [Dir(p) for p in paths if p.is_dir()] if re.search(r' (dirs?|r[a-zA-Z_]+) ', keywords) else []

        # 递归所有的文件
        rpaths = reduce(lambda x, y: x + y.select('**/*').subpaths(), [paths] + dirs) \
            if re.search(r' (r[a-zA-Z_]+) ', keywords) else []
        rfiles = [File(p) for p in rpaths if p.is_file()] if re.search(r' (r(im)?files?) ', keywords) else []
        rimfiles = [f for f in rfiles if self.is_image(f)] if re.search(r' rimfiles? ', keywords) else []
        rdirs = [Dir(p) for p in rpaths if p.is_dir()] if re.search(r' (rdirs?) ', keywords) else []

        # 3 判断是否要智能开循环处理
        m = re.search(r' (r?(path|(im)?file|dir)) ', keywords)
        if m:
            name = m.group(1)
            objs = eval(name + 's')
            print('len(' + name + 's)=', len(objs))
            for x in objs:
                locals()[name] = x
                eval(self.cmds['subinput'])
        else:  # 没有的话就直接处理所有文件
            eval(self.cmds['subinput'])

        # 4 运行结束标志
        print(f'finished in {format_timespan(tt.tocvalue())}.')


class UtoolsText(UtoolsBase):
    """ 目录路径生成工具 """

    @clipboard_decorator(copy=False, typing=True)
    def common_dir(self):
        from pyxlpr.data.datasets import CommonDir

        def func(name, unix_path=False):
            p = getattr(CommonDir, name)

            if unix_path:
                # 因为用了符号链接，实际位置会变回D:/slns，这里需要反向替换下
                p = str(p).replace('D:/slns', 'D:/home/chenkunze/slns')
                p = str(p)[2:]
            else:
                # slns比较特别，本地要重定向到D:/slns
                p = str(p).replace('D:/home/chenkunze/slns', 'D:/slns')
                p = p.replace('/', '\\')

            return p

        return fire.Fire(func, self.cmds['subinput'], 'CommonDir')

    def wdate(self):
        """ week dates 输入一周的简略日期值 """

        def func(start):
            weektag = '一二三四五六日'
            for i in range(7):
                dt = parse_datetime(start) + datetime.timedelta(i)
                type_text(dt.strftime('%y%m%d') + f'周{weektag[dt.weekday()]}')
                pyautogui.press('down')

        fire.Fire(func, self.cmds['subinput'], 'wdate')

    def input_digits(self):
        """ 输入数字 """

        def func(start, stop, step=1):
            for i in range(start, stop, step):
                pyautogui.write(str(i))
                pyautogui.press('down')

        fire.Fire(func, self.cmds['subinput'], 'input_digits')

    def __win32(self):
        """ win32相关自动化

        目前主要是word自动化相关的功能，这里有很多demo可以学习怎么用win32做word的自动化
        """
        pass

    def browser(self):
        """ 将内容复制到word，另存为html文件后，用浏览器打开查看 """
        from pyxllib.file.docxlib import rebuild_document_by_word

        file = fire.Fire(rebuild_document_by_word, self.cmds['subinput'], 'browser')
        browser(file)


class UtoolsRegex(UtoolsBase):
    def __init__(self, cmds, *, outfmt='text'):
        super().__init__(cmds, outfmt=outfmt)

    @clipboard_decorator(paste=True)
    def coderegex(self):
        # tt = TicToc()
        s = self.cmds['ClipText']
        return eval(self.cmds['subinput'])
        # print(f'finished in {format_timespan(tt.tocvalue())}.')

    @clipboard_decorator(paste=True)
    def refine_text(self, func):
        return func(self.cmds['ClipText'])


if __name__ == '__main__':
    with TicToc('utools'):
        pass
