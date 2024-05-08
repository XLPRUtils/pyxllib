#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/04/27

"""
libreoffice这个软件相关的功能

官网：https://www.libreoffice.org/download/download-libreoffice/

linux的安装：
sudo apt-get update
sudo apt-get install libreoffice -y
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime


def check_libreoffice():
    """ 检查LibreOffice是否安装 """
    executor = get_libreoffice_executor()
    try:
        subprocess.run([executor, '--version'], check=True)
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return False
    return True


def get_libreoffice_executor():
    """ 获得可执行文件的名称 """
    # 根据所在系统区分
    if sys.platform == 'win32':
        return 'soffice.exe'
    else:
        return 'libreoffice'


def infer_file_format(file_path):
    """ 推断文件所属办公文档的'主类' """
    ext = Path(file_path).suffix.lower()
    if ext in ['.doc', '.docx']:
        fmt = 'docx'
    elif ext in ['.ppt', '.pptx']:
        fmt = 'pptx'
    elif ext in ['.xls', '.xlsx']:
        fmt = 'xlsx'
    else:
        raise ValueError("不支持的文件格式")
    return fmt


class UpgradeOfficeFile:
    @classmethod
    def to_dir(cls, file_path, out_dir=None, fmt=None, timeout=10):
        """ 将doc文件转换为docx文件

        :param file_path: 待升级的文件路径
        :param out_dir: 输出文件目录
            官方接口默认只能设置导出目录，不能设置导出文件名，文件名是跟原始文件一样的
        :param fmt: 输出文件格式
            docx, xlsx, pptx
        """
        if isinstance(file_path, Path):
            file_path = file_path.as_posix()

        # 获取LibreOffice可执行文件的名称
        executor = get_libreoffice_executor()

        # 如果未指定输出路径，则默认在输入文件的同目录下生成同名的DOCX文件
        if out_dir is None:
            out_dir = os.path.dirname(file_path)

        if fmt is None:
            fmt = infer_file_format(file_path)

        # 构建转换命令
        command = [
            executor,
            '--headless',  # 无界面模式
            '--convert-to', fmt,  # 转换为docx格式
            '--outdir', str(out_dir),  # 输出目录
            file_path  # 输入文件路径
        ]

        subprocess.run(command, timeout=timeout, check=True)

        # 返回转换后的文件路径
        base_name = os.path.basename(file_path)
        name, _ = os.path.splitext(base_name)
        new_file_path = os.path.join(out_dir, f"{name}.{fmt}")

        # todo 以目标文件是否存在判断转换是否成功也是有一定bug的，可能目标文件本来就存在
        #   但如果严谨判断，就要分析subprocess.run的输出结果了，那个太麻烦，先用简便方法处理
        if not Path(new_file_path).exists():
            raise ValueError(f"升级文档失败")

        return new_file_path

    @classmethod
    def to_file(cls, in_file, out_file=None, fmt=None, timeout=10):
        """ 可以指定转换出的文件名的版本

        :param in_file: 待转换的文件路径
        :param out_file: 输出文件路径
            若未指定,默认与原文件同名,在原文件所在目录生成
        :param fmt: 输出文件格式
            docx, xlsx, pptx
            若未指定,则根据in_file的后缀名自动判断
        """
        # 将in_file转换为Path对象
        in_file = Path(in_file)

        # 如果fmt为None,则根据in_file推断
        if fmt is None:
            fmt = infer_file_format(in_file)

        # 如果out_file为None,则默认在原文件目录生成同名的新格式文件
        if out_file is None:
            out_file = in_file.with_suffix(f'.{fmt}')
        else:
            out_file = Path(out_file)

        # 确保out_file的父目录存在,不存在则创建
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # 调用upgrade_office_file函数进行转换
        temp_file = cls.to_dir(in_file, out_dir=out_file.parent, fmt=fmt, timeout=timeout)

        # 将生成的临时文件重命名为out_file
        os.rename(temp_file, out_file)

        return out_file

    @classmethod
    def to_tempfile(cls, in_file, fmt=None, *, timestamp_stem=False, create_subdir=False, timeout=10):
        """ 将文件转换为临时文件

        :param timestamp_stem: 时间戳文件名
        :param create_subdir: 是否在临时目录中创建新的子目录
        """
        if fmt is None:
            fmt = infer_file_format(in_file)

        root = Path(tempfile.gettempdir())
        if create_subdir:
            root2 = root / datetime.now().strftime('%Y%m%d.%H%M%S.%f')
            root2.mkdir(parents=True, exist_ok=True)
            root = root2

        if timestamp_stem:
            stem = datetime.now().strftime('%Y%m%d.%H%M%S.%f')
            out_file = cls.to_file(in_file, out_file=root / f"{stem}.{fmt}", fmt=fmt, timeout=timeout)
        else:
            out_file = cls.to_dir(in_file, out_dir=root, fmt=fmt, timeout=timeout)

        return out_file
