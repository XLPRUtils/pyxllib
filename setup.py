#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/05/30 11:34

"""
https://www.yuque.com/xlpr/pyxllib/depend
"""

from setuptools import setup, find_packages

xl = """
arrow
chardet
requests
qiniu
pyyaml
coloredlogs
humanfriendly
tqdm
ujson
Deprecated
pyperclip

pympler
lxml
beautifulsoup4
numpy
Jinja2
pandas>=0.23.4
"""

xlcv = """
pillow
opsdroid-get-image-size
opencv-python
"""

setup(
    name='pyxllib',  # pip 安装时用的名字
    version='0.0.79',  # 当前版本，每次更新上传到pypi都需要修改; 第4位版本号一般是修紧急bug
    author='code4101',
    author_email='877362867@qq.com',
    url='https://github.com/XLPRUtils/pyxllib',
    keywords='pyxllib',
    description='厦门理工模式识别团队通用python代码工具库',
    long_description=open('README.md', encoding='utf8').read(),  # 偷懒，就不创建f然后close了~~
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=('tests', 'tests.*', '*account.pkl')),
    include_package_data=True,
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',  # 开发的目标用户
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',  # 大部分功能都是跨平台的
    ],
    python_requires='>=3.6',  # 我的项目大量使用f字符串
    # 必须安装的模块，这里故意设为空，方便一些需要精简安装使用的场景
    # ①如需要打包生成exe，尽量避免install多余的包
    # ②对运算性能速度有极致要求的场景，不使用pyxllib.xl接口，尽量避免import任何多余的无用代码
    install_requires=open('requirements.txt').readlines(),
    # xl的扩展安装如下，xlcv同理。本库最常用的接口模式为 from pyxllib.xl import *
    # ①静态版：pip install pyxllib[xl]
    # ②开发版：python setup.py develop easy_install pyxllib[xl]
    extras_require={'xl': '\n'.join(xl.splitlines()),
                    'xlcv': '\n'.join((xl + xlcv).splitlines())},
)
