#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/05/30 11:34

"""
https://www.yuque.com/xlpr/pyxllib/depend
"""

from setuptools import setup, find_packages

basic = """
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
"""

debug = """
pympler
lxml
beautifulsoup4
numpy
Jinja2
pandas>=0.23.4
"""

image = """
pillow
PyMuPdf
"""

# shapely 要优先用 conda 尝试安装
cv = """
pillow
opsdroid-get-image-size
opencv-python
shapely
"""

text = """
python-Levenshtein
pyspellchecker
pyahocorasick
"""

data = """
oss2
sqlalchemy
mysqlclient
"""

# 这一块比较特别，可能都是比较小众的功能，强烈建议需要的时候手动安装依赖项
robot = """
pyautogui
keyboard
"""

more = """
paramiko
scp
"""

extras_require = {}
extras_require['file'] = '\n'.join(basic.splitlines())
extras_require['debug'] = '\n'.join((basic + debug).splitlines())
extras_require['image'] = '\n'.join((basic + image).splitlines())
extras_require['cv'] = '\n'.join((basic + cv).splitlines())
extras_require['text'] = '\n'.join((basic + text).splitlines())
extras_require['data'] = '\n'.join((basic + data).splitlines())
extras_require['robot'] = '\n'.join((basic + robot).splitlines())
extras_require['more'] = '\n'.join((basic + more).splitlines())
extras_require['most'] = '\n'.join((basic + debug + image + cv + text + data + more).splitlines())

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
    install_requires=open('requirements.txt').readlines(),
    extras_require=extras_require,
)
