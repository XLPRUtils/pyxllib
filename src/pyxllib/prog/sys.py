#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 11:00

import os
import pprint
import re
import socket
import subprocess
import sys
import tempfile
from statistics import mean
from urllib.parse import urlparse

from pyxllib.prog.lazyimport import lazy_import

try:
    import requests
except ModuleNotFoundError:
    requests = lazy_import('requests')

try:
    from humanfriendly import parse_size
except ModuleNotFoundError:
    parse_size = lazy_import('from humanfriendly import parse_size')

from pyxllib.prog.fmt import human_readable_size
from pyxllib.prog.run import run_once
from pyxllib.file.xlpath import cache_file


def system_information():
    """主要是测试一些系统变量值，顺便再演示一次Timer用法"""

    def pc_messages():
        """演示如何获取当前操作系统的PC环境数据"""
        # fqdn：fully qualified domain name
        print('1、socket.getfqdn() :', socket.getfqdn())  # 完全限定域名，可以理解成pcname，计算机名
        # 注意py的很多标准库功能本来就已经处理了不同平台的问题，尽量用标准库而不是自己用sys.platform作分支处理
        print('2、sys.platform     :', sys.platform)  # 运行平台，一般是win32和linux
        # li = os.getenv('PATH').split(os.path.pathsep)  # 环境变量名PATH，win中不区分大小写，linux中区分大小写必须写成PATH
        # print("3、os.getenv('PATH'):", f'数量={len(li)},', pprint.pformat(li, 4))

    def executable_messages():
        """演示如何获取被执行程序相关的数据"""
        print('1、sys.executable   :', sys.executable)  # 当前被执行脚本位置
        print('2、sys.version      :', sys.version)  # python的版本
        print('3、os.getcwd()      :', os.getcwd())  # 获得当前工作目录
        print('4、gettempdir()     :', tempfile.gettempdir())  # 临时文件夹位置
        # print('5、sys.path       :', f'数量={len(sys.path)},', pprint.pformat(sys.path, 4))  # import绝对位置包的搜索路径

    print('【pc_messages】')
    pc_messages()
    print('【executable_messages】')
    executable_messages()


def is_url(arg):
    """输入是一个字符串，且值是一个合法的url"""
    if not isinstance(arg, str): return False
    try:
        result = urlparse(arg)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def is_file(arg, exists=True):
    """相较于标准库的os.path.isfile，对各种其他错误类型也会判False

    :param exists: arg不仅需要是一个合法的文件名，还要求其实际存在
        设为False，则只判断文件名合法性，不要求其一定要存在
    """
    if not isinstance(arg, str): return False
    if len(arg) > 500: return False
    if not exists:
        raise NotImplementedError
    return os.path.isfile(arg)


def hide_console_window():
    """ 隐藏命令行窗口 """
    import ctypes
    kernel32 = ctypes.WinDLL('kernel32')
    user32 = ctypes.WinDLL('user32')
    SW_HIDE = 0
    hWnd = kernel32.GetConsoleWindow()
    user32.ShowWindow(hWnd, SW_HIDE)


def get_installed_packages():
    """ 使用pip list获取当前环境安装了哪些包 """
    output = subprocess.check_output(["pip", "list"], universal_newlines=True)
    packages = [line.split()[0] for line in output.split("\n")[2:] if line]
    return packages


def check_package(package, speccal_install_name=None):
    """ 250418周五16:36，check_install_package的简化版本，只检查、报错，提示安装依赖，但不自动进行安装
    """
    try:
        __import__(package)
    except ModuleNotFoundError:
        cmds = [sys.executable, "-m", "pip", "install"]
        cmds.append(speccal_install_name if speccal_install_name else package)
        raise ModuleNotFoundError(f'缺少依赖包：{package}，请自行安装扩展依赖：{cmds}\n')


def check_install_package(package, speccal_install_name=None, *, user=False):
    """ https://stackoverflow.com/questions/12332975/installing-python-module-within-code

    :param speccal_install_name: 注意有些包使用名和安装名不同，比如pip install python-opencv，使用时是import cv2，
        此时应该写 check_install_package('cv2', 'python-opencv')

    TODO 不知道频繁调用这个，会不会太影响性能，可以想想怎么提速优化？
    注意不要加@RunOnlyOnce，亲测速度会更慢三倍

    警告: 不要在频繁调用的底层函数里使用 check_install_package
        如果是module级别的还好，调几次其实性能影响微乎其微
        但在频繁调用的函数里使用，每百万次还是要额外的0.5秒开销的
    """
    try:
        __import__(package)
    except ModuleNotFoundError:
        cmds = [sys.executable, "-m", "pip", "install"]
        if user: cmds.append('--user')
        cmds.append(speccal_install_name if speccal_install_name else package)
        subprocess.check_call(cmds)


@run_once()
def get_hostname():
    hostname = socket.getfqdn()
    return hostname


@run_once()
def get_username():
    return os.path.split(os.path.expanduser('~'))[-1]


def get_local_ip():
    """ 获得本地ip，代码由deepseek提供 """
    try:
        # 使用UDP协议连接到外部服务器
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Google的DNS服务器和常用端口
        local_ip = s.getsockname()[0]  # 获取套接字的本地地址
        s.close()
        return local_ip
    except Exception as e:
        # 如果失败，尝试通过主机名获取IP列表
        try:
            hostname = socket.gethostname()
            ips = socket.gethostbyname_ex(hostname)[2]  # 获取所有IPv4地址
            for ip in ips:
                if ip != "127.0.0.1":  # 排除回环地址
                    return ip
            return "127.0.0.1"  # 默认返回回环地址
        except:
            raise ValueError("无法获取IP地址")


def estimate_package_size(package):
    """ 估计一个库占用的存储大小 """

    # 将cache文件存储到临时目录中，避免重复获取网页
    def get_size(package):
        r = requests.get(f'https://pypi.org/project/{package}/#files')
        if r.status_code == 404:
            return '(0 MB'  # 找不到的包默认按0MB计算
        else:
            return r.text

    s = cache_file(package + '.pypi', lambda: get_size(package))
    # 找出所有包大小，计算平均值作为这个包大小的预估
    # 注意，这里进位是x1000，不是x1024
    v = mean(list(map(parse_size, re.findall(r'\((\d+(?:\.\d+)?\s*\wB(?:ytes)?)', s))) or [0])
    return v


def estimate_pip_packages(*, print_mode=False):
    """ 检查pip list中包的大小，从大到小排序

    :param print_mode:
        0，不输出，只返回运算结果，[(package_name, package_size), ...]
        1，输出最后的美化过的运算表格
        2，输出中间计算过程
    """

    def printf(*args, **kwargs):
        # dm表示mode增量
        if print_mode > 1:
            print(*args, **kwargs)

    packages = get_installed_packages()
    package_sizes = []
    for package_name in packages:
        package_size = estimate_package_size(package_name)
        package_sizes.append((package_name, package_size))
        printf(f"{package_name}: {human_readable_size(package_size)}")

    package_sizes.sort(key=lambda x: (-x[1], x[0]))
    if print_mode > 0:
        if print_mode > 1: print('- ' * 20)
        for package_name, package_size in package_sizes:
            print(f"{package_name}: {human_readable_size(package_size)}")
    return package_sizes
