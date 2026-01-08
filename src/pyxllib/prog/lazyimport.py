#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2025/06/05

""" 惰性加载依赖包

写法一：
pd = lazy_import('pandas')

写法二：
try:
    import pandas as pd
except ModuleNotFoundError:
    pd = lazy_import('pandas')

其实工程上直接写法一就够了，如果有pandas就会正常导入返回pd。
但是这样写，IDE就懵了不知道这是import pandas的，所以一般更多用的是写法二的形式。
虽然代码冗余些，但开发的时候会方便的多。
"""

import importlib
import re


class LazyImportError:
    """延迟导入错误对象，支持子模块访问"""

    def __init__(self, module_name, install_name=None,
                 error_class=ModuleNotFoundError, extra_msg=None):
        self._module_name = module_name
        self._error_class = error_class
        self._extra_msg = extra_msg

        # 推断包名和安装命令
        self._install_name = install_name or self._get_package_name(module_name)

    def _get_package_name(self, module_name):
        """根据模块名推断包名"""
        package_map = {
            'PIL': 'Pillow',
            'cv2': 'opencv-python',
            'sklearn': 'scikit-learn',
            'yaml': 'PyYAML',
        }
        base_module = module_name.split('.')[0]
        res = package_map.get(base_module, base_module)
        res = res.replace('_', '-')
        return res

    def _raise_error(self, attempted_action=""):
        msg = f"No module named '{self._module_name}'"
        if attempted_action:
            msg += f" (attempted to {attempted_action})"
        msg += f"\n\nInstall as follows (adjust accordingly for uv, conda, etc.):\n    pip install {self._install_name}"
        if self._extra_msg:
            msg += f"\n\n注意：{self._extra_msg}"
        raise self._error_class(msg)

    def __getattr__(self, name):
        # 返回一个新的 LazyImportError 对象，支持链式访问
        child_module = f"{self._module_name}.{name}"
        return LazyImportError(child_module, self._install_name,
                               self._error_class, self._extra_msg)

    def __call__(self, *args, **kwargs):
        self._raise_error(f"call '{self._module_name}()'")

    def __getitem__(self, key):
        self._raise_error(f"access '{self._module_name}[{key}]'")

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._raise_error(f"set '{self._module_name}.{name}'")

    def __bool__(self):
        # 让 if module: 这样的检查返回 False
        return False

    def __repr__(self):
        return f"<LazyImportError: {self._module_name} not installed>"


def _parse_import_statement(statement):
    """解析导入语句，返回解析结果"""
    statement = statement.strip()

    # 情况1: from ... import ...
    from_match = re.match(r'from\s+([\w.]+)\s+import\s+(.+)', statement)
    if from_match:
        module = from_match.group(1)
        names_str = from_match.group(2)
        # 解析导入的名称列表，处理可能的空格
        names = [n.strip() for n in names_str.split(',')]
        return {'type': 'from', 'module': module, 'names': names}

    # 情况2: import ... as ...
    import_as_match = re.match(r'import\s+([\w.]+)(?:\s+as\s+(\w+))?', statement)
    if import_as_match:
        module = import_as_match.group(1)
        return {'type': 'module', 'module': module}

    # 情况3: 直接是模块名（没有import关键字）
    if re.match(r'^[\w.]+$', statement):
        return {'type': 'module', 'module': statement}

    # 无法解析
    return {'type': 'unknown'}


def _import_module(module_path, install_name=None, extra_msg=None):
    """导入模块，失败时返回LazyImportError"""
    parts = module_path.split('.')
    base_module = parts[0]

    try:
        # 尝试导入完整的模块路径
        module = importlib.import_module(module_path)

        # 对于子模块导入（如 PIL.Image），我们需要返回基础模块
        if len(parts) > 1:
            # 返回基础模块，这样 PIL.Image 会返回 PIL
            base = importlib.import_module(base_module)
            return base
        else:
            return module

    except (ModuleNotFoundError, ImportError):
        # 模块不存在，创建 LazyImportError
        # 使用基础模块名作为错误对象的模块名
        return LazyImportError(base_module, install_name, extra_msg=extra_msg)


def _import_from_module(module_path, names, install_name=None, extra_msg=None):
    """从模块导入特定对象，失败时返回LazyImportError的属性"""
    try:
        module = importlib.import_module(module_path)

        # 获取请求的对象
        if len(names) == 1:
            return getattr(module, names[0])
        else:
            return [getattr(module, name) for name in names]

    except (ModuleNotFoundError, ImportError):
        error_obj = LazyImportError(module_path, install_name, extra_msg=extra_msg)

        # 返回错误对象的属性
        if len(names) == 1:
            return getattr(error_obj, names[0])
        else:
            return [getattr(error_obj, name) for name in names]


def lazy_import(module_name, install_name=None, extra_msg=None):
    """
    智能安全导入，支持多种导入语法

    参数:
        module_name: import时使用的模块名或完整的import语句
        install_name: 安装时使用的包名，默认为模块名（会自动推断常见的映射关系）
        extra_msg: 额外提示信息

    返回:
        - 对于普通import：返回模块对象
        - 对于from import单个：返回具体对象
        - 对于from import多个：返回对象列表

    示例:
        # 普通导入
        pd = lazy_import('pandas')
        pd = lazy_import('import pandas')

        # 子模块导入（返回基础模块）
        PIL = lazy_import('PIL.Image')  # 返回PIL，使用PIL.Image时触发错误
        PIL = lazy_import('import PIL.Image')  # 同上

        # from import
        setenv = lazy_import('from envariable import setenv')
        setenv, unsetenv = lazy_import('from envariable import setenv, unsetenv')

        # 自定义安装包名
        cv2 = lazy_import('cv2', install_name='opencv-python')

    不过这个工具并不是完美的，如果只是纯粹链式访问常量等，不触发call、下标索引，是不会报错的，而是返回LazyImportError对象
    这个功能不能改，改了又会不支持PIL.Image等相关形式的处理。只能使用中尽量注意。
    """

    # 解析输入语句
    parsed = _parse_import_statement(module_name)

    if parsed['type'] == 'module':
        # 处理普通模块导入
        return _import_module(parsed['module'], install_name, extra_msg)

    elif parsed['type'] == 'from':
        # 处理 from ... import ... 语句
        return _import_from_module(parsed['module'], parsed['names'],
                                   install_name, extra_msg)

    else:
        raise ValueError(f"Cannot parse import statement: {module_name}")


if __name__ == "__main__":
    # 测试普通导入
    pd = lazy_import('pandas')
    print(f"pd: {pd}")

    # 测试子模块导入
    PIL = lazy_import('PIL.Image')
    print(f"PIL: {PIL}")

    # 测试from import
    m, n = lazy_import('from nonpackage.a import m,n', 'nonpackage>=1.0')
    print(f"m: {m}")

    # 尝试使用会触发错误
    m.func()
    # ModuleNotFoundError: No module named 'nonpackage.m.func' (attempted to call 'nonpackage.m.func()')
    #
    # Install as follows (adjust accordingly for uv, conda, etc.):
    #     pip install nonpackage>=1.0
