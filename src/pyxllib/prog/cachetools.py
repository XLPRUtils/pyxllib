#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/05/29

# 对于普通函数，一般用lru_cache即可
from functools import lru_cache
import threading

from deprecated import deprecated

from pyxllib.prog.lazyimport import lazy_import
from pyxllib.prog.uni_cache import uni_cache

try:
    # 官方文档：https://pypi.org/project/cached-property/
    from cached_property import (
        cached_property,
        threaded_cached_property,  # 线程安全
        cached_property_with_ttl,  # 限时缓存，单位秒
        threaded_cached_property_with_ttl  # 线程 + 限时
    )
except ModuleNotFoundError:
    cached_property = lazy_import('from cached_property import cached_property')
    threaded_cached_property = lazy_import('from cached_property import threaded_cached_property')
    cached_property_with_ttl = lazy_import('from cached_property import cached_property_with_ttl')
    threaded_cached_property_with_ttl = lazy_import('from cached_property import threaded_cached_property_with_ttl')


# 进一步封装的更通用、自己常用的装饰器

@deprecated(reason="请使用 pyxllib.prog.uni_cache.uni_cache 代替")
def xlcache(maxsize=128, *, ttl=None, lock=None, property=False):
    """ 那些工具接口太复杂难记，自己封装一个统一的工具

    就是一个装饰器，最大缓存多少项，然后是否要开多线程安全，是否要设置限时重置，是否是作为类成员属性修饰

    :param property: 是否作为类成员属性修饰，不过一般不建议通过这里设置，
        而是外部再加一层@property，不然IDE会识别不了这是一个property，影响开发

    """
    if property:
        def decorator(func):
            if ttl is not None:
                if lock:
                    return threaded_cached_property_with_ttl(ttl)(func)
                else:
                    return cached_property_with_ttl(ttl)(func)
            else:
                if lock:
                    return threaded_cached_property(func)
                else:
                    return cached_property(func)
        return decorator
    else:
        return uni_cache(maxsize=maxsize, ttl=ttl)
