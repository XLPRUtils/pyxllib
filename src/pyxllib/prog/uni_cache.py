# -*- coding: utf-8 -*-
"""
uni_cache.py - Unified & Universal Cache

基于 cachebox 实现的高性能大一统缓存装饰器。
它整合了函数缓存(Memoization)、单例模式、多例模式，解决了 lru_cache 不支持不可哈希参数、
无法精确控制 OOP 缓存作用域、缺乏生命周期管理接口等痛点。

核心特性与用法示例：

1. 基础用法：支持 list/dict 参数，自动参数对齐
   原生 lru_cache 遇到 list 参数会报错，且 func(1, b=2) 与 func(1, 2) 视为不同。
   uni_cache 通过 inspect 签名绑定和零拷贝递归 Tuple 转换解决此问题。

   >>> @uni_cache
   ... def analyze(data_list, config): return sum(data_list)
   >>> analyze([1, 2], {'k': 'v'})  # 正常运行，不会报错

2. OOP 模式 A：实例隔离 (Instance Isolation)
   场景：不同对象的同名方法需要独立缓存。
   策略：mode='id, str' (第1个参数 self 用 id 区分，后续参数用值区分)。

   >>> class User:
   ...     @uni_cache(mode='id, str')
   ...     def get_report(self, day): ...

3. OOP 模式 B：跨实例共享 (Shared Cache)
   场景：工具类方法，结果只与参数有关，与 self 无关。
   策略：mode='ignore, str' (忽略 self，避免重复计算)。

   >>> class Tools:
   ...     @uni_cache(mode='ignore, str')
   ...     def heavy_calc(self, x): ...

4. 大对象性能优化 (Reference Mode)
   场景：参数包含大图片或大文本，全量 Hash 极慢。
   策略：指定参数名=id，仅对比内存地址。

   >>> @uni_cache(mode='ignore, img=id, str')
   ... def process(ctx, img, option='fast'): ...

5. 生命周期管理
   自动挂载 .cache (可配置 namespace) 控制器。

   >>> func.cache.refresh(x)  # 强制执行并更新缓存（无论之前有没执行过都强制执行，然后存到缓存）
   >>> func.cache.new(x)      # 瞬态执行(新建一个实例，但这个不在此缓存系统中管理，是一个全新的独立对象)
   >>> func.cache.clear()     # 清空缓存
   >>> func.cache.info()      # 查看容量和命中率

参数说明：
- mode: 缓存策略字符串 (如 'ignore, c=id, str')
    * str: (默认) 值匹配，容器自动转 tuple
    * id: 引用匹配，用于大对象/实例隔离
    * ignore: 忽略该参数
- maxsize: 最大容量 (默认 128)，防内存泄漏
- ttl: 有效期 (秒)
- namespace: 管理接口名称 (默认 'cache')
"""

import functools
import inspect
import threading

from pyxllib.prog.lazyimport import lazy_import

try:
    from cachebox import LRUCache, TTLCache
except ModuleNotFoundError:
    LRUCache, TTLCache = lazy_import("from cachetools import LRUCache, TTLCache")


def __1_缓存策略解析():
    """ 负责将函数参数解析为可哈希的键 """


def _to_tuple(obj):
    """ 递归将 list/dict/set 转换为 tuple，实现可哈希转换

    :param obj: 输入对象
    :return: 转换后的可哈希对象

    >>> _to_tuple([1, 2, [3, 4]])
    (1, 2, (3, 4))
    >>> _to_tuple({'a': 1, 'b': [2, 3]})
    (('a', 1), ('b', (2, 3)))
    """
    if isinstance(obj, (list, tuple)):
        return tuple(_to_tuple(x) for x in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, _to_tuple(v)) for k, v in obj.items()))
    if isinstance(obj, set):
        return tuple(sorted(_to_tuple(x) for x in obj))
    return obj


def process_val(val, rule='str'):
    """ 根据规则处理单个参数值，返回可哈希的对象

    :param val: 参数值
    :param str rule: 处理规则
        - 'str': 默认规则，尝试转换为 tuple 或字符串。
        - 'id': 返回对象的 id(val)，适用于不可序列化的大对象。
        - 'ignore': 忽略该参数，返回 None。
    :return: 处理后的可哈希对象

    >>> process_val([1, 2], 'str')
    (<class 'list'>, (1, 2))
    >>> process_val((1, 2), 'str')
    (1, 2)
    >>> d = {'a': 1}; process_val(d, 'id') == id(d)
    True
    >>> process_val(123, 'ignore')
    """
    if rule == 'ignore':
        return None
    if rule == 'id':
        return id(val)

    if rule == 'str':
        # 1. 基础不可变类型 (包含 tuple)：直接返回
        # Python 的 hash((1,2)) 和 hash([1,2]) 本身机制不同，
        # 但为了防止 list 转 tuple 后撞车，我们在下面处理 list/dict 时加了 type 标记。
        # 所以这里的 tuple 可以放心直通。
        if isinstance(val, (int, float, bool, str, bytes, type(None), tuple)):
            return val

        # 2. 可变容器：转 Tuple 并附加类型指纹
        if isinstance(val, (list, dict, set)):
            # 结果形如: (<class 'list'>, (1, 2))
            return (type(val), _to_tuple(val))

        # 3. 其他自定义对象
        return str(val)

    return val


class CachePolicyParser:
    """ 解析 mode 字符串并提供规则查询服务 """

    def __init__(self, mode_str='str'):
        """
        :param str mode_str: 规则字符串，如 'ignore, heavy=id, str'
        """
        self.raw_mode = mode_str
        self._parse(mode_str)

    def _parse(self, mode_str):
        raw_items = [x.strip() for x in mode_str.split(',') if x.strip()]
        self.pos_rules = []
        self.named_rules = {}

        for item in raw_items:
            if '=' in item:
                key, val = item.split('=', 1)
                self.named_rules[key.strip()] = val.strip()
            else:
                self.pos_rules.append(item)

        if not self.pos_rules:
            self.pos_rules = ['str']

        self.default_rule = self.pos_rules[-1]
        self.specific_pos_len = len(self.pos_rules) - 1

    def get_rule(self, index, name=None):
        """ 获取指定位置或名称的规则

        :param int index: 参数索引
        :param str name: 参数名称
        :return str: 匹配到的规则名称
        """
        if name and name in self.named_rules:
            return self.named_rules[name]
        if index < self.specific_pos_len:
            return self.pos_rules[index]
        return self.default_rule


def __2_基础工具与控制器():
    """ 存放缓存管理相关的辅助类 """


class CacheControl:
    """ 缓存管理控制器，提供清理、统计、强制执行等接口 """

    def __init__(self, wrapper):
        """
        :param UniCacheWrapper wrapper: 包装器实例
        """
        self._wrapper = wrapper

    def clear(self):
        """ 清空所有缓存内容 """
        with self._wrapper._lock:
            self._wrapper._cache.clear()

    def info(self):
        """ 查看缓存统计信息

        :return dict: 统计信息字典，包含 target, mode, backend, count, usage, ttl 等字段
        """
        curr_size = len(self._wrapper._cache)
        max_size = getattr(self._wrapper._cache, 'maxsize', 0)
        display_max = max_size if max_size > 0 else 'Infinite'
        return {
            'target': self._wrapper._name,
            'mode': self._wrapper._mode_str,
            'backend': type(self._wrapper._cache).__name__,
            'count': curr_size,
            'usage': f'{curr_size}/{display_max}',
            'ttl': getattr(self._wrapper._cache, 'ttl', None)
        }

    def __call__(self, *args, **kwargs):
        """ 调用 info() 的快捷方式 """
        return self.info()

    def new(self, *args, **kwargs):
        """ 瞬态模式：强制执行目标函数，但不更新或读取缓存 """
        return self._wrapper._target(*args, **kwargs)

    def refresh(self, *args, **kwargs):
        """ 覆盖模式：强制执行目标函数，并更新缓存结果 """
        key = self._wrapper._generate_key(args, kwargs)
        res = self._wrapper._target(*args, **kwargs)
        with self._wrapper._lock:
            self._wrapper._cache[key] = res
        return res


def __3_核心包装器():
    """ 缓存装饰器的核心逻辑实现 """


class UniCacheWrapper:
    """ 统一缓存包装器，支持多种缓存策略、TTL、最大容量 """

    def __init__(self, target, mode, namespace, maxsize, ttl):
        """
        :param callable target: 目标函数或类
        :param str mode: 缓存策略字符串
        :param str namespace: 管理接口挂载的属性名
        :param int maxsize: 最大缓存容量
        :param float|int ttl: 缓存生存时间（秒）
        """
        self._target = target
        self._name = getattr(target, '__name__', str(target))
        self._mode_str = mode
        self._namespace = namespace
        self._lock = threading.RLock()

        self._init_backend(maxsize, ttl)
        self._policy = CachePolicyParser(mode)

        try:
            self._sig = inspect.signature(target)
        except (ValueError, TypeError):
            self._sig = None

        control = CacheControl(self)
        if namespace:
            setattr(self, namespace, control)
        else:
            # 如果没有 namespace，则直接挂载到 self 上
            self.clear = control.clear
            self.info = control.info
            self.new = control.new
            self.refresh = control.refresh

        functools.update_wrapper(self, target)

    def __repr__(self):
        return f'<UniCacheWrapper {self._name} mode={self._mode_str}>'

    def __get__(self, instance, owner):
        """ 支持描述符协议，处理实例方法绑定 """
        if instance is None:
            return self

        def bound_call(*args, **kwargs):
            return self(instance, *args, **kwargs)

        functools.update_wrapper(bound_call, self._target)
        if self._namespace:
            setattr(bound_call, self._namespace, getattr(self, self._namespace))
        else:
            bound_call.clear = self.clear
            bound_call.info = self.info
            bound_call.new = self.new
            bound_call.refresh = self.refresh

        return bound_call

    def _init_backend(self, maxsize, ttl):
        """ 根据配置选择最佳的存储容器 """
        if ttl and ttl > 0:
            self._cache = TTLCache(maxsize=maxsize, ttl=ttl)
        else:
            self._cache = LRUCache(maxsize=maxsize)

    def _generate_key(self, args, kwargs):
        """ 生成缓存键 """
        if self._sig:
            try:
                bound = self._sig.bind(*args, **kwargs)
                bound.apply_defaults()
                arguments = bound.arguments.items()
            except TypeError:
                # 参数不匹配时回退到原始参数处理
                arguments = self._fallback_arguments(args, kwargs)
        else:
            arguments = self._fallback_arguments(args, kwargs)

        key_parts = []
        for i, (name, val) in enumerate(arguments):
            rule = self._policy.get_rule(index=i, name=name)
            key_parts.append(process_val(val, rule))
        return hash(tuple(key_parts))

    def _fallback_arguments(self, args, kwargs):
        """ 回退的参数处理逻辑 """
        arguments = []
        for i, arg in enumerate(args):
            arguments.append((f'arg_{i}', arg))
        for k, v in kwargs.items():
            arguments.append((k, v))
        return arguments

    def __call__(self, *args, **kwargs):
        """ 执行缓存调用逻辑 """
        key = self._generate_key(args, kwargs)

        with self._lock:
            if key in self._cache:
                return self._cache[key]

            res = self._target(*args, **kwargs)
            self._cache[key] = res
            return res

    def __getattr__(self, item):
        return getattr(self._target, item)


def __4_装饰器入口():
    """ 暴露给外部使用的装饰器函数 """


def uni_cache(_func=None, *, mode='str', maxsize=128, ttl=None, namespace='cache'):
    """ 通用大一统缓存装饰器

    :param callable _func: 被装饰的函数
    :param str mode: 缓存策略，控制每个参数如何参与缓存键的生成。
        - 支持按位置或名称指定规则，如 'ignore, heavy=id, str'。
        - 规则选项：'str' (默认，值匹配), 'id' (引用匹配), 'ignore' (忽略)。
    :param int maxsize: 最大缓存条目数，默认 128。
    :param float|int ttl: 缓存有效期（秒），默认 None 表示不限时。
    :param str namespace: 管理接口挂载的属性名，默认 'cache'。若设为 None，则直接挂载 clear/info 等方法。
    :return: 装饰后的函数包装器

    >>> @uni_cache(mode='str')
    ... def add(a, b): return a + b
    >>> add(1, 2)
    3
    >>> add(1, b=2)  # 参数对齐，命中缓存
    3
    >>> add.cache.info()['count']
    1
    >>> @uni_cache(ttl=1)
    ... def get_time(): return time.time()
    >>> t1 = get_time(); time.sleep(0.1); t2 = get_time()
    >>> t1 == t2
    True
    """

    def decorator(target):
        return UniCacheWrapper(target, mode, namespace, maxsize, ttl)

    return decorator(_func) if _func else decorator
