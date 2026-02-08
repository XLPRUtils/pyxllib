"""ctor_proxy.py - 构造函数代理

托管类对象的构造过程，支持命名单例、参数指纹复用及混合索引模式。
核心功能是提供一个中间层，允许在不同地点配置初始化参数，并基于名称或参数指纹来复用实例。

主要功能：
1. 命名单例：通过 name 字符串来全局共享一个实例。
2. 参数指纹：通过初始化参数的 Hash 值来共享实例（类似 functools.lru_cache）。
3. 混合模式：同时支持名称和参数指纹，灵活管理实例生命周期。

>>> class User:
...     def __init__(self, name, age):
...         self.name = name
...         self.age = age
...     def __repr__(self):
...         return f'User({self.name}, {self.age})'

>>> # 1. 简单命名单例
>>> p1 = ConstructorProxy(User, 'admin').config('Alice', 30)
>>> u1 = p1.get()
>>> u1
User(Alice, 30)
>>> # 在其他地方获取同一个名为 'admin' 的实例，无需再次配置参数
>>> u2 = ConstructorProxy(User, 'admin').get()
>>> u1 is u2
True

>>> # 2. 基于指纹的复用 (name=None)
>>> p3 = ConstructorProxy(User).config('Bob', 20)
>>> u3 = p3.get()
>>> p4 = ConstructorProxy(User).config('Bob', 20)
>>> u4 = p4.get()
>>> u3 is u4  # 参数相同，实例复用
True

>>> # 3. 参数变化会自动创建新实例
>>> p5 = ConstructorProxy(User).config('Bob', 21)
>>> u5 = p5.get()
>>> u3 is u5
False
"""

import inspect
import threading
import time
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union  # noqa: UP035

T = TypeVar('T')


class InstanceContext:
    """实例上下文

    用于存储实例对象及其创建时的配置元数据。

    :param instance: 实际创建的对象实例
    :param tuple args: 创建时的位置参数
    :param dict kwargs: 创建时的关键字参数
    """

    def __init__(self, instance, args, kwargs):
        self.instance = instance
        self.args = args
        self.kwargs = kwargs
        self.created_at = time.time()


class ConstructorProxy:
    """构造函数代理

    托管类对象的构造过程，支持命名单例、参数指纹复用及混合索引模式。

    :param Type[T] target_class: 需要被托管的目标类
    :param str|list|None name: 索引键
        - str: 指定一个全局唯一的名称，用于查找单例。
        - None: 表示匿名，将使用参数指纹(fingerprint)作为索引键。
        - list: 可以包含多个名称或 None，同时支持多种查找方式。
    """

    # 全局注册表：Key -> InstanceContext
    # Key 可以是字符串(name) 或 整数(fingerprint hash)
    _registry: Dict[Union[str, int], InstanceContext] = {}
    _lock = threading.RLock()

    def __init__(self, target_class: Type[T], name: Union[str, List[Optional[str]], None] = None):
        self.target_class = target_class

        # 归一化 name 为列表形式
        if name is None:
            self._names = [None]
        elif isinstance(name, list):
            self._names = name
        else:
            self._names = [name]

        # 暂存初始化参数
        self._args: Tuple = ()
        self._kwargs: Dict = {}

    def config(self, *args, **kwargs) -> 'ConstructorProxy[T]':
        """配置初始化参数（支持链式调用）

        此处不执行实例化，仅存储参数，供后续 get() 使用。

        :return ConstructorProxy[T]: 返回自身以支持链式调用
        """
        self._args = args
        self._kwargs = kwargs
        return self

    def get(self) -> T:
        """获取或创建实例 (Get-or-Create)

        :return T: 目标类的实例

        逻辑流程：
        1. 根据 name 配置计算所有的查找键 (Keys)。
        2. 在全局注册表中查找这些 Keys。
        3. 如果找到已存在的实例：
           - 检查是否冲突（不同 Key 指向不同实例）。
           - (可选) 检查参数一致性。
           - 将本次所有未注册的 Key 关联到该实例。
           - 返回旧实例。
        4. 如果未找到：
           - 使用 config 的参数创建新实例。
           - 注册所有 Key。
           - 返回新实例。
        """
        with self._lock:
            # 1. 计算所有实际的查找键 (Keys)
            keys = self._resolve_keys()

            # 2. 在注册表中查找这些 Key
            found_ctx: Optional[InstanceContext] = None
            found_by_key = None

            for key in keys:
                if key in self._registry:
                    ctx = self._registry[key]

                    # 冲突检测：如果不同的 Key 指向了不同的实例
                    if found_ctx is not None and found_ctx is not ctx:
                        raise ValueError(
                            f'实例引用歧义 (Ambiguous Instance): '
                            f"键 '{found_by_key}' 指向实例 A，但键 '{key}' 指向实例 B。"
                            '请检查是否尝试将两个已存在的不同实例强制合并为一个别名组。'
                        )

                    found_ctx = ctx
                    found_by_key = key

            # 3. 命中逻辑
            if found_ctx:
                # 一致性检查（可选但推荐）：
                # 如果是通过名字找到的，但当前 config 生成的指纹与实例原本的参数不一致，
                # 说明用户想用新的参数获取一个旧的单例，这通常是逻辑错误。
                # 只有当 name 列表中包含 None (即意图使用指纹) 时，我们才严格校验配置的一致性，
                # 否则纯命名模式下，通常假设用户只想拿旧对象，不关心参数。
                if None in self._names:
                    # 简单的参数全等比对
                    if not self._check_config_consistency(found_ctx):
                        raise ValueError(
                            f"配置冲突: 找到名为 '{found_by_key}' 的实例，"
                            '但其创建参数与当前传入的 .config() 参数不一致。'
                            '若需覆盖旧实例，请使用 .recreate()。'
                        )

                # 自动关联：将本次所有尚未注册的 Key 指向这个已存在的实例
                for key in keys:
                    if key not in self._registry:
                        self._registry[key] = found_ctx

                return found_ctx.instance

            # 4. 未命中逻辑：创建新实例
            instance = self.target_class(*self._args, **self._kwargs)
            new_ctx = InstanceContext(instance, self._args, self._kwargs)

            # 注册所有 Key
            for key in keys:
                self._registry[key] = new_ctx

            return instance

    def recreate(self) -> T:
        """强制重新实例化

        销毁旧引用，创建新对象。
        注意：这只会重置当前 proxy 指定的 name/fingerprint 对应的引用。
        如果旧实例还有其他别名引用，那些别名不会受影响（指向旧实例）。

        :return T: 新创建的实例
        """
        with self._lock:
            keys = self._resolve_keys()

            # 1. 移除旧引用
            for key in keys:
                if key in self._registry:
                    del self._registry[key]

            # 2. 创建新实例并注册
            instance = self.target_class(*self._args, **self._kwargs)
            new_ctx = InstanceContext(instance, self._args, self._kwargs)

            for key in keys:
                self._registry[key] = new_ctx

            return instance

    def clear(self) -> None:
        """清除当前定义的所有索引引用

        从注册表中移除当前 proxy 关注的所有 Key。
        """
        with self._lock:
            keys = self._resolve_keys()
            for key in keys:
                if key in self._registry:
                    del self._registry[key]

    # --- 内部辅助方法 ---

    def _resolve_keys(self) -> List[Union[str, int]]:
        """将 self._names 中的 None 替换为实际计算出的指纹 Hash"""
        resolved_keys = []
        fingerprint = None

        for name in self._names:
            if name is None:
                # 惰性计算指纹，只计算一次
                if fingerprint is None:
                    fingerprint = self._calculate_fingerprint()
                resolved_keys.append(fingerprint)
            else:
                resolved_keys.append(name)
        return resolved_keys

    def _calculate_fingerprint(self) -> int:
        """计算 (TargetClass, args, kwargs) 的唯一指纹

        尝试使用 inspect.signature 绑定参数以忽略 (a=1, b=2) vs (b=2, a=1) 的顺序差异。
        """
        # 1. 尝试标准化参数
        bound_args = None
        try:
            sig = inspect.signature(self.target_class)
            # bind 可能会失败，比如 C 扩展类没有签名，或者参数不匹配
            bound = sig.bind(*self._args, **self._kwargs)
            bound.apply_defaults()
            # 使用绑定后的参数字典作为指纹依据
            args_for_hash = ()
            kwargs_for_hash = bound.arguments
        except (ValueError, TypeError):
            # 降级方案：直接使用原始参数
            args_for_hash = self._args
            kwargs_for_hash = self._kwargs

        # 2. 生成 Hash 组件
        # 包含类名，防止不同类但参数相同的情况冲突
        class_key = self.target_class
        args_key = self._make_hashable(args_for_hash)
        # 将 kwargs 转为排序后的 tuple item 列表
        kwargs_key = self._make_hashable(tuple(sorted(kwargs_for_hash.items())))

        return hash((class_key, args_key, kwargs_key))

    def _make_hashable(self, value: Any) -> Any:
        """递归将 list, dict 等不可哈希类型转换为 tuple, frozenset"""
        if isinstance(value, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in value.items()))
        elif isinstance(value, (list, set)):
            return tuple(self._make_hashable(v) for v in value)
        elif isinstance(value, tuple):
            return tuple(self._make_hashable(v) for v in value)
        else:
            return value

    def _check_config_consistency(self, ctx: InstanceContext) -> bool:
        """检查当前配置与上下文中的配置是否一致"""
        # 简化版：直接全等比较。注意：如果是 list/dict，== 对比是内容对比，符合预期。
        return (ctx.args == self._args) and (ctx.kwargs == self._kwargs)
