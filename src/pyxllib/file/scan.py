import os
import re
from typing import List, Tuple, Callable

from pyxllib.text.pstr import PStr

Predicate = Callable[[os.DirEntry], bool]


class FilterFactory:
    """
    用于生成各种常用的 Predicate 函数

    虽然这里的filter一般是用于生成bool判断进行过滤
    但其实也可以作为一个回调、捕捉器使用，捕捉特定目标的entry，输出日志进行检查等
    """

    @classmethod
    def is_file(cls):
        """是否为文件"""
        return lambda e: e.is_file()

    @classmethod
    def is_dir(cls):
        """是否为目录"""
        return lambda e: e.is_dir()

    @classmethod
    def match_name(cls, patterns, ignore_case=False):
        """
        根据文件名进行匹配

        :param patterns: 匹配模式，可以是字符串、PStr 对象或它们的列表/数组
        :param ignore_case: 是否忽略大小写，默认为 False
        """
        if isinstance(patterns, (str, PStr)):
            patterns = [patterns]

        processed_patterns = [PStr.auto(p, ignore_case=ignore_case) for p in patterns]
        return lambda e: any(p.match(e.name) for p in processed_patterns)

    @classmethod
    def match_path(cls, patterns, ignore_case=False):
        """
        根据文件路径进行匹配

        :param patterns: 匹配模式，可以是字符串、PStr 对象或它们的列表/数组
        :param ignore_case: 是否忽略大小写，默认为 False
        """
        if isinstance(patterns, (str, PStr)):
            patterns = [patterns]

        processed_patterns = [PStr.auto(p, ignore_case=ignore_case) for p in patterns]
        return lambda e: any(p.match(e.path) for p in processed_patterns)


class PathSelector:
    """
    PathSelector Core
    基于 ACL (Allow/Deny) 模型的规则引擎
    """

    def __init__(self):
        # 语义清晰：这是一组 ACL 规则
        self.enter_rules: List[Tuple[Predicate, bool]] = []
        self.yield_rules: List[Tuple[Predicate, bool]] = []

    # =========================================================
    # 1. 导航权限控制 (Navigation ACL)
    # =========================================================

    def allow_enter(self, predicate: Predicate) -> "PathSelector":
        """【Allow】添加规则：允许/强制进入 (Target: True)"""
        self.enter_rules.append((predicate, True))
        return self

    def deny_enter(self, predicate: Predicate) -> "PathSelector":
        """【Deny】添加规则：拒绝/禁止进入 (Target: False)"""
        self.enter_rules.append((predicate, False))
        return self

    # =========================================================
    # 2. 产出权限控制 (Selection ACL)
    # =========================================================

    def allow_yield(self, predicate: Predicate) -> "PathSelector":
        """【Allow】添加规则：允许/强制产出 (Target: True)"""
        self.yield_rules.append((predicate, True))
        return self

    def deny_yield(self, predicate: Predicate) -> "PathSelector":
        """【Deny】添加规则：拒绝/禁止产出 (Target: False)"""
        self.yield_rules.append((predicate, False))
        return self

    # =========================================================
    # 3. 执行层 (Execution)
    # =========================================================

    def is_enter(self, entry: os.DirEntry) -> bool:
        # 默认策略：Allow All
        return self._check(entry, self.enter_rules, default=True)

    def is_yield(self, entry: os.DirEntry) -> bool:
        # 默认策略：Allow All
        return self._check(entry, self.yield_rules, default=True)

    def _check(
        self, entry: os.DirEntry, rules: List[Tuple[Predicate, bool]], default: bool
    ) -> bool:
        decision = default
        for predicate, target_action in rules:
            if decision == target_action:
                continue
            if predicate(entry):
                decision = target_action
        return decision
