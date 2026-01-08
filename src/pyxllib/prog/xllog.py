#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com

from loguru import logger as _global_logger
from fastcore.basics import GetAttr


class XlLogger(GetAttr):
    _default = '_internal'  # 找不到的方法找 _internal

    def __init__(self, name=None, min_level="INFO", _internal=None):
        """
        :param _internal: 私有参数，用于在 bind 等操作时克隆实例
        """
        # 1. 逻辑分叉：是全新初始化，还是克隆旧的？
        if _internal:
            self._internal = _internal
        else:
            self._internal = _global_logger.bind(name=name) if name else _global_logger

        # 2. 存储配置
        self._min_level_no = self._get_level_no(min_level)
        # 存下 min_level 的字符串形式，方便克隆时使用
        self._min_level_str = min_level

    def _get_level_no(self, level_name):
        try:
            return _global_logger.level(level_name.upper()).no
        except ValueError:
            return 0

    def set_level(self, level_name):
        """动态调整当前文件的日志门槛"""
        self._min_level_no = self._get_level_no(level_name)

    # =========================================================
    # 核心修复：拦截 bind，防止逃逸
    # =========================================================
    def bind(self, *args, **kwargs):
        # 1. 调用内部原本的 bind，得到一个新的原生 logger
        new_inner_logger = self._internal.bind(*args, **kwargs)

        # 2. 【关键】不要直接返回 new_inner_logger，而是用它再造一个 XlLogger
        # 这样返回给用户的依然是你的包装壳，大坝依然存在！
        return XlLogger(_internal=new_inner_logger, min_level=self._min_level_str)

    # 同理，如果通过 opt 改变了配置，也要重新包装
    def opt(self, *args, **kwargs):
        new_inner_logger = self._internal.opt(*args, **kwargs)
        return XlLogger(_internal=new_inner_logger, min_level=self._min_level_str)

    # =========================================================
    # 核心大坝 (保持不变)
    # =========================================================
    def _log(self, level, msg, *args, **kwargs):
        if self._get_level_no(level) >= self._min_level_no:
            # depth=2 很重要
            self._internal.opt(depth=2).log(level, msg, *args, **kwargs)

    # 显式定义常用方法 (IDE提示 + 路由到 _log)
    def trace(self, message, *args, **kwargs):
        self._log("TRACE", message, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._log("DEBUG", msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._log("INFO", msg, *args, **kwargs)

    def success(self, msg, *args, **kwargs):
        self._log("SUCCESS", msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._log("WARNING", msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._log("ERROR", msg, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self._log("CRITICAL", message, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        if self._get_level_no("ERROR") >= self._min_level_no:
            self._internal.opt(depth=1).exception(msg, *args, **kwargs)

    # ... 其他自定义方法 (log_json 等) ...
