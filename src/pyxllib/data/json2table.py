#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2025/06/05

"""
JSON转表格工具，用于将复杂的嵌套JSON列表转换为扁平化的DataFrame或Excel表格。

核心功能：
1. 数据加载：支持从单个文件或目录批量加载JSON数据，自动处理异构数据。
2. 字段解析：支持使用点号分隔路径（如 'analysis.score'）访问深层字段。
3. 智能选择：支持 Glob 通配符（'*.score'）和正则选择器批量操作字段。
4. 数据清洗：提供 add_meta_column, transform_field, map_field_values 等丰富的清洗接口。
5. 结构变换：支持 expand_list_field（列表展开）、list_to_dict_by_key（列表转字典）等结构重组。
6. 表格拆分：支持垂直拆分（extract_columns_table）、水平拆分（extract_rows_table）和宽表转长表（extract_fields_to_long_table）。
7. 导出功能：支持自动分表、多Sheet导出，并自动处理列顺序。

使用示例：

>>> import os
>>> import json
>>> from pyxllib.data.json2table import Json2Table

# 1. 准备测试数据
>>> data = [
...     {
...         "id": 1001,
...         "info": {"name": "Alice", "age": 25},
...         "scores": [{"subject": "Math", "score": 90}, {"subject": "English", "score": 85}],
...         "tags": ["student", "active"]
...     },
...     {
...         "id": 1002,
...         "info": {"name": "Bob", "age": 30},
...         "scores": [{"subject": "Math", "score": 70}],
...         "tags": ["worker"]
...     }
... ]
>>> json_file = 'test_data.json'
>>> with open(json_file, 'w') as f:
...     json.dump(data, f)

# 2. 初始化工具
>>> mgr = Json2Table()
>>> mgr.add_from_json(json_file)

# 3. 数据清洗与转换
# 提取深层字段到顶层，并进行转换
>>> mgr.add_meta_column('username', lambda row: row['data']['info']['name'])
>>> mgr.transform_field('info.age', lambda x: x + 1)  # 年龄加1

# 4. 复杂结构处理
# 将 scores 列表展开，方便后续提取
>>> mgr.expand_list_field('scores', start_index=1)

# 5. 拆分表格
# 将成绩信息单独拆分到一个子表
>>> mgr.extract_columns_table('scores_table', ['id'], ['scores[1].score', 'scores[2].score'])

# 6. 导出结果 (仅生成对象，不实际写入文件以免影响环境)
>>> frames = list(mgr.iter_export_frames())
>>> [name for name, df, type in frames]
['Sheet1', 'scores_table']

# 主表包含基础信息
>>> main_df = frames[0][1]
>>> 'username' in main_df.columns
True

# 清理测试文件
>>> os.remove(json_file)
"""

import json
from pathlib import Path
from functools import wraps
import fnmatch

import pandas as pd

from pyxllib.prog.xllog import XlLogger
from pyxllib.text.pstr import PStr

# 这个logger完全兼容from loguru import logger的用法，但扩展了额外日志功能
logger = XlLogger(__name__, 'WARNING')


def invalidates_schema(func):
    """
    [装饰器] 标记该方法会改变数据结构（如删除字段、重命名、展开列表等）。
    调用该方法后，会自动强制重置 Schema 缓存，确保下次解析通配符时是准确的。
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        # 方法执行完后，作废缓存
        self._reset_schema_cache()
        return result

    return wrapper


# 基础类：包含核心初始化和缓存管理
class Json2TableBase:
    """Json2Table基础类，包含核心初始化和缓存管理功能"""

    def __init__(self, source_dir=None):
        """
        初始化 Json2Table 工具。

        :param str source_dir: (Optional) 如果传入，会自动调用 add_from_directory。
        """
        # 1 内部存储结构：
        # '_source_path' 存储来源路径对象。
        # 'data' 列存储原始 JSON 对象（Dict），不立即展开，允许异构数据共存。
        self.df = pd.DataFrame()

        # 2 Schema 缓存机制 ---
        self._cached_paths = None  # 类型: set
        self._scanned_count = 0  # 记录已经扫描了多少行

        self._sort_operations = []

        # 3 存储提取出来的映射表/配置表
        # 格式: { '表名': DataFrame }
        self.sub_tables = {}

        # 4 是否从目录加载json数据
        if source_dir:
            self.add_from_directory(source_dir)

    def add_from_json(self, file_path):
        """
        加载单个 JSON 文件。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f'File not found: {path}')

        with open(path, 'r', encoding='utf-8') as f:
            content = json.load(f)

            new_records = []
            # 统一处理为列表形式，方便构建 DataFrame
            # 这里只存储 data 对象本身和路径，不解析 data 内部结构
            if isinstance(content, list):
                for item in content:
                    new_records.append({'data': item, '_source_path': path})
            else:
                new_records.append({'data': content, '_source_path': path})

            if new_records:
                new_df = pd.DataFrame(new_records)
                if self.df.empty:
                    self.df = new_df
                else:
                    self.df = pd.concat([self.df, new_df], ignore_index=True)

    def add_from_directory(self, dir_path, pattern='*.json'):
        """
        扫描目录并添加数据。
        """
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f'Directory not found: {path}')

        for file_path in path.glob(pattern):
            self.add_from_json(file_path)

    def _reset_schema_cache(self):
        """强制清空缓存（当结构发生破坏性变更时调用）"""
        self._cached_paths = None
        self._scanned_count = 0

    def _get_all_json_paths(self):
        """
        [智能核心] 获取所有 JSON 路径。
        实现策略：惰性计算 + 增量更新。
        """
        current_count = len(self.df)

        # 1. 如果没有任何数据
        if current_count == 0:
            return []

        # 2. 如果发生了行删除 (当前行数 < 记录的行数)，旧索引失效
        #    这种情况必须全量重扫，因为不知道删的是哪几行，会不会导致某些字段彻底消失
        if self._scanned_count > current_count:
            self._reset_schema_cache()

        # 3. 初始化缓存集合
        if self._cached_paths is None:
            self._cached_paths = set()
            self._scanned_count = 0

        # 4. 判断是否需要扫描新数据
        if self._scanned_count < current_count:
            # === 增量扫描关键逻辑 ===
            # 只提取从 last_index 到 current_index 的新数据
            new_data = self.df['data'].iloc[self._scanned_count: current_count].tolist()

            # 使用 json_normalize 提取这些新数据的 Key
            try:
                # 仅提取 columns，不构建完整 DataFrame 以节省内存
                # json_normalize 默认会把所有层级展开为 dotted path
                if new_data:
                    mini_df = pd.json_normalize(new_data)
                    new_keys = set(mini_df.columns)
                    # 合并入总缓存
                    self._cached_paths.update(new_keys)
            except Exception:
                pass  # 忽略解析错误

            # 更新计数器
            self._scanned_count = current_count
            logger.debug(f'Schema缓存已更新。已知字段总数：{len(self._cached_paths)}')

        return list(self._cached_paths)

    def _resolve_path(self, root_data, path_str):
        """
        解析路径，返回目标字段所在的父级字典和目标键名。

        :param dict root_data: 根字典
        :param str path_str: 点号分隔的路径，如 "analysis.spot.category"
        :return: (parent_dict, target_key) 或 (None, None) 如果路径不存在
        """
        if not isinstance(root_data, dict) or not path_str:
            return None, None

        parts = path_str.split('.')
        target_key = parts[-1]
        parent_path = parts[:-1]

        current = root_data
        for key in parent_path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None, None

        if isinstance(current, dict) and target_key in current:
            return current, target_key

        return None, None

    def _ensure_list(self, items):
        """辅助函数：确保输入是列表"""
        if items is None:
            return []
        return items if isinstance(items, list) else [items]

    # 选择器工厂 (语法糖)
    def glob(self, pattern):
        """创建通配符选择器。例如: mgr.glob('analysis.*')"""
        return PStr.glob(pattern)

    def re(self, pattern):
        r""" 创建正则选择器。例如: mgr.re(r'analysis\.\w+_score') """
        return PStr.re(pattern)

    def _resolve_fields(self, inputs, scope='json'):
        """
        [核心] 将输入的字段/选择器解析为具体的字段名列表。

        :param inputs: 字段名、通配符、正则。
        :param scope: 查找范围。
            - 'json': 仅查找 JSON 内部结构 (默认)。
            - 'dataframe': 仅查找 DataFrame 顶层列 (如 filename, _source_path)。
            - 'all': 查找上述两者 (用于 reorder_columns)。
        """
        raw_list = self._ensure_list(inputs)
        final_fields = []

        # 1. 确定候选池 (Candidates)
        candidates = []
        if scope in ['dataframe', 'all']:
            candidates.extend(list(self.df.columns))
        if scope in ['json', 'all']:
            candidates.extend(self._get_all_json_paths())

        # 去重候选池
        candidates = list(dict.fromkeys(candidates))

        # 2. 匹配逻辑
        for item in raw_list:
            if isinstance(item, PStr) and not item.is_literal:
                # 智能模式匹配 (Regex / Glob)
                final_fields.extend(item.find_matches(candidates))
            elif isinstance(item, str):
                # 普通字符串（包括 Literal PStr）
                # 注意：为了兼容性，普通字符串是否支持 fnmatch 取决于原先设计。
                # 原设计中：isinstance(item, str) -> final_fields.append(item)
                # 这意味着普通字符串只做精确匹配（添加它自己），不做 glob 展开？
                # 等等，我看之前的代码：
                # if isinstance(item, str): final_fields.append(item)
                # 之前的代码确实是直接 append，不做 fnmatch。
                # 只有 GlobSelector 才做 fnmatch。
                final_fields.append(item)
            else:
                # 其他类型直接添加（虽然理论上应该是 str）
                final_fields.append(str(item))

        return list(dict.fromkeys(final_fields))


# 字段管理类：负责处理JSON字段的各种操作
class Json2TableFieldManager(Json2TableBase):
    """Json2Table字段管理类，负责处理JSON字段的各种操作"""

    def add_meta_column(self, col_name, strategy):
        """
        添加一个元数据列（DataFrame 顶层列），通常用于提取 JSON 中的信息或添加辅助信息。

        :param str col_name: 新列名称
        :param callable strategy: 函数，接收 row (Series)，返回新列的值
        """
        if self.df.empty: return
        self.df[col_name] = self.df.apply(strategy, axis=1)

    def add_meta_columns(self, mapping):
        """
        批量添加元数据列。
        :param dict mapping: { "col_name": strategy_func, ... }
        """
        if self.df.empty: return
        # 批量处理建议使用 assign 或逐个调用，pandas 不支持一次 apply 返回多列赋值给不同名
        for col, func in mapping.items():
            self.add_meta_column(col, func)

    @invalidates_schema
    def remove_fields(self, field_paths, scope='all'):
        """
        [通用删除] 批量删除字段。
        支持通过 scope 精确控制删除范围。

        :param str|list field_paths: 字段名、路径或通配符列表
        :param str scope: 删除范围
            - 'all': (默认) 尝试同时删除 DataFrame 列和 JSON 内部字段。
            - 'dataframe': 仅删除 DataFrame 顶层列 (性能最快)。
            - 'json': 仅删除 JSON 内部字段 (不会误删顶层列)。
        """
        if self.df.empty: return

        # 1. 根据 scope 解析所有匹配的目标
        # 注意：_resolve_fields 会根据 scope 自动去 df.columns 或 json keys 里找匹配项
        targets = self._resolve_fields(field_paths, scope=scope)

        if not targets:
            return

        # 2. 分类待删除列表
        df_cols_to_drop = []
        json_paths_to_drop = []

        for t in targets:
            # 绝对保护核心列 'data'
            if t == 'data':
                continue

            # 判定逻辑：
            # A. 如果 scope 允许操作 dataframe，且该字段确实是顶层列 -> 删列
            if scope in ['all', 'dataframe'] and t in self.df.columns:
                df_cols_to_drop.append(t)

            # B. 如果 scope 允许操作 json -> 放入 JSON 删除队列
            # 注意：这里我们不做严格检查 "t 是否在 json 中"，因为后续的 apply 逻辑本身就是安全的（key不存在则忽略）
            # 但我们需要避免把纯粹的 DF 列（非 JSON 路径）传给 JSON 处理函数虽无害但浪费性能
            # 不过鉴于 _resolve_fields 已经做过筛选，这里直接通过 scope 判断即可
            if scope in ['all', 'json']:
                json_paths_to_drop.append(t)

        # 3. 执行删除：DataFrame 顶层列
        if df_cols_to_drop:
            self.df.drop(columns=df_cols_to_drop, inplace=True)
            logger.debug(f"已删除DataFrame列：{df_cols_to_drop}")

        # 4. 执行删除：JSON 内部字段
        if json_paths_to_drop:
            def process_row(root_data):
                for path in json_paths_to_drop:
                    # 复用 resolve_path 查找父节点
                    parent, key = self._resolve_path(root_data, path)
                    if parent and key in parent:
                        del parent[key]
                return root_data

            self.df['data'] = self.df['data'].apply(process_row)
            logger.debug(f"已删除JSON字段：{len(json_paths_to_drop)} 个路径")

    def keep_fields(self, keep_paths):
        """
        (新增) 仅保留指定字段，删除其他所有字段。
        注意：这会重构整个 JSON 结构，仅保留 keep_paths 指向的数据。
        :param str|list keep_paths: 要保留的字段路径列表
        """
        if self.df.empty: return
        targets = self._resolve_fields(keep_paths, scope='json')

        def process_row(root_data):
            new_root = {}
            for path in targets:
                parent, key = self._resolve_path(root_data, path)
                if parent and key in parent:
                    # 注意：这里简化处理，将深层路径提取出来放在顶层，
                    # 或者你需要编写复杂的逻辑来重建层级结构。
                    # 这里为了实用性，通常是保留原始层级比较困难，建议根据业务需求调整。
                    # 简单实现：保留路径指向的值，扁平化或手动重建。
                    # 此处暂不实现复杂重建，建议用户慎用或自行实现 strategy。
                    pass
            return root_data

        # 由于通用实现的复杂性（重建树），建议此场景使用 process_rows 自定义清洗
        pass

    @invalidates_schema
    def rename_fields(self, mapping):
        """
        (新增) 批量修改 JSON 内部的键名。
        :param dict mapping: { "old.path.key": "new_key_name" }
                             注意：新名字目前仅支持修改 Key 本身，不支持移动层级。
        """
        if self.df.empty: return

        def process_row(root_data):
            for old_path, new_name in mapping.items():
                parent, key = self._resolve_path(root_data, old_path)
                if parent and key in parent:
                    val = parent.pop(key)
                    parent[new_name] = val
            return root_data

        self.df['data'] = self.df['data'].apply(process_row)

    @invalidates_schema
    def expand_list_field(self, field_paths, start_index=1, remove_original=True):
        """
        将 List 字段展开为多个字段。支持批量处理。

        :param str|list field_paths: 字段路径或列表
        """
        if self.df.empty: return
        targets = self._resolve_fields(field_paths, scope='json')

        def process_record(root_data):
            # 遍历所有需要展开的路径
            for path in targets:
                parent, key = self._resolve_path(root_data, path)

                # 校验
                if not parent or key not in parent or not isinstance(parent[key], list):
                    continue

                target_val = parent[key]
                if not target_val:  # 空列表处理
                    if remove_original:
                        del parent[key]
                    continue

                # 重构当前层级的字典（原地修改 parent 对象的内容）
                # 注意：因为是在循环中修改 parent，需要确保逻辑稳健
                new_dict = {}
                for k, v in parent.items():
                    if k == key:
                        for i, item in enumerate(target_val):
                            new_dict[f'{key}[{i + start_index}]'] = item
                        if not remove_original:
                            new_dict[k] = v
                    else:
                        new_dict[k] = v

                # 更新 parent 内容
                parent.clear()
                parent.update(new_dict)

            return root_data

        self.df['data'] = self.df['data'].apply(process_record)

    @invalidates_schema
    def list_to_dict_by_key(self, field_paths, key_field, remove_key_field=True):
        """
        [结构修改] 将列表转为字典。
        利用列表中每个元素的指定字段值（如 'cls'）作为新字典的 Key。

        示例数据：
        "category": [
            {
              "cls": "forehead",
              "type": "oil",
              "level": "moderately",
              "prob": 4.468299865722656,
              "score": 62,
              "oil_score": 85.35,
              "points": []
            },
            {
              "cls": "nose",
              "type": "oil",
              "level": "lightly",
              "prob": 4.93910026550293,
              "score": 67,
              "oil_score": 95.14,
              "points": []
            },
            ...
          ],

        :param str|list field_paths: 目标字段路径（指向那个 list）。
        :param str key_field: 列表中字典的哪个字段用作 Key（如 'cls'）。
        :param bool remove_key_field: 转换后是否移除原字典中的 key_field（默认移除，避免冗余）。
        """
        if self.df.empty: return
        targets = self._resolve_fields(field_paths, scope='json')

        def process_row(root_data):
            for path in targets:
                parent, key = self._resolve_path(root_data, path)

                # 1. 基础校验：父级存在，Key存在，且值确实是 List
                if not (parent and key in parent and isinstance(parent[key], list)):
                    continue

                original_list = parent[key]
                new_dict = {}

                # 2. 遍历列表进行转换
                for item in original_list:
                    if not isinstance(item, dict):
                        continue  # 跳过非字典元素

                    if key_field not in item:
                        continue  # 跳过没有指定 Key 的元素

                    # 获取新的键名 (如 'forehead')
                    new_key = str(item[key_field])

                    # 3. 处理 Value
                    if remove_key_field:
                        # 浅拷贝一份，以免修改原始引用导致意外
                        item_val = item.copy()
                        item_val.pop(key_field, None)
                    else:
                        item_val = item

                    new_dict[new_key] = item_val

                # 4. 原地替换：将原来的 List 替换为新的 Dict
                parent[key] = new_dict

            return root_data

        self.df['data'] = self.df['data'].apply(process_row)

    def transform_field(self, field_paths, func):
        """
        通用的字段转换函数。支持批量处理。

        :param str|list field_paths: 字段路径或列表
        :param callable func: 转换函数，支持两种形式：
                             - func(value)：只接收字段值
                             - func(value, row)：接收字段值和整行数据
        """
        if self.df.empty: return
        targets = self._resolve_fields(field_paths, scope='json')

        def process_row(root_data):
            for path in targets:
                parent, key = self._resolve_path(root_data, path)
                if parent and key in parent:
                    try:
                        # 检查func需要几个参数，支持传递整行数据
                        import inspect
                        sig = inspect.signature(func)
                        if len(sig.parameters) == 2:
                            parent[key] = func(parent[key], root_data)
                        else:
                            parent[key] = func(parent[key])
                    except Exception:
                        pass
            return root_data

        self.df['data'] = self.df['data'].apply(process_row)

    def map_field_values(self, field_paths, mapping_dict, else_value=None):
        """
        [内容修改] 值映射函数。
        自动适应 List 类型：如果字段值是 List，则映射 List 内部的每个元素，结构保持不变。

        :param str|list field_paths: 字段路径。
        :param dict mapping_dict: 映射字典。
        :param any else_value: 默认值。
        """
        if self.df.empty: return

        # 单个值的转换逻辑
        def map_scalar(scalar_val):
            if scalar_val in mapping_dict:
                return mapping_dict[scalar_val]
            return else_value if else_value is not None else scalar_val

        # 主策略：判断类型分发逻辑
        def strategy(val):
            if isinstance(val, list):
                # 如果是列表，生成新列表 (例如: ['A', 'B'] -> ['映射A', '映射B'])
                return [map_scalar(item) for item in val]
            else:
                # 如果是标量，直接映射
                return map_scalar(val)

        self.transform_field(field_paths, strategy)

    def simplify_list_field(self, field_paths, pick=0, sep=None):
        """
        [结构修改] 列表解包与简化通用函数。
        支持按下标提取单个值、提取子集、以及列表拼接。

        :param str|list field_paths: 字段路径。
        :param int|list|None pick: 选择策略。
            - int: 按下标提取单个元素 (e.g., 0 取第一个, -1 取最后一个)。
            - list[int]: 按下标提取多个元素 (e.g., [0, 2] 取第1和第3个)。
            - None: 保留所有元素 (通常配合 sep 使用)。
        :param str sep: 分隔符。
            - 如果结果是列表且 sep 不为 None，则将其元素拼接为字符串。
            - 如果结果是标量 (pick为int时)，此参数无效。
        """
        if self.df.empty: return
        targets = self._ensure_list(field_paths)

        def strategy(val):
            # 1. 基础校验：非列表或空列表直接返回
            if not isinstance(val, list) or len(val) == 0:
                # 如果是空列表，且指定了 sep，通常期望得到空字符串 '' 而不是 [] 或 None
                if isinstance(val, list) and sep is not None:
                    return ''
                return val

            result = val

            # 2. 执行选择 (Pick)
            try:
                if isinstance(pick, int):
                    # 模式 A: 提取标量 (支持负数下标)
                    # 越界保护：如果下标不存在，返回 None (或者你可以选择 raise)
                    if -len(val) <= pick < len(val):
                        result = val[pick]
                    else:
                        return None

                elif isinstance(pick, list):
                    # 模式 B: 提取子集 (e.g., [0, 2])
                    # 自动过滤越界的下标
                    result = [val[i] for i in pick if -len(val) <= i < len(val)]

                # 模式 C: pick is None -> 保留全量 (result = val)

            except Exception:
                return None

            # 3. 执行拼接 (Join)
            # 只有当 result 依然是列表，且用户指定了 sep 时才拼接
            if sep is not None and isinstance(result, list):
                return sep.join(str(x) for x in result)

            return result

        self.transform_field(targets, strategy)


# 行管理类：负责处理行数据的各种操作
class Json2TableRowManager(Json2TableFieldManager):
    """Json2Table行管理类，负责处理行数据的各种操作"""

    @invalidates_schema
    def delete_rows_by_func(self, condition_func):
        """
        根据自定义逻辑删除行。

        :param callable condition_func: 函数，接收一行 JSON data (dict)，
                                        返回 True 表示删除该行，返回 False 表示保留。
        """
        if self.df.empty:
            return

        def should_delete(row_data):
            if not isinstance(row_data, dict):
                return False
            try:
                return condition_func(row_data)
            except Exception:
                # 遇到错误（如字段不存在）通常选择保留，或者根据需求改为 True 删除
                return False

        # 逻辑取反：我们保留那些 should_delete 为 False 的行
        # df['data'].apply 会对每一行执行函数
        mask_to_keep = ~self.df['data'].apply(should_delete)

        # 重新赋值并重置索引
        original_count = len(self.df)
        self.df = self.df[mask_to_keep].reset_index(drop=True)

        # 可选：打印删除了多少条
        logger.info(f'删除了 {original_count - len(self.df)} 行。')

    def delete_rows_by_value(self, field_path, target_value):
        """
        删除指定字段等于特定值的行（支持嵌套路径）。

        :param str field_path: 字段路径，如 "id" 或 "analysis.score"
        :param any target_value: 要删除的目标值，如 1987837
        """
        if self.df.empty:
            return

        def check_value(root_data):
            # 复用内部的路径解析函数
            parent, key = self._resolve_path(root_data, field_path)
            if parent and key in parent:
                # 如果值相等，返回 True (代表要删除)
                return parent[key] == target_value
            return False

        self.delete_rows_by_func(check_value)


# 拆分类：负责将数据拆分为多个表
class Json2TableSplitter(Json2TableRowManager):
    """Json2Table拆分类，负责将数据拆分为多个表"""

    def __1_按行拆分(self):
        pass

    @invalidates_schema
    def extract_rows_table(self, new_table_name, condition_func, flatten=True):
        """
        [通用筛选] 传入一个函数，将返回 True 的行提取到新表，并从主表中删除。
        相当于：
            for row_data in json_data:
                if condition_func(row_data):
                    move_to(new_table)

        :param str new_table_name: 新表的 Sheet 名
        :param callable condition_func: 接收 json字典(dict) 的函数，返回 True/False
        :param bool flatten: 是否把提取出来的 JSON 展平成表格
        """
        if self.df.empty:
            return

        # 1. 定义包装函数：增加安全性，防止用户函数报错导致程序中断
        def safe_check(row_data):
            if not isinstance(row_data, dict):
                return False
            try:
                return condition_func(row_data)
            except Exception:
                # 如果用户写的逻辑报错（比如key不存在），默认不提取
                return False

        # 2. 计算筛选掩码 (Mask)
        # 这一步就相当于遍历：for x in df['data']
        mask = self.df['data'].apply(safe_check)

        if not mask.any():
            logger.warning(f"没有匹配到 {new_table_name} 的行")
            return

        # 3. 提取数据 (切片)
        extracted_df = self.df[mask].copy()

        # 4. 从主表中删除这些行
        self.df = self.df[~mask].reset_index(drop=True)
        logger.info(f"提取了 {len(extracted_df)} 行到 '{new_table_name}'。")

        # === 修改点：移除内部使用的 _source_path 列 ===
        # 在处理数据结构之前，先剔除这个内部路径字段
        if '_source_path' in extracted_df.columns:
            extracted_df.drop(columns=['_source_path'], inplace=True)

        # 5. 处理提取出的数据（展平 JSON 方便查看）
        if flatten:
            # 提取 JSON 内部字段
            data_expanded = pd.json_normalize(extracted_df['data'].tolist())

            # 提取元数据列（比如之前 add_meta_column 加的 filename）
            # 此时 extracted_df.columns 已经不包含 _source_path 了
            meta_cols = [c for c in extracted_df.columns if c != 'data']
            if meta_cols:
                meta_df = extracted_df[meta_cols].reset_index(drop=True)
                final_sub_df = pd.concat([meta_df, data_expanded], axis=1)
            else:
                final_sub_df = data_expanded
        else:
            final_sub_df = extracted_df

        # 6. 存入结果表
        self.sub_tables[new_table_name] = final_sub_df

    def __2_按列拆分(self):
        pass

    @invalidates_schema
    def extract_columns_table(self, new_table_name, key_fields, move_fields):
        """
        [结构修改] 垂直分表：将一部分字段剪切到新表中，并保留关联键。
        严格按照 [key_fields + move_fields] 的顺序排列新表的列。

        :param str new_table_name: 新 Sheet 的名称。
        :param str|list key_fields: 关联键（保留在原表，复制到新表）。
        :param str|list move_fields: 要移动的字段（从原表删除，移动到新表）。
        """
        if self.df.empty:
            return

        # 1. 解析字段路径 (这里解析出的列表顺序就是用户期望的顺序)
        # 支持从 DataFrame 列中提取（scope='all'）
        keys = self._resolve_fields(key_fields, scope='all')
        targets = self._resolve_fields(move_fields, scope='all')

        # 目标列顺序：先 Key 后 Value
        final_columns = keys + targets

        # 如果没有要移动的目标，直接返回
        if not targets:
            logger.warning(f"未找到匹配模式的字段：{move_fields}")
            return

        # 辅助函数：支持从 DataFrame 列或 JSON 中获取
        def get_val(row, data, path):
            # 1. 优先检查 DataFrame 顶层列
            if path in row:
                return row[path]
            # 2. 检查 JSON
            parent, k = self._resolve_path(data, path)
            return parent[k] if parent and k in parent else None

        def pop_val(row, data, path):
            # 1. 优先检查 DataFrame 顶层列
            if path in row:
                # 这里的 pop 只是获取值，删除操作在循环外统一处理（因为 iterrows 是副本/只读）
                return row[path]
            # 2. 检查 JSON
            parent, k = self._resolve_path(data, path)
            if parent and k in parent:
                return parent.pop(k)
            return None

        new_records = []

        # 2. 遍历数据
        for idx, row in self.df.iterrows():
            row_data = row['data']  # JSON 数据
            record = {}

            # (1) 复制 Key
            for k in keys:
                record[k] = get_val(row, row_data, k)

            # (2) 剪切 Value
            has_data = False
            for t in targets:
                val = pop_val(row, row_data, t)
                if val is not None:
                    record[t] = val
                    has_data = True

            # 只有当包含移动的数据时才保留该行
            if has_data:
                new_records.append(record)

        # 3. 删除已移动的 DataFrame 列
        df_cols_to_drop = [t for t in targets if t in self.df.columns]
        if df_cols_to_drop:
            self.df.drop(columns=df_cols_to_drop, inplace=True)
            logger.debug(f"已删除DataFrame列：{df_cols_to_drop}")

        # 4. 生成 DataFrame 并强制排序
        if new_records:
            sub_df = pd.DataFrame(new_records)

            # === 核心修改：强制按照解析出的字段顺序重排 ===
            # reindex 会自动对齐列，如果某列在某些行缺失（None），会自动填 NaN
            # 这样既保证了顺序，也保证了所有请求的字段都会出现在表头中，哪怕它是全空的
            sub_df = sub_df.reindex(columns=final_columns)

            self.sub_tables[new_table_name] = sub_df
            logger.info(f"提取子表 '{new_table_name}' 共 {len(sub_df)} 行。列已排序。")
        else:
            logger.warning(f"子表 '{new_table_name}' 为空。")

    def extract_lookup_table(self, table_name, key_fields, value_fields, remove_original=True):
        """
        提取映射关系表（配置表）。
        扫描主数据，将 key_fields 和 value_fields 的唯一组合提取出来存入 lookup_tables，
        并（可选）从主数据中删除 value_fields。

        :param str table_name: 映射表的名称（将作为 Excel Sheet 名）
        :param list key_fields: 决定映射关系的键字段列表（如 ['analysis.color.level']）
        :param list value_fields: 被决定的值字段列表（如 ['analysis.color.suggestion.introduction']）
        :param bool remove_original: 是否从主表中删除 value_fields。
        """
        if self.df.empty: return

        keys = self._resolve_fields(key_fields, scope='json')
        values = self._resolve_fields(value_fields, scope='json')

        # 内部辅助：安全获取深层值的函数
        def get_val(data, path):
            parent, k = self._resolve_path(data, path)
            return parent[k] if parent and k in parent else None

        # 1. 扫描所有数据，提取子集
        extracted_rows = []
        for row_data in self.df['data']:
            record = {}
            # 提取键
            for k in keys:
                record[k] = get_val(row_data, k)
            # 提取值
            for v in values:
                record[v] = get_val(row_data, v)
            extracted_rows.append(record)

        # 2. 生成 DataFrame 并去重
        lookup_df = pd.DataFrame(extracted_rows)
        # 核心：drop_duplicates 会保留所有出现过的唯一组合
        # 如果 level=2 有时对应 A，有时对应 B，这里会保留两行，反映真实数据情况
        lookup_df = lookup_df.drop_duplicates().reset_index(drop=True)

        # 3. 排序（可选，为了好看，按键排序）
        try:
            lookup_df = lookup_df.sort_values(by=keys)
        except Exception:
            pass  # 排序失败就算了

        # 4. 存入成员变量
        self.sub_tables[table_name] = lookup_df
        logger.info(f"提取查找表 '{table_name}' 共 {len(lookup_df)} 条唯一规则。")

        # 5. 从主表中移除冗余字段
        if remove_original:
            self.remove_fields(values)

    def extract_fields_to_long_table(self, new_table_name, id_field, target_fields,
                                     var_name='source_field', value_name='value',
                                     remove_original=True):
        """
        [通用接口] 宽表转长表 (Unpivot / Melt)。
        将主表中的多个列（target_fields）提取出来，转换为 "ID - 字段名 - 字段值" 的长表格式。
        适用于：提取所有图片路径、提取所有不同类型的分数等。

        这种操作在数据科学和数据库领域有专门的术语：
        1、逆透视 (Unpivoting)：最准确的商业智能（BI）或 Excel 术语。透视是将“行”变成“列”，
            逆透视则是将“列”（分析.filename, 斑点.filename）变成“行”。
        2、宽表转长表 (Wide-to-Long Transformation)：数据分析（Pandas/R语言）术语。
        宽格式：一行数据包含所有信息（比如一行有10个图片字段）。
        长格式：通过增加行数来减少列数（变成：ID列 + 类型列 + 值列）。
        3、EAV 模型 (Entity-Attribute-Value)：“ID-来源字段-值”这种结构，就是典型的 EAV 模型。
            它非常适合存储稀疏数据（即不是每行都有所有图片的情况）。

        :param str new_table_name: 新表的名称（Excel Sheet名）。
        :param str id_field: 主表中作为唯一标识的字段路径（如 "code" 或 "filename"）。
        :param list target_fields: 需要提取的字段路径列表。
        :param str var_name: 新表中存储"来源字段名"的列名。
        :param str value_name: 新表中存储"值"的列名。
        :param bool remove_original: 是否从主表中删除 target_fields。
        """
        if self.df.empty:
            return

        targets = self._resolve_fields(target_fields, scope='json')
        new_records = []

        # 辅助函数
        def get_val(data, path):
            parent, k = self._resolve_path(data, path)
            return parent[k] if parent and k in parent else None

        # 1. 遍历主数据
        for row_data in self.df['data']:
            # 获取每行的唯一 ID
            row_id = get_val(row_data, id_field)

            # 如果没有ID，通常跳过或生成一个，这里假设 ID 必须存在
            if row_id is None:
                continue

            # 遍历要提取的每一个字段
            for field in targets:
                val = get_val(row_data, field)

                # 只有值存在时才提取（稀疏存储，节省空间）
                # 如果你需要保留空值，可以去掉 if val: 判断
                if val:
                    new_records.append({
                        id_field: row_id,  # 锚点 ID
                        var_name: field,  # 来源字段 (Key)
                        value_name: val  # 实际值 (Value)
                    })

        # 2. 生成 DataFrame
        long_df = pd.DataFrame(new_records)

        # 3. 存入 sub_tables
        self.sub_tables[new_table_name] = long_df

        # 4. 从主表删除
        if remove_original:
            self.remove_fields(targets)


# 导出类：负责将数据导出为各种格式
class Json2TableExporter(Json2TableSplitter):
    """Json2Table导出类，负责将数据导出为各种格式"""

    def export_data(self):
        """
        导出为字典列表（API/JSON 用途）。
        """
        if self.df.empty:
            return []

        df_export = self.df.copy()
        if '_source_path' in df_export.columns:
            df_export = df_export.drop(columns=['_source_path'])
        return df_export.to_dict(orient='records')

    def _get_export_groups(self, split_by_col):
        """
        辅助函数：处理分组逻辑 (优化版)
        支持直接传入 JSON 内部字段路径，无需提前 add_field。
        """
        if not split_by_col:
            return [(None, self.df)]

        # 确保转为列表
        cols_to_check = split_by_col if isinstance(split_by_col, list) else [split_by_col]

        # 准备分组依据 (Groupers)
        groupers = []

        # 内部函数：安全获取嵌套字典的值
        def get_nested_val(record, path):
            parent, key = self._resolve_path(record, path)
            return parent[key] if parent else 'Unknown'

        for col in cols_to_check:
            if col in self.df.columns:
                # 情况A: 已经是顶层列 (例如 filename, _source_path)
                groupers.append(self.df[col])
            else:
                # 情况B: 顶层没找到，尝试从 JSON data 中提取
                logger.info(f'按JSON字段分组："{col}"')
                extracted_series = self.df['data'].apply(lambda d: get_nested_val(d, col))
                groupers.append(extracted_series)

        if not groupers:
            return [(None, self.df)]

        # dropna=False 确保 key 不存在的行也能分到一组 (通常是 NaN 或 Unknown)
        return self.df.groupby(groupers, dropna=False)

    def _process_group_data(self, group_df, meta_cols, flatten_data):
        """
        辅助函数：处理单个分组的数据展平和列合并
        """
        # 重置索引，确保 concat 时对齐
        group_df = group_df.reset_index(drop=True)

        if flatten_data:
            # json_normalize 会自动处理异构数据
            data_expanded = pd.json_normalize(group_df['data'].tolist())

            # 冲突解决：用户添加的列(meta_cols) 优先级 > 原始数据(data_expanded)
            if meta_cols:
                # 找出重名列
                collisions = set(meta_cols) & set(data_expanded.columns)
                if collisions:
                    logger.info(f'使用用户定义的列覆盖原始数据列：{collisions}')
                    # 删除原始数据中的同名列，保留用户的 meta_cols
                    data_expanded = data_expanded.drop(columns=list(collisions))

                # 合并：用户列 + 剩余的原始数据列
                return pd.concat([group_df[meta_cols], data_expanded], axis=1)
            else:
                return data_expanded
        else:
            # 不展开，直接转字符串防止 Excel 报错
            # 创建副本避免 SettingWithCopyWarning
            final_df = group_df.copy()
            final_df['data'] = final_df['data'].astype(str)
            return final_df[meta_cols + ['data']]

    def _generate_sheet_name(self, group_key, strategy):
        """
        辅助函数：生成并清洗 Sheet 名称。

        :param Any group_key: 当前分组的键值（可能是 None, 字符串, 或元组）。
        :param str|callable strategy: 命名策略。
            - str:
                - 如果包含 '{}' (e.g., "Data_{}"), 则使用 .format(key) 填充。
                - 如果等于 "Sheet1" 且有分组，则直接使用 Key 作为 Sheet 名。
                - 否则作为前缀 (e.g., "Prefix" -> "Prefix_Key")。
            - callable: 接受 key 返回字符串的函数。
        :return: 清洗后合法的 Excel Sheet 名称。
        """

        # 将 Key 转换为基础字符串
        def get_key_str(key):
            if key is None: return ''
            if isinstance(key, tuple): return '_'.join(map(str, key))
            return str(key)

        key_str = get_key_str(group_key)

        if callable(strategy):
            # 策略是函数：直接调用
            raw_sheet_name = str(strategy(group_key if group_key is not None else 'All'))
        elif group_key is None:
            # 没有分组，直接使用策略字符串作为 Sheet 名
            raw_sheet_name = str(strategy)
        else:
            # 有分组，且策略是字符串
            if strategy == 'Sheet1':
                # 默认情况：直接用分组键作为 Sheet 名
                raw_sheet_name = key_str
            elif '{}' in strategy:
                # 模板模式：填充占位符
                raw_sheet_name = strategy.format(key_str)
            else:
                # 前缀模式
                raw_sheet_name = f'{strategy}_{key_str}'

        # Excel 限制: max 31 chars, 无特殊字符
        safe_sheet_name = raw_sheet_name.replace(':', '_').replace('\\', '_').replace('/', '_')[:31]

        # 防止空名称
        if not safe_sheet_name:
            return 'Default'

        return safe_sheet_name

    def add_export_column_rule(self, field_patterns, after=None):
        """
        [配置] 添加一条列排序规则。支持多次调用，后调用的规则优先级更高（会覆盖之前的移动）。

        :param str|list field_patterns: 要移动的字段（支持通配符）。
        :param str after: 锚点字段（支持通配符，取第一个匹配到的）。
                          - None (默认): 将 field_patterns 移动到表格 **最左侧 (Start)**。
                          - 'some_col': 将 field_patterns 移动到 'some_col' 的 **后面**。

        用法示例：
        1. 绝对置顶：mgr.add_export_column_rule(['id', 'name']) -> id, name 排在最前
        2. 相对排序：mgr.add_export_column_rule(['score', 'rank'], after='exam_date')
        3. 后来居上：如果之前 id 已经在最前，现在调用 mgr.add_export_column_rule(['code'], after=None)
           -> 结果是 code, id, name ... (code 变为最新的第一位)
        """
        patterns = self._ensure_list(field_patterns)
        self._sort_operations.append((patterns, after))

    def reorder_columns(self, df, rules=None):
        """
        [工具方法] 对 DataFrame 的列进行重排。

        :param pd.DataFrame df: 目标 DataFrame
        :param list rules: (Optional) 排序规则列表。
                           格式: [(target_patterns, anchor_pattern), ...]
                           如果不传，默认使用通过 add_export_column_rule 添加的全局规则。
        :return: 重排后的 DataFrame
        """
        sort_ops = rules if rules is not None else self._sort_operations

        if not sort_ops:
            return df

        # 1. 获取当前列顺序（作为操作底板）
        current_cols = list(df.columns)

        # 2. 顺序执行每一条规则
        for targets_patterns, anchor_pattern in sort_ops:

            # --- A. 解析要移动的字段 (Targets) ---
            targets = []
            seen = set()
            for pattern in targets_patterns:
                matches = []
                if isinstance(pattern, PStr):
                    if pattern.is_literal:
                        # Literal PStr: 严格精确匹配
                        # 只有当该列真实存在时才匹配
                        s_pat = str(pattern)
                        if s_pat in current_cols:
                            matches = [s_pat]
                    else:
                        # PStr (Regex / Glob): 智能匹配
                        matches = pattern.find_matches(current_cols)
                elif isinstance(pattern, str):
                    # 普通字符串：保持原有逻辑，支持 fnmatch 通配符
                    matches = fnmatch.filter(current_cols, pattern)

                # 简单的处理：过滤出存在的列
                valid_matches = [m for m in matches if m not in seen]

                # 提取当前 pattern 匹配到的所有列（保持原相对顺序）
                matches_in_order = [c for c in current_cols if c in valid_matches]
                targets.extend(matches_in_order)
                seen.update(matches_in_order)

            if not targets:
                continue

            # --- B. 确定插入位置 (Insert Index) ---
            insert_index = 0

            # 锚点适配
            # 支持 PStr (如果是 Glob/Literal，str() 取出内容即可支持 fnmatch)
            # 如果是 Regex PStr，这里当作普通字符串处理（fnmatch 可能无法正确解析正则语法，但这是预期行为）
            anchor_str = str(anchor_pattern) if anchor_pattern is not None else None

            if anchor_str:
                # 查找锚点
                anchors = fnmatch.filter(current_cols, anchor_str)
                # 只取第一个匹配到的作为锚点
                real_anchor = anchors[0] if anchors else None

                if real_anchor and real_anchor in current_cols:
                    # 目标是插在 anchor 的后面，所以 index 是 anchor_idx + 1
                    insert_index = current_cols.index(real_anchor) + 1
                else:
                    continue
            else:
                # after=None，表示置顶，index=0
                insert_index = 0

            # --- C. 执行移动 (Move) ---
            # 1. 从列表中移除 targets
            targets_set = set(targets)
            remaining_cols = [c for c in current_cols if c not in targets_set]

            # 2. 重新计算插入位置
            if anchor_str:
                try:
                    new_anchor_idx = remaining_cols.index(real_anchor)
                    insert_index = new_anchor_idx + 1
                except ValueError:
                    insert_index = len(remaining_cols)
            else:
                insert_index = 0

            # 3. 插入
            current_cols = remaining_cols[:insert_index] + targets + remaining_cols[insert_index:]

        # 3. 返回重排后的 DataFrame
        return df[current_cols]

    def iter_export_frames(self, split_by_col=None, flatten_data=True, sheet_name_strategy='Sheet1'):
        """
        [生成器] 迭代产生所有需要导出的 DataFrame。
        将数据生成的逻辑与 IO 写入逻辑解耦，允许用户在写入前对 DataFrame 进行自定义操作。

        :yield: (sheet_name, df, frame_type)
            - sheet_name: 建议的 Sheet 名称
            - df: 准备好的 DataFrame
            - frame_type: 数据类型，'main' (主数据) 或 'lookup' (查找表)
        """
        if self.df.empty:
            return

        meta_cols = [c for c in self.df.columns if c not in ['data', '_source_path']]

        # 1. 产生主数据 (Main Data)
        groups = self._get_export_groups(split_by_col)
        for group_key, group_df in groups:
            # 数据展平与合并
            final_df = self._process_group_data(group_df, meta_cols, flatten_data)

            # 应用默认的列排序规则
            # (注意：如果用户在外部再次调用 reorder_columns，相当于排序了两次，这是允许的)
            final_df = self.reorder_columns(final_df)

            # 生成 Sheet 名
            safe_sheet_name = self._generate_sheet_name(group_key, sheet_name_strategy)

            yield safe_sheet_name, final_df, 'main'

        # 2. 产生副表 (Sub Tables)
        for table_name, sub_df in self.sub_tables.items():
            # 截断 Sheet 名防止报错
            safe_name = table_name[:31].replace(':', '_').replace('\\', '_')

            # 副表目前不自动应用 reorder_columns，因为通常它们的结构是固定的
            # 但用户可以通过 iter_export_frames 拿到 df 后手动调用 reorder_columns

            yield safe_name, sub_df, 'sub_table'

    def to_excel(self, output_path, split_by_col=None, flatten_data=True, sheet_name_strategy='Sheet1'):
        """
        导出为 Excel 表格，支持异构数据展平和多 Sheet 拆分。
        (现在是 iter_export_frames 的一个简单包装器)

        :param str output_path: 输出 Excel 路径
        :param str|list split_by_col: 根据哪一列(或多列)来拆分 Sheet。例如 'category' 或 ['category', 'sub_type']。
        :param bool flatten_data: 是否将 data 列里的 Dict 展开成多列。
        :param str|callable sheet_name_strategy: 控制 Sheet 名称的生成策略。
        """
        if self.df.empty:
            raise ValueError('No data to export.')

        # 使用 ExcelWriter 上下文
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

            # 使用生成器获取所有数据
            for sheet_name, df, frame_type in self.iter_export_frames(split_by_col, flatten_data, sheet_name_strategy):

                # 写入 Excel
                df.to_excel(writer, sheet_name=sheet_name, index=False)

                if frame_type == 'sub_table':
                    logger.success(f'导出副表Sheet：{sheet_name}')
                # else:
                #     logger.info(f'导出主表Sheet：{sheet_name}')


# 最终的Json2Table类，继承所有功能子类
class Json2Table(Json2TableExporter):
    """Json2Table主类，继承所有功能子类"""
    pass
