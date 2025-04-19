#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/04/01

from collections import Counter

import jmespath


class JsonTool:
    @classmethod
    def find_paths_by_condition(cls, obj, condition, current_path=''):
        """
        根据提供的条件递归搜索JSON对象，查找满足条件的路径。
        """
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_path = f"{current_path}.{k}" if current_path else k
                if condition(k, v):
                    yield new_path
                else:
                    if isinstance(v, (dict, list)):
                        yield from JsonTool.find_paths_by_condition(v, condition, new_path)
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                new_path = f"{current_path}[{index}]"
                if isinstance(item, (dict, list)):
                    yield from JsonTool.find_paths_by_condition(item, condition, new_path)

    @classmethod
    def find_paths_with_key(cls, obj, key, current_path=''):
        """
        查找包含指定键名的路径。
        """
        # 定义一个简单的条件函数，仅基于键名
        condition = lambda k, v: k == key
        return JsonTool.find_paths_by_condition(obj, condition, current_path)


class JsonlTool:
    @classmethod
    def check_key_positions(cls, json_list, key):
        """
        在JSON对象列表中统计每种包含指定键名的路径出现的次数。

        :param json_list: 包含多个JSON对象的列表。
        :param key: 要查找的键名。
        :return: 一个Counter对象，统计每种路径出现的次数。
        """
        paths_counter = Counter()
        for json_obj in json_list:
            # 使用JsonTool查找当前对象中包含指定键的路径
            paths = list(JsonTool.find_paths_with_key(json_obj, key))
            paths_counter.update(paths)

        return paths_counter

    @classmethod
    def check_path_values(cls, json_list, jmespath_expression, value_extractor=lambda x: x):
        """
        根据提供的jmespath表达式，在JSON对象列表中直接统计表达式取值的次数。
        支持自定义函数来获取要统计的键。

        :param json_list: 包含多个JSON对象的列表。
        :param jmespath_expression: 要应用的jmespath表达式。
        :param value_extractor: 一个函数，用于从jmespath.search的结果中提取要统计的值。
        :return: 一个Counter对象，统计提取值出现的次数。
        """
        values_counter = Counter()
        for json_obj in json_list:

            # 使用jmespath搜索当前对象
            result = jmespath.search(jmespath_expression, json_obj)
            # 使用自定义的value_extractor函数提取要统计的值
            extracted_values = value_extractor(result)
            if not isinstance(extracted_values, list):
                extracted_values = [extracted_values]

            for value in extracted_values:
                values_counter[value] += 1

        values_counter = Counter(dict(values_counter.most_common()))
        return values_counter
