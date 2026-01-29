# -*- coding: utf-8 -*-
import pytest
from loguru import logger
from pyxllib.prog.uni_cache import uni_cache

class MockService:
    def __init__(self, name):
        self.name = name
        self.exec_count = 0

    @uni_cache(mode='id', maxsize=1)
    def get_data_id(self):
        self.exec_count += 1
        return f"data_for_{self.name}_{self.exec_count}"

    @uni_cache(maxsize=1)  # 默认 mode='str'
    def get_data_str(self):
        self.exec_count += 1
        return f"data_for_{self.name}_{self.exec_count}"

    @uni_cache(mode='ignore', maxsize=1)
    def get_data_ignore(self):
        self.exec_count += 1
        return f"shared_data_{self.exec_count}"

class MockServiceMax2:
    def __init__(self, name):
        self.name = name
        self.exec_count = 0

    @uni_cache(mode='id', maxsize=2)
    def get_data(self):
        self.exec_count += 1
        return f"data_for_{self.name}_{self.exec_count}"

def test_eviction_mode_id():
    """验证 mode='id', maxsize=1 时，不同实例会互相覆盖（淘汰）"""
    s1 = MockService("s1")
    s2 = MockService("s2")

    # 1. s1 执行
    res1_1 = s1.get_data_id()
    assert s1.exec_count == 1
    
    # 2. s1 再次执行 -> 命中缓存
    res1_2 = s1.get_data_id()
    assert s1.exec_count == 1
    assert res1_1 == res1_2

    # 3. s2 执行 -> 因为 maxsize=1，s1 的缓存被淘汰
    res2_1 = s2.get_data_id()
    assert s2.exec_count == 1
    
    # 4. s1 再次执行 -> 缓存已失效，重新执行
    res1_3 = s1.get_data_id()
    assert s1.exec_count == 2
    assert res1_3 != res1_1
    
    logger.success("test_eviction_mode_id: 验证成功，maxsize=1 导致实例间覆盖")

def test_eviction_mode_str():
    """验证默认 mode='str' 时，如果 self 的 str 不同，依然会发生覆盖"""
    s1 = MockService("s1")
    s2 = MockService("s2")
    
    # 默认情况下，MockService 没有重写 __str__，其 str 包含内存地址，是唯一的
    
    # 1. s1 执行
    s1.get_data_str()
    assert s1.exec_count == 1
    
    # 2. s2 执行 -> 覆盖 s1
    s2.get_data_str()
    assert s2.exec_count == 1
    
    # 3. s1 执行 -> 重新计算
    s1.get_data_str()
    assert s1.exec_count == 2
    
    logger.success("test_eviction_mode_str: 验证成功，默认 str 模式下不同实例依然会竞争 maxsize")

def test_self_as_first_param():
    """验证 self 确实作为第1个参数进入策略分析"""
    s1 = MockService("s1")
    s2 = MockService("s2")

    # get_data_ignore 使用了 mode='ignore'，这意味着第1个参数(self)被忽略
    # 那么所有实例应该共享同一个缓存键（因为没有其他参数）
    
    # 1. s1 执行
    res1 = s1.get_data_ignore()
    assert s1.exec_count == 1
    
    # 2. s2 执行 -> 应该命中 s1 留下的缓存！
    res2 = s2.get_data_ignore()
    assert s2.exec_count == 0  # s2 根本没执行自己的逻辑，直接拿了缓存
    assert res1 == res2
    
    logger.success("test_self_as_first_param: 验证成功，self 作为第1个参数被 ignore 后实现了跨实例共享")

def test_no_eviction_maxsize_2():
    """验证 maxsize=2 时，两个不同实例可以同时持有缓存而不冲突"""
    s1 = MockServiceMax2("s1")
    s2 = MockServiceMax2("s2")

    # 1. s1 执行
    res1_1 = s1.get_data()
    assert s1.exec_count == 1

    # 2. s2 执行
    res2_1 = s2.get_data()
    assert s2.exec_count == 1

    # 3. s1 再次执行 -> 应该依然命中缓存，因为 maxsize=2
    res1_2 = s1.get_data()
    assert s1.exec_count == 1
    assert res1_1 == res1_2

    # 4. s2 再次执行 -> 应该依然命中缓存
    res2_2 = s2.get_data()
    assert s2.exec_count == 1
    assert res2_1 == res2_2

    # 5. 引入第三个实例 s3 -> 挤出最久未使用的那个（LRU）
    s3 = MockServiceMax2("s3")
    s3.get_data()
    
    # 因为步骤4刚访问过 s2，所以此时 s1 是最久未使用的（LRU）
    # 再次访问 s1 应该会重新计算
    s1.get_data()
    assert s1.exec_count == 2
    
    logger.success("test_no_eviction_maxsize_2: 验证成功，maxsize=2 允许两个实例并存")

if __name__ == "__main__":
    pytest.main([__file__])
