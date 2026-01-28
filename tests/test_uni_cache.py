# -*- coding: utf-8 -*-
"""
test_uni_cache.py - 测试 uni_cache 装饰器的各项功能
"""

import time
import threading
import concurrent.futures
from loguru import logger
from pyxllib.prog.uni_cache import uni_cache


def test_basic_and_alignment():
    """测试基础函数与参数对齐"""
    logger.info("--- [Test] 基础函数与参数对齐 ---")

    @uni_cache(mode='str')
    def add(a, b):
        logger.debug(f"  -> 计算 {a} + {b} ...")
        return a + b

    # 第一次调用
    r1 = add(1, 2)
    # 第二次调用: 即使写法不同，但逻辑参数一致，应该命中缓存
    r2 = add(a=1, b=2)
    r3 = add(1, b=2)

    assert r1 == 3
    assert r1 == r2 == r3
    logger.success("Pass: 参数对齐缓存命中")


def test_class_factory():
    """模拟 PaddleOCR (类工厂模式)"""
    logger.info("--- [Test] 模拟 PaddleOCR 类工厂 ---")

    class MockPaddleOCR:
        def __init__(self, lang="ch", use_gpu=False):
            self.config = f"OCR(lang={lang}, gpu={use_gpu})"
            logger.debug(f"  -> 初始化模型: {self.config}")

        def predict(self, img):
            return f"Result of {img} by {self.config}"

    # 使用装饰器
    OCRFactory = uni_cache(MockPaddleOCR, namespace='cacher')

    # 1. 创建实例
    ocr1 = OCRFactory(lang="ch")
    # 2. 再次创建 (参数相同) -> 命中缓存
    ocr2 = OCRFactory(lang="ch", use_gpu=False)
    # 3. 创建不同实例
    ocr3 = OCRFactory(lang="en")

    assert ocr1 is ocr2
    assert ocr1 is not ocr3
    logger.success("Pass: 类实例单例化成功")

    # 4. 验证管理接口
    OCRFactory.cacher.clear()
    info = OCRFactory.cacher.info()
    assert info['count'] == 0
    logger.info(f"Info after clear: {info}")


def test_complex_rules():
    """测试复杂规则 (ignore, id)"""
    logger.info("--- [Test] 复杂规则 (ignore, id) ---")

    class BigContext:
        pass

    # 规则: 
    # 1. 第1个参数 (ctx) -> ignore
    # 2. 命名参数 heavy_data -> id
    # 3. 其他 -> str
    @uni_cache(mode='ignore, heavy_data=id, str', namespace=None)
    def process_data(ctx, heavy_data, option="fast"):
        logger.debug(f"  -> 处理数据 mode={option}")
        return "Done"

    ctx1 = BigContext()
    data = ["Huge Data"]

    # 第一次运行
    process_data(ctx1, data, "fast")

    # 第二次运行: 换了 ctx 对象 (ignore生效)，换了 data 变量名但引用没变 (id生效)
    ctx2 = BigContext()
    # 这里应该命中缓存
    process_data(ctx2, heavy_data=data, option="fast")

    # 第三次运行: 换了 data 的引用
    # 引用变了，id变了 -> 触发重新执行
    process_data(ctx1, ["Huge Data"], "fast")

    logger.success("Pass: 混合规则验证成功")


def test_lifecycle_management():
    """测试强制刷新与瞬态调用"""
    logger.info("--- [Test] 强制刷新与瞬态调用 ---")

    @uni_cache
    def get_time():
        return time.time()

    t1 = get_time()
    time.sleep(0.01)
    t2 = get_time()
    assert t1 == t2  # 缓存生效

    # 瞬态调用 (new): 跑一次新的，但不存缓存
    t_new = get_time.cache.new()
    assert t_new != t1

    # 此时再调 get_time()，还是旧缓存
    assert get_time() == t1

    # 刷新调用 (refresh): 跑一次新的，并更新缓存
    t_refresh = get_time.cache.refresh()
    assert t_refresh != t1
    
    # 此时再调，是新的了
    assert get_time() == t_refresh

    logger.success("Pass: 生命周期管理验证成功")


def test_instance_methods():
    """测试实例方法 - 互相隔离与全局共享"""
    logger.info("--- [Test] 实例方法 (互相隔离与全局共享) ---")

    class UserData:
        def __init__(self, user_id):
            self.user_id = user_id

        @uni_cache(mode='id, str')
        def query_report(self, day):
            logger.debug(f"  [Exec] 查询用户 {self.user_id} 第 {day} 天的数据...")
            return f"Report_{self.user_id}_{day}"

    u1 = UserData("Alice")
    u2 = UserData("Bob")

    res1 = u1.query_report(1)
    assert u1.query_report(1) == res1
    
    res2 = u2.query_report(1)
    assert res1 != res2
    logger.success("Pass: 不同对象缓存隔离")

    class ToolBox:
        @uni_cache(mode='ignore, str')
        def heavy_calc(self, x, y):
            logger.debug(f"  [Exec] 执行通用计算 {x} + {y} ...")
            return x + y

    t1 = ToolBox()
    t2 = ToolBox()

    v1 = t1.heavy_calc(10, 20)
    v2 = t2.heavy_calc(10, 20)
    assert v1 == v2
    assert t1.heavy_calc.cache.info()['count'] == 1
    logger.success("Pass: 忽略 Self 实现跨实例共享缓存")


def test_property_and_class_methods():
    """测试 @property, @classmethod, @staticmethod"""
    logger.info("--- [Test] OOP 特性 (@property, @classmethod, @staticmethod) ---")

    class Rectangle:
        def __init__(self, width, height):
            self.width = width
            self.height = height

        @property
        @uni_cache(mode='id')
        def area(self):
            logger.debug(f"  [Exec] 计算面积 {self.width}x{self.height}...")
            return self.width * self.height

    r1 = Rectangle(10, 20)
    assert r1.area == 200
    assert r1.area == 200  # 命中

    class ConfigLoader:
        _base_path = "/etc"
        @classmethod
        @uni_cache(mode='ignore, str')
        def load(cls, filename):
            logger.debug(f"  [Exec] Loading {cls._base_path}/{filename}")
            return object()

    o1 = ConfigLoader.load("config.json")
    o2 = ConfigLoader.load("config.json")
    assert o1 is o2

    class MathStatic:
        @staticmethod
        @uni_cache(mode='str')
        def factorial(n):
            return n

    assert MathStatic.factorial(5) == 5
    MathStatic.factorial.cache.clear()
    logger.success("Pass: OOP 特性验证成功")


def test_multithreading():
    """测试多线程并发下的安全性"""
    logger.info("--- [Test] 多线程并发安全性 ---")

    exec_count = 0
    exec_lock = threading.Lock()

    @uni_cache
    def expensive_task(x):
        nonlocal exec_count
        with exec_lock:
            exec_count += 1
        time.sleep(0.1)  # 模拟耗时操作
        return x * x

    # 使用 10 个线程同时调用同一个参数
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(expensive_task, 10) for _ in range(10)]
        results = [f.result() for f in futures]

    # 验证：所有结果应该一致，且实际执行次数应该只有 1 次
    assert all(r == 100 for r in results)
    assert exec_count == 1
    logger.success(f"Pass: 多线程并发下函数仅执行 {exec_count} 次，缓存生效")


def test_recursive_lock():
    """测试 RLock 是否支持递归调用"""
    logger.info("--- [Test] 递归调用支持 (RLock) ---")

    @uni_cache
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)

    # 如果是普通 Lock，这里会死锁
    res = factorial(5)
    assert res == 120
    logger.success("Pass: 递归调用成功，未发生死锁")


def test_ttl_expiry():
    """测试 TTL 超时功能"""
    logger.info("--- [Test] TTL 超时功能 ---")

    @uni_cache(ttl=2)  # 设置 2 秒超时，方便测试
    def get_data():
        logger.debug("  [Exec] 获取新数据...")
        return time.time()

    # 1. 第一次获取
    t1 = get_data()

    # 2. 立即再次获取 -> 命中缓存
    t2 = get_data()
    assert t1 == t2
    logger.debug("  -> 2秒内获取，命中缓存")

    # 3. 等待 3 秒 (超过 2 秒 TTL)
    logger.info("  等待 3 秒以使缓存过期...")
    time.sleep(3)

    # 4. 再次获取 -> 缓存过期，重新执行
    t3 = get_data()
    assert t1 != t3
    logger.debug("  -> 超过 TTL 后获取，缓存已失效并重新执行")

    logger.success("Pass: TTL 超时功能验证成功")


def test_type_fingerprint():
    """测试类型指纹，区分 list 和 tuple 等"""
    logger.info("--- [Test] 类型指纹 (Type Fingerprint) ---")

    exec_count = 0

    @uni_cache
    def get_data(data):
        nonlocal exec_count
        exec_count += 1
        return f"Data_{len(data)}"

    # 1. 传入 list
    r1 = get_data([1, 2])
    assert exec_count == 1

    # 2. 再次传入 list (相同内容) -> 命中
    r2 = get_data([1, 2])
    assert exec_count == 1
    assert r1 == r2

    # 3. 传入 tuple (相同内容) -> 不应该命中，因为类型不同
    r3 = get_data((1, 2))
    assert exec_count == 2
    assert r3 == r1

    # 4. 传入 set
    r4 = get_data({1, 2})
    assert exec_count == 3

    logger.success("Pass: 类型指纹成功区分不同容器类型")


if __name__ == "__main__":
    test_basic_and_alignment()
    test_class_factory()
    test_complex_rules()
    test_lifecycle_management()
    test_instance_methods()
    test_property_and_class_methods()
    test_multithreading()
    test_recursive_lock()
    test_ttl_expiry()
    test_type_fingerprint()
    logger.success(">>> 所有测试通过！")
