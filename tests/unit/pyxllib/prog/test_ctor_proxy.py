import pytest
import time
from ctor_proxy import ConstructorProxy


class MockDatabase:
    def __init__(self, host='localhost', port=5432):
        self.host = host
        self.port = port
        self.created_at = time.time()

    def __repr__(self):
        return f"MockDatabase(host='{self.host}', port={self.port}, created_at={self.created_at})"


class TestConstructorProxy:
    def setup_method(self):
        # 每个测试前清理注册表，防止状态污染
        ConstructorProxy._registry.clear()

    def test_basic_lazy_loading(self):
        """测试基础链式调用和惰性加载"""
        proxy = ConstructorProxy(MockDatabase, 'db_basic')
        proxy.config(host='192.168.1.1')

        # 此时不应实例化
        assert 'db_basic' not in ConstructorProxy._registry

        # 获取实例
        db = proxy.get()
        assert isinstance(db, MockDatabase)
        assert db.host == '192.168.1.1'
        assert 'db_basic' in ConstructorProxy._registry

    def test_singleton_by_name(self):
        """测试具名单例模式"""
        proxy1 = ConstructorProxy(MockDatabase, 'shared_db').config(host='1.1.1.1')
        db1 = proxy1.get()

        proxy2 = ConstructorProxy(MockDatabase, 'shared_db').config(host='1.1.1.1')
        db2 = proxy2.get()

        assert db1 is db2
        assert db1.host == '1.1.1.1'

    def test_anonymous_fingerprint(self):
        """测试基于配置指纹的匿名复用"""
        # 相同的配置应该返回同一个实例
        db1 = ConstructorProxy(MockDatabase).config(host='localhost', port=5432).get()
        db2 = ConstructorProxy(MockDatabase).config(host='localhost', port=5432).get()

        assert db1 is db2

        # 不同的配置应该返回不同实例
        db3 = ConstructorProxy(MockDatabase).config(host='remote', port=5432).get()
        assert db1 is not db3

    def test_hybrid_alias(self):
        """测试混合索引 (具名 + 匿名)"""
        # 1. 创建一个既有名字又有指纹的实例
        proxy = ConstructorProxy(MockDatabase, name=['master', None])
        db_master = proxy.config(host='master_host').get()

        # 2. 通过名字获取
        db_by_name = ConstructorProxy(MockDatabase, 'master').get()
        assert db_by_name is db_master

        # 3. 通过相同的配置（匿名）获取
        db_anon = ConstructorProxy(MockDatabase).config(host='master_host').get()
        assert db_anon is db_master

    def test_multi_alias(self):
        """测试多名模式"""
        db = ConstructorProxy(MockDatabase, name=['alias_a', 'alias_b']).config(port=8888).get()

        db_a = ConstructorProxy(MockDatabase, 'alias_a').get()
        db_b = ConstructorProxy(MockDatabase, 'alias_b').get()

        assert db is db_a
        assert db is db_b

    def test_recreate(self):
        """测试强制重新实例化"""
        proxy = ConstructorProxy(MockDatabase, 'reload_db').config(host='v1')
        db1 = proxy.get()

        # 强制重建
        db2 = proxy.config(host='v2').recreate()

        assert db1 is not db2
        assert db2.host == 'v2'

        # 再次获取应该得到新的实例
        db3 = proxy.get()
        assert db3 is db2

    def test_clear(self):
        """测试清除引用"""
        proxy = ConstructorProxy(MockDatabase, 'temp_db')
        db = proxy.get()

        assert 'temp_db' in ConstructorProxy._registry

        proxy.clear()
        assert 'temp_db' not in ConstructorProxy._registry

        # 再次获取应该是新实例
        db_new = proxy.get()
        assert db is not db_new

    def test_config_consistency_check(self):
        """测试配置一致性检查（混合模式下）"""
        # 1. 创建 'strict_db'，配置为 port=1000
        ConstructorProxy(MockDatabase, name=['strict_db', None]).config(port=1000).get()

        # 2. 尝试获取 'strict_db'，但配置不同 (port=2000)
        # 预期应该抛出 ValueError，因为 name=['strict_db', None] 包含 None，会触发严格检查
        with pytest.raises(ValueError, match='配置冲突'):
            ConstructorProxy(MockDatabase, name=['strict_db', None]).config(port=2000).get()

    def test_ambiguous_instance_error(self):
        """测试实例引用歧义冲突"""
        # 1. 创建实例 A，名为 'A'
        ConstructorProxy(MockDatabase, 'A').config(host='host_A').get()

        # 2. 创建实例 B，名为 'B'
        ConstructorProxy(MockDatabase, 'B').config(host='host_B').get()

        # 3. 尝试将 'A' 和 'B' 绑定到同一个 new proxy，这将引发冲突
        with pytest.raises(ValueError, match='实例引用歧义'):
            ConstructorProxy(MockDatabase, name=['A', 'B']).get()

    def test_auto_association(self):
        """测试自动关联 (Auto Association)"""
        # 1. 创建 'primary'
        db = ConstructorProxy(MockDatabase, 'primary').config(host='h1').get()

        # 2. 创建一个新别名 'secondary'，同时引用 'primary'
        # 逻辑：'primary' 已存在，'secondary' 不存在。
        # 系统应返回 'primary' 的实例，并将 'secondary' 指向它。
        db_new = ConstructorProxy(MockDatabase, name=['primary', 'secondary']).get()

        assert db_new is db
        assert ConstructorProxy._registry['secondary'].instance is db

        # 验证 'secondary' 是否真的生效
        assert ConstructorProxy(MockDatabase, 'secondary').get() is db

    def test_fingerprint_order_independence(self):
        """测试参数指纹对 kwargs 顺序不敏感"""
        # 使用 MockDatabase 实际支持的参数
        db1 = ConstructorProxy(MockDatabase).config(host='server', port=80).get()
        db2 = ConstructorProxy(MockDatabase).config(port=80, host='server').get()

        assert db1 is db2


if __name__ == '__main__':
    pytest.main(['-v', __file__])
