import os
import shutil
import tempfile
import zipfile
import pytest
from pyxllib.file.walker import DirWalker

@pytest.fixture
def temp_dir():
    """创建一个临时的目录环境进行测试"""
    # 使用 tempfile.mkdtemp 创建一个临时目录，并进行路径规范化
    root = os.path.realpath(tempfile.mkdtemp())
    
    # 建立测试目录结构
    # root/
    #   - file1.txt
    #   - file2.py
    #   - dir1/
    #     - file3.txt
    #     - file4.py
    #   - dir2/
    #     - file5.txt
    #     - subdir1/
    #       - file6.py
    #   - .hidden_dir/
    #     - secret.txt
    
    with open(os.path.join(root, 'file1.txt'), 'w') as f: f.write('content1')
    with open(os.path.join(root, 'file2.py'), 'w') as f: f.write('print("hello")')
    
    dir1 = os.path.join(root, 'dir1')
    os.makedirs(dir1)
    with open(os.path.join(dir1, 'file3.txt'), 'w') as f: f.write('content3')
    with open(os.path.join(dir1, 'file4.py'), 'w') as f: f.write('print("world")')
    
    dir2 = os.path.join(root, 'dir2')
    os.makedirs(dir2)
    with open(os.path.join(dir2, 'file5.txt'), 'w') as f: f.write('content5')
    
    subdir1 = os.path.join(dir2, 'subdir1')
    os.makedirs(subdir1)
    with open(os.path.join(subdir1, 'file6.py'), 'w') as f: f.write('print("subdir")')
    
    hidden_dir = os.path.join(root, '.hidden_dir')
    os.makedirs(hidden_dir)
    with open(os.path.join(hidden_dir, 'secret.txt'), 'w') as f: f.write('shhh')
    
    yield root
    
    # 测试结束后删除临时目录
    shutil.rmtree(root)

def test_basic_walk(temp_dir):
    """测试基本的遍历功能"""
    dw = DirWalker(temp_dir, enter=True, select=True)
    
    # 测试 iter_files
    files = list(dw.iter_files())
    file_names = {os.path.basename(f.path) for f in files}
    expected_files = {'file1.txt', 'file2.py', 'file3.txt', 'file4.py', 'file5.txt', 'file6.py', 'secret.txt'}
    assert file_names == expected_files

    # 测试 iter_dirs
    dirs = list(dw.iter_dirs())
    dir_names = {os.path.basename(d.path) for d in dirs}
    expected_dirs = {'dir1', 'dir2', 'subdir1', '.hidden_dir'}
    assert dir_names == expected_dirs

def test_filtering_rules(temp_dir):
    """测试过滤规则"""
    # 1. 测试 match_ext
    dw = DirWalker(temp_dir, enter=True, select=False)
    dw.include_file.match_ext('.py')
    py_files = list(dw.iter_files())
    assert all(f.name.endswith('.py') for f in py_files)
    assert len(py_files) == 3

    # 2. 测试 match_name
    dw = DirWalker(temp_dir, enter=True, select=True)
    dw.skip_dir.match_name('.*')  # 跳过进入隐藏目录
    dw.exclude.match_name('.*')   # 排除隐藏文件
    files = list(dw.iter_files())
    file_names = {f.name for f in files}
    assert 'secret.txt' not in file_names
    assert 'file1.txt' in file_names

    # 3. 测试 skip_dir
    dw = DirWalker(temp_dir, enter=True, select=True)
    dw.skip_dir.match_name('dir1')
    files = list(dw.iter_files())
    file_names = {f.name for f in files}
    assert 'file3.txt' not in file_names
    assert 'file4.py' not in file_names
    assert 'file1.txt' in file_names

def test_walk_format(temp_dir):
    """测试 walk 方法返回的格式 (类似 os.walk)"""
    dw = DirWalker(temp_dir, enter=True, select=True)
    dw.exclude.match_name('.*')
    
    results = list(dw.walk())
    # results[0] 应为 (root, [file1.txt, file2.py], [dir1, dir2])
    root_res = results[0]
    assert root_res[0] == temp_dir
    assert set(root_res[1]) == {'file1.txt', 'file2.py'}
    assert set(root_res[2]) == {'dir1', 'dir2'}

def test_pack_zip(temp_dir):
    """测试打包功能"""
    # 必须设置 select=False，否则默认全选，我们的 include_file 就没意义了
    dw = DirWalker(temp_dir, enter=True, select=False)
    dw.include_file.match_ext('.py')
    
    zip_path = os.path.join(temp_dir, 'test.zip')
    dw.pack_zip(zip_path)
    
    assert os.path.exists(zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zip_files = zf.namelist()
        # 应该只包含 .py 文件
        assert all(f.endswith('.py') for f in zip_files)
        assert len(zip_files) == 3

def test_chaining(temp_dir):
    """测试链式调用"""
    dw = DirWalker(temp_dir, enter=True) \
        .include_file.match_ext('.txt') \
        .exclude.match_name('file3.txt')
    
    files = list(dw.iter_files())
    file_names = {f.name for f in files}
    assert 'file1.txt' in file_names
    assert 'file5.txt' in file_names
    assert 'file3.txt' not in file_names
    assert 'file2.py' not in file_names

def test_size_filter(temp_dir):
    """测试文件大小过滤"""
    # 创建一个稍大的文件
    big_file = os.path.join(temp_dir, 'big.dat')
    with open(big_file, 'wb') as f:
        f.write(b'0' * 1024) # 1KB
        
    dw = DirWalker(temp_dir, enter=True, select=False)
    dw.include_file.match_size(min_size=500)
    
    files = list(dw.iter_files())
    assert len(files) == 1
    assert files[0].name == 'big.dat'

def test_relpath_filter(temp_dir):
    """测试相对路径过滤"""
    dw = DirWalker(temp_dir, enter=True, select=False)
    # 使用通配符匹配相对路径末尾
    dw.include_file.match_relpath('*file3.txt')
    dw.include_file.match_relpath('*file4.py')
    
    files = list(dw.iter_files())
    file_names = {f.name for f in files}
    assert file_names == {'file3.txt', 'file4.py'}
    assert 'file1.txt' not in file_names

def test_custom_predicate(temp_dir):
    """测试自定义判断函数"""
    dw = DirWalker(temp_dir, enter=True, select=False)
    
    # 只选中文件名包含 'file' 且序号为偶数的文件
    def my_filter(e):
        if 'file' not in e.name: return False
        import re
        res = re.search(r'file(\d+)', e.name)
        if res:
            num = int(res.group(1))
            return num % 2 == 0
        return False
        
    dw.include_file.custom(my_filter)
    
    files = list(dw.iter_files())
    file_names = {f.name for f in files}
    assert 'file2.py' in file_names
    assert 'file4.py' in file_names
    assert 'file6.py' in file_names
    assert 'file1.txt' not in file_names

def test_time_filter(temp_dir):
    """测试时间过滤"""
    import time
    from pyxllib.prog.xltime import XlTime
    
    # 获取当前时间
    now = time.time()
    
    dw = DirWalker(temp_dir, enter=True, select=True)
    # 选中在当前时间之前修改的文件（即所有文件）
    dw.include_file.match_mtime(max_time=now + 100)
    
    files = list(dw.iter_files())
    assert len(files) > 0
    
    # 选中在未来修改的文件（应该没有）
    dw = DirWalker(temp_dir, enter=True, select=False)
    dw.include_file.match_mtime(min_time=now + 1000)
    files = list(dw.iter_files())
    assert len(files) == 0
