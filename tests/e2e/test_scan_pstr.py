
import sys
import os

# Add src to sys.path to ensure we can import pyxllib
src_path = r"d:\home\chenkunze\slns\pyxllib\src"
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pyxllib.file.scan import FilterFactory
from pyxllib.text.pstr import PStr
from loguru import logger

class MockDirEntry:
    def __init__(self, name, path, is_dir_val=False, is_file_val=True):
        self.name = name
        self.path = path
        self._is_dir = is_dir_val
        self._is_file = is_file_val

    def is_dir(self):
        return self._is_dir

    def is_file(self):
        return self._is_file
    
    def __repr__(self):
        return f"<DirEntry '{self.name}'>"

def test_match_name():
    logger.info("Testing match_name...")
    
    entries = [
        MockDirEntry("test.py", "/path/to/test.py"),
        MockDirEntry("data.csv", "/path/to/data.csv"),
        MockDirEntry("image.png", "/path/to/image.png"),
        MockDirEntry("test_data.txt", "/path/to/test_data.txt"),
        MockDirEntry("NODE_MODULES", "/path/to/NODE_MODULES"),
    ]
    
    # 1. Literal match
    pred = FilterFactory.match_name("test.py")
    matched = [e for e in entries if pred(e)]
    assert len(matched) == 1
    assert matched[0].name == "test.py"
    logger.success("Literal match passed")
    
    # 2. Regex match
    pred = FilterFactory.match_name(PStr.re(r"test.*"))
    matched = [e for e in entries if pred(e)]
    assert len(matched) == 2
    names = sorted([e.name for e in matched])
    assert names == ["test.py", "test_data.txt"]
    logger.success("Regex match passed")
    
    # 3. Glob match
    pred = FilterFactory.match_name(PStr.glob("*.png"))
    matched = [e for e in entries if pred(e)]
    assert len(matched) == 1
    assert matched[0].name == "image.png"
    logger.success("Glob match passed")

    # 4. List match
    pred = FilterFactory.match_name([".git", "data.csv", PStr.glob("*.png")])
    matched = [e for e in entries if pred(e)]
    assert len(matched) == 2
    names = sorted([e.name for e in matched])
    assert names == ["data.csv", "image.png"]
    logger.success("List match passed")

    # 5. Case-insensitive match
    pred = FilterFactory.match_name("node_modules", ignore_case=True)
    matched = [e for e in entries if pred(e)]
    assert len(matched) == 1
    assert matched[0].name == "NODE_MODULES"
    logger.success("Case-insensitive match passed")

    # 6. Re-wrapping PStr
    p1 = PStr.re("abc")
    assert not p1.ignore_case
    p2 = PStr(p1, ignore_case=True)
    assert p2.ignore_case
    assert p2.is_re
    assert p2.match("ABC")
    logger.success("PStr re-wrapping test passed")

def test_match_path():
    logger.info("Testing match_path...")
    
    entries = [
        MockDirEntry("file1.txt", "/data/logs/file1.txt"),
        MockDirEntry("file2.txt", "/data/users/file2.txt"),
        MockDirEntry("file3.txt", "/var/logs/file3.txt"),
    ]
    
    # 1. Regex match
    pred = FilterFactory.match_path(PStr.re(r".*/logs/.*"))
    matched = [e for e in entries if pred(e)]
    assert len(matched) == 2
    paths = sorted([e.path for e in matched])
    assert paths == ["/data/logs/file1.txt", "/var/logs/file3.txt"]
    logger.success("Regex path match passed")

    # 2. List path match
    pred = FilterFactory.match_path(["/data/users/file2.txt", PStr.re(r"/var/.*")])
    matched = [e for e in entries if pred(e)]
    assert len(matched) == 2
    paths = sorted([e.path for e in matched])
    assert paths == ["/data/users/file2.txt", "/var/logs/file3.txt"]
    logger.success("List path match passed")

if __name__ == "__main__":
    test_match_name()
    test_match_path()
