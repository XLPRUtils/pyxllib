"""
walker.py - Directory & File Walker Utility

基于 os.scandir 实现的高性能目录遍历工具。
它支持灵活的规则配置（允许/拒绝、进入/跳过），支持链式调用，并且可以方便地进行文件打包。
解决了原生 os.walk 在复杂过滤逻辑下的低效和代码冗长问题。

核心用法示例：
>>> dw = DirWalker('.', enter=True)
>>> dw.skip_dir.match_name(['.*', '__pycache__'])  # 跳过隐藏目录和缓存目录
>>> dw.include_file.match_ext(['.py', '.txt'])    # 只选取 py 和 txt 文件
>>> for entry in dw.iter_files():
...     print(entry.path)
"""

import os
from typing import List, Tuple, Callable
import zipfile

from loguru import logger

from pyxllib.text.pstr import PStr
from pyxllib.prog.xltime import XlTime

Predicate = Callable[[os.DirEntry], bool]


class FilterFactory:
    """
    用于生成各种常用的 Predicate 函数。

    虽然这里的 filter 一般是用于生成 bool 判断进行过滤，
    但其实也可以作为一个回调、捕捉器使用，捕捉特定目标的 entry，输出日志进行检查等。

    注意这里的设计，每个方法都是返回一个函数，而不是直接返回判断结果，这是为了效率考虑。
    例如要遍历 1 万个文件执行 match_name 规则，这 1 万个文件就不用重复进行 PStr 的初始化预处理过程了。
    """

    @classmethod
    def custom(cls, func: Predicate):
        """添加自定义的判断函数

        :param func: 自定义的判断函数，接收 os.DirEntry 返回 bool
        :return: 传入的函数本身
        """
        return func

    @classmethod
    def is_file(cls):
        """判断是否为文件

        :return: 判断函数
        """
        return lambda e: e.is_file()

    @classmethod
    def is_dir(cls):
        """判断是否为目录

        :return: 判断函数
        """
        return lambda e: e.is_dir()

    @classmethod
    def match_name(cls, patterns, ignore_case=False):
        """根据文件名进行匹配

        :param patterns: 匹配模式。
            - str: 单个匹配模式。
            - list|tuple: 多个匹配模式。
        :param bool ignore_case: 是否忽略大小写，默认为 False
        :return: 判断函数

        >>> f = FilterFactory.match_name('*.py')
        """
        if isinstance(patterns, str):
            patterns = [patterns]

        processed_patterns = [PStr.auto(p, ignore_case=ignore_case) for p in patterns]
        return lambda e: any(p.match(e.name) for p in processed_patterns)

    @classmethod
    def match_path(cls, patterns, ignore_case=False):
        """根据文件路径进行匹配

        :param patterns: 匹配模式。
            - str: 单个匹配模式。
            - list|tuple: 多个匹配模式。
        :param bool ignore_case: 是否忽略大小写，默认为 False
        :return: 判断函数
        """
        if isinstance(patterns, str):
            patterns = [patterns]

        processed_patterns = [PStr.auto(p, ignore_case=ignore_case) for p in patterns]
        return lambda e: any(p.match(e.path) for p in processed_patterns)

    @classmethod
    def match_ext(cls, extensions, ignore_case=True):
        """匹配文件扩展名

        :param extensions: 扩展名。
            - str: 单个扩展名，如 '.py' 或 'jpg'。
            - list|tuple: 多个扩展名。
        :param bool ignore_case: 是否忽略大小写，默认 True
        :return: 判断函数
        """
        if isinstance(extensions, str):
            extensions = [extensions]

        # 预处理：1.确保带点 2.根据忽略大小写转换
        processed = []
        for ext in extensions:
            if not ext.startswith('.'):
                ext = '.' + ext
            processed.append(ext.lower() if ignore_case else ext)
        # 以此转为tuple，endswith支持传入tuple
        processed = tuple(processed)

        def _check(e):
            name = e.name
            if ignore_case:
                name = name.lower()
            return name.endswith(processed)

        return _check

    @classmethod
    def match_prefix(cls, prefixes, ignore_case=False):
        """匹配文件名前缀

        :param prefixes: 前缀字符串或列表
        :param bool ignore_case: 是否忽略大小写
        :return: 判断函数
        """
        if isinstance(prefixes, str):
            prefixes = [prefixes]

        processed = tuple(p.lower() if ignore_case else p for p in prefixes)

        def _check(e):
            name = e.name
            if ignore_case:
                name = name.lower()
            return name.startswith(processed)

        return _check

    @classmethod
    def match_suffix(cls, suffixes, ignore_case=False):
        """匹配文件名后缀 (比 match_ext 更泛用，不局限于扩展名)

        例如：匹配 '_backup.tar.gz' 结尾的文件

        :param suffixes: 后缀字符串或列表
        :param bool ignore_case: 是否忽略大小写
        :return: 判断函数
        """
        if isinstance(suffixes, str):
            suffixes = [suffixes]

        processed = tuple(s.lower() if ignore_case else s for s in suffixes)

        def _check(e):
            name = e.name
            if ignore_case:
                name = name.lower()
            return name.endswith(processed)

        return _check

    @classmethod
    def match_size(cls, min_size=0, max_size=None):
        """匹配文件大小 (字节)

        注意：这会触发 stat() 调用，比单纯检查文件名稍微耗时一点点。

        :param int min_size: 最小字节数
        :param int|None max_size: 最大字节数 (None表示不限制上限)
        :return: 判断函数
        """
        return lambda e: (min_size <= e.stat().st_size) and (max_size is None or e.stat().st_size <= max_size)

    @classmethod
    def match_mtime(cls, min_time=None, max_time=None):
        """匹配修改时间 (Modification Time)

        :param min_time: 最早时间 (时间戳、datetime对象或 'YYYY-MM-DD' 字符串)
        :param max_time: 最晚时间
        :return: 判断函数
        """
        min_ts = XlTime(min_time)
        max_ts = XlTime(max_time)

        def _check(e):
            mtime = e.stat().st_mtime
            if min_ts is not None and mtime < min_ts:
                return False
            if max_ts is not None and mtime > max_ts:
                return False
            return True

        return _check

    @classmethod
    def match_ctime(cls, min_time=None, max_time=None):
        """匹配创建时间 (Creation Time)

        注意：在 Unix 上 ctime 可能是元数据变更时间，Windows 上是创建时间。

        :param min_time: 最早时间
        :param max_time: 最晚时间
        :return: 判断函数
        """
        min_ts = XlTime(min_time)
        max_ts = XlTime(max_time)

        def _check(e):
            ctime = e.stat().st_ctime
            if min_ts is not None and ctime < min_ts:
                return False
            if max_ts is not None and ctime > max_ts:
                return False
            return True

        return _check

    # ==========================
    # 预设的常用文件扩展名集合
    # ==========================
    EXT_IMAGES = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif',
        '.svg', '.ico', '.psd', '.heic', '.raw'
    }

    EXT_VIDEOS = {
        '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v',
        '.mpeg', '.mpg', '.3gp'
    }

    EXT_AUDIOS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}

    EXT_DOCS = {
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt',
        '.md', '.csv', '.json', '.xml'
    }

    EXT_CODES = {
        '.py', '.java', '.c', '.cpp', '.h', '.js', '.ts', '.html', '.css',
        '.go', '.rs', '.php', '.sh', '.bat'
    }

    # ==========================
    # 封装方法
    # ==========================

    @classmethod
    def is_image(cls):
        """匹配常见图片文件"""
        return cls.match_ext(cls.EXT_IMAGES, ignore_case=True)

    @classmethod
    def is_video(cls):
        """匹配常见视频文件"""
        return cls.match_ext(cls.EXT_VIDEOS, ignore_case=True)

    @classmethod
    def is_audio(cls):
        """匹配常见音频文件"""
        return cls.match_ext(cls.EXT_AUDIOS, ignore_case=True)

    @classmethod
    def is_doc(cls):
        """匹配常见文档文件"""
        return cls.match_ext(cls.EXT_DOCS, ignore_case=True)

    @classmethod
    def is_code(cls):
        """匹配常见代码源文件"""
        return cls.match_ext(cls.EXT_CODES, ignore_case=True)


class RuleBuilder:
    """
    规则构建器。

    中间层，衔接 FilterRule 和 PathSelector。
    """

    def __init__(self, parent, rules, target, base_condition=None):
        """
        :param parent: 父级对象 (DirWalker)
        :param list rules: 需要往哪里添加规则数组
        :param bool target: 规则的目标动作，True 表示允许，False 表示拒绝
        :param base_condition: 基础前置条件，例如 lambda e: e.is_file()
        """
        self.parent = parent
        self.rules = rules
        self.target = target
        self.base_condition = base_condition

    def __getattr__(self, method):
        # 获取原始的规则生成工厂 (如 match_name)
        rule_factory = getattr(FilterFactory, method)

        def wrapper(*args, **kwargs):
            # 1. 生成原始的判断函数
            predicate = rule_factory(*args, **kwargs)

            # 2. 如果存在基础条件（如必须是文件），则进行逻辑“与”组合
            if self.base_condition:
                original_predicate = predicate
                base_condition = self.base_condition

                # 新的复合判断函数
                def composite_predicate(e: os.DirEntry) -> bool:
                    return base_condition(e) and original_predicate(e)

                final_predicate = composite_predicate
            else:
                final_predicate = predicate

            # 3. 添加规则
            self.rules.append((final_predicate, self.target))

            # 返回 PathSelector，支持连续链式调用
            return self.parent

        return wrapper


class DirWalker:
    """目录检索遍历工具。"""

    def __init__(self, root, enter=False, select=False):
        """
        :param str root: 根目录
        :param bool enter: 默认是否进入子目录
        :param bool select: 默认是否选择目标
            根据不同需求场景，可能初始 False，即通过后续白名单机制添加目标（白名单也可以再混合黑名单）；
            也可以初始 True，后续通过黑名单排除。
        """
        self.root = root

        # 配置初始状态：默认初始状态空，不进入子目录，不选任何文件/目录
        self.default_enter = enter
        self.default_select = select

        # 最终组装的规则
        # 规则有先后顺序，可以不断增加允许、拒绝规则
        # 一个目标如果前面被允许，后面依然可能被拒绝，被拒绝的也可以后面又被允许
        self.enter_rules: List[Tuple[Predicate, bool]] = []
        self.select_rules: List[Tuple[Predicate, bool]] = []

    def __1_配置规则(self):
        """ 用于在 IDE 大纲中标记规则配置部分 """
        pass

    @property
    def enter_dir(self) -> RuleBuilder:
        """进入目录规则"""
        return RuleBuilder(self, self.enter_rules, True)

    @property
    def skip_dir(self) -> RuleBuilder:
        """跳过目录规则"""
        return RuleBuilder(self, self.enter_rules, False)

    @property
    def include(self) -> RuleBuilder:
        """包含规则"""
        return RuleBuilder(self, self.select_rules, True)

    @property
    def exclude(self) -> RuleBuilder:
        """排除规则"""
        return RuleBuilder(self, self.select_rules, False)

    @property
    def include_file(self) -> RuleBuilder:
        """选中：必须是文件 AND 满足后续条件"""
        return RuleBuilder(self, self.select_rules, True, base_condition=lambda e: e.is_file())

    @property
    def exclude_file(self) -> RuleBuilder:
        """排除：必须是文件 AND 满足后续条件"""
        return RuleBuilder(self, self.select_rules, False, base_condition=lambda e: e.is_file())

    @property
    def include_dir(self) -> RuleBuilder:
        """选中：必须是目录 AND 满足后续条件"""
        return RuleBuilder(self, self.select_rules, True, base_condition=lambda e: e.is_dir())

    @property
    def exclude_dir(self) -> RuleBuilder:
        """排除：必须是目录 AND 满足后续条件"""
        return RuleBuilder(self, self.select_rules, False, base_condition=lambda e: e.is_dir())

    def __2_判断逻辑(self):
        """ 用于在 IDE 大纲中标记逻辑检查部分 """
        pass

    def should_enter(self, entry: os.DirEntry) -> bool:
        """是否需要进入目标目录

        :param os.DirEntry entry: 目录项
        :return bool: 是否进入
        """
        return self._check(entry, self.enter_rules, default=self.default_enter)

    def should_select(self, entry: os.DirEntry) -> bool:
        """是否需要选择目标文件/目录

        :param os.DirEntry entry: 目录项
        :return bool: 是否选中
        """
        return self._check(entry, self.select_rules, default=self.default_select)

    def _check(self, entry: os.DirEntry, rules: List[Tuple[Predicate, bool]], default: bool) -> bool:
        """检查目标是否符合规则

        :param entry: 目标文件/目录
        :param rules: 规则数组
        :param bool default: 默认动作
        :return bool: 是否符合规则
        """
        decision = default
        for predicate, target_action in rules:
            if decision == target_action:
                continue
            if predicate(entry):
                decision = target_action
        return decision

    def __3_迭代遍历(self):
        """ 用于在 IDE 大纲中标记遍历功能部分 """
        pass

    def iter(self, path=None):
        """遍历目录

        :param str path: 当前遍历的路径，第一次调用不需要传，默认为 self.root
        :return: 生成器，产生 os.DirEntry 对象
        """
        # 1. 确定当前遍历路径
        cur_path = path if path is not None else self.root

        # 用于暂存需要稍后进入的子目录，保证当前层级优先遍历
        subdirs_to_visit = []

        try:
            with os.scandir(cur_path) as it:
                for entry in it:
                    # 2. 检查是否选中该目标（优先 yield 当前层级的结果）
                    if self.should_select(entry):
                        yield entry

                    # 3. 如果是目录且符合进入条件，先存起来，不要立即递归
                    if entry.is_dir() and self.should_enter(entry):
                        subdirs_to_visit.append(entry.path)
        except Exception as e:
            logger.error(f'访问 {cur_path} 失败: {e}')
            return

        # 4. 当前层级遍历完后，再递归进入子目录
        for subdir_path in subdirs_to_visit:
            yield from self.iter(subdir_path)

    def iter_files(self, path=None):
        """只迭代文件

        :param str path: 遍历路径
        :return: 生成器
        """
        for entry in self.iter(path):
            if entry.is_file():
                yield entry

    def iter_dirs(self, path=None):
        """只迭代目录

        :param str path: 遍历路径
        :return: 生成器
        """
        for entry in self.iter(path):
            if entry.is_dir():
                yield entry

    def walk(self, path=None):
        """类似 os.walk 的遍历方式，但应用了 ScanDir 的过滤规则。

        :param str path: 遍历路径
        :return: Generator[Tuple[str, List[str], List[str]]]
                 返回三元组 (当前目录路径, 文件名列表, 子目录名列表)
        """
        # 1. 确定当前遍历路径
        cur_path = path if path is not None else self.root

        # 用于存储当前层级的结果
        files = []
        dirs = []

        # 用于存储需要递归进入的路径
        subdirs_to_visit = []

        try:
            with os.scandir(cur_path) as it:
                for entry in it:
                    if entry.is_dir():
                        if self.should_select(entry):
                            dirs.append(entry.name)

                        if self.should_enter(entry):
                            subdirs_to_visit.append(entry.path)

                    elif entry.is_file():
                        if self.should_select(entry):
                            files.append(entry.name)

        except Exception as e:
            logger.error(f'访问 {cur_path} 失败: {e}')
            return

        # 2. Yield 当前层级结果
        yield cur_path, files, dirs

        # 3. 递归遍历子目录
        for subdir_path in subdirs_to_visit:
            yield from self.walk(subdir_path)

    def __4_实用工具(self):
        """ 用于在 IDE 大纲中标记实用工具部分 """
        pass

    def pack_zip(self, save_path: str):
        """将选中的文件或目录打包成 zip 文件

        :param str save_path: zip 文件的保存路径
        """
        # 确保保存路径的目录存在
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        count = 0
        with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 开启递归扫描写入
            count = self._pack_recursive(zf, self.root)

        logger.info(f'已将 {count} 个项目打包至 {save_path}')

    def _pack_recursive(self, zf: zipfile.ZipFile, current_path: str) -> int:
        """内部递归打包函数

        :param zf: zipfile.ZipFile 对象
        :param str current_path: 当前路径
        :return int: 打包的项目数量
        """
        count = 0
        try:
            with os.scandir(current_path) as it:
                for entry in it:
                    # 1. 检查是否选中
                    if self.should_select(entry):
                        if entry.is_dir():
                            # 【核心逻辑】：如果目录被选中，直接打包整个目录，并跳过对该目录的进一步 Filter 递归
                            count += self._write_whole_dir_to_zip(zf, entry.path)
                            continue
                        else:
                            # 如果是文件被选中，直接写入
                            arcname = os.path.relpath(entry.path, self.root)
                            zf.write(entry.path, arcname)
                            count += 1

                    # 2. 如果未被选中（或者是文件已处理），检查是否需要递归进入寻找深层目标
                    if entry.is_dir() and self.should_enter(entry):
                        count += self._pack_recursive(zf, entry.path)

        except Exception as e:
            logger.error(f'打包 {current_path} 失败: {e}')

        return count

    def _write_whole_dir_to_zip(self, zf: zipfile.ZipFile, dir_path: str) -> int:
        """辅助函数：无视规则，暴力递归写入某个目录下的所有内容

        :param zf: zipfile.ZipFile 对象
        :param str dir_path: 目录路径
        :return int: 写入的文件数量

        todo 可能会有需要打包空目录，只要目录不要文件的情况，这个等以后遇到可以再考虑怎么设计优化
        """
        count = 0
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                abs_path = os.path.join(root, file)
                # 计算相对路径，保持目录结构
                arcname = os.path.relpath(abs_path, self.root)
                zf.write(abs_path, arcname)
                count += 1

        return count


if __name__ == '__main__':
    dw = DirWalker(r'D:\home\chenkunze\slns\pyxllib', True)
    dw.skip_dir.match_name(['.*', '__pycache__'])  # 不进入 .git、.venv 这类子目录
    dw.include.is_file()
    dw.exclude_file.match_ext('*.pyc')

    dw.pack_zip(r'D:\home\chenkunze\slns\pyxllib.zip')
