#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2018/07/12 09:32


"""
任何模块代码的第一个字符串用来写文档注释

因为util下的debuglib、textlib、filelib的功能拆分并不是特别精确，所以一般不对外开放接口
而是由util作为统一的一个大工具箱接口对外开放
"""

import filecmp
import shutil
import sys, json
import textwrap
from os.path import getmtime
from os.path import join as pathjoin
from collections import OrderedDict, Counter, defaultdict

from bs4 import BeautifulSoup

from pyxllib.debug import *
from pyxllib.image import *
from pyxllib.util.filelib import *


def ________B_数据结构________():
    pass


def dict__sub__(d1, d2):
    """在d1中删除d2存在Keys的方法"""
    d = {}
    for k in [k for k in d1 if k not in d2]:
        d[k] = d1[k]
    return d


def ________C_文本处理________():
    pass


def read_from_ubuntu(url):
    """从paste.ubuntu.com获取数据"""
    if isinstance(url, int):  # 允许输入一个数字ID来获取网页内容
        url = 'https://paste.ubuntu.com/' + str(url) + '/'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    content = soup.find_all(name='div', attrs={'class': 'paste'})[2]
    return content.get_text()


def 从部门Confluence获取数据(url):
    cookies = getattr(从部门Confluence获取数据, 'cookies', None)
    if cookies:  # 如果存储了cookies，尝试使用
        r = requests.get(url, cookies=cookies)
    if not cookies or not r.cookies._cookies:  # 如果没有cookies或者读取失败（使用_cookies是否为空判断登陆是否成功），则重新登陆获取cookies
        r = requests.get('http://doc.klxuexi.org/login.action', auth=('chenkz', 'klxx11235813'))
        cookies = r.cookies
        r = requests.get(url, cookies=cookies)
    从部门Confluence获取数据.cookies = cookies
    return r.text


def ReadFromUrl(url):
    """从url获得文本数据

    对特殊的网页有专门优化
    """
    if 'paste.ubuntu.com' in url:
        return read_from_ubuntu(url)
    elif url.startswith('http://doc.klxuexi.org'):  # 这个写法主要是针对工时统计表
        # TODO：如果是工时表，应该做成DataFrame结构数据
        text = 从部门Confluence获取数据(url)
        soup = BeautifulSoup(text, 'lxml')  # 解析网页得到soup对象
        content = soup.find_all(name='div', attrs={'id': 'main-content'})[0]  # 提取conteng部分的内容
        return content.get_text('\t')  # 仅获得文本信息，每项用\t隔开
    else:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'lxml')
        return soup.get_text()


class CWord:
    def __init__(self, arg1=None, *, visible=None):
        """visible默认为None，代表不对现有窗口显示情况做更改
                如果设为bool值true或false，
        """
        self.__dict__['app'] = win32.gencache.EnsureDispatch('Word.Application')
        if isinstance(visible, bool): self.app.Visible = visible

        if isinstance(arg1, str):  # 输入的是一个文件名则打开
            file = arg1
            self.app.Documents.Open(file)  # 打开文件
            # self.__dict__ = app.Documents(Path(file).name).__dict__
            self.__dict__['doc'] = self.app.Documents(File(file).name)  # 存储到doc成员变量
        else:  # 如果输入参数不合法，新建一个空word
            self.app.Documents.Add()
            self.__dict__['doc'] = self.app.ActiveDocument

    def CheckAttr(self):
        """ 输出 self.app.Documents 的成员 """
        showdir(self.doc)

    def CntPage(self):
        """统计word页数"""
        return self.doc.ActiveWindow.Panes(1).Pages.Count

    def GetParagraphs(self):
        """以 yield 形式获得每个段落文本内容"""
        cntPar = self.doc.Paragraphs.Count
        for i in range(1, cntPar + 1):
            yield str(self.doc.Paragraphs(i))

    def Close(self):
        """ 关闭word文件 """
        self.doc.Close(False)
        # self.app.Quit()

    def __getattr__(self, item):
        """ 智能获取成员 """
        if item in self.__dict__:
            return self.__dict__[item]
        elif self.doc:
            return getattr(self.doc, item)
        else:
            return None

    def __setattr__(self, key, value):
        """ 智能设置成员 """
        if key in self.__dict__:
            self.__dict__[key] = value
        elif self.doc:
            setattr(self.doc, key, value)
        else:
            pass

    def __str__(self):
        """ 获得word文件的全文纯文本 """
        return str(self.doc.Content)


class CExcel:
    def __init__(self, arg1=None, *, visible=None):
        """visible默认为None，代表不对现有窗口显示情况做更改
                如果设为bool值true或false，
        """
        import win32com.client as win32
        import win32com

        self.__dict__['app'] = win32.gencache.EnsureDispatch('Excel.Application')
        if isinstance(visible, bool): self.app.Visible = visible

        if isinstance(arg1, str):  # 输入的是一个文件名则打开
            file = arg1
            self.app.Workbooks.Open(file)  # 打开文件
            # self.__dict__ = app.Documents(Path(file).name).__dict__
            self.__dict__['wb'] = self.app.Workbooks(File(file).name)  # 存储到doc成员变量
        else:  # 如果输入参数不合法，新建一个空word
            # self.app.Workbooks.Add()
            # self.__dict__['wb'] = self.app.ActiveWorkbook
            self.__dict__['wb'] = self.app.Workbooks.Add()
        self.__dict__['st'] = self.wb.ActiveSheet

    def WriteCell(self, row=0, col=0, val='', *, fontColor=None):
        """往当前激活的st增加数据

        row可以写0，表示往新的一行加数据
        col可以列名，智能进行列查找

        color可以设置颜色
        """
        # 空表还有bug
        if row == 0: row = self.st.UsedRange.Count + 1
        dprint(row)
        # if col == 0: row = self.st.UsedRange.Count + 1

        self.st.Cells(row, col).Value = str(val)

    def GetVal(self, row, col):
        """获得指定单元格的值"""
        return self.st.Cells(row, col).Text


def EnsureContent(ob=None, encoding='utf8'):
    """
    未输入ob参数时，自动从控制台获取文本

    输入的如果是字符串内容，则返回字符串内容
    输入的如果是文件名，则读取文件的内容返回
    输入的如果是url，则返回html爬取内容
    """
    # TODO: 如果输入的是一个文件指针，也能调用f.read()返回所有内容
    # TODO: 增加鲁棒性判断，如果输入的不是字符串类型也要有出错判断

    if ob is None:
        return sys.stdin.read()  # 注意输入是按 Ctrl + D 结束
    elif ob.find('\n') >= 0 or len(ob) > 200:  # 如果存在回车符，不用想了，直接认为是字符串
        return ob
    elif os.path.exists(ob):  # 如果存在这样的文件，那就读取文件内容
        if ob.endswith('.pdf'):  # 如果是pdf格式，则读取后转换为文本格式
            res = map(lambda page: page.getText('text'), fitz.open(ob))
            return '\n'.join(res)
        elif ob.endswith('.docx'):
            import textract
            text = textract.process(ob)
            return text.decode(encoding, errors='ignore')
        elif ob.endswith('.doc'):
            a = CWord(ob)
            s = str(a)
            a.Close()
            return s
        else:  # 其他按文本格式处理
            if ob.endswith(r'.tex'):
                encoding = 'gbk'  # TODO：强制转为gbk，这个后续要改
            with open(ob, 'r', errors='ignore', encoding=encoding) as f:
                # 默认编码是跟平台有关，比如windows是gbk
                return f.read()
    elif ob.startswith('http'):
        try:
            return ReadFromUrl(ob)
        except:
            # 读取失败则返回原内容
            return ob
    elif isinstance(ob, pd.DataFrame):
        # 还未开发
        pass
    else:
        # 判断不了的情况，也认为是字符串
        return ob


################################################################################
# 文 本 处 理
################################################################################


def PrintFullTable(df, *, 最后一列左对齐=False, columns=None):
    if isinstance(df, (list, tuple)):
        df = pd.DataFrame.from_records(df, columns=columns)

    if len(df) < 1:
        return
    """参考文档： http://pandas.pydata.org/pandas-docs/stable/options.html"""
    with pd.option_context('display.max_rows', None,  # 没有行数限制
                           'display.max_columns', None,  # 没有列数限制（超过列数会分行显示）
                           'display.width', None,  # 没有列宽限制
                           'display.max_colwidth', 10 ** 6,  # 单列宽度上限
                           # 'display.colheader_justify', 'left', # 列标题左对齐
                           'display.unicode.east_asian_width', True,  # 中文输出必备选项，用来控制正确的域宽
                           ):
        if 最后一列左对齐:  # 但是这里涉及中文的时候又会出错~~
            # df.iloc[:, -1] = (lambda s: s.str.ljust(s.str.len().max()))(df.iloc[:, -1]) # 最后一列左对齐
            def func(s):
                return s.str.ljust(s.str.len().max())

            df.iloc[:, -1] = func(df.iloc[:, -1])  # 最后一列左对齐
        print(df)
        # print(df.info())


def 重定向到浏览器显示(fileName=None):
    """第一次执行时，必须给一个参数，代表重定向的输出文件名

    第二次执行时，不要输入参数，此时会弹出chrome.exe显示已保存的所有输出内容
    """
    if fileName:
        重定向到浏览器显示.fileName = fileName
        重定向到浏览器显示.oldTarget = sys.stdout
        sys.stdout = open(fileName, 'w')
    else:  # 如果没写参数，则显示
        sys.stdout = 重定向到浏览器显示.oldTarget
        subprocess.run(['chrome.exe', 重定向到浏览器显示.fileName])


def regular_cut(old_str, pattern, flags=0):
    r"""返回第1个参数是新的new_str，去掉了被提取的元素
    返回第2个参数是提取出来的数据列表

    >>> regular_cut('abc123de4f', r'\d+')
    ('abcdef', ['123', '4'])
    >>> regular_cut('abc123de4f', r'c(\d+)')
    ('abde4f', ['123'])
    """
    new_str = re.sub(pattern, '', old_str, flags=flags)
    elements = re.findall(pattern, old_str, flags=flags)
    return new_str, elements


def research(pattern, string):
    """ .能匹配所有字符
        返回第一个匹配的字符串（的group(1)），结果会去掉左右空白
        如果找不到则返回空字符串
    """
    m = re.search(pattern, string, flags=re.DOTALL)
    return m.group(1).strip() if m else ''


def ________D_文件目录相关函数________():
    """"""
    pass


################################################################################
### 目录相关函数
################################################################################


def SetWkDir(wkDir=None, key=None):
    r"""用过的工作目录存在字典wkdirs {int:string}，原始目录用0索引，上一个目录用-1索引
    新设的目录可以添加自己的索引key

    SetWkDir(0)
    SetWkDir('word')
    SetWkDir(0)
    SetWkDir(-1)
    """
    wkDir = str(wkDir)
    SetWkDir.wkdirs = getattr(SetWkDir, 'wkdirs', {'0': os.getcwd(), '-1': os.getcwd()})
    wkdirs = SetWkDir.wkdirs
    lastWkDir = os.getcwd()

    if wkDir in wkdirs:
        os.chdir(wkdirs[wkDir])
    elif wkDir == '':  # 如果输入空字符串，则返回当前工作目录（这在使用os.path.dirname('a.tex')时是会发生的）
        return os.getcwd()
    else:
        os.chdir(wkDir)  # 切换到目标工作目录
        if key:  # 如果输入了key，则存储当前工作目录
            wkdirs[key] = wkDir

    wkdirs[-1] = lastWkDir
    return lastWkDir  # 返回切换前的目录


def SmartCopyFiles(files, inFolder, outFolder):
    """将files里的文件移到folder目录里，如果folder里已经存在对应文件则自动进行备份"""
    inFd = File(inFolder)
    outFd = File(outFolder)

    for file in files:
        inFile = inFd / file
        if not inFile.exists():  # 如果原目录里并不含有该文件则continue
            continue
        outFile = outFd / file
        if outFile.exists():
            if filecmp.cmp(inFile, outFile):
                continue  # 如果两个文件是相同的，不用操作，可以直接处理下一个
            else:
                File(outFile).backup()  # 如果不相同，则对outFile进行备份
        shutil.copy2(inFile, outFile)
        File(outFile).backup()  # 对拷贝过来的文件也提前做好备份


def MyMove(folder1, folder2):
    """将目录1里的文件复制到目录2，如果目录2已存在文件，则对其调用备份"""
    pass


def 多规则字符串筛选(列表, glob筛选=None, *, 正则筛选=None, 指定名筛选=None, 去除备份文件=False):
    """该函数主要供文件处理使用，其它字符串算法慎用"""
    if glob筛选:
        列表 = list(filter(lambda x: File(x).match(glob筛选), 列表))

    # 只挑选出满足正则条件的文件名（目录名）
    if 正则筛选:
        列表 = list(filter(lambda x: re.match(正则筛选, x, flags=re.IGNORECASE), 列表))

    if 指定名筛选:
        列表 = list(filter(lambda x: x in 指定名筛选, 列表))

    if 去除备份文件:
        列表 = list(filter(lambda x: not File(x).backup_time, 列表))

    return 列表


def 递归删除空目录(rootPath):
    fd = CBaseFolder(rootPath)
    for f in fd.Folders():
        d = CBaseFolder(pathjoin(rootPath, f))
        if d.大小():
            递归删除空目录(d.name)
        else:
            d.删除()


class CBaseFolder(object):
    def __init__(self, s='.'):
        """TODO：应该支持'a/*.eps'这种操作，指定目录的同时，也进行了文件筛选"""
        self.path = GetFullPathClass(s)
        self.name = str(self.path)
        self.files = self.Files()
        if os.path.exists(self.name):
            self.folderStats = os.stat(s)

    def Files(self, glob筛选=None, *, 正则筛选=None, 指定名筛选=None, 去除备份文件=False):
        """注意：正则规则和一般的文件匹配规则是不一样的！
        比如glob中的'*.log'，在正则中应该谢伟'.*[.]log'

        使用举例：
        f.Files('*.tex', 正则筛选=r'ch[a-zA-Z]*[.]tex', 指定名筛选 = ('chePre.tex', 'cheRev.tex'))
        """
        if not self:
            return list()

        ls = os.listdir(self.name)
        ls = list(filter(self.IsFile, ls))
        files = 多规则字符串筛选(ls, glob筛选, 正则筛选=正则筛选, 指定名筛选=指定名筛选, 去除备份文件=去除备份文件)
        files = natural_sort(files)
        return files

    def 递归获取文件(self, 过滤器=lambda x: True):
        """过滤器输入一个函数，文件名要满足指定条件才会被提取"""
        for f in list(filter(过滤器, self.files)): yield pathjoin(self.path, f)
        for folder in self.Folders():
            try:  # 有些系统目录会读取不了
                fd = CBaseFolder(pathjoin(self.name, folder))
                for f in fd.递归获取文件(过滤器): yield f
            except:
                continue

    def Folders(self, glob筛选=None, *, 正则筛选=None, 指定名筛选=None, 去除备份文件=False):
        ls = os.listdir(self.name)
        ls = list(filter(self.IsFolder, ls))
        return 多规则字符串筛选(ls, glob筛选, 正则筛选=正则筛选, 指定名筛选=指定名筛选, 去除备份文件=去除备份文件)

    def IsFile(self, f):
        return os.path.isfile(os.path.join(self.name, f))

    def IsFolder(self, f):
        if f in ('$RECYCLE.BIN', 'Recovery', 'System Volume Information'):  # 过滤掉无需访问的目录
            return False
        else:
            return os.path.isdir(os.path.join(self.name, f))

    def FilesRename(self, origin, target, *, 目标目录=None):
        """使用正则规则匹配文件名，并重命名"""
        files = self.Files(正则筛选=origin)
        if not 目标目录:  # 如果没有设置目标目录，则以该类所在目录为准
            目标目录 = self.name
        for fn in files:
            f = File(os.path.join(self.name, fn))
            目标名称 = re.sub(origin, target, fn, flags=re.IGNORECASE)
            f.rename(os.path.join(目标目录, 目标名称))

    def 递归输出文件列表(self, *, 当前层级=0, 控制宽度=None):
        """占秋意见：可以根据扩展名简化输出内容"""
        fileString = ', '.join(self.files)
        if isinstance(控制宽度, int):
            fileString = textwrap.shorten(fileString, 控制宽度, placeholder='...')
        s = '{}【{}】（{}）: {}'.format('\t' * 当前层级, self.path.stem, len(self.files), fileString)
        print(s)
        for folder in self.Folders():
            try:  # 有些系统目录会读取不了
                fd = CBaseFolder(pathjoin(self.name, folder))
                fd.递归输出文件列表(当前层级=当前层级 + 1, 控制宽度=控制宽度)
            except:
                continue

    def 大小(self):
        return File(self.name).size

    def 删除(self):
        shutil.rmtree(self.name, ignore_errors=True)

    def __bool__(self):
        return os.path.isdir(str(self.path))

    def __str__(self):
        return self.name


def 自定义正则规则转为标准正则表达式(s):
    s = s.replace('？', r'[\u4e00-\u9fa5]')  # 中文问号匹配任意一个中文字符
    return s


def 文件搜索匹配(源目录, 自定义正则规则, *, 目标类型=('文件', '目录')):
    """ 目标类型=('文件','目录') """
    匹配文件 = list()
    源目录前缀长度 = len(源目录)
    正则规则 = 自定义正则规则转为标准正则表达式(自定义正则规则)
    所有目录 = tuple(os.walk(源目录))
    for 当前目录名, 包含目录, 包含文件 in 所有目录:
        parts = File(当前目录名).parts
        if '.git' in parts or '$RECYCLE.BIN' in parts:  # 去掉'.git'这个备份目录，'$RECYCLE.BIN'这个不知啥鬼目录
            continue
        相对目录 = 当前目录名[源目录前缀长度 + 1:]
        if '目录' in 目标类型:
            if re.search(正则规则, 相对目录):
                匹配文件.append(相对目录)
        if '文件' in 目标类型:
            for 文件 in 包含文件:
                相对路径 = os.path.join(相对目录, 文件)
                if re.search(正则规则, 相对路径):
                    匹配文件.append(相对路径)
    return natural_sort(匹配文件)


def 文件复制(源目录, 自定义正则规则, 目标目录, 新正则名称, *, 目标类型=('文件',)):
    """ 目标类型=('文件','目录') """
    匹配文件 = 文件搜索匹配(源目录, 自定义正则规则, 目标类型=目标类型)
    # 获得匹配文件后，需要从后往前改，否则产生连锁反应会索引不到


def 文件重命名(源目录, 自定义正则规则, 新正则名称, *, 目标类型=('文件',), 目标目录=None, 调试=True, 输出=True, 覆盖操作=False):
    """因为这个操作风险非常大，所以默认情况下是调试模式，必须手动指定进入非调试模式才会进行实际工作

    使用示例：文件重命名(r'D:\2017LaTeX\B暑假教材\高数\高一教师版 - 测试\figs - 副本',
                            r'^(.*?)-eps-converted-to[.]png', r'\1.png', 调试=False)
    """
    ls = list()
    匹配文件 = 文件搜索匹配(源目录, 自定义正则规则, 目标类型=目标类型)
    正则规则 = 自定义正则规则转为标准正则表达式(自定义正则规则)
    if not 目标目录:
        目标目录 = 源目录
    for f in reversed(匹配文件):
        f2 = re.sub(正则规则, 新正则名称, f)
        ls.append([f, f2])
        if not 调试:
            targetName = os.path.join(目标目录, f2)
            f3 = File(targetName)
            if f3:
                print('文件已存在：', f3.name)
                if 覆盖操作:
                    f3.delete()
                    os.rename(os.path.join(源目录, f), targetName)
            else:
                os.rename(os.path.join(源目录, f), targetName)

    df = pd.DataFrame.from_records(ls, columns=('原文件名', '目标文件名'))
    if 输出:
        PrintFullTable(df)
    return df


# def 目录下查找文本(目录, 文件名筛选, 目标文本):
#     ls = 文件搜索匹配(目录, 文件名筛选, 目标类型=('文件',))
#     ls = list(filter(lambda x: Path(x).backup_time == '', ls)) # 去除备份文件
#     ls = natural_sort(ls)
#     文件名 = list()
#     出现次数 = list()
#     行号 = list()
#     for i, fileName in enumerate(ls):
#         cl = ContentLine(pathjoin(目录, fileName))
#         lines = cl.regular_search(目标文本)
#         if lines:
#             文件名.append(fileName)
#             出现次数.append(len(lines))
#             行号.append(str(lines))
#
#     pf = pd.DataFrame({'文件名': 文件名, '出现次数': 出现次数, '行号': 行号}, columns=['文件名', '出现次数', '行号'])
#     pf.sort_values(by=['出现次数'], ascending=False, inplace=True)
#     PrintFullTable(pf, 最后一列左对齐=True)
#     return pf


def 目录下查找文本(目录, 文件名筛选, 目标文本, *, 模式='表格'):
    """
    表格模式：统计每个文件中出现的总次数
    行文本模式：显示所有匹配的行文本
    """
    ls = 文件搜索匹配(目录, 文件名筛选, 目标类型=('文件',))
    ls = list(filter(lambda x: File(x).backup_time == '', ls))  # 去除备份文件
    ls = natural_sort(ls)
    if 模式 == '表格':
        table = list()
        for i, fileName in enumerate(ls):
            cl = ContentLine(pathjoin(目录, fileName))
            lines = cl.regular_search(目标文本)
            if lines:
                table.append((fileName, len(lines), refine_digits_set(lines)))

        pf = pd.DataFrame.from_records(table, columns=('文件名', '出现次数', '行号'))
        for i in range(len(pf)):
            pf['出现次数'][i] = int(pf['出现次数'][i])
        pf.sort_values(by=['出现次数'], ascending=False, inplace=True)
        PrintFullTable(pf, 最后一列左对齐=True)
        return pf
    elif 模式 == '行文本':
        for i, fileName in enumerate(ls):
            cl = ContentLine(pathjoin(目录, fileName))
            lines = cl.regular_search(目标文本)
            if lines:
                print()
                # print(fileName) # 不输出根目录版本
                print(pathjoin(目录, fileName))  # 输出根目录版本
                print(cl.lines_content(lines))
    else:
        raise TypeError


def 目录下统计单词出现频数(目录, 文件名筛选, 目标文本=r'(\\?[a-zA-Z]+)(?![a-zA-Z])'):
    """默认会找所有单词，以及tex命令"""
    ls = 文件搜索匹配(目录, 文件名筛选, 目标类型=('文件',))
    s = list()
    for fileName in [f for f in ls if not File(f).backup_time]:  # 去除备份文件
        c = File(pathjoin(目录, fileName)).read()
        s.append(c)
    s = '\n'.join(s)

    # ls = re.findall(r'(?<=\\)([a-zA-Z]+)(?![a-zA-Z])', s) # 统计tex命令数量
    # ls = re.findall(r'([a-zA-Z]+)(?![a-zA-Z])', s)  # 统计单词次数
    # ls = re.findall(r'(\\?[a-zA-Z]+)(?![a-zA-Z])', s)  # tex和普通单词综合性搜索
    ls = re.findall(目标文本, s)
    d = OrderedDict(sorted(Counter(ls).items(), key=lambda t: -t[1]))
    pf = pd.DataFrame({'关键字': list(d.keys()), '出现次数': list(d.values())}, columns=['关键字', '出现次数'])
    PrintFullTable(pf)
    return pf


def GetFullPathClass(s):
    """如果输入的是相对路径，会解析为绝对路径"""
    # p = Path(s).resolve() # 奕本的电脑这句话运行不了
    p = File(s)
    if not s.startswith('\\') and not p.drive:
        p = File.cwd() / p
    return p


def ________E_图像处理________():
    pass


################################################################################
### 图 像 处 理 操 作
################################################################################
import PIL
import PIL.ExifTags
from PIL import Image


def 图像实际视图(img):
    """Image.open读取图片时，是手机严格正放时拍到的图片效果，
    但手机拍照时是会记录旋转位置的，即可以判断是物理空间中，实际朝上、朝下的方向，
    从而识别出正拍（代号1），顺时针旋转90度拍摄（代号8），顺时针180度拍摄（代号3）,顺时针270度拍摄（代号6）。
    windows上的图片查阅软件能识别方向代号后正确摆放；
    为了让python处理图片的时候能增加这个属性的考虑，这个函数能修正识别角度返回新的图片。
    """
    exif_data = img._getexif()
    if exif_data:
        exif = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in exif_data.items()
            if k in PIL.ExifTags.TAGS
        }
        方向 = exif['Orientation']
        if 方向 == 8:
            img = img.transpose(PIL.Image.ROTATE_90)
        elif 方向 == 3:
            img = img.transpose(PIL.Image.ROTATE_180)
        elif 方向 == 6:
            img = img.transpose(PIL.Image.ROTATE_270)
    return img


def 查看图片的Exif信息(img):
    exif_data = img._getexif()
    if exif_data:
        exif = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in exif_data.items()
            if k in PIL.ExifTags.TAGS
        }
    else:
        exif = None
    return exif


def 缩放目录下所有png图片(folder, rate=120):
    """rate可以控制缩放比例，正常是100
    再乘120是不想缩放太多，理论上乘100是正常比例
    """
    fd = CBaseFolder(folder)
    for imgFile in fd.Files('*.png'):
        files = pathjoin(fd.name, imgFile)
        try:
            im = Image.open(files)
            if 'dpi' in im.info:
                if im.info['dpi'][0] in (305, 610):  # 这个是magick转换过来的特殊值，不能缩放
                    continue  # 180920周四，610是为欧龙加的，欧龙双师需要设置-density 240
                # print(files, im.info)  # dpi: 600
                # print(查看图片的Exif信息(im))
                s = list(im.size)
                s[0] = int(s[0] / im.info['dpi'][0] * rate)
                s[1] = int(s[1] / im.info['dpi'][1] * rate)
                im = im.resize(s, Image.ANTIALIAS)
            im.save(files)
        except:
            print('无法处理图片：', files)
            continue


def 缩放目录下所有png图片2(folder, scale=1.0):
    """rate可以控制缩放比例，正常是100
    再乘120是不想缩放太多，理论上乘100是正常比例
    """
    fd = CBaseFolder(folder)
    for imgFile in fd.Files('*.png'):
        files = pathjoin(fd.name, imgFile)
        im = Image.open(files)
        s = list(im.size)
        s[0] = int(s[0] * scale)
        s[1] = int(s[1] * scale)
        im = im.resize(s, Image.ANTIALIAS)
        im.save(files)


def 查看目录下png图片信息(folder):
    fd = CBaseFolder(folder)
    ls = list()
    for imgFile in fd.Files('*.png'):
        file = File(pathjoin(fd.name, imgFile))
        im = Image.open(file.name)
        d0, d1 = im.info['dpi'] if 'dpi' in im.info else ('', '')
        # 处理eps格式
        epsFile = file.with_suffix('.eps')
        if epsFile:
            epsSize, epsIm = epsFile.大小(), Image.open(epsFile.name)
            boundingBox = epsIm.info['BoundingBox'].replace(' ', ',') if 'BoundingBox' in epsIm.info else ''
        else:
            epsSize, boundingBox = '', ''
        # 处理pdf格式
        pdfFile = File(file.name[:-4] + '-eps-converted-to.pdf')
        pdfSize = pdfFile.size if pdfFile else ''
        # 存储到列表
        ls.append((imgFile, im.size[0], im.size[1], d0, d1,
                   file.size, file.mtime.strftime(' %y%m%d-%H%M%S'),
                   epsSize, boundingBox, pdfSize))
    df = pd.DataFrame.from_records(ls,
                                   columns=('fileName', 'width', 'height', 'dpi_w', 'dpi_h', 'size', 'time', 'epsSize',
                                            'boundingBox', 'pdfSize'))
    PrintFullTable(df)
    return df


def MakeColorTransparent(image, color, thresh2=0):
    """
    将指定颜色转为透明

    https://stackoverflow.com/questions/765736/using-pil-to-make-all-white-pixels-transparent/765829"""
    from PIL import ImageMath

    def Distance2(a, b):
        return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])

    image = image.convert("RGBA")
    red, green, blue, alpha = image.split()
    image.putalpha(ImageMath.eval("""convert(((((t - d(c, (r, g, b))) >> 31) + 1) ^ 1) * a, 'L')""",
                                  t=thresh2, d=Distance2, c=color, r=red, g=green, b=blue, a=alpha))
    return image


____other = """"""

mydecrypt = base64.b64decode


class LengthFormatter:
    """ 长度换算类，可以在允许浮点小误差的场景使用
    TODO 需要精确运算也是可以的，那就要用分数类来存储底层值了

    # 默认标准长度是mm，所以初始化一个233数字，就等于用233mm初始化
    >>> LengthFormatter(233).cm  # 然后可以转化为cm，计算厘米单位下的长度值
    23.3
    >>> LengthFormatter('233pt')  # 可以用带单位的字符串
    81.78mm
    >>> LengthFormatter('233.45 pt')  # 支持小数、有空格等格式
    81.94mm

    应用举例：把长度超过12cm的hspace都限制在12cm以内
    >> s = NestEnv(s).latexcmd1('hspace').bracket('{', inner=True).\
           replace(lambda x: '12cm' if LengthFormatter(x).cm > 12 else x)
    """

    # 所有其他单位长度与参照长度mm之间比例关系
    ratio = {'pt': 0.351,  # 点
             'bp': 0.353,  # 大点，≈1pt
             'dd': 0.376,  # 迪多，=1.07pt
             'pc': 4.218,  # 派卡，=12pt
             'sp': 1 / 65536,  # 定标点，65536sp=1pt
             'cm': 10,  # 厘米
             'cc': 4.513,  # 西塞罗
             'in': 25.4,  # 英寸，=72.27pt
             'em': 18,  # 1em≈当前字体中M的宽度，在正文12pt情况下，一般为18pt
             'ex': 12,  # 1ex≈当前字体中x的高度，暂按12pt处理
             }

    def __init__(self, v=0):
        if isinstance(v, (int, float)):
            self.__dict__['mm'] = v
        elif isinstance(v, str):
            m = re.match(r'(-?\d+(?:\.\d*)?)\s*(' + '|'.join(list(self.ratio.keys()) + ['mm']) + ')$', v)
            if not m: raise ValueError(f'不存在的长度单位类型：{v}')
            self.__dict__['mm'] = 0
            self.__setitem__(m.group(2), float(m.group(1)))
        else:
            raise ValueError(f'不存在的长度单位类型：{v}')

    def __repr__(self):
        return '{:.2f}mm'.format(self.__dict__['mm'])

    def __getattr__(self, key):
        if key == 'mm':
            return self.__dict__['mm']
        elif key in self.ratio.keys():
            return self.__dict__['mm'] / self.ratio[key]
        else:
            raise ValueError(f'不存在的长度单位类型：{key}')

    def __setitem__(self, key, value):
        if key == 'mm':
            self.__dict__['mm'] = value
        elif key in self.ratio.keys():
            self.__dict__['mm'] = value * self.ratio[key]
        else:
            raise ValueError(f'不存在的长度单位类型：{key}')


if __name__ == '__main__':
    print('测试')
