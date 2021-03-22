#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2018/07/12 09:09

"""
各种文件遍历功能

这里要强调，推荐os.walk功能
"""

from pyxllib.text import *
import pyxllib.basic.stdlib.zipfile as zipfile  # 重写了标准库的zipfile文件，cp437改为gbk，解决zip中文乱码问题

try:
    import paramiko
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'paramiko'])
    import paramiko

# 对 paramiko 进一步封装的库
# try:
#     import fabric
# except ModuleNotFoundError:
#     subprocess.run(['pip3', 'install', 'fabric'])
#     import fabric

try:
    import scp
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'scp'])
    import scp

# 需要使用的第三方软件
# BCompare.exe， bcompare函数要用


____section_1_normal = """
一些通用文件、文件夹工具
"""


def add_quote(s):
    return f'"{s}"'


def recreate_folders(*dsts):
    """重建一个空目录"""
    for dst in dsts:
        try:
            # 删除一个目录（含内容），设置ignore_errors可以忽略目录不存在时的错误
            shutil.rmtree(dst, ignore_errors=True)
            os.makedirs(dst)  # 重新新建一个目录，注意可能存在层级关系，所以要用makedirs
        except TypeError:
            pass


class UsedRecords:
    """存储用户的使用记录到一个文件"""

    def __init__(self, filename, default_value=None, *, use_temp_root=False, limit_num=30):
        """记录存储文件
        :param filename: 文件路径与名称
        :param default_value:
        :param use_temp_root: 使用临时文件夹作为根目录
        :param limit_num: 限制条目上限
        """
        from os.path import join, dirname, basename, exists
        # 1 文件名处理
        if use_temp_root:
            dirname = join(os.getenv('TEMP'), 'code4101py_config')
            basename = basename(filename)
            fullname = join(dirname, basename)
        else:
            dirname = dirname(filename)
            basename = basename(filename)
            fullname = filename

        # 2 读取值
        if exists(fullname):
            ls = ensure_content(fullname).splitlines()
        else:
            ls = list(default_value)

        # 3 存储到类
        self.dirname = dirname
        self.basename = basename
        self.fullname = fullname
        self.ls = ls
        self.limit_num = limit_num

    def save(self):
        """保存记录文件"""
        File(self.dirname + '/').ensure_parent()
        File(self.fullname).write('\n'.join(self.ls), if_exists='delete')

    def add(self, s):
        """新增一个使用方法
        如果s在self.ls里，则把方法前置到第一条
        否则在第一条添加新方法

        如果总条数超过30要进行删减
        """
        if s in self.ls:
            del self.ls[self.ls.index(s)]

        self.ls = [s] + list(self.ls)

        if len(self.ls) > self.limit_num:
            self.ls = self.ls[:self.limit_num]

    def __str__(self):
        res = list()
        res.append(self.fullname)
        for t in self.ls:
            res.append(t)
        return '\n'.join(res)


def checkpathfile(name):
    r"""判断环境变量path下是否有name这个文件，有则返回绝对路径，无则返回None
    常用的有：BCompare.exe、Chrome.exe、mogrify.exe、xelatex.exe

    >> checkpathfile('xelatex.exe')
    'C:\\CTEX\\MiKTeX\\miktex\\bin\\xelatex.exe'
    >> checkpathfile('abcd.exe')
    """
    for path in os.getenv('path').split(';'):
        fn = os.path.join(path, name)
        if os.path.exists(fn):
            return fn
    return None


def filename_tail(fn, tail):
    """在文件名末尾和扩展名前面加上一个tail"""
    names = os.path.splitext(fn)
    return names[0] + tail + names[1]


def hasext(f, *exts):
    """判断文件f是否是exts扩展名中的一种，如果不是返回False，否则返回对应的值

    所有文件名统一按照小写处理
    """
    ext = os.path.splitext(f)[1].lower()
    exts = tuple(map(lambda x: x.lower(), exts))
    if ext in exts:
        return ext
    else:
        return False


def isdir(fn):
    """判断输入的是不是合法的路径格式，且存在确实是一个文件夹"""
    try:
        return os.path.isdir(fn)
    except ValueError:  # 出现文件名过长的问题
        return False
    except TypeError:  # 输入不是字符串类型
        return False


____section_4_mygetfiles = """
py有os.walk可以递归遍历得到一个目录下的所有文件
但是“我们”常常要过滤掉备份文件（171020-153959），Old、temp目、.git等目录
特别是windows还有一个很坑爹的$RECYCLE.BIN目录。
所以在os.walk的基础上，再做了封装得到myoswalk。

然后在myoswalk基础上，实现mygetfiles。
"""


def gen_file_filter(s):
    """生成一个文件名过滤函数"""
    if s[0] == '.':
        return lambda x: x.endswith(s)
    else:
        s = s.replace('？', r'[\u4e00-\u9fa5]')  # 中文问号可以匹配任意中文字符
        return lambda x: re.search(s, x)


def myoswalk(root, filter_rule=None, recur=True):
    """
    :param root: 根目录
    :param filter_rule:
        字符串
            以点.开头的，统一认为是进行后缀格式识别
        其他字符串类型会认为是一个正则规则，只要相对root的全名能search到规则即认为匹配
            可以将中文问号用于匹配任意汉字
        也可以输入自定义函数： 输入参数是相对root目录下的文件全名
    :param recur: 是否进行子文件夹递归
    :return:
    """
    if isinstance(filter_rule, str):
        filter_rule = gen_file_filter(filter_rule)

    # prefix_len = len(root)  # 计算出前缀长度
    for dirpath, dirnames, filenames in os.walk(root):
        # relative_root = dirpath[prefix_len+1:]  # 我想返回相对路径，但是好像不太规范会对很多东西造成麻烦
        #  过滤掉特殊目录
        for t in ('.git', '$RECYCLE.BIN', '__pycache__', 'temp', 'Old', 'old'):
            try:
                del dirnames[dirnames.index(t)]
            except ValueError:
                pass
        # 去掉备份文件
        dirnames = list(filter(lambda x: not File(x).backup_time and '-冲突-' not in x, dirnames))
        filenames = list(filter(lambda x: not File(x).backup_time and '-冲突-' not in x, filenames))

        # 调用特殊过滤规则
        if filter_rule:
            dirnames = list(filter(lambda x: filter_rule(f'{dirpath}\\{x}'), dirnames))
            filenames = list(filter(lambda x: filter_rule(f'{dirpath}\\{x}'), filenames))

        # 如果该文件夹下已经没有文件，不返回该目录
        if not (filenames or dirnames):
            continue

        # 返回生成结果
        yield dirpath, dirnames, filenames

        if not recur:  # 不进行递归
            break


def getfiles(root, filter_rule=None):
    """对os.walk进一步封装，返回所有匹配的文件

    可以这样遍历一个目录下的所有文件：
    for f in getfiles(r'C:\pycode\code4101py', r'.py'):
        print(f)
    筛选规则除了“.+后缀”，还可以写正则匹配
    """
    if isinstance(filter_rule, str):
        filter_rule = gen_file_filter(filter_rule)

    for root, _, files in os.walk(root, filter_rule):
        for f in files:
            if filter_rule and not filter_rule(f):
                continue
            yield root + '\\' + f


def tex_content_filefilter(f):
    """只获取正文类tex文件"""
    if f.endswith('.tex') and 'Conf' not in f and 'settings' not in f:
        return True
    else:
        return False


def tex_conf_filefilter(f):
    """只获取配置类tex文件"""
    if f.endswith('.tex') and ('Conf' in f or 'settings' in f):
        return True
    else:
        return False


def mygetfiles(root, filter_rule=None, recur=True):
    """对myoswalk进一步封装，返回所有匹配的文件
    会递归查找所有子文件

    可以这样遍历一个目录下的所有文件：
    for f in mygetfiles(r'C:\pycode\code4101py', r'.py'):
        print(f)
    这个函数已经自动过滤掉备份文件了
    筛选规则除了“.+后缀”，还可以写正则匹配

    参数含义详见myoswalk
    """
    for root, _, files in myoswalk(root, filter_rule, recur):
        for f in files:
            yield root + '\\' + f


def _test_getfile_speed():
    """
    遍历D盘所有文件(205066个) 用时0.65秒
    遍历D盘所有tex文件(7796个) 用时0.95秒
    有筛选遍历D盘所有文件(193161个) 用时1.19秒
    有筛选遍历D盘所有tex文件(4464个) 用时1.22秒
        + EnsureContent： 3.18秒，用list存储所有文本要 310 MB 开销，转str拼接差不多也是这个值
        + re.sub(r'\$.*?\$', r'', s)： 4.48秒
    """
    timer = Timer(start_now=True)
    ls = list(getfiles(r'D:\\'))
    timer.stop_and_report(f'遍历D盘所有文件({len(ls)}个)')

    timer = Timer(start_now=True)
    ls = list(getfiles(r'D:\\', '.tex'))
    timer.stop_and_report(f'遍历D盘所有tex文件({len(ls)}个)')

    timer = Timer(start_now=True)
    ls = list(mygetfiles(r'D:\\'))
    timer.stop_and_report(f'有筛选遍历D盘所有文件({len(ls)}个)')

    timer = Timer(start_now=True)
    ls = list(mygetfiles(r'D:\\', '.tex'))
    timer.stop_and_report(f'有筛选遍历D盘所有tex文件({len(ls)}个)')


____section_5_filedfs = """
对目录的遍历查看目录结构
"""


def file_generator(f):
    """普通文件迭代生成器
    :param f: 搜索目录
    """
    if os.path.isdir(f):
        try:
            dirpath, dirnames, filenames = myoswalk(f).__next__()
        except StopIteration:
            return []

        ls = filenames + dirnames
        ls = map(lambda x: dirpath + '/' + x, ls)
        return ls
    else:
        return []


def pyfile_generator(f):
    """py文件迭代生成器
    :param f: 搜索目录
    """
    if os.path.isdir(f):
        try:
            dirpath, dirnames, filenames = myoswalk(f).__next__()
        except StopIteration:
            return []
        filenames = list(filter(lambda x: x.endswith('.py'), filenames))
        ls = filenames + dirnames
        ls = map(lambda x: dirpath + '/' + x, ls)
        return ls
    else:
        return []


def texfile_generator(f):
    """tex  文件迭代生成器
    :param f: 搜索目录
    """
    if os.path.isdir(f):
        try:
            dirpath, dirnames, filenames = myoswalk(f).__next__()
        except StopIteration:
            return []

        filenames = list(filter(lambda x: x.endswith('.tex'), filenames))
        ls = filenames + dirnames
        ls = map(lambda x: dirpath + '/' + x, ls)
        return ls
    else:
        return []


def file_str(f):
    """
    :param f: 输入完整路径的文件夹或文件名
    :return: 返回简化的名称
        a/b     ==> <b>
        a/b.txt ==> b.txt
    """
    name = os.path.basename(f)
    if os.path.isdir(f):
        s = '<' + name + '>'
    else:
        s = name
    return s


def filedfs(root,
            child_generator=file_generator, select_depth=None, linenum=True,
            mystr=file_str, msghead=True, lsstr=None, show_node_type=False, prefix='\t'):
    """对文件结构的递归遍历
    注意这里的子节点生成器有对非常多特殊情况进行过滤，并不是通用的文件夹查看工具
    """
    if isinstance(child_generator, str):
        if child_generator == '.py':
            child_generator = pyfile_generator
        elif child_generator == '.tex':
            child_generator = texfile_generator
        else:
            raise ValueError

    return dfs_base(root, child_generator=child_generator, select_depth=select_depth, linenum=linenum,
                    mystr=mystr, msghead=msghead, lsstr=lsstr, show_node_type=show_node_type, prefix=prefix)


____section_6_viewfiles = """
使用外部程序查看文件
"""


def genfilename(fd='.'):
    """生成一个fd目录下的文件名
    注意只是文件名，并未实际产生文件，输入目录是为了防止生成重名文件（以basename为标准的无重名）

    格式为：180827周一195802，如果出现重名，前面的6位记为数值d1，是年份+月份+日期的标签
        后面的6位记为数值d2，类似小时+分钟+秒的标签，但是在出现重名时，
        d2会一直自加1直到没有重名文件，所以秒上是可能会出现“99”之类的值的。
    """
    # 1 获取前段标签
    s1 = Datetime().briefdateweek()  # '180827周一'

    # 2 获取后端数值标签
    d2 = int(datetime.datetime.now().strftime('%H%M%S'))

    # 3 获取目录下文件，并迭代确保生成一个不重名文件
    ls = os.listdir(fd)
    files = set(map(lambda x: os.path.basename(os.path.splitext(x)[0]), ls))  # 收集basename

    while s1 + str(d2) in files:
        d2 += 1

    return s1 + str(d2)


____section_7_PackFile = """
处理压缩文件
"""


class PackFile:
    def __init__(self, file, mode=None):
        """
        :param file: 要处理的文件
        :param mode: 要处理的格式，不输入会有一套智能匹配算法
            'rar'：
            'zip'： docx后缀的，默认采用zip格式解压
        """
        # 1 确定压缩格式
        name, ext = os.path.splitext(file)
        ext = ext.lower()
        if not mode:
            if ext in ('.docx', '.zip'):
                mode = 'zip'
            elif ext == '.rar':
                mode = 'rar'
            else:
                dprint(ext)  # 从文件扩展名无法得知压缩格式
                raise ValueError
        self.mode = mode

        # 2 确定是用的解压“引擎”
        if mode == 'zip':
            self.proc = zipfile.ZipFile(file)
        elif mode == 'rar':
            try:
                from unrar.rarfile import RarFile
            except ModuleNotFoundError:
                dprint()  # 缺少unrar模块，安装详见： https://blog.csdn.net/code4101/article/details/79328636
                raise ModuleNotFoundError
            self.proc = RarFile(file)
        # 3 解压文件夹目录，None表示还未解压
        self.tempfolder = None

    def open(self, member, pwd=None):
        """Return file-like object for 'member'.

           'member' may be a filename or a RarInfo object.
        """
        return self.proc.open(member, pwd)

    def read(self, member, pwd=None):
        """Return file bytes (as a string) for name."""
        return self.proc.read(member, pwd)

    def namelist(self):
        """>> self.namelist()  # 获得文件清单列表
             1           [Content_Types].xml
             2                   _rels/.rels
            ......
            20            word/fontTable.xml
            21              docProps/app.xml
        """
        return self.proc.namelist()

    def setpassword(self, pwd):
        """Set default password for encrypted files."""
        return self.proc.setpassword(pwd)

    def getinfo(self, name):
        """
        >> self.getinfo('word/document.xml')  # 获得某个文件的信息
        <ZipInfo filename='word/document.xml' compress_type=deflate file_size=140518 compress_size=10004>
        """
        return self.proc.getinfo(name)

    def infolist(self, prefix=None, zipinfo=True):
        """>> self.infolist()  # getinfo的多文件版本
             1           <ZipInfo filename='[Content_Types].xml' compress_type=deflate file_size=1495 compress_size=383>
             2                    <ZipInfo filename='_rels/.rels' compress_type=deflate file_size=590 compress_size=243>
            ......
            20            <ZipInfo filename='word/fontTable.xml' compress_type=deflate file_size=1590 compress_size=521>
            21               <ZipInfo filename='docProps/app.xml' compress_type=deflate file_size=720 compress_size=384>

            :param prefix:
                可以筛选文件的前缀，例如“word/”可以筛选出word目录下的
            :param zipinfo:
                返回的list每个元素是zipinfo数据类型
        """
        ls = self.proc.infolist()
        if prefix:
            ls = list(filter(lambda t: t.filename.startswith(prefix), ls))
        if not zipinfo:
            ls = list(map(lambda x: x.filename, ls))
        return ls

    def printdir(self):
        """Print a table of contents for the RAR file."""
        return self.proc.printdir()

    def testrar(self):
        """Read all the files and check the CRC."""
        return self.proc.testrar()

    def extract(self, member, path=None, pwd=None):
        """注意，如果写extract('word/document.xml', 'a')，那么提取出来的文件是在'a/word/document.xml'
        """
        return self.proc.extract(member, path, pwd)

    def extractall(self, path=None, members=None, pwd=None):
        """Extract all members from the archive to the current working
           directory. `path' specifies a different directory to extract to.
           `members' is optional and must be a subset of the list returned
           by namelist().
        """
        return self.proc.extractall(path, members, pwd)

    def extractall2tempfolder(self):
        """将文件解压到一个临时文件夹，并返回临时文件夹目录"""
        if not self.tempfolder:
            self.tempfolder = tempfile.mkdtemp()
            self.proc.extractall(path=self.tempfolder)
        return self.tempfolder

    def clear_tempfolder(self):
        """删除创建的临时文件夹内容"""
        filesdel(self.tempfolder)

    def __enter__(self):
        """使用with ... as ...语法能自动建立解压目录和删除
        注意：这里返回的不是PackFile对象，而是解压后的目录
        """
        path = self.extractall2tempfolder()
        return path

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_tempfolder()


____section_temp = """
临时添加的新功能
"""


def filetext_replace(files, func, *,
                     count=-1, start=1, bc=False, write=False, if_exists=None):
    r"""遍历目录下的文本文件进行批量处理的功能函数

    :param files: 文件匹配规则，详见filesmatch用法
    :param func: 通用文本处理函数
    :param count: 匹配到count个文件后结束，防止满足条件的文件太多，程序会跑死
    :param start: 从编号几的文件开始查找，一般在遇到意外调试的时候使用
    :param bc: 使用beyond compare软件
        注意bc的优先级比write高，如果bc和write同时为True，则会开bc，但并不会执行write
    :param write: 是否原地修改文件内容进行保存
    :param if_exists: 是否进行备份，详见writefile里的参数文件
    :return: 满足条件的文件清单
    """
    ls = []
    total = 0
    for f in filesmatch(files):
        # if 'A4-Exam' in f:
        #     continue
        total += 1
        if total < start:
            continue
        s0 = File(f).read()
        s1 = func(s0)
        if s0 != s1:
            match = len(ls) + 1
            dprint(f, total, match)
            if bc:
                bcompare(f, s1)
            elif write:  # 如果开了bc，程序是绝对不会自动写入的
                File(f).write(s1, if_exists=if_exists)
            ls.append(f)
            if len(ls) == count:
                break

    match_num = len(ls)
    dprint(total, match_num)
    return ls


def change_ext(filename, ext):
    """更改文件名后缀
    返回第1个参数是新的文件名，第2个参数是这个文件是否存在

    输入的fileName可以没有扩展名，如'A/B/C/a'，仍然可以找对应的扩展名为ext的文件
    输入的ext不要含有'.'，例如正确格式是输入'tex'、'txt'
    """
    name = os.path.splitext(filename)[0]  # 'A/B/C/a.txt' --> 'A/B/C/a'
    newname = name + '.' + ext
    return newname, os.path.exists(newname)


def download_file(url, fn=None, *, encoding=None, if_exists=None, ext=None, temp=False):
    """类似writefile，只是源数据是从url里下载
    :param url: 数据下载链接
    :param fn: 保存位置，会从url智能提取文件名
    :param if_exists: 详见writefile参数解释
    :para temp: 将文件写到临时文件夹
    :return:
    """
    if not fn: fn = url.split('/')[-1]
    root = Dir.TEMP if temp else None
    fn = File(fn, root, suffix=ext).write(requests.get(url).content,
                                          encoding=encoding, if_exists=if_exists, etag=(not fn))
    return fn.to_str()


____other = """
"""


def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


class DataSyncBase:
    """ 在windows和linux之间同步文件数据

    DataSync只有在win平台才能用

    TODO 如果本地目录a已存在，从服务器a再下载，会跑到a/a的位置
    TODO 选择性从服务器拷贝？ 怎么获得服务器有哪些文件内容，使用ssh来获得？
    """

    def __init__(self, server, port, user, password):
        self.ssh = createSSHClient(server, port, user, password)
        self.scp = scp.SCPClient(self.ssh.get_transport())

    @classmethod
    def _remote_path(cls, local_path):
        """ 本地路径和服务器路径的一些相对位置 """

        # 本功能需要在继承的类上定制功能，例如：
        # dst = local_path.replace('D:/datasets', '/home/datasets')
        # return dst

        raise NotImplementedError

    @classmethod
    def _proc_paths(cls, in_arg):
        # 转为文件清单来同步
        if isinstance(in_arg, Dir) and in_arg.subs:
            # 可以输入Dir且选中部分子文件
            paths = [str(p) for p in in_arg.subpaths()]
        else:
            paths = [str(in_arg)]
        return paths

    def put(self, path):
        paths = self._proc_paths(path)

        tt = TicToc()
        for p in paths:
            tt.tic()
            # 其实scp也支持同时同步多文件，但是目标位置没法灵活控制，所以我这里还是一个一个同步
            q = self._remote_path(p)
            self.ssh.exec_command(f'mkdir -p {os.path.dirname(q)}')  # 如果不存在父目录则建立
            self.scp.put(p, q, recursive=True)
            t = tt.tocvalue()
            speed = humanfriendly.format_size(file_or_dir_size(p) / t, binary=True)
            print(f'upload to {q}, ↑{speed}/s, {t:.2f}s')

    def get(self, path):
        """ 只需要写本地文件路径，会推断服务器上的位置 """
        paths = self._proc_paths(path)

        tt = TicToc()
        for p in paths:
            tt.tic()
            # 目录的同步必须要开recursive，其他没什么区别
            Dir(os.path.dirname(p)).ensure_dir()
            self.scp.get(self._remote_path(p), p, recursive=True, preserve_times=True)
            t = tt.tocvalue()
            speed = humanfriendly.format_size(file_or_dir_size(p) / t, binary=True)
            print(f'download {p}, ↓{speed}/s, {t:.2f}s')


def cache_file(file, make_data_func, *, reset=False, **kwargs):
    """
    :param file: 需要缓存的文件路径
    :param make_data_func: 如果文件不存在，则需要生成一份，要提供数据生成函数
    :param reset: 如果file是否已存在，都用make_data_func强制重置一遍
    :param kwargs: 可以传递read、write支持的扩展参数
    :return: 从缓存文件直接读取到的数据
    """
    file = File(file)
    if file and not reset:  # 文件存在，直接读取返回
        data = file.read(**kwargs)
    else:  # 文件不存在则要生成一份数据
        data = make_data_func()
        file.write(data, **kwargs)
    return data
