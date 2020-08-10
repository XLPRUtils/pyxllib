#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/05/30 11:50


"""
使用gitpython库，在python调用git进行一些版本分析的功能

Git
    list_commits，输出仓库的commit历史记录
    bcompare，对比一个文件在不同版本的内容，也会输出这个文件的历史commit清单
        show，获得一个文件某个版本的文本

TODO 清单
1、输入一个sha，分析某一次commit的细节（GUI有相应功能，不紧急）
2、按照周几、24小时制、时间轴等判断提交频率，结合files_changed、insertions、deletions判断工作量（不紧急）
3、将数据以图片的直观形式展现
"""

import subprocess

try:
    import git
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'gitpython'])
    import git

from pyxllib.util.filelib import *


class Git:
    def __init__(self, repo_path):
        self.g = git.Git(repo_path)

    def commits_name(self, n=None, file=None):
        """ 每条commit的标题名称
        :param n: 输出条目数，默认为全部
        :param file: 仅摘取file文件（文件夹）的历史记录
        :return: 一个list，每个元素存储每个commit的名称
        """
        cmd = ['--pretty=%s']
        if n:
            cmd.append(f'-{n}')
        if file:
            cmd.append(file)
        return self.g.log(*cmd).splitlines()

    def commits_time0(self, n=None, file=None):
        """每条commit的提交时间
        使用excel可以识别的时间格式"""
        cmd = ['--pretty=%cd', '--date=format:%Y/%m/%d %H:%M']
        if n:
            cmd.append(f'-{n}')
        if file:
            cmd.append(file)
        t = self.g.log(*cmd)

        return t.splitlines()

    def commits_time(self, n=None, file=None):
        """每条commit的提交时间
        参数含义参考commits_name里的解释"""
        cmd = ['--pretty=%cd', '--date=format:%y%m%d-%w-%H:%M']
        if n:
            cmd.append(f'-{n}')
        if file:
            cmd.append(file)
        t = self.g.log(*cmd)

        def myweektag(m):
            s = m.group().replace('0', '7')
            s = digit2weektag(s[1])
            return s

        t = re.sub(r'-\d-', myweektag, t)
        return t.splitlines()

    def commits_sha(self, n=None, file=None) -> [str, str, ...]:
        """每条commit的sha哈希值
        参数含义参考commits_name里的解释"""
        cmd = ['--pretty=%h']
        if n:
            cmd.append(f'-{n}')
        if file:
            cmd.append(file)
        return self.g.log(*cmd).splitlines()

    def commits_stat(self, n=None, file=None):
        """统计每个commits修改的文件数和插入行数、删除行数
        :param n: 输出条目数，默认为全部
        :param file: 仅摘取file文件（文件夹）的历史记录
        :return: 一个n*3的二维list，第0列是修改的文件数，第1列是插入的代码行数，第2列是删除的代码行数
        """
        cmd = ['--stat']
        if n:
            cmd.append(f'-{n}')
        if file:
            cmd.append(file)
        s = self.g.log(*cmd)

        file_changed = []
        insertion = []
        deletions = []
        for m in re.finditer(r'(\d+) files? changed, (\d+) insertions?\(\+\)(?:, (\d+) deletions?\(-\))?', s):
            file_changed.append(int(m.group(1)))
            insertion.append(int(m.group(2)))
            if m.lastindex > 2:
                deletions.append(int(m.group(3)))
            else:
                deletions.append(0)

        arr = [file_changed, insertion, deletions]
        return arr

    def commits_author(self, n=None, file=None):
        """每条commit的作者
        参数含义参考commits_name里的解释"""
        cmd = ['--pretty=%an']
        if n:
            cmd.append(f'-{n}')
        if file:
            cmd.append(file)
        return self.g.log(*cmd).splitlines()

    def list_commits(self, n=None, file=None, exceltime=False) -> pd.DataFrame:
        """
        :param n: 要显示的条目数
        :param file: 只分析某个文件（文件夹）的提交情况。
            None： 不指定文件，默认输出所有commit
            注意此时insertions，deletions也是针对单个文件进行的
            另外科普下git.log的功能
                是纯粹以目录为准查找的，所以对于重命名、移动之类的操作，文件操作历史就无法衔接了
                但同样的，一个文件即使当前版本不存在，但仍能调出commit记录
        :param exceltime:
            False，默认的时间戳格式
            True，excel能识别的时间格式
        :return: 返回类似下述结构的DataFrame
                                    C:/pycode / commit名称             时间      sha  files_changed  insertions  deletions
              0                     retool.py整合到text.py  180807周二20:36  a557cd8              4         388        395
              1                       解决图片位置移动问题  180806周一21:28  5f0f63a              6          90         44
              2                                         ..  180806周一14:46  c7314a2              1           1         10
        """
        times = self.commits_time0(n, file) if exceltime else self.commits_time(n, file)
        ls = [self.commits_name(n, file), times, self.commits_sha(n, file)]
        ls.extend(self.commits_stat(n, file))
        ls.append(self.commits_author(n, file))
        ls = swap_rowcol(ls)
        t = file or self.g.working_dir
        col_tag = (t + ' / commit名称', '时间', 'sha', 'files_changed', 'insertions', 'deletions', 'author')
        df = pd.DataFrame.from_records(ls, columns=col_tag)
        return df

    def sha_id(self, s, file=None):
        """类似smartsha，但是返回的是sha对应的序号
        0对应最近一次提交，1对应上上次提交...

        这个实现和smartsha有点重复冗余，不是很优雅~~
        """
        shas = self.commits_sha(file=file)

        if s is None:
            return 0
        elif isinstance(s, str) and re.match(r'[a-z1-9]+', s):  # 正常的一段sha
            for i, t in enumerate(shas):
                if t.startswith(s):
                    return i
        elif isinstance(s, int):
            return s
        else:  # 其他情况，去匹配commit的名称
            names = self.commits_name(file=file)
            for i, t in enumerate(names):
                if s in t:
                    return i
            else:
                dprint(s)  # 没有找到这个关键词对应的commit
                raise ValueError

    def smartsha(self, s, file=None):
        """输入一段文本，智能识别sha
            None  -->  None
            sha   -->  sha
            数字  -->  0,1,2,3索引上次提交、上上次提交...
            文本  -->  找commit有出现关键词的commit

        可以只抓取某个文件file的修改版本
        """
        if s is None:
            return s
        elif isinstance(s, str) and re.match(r'[a-z0-9]+', s):  # 正常的一段sha
            return s

        shas = self.commits_sha(file=file)
        num = len(shas)

        if isinstance(s, int):  # 一个整数，0表示HEAD，最近的一次提交
            if s < num:
                return shas[s]
            else:
                dprint(num, s)  # 下标越界
                raise ValueError
        else:  # 其他情况，去匹配commit的名称
            names = self.commits_name(file=file)
            for i, t in enumerate(names):
                if s in t:
                    return shas[i]
            else:
                dprint(s)  # 没有找到这个关键词对应的commit
                raise ValueError

    def show(self, file, sha=0) -> str:
        """git show命令，可以查看文件内容
        :param file: 需要调用的文件相对路径
        :param sha: 支持smartsha的那些规则，注意如果是数字，是返回path文件对应的版本，而不是全局的commit的版本列表
        :return: 某个文件对应版本的文本内容
        """
        file = file.replace('\\', '/')  # 好像路径一定要用反斜杆/

        sha = self.smartsha(sha, file)

        if sha:
            s = self.g.show(f'{sha}:{file}')
        else:
            s = ensure_content(pathjoin(self.g.working_dir, file))
        return s

    def bcompare(self, file, sha1=0, sha2=None):
        """
        :param file: 文件相对路径
        :param sha1: 需要调取出的旧版本sha编号
            0，1，2可以索引上次提交，上上次提交的sha
            当sha1输入负数，例如-1时，则sha1调出的是文件最早版本
        :param sha2: 需要调取出的另一个版本sha编号
            默认是None，和当前文件内容比较
            也可以输入数字等索引sha
        :return: 无，直接调出 bcompare 软件对比结果

        还会在命令行输出这个文件的版本信息
        """
        # 1 获取s1内容
        sha1_id = self.sha_id(sha1)
        s1 = self.show(file, sha1_id)

        # 2 获取s2内容
        if sha2 is not None:
            sha2_id = self.sha_id(sha2)
            s2 = self.show(file, sha2_id)
        else:
            sha2_id = None
            s2 = os.path.join(self.g.working_dir, file)  # 存储文件名而不是内容

        # 3 对比
        dprint(sha1, sha2, sha1_id, sha2_id)
        print(dataframe_str(self.list_commits(file=file)))

        bcompare(s1, s2)

    def find_pattern(self, pattern, files=None) -> pd.DataFrame:
        """
        :param pattern: re.compile 对象
        :param files: 要搜索的文件清单

        191108周五10:40，目前跑 c:/pycode 要2分钟
        >> chrome(Git('C:/pycode/').find_pattern(re.compile(r'ssb等改明文')))

        """
        # 1 主对象
        df = self.list_commits()
        all_shas = list(reversed(list(df['sha'])))

        # 2 files没有设置则匹配所有文件
        # TODO 当前已经不存在的文件这样是找不到的，有办法把历史文件也挖出来？
        if not files:
            with Dir(self.g.working_dir):
                files = filesmatch('**/*.py')

        # 3 遍历文件
        for file in files:
            d = {}
            for sha in self.commits_sha(file=file):
                try:
                    cnt = len(pattern.findall(self.show(file, sha)))
                    if cnt: d[sha] = cnt
                except git.exc.GitCommandError:
                    pass
            if d:
                li = []
                v = 0
                for sha in all_shas:
                    if sha in d:
                        v = d[sha]
                    li.append(v)
                df[file] = list(reversed(li))

        return df
