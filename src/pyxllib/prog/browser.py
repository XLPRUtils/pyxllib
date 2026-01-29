import os
import subprocess
import tempfile
import hashlib
import webbrowser
import pathlib
import platform

from loguru import logger


def get_hash(data):
    """ 计算数据的哈希值，用于生成唯一文件名

    :param Any data: 输入数据，会转为字符串处理
    :return str: 哈希字符串
    """
    if not isinstance(data, (str, bytes)):
        data = str(data)
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.md5(data).hexdigest()


class Explorer:
    """ 资源管理器类，负责调用系统程序打开文件 """

    def __init__(self, app='explorer', shell=False):
        """ 初始化 Explorer

        :param str app: 应用程序名称或路径
        :param bool shell: 是否通过 shell 执行命令
        """
        self.app = app
        self.shell = shell

    def __call__(self, *args, wait=False, **kwargs):
        """ 调用程序打开文件或执行命令

        :param args: 命令行参数
        :param bool wait: 是否等待程序运行结束
        :param kwargs: 传递给 subprocess 的其他参数
        """
        actual_shell = kwargs.get('shell', self.shell)
        
        # 特殊处理 Mac 的 open 命令
        if platform.system() == 'Darwin' and self.app.startswith('open '):
            cmd = self.app + ' ' + ' '.join(f'"{arg}"' for arg in args)
            actual_shell = True
        else:
            cmd = [self.app] + [str(arg) for arg in args]

        try:
            if wait:
                return subprocess.run(cmd, shell=actual_shell, **kwargs)
            else:
                return subprocess.Popen(cmd, shell=actual_shell, **kwargs)
        except FileNotFoundError:
            # 如果是已知浏览器尝试失败，尝试退而求其次使用系统默认方式
            if self.app in ('chrome', 'msedge', 'google-chrome', 'chrome.exe'):
                logger.warning(f"未找到程序 {self.app}，尝试使用系统默认浏览器")
                return webbrowser.open(str(args[0]) if args else '')
            raise FileNotFoundError(f"未找到程序或命令：{self.app}")


class Browser(Explorer):
    """ 浏览器类，支持将各种数据转为 HTML/TXT 并在浏览器中查看 """

    def __init__(self, app=None):
        """ 初始化 Browser

        :param str app: 浏览器程序路径。如果为 None，则尝试自动检测或使用系统默认浏览器
        """
        if app is None:
            app = self._detect_browser()
        super().__init__(app or 'webbrowser')

    def _detect_browser(self):
        """ 尝试检测系统中安装的浏览器 """
        sys_platform = platform.system()
        if sys_platform == 'Windows':
            # 常见的浏览器路径
            paths = [
                os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), 'Google\\Chrome\\Application\\chrome.exe'),
                os.path.join(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'), 'Google\\Chrome\\Application\\chrome.exe'),
                'chrome',
                'msedge'
            ]
            for p in paths:
                if os.path.exists(p) or self._is_in_path(p):
                    return p
        elif sys_platform == 'Linux':
            for p in ['google-chrome', 'firefox']:
                if self._is_in_path(p):
                    return p
        elif sys_platform == 'Darwin':  # macOS
            return 'open -a "Google Chrome"'
        
        return 'webbrowser'

    @staticmethod
    def _is_in_path(name):
        """ 检查程序是否在环境变量 PATH 中 """
        return any(os.access(os.path.join(path, name), os.X_OK) 
                   for path in os.environ.get('PATH', '').split(os.pathsep))

    def _write_temp(self, content, suffix='.html', name=None):
        """ 将内容写入临时文件

        :param Any content: 文件内容
        :param str suffix: 文件后缀
        :param str name: 指定文件名，如果未提供则根据内容生成哈希名
        :return pathlib.Path: 临时文件路径
        """
        temp_dir = pathlib.Path(tempfile.gettempdir()) / 'pyxllib_browser'
        temp_dir.mkdir(parents=True, exist_ok=True)

        if name is None:
            name = get_hash(content)

        file_path = temp_dir / f"{name}{suffix}"

        # 写入内容
        if isinstance(content, bytes):
            file_path.write_bytes(content)
        else:
            file_path.write_text(str(content), encoding='utf-8')

        return file_path

    def to_file(self, arg, name=None, **kwargs):
        """ 将数据转换为适合浏览器查看的文件路径

        :param Any arg: 数据、路径或 URL
        :param str name: 临时文件名
        :param kwargs: 额外参数，如 to_html_args (针对 DataFrame)
        :return str: 文件路径或 URL 字符串
        """
        # 1. 如果已经是存在的路径或 URL，直接返回
        if isinstance(arg, (str, pathlib.Path)):
            s_arg = str(arg)
            if os.path.exists(s_arg) or s_arg.startswith(('http://', 'https://', 'file://')):
                return s_arg

        # 2. 尝试转换各种对象
        # 优先检测是否有 to_html 方法（如 pandas.DataFrame）
        if hasattr(arg, 'to_html') and callable(arg.to_html):
            to_html_args = kwargs.get('to_html_args', {})
            return str(self._write_temp(arg.to_html(**to_html_args), suffix='.html', name=name))

        # 检测是否有 render 方法（如 pyecharts）
        if hasattr(arg, 'render') and callable(arg.render):
            try:
                # pyecharts 的 render 默认生成到当前目录的 render.html，我们可以指定路径
                # 先创建一个临时路径名
                temp_file = self._write_temp('', suffix='.html', name=name)
                arg.render(path=str(temp_file))
                return str(temp_file)
            except Exception:
                pass

        # 3. 兜底方案：转为字符串作为 txt 查看
        suffix = '.txt'
        content = str(arg)
        # 如果看起来像 html，用 html 后缀
        if content.strip().startswith('<') and content.strip().endswith('>'):
            suffix = '.html'

        return str(self._write_temp(content, suffix=suffix, name=name))

    def __call__(self, arg, name=None, wait=False, **kwargs):
        """ 在浏览器中查看数据

        :param Any arg: 数据、路径或 URL
        :param str name: 临时文件名
        :param bool wait: 是否等待
        :param kwargs: 其他参数，支持 to_html_args 等
        """
        # 1. 提取 to_file 需要的参数
        to_file_keys = ['to_html_args']
        to_file_kwargs = {k: kwargs.pop(k) for k in to_file_keys if k in kwargs}

        target = self.to_file(arg, name=name, **to_file_kwargs)

        if self.app == 'webbrowser':
            webbrowser.open(target)
        else:
            super().__call__(target, wait=wait, **kwargs)


def view_files(procname, *files, wait=False, **kwargs):
    """ 调用指定程序打开多个文件

    :param str procname: 程序名
    :param files: 文件路径或数据
    :param bool wait: 是否等待
    :param kwargs:
        - name: 临时文件名基础
    """
    explorer = Explorer(procname)
    b = Browser()  # 借用 Browser 的 to_file 逻辑

    paths = []
    for i, f in enumerate(files):
        name = kwargs.get('name')
        if name and i > 0:
            name = f"{name}_{i}"
        paths.append(b.to_file(f, name=name))

    explorer(*paths, wait=wait)


# 单例对象
browser = Browser()


if __name__ == '__main__':
    import fire
    fire.Fire({
        'show': browser,
        'view': view_files
    })
