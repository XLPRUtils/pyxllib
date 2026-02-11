import builtins
import os
import subprocess
import webbrowser
import pathlib
import platform

from loguru import logger


class Explorer:
    """资源管理器类，负责调用系统程序打开文件"""

    def __init__(self, app="explorer", shell=False):
        """初始化 Explorer

        :param str app: 应用程序名称或路径
        :param bool shell: 是否通过 shell 执行命令
        """
        self.app = app
        self.shell = shell

    def __call__(self, *args, wait=False, **kwargs):
        """调用程序打开文件或执行命令

        :param args: 命令行参数
        :param bool wait: 是否等待程序运行结束
        :param kwargs: 传递给 subprocess 的其他参数
        """
        actual_shell = kwargs.get("shell", self.shell)

        # 特殊处理 Mac 的 open 命令
        if platform.system() == "Darwin" and self.app.startswith("open "):
            cmd = self.app + " " + " ".join(f'"{arg}"' for arg in args)
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
            if self.app in ("chrome", "msedge", "google-chrome", "chrome.exe"):
                logger.warning(f"未找到程序 {self.app}，尝试使用系统默认浏览器")
                return webbrowser.open(str(args[0]) if args else "")
            raise FileNotFoundError(f"未找到程序或命令：{self.app}")


class Browser(Explorer):
    """浏览器类，支持将各种数据转为 HTML/TXT 并在浏览器中查看"""

    def __init__(self, app=None):
        """初始化 Browser

        :param str app: 浏览器程序路径。如果为 None，则尝试自动检测或使用系统默认浏览器
        """
        if app is None:
            app = self._detect_browser()
        super().__init__(app or "webbrowser")

    def _detect_browser(self):
        """尝试检测系统中安装的浏览器"""
        sys_platform = platform.system()
        if sys_platform == "Windows":
            # 常见的浏览器路径
            paths = [
                os.path.join(
                    os.environ.get("ProgramFiles", "C:\\Program Files"), "Google\\Chrome\\Application\\chrome.exe"
                ),
                os.path.join(
                    os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
                    "Google\\Chrome\\Application\\chrome.exe",
                ),
                "chrome",
                "msedge",
            ]
            for p in paths:
                if os.path.exists(p) or self._is_in_path(p):
                    return p
        elif sys_platform == "Linux":
            for p in ["google-chrome", "firefox"]:
                if self._is_in_path(p):
                    return p
        elif sys_platform == "Darwin":  # macOS
            return 'open -a "Google Chrome"'

        return "webbrowser"

    @staticmethod
    def _is_in_path(name):
        """检查程序是否在环境变量 PATH 中"""
        return any(
            os.access(os.path.join(path, name), os.X_OK) for path in os.environ.get("PATH", "").split(os.pathsep)
        )

    def __call__(self, arg, name=None, wait=False, mode="auto", **kwargs):
        """在浏览器或控制台中查看数据

        :param Any arg: 数据、路径或 URL
        :param str name: 临时文件名
        :param bool wait: 是否等待
        :param str mode: 展示模式
            - auto: 自动选择，Windows 默认 browser，其他环境默认 console (如果 arg 是数据)
            - browser: 在浏览器中打开
            - console: 在控制台输出
            - text: 返回纯文本字符串
            - html: 返回 HTML 字符串
        :param kwargs: 其他参数
        """
        from pyxllib.text.document import Document

        if mode == "auto":
            # 如果是已有路径，强制 browser 模式
            if isinstance(arg, (str, pathlib.Path)):
                s_arg = str(arg)
                if os.path.exists(s_arg) or s_arg.startswith(("http://", "https://", "file://")):
                    mode = "browser"
                else:
                    # 不是路径，当作数据
                    if platform.system() == "Windows":
                        mode = "browser"
                    else:
                        mode = "console"
            else:
                # 非字符串，肯定是数据
                if platform.system() == "Windows":
                    mode = "browser"
                else:
                    mode = "console"

        # 根据模式分发
        if mode == "console":
            return Document(arg).print(**kwargs)
        elif mode == "text":
            return Document(arg).render_text(**kwargs)
        elif mode == "html":
            return Document(arg).render_html(**kwargs)

        # mode == "browser"
        # 1. 检查是否直接是路径/URL
        if isinstance(arg, (str, pathlib.Path)):
            s_arg = str(arg)
            if os.path.exists(s_arg) or s_arg.startswith(("http://", "https://", "file://")):
                return super().__call__(s_arg, wait=wait, **kwargs)

        # 2. 否则通过 Document 生成临时文件并打开
        return Document(arg).browser(name=name, wait=wait, **kwargs)


def view_files(procname, *files, wait=False, **kwargs):
    """调用指定程序打开多个文件

    :param str procname: 程序名
    :param files: 文件路径或数据
    :param bool wait: 是否等待
    :param kwargs:
        - name: 临时文件名基础
    """
    from pyxllib.text.document import Document

    explorer = Explorer(procname)

    paths = []
    for i, f in enumerate(files):
        name = kwargs.get("name")
        if name and i > 0:
            name = f"{name}_{i}"

        if isinstance(f, (str, pathlib.Path)) and (os.path.exists(str(f)) or str(f).startswith(("http", "file"))):
            paths.append(str(f))
        else:
            paths.append(str(Document(f).to_file(name=name)))

    explorer(*paths, wait=wait)


# 单例对象
browser = Browser()


if __name__ == "__main__":
    import fire

    fire.Fire({"show": browser, "view": view_files})
