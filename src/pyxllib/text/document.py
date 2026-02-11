#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  pyxllib/text/document.py

"""document.py - 通用富文本文档构建器

提供了一套统一的接口来构建包含文本、代码、表格、图片等元素的文档，
并支持导出为 HTML、Markdown、纯文本等多种格式。
主要用于生成分析报告、数据展示、日志记录等场景。

核心功能：
1. 统一的 Block 抽象：Header, Text, Code, Table, Image, Markdown, HTML。
2. 灵活的 Document 构建器：链式调用，方便添加内容。
3. 多格式渲染：支持渲染为 HTML (带默认 CSS)、Markdown、Text。
4. 便捷预览：支持直接在浏览器中预览生成的 HTML。

使用示例：

>>> d = Document(title="测试报告")
>>> _ = d.add_header("1. 简介")
>>> _ = d.add_text("这是一个自动生成的报告。")
>>> _ = d.add_code("print('Hello')", language="python")
>>> print(d.render_text())
=== 测试报告 ===
<BLANKLINE>
<BLANKLINE>
1. 简介
=====
<BLANKLINE>
这是一个自动生成的报告。
<BLANKLINE>
print('Hello')
<BLANKLINE>
"""

import builtins
import os
import subprocess
import tempfile
import hashlib
import webbrowser
import pathlib
import platform
import html
import base64
import io
import json
from datetime import datetime
from typing import List, Union, Any, Optional

from loguru import logger
from pyxllib.prog.lazyimport import lazy_import
from pyxllib.text.renderer import to_text

pd = lazy_import("pandas")


def get_hash(data):
    """计算数据的哈希值，用于生成唯一文件名

    :param Any data: 输入数据，会转为字符串处理
    :return str: 哈希字符串

    >>> get_hash("hello")
    '5d41402abc4b2a76b9719d911017c592'
    >>> get_hash(123)
    '202cb962ac59075b964b07152d234b70'
    """
    if not isinstance(data, (str, bytes)):
        data = str(data)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.md5(data).hexdigest()


def __1_Block_Definitions():
    """文档块定义 (中间表示层)"""
    pass


class Block:
    """文档块基类

    所有具体的文档内容块（如标题、文本、表格）都应继承自此类，
    并实现相应的渲染方法。
    """

    def to_html(self) -> str:
        """渲染为 HTML 格式"""
        raise NotImplementedError

    def to_md(self) -> str:
        """渲染为 Markdown 格式"""
        raise NotImplementedError

    def to_text(self) -> str:
        """渲染为纯文本格式"""
        raise NotImplementedError

    def to_dict(self) -> dict:
        """转换为字典格式，用于序列化"""
        raise NotImplementedError


class HeaderBlock(Block):
    """标题块"""

    def __init__(self, text: str, level: int = 1):
        """初始化标题块

        :param str text: 标题文本
        :param int level: 标题级别 (1-6)
        """
        self.text = str(text)
        self.level = max(1, min(6, level))  # 限制 1-6 级

    def to_html(self):
        # 自动生成锚点，方便跳转
        anchor = hashlib.md5(self.text.encode()).hexdigest()[:8]
        return f'<h{self.level} id="{anchor}">{html.escape(self.text)}</h{self.level}>'

    def to_md(self):
        return f"{'#' * self.level} {self.text}\n"

    def to_text(self):
        underline = "=" if self.level == 1 else "-"
        return f"\n{self.text}\n{underline * len(self.text)}\n"

    def to_dict(self):
        return {"type": "header", "text": self.text, "level": self.level}


class TextBlock(Block):
    """普通文本或代码块"""

    def __init__(self, text: Any, code: bool = False, language: str = None):
        """初始化文本块

        :param Any text: 文本内容
        :param bool code: 是否为代码块
        :param str language: 代码语言（仅当 code=True 时有效）
        """
        self.text = str(text)
        self.code = code
        self.language = language

    def to_html(self):
        if self.code:
            lang_attr = f' class="language-{self.language}"' if self.language else ""
            return f"<pre><code{lang_attr}>{html.escape(self.text)}</code></pre>"
        # 处理换行符，转为 <br>
        safe_text = html.escape(self.text).replace("\n", "<br>")
        return f"<p>{safe_text}</p>"

    def to_md(self):
        if self.code:
            lang_tag = self.language if self.language else ""
            return f"```{lang_tag}\n{self.text}\n```\n"
        return f"{self.text}\n"

    def to_text(self):
        return f"{self.text}\n"

    def to_dict(self):
        return {"type": "text", "text": self.text, "code": self.code, "language": self.language}


class MarkdownBlock(Block):
    """原生 Markdown 块"""

    def __init__(self, text: str):
        """初始化 Markdown 块

        :param str text: Markdown 源码
        """
        self.text = str(text)

    def to_html(self):
        # 保底机制：不依赖三方库，直接显示源码，用特殊样式包裹
        # 如果未来想支持渲染，可以在这里 lazy_import markdown
        return (
            f'<div class="markdown-block" style="border-left: 3px solid #6c757d; padding-left: 1rem; color: #495057;">'
            f"<pre>{html.escape(self.text)}</pre>"
            f'<small style="color: #adb5bd;">(Markdown Raw)</small>'
            f"</div>"
        )

    def to_md(self):
        return f"{self.text}\n"

    def to_text(self):
        return f"{self.text}\n"

    def to_dict(self):
        return {"type": "markdown", "text": self.text}


class TableBlock(Block):
    """表格块"""

    def __init__(self, data: Any, title: str = None, show_row_index: bool = False, show_col_index: bool = True):
        """初始化表格块

        :param Any data: 表格数据，支持 DataFrame, list, tuple 等
        :param str title: 表格标题
        :param bool show_row_index: 是否显示行索引
        :param bool show_col_index: 是否显示列索引
        """
        self.title = title
        self.show_row_index = show_row_index
        self.show_col_index = show_col_index
        # 尝试转 DataFrame
        if isinstance(data, pd.DataFrame):
            self.df = data
        elif isinstance(data, (list, tuple)) and data:
            try:
                self.df = pd.DataFrame(data)
            except:
                self.df = pd.DataFrame([str(data)])
        else:
            self.df = pd.DataFrame([str(data)])

    def to_html(self):
        caption = f"<caption>{html.escape(self.title)}</caption>" if self.title else ""
        # 默认样式：Bootstrap 风格
        table_html = self.df.to_html(
            classes="table table-striped table-hover table-sm",
            border=0,
            escape=True,
            index=self.show_row_index,
            header=self.show_col_index,
        )
        # 插入 caption
        if caption:
            table_html = table_html.replace("<table", f"<table", 1).replace(">", f">{caption}", 1)
        return f'<div class="table-responsive">{table_html}</div>'

    def to_md(self):
        title = f"**{self.title}**\n" if self.title else ""
        try:
            # 需要 tabulate 库支持 markdown 表格，如果不存在则回退
            # to_markdown(index=..., headers=...)
            headers = "keys" if self.show_col_index else []
            return title + self.df.to_markdown(index=self.show_row_index, headers=headers) + "\n"
        except (ImportError, AttributeError):
            return title + self.df.to_string(index=self.show_row_index, header=self.show_col_index) + "\n"

    def to_text(self):
        title = f"[{self.title}]\n" if self.title else ""
        return title + self.df.to_string(index=self.show_row_index, header=self.show_col_index) + "\n"

    def to_dict(self):
        # 使用 split 格式可以保留索引和列名信息
        return {
            "type": "table",
            "title": self.title,
            "data": self.df.to_dict(orient="split"),
            "show_row_index": self.show_row_index,
            "show_col_index": self.show_col_index,
        }


class ImageBlock(Block):
    """支持路径或Bytes的图片块"""

    def __init__(self, src: Union[str, bytes], title: str = None):
        """初始化图片块

        :param str|bytes src: 图片源，可以是文件路径、URL 或 bytes 数据
        :param str title: 图片标题/说明
        """
        self.title = title
        self.src = src

    def _get_b64(self):
        """获取图片的 base64 编码"""
        if isinstance(self.src, bytes):
            return base64.b64encode(self.src).decode("utf-8")
        elif isinstance(self.src, (str, pathlib.Path)):
            if str(self.src).startswith("http"):
                return None  # 网络图片不转base64
            if os.path.exists(self.src):
                with open(self.src, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
        return None

    def to_html(self):
        caption = f"<figcaption>{html.escape(self.title)}</figcaption>" if self.title else ""
        b64 = self._get_b64()

        if b64:
            src_str = f"data:image/png;base64,{b64}"
        else:
            src_str = str(self.src)

        return f'<figure><img src="{src_str}" style="max-width:100%; height:auto;" />{caption}</figure>'

    def to_md(self):
        return f"![{self.title or 'image'}]({self.src})\n"

    def to_text(self):
        return f"[Image: {self.title or str(self.src)[:20]}]\n"

    def to_dict(self):
        # 尝试获取 base64，如果获取不到则使用 src 字符串
        # 注意：如果是 bytes 类型 src，_get_b64 会处理；如果是本地路径，也会处理
        # 这里的策略是：如果有 base64 则提供，方便完全内嵌；同时也提供 src 原始值（如果是字符串）
        return {
            "type": "image",
            "src": str(self.src) if not isinstance(self.src, bytes) else None,
            "base64": self._get_b64(),
            "title": self.title,
        }


class RawHtmlBlock(Block):
    """原始 HTML 块，不进行转义"""

    def __init__(self, html_content: str):
        """初始化原始 HTML 块

        :param str html_content: HTML 内容
        """
        self.html_content = str(html_content)

    def to_html(self):
        return self.html_content

    def to_md(self):
        # Markdown 原生支持 HTML，直接嵌入
        return self.html_content + "\n"

    def to_text(self):
        # 文本模式下，无法渲染 HTML，只能显示“Raw HTML”占位符或源码
        return "[Raw HTML Block]\n"

    def to_dict(self):
        return {"type": "raw_html", "content": self.html_content}


def __2_Document_Components():
    """文档组件类"""
    pass


# 内置简单的 CSS，确保单文件 HTML 也很美观
DEFAULT_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; padding: 2rem; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; }
h1, h2, h3, h4, h5, h6 { color: #2c3e50; margin-top: 1.5rem; margin-bottom: 0.5rem; font-weight: 600; }
h1 { border-bottom: 2px solid #eee; padding-bottom: 0.5rem; }
a { color: #3498db; text-decoration: none; }
a:hover { text-decoration: underline; }
code { background-color: #f8f9fa; padding: 0.2rem 0.4rem; border-radius: 4px; font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; font-size: 0.9em; }
pre { background-color: #f8f9fa; padding: 1rem; border-radius: 6px; overflow-x: auto; }
table { width: auto; max-width: 100%; margin-bottom: 1rem; color: #212529; border-collapse: collapse; }
th, td { padding: 0.4rem; }
th { text-align: left; background-color: #e9ecef; }
img { border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
blockquote { border-left: 4px solid #eee; padding-left: 1rem; color: #666; }
.metadata { font-size: 0.8rem; color: #999; margin-bottom: 2rem; border-bottom: 1px solid #eee; padding-bottom: 1rem; }
"""


class DocumentBase:
    """文档基类，管理基础数据结构"""

    def __init__(self, title=None):
        self.title = title
        self.blocks: List[Block] = []
        self.css = DEFAULT_CSS


class DocumentBuilder(DocumentBase):
    """文档构建器，负责添加内容"""

    def add(self, content: Any, title: str = None, **kwargs):
        """添加内容

        自动根据内容类型判断并添加合适的 Block。
        注意：Document 尽量保持"愚钝"，只处理明确的类型。
        复杂的类型推断（如判断是否是 pyecharts 对象）应由调用者（如 Browser）处理。

        :param Any content: 内容数据
        :param str title: 标题（用于表格或图片）
        :return Document: 返回自身以支持链式调用
        """
        # 1. 如果本身就是 Block 对象
        if isinstance(content, Block):
            self.blocks.append(content)

        # 2. DataFrame -> Table
        elif isinstance(content, pd.DataFrame):
            self.blocks.append(TableBlock(content, title=title))

        # 3. 列表/字典 -> Table (尝试转换)
        elif isinstance(content, (list, tuple, dict)) and not isinstance(content, (str, bytes)):
            try:
                # 简单的启发式：如果是 list of dicts，转表格
                self.blocks.append(TableBlock(pd.DataFrame(content), title=title))
            except:
                # 转换失败，转文本
                self.blocks.append(TextBlock(str(content)))

        # 4. 默认 -> Text
        else:
            self.blocks.append(TextBlock(str(content)))

        return self

    def add_header(self, text, level=1):
        """添加标题

        :param str text: 标题文本
        :param int level: 标题级别 (1-6)
        """
        self.blocks.append(HeaderBlock(text, level))
        return self

    def add_text(self, text, code=False):
        """添加文本

        :param str text: 文本内容
        :param bool code: 是否作为代码块显示
        """
        self.blocks.append(TextBlock(text, code))
        return self

    def add_code(self, text, language=None):
        """添加代码块

        :param str text: 代码内容
        :param str language: 编程语言名称
        """
        self.blocks.append(TextBlock(text, code=True, language=language))
        return self

    def add_html(self, html_content):
        """添加 HTML 内容 (别名 add_raw_html)

        :param str html_content: HTML 源码
        """
        self.blocks.append(RawHtmlBlock(html_content))
        return self

    def add_md(self, md_content):
        """添加 Markdown 内容

        :param str md_content: Markdown 源码
        """
        self.blocks.append(MarkdownBlock(md_content))
        return self

    def add_raw_html(self, html_content):
        """添加原始 HTML 内容（不转义）

        :param str html_content: HTML 源码
        """
        self.blocks.append(RawHtmlBlock(html_content))
        return self

    def add_echart(self, chart):
        """添加 pyecharts 图表对象

        :param chart: pyecharts 图表对象
        """
        if hasattr(chart, "render_embed"):
            self.blocks.append(RawHtmlBlock(chart.render_embed()))
        else:
            # 兜底：如果不是 chart 对象，当做文本处理，并警告
            self.blocks.append(TextBlock(str(chart)))
        return self

    def _process_row_index(self, df, row_index):
        """处理行索引逻辑"""
        show_row_index = False
        if row_index is not None:
            if isinstance(row_index, int) and not isinstance(row_index, bool):
                # 如果是整数（且不是bool），重置索引起始值
                df.index = pd.RangeIndex(start=row_index, stop=row_index + len(df), step=1)
                show_row_index = True
            elif row_index is True:
                # 如果是 True，显示现有索引
                show_row_index = True
            # 如果是 False，保持 show_row_index=False
        return df, show_row_index

    def _process_col_index(self, df, col_index):
        """处理列索引逻辑"""
        show_col_index = True
        if col_index is not None:
            if isinstance(col_index, (list, tuple)):
                # 如果提供了列表，重设列名
                # 注意：这里假设列表长度匹配
                if len(col_index) == len(df.columns):
                    df.columns = col_index
                show_col_index = True
            elif col_index is False:
                show_col_index = False
            # 如果是 True，保持 show_col_index=True
        else:
            # col_index is None -> Default behavior?
            # User didn't specify behavior for None, but usually we show headers.
            pass
        return df, show_col_index

    def _add_table_common(self, df, title, row_index, col_index):
        df = df.copy()
        df, show_row_index = self._process_row_index(df, row_index)
        df, show_col_index = self._process_col_index(df, col_index)
        self.blocks.append(TableBlock(df, title, show_row_index=show_row_index, show_col_index=show_col_index))

    def _add_table_from_list(self, data, title, row_index, col_index):
        df = pd.DataFrame(data)
        # 如果列名是默认的整数索引，且用户没有指定 col_index（即 col_index=True/None），重命名为 col0...
        # 如果用户指定了 col_index 为列表，_process_col_index 会处理
        if isinstance(df.columns, pd.RangeIndex) and len(df.columns) > 0:
            if col_index is True or col_index is None:
                df.columns = [f"col{i}" for i in df.columns]

        self._add_table_common(df, title, row_index, col_index)

    def add_table(self, data, title=None, row_index=None, col_index=True):
        """添加表格

        :param data: 表格数据
        :param str title: 表格标题
        :param row_index: 行索引控制
            - None/False: 不显示行索引
            - True: 显示行索引
            - int: 显示行索引，并设置起始值为该整数
        :param col_index: 列索引控制
            - True/None: 显示列索引 (默认)
            - False: 不显示列索引
            - list/tuple: 显示列索引，并使用该列表作为列名
        """
        if isinstance(data, pd.DataFrame):
            self._add_table_common(data, title, row_index, col_index)
        elif isinstance(data, (list, tuple)):
            self._add_table_from_list(data, title, row_index, col_index)
        elif isinstance(data, dict):
            # dict 转 DataFrame 通常已经有 keys 作为 columns
            self._add_table_common(pd.DataFrame(data), title, row_index, col_index)
        else:
            self.blocks.append(TableBlock(data, title, show_row_index=bool(row_index), show_col_index=bool(col_index)))
        return self

    def add_image(self, src, title=None):
        """添加图片

        :param src: 图片源
        :param title: 图片标题
        """
        self.blocks.append(ImageBlock(src, title))
        return self


class DocumentRenderer(DocumentBuilder):
    """文档渲染器，负责导出和预览"""

    def _write_temp(self, content, suffix=".html", name=None):
        """将内容写入临时文件"""
        temp_dir = pathlib.Path(tempfile.gettempdir()) / "pyxllib_browser"
        temp_dir.mkdir(parents=True, exist_ok=True)

        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            hash_val = get_hash(content)[:4]
            name = f"{timestamp}_{hash_val}"

        file_path = temp_dir / f"{name}{suffix}"

        if isinstance(content, bytes):
            file_path.write_bytes(content)
        else:
            file_path.write_text(str(content), encoding="utf-8")

        return file_path

    def to_file(self, path=None, name=None, **kwargs):
        """生成文件

        :param path: 指定路径，如果为None则生成临时文件
        :param name: 临时文件名
        """
        content = self.render_html(**kwargs)

        if path:
            p = pathlib.Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return p
        else:
            return self._write_temp(content, suffix=".html", name=name)

    # --- Renders ---

    def _get_render_blocks(self, heading_numbering):
        if not heading_numbering:
            yield from self.blocks
            return

        counters = [0] * 6
        for b in self.blocks:
            if isinstance(b, HeaderBlock):
                level = b.level
                counters[level - 1] += 1
                for i in range(level, 6):
                    counters[i] = 0

                prefix = ".".join(str(c) for c in counters[:level]) + " "
                yield HeaderBlock(prefix + b.text, level=level)
            else:
                yield b

    def render_html(self, heading_numbering=None) -> str:
        """渲染为 HTML

        :param heading_numbering: 是否开启标题自动编号
            - True: 自动添加编号，如 "1 ", "1.1 " 等
            - False/None: 不添加编号 (默认)
        :return str: HTML 源码
        """
        body_content = "\n".join([b.to_html() for b in self._get_render_blocks(heading_numbering)])

        title_str = html.escape(str(self.title)) if self.title else "Document"
        h1_tag = f"<h1>{html.escape(str(self.title))}</h1>" if self.title else ""

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title_str}</title>
    <style>{self.css}</style>
</head>
<body>
    {h1_tag}
    {body_content}
</body>
</html>"""

    def render_md(self, heading_numbering=None) -> str:
        """渲染为 Markdown

        :param heading_numbering: 是否开启标题自动编号
        :return str: Markdown 源码
        """
        content = "\n\n".join([b.to_md() for b in self._get_render_blocks(heading_numbering)])
        if self.title:
            return f"# {self.title}\n\n{content}"
        else:
            return content

    def render_text(self, heading_numbering=None) -> str:
        """渲染为纯文本

        :param heading_numbering: 是否开启标题自动编号
        :return str: 纯文本内容
        """
        content = "\n".join([b.to_text() for b in self._get_render_blocks(heading_numbering)])
        if self.title:
            return f"=== {self.title} ===\n\n{content}"
        else:
            return content

    def render_json(self, **kwargs) -> str:
        """导出为 JSON 字符串

        :param kwargs: 传递给 json.dumps 的参数
        :return str: JSON 字符串
        """
        data = {"title": self.title, "blocks": [b.to_dict() for b in self.blocks]}
        return json.dumps(data, ensure_ascii=False, indent=2, **kwargs)

    def print(self, **kwargs):
        """输出到控制台

        :param kwargs: 传递给 render_text 的参数
        """
        print(self.render_text(**kwargs))
        return self

    def browser(self, name=None, wait=False, **kwargs):
        """在浏览器中打开

        :param name: 指定文件名
        :param wait: 是否等待浏览器关闭
        :param kwargs: 传递给 render_html 的参数，如 heading_numbering
        """
        # 局部导入避免循环引用
        from pyxllib.prog.browser import browser as open_in_browser

        # 提取渲染相关参数
        render_kwargs = {}
        if "heading_numbering" in kwargs:
            render_kwargs["heading_numbering"] = kwargs.pop("heading_numbering")

        target = self.to_file(name=name, **render_kwargs)
        open_in_browser(str(target), wait=wait, **kwargs)
        return self


class Document(DocumentRenderer):
    """通用文档构建器

    用于构建和渲染富文本文档。
    继承自 DocumentRenderer (-> DocumentBuilder -> DocumentBase)。
    """

    def __init__(self, content=None, title=None, **kwargs):
        """初始化文档

        :param Any content: 初始内容
        :param str title: 文档标题
        """
        super().__init__(title=title)

        if content is not None:
            self.add(content, **kwargs)


if __name__ == "__main__":
    d = Document(title=None)
    d.add_header("1 数据概览")
    d.add_text("这是一段测试文本，包含\n换行符。")

    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    d.add_table(df)

    d.add_header("2 代码示例", level=2)
    d.add_text("print('hello world')", code=True)

    # d.browser()
