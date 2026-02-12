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
>>> print(d.render_text())  # doctest: +SKIP
=== 测试报告 ===
<BLANKLINE>
<BLANKLINE>
1. 简介
=====
<BLANKLINE>
这是一个自动生成的报告。
<BLANKLINE>
print('Hello')
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
import inspect
from datetime import datetime
from typing import List, Union, Any, Optional

from loguru import logger
from pyxllib.prog.lazyimport import lazy_import

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


def __1_FormatBlock_Definitions():
    """文档块定义 (中间表示层)"""
    pass


class Block:
    """文档块基类

    所有具体的文档内容块（如标题、文本、表格）都应继承自此类，
    并实现相应的渲染方法。

    :cvar str type_name: 块类型名称，用于序列化和反序列化
    """
    type_name = "block"

    def to_html(self, **kwargs):
        """转换为 HTML (表现层)"""
        # 保底：预格式化文本
        return f"<pre>{html.escape(self.to_text(**kwargs))}</pre>"

    def to_md(self, **kwargs):
        """转换为 Markdown (表现层)"""
        # 保底：代码块
        return f"```\\n{self.to_text(**kwargs)}\\n```"

    def to_text(self, **kwargs):
        """转换为纯文本 (表现层)"""
        return str(self)

    def to_dict(self, **kwargs):
        """转换为字典格式 (数据层)"""
        raise NotImplementedError(f"{self.__class__.__name__} does not support conversion to dict.")

    def to_json(self, **kwargs):
        """转换为 JSON 字符串 (传输层)"""
        dump_args = {"ensure_ascii": False, "indent": 2, "default": str}
        dump_args.update(kwargs)
        return json.dumps(self.to_dict(), **dump_args)

    @classmethod
    def from_dict(cls, data: dict):
        """从字典构建对象 (反序列化)

        如果直接从 Block 调用，则作为工厂方法，根据 type 字段分发给具体的子类。
        如果是子类调用，则必须由子类实现具体逻辑。
        """
        if cls is Block:
            # 工厂模式
            block_type = data.get("type")
            if not block_type:
                # 尝试推断
                if "blocks" in data:
                    block_type = "document"
                elif "text" in data and "level" in data:
                    block_type = "header"
                elif "text" in data and "code" in data:
                    block_type = "text"
                else:
                    raise ValueError("Cannot infer block type from data.")

            # 查找对应的子类
            for subclass in cls.__subclasses__():
                # 递归查找所有子类（包括子类的子类，如 Document）
                # 这里简单处理，假设只有一层继承，或者 Document 在直接子类中
                # 实际上 __subclasses__ 只返回直接子类。
                # Document 继承 Block，所以 Document 是 Block 的直接子类。
                
                # 检查 type_name
                if getattr(subclass, "type_name", "") == block_type:
                    return subclass.from_dict(data)
            
            # 兼容旧代码的 name_map 方式 (Optional)
            name_map = {
                "HeaderBlock": "header",
                "TextBlock": "text",
                "MarkdownBlock": "markdown",
                "TableBlock": "table",
                "ImageBlock": "image",
                "JsonBlock": "json",
                "HtmlBlock": "raw_html",
                "Document": "document",
            }
            for subclass in cls.__subclasses__():
                if name_map.get(subclass.__name__) == block_type:
                    return subclass.from_dict(data)

            raise ValueError(f"Unknown or unsupported block type: {block_type}")
        else:
            raise NotImplementedError(f"{cls.__name__} must implement from_dict.")

    @classmethod
    def from_json(cls, json_str: str):
        """从 JSON 字符串构建"""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path, encoding="utf-8"):
        """从文件加载"""
        path = pathlib.Path(path)
        if path.suffix == ".json":
            return cls.from_json(path.read_text(encoding=encoding))
        raise NotImplementedError(f"Cannot load {cls.__name__} from {path.suffix} file.")

    def get_fingerprint(self):
        """获取内容指纹"""
        return get_hash(self.to_text())[:4]

    def get_default_filename(self, suffix):
        """生成默认文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        fingerprint = self.get_fingerprint()
        if not suffix.startswith("."):
            suffix = "." + suffix
        return f"{timestamp}_{fingerprint}{suffix}"

    def to_temp_file(self, content, suffix, name=None):
        """写入临时文件"""
        temp_dir = pathlib.Path(tempfile.gettempdir()) / "pyxllib_browser"
        temp_dir.mkdir(parents=True, exist_ok=True)

        if name is None:
            name = self.get_default_filename(suffix)

        if not pathlib.Path(name).suffix and suffix:
            if not suffix.startswith("."):
                suffix = "." + suffix
            name += suffix

        file_path = temp_dir / name

        if isinstance(content, bytes):
            file_path.write_bytes(content)
        else:
            file_path.write_text(str(content), encoding="utf-8")

        return file_path

    def print(self, *, fmt="text", **kwargs):
        """输出到控制台"""
        if fmt == "html":
            print(self.to_html(**kwargs))
        elif fmt == "md":
            print(self.to_md(**kwargs))
        else:
            print(self.to_text(**kwargs))
        return self

    def browse(self, name=None, *, fmt="html", wait=False, **kwargs):
        """在浏览器中打开"""
        from pyxllib.prog.browser import browser as open_in_browser

        if fmt == "html":
            content = self.to_html(**kwargs)
            suffix = ".html"
        elif fmt == "md":
            content = self.to_md(**kwargs)
            suffix = ".md"
        else:
            content = self.to_text(**kwargs)
            suffix = ".txt"

        file_path = self.to_temp_file(content, suffix, name=name)
        open_in_browser(str(file_path), wait=wait, **kwargs)
        return self

    def to_file(self, path=None, name=None, *, fmt=None, encoding="utf-8", **kwargs):
        """保存到文件"""
        if path is None:
            if fmt is None:
                fmt = "html"
        else:
            path = pathlib.Path(path)
            if fmt is None:
                suffix = path.suffix.lower()
                if suffix in (".html", ".htm"):
                    fmt = "html"
                elif suffix in (".md", ".markdown"):
                    fmt = "md"
                elif suffix == ".json":
                    fmt = "json"
                else:
                    fmt = "text"

        if fmt == "html":
            content = self.to_html(**kwargs)
            suffix = ".html"
        elif fmt == "md":
            content = self.to_md(**kwargs)
            suffix = ".md"
        elif fmt == "json":
            content = self.to_json(**kwargs)
            suffix = ".json"
        else:
            content = self.to_text(**kwargs)
            suffix = ".txt"

        if path is None:
            return self.to_temp_file(content, suffix=suffix, name=name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding=encoding)
            return path


class HeaderBlock(Block):
    """标题块"""
    type_name = "header"

    def __init__(self, text: str, level: int = 1):
        """
        :param str text: 标题文本
        :param int level: 标题级别 (1-6)
        """
        self.text = str(text)
        self.level = max(1, min(6, level))

    def to_html(self, **kwargs):
        anchor = hashlib.md5(self.text.encode()).hexdigest()[:8]
        return f'<h{self.level} id="{anchor}">{html.escape(self.text)}</h{self.level}>'

    def to_md(self, **kwargs):
        return f"{'#' * self.level} {self.text}\n"

    def to_text(self, **kwargs):
        underline = "=" if self.level == 1 else "-"
        return f"\n{self.text}\n{underline * len(self.text)}\n"

    def to_dict(self, **kwargs):
        return {"type": self.type_name, "text": self.text, "level": self.level}

    @classmethod
    def from_dict(cls, data):
        return cls(text=data["text"], level=data.get("level", 1))


class TextBlock(Block):
    """普通文本或代码块"""
    type_name = "text"

    def __init__(self, text: Any, code: bool = False, language: str = None):
        """
        :param Any text: 文本内容
        :param bool code: 是否为代码块
        :param str language: 代码语言（仅当 code=True 时有效）
        """
        self.text = str(text)
        self.code = code
        self.language = language

    def to_html(self, **kwargs):
        if self.code:
            lang_attr = f' class="language-{self.language}"' if self.language else ""
            return f"<pre><code{lang_attr}>{html.escape(self.text)}</code></pre>"
        safe_text = html.escape(self.text).replace("\n", "<br>")
        return f"<p>{safe_text}</p>"

    def to_md(self, **kwargs):
        if self.code:
            lang_tag = self.language if self.language else ""
            return f"```{lang_tag}\n{self.text}\n```\n"
        return f"{self.text}\n"

    def to_text(self, **kwargs):
        return f"{self.text}\n"

    def to_dict(self, **kwargs):
        return {"type": self.type_name, "text": self.text, "code": self.code, "language": self.language}

    @classmethod
    def from_dict(cls, data):
        return cls(text=data["text"], code=data.get("code", False), language=data.get("language"))


class MarkdownBlock(Block):
    """原生 Markdown 块"""
    type_name = "markdown"

    def __init__(self, text: str):
        self.text = str(text)

    def to_html(self, **kwargs):
        # 简单处理：作为 raw text 显示，或者如果引入 markdown 库可以渲染
        # 这里保持原样，用 div 包裹
        return (
            f'<div class="markdown-block" style="border-left: 3px solid #6c757d; padding-left: 1rem; color: #495057;">'
            f"<pre>{html.escape(self.text)}</pre>"
            f'<small style="color: #adb5bd;">(Markdown Raw)</small>'
            f"</div>"
        )

    def to_md(self, **kwargs):
        return f"{self.text}\n"

    def to_text(self, **kwargs):
        return f"{self.text}\n"

    def to_dict(self, **kwargs):
        return {"type": self.type_name, "text": self.text}

    @classmethod
    def from_dict(cls, data):
        return cls(text=data["text"])


class TableBlock(Block):
    """表格块"""
    type_name = "table"

    def __init__(self, data: Any, title: str = None, show_row_index: bool = False, show_col_index: bool = True):
        self.title = title
        self.show_row_index = show_row_index
        self.show_col_index = show_col_index
        if isinstance(data, pd.DataFrame):
            self.df = data
        elif isinstance(data, (list, tuple)) and data:
            try:
                self.df = pd.DataFrame(data)
            except:
                self.df = pd.DataFrame([str(data)])
        else:
            self.df = pd.DataFrame([str(data)])

    def to_html(self, **kwargs):
        caption = f"<caption>{html.escape(self.title)}</caption>" if self.title else ""
        table_html = self.df.to_html(
            classes="table table-striped table-hover table-sm",
            border=0,
            escape=True,
            index=self.show_row_index,
            header=self.show_col_index,
        )
        if caption:
            table_html = table_html.replace("<table", f"<table", 1).replace(">", f">{caption}", 1)
        return f'<div class="table-responsive">{table_html}</div>'

    def to_md(self, **kwargs):
        title = f"**{self.title}**\n" if self.title else ""
        try:
            headers = "keys" if self.show_col_index else []
            return title + self.df.to_markdown(index=self.show_row_index, headers=headers) + "\n"
        except (ImportError, AttributeError):
            return title + self.df.to_string(index=self.show_row_index, header=self.show_col_index) + "\n"

    def to_text(self, **kwargs):
        title = f"[{self.title}]\n" if self.title else ""
        return title + self.df.to_string(index=self.show_row_index, header=self.show_col_index) + "\n"

    def to_dict(self, **kwargs):
        return {
            "type": self.type_name,
            "title": self.title,
            "data": self.df.to_dict(orient="split"),
            "show_row_index": self.show_row_index,
            "show_col_index": self.show_col_index,
        }

    @classmethod
    def from_dict(cls, data):
        df_data = data.get("data", {})
        if isinstance(df_data, dict) and "data" in df_data and "index" in df_data and "columns" in df_data:
            try:
                df = pd.DataFrame(df_data["data"], index=df_data["index"], columns=df_data["columns"])
            except Exception:
                df = pd.DataFrame(df_data)
        else:
            df = pd.DataFrame(df_data)

        return cls(
            data=df,
            title=data.get("title"),
            show_row_index=data.get("show_row_index", False),
            show_col_index=data.get("show_col_index", True),
        )


class ImageBlock(Block):
    """支持路径或Bytes的图片块"""
    type_name = "image"

    def __init__(self, src: Union[str, bytes], title: str = None):
        self.title = title
        self.src = src

    def _get_b64(self):
        if isinstance(self.src, bytes):
            return base64.b64encode(self.src).decode("utf-8")
        elif isinstance(self.src, (str, pathlib.Path)):
            if str(self.src).startswith("http"):
                return None
            if os.path.exists(self.src):
                with open(self.src, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
        return None

    def to_html(self, **kwargs):
        caption = f"<figcaption>{html.escape(self.title)}</figcaption>" if self.title else ""
        b64 = self._get_b64()
        if b64:
            src_str = f"data:image/png;base64,{b64}"
        else:
            src_str = str(self.src)
        
        # Add download attribute if title is present
        download_attr = f' download="{html.escape(self.title)}.png"' if self.title else ""
        title_attr = f' title="{html.escape(self.title)}"' if self.title else ""
        
        # Wrap in anchor tag for click-to-download/view
        img_tag = f'<img src="{src_str}" style="max-width:100%; height:auto;"{title_attr} />'
        
        # If it's a base64 image, we can make it downloadable easily
        if b64:
             return f'<figure><a href="{src_str}"{download_attr}>{img_tag}</a>{caption}</figure>'
        
        return f'<figure>{img_tag}{caption}</figure>'

    def to_md(self, **kwargs):
        return f"![{self.title or 'image'}]({self.src})\n"

    def to_text(self, **kwargs):
        return f"[Image: {self.title or str(self.src)[:20]}]\n"

    def to_dict(self, **kwargs):
        return {
            "type": self.type_name,
            "src": str(self.src) if not isinstance(self.src, bytes) else None,
            "base64": self._get_b64(),
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data):
        b64 = data.get("base64")
        if b64:
            src = base64.b64decode(b64)
        else:
            src = data.get("src")
        return cls(src=src, title=data.get("title"))


class JsonBlock(Block):
    """JSON 数据块"""
    type_name = "json"

    def __init__(self, data: Any, title: str = None, max_items=10, max_value_length=100):
        self.title = title
        self.max_items = max_items
        self.max_value_length = max_value_length

        if isinstance(data, (str, pathlib.Path)):
            file_path = pathlib.Path(data)
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding="utf-8")
                    self.data = json.loads(content)
                    if not self.title:
                        self.title = file_path.name
                except Exception as e:
                    self.data = {"error": f"Failed to load json file: {e}", "path": str(file_path)}
            else:
                try:
                    self.data = json.loads(str(data))
                except json.JSONDecodeError:
                    self.data = str(data)
        else:
            self.data = data

    def to_html(self, **kwargs):
        from pyxllib.prog.specialist.common import NestedDict
        html_table = NestedDict.to_html_table(self.data, max_items=self.max_items)
        caption = f"<figcaption><strong>{html.escape(self.title)}</strong></figcaption>" if self.title else ""
        return f'<figure class="json-block">{caption}{html_table}</figure>'

    def to_md(self, **kwargs):
        json_str = json.dumps(self.data, ensure_ascii=False, indent=2, default=str)
        title_str = f"**{self.title}**\n" if self.title else ""
        return f"{title_str}```json\n{json_str}\n```\n"

    def to_text(self, **kwargs):
        title_str = f"[{self.title}]\n" if self.title else ""
        return f"{title_str}{json.dumps(self.data, ensure_ascii=False, indent=2, default=str)}\n"

    def to_dict(self, **kwargs):
        return {
            "type": self.type_name,
            "data": self.data,
            "title": self.title,
            "max_items": self.max_items,
            "max_value_length": self.max_value_length,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            data=data.get("data"),
            title=data.get("title"),
            max_items=data.get("max_items", 10),
            max_value_length=data.get("max_value_length", 100),
        )


class HtmlBlock(Block):
    """原始 HTML 块"""
    type_name = "raw_html"

    def __init__(self, html_content: str):
        self.html_content = str(html_content)

    def to_html(self, **kwargs):
        return self.html_content

    def to_md(self, **kwargs):
        return self.html_content + "\n"

    def to_text(self, **kwargs):
        return "[Raw HTML Block]\n"

    def to_dict(self, **kwargs):
        return {"type": self.type_name, "content": self.html_content}

    @classmethod
    def from_dict(cls, data):
        return cls(html_content=data["content"])


def __2_Document_Components():
    """文档组件类"""
    pass


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


class Document(Block):
    """通用文档构建器
    
    集成内容管理（Builder）和格式渲染（Renderer）。
    """
    type_name = "document"

    def __init__(self, content=None, title=None, **kwargs):
        """
        :param Any content: 初始内容
        :param str title: 文档标题
        """
        self.title = title
        self.blocks: List[Block] = []
        self.css = DEFAULT_CSS

        if content is not None:
            self.add(content, **kwargs)

    def add(self, content: Any, title: str = None, **kwargs):
        """添加内容"""
        if isinstance(content, Block):
            self.blocks.append(content)
        elif hasattr(content, "to_document") and callable(content.to_document):
            if title is not None:
                kwargs["title"] = title
            self.blocks.append(content.to_document(**kwargs))
        elif isinstance(content, pd.DataFrame):
            self.blocks.append(TableBlock(content, title=title))
        elif isinstance(content, (list, tuple, dict)) and not isinstance(content, (str, bytes)):
            try:
                self.blocks.append(TableBlock(pd.DataFrame(content), title=title))
            except:
                self.blocks.append(TextBlock(str(content)))
        else:
            self.blocks.append(TextBlock(str(content)))
        return self

    def add_header(self, text, level=1):
        self.blocks.append(HeaderBlock(text, level))
        return self

    def add_text(self, text, code=False):
        self.blocks.append(TextBlock(text, code))
        return self

    def add_code(self, text, language=None):
        self.blocks.append(TextBlock(text, code=True, language=language))
        return self

    def add_html(self, html_content):
        self.blocks.append(HtmlBlock(html_content))
        return self

    def add_md(self, md_content):
        self.blocks.append(MarkdownBlock(md_content))
        return self

    def add_echart(self, chart):
        if hasattr(chart, "render_embed"):
            self.blocks.append(HtmlBlock(chart.render_embed()))
        else:
            self.blocks.append(TextBlock(str(chart)))
        return self

    def _process_row_index(self, df, row_index):
        show_row_index = False
        if row_index is not None:
            if isinstance(row_index, int) and not isinstance(row_index, bool):
                df.index = pd.RangeIndex(start=row_index, stop=row_index + len(df), step=1)
                show_row_index = True
            elif row_index is True:
                show_row_index = True
        return df, show_row_index

    def _process_col_index(self, df, col_index):
        show_col_index = True
        if col_index is not None:
            if isinstance(col_index, (list, tuple)):
                if len(col_index) == len(df.columns):
                    df.columns = col_index
                show_col_index = True
            elif col_index is False:
                show_col_index = False
        return df, show_col_index

    def _add_table_common(self, df, title, row_index, col_index):
        df = df.copy()
        df, show_row_index = self._process_row_index(df, row_index)
        df, show_col_index = self._process_col_index(df, col_index)
        self.blocks.append(TableBlock(df, title, show_row_index=show_row_index, show_col_index=show_col_index))

    def _add_table_from_list(self, data, title, row_index, col_index):
        df = pd.DataFrame(data)
        if isinstance(df.columns, pd.RangeIndex) and len(df.columns) > 0:
            if col_index is True or col_index is None:
                df.columns = [f"col{i}" for i in df.columns]
        self._add_table_common(df, title, row_index, col_index)

    def add_table(self, data, title=None, row_index=None, col_index=True):
        if isinstance(data, pd.DataFrame):
            self._add_table_common(data, title, row_index, col_index)
        elif isinstance(data, (list, tuple)):
            self._add_table_from_list(data, title, row_index, col_index)
        elif isinstance(data, dict):
            self._add_table_common(pd.DataFrame(data), title, row_index, col_index)
        else:
            self.blocks.append(TableBlock(data, title, show_row_index=bool(row_index), show_col_index=bool(col_index)))
        return self

    def add_json(self, data, title=None, max_items=10, max_value_length=100):
        self.blocks.append(JsonBlock(data, title, max_items=max_items, max_value_length=max_value_length))
        return self

    def add_image(self, src, title=None):
        self.blocks.append(ImageBlock(src, title))
        return self

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

    def to_html(self, heading_numbering=None, full_page=True, **kwargs) -> str:
        """转换为 HTML
        
        :param heading_numbering: 是否开启标题编号
        :param full_page: 是否生成完整的 HTML 页面（包含 head, body）。
                          如果是嵌套在其他文档中，应设为 False。
        """
        body_parts = []
        for b in self._get_render_blocks(heading_numbering):
            # 如果 Block 是 Document，则递归调用并设置 full_page=False
            if isinstance(b, Document):
                body_parts.append(b.to_html(heading_numbering=heading_numbering, full_page=False, **kwargs))
            else:
                body_parts.append(b.to_html(**kwargs))
        
        body_content = "\n".join(body_parts)

        if not full_page:
            # 作为片段返回
            h1_tag = f"<h2>{html.escape(str(self.title))}</h2>" if self.title else ""
            return f'<div class="sub-document">\n{h1_tag}\n{body_content}\n</div>'

        # 生成完整页面
        title_str = html.escape(str(self.title)) if self.title else "Document"
        h1_tag = f"<h1>{title_str}</h1>" if self.title else ""

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

    def to_md(self, heading_numbering=None, **kwargs) -> str:
        content = "\n\n".join([b.to_md(**kwargs) for b in self._get_render_blocks(heading_numbering)])
        if self.title:
            return f"# {self.title}\n\n{content}"
        else:
            return content

    def to_text(self, heading_numbering=None, **kwargs) -> str:
        content = "\n".join([b.to_text(**kwargs) for b in self._get_render_blocks(heading_numbering)])
        if self.title:
            return f"=== {self.title} ===\n\n{content}"
        else:
            return content

    def to_dict(self, **kwargs):
        return {"type": self.type_name, "title": self.title, "blocks": [b.to_dict() for b in self.blocks]}

    @classmethod
    def from_dict(cls, data):
        doc = cls(title=data.get("title"))
        for block_data in data.get("blocks", []):
            doc.add(Block.from_dict(block_data))
        return doc


class DocumentableMixin:
    """具备直接导出文档能力的对象混入类"""

    def to_document(self, **kwargs):
        raise NotImplementedError

    def _smart_dispatch(self, render_method_name, **kwargs):
        try:
            sig = inspect.signature(self.to_document)
        except ValueError:
            gen_args = {}
            render_args = kwargs
        else:
            gen_args = {}
            render_args = {}
            params = sig.parameters
            for k, v in kwargs.items():
                if k in params:
                    gen_args[k] = v
                else:
                    render_args[k] = v

        doc = self.to_document(**gen_args)

        if not hasattr(doc, render_method_name):
            raise AttributeError(f"Document object has no attribute '{render_method_name}'")

        return getattr(doc, render_method_name)(**render_args)

    def to_text(self, **kwargs):
        return self._smart_dispatch("to_text", **kwargs)

    def to_html(self, **kwargs):
        return self._smart_dispatch("to_html", **kwargs)

    def to_md(self, **kwargs):
        return self._smart_dispatch("to_md", **kwargs)

    def print(self, **kwargs):
        return self._smart_dispatch("print", **kwargs)

    def browse(self, **kwargs):
        return self._smart_dispatch("browse", **kwargs)


if __name__ == "__main__":
    d = Document(title=None)
    d.add_header("1 数据概览")
    d.add_text("这是一段测试文本，包含\n换行符。")

    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    d.add_table(df)

    d.add_header("2 代码示例", level=2)
    d.add_text("print('hello world')", code=True)

    # Test nesting
    sub_doc = Document(title="Sub Document")
    sub_doc.add_text("This is inside a sub document.")
    d.add(sub_doc)

    # d.browse()
    print(d.to_text())
