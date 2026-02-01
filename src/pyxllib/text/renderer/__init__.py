from functools import singledispatch
import html
import pprint


@singledispatch
def to_str(data):
    """通用兜底：使用 pprint"""
    return pprint.pformat(data)


@singledispatch
def to_html(data, *args, **kwargs):
    """通用兜底：转义后放入 <pre>"""
    return f"<pre>{html.escape(to_str(data))}</pre>"


@to_html.register(dict)
def _(data, *args, **kwargs):
    """字典的 HTML 处理"""
    from pyxllib.prog.specialist.common import NestedDict

    return NestedDict.to_html_table(data)


@to_html.register(list)
def _(data, *args, **kwargs):
    """列表的 HTML 处理：递归调用 to_html"""
    items = [f"<li>{to_html(x, *args, **kwargs)}</li>" for x in data]
    return f"<ul>{''.join(items)}</ul>"
