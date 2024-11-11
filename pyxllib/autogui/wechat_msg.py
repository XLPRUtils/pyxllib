#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/11/05

"""
微信 消息框中不同消息结构的解析器 工具
"""

import re
import json

# 定义全局字典
msg_parsers = {}


# 定义装饰器
def register_tool(key):
    def decorator(func):
        # 将函数添加到全局字典中，键为传入的key参数
        msg_parsers[key] = func
        return func

    return decorator


def fmt_text(text):
    return text.replace('\n', ' ')


def parse_time_string(time_str):
    """
    解析不同格式的时间字符串为标准的 Python datetime 类型。

    支持以下格式：
    1. 完整的日期和时间，如 "2024年10月31日 22:03"，解析为对应的 datetime 对象。
    2. 仅包含时间的字符串，如 "19:18"，会被解析为当天的时间点，日期部分设为当前日期。
    3. 特殊字符串 "以下为新消息"，解析为当前时间的 datetime 对象。

    示例输入：
    - "2024年10月31日 22:03"
    - "2024年11月1日 20:09"
    - "2024年11月1日 21:08"
    - "19:18"
    - "20:16"
    - "以下为新消息"
    """
    from datetime import datetime

    # 定义当前日期
    now = datetime.now()

    # 定义不同的正则表达式模式
    full_date_pattern = r"(\d{4})年(\d{1,2})月(\d{1,2})日\s+(\d{1,2}):(\d{2})"
    partial_time_pattern = r"^(\d{1,2}):(\d{2})$"
    new_message_pattern = r"^以下为新消息$"

    # 解析完整的日期时间
    match = re.match(full_date_pattern, time_str)
    if match:
        year, month, day, hour, minute = map(int, match.groups())
        return datetime(year, month, day, hour, minute)

    # 解析只有时间（小时:分钟），默认当天的日期
    match = re.match(partial_time_pattern, time_str)
    if match:
        hour, minute = map(int, match.groups())
        return datetime(now.year, now.month, now.day, hour, minute)

    # 解析 "以下为新消息" 为当前时间
    if re.match(new_message_pattern, time_str):
        return now

    # 如果未匹配到任何模式，返回 None 表示无法解析
    return None


@register_tool("1b2p3p3b3p")
def 系统_查看更多消息(node):
    node.msg_type = 'system'
    node.content_type = 'button_more'
    node.render_text = f'⚙️: 点击此处可 {node.text}'


@register_tool("1l2t")
def 系统_时间标签(node):
    node.msg_type = 'system'
    node.content_type = 'time'
    node.time = parse_time_string(node.text)
    node.render_text = f'⚙️: {node.text}'


# 刚刚发送的特殊时间标签
@register_tool("1l2p3p3p3t3p3p")
def 系统_时间标签2(node):
    node.msg_type = 'system'
    node.content_type = 'time'
    node.time = parse_time_string(node.text)
    node.render_text = f'⚙️: {node.text}'


@register_tool("1l2p3p3t3p")
def 系统_撤回消息(node):
    node.msg_type = 'system'

    if node.text.endswith('撤回了一条消息'):
        node.content_type = 'recall'
        if node.text == '你撤回了一条消息':
            node.user = node.user2 = '你'
        else:
            node.user = re.search(r'"(.+)"', node.text).group(1)
            node.user2 = node.user
        node.render_text = f'⚙️: {node.text}'
    elif node.text.endswith('现在可以开始聊天了。'):
        node.content_type = 'add_friend'
        node.user = re.search(r'你已添加了(.+?)，现在可以开始聊天了。', node.text).group(1)
        node.user2 = node.user
        node.render_text = f'⚙️: {node.text}'
    else:
        raise ValueError


@register_tool("1l2p3p3p4p5p6p7t3b")
def 发送_文本(node):
    node.msg_type = 'send'
    node.content_type = 'text'
    node.user = node[0][2].text
    node.render_text = f'↑{node.user}: {fmt_text(node.text)}'


@register_tool("1l2p3p3p4p5p6p7p7p8p8b3b")
def 发送_图片(node):
    node.msg_type = 'send'
    node.content_type = 'image'
    node.user = node[0][2].text


@register_tool("1l2p3p3p4p5p6p7p7p8p9t9p9b8b3b")
def 发送_视频(node):
    node.msg_type = 'send'
    node.content_type = 'video'
    node.user = node[0][2].text


@register_tool("1l2p3p3p4p5p6p7p7p8p9p10p11t6p7p8p9p10p11t7b3b")
def 发送_文本_引用文本(node):
    node.msg_type = 'send'
    node.content_type = 'text'
    node.user = node[0][2].text
    quoted_node = node[0][1][0][0]  # 引用的层级结构
    node.text = quoted_node[0][1][0][0][0][0].text
    node.cite_text = quoted_node[1][0][0][0][0][0].text


@register_tool("1l2p3p3p4p5p6p7p7p8p9p10p11t6p7p8p9p10p11t11t11p12b7b3b")
def 发送_文本_引用图片(node):
    node.msg_type = 'send'
    node.content_type = 'text'
    node.user = node[0][2].text
    quoted_node = node[0][1][0][0]
    node.text = quoted_node[0][1][0][0][0][0].text
    node.cite_text = f'{quoted_node[1][0][0][0][0][0].text} : [图片]'


# 引用文件
@register_tool("1l2p3p3p4p5p6p7p7p8p9p10p11t6p7p8p9p10p11p12t11p12b7b3b")
def 发送_文本_引用文件(node):
    node.msg_type = 'send'
    node.content_type = 'text'
    node.user = node[0][2].text
    quoted_node = node[0][1][0][0]
    node.text = quoted_node[0][1][0][0][0][0].text
    node.cite_text = f'[文件] {quoted_node[1][0][0][0][0][0][0].text}'


# 发送文件
@register_tool("1l2p3p3p4p5p6p7p8p9p10t10p11t11t9p10p10p8p9t7b7b3b")
def 发送_文本_引用文件(node):
    node.msg_type = 'send'
    node.content_type = 'file'
    node.user = node[0][2].text
    file_node = node[0][1][0][0][0][0]
    desc = {
        'name': file_node[0][0][0].text,
        'size': file_node[0][0][1][0].text,
        'platform': file_node[1][0].text
    }
    node.text = json.dumps(desc, ensure_ascii=False)


# 发送链接
@register_tool("1l2p3p3p4p5p6p7p8t8p9t9p9b8p9b9t7b3b")
def 发送_链接(node):
    node.msg_type = 'send'
    node.content_type = 'link'
    node.user = node[0][2].text
    link_node = node[0][1][0][0]
    desc = {
        'title': link_node[0][0][0].text,
        'head': link_node[0][0][1][0].text,
        'author': link_node[0][0][2][1].text
    }
    node.text = json.dumps(desc, ensure_ascii=False)


# 转发消息
@register_tool("1l2p3p3p4p5p6p7p8t8p9t9t9t8p9p9t7b3b")
def 发送_转发消息(node):
    node.msg_type = 'send'
    node.content_type = 'messages'
    node.user = node[0][2].text


@register_tool("1l2p3b3p4p5p6p7t3p")
def 接收_文本(node):
    node.msg_type = 'receive'
    node.content_type = 'text'
    node.user = node[0][0].text
    node.render_text = f'↓{node.user}: {fmt_text(node.text)}'


@register_tool("1l2p3b3p4p5p6p7p7b3p")
def 接收_图片(node):
    node.msg_type = 'receive'
    node.content_type = 'image'
    node.user = node[0][0].text
    node.render_text = f'↓{node.user}: [图片]'


@register_tool("1l2p3b3p4p5p6p7p8p9p10t10p11t11t9p10p10p8p9t7b7b3p")
def 接收_文件(node):
    node.msg_type = 'receive'
    node.content_type = 'file'
    node.user = node[0][0].text
    node.render_text = f'↓{node.user}: {fmt_text(node.text)}'


@register_tool("1l2p3b3p4p5p6p7p8b8p8t7b3p")
def 接收_语音(node):
    node.msg_type = 'receive'
    node.content_type = 'voice'
    node.user = node[0][0].text


@register_tool("1l2p3b3p4p5p6p7p8t8p9t9p9b8p9b9t7b3p")
def 接收_链接(node):
    node.msg_type = 'receive'
    node.content_type = 'link'
    node.user = node[0][0].text
    link_node = node[0][1][0][0][0][0]  # 链接内容的层级结构
    desc = {
        'title': link_node[0].text,
        'head': link_node[1][0].text,  # 文章开头部分
        'author': link_node[2][1].text
    }
    node.text = json.dumps(desc, ensure_ascii=False)


@register_tool("1l2p3b3p4p5t4p5p6p7t3p")
def 群接收_文本(node):
    node.msg_type = 'receive'
    node.content_type = 'text'
    node.user = node[0][0].text
    node.user2 = node[0][1][0][0].text


@register_tool("1l2p3b3p4p5t4p5p6p7p7b3p")
def 群接收_图片(node):
    node.msg_type = 'receive'
    node.content_type = 'image'
    node.user = node[0][0].text
    node.user2 = node[0][1][0][0].text


@register_tool("1l2p3b3p4p5t4p5p6p7p8p9p10t10p11t11t9p10p10p7b7b3p")
def 群接收_文件(node):
    node.msg_type = 'receive'
    node.content_type = 'file'
    node.user = node[0][0].text
    file_node = node[0][1]  # 文件信息的节点
    node.user2 = file_node[0][0].text
    file_info_node = file_node[1][0][0][0][0]
    desc = {
        'name': file_info_node[0][0].text,
        'size': file_info_node[0][1][0].text
    }
    node.text = json.dumps(desc, ensure_ascii=False)


@register_tool("1l2p3b3p4p5t4p5p6p7p8p9p10t10p11t11t9p10p10p8p9t7b7b3p")
def 群接收_文件2(node):
    node.msg_type = 'receive'
    node.content_type = 'file'
    node.user = node[0][0].text
    node.user2 = node[0][1][0][0].text
    file_info_node = node[0][1][1][0][0][0]
    desc = {
        'name': file_info_node[0][0][0].text,
        'size': file_info_node[0][0][1][0].text,
        'platform': file_info_node[1][0].text
    }
    node.text = json.dumps(desc, ensure_ascii=False)


@register_tool("1l2p3b3p4p5t4p5p6p7b3p")
def 群接收_动画表情(node):
    node.msg_type = 'receive'
    node.content_type = 'emoji'
    node.user = node[0][0].text
    node.user2 = node[0][1][0][0].text


@register_tool("1l2p3b3p4p5t4p5p6p7p8t8p9t9p9b8p9b9t7b3p")
def 群接收_链接_有公众号作者名(node):
    node.msg_type = 'receive'
    node.content_type = 'link'
    node.user = node[0][0].text
    link_node = node[0][1]
    node.user2 = link_node[0][0].text
    link_info_node = link_node[1][0][0][0]
    desc = {
        'title': link_info_node[0].text,
        'head': link_info_node[1][0].text,  # 文章开头的一小部分
        'author': link_info_node[2][0].text
    }
    node.text = json.dumps(desc, ensure_ascii=False)


@register_tool("1l2p3b3p4p5t4p5p6p7p8t8p9t9p9b7b3p")
def 群接收_链接_无公众号作者名(node):
    node.msg_type = 'receive'
    node.content_type = 'link'
    node.user = node[0][0].text
    link_node = node[0][1]
    node.user2 = link_node[0][0].text
    link_info_node = link_node[1][0][0][0]
    desc = {
        'title': link_info_node[0].text,
        'head': link_info_node[1][0].text  # 文章开头的一小部分
    }
    node.text = json.dumps(desc, ensure_ascii=False)


@register_tool("1l2p3b3p4p5t4p5p6p7p8b8p9p8p9p10p10t8p9p7b3p")
def 群接收_视频(node):
    node.msg_type = 'receive'
    node.content_type = 'video'
    node.user = node[0][0].text
    video_node = node[0][1]
    node.user2 = video_node[0][0].text
    video_info_node = video_node[1][0][0][0]
    desc = {
        'author': video_info_node[2][0][1].text
    }
    node.text = json.dumps(desc, ensure_ascii=False)


@register_tool("1l2p3b3p4p5t4p5p6p7p8t8p9t8t7b3p")
def 群接收_视频2(node):
    node.msg_type = 'receive'
    node.content_type = 'video'
    node.user = node[0][0].text
    video_node = node[0][1]
    node.user2 = video_node[0][0].text
    video_info_node = video_node[1][0][0][0]
    desc = {
        'title': video_info_node[0].text,
        'head': video_info_node[1][0].text,
        'duration': video_info_node[2].text
    }
    node.text = json.dumps(desc, ensure_ascii=False)


@register_tool("1l2p3b3p4p5t4p5p6p7p8p9b9t8t8b8p9t9t7b3p")
def 群接收_小程序(node):
    node.msg_type = 'receive'
    node.content_type = 'applet'
    node.user = node[0][0].text
    applet_node = node[0][1]
    node.user2 = applet_node[0][0].text
    applet_info_node = applet_node[1][0][0][0]
    desc = {
        'name': applet_info_node[0][1].text,
        'title': applet_info_node[1].text,
        'footnote': applet_info_node[3][1].text
    }
    node.text = json.dumps(desc, ensure_ascii=False)


@register_tool("1l2p3p3p4p5l6p7p7t7p3p")
def 群接收_拍一拍(node):
    node.msg_type = 'receive'
    node.content_type = 'shake'
    node.text = node[0][1][0][0].text
    node.user = re.search(r'"(.+?)" 拍了拍 "', node.text).group(1)


@register_tool("1l2p3b3p4p5t4p5p6p7p8t4p5p6p7p8p9t5b3p")
def 群接收_引用消息(node):
    node.msg_type = 'receive'
    node.content_type = 'text'
    node.user = node[0][0].text
    quoted_node = node[0][1]
    node.user2 = quoted_node[0][0].text
    node.text = quoted_node[1][0][0][0][0].text
    node.cite_text = quoted_node[2][0][0][0][0][0].text


@register_tool("1l2p3b3p4p5t4p5p6p7p8t4p5p6p7p8p9p10t9p10b5b3p")
def 群接收_引用文件(node):
    node.msg_type = 'receive'
    node.content_type = 'text'
    node.user = node[0][0].text
    quoted_node = node[0][1]
    node.user2 = quoted_node[0][0].text
    node.text = quoted_node[1][0][0][0][0].text
    file_node = quoted_node[2][0][0][0][0][0][0]
    node.cite_text = f'[文件] {file_node.text}'
