#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/08/09

import os
import json
from json import dumps as jsonDumps
from json import loads as jsonLoads
from pathlib import Path
from typing import List, Literal
import base64
import logging
from unittest.mock import MagicMock
import math
from types import SimpleNamespace
from typing import Union

import requests
from loguru import logger as loguru_logger
from tqdm import tqdm
from joblib import Parallel, delayed
from openai import OpenAI, AsyncOpenAI

from pyxllib.text.jinjalib import set_template
from pyxllib.file.specialist import download_file


def __1_Chat类的基础功能():
    pass


class AKPool:
    """ 轮询获取api_key """

    def __init__(self, apikeys: list):
        self._pool = self._POOL(apikeys)

    def fetch_key(self):
        return next(self._pool)

    @classmethod
    def _POOL(cls, apikeys: list):
        while True:
            for x in apikeys:
                yield x


class MsgBase:
    role_name: str
    text: str

    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text

    def __iter__(self):
        yield "role", self.role_name
        yield "content", self.text


system_msg = type("system_msg", (MsgBase,), {"role_name": "system"})
user_msg = type("user_msg", (MsgBase,), {"role_name": "user"})
assistant_msg = type("assistant_msg", (MsgBase,), {"role_name": "assistant"})


class Temque:
    """ 一个先进先出, 可设置最大容量, 可固定元素的队列 """

    def __init__(self, maxlen: int = None):
        self.core: List[dict] = []
        self.maxlen = maxlen or float("inf")

    def _trim(self):
        core = self.core
        if len(core) > self.maxlen:
            dc = len(core) - self.maxlen
            indexes = []
            for i, x in enumerate(core):
                if not x["pin"]:
                    indexes.append(i)
                if len(indexes) == dc:
                    break
            for i in indexes[::-1]:
                core.pop(i)

    def add_many(self, *objs):
        for x in objs:
            self.core.append({"obj": x, "pin": False})
        self._trim()

    def __iter__(self):
        for x in self.core:
            yield x["obj"]

    def pin(self, *indexes):
        for i in indexes:
            self.core[i]["pin"] = True

    def unpin(self, *indexes):
        for i in indexes:
            self.core[i]["pin"] = False

    def copy(self):
        que = self.__class__(maxlen=self.maxlen)
        que.core = self.core.copy()
        return que

    def deepcopy(self):
        ...  # 创建这个方法是为了以代码提示的方式提醒用户: copy 方法是浅拷贝

    def __add__(self, obj: 'list|Temque'):
        que = self.copy()
        if isinstance(obj, self.__class__):
            que.core += obj.core
            que._trim()
        else:
            que.add_many(*obj)
        return que


class Chat:
    """
    [文档](https://lcctoor.github.io/arts/arts/openai2)

    获取api_key:
    * [获取链接1](https://platform.openai.com/account/api-keys)
    * [获取链接2](https://www.baidu.com/s?wd=%E8%8E%B7%E5%8F%96%20openai%20api_key)
    """

    recently_request_data: dict  # 最近一次请求所用的参数
    default_api_key = None
    default_base_url = "https://platform.llmprovider.ai/v1"

    def __init__(self,
                 # kwargs
                 api_key: Union[str, AKPool] = None,
                 base_url: str = None,  # base_url 参数用于修改基础URL
                 timeout=None,
                 max_retries=None,
                 http_client=None,
                 # request_kwargs
                 model: Literal["gpt-4-1106-preview", "gpt-4-vision-preview",
                 "gpt-4", "gpt-4-0314", "gpt-4-0613",
                 "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-0613", "gpt-3.5-turbo",
                 "gpt-4o", "gpt-4-turbo"] = "gpt-4o-mini",
                 # Chat
                 msg_max_count: int = None,
                 # kwargs
                 **kwargs,
                 ):
        """

        :param kwargs: 主要是供OpenAI初始化时使用的参数

        """
        api_key = api_key or self.default_api_key or os.getenv("OPENAI_API_KEY")

        api_base = kwargs.pop('api_base', None)
        base_url = base_url or api_base or self.default_base_url or os.getenv("OPENAI_BASE_URL")
        MsgMaxCount = kwargs.pop('MsgMaxCount', None)
        msg_max_count = msg_max_count or MsgMaxCount

        if base_url: kwargs["base_url"] = base_url
        if timeout: kwargs["timeout"] = timeout
        if max_retries: kwargs["max_retries"] = max_retries
        if http_client: kwargs["http_client"] = http_client

        self.reset_api_key(api_key)
        self._kwargs = kwargs
        self._request_kwargs = {'model': model}
        self._messages = Temque(maxlen=msg_max_count)

    def __1_原作者提供功能(self):
        pass

    def reset_api_key(self, api_key: Union[str, AKPool]):
        if isinstance(api_key, AKPool):
            self._akpool = api_key
        else:
            self._akpool = AKPool([api_key])

    def request(self, text: str = None, **kwargs):
        """
        :param kwargs: 支持本次请求的参数覆盖全局 self._request_kwargs 的参数
        """
        messages = [{"role": "user", "content": text}]
        messages += (kwargs.pop('messages', None) or [])  # 兼容官方包[openai]用户, 使其代码可以无缝切换到[openai2]
        assert messages
        self.recently_request_data = {
            'api_key': (api_key := self._akpool.fetch_key()),
        }
        completion = OpenAI(api_key=api_key, **self._kwargs).chat.completions.create(**{
            **self._request_kwargs,  # 全局参数
            **kwargs,  # 单次请求的参数覆盖全局参数
            "messages": list(self._messages + messages),
            "stream": False,
        })
        answer: str = completion.choices[0].message.content
        self._messages.add_many(*messages, {"role": "assistant", "content": answer})
        return answer

    def stream_request(self, text: str = None, **kwargs):
        messages = [{"role": "user", "content": text}]
        messages += (kwargs.pop('messages', None) or [])  # 兼容官方包[openai]用户, 使其代码可以无缝切换到[openai2]
        assert messages
        self.recently_request_data = {
            'api_key': (api_key := self._akpool.fetch_key()),
        }
        completion = OpenAI(api_key=api_key, **self._kwargs).chat.completions.create(**{
            **self._request_kwargs,  # 全局参数
            **kwargs,  # 单次请求的参数覆盖全局参数
            "messages": list(self._messages + messages),
            "stream": True,
        })
        answer: str = ""
        for chunk in completion:
            if chunk.choices and (content := chunk.choices[0].delta.content):
                answer += content
                yield content
        self._messages.add_many(*messages, {"role": "assistant", "content": answer})

    async def async_request(self, text: str = None, **kwargs):
        messages = [{"role": "user", "content": text}]
        messages += (kwargs.pop('messages', None) or [])  # 兼容官方包[openai]用户, 使其代码可以无缝切换到[openai2]
        assert messages
        self.recently_request_data = {
            'api_key': (api_key := self._akpool.fetch_key()),
        }
        completion = await AsyncOpenAI(api_key=api_key, **self._kwargs).chat.completions.create(**{
            **self._request_kwargs,  # 全局参数
            **kwargs,  # 单次请求的参数覆盖全局参数
            "messages": list(self._messages + messages),
            "stream": False,
        })
        answer: str = completion.choices[0].message.content
        self._messages.add_many(*messages, {"role": "assistant", "content": answer})
        return answer

    async def async_stream_request(self, text: str = None, **kwargs):
        messages = [{"role": "user", "content": text}]
        messages += (kwargs.pop('messages', None) or [])  # 兼容官方包[openai]用户, 使其代码可以无缝切换到[openai2]
        assert messages
        self.recently_request_data = {
            'api_key': (api_key := self._akpool.fetch_key()),
        }
        completion = await AsyncOpenAI(api_key=api_key, **self._kwargs).chat.completions.create(**{
            **self._request_kwargs,  # 全局参数
            **kwargs,  # 单次请求的参数覆盖全局参数
            "messages": list(self._messages + messages),
            "stream": True,
        })
        answer: str = ""
        async for chunk in completion:
            if chunk.choices and (content := chunk.choices[0].delta.content):
                answer += content
                yield content
        self._messages.add_many(*messages, {"role": "assistant", "content": answer})

    def rollback(self, n=1):
        '''
        回滚对话
        '''
        self._messages.core[-2 * n:] = []
        for x in self._messages.core[-2:]:
            x = x["obj"]
            print(f"[{x['role']}]:{x['content']}")

    def pin_messages(self, *indexes):
        '''
        锁定历史消息
        '''
        self._messages.pin(*indexes)

    def unpin_messages(self, *indexes):
        '''
        解锁历史消息
        '''
        self._messages.unpin(*indexes)

    def fetch_messages(self):
        return list(self._messages)

    def add_dialogs(self, *ms: Union[dict, system_msg, user_msg, assistant_msg]):
        '''
        添加历史对话
        '''
        messages = [dict(x) for x in ms]
        self._messages.add_many(*messages)

    def __getattr__(self, name):
        # match name:  # 兼容旧代码
        #     case 'asy_request':
        #         return self.async_request
        #     case 'forge':
        #         return self.add_dialogs
        #     case 'pin':
        #         return self.pin_messages
        #     case 'unpin':
        #         return self.unpin_messages
        #     case 'dump':
        #         return self._dump
        #     case 'load':
        #         return self._load
        # raise AttributeError(name)

        if name == 'asy_request':
            return self.async_request
        elif name == 'forge':
            return self.add_dialogs
        elif name == 'pin':
            return self.pin_messages
        elif name == 'unpin':
            return self.unpin_messages
        elif name == 'dump':
            return self._dump
        elif name == 'load':
            return self._load
        else:
            raise AttributeError(name)

    def _dump(self, fpath: str):
        """ 存档 """
        jt = jsonDumps(self.fetch_messages(), ensure_ascii=False)
        Path(fpath).write_text(jt, encoding="utf8")
        return True

    def _load(self, fpath: str):
        """ 载入存档 """
        jt = Path(fpath).read_text(encoding="utf8")
        self._messages.add_many(*jsonLoads(jt))
        return True

    def __2_扩展功能(self):
        pass

    def add_message(self, message):
        """ 添加单条历史消息记录 """
        self._messages.add_many(message)

    def add_messages(self, messages: list[dict]):
        """ 添加多条历史消息记录 """
        self._messages.add_many(*messages)

    def parse_cont_to_conts(self, in_cont) -> list[dict]:
        """ 自定义的一套内容设置格式，用来简化openai源生的配置逻辑

        :return: 注意虽然看似输入是一个cont，实际可能解析出多个conts，所以返回的是list类型
        """

        def encode_image(image_path):
            """ 读取本地图片文件 """
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        def parse_image_file(image_file):
            suffix = Path(image_file).suffix[1:].lower()
            if suffix == 'jpg':
                suffix = 'jpeg'
            base64_image = encode_image(image_file)
            image_url = {'url': f"data:image/{suffix};base64,{base64_image}"}
            return image_url

        if isinstance(in_cont, str):
            return [{"type": "text", "text": in_cont}]

        if isinstance(in_cont, dict):
            if 'image_url' in in_cont and isinstance(in_cont['image_url'], str):
                url = in_cont['image_url']
                return [{"type": "image_url", "image_url": {'url': url}}]
            if 'image_file' in in_cont:
                image_url = parse_image_file(in_cont['image_file'])
                in_cont.pop('image_file')
                if in_cont:
                    image_url.update(in_cont)
                return [{"type": "image_url", "image_url": image_url}]

            if 'image_urls' in in_cont:
                return [{"type": "image_url", "image_url": {'url': x}} for x in in_cont['image_urls']]
            if 'image_files' in in_cont:
                return [{"type": "image_url", "image_url": parse_image_file(x)} for x in in_cont['image_files']]

            return [in_cont]
        else:
            raise ValueError

    def parse_conts_to_message(self, conts, role='user'):
        """ 可以用一种简化的格式，来配置用户要提问的信息内容

        :param conts:
            str, 单条文本提问
            list,
                类似这样的数据，可以完全兼容openai的格式
                    [
                    {"type": "text", "text": "What’s in this image?"},
                    {
                      "type": "image_url",
                      "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                      },
                    },
                    ]
              也可以简化：用我这里自定义的简化格式
                'text'    文本格式直接输入
                {'image_url': 'https://...'}
                {'image_file': 'path/to/file'}
        """
        message = {"role": role}

        if isinstance(conts, str):
            message["content"] = conts
        else:
            conts = conts if isinstance(conts, (list, tuple)) else [conts]
            message["content"] = []
            for x in conts:
                message["content"] += self.parse_cont_to_conts(x)

        return message

    def add_message_from_conts(self, conts, role='user'):
        """ 添加content版本的message """
        message = self.parse_conts_to_message(conts, role=role)
        self._messages.add_many(message)

    def add_system_prompt(self, conts):
        self.add_message_from_conts(conts, 'system')

    def query(self, conts=None, response_json=None, **kwargs):
        """ 提交一个请求，并获得回复结果

        :param conts:
            None, content可以不输入，这种一般是外部已经提前配置好各种add_message，这里只要直接请求即可
            str, 常规的文本提问使用
            list, 一般是搭配图片的提问使用
        :param response_json: 是否返回json格式
            注意就算这个参数为True，提示里也要显示说明需要json格式。官方示例是在初始的role=system中配置
        """
        # 1 total_messages
        if conts:
            messages = [self.parse_conts_to_message(conts)]
            messages += (kwargs.pop('messages', None) or [])
        else:
            messages = []
        total_messages = list(self._messages + messages)
        assert total_messages

        if response_json:
            kwargs['response_format'] = {"type": "json_object"}

        # 2 请求结果
        self.recently_request_data = {
            'api_key': (api_key := self._akpool.fetch_key()),
        }
        completion = OpenAI(api_key=api_key, **self._kwargs).chat.completions.create(**{
            **self._request_kwargs,  # 全局参数
            **kwargs,  # 单次请求的参数覆盖全局参数
            "messages": total_messages,
            "stream": False,
        })
        answer: str = completion.choices[0].message.content
        self._messages.add_many(*messages, {"role": "assistant", "content": answer})

        if response_json:
            answer = json.loads(answer)

        return answer

    def query_image(self, prompt,
                    size='1024x1024', model='dall-e-3', quality='standard', n=1,
                    save_file=None,
                    **kwargs):
        """ 生成图片，注意，此功能不支持引用旧历史记录，而是一个独立的记录来请求model获得一张图

        :param save_file: 指定保存到本地某个位置
        """
        self.recently_request_data = {
            'api_key': (api_key := self._akpool.fetch_key()),
        }
        client = OpenAI(api_key=api_key, **self._kwargs)
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
            **kwargs
        )
        image_url = response.data[0].url
        self.add_message_from_conts(prompt, 'user')
        self.add_message_from_conts([{'image_url': image_url}], 'assistant')

        if save_file:
            download_file(image_url, save_file)

        return image_url


class Chat2:
    default_api_key = None
    default_base_url = 'https://beta.gpt4api.plus'

    def __init__(self, api_key=None,
                 gizmo_id=None,
                 *,
                 base_url=None,
                 model='gpt-4o-mini'):
        """
        :param gizmo_id: gpts的id，比如 https://chatgpt.com/g/g-ABCD1234-ce-shi。则id就是"g-ABCD1234"
            后面的gpts的名称变了没关系，前面前缀的id不要变就行
        """
        self.api_key = api_key or self.default_api_key
        self.base_url = base_url or self.default_base_url
        self.gizmo_id = gizmo_id  # 是否有action模块需要点击确认

        self.headers = {
            'Authorization': f'Bearer {self.api_key}'
        }
        self.session_state = {
            'stream': False,
            'model': model,
        }

        self.messages = list()  # 历史记录

    def _upload_file(self, file_path, file_type='my_files', parent_payload=None):
        """ 上传文件

        :param payload: 如果传入外部主payload自动，会自动添加已上传的文件的状态内容信息
        """
        # 1 请求
        url = f"{self.base_url}/chat/uploaded"
        payload = {'type': file_type}
        if 'conversation_id' in self.session_state:
            payload['conversation_id'] = self.session_state['conversation_id']
        files = [
            ('files', (Path(file_path).name, open(file_path, 'rb')))
        ]
        resp = requests.post(url, headers=self.headers, data=payload, files=files)
        resp_json = resp.json()

        # 2 保存状态
        self.session_state['conversation_id'] = resp_json['conversation_id']
        self.messages.append(resp_json)
        if parent_payload:
            key_name = 'attachments' if file_type == 'my_files' else 'parts'
            if key_name not in parent_payload:
                parent_payload[key_name] = []
            parent_payload[key_name].append(resp_json[key_name[:-1]])

        return resp_json

    def _upload_image(self, image_path, parent_payload=None):
        """ 上传图片 """
        return self._upload_file(image_path, 'multimodal', parent_payload=parent_payload)

    def query(self, message, *, files=None, images=None):
        # 1 预备
        if self.gizmo_id:
            url = f"{self.base_url}/chat/action"
        else:
            url = f"{self.base_url}/chat/all-tools"

        headers = self.headers.copy()
        # headers['Content-Type'] = 'application/json'

        # 2 api初步配置
        payload = self.session_state.copy()
        payload['message'] = message
        if self.gizmo_id:
            payload['gizmo_id'] = self.gizmo_id

        # 3 添加文件
        if files:
            if isinstance(files, str):
                files = [files]
            for file in files:
                self._upload_file(file, parent_payload=payload)

        # 4 添加图片
        if images:
            if isinstance(images, str):
                images = [images]
            for image in images:
                self._upload_image(image, parent_payload=payload)

        # 5 请求
        # resp = requests.post(url, headers=headers, data=json.dumps(payload))
        resp = requests.post(url, headers=headers, json=payload)  # 我觉得应该可以改这样调用，但还没测试
        resp_json = resp.json()

        # 6 保存状态
        self.session_state['conversation_id'] = resp_json['conversation_id']
        self.session_state['parent_message_id'] = resp_json['message_id']
        self.messages.append(resp_json)

        return '\n\n'.join(resp_json['contents'][-1]['message']['content']['parts'])


class DifyChat:
    default_base_url = 'http://localhost'

    def __init__(self, app_key, *,
                 conversation_id=None,
                 user=None,
                 base_url=None):
        self.app_key = app_key
        self.base_url = base_url or self.default_base_url

        self.headers = {
            'Authorization': f'Bearer {self.app_key}',
        }

        self.conversation_state = {
            'inputs': {},
            # 'response_mode': 'streaming',
            'conversation_id': conversation_id or '',
            'user': user or 'code4101',
        }

    def query(self, query, *, files=None, images=None):
        """ files、images等后期再加
        """
        # 1 请求
        payload = self.conversation_state.copy()
        payload['query'] = query
        resp = requests.post(f'{self.base_url}/v1/chat-messages',
                             headers=self.headers, json=payload)
        resp_json = resp.json()  # 如果需要可以打印这个字典看结构

        # 2 保存状态
        self.conversation_state['conversation_id'] = resp_json.get('conversation_id', '')

        return resp_json['answer']


def __2_信息摘要():
    pass


# 用来记录一些常用的提示词
xlprompts = SimpleNamespace()

# 通用的信息压缩
xlprompts.compress_content_common = """
角色: 你是一个专业的信息摘要助手，擅长快速提取和总结大量文本中的关键信息。
任务: 你的任务是对输入的"从全文中节选的部分文本内容"，提取出最重要的要点，确保摘要简洁明了，保持原意并覆盖主要内容。你需要注意以下几点：

重点提取: 确保抓住文本中的关键主题、观点和结论。
简明扼要: 摘要应尽量简短，但不能遗漏重要信息。
客观性: 保持中立，不加入任何个人意见或评论。
上下文理解: 理解并保留信息的背景和细节，以免丢失关键上下文。
适应性: 无论输入文本的类型、领域或长度如何，你都能有效生成高质量的摘要。
""".strip()

# 根据用户的问题，进行信息压缩
xlprompts.compress_content_with_query = set_template("""
角色: 你是一个智能的信息摘要助手，能够根据用户提供的查询<query>灵活调整摘要的侧重点。
任务: 你的任务是对输入的"从全文中节选的部分文本内容"中，提取出与用户查询<query>最相关的内容，并根据这些重点信息生成一个简洁的摘要。
确保摘要既包含用户感兴趣的内容，又能保留必要的背景信息。你需要注意以下几点：

查询相关性: 优先提取与用户查询相关的内容，将其放在摘要的核心部分。
简洁性: 保证摘要清晰简洁，避免冗长和不必要的细节。
全面性: 在满足查询相关性的前提下，尽量保留与主题相关的背景信息。
上下文理解: 理解用户查询的意图，并在生成摘要时结合上下文做出适当的调整。
适应性: 无论输入文本的类型、领域或长度如何，你都能有效生成侧重用户查询的高质量摘要。

<query>
{{ query }}
""")

xlprompts.compress_qq_message = """
### 目的
你是一个信息整理助手，这里任务是负责整理我的QQ聊天记录，便于后续查阅和使用，特制定以下规则。

### 整理步骤
1. **按时间线排序**：将聊天记录按时间顺序整理。
2. **聚合信息**：将内容相似或相关的聊天记录聚合，避免过于细碎。
3. **引入人员信息**：在描述中适当引入发言人员的名字，确保信息来源清晰。
4. **内容简洁**：确保每条记录简洁明了，保留关键内容。

### 具体格式
1. **时间格式**：以“YYYY-MM-DD HH:MM:SS”格式记录时间。
2. **内容聚合**：
   - **主题聚合**：将同一主题的聊天记录合并整理。
   - **人名引入**：在描述中加入发言人员名字，例如“韩锦锦提到…”，“代号4101解释…”
3. **示例**：
```markdown
### 2024-02-21
- **童浩关于空表的处理**：童浩提到已经将空表都补上，方便自动测试，@23研一 童浩 数据已修改对应实际内容，避免遇到没有表格的题目。
- **关于条件格式**：代号4101询问是否有办法做出多条件的约束，22软工-郑创鸿表示条件格式不太好做，23研一 童浩确认空表已补上。代号4101提到WPS偏爱条件格式，建议确认完成后查看结果。
- **模型数据准备**：代号4101建议使用GPT-4帮助生成正确数据，并呼吁团队注意在一些题目中替换正确数据以进行测试。

### 2024-02-22
- **多轮回答格式调整**：代号4101通知韩锦锦关于多轮回答的格式调整已完成，可以开始标注。
```

### 注意事项
- **避免冗长**：每条记录保持简洁，去除无关信息。
- **保留原意**：整理过程中，确保保留原始信息的准确性和完整性。
""".strip()


class TextSpliter:
    """ 一组文本切分算法 """

    @classmethod
    def avg(cls, content, segment_len, overlap_len):
        """ 均匀切分，不过这个具体实现还是比较基础版的

        >> TextSpliter.avg('123456789', 10, 0)
        ['123456789']
        >> TextSpliter.avg('123456789', 9, 0)
        ['12345', '6789']
        >> TextSpliter.avg('123456789', 9, 1)
        ['12345', '56789']
        >> TextSpliter.avg('123456789', 9, 2)
        ['12345', '45678', '789']
        >> TextSpliter.avg('123456789', 5, 3)
        ['12345', '34567', '56789']
        """
        content_len = len(content)
        num = content_len // segment_len + 1  # 目标切分数量
        segment_len = math.ceil(content_len / num)  # 重置后的segment_len
        return [content[i:i + segment_len] for i in range(0, content_len - overlap_len, segment_len - overlap_len)]


class CompressContent:

    @classmethod
    def basic(cls, contents, extra_prompts,
              segment_len=5000, overlap_len=200, *,
              max_workers=1,
              expect_len=None,
              max_round=-1,
              text_spliter=None,
              model='gpt-4o-mini',
              soft_merge_prompt=None,
              calc_len=None,
              logger=None,
              ):
        """ 从一段（长）文本内容中进行信息抽取

        :param str|list[str] contents: 原文内容
            list[str]，表示本身已经有些硬分割
        :param str|list[str] extra_prompts: 提取目标信息的提示词
            str，无论递归多少轮，都用同一个提示词
            list[str]，第0轮用prompts[0]，第1轮使用prompts[1]，以此类推，如果轮次超出prompts长度，则使用prompts[-1]

        :param int segment_len: 每段文本的最大长度 （原content过长时，需要拆分成多段分开处理）
            理论上最终期望摘取的信息长度应该是不超过segment_len的一半
        :param int overlap_len: 段与段之间的重叠文本长度
        :param int max_round: 最大处理轮次，-1表示无限制，直到所有内容都被处理完
            比如max_round=1，就表示只处理一个轮次
        :param int expect_len: 最终期望的内容长度，默认是segment_len的一半
        :param func calc_len: 文本长度的算法， 默认使用len
        :param int max_workers: 提取信息时，能开的最大并发数

        :param func text_spliter: 文本分割器，默认使用均切
            均匀切分，在不超过segment_len的情况下，尽量保持每段文本长度相同
            后续要支持langchain里分割算法，觉得那个切法更泛用，虽然效率估计更低
        :param str model: 使用的模型
        :param soft_merge_prompt: 最后一轮合并后，如果已经满足要求，是否要加一段提示词，再优化下描述，避免暴力合并效果过于生硬

        :param logger: 日志记录器，输入None表示不记录。 暂未实装，后续可能要用来评估token长度。
        """
        # 1 预备
        if isinstance(contents, str):
            contents = [contents]

        if isinstance(extra_prompts, str):
            extra_prompts = [extra_prompts]

        expect_len = expect_len or segment_len // 2
        logger = logger or MagicMock()
        if logger is True:
            logger = loguru_logger

        calc_len = calc_len or len

        if text_spliter is None:
            def text_spliter(content, segment_len=segment_len, overlap_len=overlap_len):
                return TextSpliter.avg(content, segment_len, overlap_len)

        # 2 循环提取信息
        round_num = 0
        while True:
            # 1 判断现在的长度，是否需要精简信息提取
            last_content_len = sum([calc_len(c) for c in contents])
            if last_content_len <= expect_len:
                break

            # 2 切分文本
            contents2 = []
            for content in contents:
                contents2.extend(text_spliter(content))

            # 3 单个提取的接口
            def extract_info(i, content, prompt):
                chat = Chat(model=model)
                chat.add_system_prompt(prompt)
                summary = chat.query(content)
                logger.debug(f'>>> 第{i}块原始内容：\n{content}\n\n>>> 提取信息：{summary}\n\n')
                return summary

            # 4 (并发)提取信息
            prompt_idx = min(round_num, len(extra_prompts) - 1)
            backend = 'threading' if max_workers != 1 else 'sequential'
            parallel = Parallel(n_jobs=max_workers, backend=backend, return_as='generator')
            logger.debug(f'>>> 提取信息轮次：{round_num}，使用提示词：\n{extra_prompts[prompt_idx]}')
            tasks = [delayed(extract_info)(i, c, extra_prompts[prompt_idx]) for i, c in enumerate(contents2, start=1)]
            contents = list(tqdm(parallel(tasks), total=len(contents2),
                                 desc=f'round {round_num} 提取信息', disable=isinstance(logger, MagicMock)))

            # 5 合并数据
            contents = ['\n\n'.join(contents)]
            round_num += 1

            if max_round > 0 and round_num >= max_round:
                break

        # 3 提取结果
        res = '\n\n'.join(contents)
        if soft_merge_prompt:
            chat = Chat(model=model)
            chat.add_system_prompt(soft_merge_prompt)
            res = chat.query(res)

        return res

    @classmethod
    def common(cls, content, extra_prompts=None, **kwargs):
        """ 通用的信息压缩 """
        return cls.basic(content,
                         extra_prompts or xlprompts.compress_content_common,
                         **kwargs)

    @classmethod
    def with_query(cls, content, query=None, extra_prompts=None, **kwargs):
        """ 通用的信息压缩，根据用户的问题，进行信息压缩
        但如果没传入query参数，也会自动改回common机制处理
        """
        if not query:
            return cls.common(content, extra_prompts, **kwargs)

        return cls.basic(content,
                         extra_prompts or xlprompts.compress_content_with_query.render(query=query),
                         **kwargs)

    @classmethod
    def qq_message(cls, content, extra_prompts=None, **kwargs):
        """ QQ聊天记录整理 """
        return cls.basic(content,
                         extra_prompts or xlprompts.compress_qq_message,
                         **kwargs)
