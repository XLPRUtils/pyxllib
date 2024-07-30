import os
import json
from json import dumps as jsonDumps
from json import loads as jsonLoads
from pathlib import Path
from typing import List, Literal
import base64

from openai import OpenAI, AsyncOpenAI

from pyxllib.file.specialist import download_file


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
    default_base_url = "https://console.chatdata.online/v1"

    def __init__(self,
                 # kwargs
                 api_key: str | AKPool = None,
                 base_url: str = None,  # base_url 参数用于修改基础URL
                 timeout=None,
                 max_retries=None,
                 http_client=None,
                 # request_kwargs
                 model: Literal["gpt-4-1106-preview", "gpt-4-vision-preview",
                 "gpt-4", "gpt-4-0314", "gpt-4-0613",
                 "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-0613", "gpt-3.5-turbo",
                 "gpt-4o", "gpt-4-turbo"] = "gpt-3.5-turbo",
                 # Chat
                 msg_max_count: int = None,
                 # kwargs
                 **kwargs,
                 ):
        """

        :param kwargs: 主要是供OpenAI初始化时使用的参数

        """
        api_key = api_key or self.default_api_key or os.environ.get("OPENAI_API_KEY")

        api_base = kwargs.pop('api_base', None)
        base_url = base_url or api_base or self.default_base_url or os.environ.get("OPENAI_BASE_URL")
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

    def reset_api_key(self, api_key: str | AKPool):
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

    def add_dialogs(self, *ms: dict | system_msg | user_msg | assistant_msg):
        '''
        添加历史对话
        '''
        messages = [dict(x) for x in ms]
        self._messages.add_many(*messages)

    def __getattr__(self, name):
        match name:  # 兼容旧代码
            case 'asy_request':
                return self.async_request
            case 'forge':
                return self.add_dialogs
            case 'pin':
                return self.pin_messages
            case 'unpin':
                return self.unpin_messages
            case 'dump':
                return self._dump
            case 'load':
                return self._load
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
