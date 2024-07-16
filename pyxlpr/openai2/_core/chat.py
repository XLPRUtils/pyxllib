from json import dumps as jsonDumps
from json import loads as jsonLoads
from pathlib import Path
from typing import List, Literal
from openai import OpenAI, AsyncOpenAI


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

    def __init__(self,
                 # kwargs
                 api_key: str|AKPool,
                 base_url: str = None,  # base_url 参数用于修改基础URL
                 timeout=None,
                 max_retries=None,
                 http_client=None,
                 # request_kwargs
                 model: Literal["gpt-4-1106-preview", "gpt-4-vision-preview", "gpt-4", "gpt-4-0314", "gpt-4-0613",
                                "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-0613", "gpt-3.5-turbo"] = "gpt-3.5-turbo",
                 # Chat
                 msg_max_count: int=None,
                 # kwargs
                 **kwargs,
                 ):
        api_base = kwargs.pop('api_base', None)
        base_url = base_url or api_base
        MsgMaxCount = kwargs.pop('MsgMaxCount', None)
        msg_max_count = msg_max_count or MsgMaxCount

        if base_url: kwargs["base_url"] = base_url
        if timeout: kwargs["timeout"] = timeout
        if max_retries: kwargs["max_retries"] = max_retries
        if http_client: kwargs["http_client"] = http_client

        self.reset_api_key(api_key)
        self._kwargs = kwargs
        self._request_kwargs = {'model':model}
        self._messages = Temque(maxlen=msg_max_count)

    def reset_api_key(self, api_key: str|AKPool):
        if isinstance(api_key, AKPool):
            self._akpool = api_key
        else:
            self._akpool = AKPool([api_key])

    def request(self, text: str=None, **kwargs):
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

    def stream_request(self, text: str=None, **kwargs):
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

    async def async_request(self, text: str=None, **kwargs):
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

    async def async_stream_request(self, text: str=None, **kwargs):
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
        self._messages.core[-2 * n :] = []
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

    def add_dialogs(self, *ms: dict|system_msg|user_msg|assistant_msg):
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