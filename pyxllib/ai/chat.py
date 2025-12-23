#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/08/09

import asyncio
import os
import re
import json
import math
import requests
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, List, Dict, Union, Any
import base64
from base64 import b64encode, b64decode

from loguru import logger as loguru_logger

from pyxllib.prog.lazyimport import lazy_import

try:
    import litellm
except ModuleNotFoundError:
    litellm = lazy_import('import litellm')

from pyxllib.text.jinjalib import set_template


def __1_扩展litellm支持的供应商():
    pass


# 1. 扩展自定义供应商
ROUTER_CONFIG = [
    {
        "model_name": "302ai/*",
        "litellm_params": {
            "model": "openai/*",  # 映射到 OpenAI 协议
            "api_base": "https://api.302.ai/v1",
            # 等价于os.getenv('302AI_API_KEY')，但是这样写，litellm只会在开始使用的时候才加载
            # 但在这里这样惰性加载其实还不够，所以后续用了_LAZY_ROUTER全局变量来缓存
            "api_key": "os.environ/302AI_API_KEY",
        }
    },
    {
        "model_name": "aihubmix/*",
        "litellm_params": {
            "model": "openai/*",
            "api_base": "https://api.aihubmix.com/v1",
            "api_key": "os.environ/AIHUBMIX_API_KEY",
        }
    }
]

_LAZY_ROUTER = None
_NATIVE_FUNCS = {}


def get_router():
    """惰性加载函数：只有第一次被调用时才初始化 Router"""
    global _LAZY_ROUTER
    if _LAZY_ROUTER is None:
        _LAZY_ROUTER = litellm.Router(model_list=ROUTER_CONFIG)
    return _LAZY_ROUTER


# 提取你配置的前缀，用于自动判断 (结果是 ['302ai/', 'aihubmix/'])
# 这样你以后加新厂商，只需要改 model_list，不用改下面的逻辑
ROUTER_PREFIXES = tuple(item['model_name'].replace('*', '') for item in ROUTER_CONFIG)

# 2. 定义需要支持的功能列表
# 这里列出了 LiteLLM 中常用的核心方法，你想支持哪个就加哪个
TARGET_FUNCTIONS = [
    # 对话补全
    "completion", "acompletion",
    # 向量嵌入
    "embedding", "aembedding",
    # 图片生成
    "image_generation", "aimage_generation",
    # 语音转文字 (Whisper)
    "transcription", "atranscription",
    # 文字转语音 (TTS)
    "speech", "aspeech",
    # 内容审核
    "moderation", "amoderation"
]

# 3. 动态补丁工厂 (The Better Mechanism)

# 用一个字典保存原生的 litellm 方法，防止被覆盖后找不到，也防止递归死循环
_NATIVE_FUNCS = {}


def create_proxy_function(func_name, original_func):
    """
    创建一个代理函数，根据 model 前缀决定走 Router 还是 原生逻辑
    """

    # 判断是否为异步函数 (根据命名习惯以 'a' 开头，或者检查 inspect.iscoroutinefunction)
    is_async = func_name.startswith('a') or asyncio.iscoroutinefunction(original_func)

    if is_async:
        async def async_proxy(model, **kwargs):
            # 1. 判断是否命中自定义前缀
            if model and str(model).startswith(ROUTER_PREFIXES):
                router = get_router()
                # 确保 Router 对象里也有这个方法 (比如 router.acompletion)
                if hasattr(router, func_name):
                    router_method = getattr(router, func_name)
                    return await router_method(model=model, **kwargs)

            # 2. 没命中或 Router 不支持该方法，回退到原生方法
            return await original_func(model=model, **kwargs)

        return async_proxy

    else:
        def sync_proxy(model, **kwargs):
            # 1. 判断是否命中自定义前缀
            if model and str(model).startswith(ROUTER_PREFIXES):
                router = get_router()
                if hasattr(router, func_name):
                    router_method = getattr(router, func_name)
                    return router_method(model=model, **kwargs)

            # 2. 回退到原生方法
            return original_func(model=model, **kwargs)

        return sync_proxy


# 4. 执行批量替换
def apply_patches():
    for name in TARGET_FUNCTIONS:
        # 确保 litellm 库里确实有这个方法 (防止版本差异报错)
        if not hasattr(litellm, name):
            continue

        # 获取原生方法
        original = getattr(litellm, name)

        # 存入备份字典 (只存一次，防止多次调用 apply_patches 导致嵌套)
        if name not in _NATIVE_FUNCS:
            _NATIVE_FUNCS[name] = original

        # 生成代理函数
        proxy_func = create_proxy_function(name, _NATIVE_FUNCS[name])

        # 替换 litellm 模块下的方法
        setattr(litellm, name, proxy_func)


# 立即执行补丁
apply_patches()


def __2_基于skk的chat重构后的个人chat():
    pass


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


class Multimodal_Part:
    def __init__(self, part: dict):
        self.part = part

    @classmethod
    def text(cls, _text: str):
        return cls({'type': 'text', 'text': _text})

    @classmethod
    def jpeg(cls, _bytestring: bytes):
        return cls({'type': 'image_url',
                    'image_url': {'url': f"data:image/jpeg;base64,{b64encode(_bytestring).decode('utf8')}"}})

    @classmethod
    def png(cls, _bytestring: bytes):
        return cls({'type': 'image_url',
                    'image_url': {'url': f"data:image/png;base64,{b64encode(_bytestring).decode('utf8')}"}})


def get_multimodal_content(*speeches: str | Multimodal_Part | dict):
    content = []
    for x in speeches:
        if x:
            if type(x) is str: x = Multimodal_Part.text(x)
            if type(x) is Multimodal_Part: x = x.part
            content.append(x)
    if len(content) == 1 and content[0]['type'] == 'text':
        content = content[0]['text']
    return content or ''


class ModelAnswer(str):
    """
    继承自 str 的结果对象。

    既保持了字符串的特性（可以直接 print、== 判断、写入文件），
    又携带了模型的思考过程 (reasoning) 和原始响应 (raw)。
    """

    def __new__(cls, content, reasoning=None, usage=None, raw_response=None):
        # 确保 content 是字符串，如果是 None 则转为空串
        content = content or ""
        obj = str.__new__(cls, content)
        # 兼容不同厂商的字段，有的叫 reasoning_content，有的在 content 里用 <think> 包裹
        obj.think = reasoning or ""
        obj.usage = usage  # 记录 token 消耗
        obj.raw = raw_response  # 保留原始响应对象以备查
        return obj

    def __repr__(self):
        # 交互式命令行显示时，提示这是一个特殊的 Answer 对象
        return f"ModelAnswer(len={len(self)}, think_len={len(self.think)})"


class SkkChat:
    """ 大模型通用聊天类

    类的框架原代码设计参考自skk：https://github.com/canbiaoxu/skk/tree/main/skk/openai

    后让AI整改了：
    1、底层改成litellm，从而更灵活地支持多种不同供应商
    2、为了获取ollama场景下的推理think内容，做了特殊改动
        （只修补了qwen3-vl的，deepseek-r1还是暂时没办法取出think，其实ollama官方自己ui也没取出来，但dify是能取的）
    """

    recently_request_data: dict  # 最近一次请求所用的参数

    def __init__(self,
                 model: str,  # 例如: "gpt-3.5-turbo", "deepseek/deepseek-chat", "claude-3-opus-20240229"
                 api_key: str = None,
                 base_url: str = None,
                 timeout=None,
                 max_retries=None,
                 http_client=None,
                 msg_max_count: int = None,
                 **kwargs,
                 ):
        """
        :param model: 模型名称，litellm 格式 (e.g. 'provider/model-name' or just 'model-name')
        :param api_key: 如果不传，litellm 会尝试读取环境变量 (如 OPENAI_API_KEY, ANTHROPIC_API_KEY)
        :param base_url: 自定义 API 地址
        """
        # 1. 统一管理配置 (Configuration Layer)
        # 将所有连接参数存入 self.config，方便后续合并
        self.config = kwargs.copy()
        self.config['model'] = model
        if api_key: self.config['api_key'] = api_key
        if base_url: self.config['base_url'] = base_url
        if timeout: self.config['timeout'] = timeout
        if max_retries: self.config['max_retries'] = max_retries

        # 2. 初始化消息队列
        MsgMaxCount = kwargs.pop('MsgMaxCount', None)
        msg_max_count = msg_max_count or MsgMaxCount

        # 整理 litellm 的通用参数
        self._request_kwargs = kwargs
        if timeout: self._request_kwargs["timeout"] = timeout
        if max_retries: self._request_kwargs["max_retries"] = max_retries
        if http_client: self._request_kwargs["http_client"] = http_client

        # 初始化消息队列
        self._messages = Temque(maxlen=msg_max_count)

    def reset_api_key(self, api_key: str):
        self.config['api_key'] = api_key

    def _resolve_connection_params(self, runtime_kwargs: dict) -> dict:
        """
        [核心机制] 将实例默认配置与单次请求参数合并，并执行策略调整。
        功能：
        1. 合并参数 (Runtime Override)
        2. Ollama 自动补丁 (Auto Patching): ollama/ -> openai/ + /v1
        """
        # 1. 合并参数: 实例配置 < 运行时参数
        # 注意: 这里使用 copy 避免污染实例本身的配置
        final_params = self.config.copy()
        final_params.update(runtime_kwargs)

        # 2. 提取关键字段
        model = str(final_params.get('model', ''))
        base_url = final_params.get('base_url') or final_params.get('api_base')

        # 3. 策略：如果是 ollama 协议，自动伪装成 openai 协议以保留 thinking
        if model.startswith('ollama/'):
            # 3.1 切换协议头
            final_params['model'] = model.replace('ollama/', 'openai/', 1)

            # 3.2 修正 Base URL (追加 /v1)
            if not base_url:
                base_url = "http://localhost:11434"  # Ollama 默认端口

            if "/v1" not in base_url:
                base_url = base_url.rstrip('/') + "/v1"

            final_params['base_url'] = base_url
            final_params['api_base'] = base_url  # litellm 有时认 api_base

            # 3.3 填充 Dummy Key (OpenAI 协议必须有 Key)
            if not final_params.get('api_key'):
                final_params['api_key'] = "ollama"

        # 4. 记录调试信息 (供调试查阅)
        self.recently_request_data = {
            'model': final_params.get('model'),
            'api_key': final_params.get('api_key'),
            'base_url': final_params.get('base_url')
        }

        return final_params

    def _merge_api_params(self, kwargs):
        """
        [工具] 合并 API Key 和 Base URL 到请求参数中
        """
        # 复制一份防止修改原字典
        params = kwargs.copy()
        if self.base_url:
            params['base_url'] = self.base_url
        if self.api_key:
            params['api_key'] = self.api_key
        return params

    def _extract_content_and_think(self, response_obj):
        """
        [终极版] 暴力挖掘思考过程
        解决了 LiteLLM 自动丢弃 Ollama 'thinking' 字段的问题

        todo ai这里搞的一堆修复，都无法解决litellm会丢失thinking字段的问题
        """
        # 1. 基础获取
        choice = response_obj.choices[0]
        message = choice.message
        content = message.content or ""
        reasoning = None

        # --- 第一层：标准渠道 (DeepSeek API / Azure) ---
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            reasoning = message.reasoning_content

        # --- 第二层：LiteLLM 的保留地 (provider_specific_fields) ---
        if not reasoning and hasattr(message, "provider_specific_fields"):
            # Ollama 的 extra fields 有时会在这里
            fields = message.provider_specific_fields
            if fields:
                reasoning = fields.get("thinking") or fields.get("reasoning")

        # --- 第三层：挖掘被隐藏的原始数据 (针对 Ollama/Qwen) ---
        if not reasoning:
            # LiteLLM 有时会把原始响应藏在 _hidden_params 或 model_extra 中
            # 这是一个 Pydantic 的特性，专门存"未定义字段"
            try:
                # 兼容不同版本的 Pydantic/LiteLLM
                extra = getattr(message, "model_extra", None) or getattr(message, "_extra", {})
                if extra:
                    reasoning = extra.get("thinking") or extra.get("reasoning")
            except Exception:
                pass

        # --- 第四层：文本清洗 (DeepSeek R1 / Qwen 本地版常见) ---
        # 很多时候它们其实就在正文里，只是被 <think> 包裹了
        if "<think>" in content and "</think>" in content:
            match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
            if match:
                found_think = match.group(1).strip()
                if not reasoning:
                    reasoning = found_think
                # 从正文剔除，保持纯净
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        return content, reasoning

    def _prepare_request_messages(self, text, speeches, kwargs):
        """ 准备消息列表，返回 (content, new_msgs, remaining_kwargs, full_messages) """
        content = get_multimodal_content(text, *speeches)
        new_messages = [{"role": "user", "content": content}]

        # 提取临时 messages 参数，不让它进入 litellm 的 kwargs
        temp_messages = (kwargs.pop('messages', None) or [])

        full_messages = list(self._messages + temp_messages + new_messages)
        return content, new_messages, kwargs, full_messages

    def _process_complete_response(self, response, new_msgs) -> 'ModelAnswer':
        """
        [工具] 处理非流式请求的最终结果：更新历史、封装 ModelAnswer
        """
        # msg = response.choices[0].message
        content, reasoning = self._extract_content_and_think(response)

        # 历史记录只存纯净的 Content，不存 reasoning（防止上下文超长）
        # 注意：如果 content 里本身包含 <think> 标签且没剔除，这里也会存进去。
        # 如果追求极致纯净，上面 _extract_content_and_think 需要返回清洗后的 content。
        self._messages.add_many(*new_msgs, {"role": "assistant", "content": content})

        return ModelAnswer(
            content,
            reasoning=reasoning,
            usage=response.usage,
            raw_response=response
        )

    def reset_api_key(self, api_key: str):
        self.api_key = api_key

    def _prepare_request(self, text, speeches, kwargs):
        """通用请求准备逻辑"""
        content = get_multimodal_content(text, *speeches)
        new_messages = [{"role": "user", "content": content}]
        temp_messages = (kwargs.pop('messages', None) or [])

        # 记录调试信息
        self.recently_request_data = {
            'model': self.model,
            'api_key': self.api_key,  # 仅作记录，实际 key 传递给 litellm
            'base_url': self.base_url
        }

        # 合并参数: 全局参数 < 单次请求参数
        req_params = {**self._request_kwargs, **kwargs}

        # 构建完整对话历史
        full_messages = list(self._messages + temp_messages + new_messages)

        return content, new_messages, req_params, full_messages

    def request(self, text: str | Multimodal_Part | dict, *speeches, **kwargs):
        """
        同步请求
        :param kwargs: 支持临时覆盖 model, api_key, base_url 等参数
        """
        content, new_msgs, kwargs, full_messages = self._prepare_request_messages(text, speeches, kwargs)
        final_params = self._resolve_connection_params(kwargs)

        response = litellm.completion(
            messages=full_messages,
            stream=False,
            **final_params
        )

        return self._process_complete_response(response, new_msgs)

    async def async_request(self, text: str | Multimodal_Part | dict, *speeches, **kwargs):
        """ 异步请求 """
        content, new_msgs, kwargs, full_messages = self._prepare_request_messages(text, speeches, kwargs)
        final_params = self._resolve_connection_params(kwargs)

        response = await litellm.acompletion(
            messages=full_messages,
            stream=False,
            **final_params
        )

        return self._process_complete_response(response, new_msgs)

    def stream_request(self, text: str | Multimodal_Part | dict, *speeches, **kwargs):
        """ 流式请求 """
        content, new_msgs, kwargs, full_messages = self._prepare_request_messages(text, speeches, kwargs)
        final_params = self._resolve_connection_params(kwargs)

        # 强制 stream=True
        final_params['stream'] = True

        response = litellm.completion(
            messages=full_messages,
            **final_params
        )

        full_content = []
        # 注意：流式暂不深度支持 thinking 实时分离，主要保证不崩，历史记录干净
        # 如果需要 thinking，通常建议使用非流式 request
        for chunk in response:
            if chunk.choices and (delta := chunk.choices[0].delta.content):
                full_content.append(delta)
                yield delta

        answer_text = "".join(full_content)
        # 简单清洗一下历史记录里的 think 标签（如果有），防止污染
        if "<think>" in answer_text and "</think>" in answer_text:
            answer_text = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL).strip()

        self._messages.add_many(*new_msgs, {"role": "assistant", "content": answer_text})

    async def async_stream_request(self, text: str | Multimodal_Part | dict, *speeches, **kwargs):
        """ 异步流式 """
        content, new_msgs, kwargs, full_messages = self._prepare_request_messages(text, speeches, kwargs)
        final_params = self._resolve_connection_params(kwargs)
        final_params['stream'] = True

        response = await litellm.acompletion(
            messages=full_messages,
            **final_params
        )

        full_content = []
        async for chunk in response:
            if chunk.choices and (delta := chunk.choices[0].delta.content):
                full_content.append(delta)
                yield delta

        answer_text = "".join(full_content)
        if "<think>" in answer_text and "</think>" in answer_text:
            answer_text = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL).strip()

        self._messages.add_many(*new_msgs, {"role": "assistant", "content": answer_text})

    def rollback(self, n=1):
        '''
        回滚对话
        '''
        self._messages.core[-2 * n:] = []
        # 简单打印最后剩下的两条消息用于确认
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

    def dalle(self, *speeches,
              size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = '1024x1024',
              return_format: Literal["url", "bytes"] = 'bytes',
              **kwargs):
        """ 图片生成 """
        # 参数整理
        kwargs['prompt'] = speeches[0]
        kwargs['size'] = size
        kwargs['response_format'] = 'b64_json' if return_format == 'bytes' else 'url'

        # 使用统一的解析器处理连接参数 (支持临时换 key/model)
        final_params = self._resolve_connection_params(kwargs)

        # DALL-E 接口比较特殊，不需要 messages 参数，直接传 kwargs
        response = litellm.image_generation(**final_params)

        if return_format == 'bytes':
            return [b64decode(img.b64_json) for img in response.data]
        else:
            return [img.url for img in response.data]

    async def async_dalle(self,
                          *speeches,
                          size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = '1024x1024',
                          image_count=1,
                          quality: Literal["standard", "hd"] = 'standard',
                          return_format: Literal["url", "bytes"] = 'bytes',
                          **kwargs
                          ):
        self.recently_request_data = {'model': self.model, 'api_key': self.api_key}

        kvs = {
            **self._request_kwargs,
            **kwargs,
            'prompt': speeches[0],
            'size': size,
        }

        if image_count and image_count != 1: kvs['n'] = image_count
        if quality and quality != 'standard': kvs['quality'] = quality

        resp_fmt = 'b64_json' if return_format == 'bytes' else 'url'
        kvs['response_format'] = resp_fmt
        if self.base_url: kvs['base_url'] = self.base_url
        if self.api_key: kvs['api_key'] = self.api_key

        # litellm 异步画图使用 aimage_generation
        response = await litellm.aimage_generation(
            model=self.model,
            **kvs
        )

        if return_format == 'bytes':
            return [b64decode(img.b64_json) for img in response.data]
        else:
            return [img.url for img in response.data]

    def __getattr__(self, name):
        # 兼容旧项目 API
        match name:
            case 'asy_request':
                return self.async_request
            case 'forge':
                return self.add_dialogs
            case 'pin':
                return self.pin_messages
            case 'unpin':
                return self.unpin_messages
            case 'dump':
                def _dump(fpath):
                    jt = json.dumps(self.fetch_messages(), ensure_ascii=False)
                    Path(fpath).write_text(jt, encoding="utf8")
                    return True

                return _dump
            case 'load':
                def _load(fpath):
                    jt = Path(fpath).read_text(encoding="utf8")
                    self._messages.add_many(*json.loads(jt))
                    return True

                return _load
        raise AttributeError(name)


def __3_自定义Chat():
    pass


class Chat(SkkChat):
    """
    [文档](https://github.com/canbiaoxu/skk/tree/main/skk/openai)

    获取api_key:
    * [获取链接1](https://platform.openai.com/account/api-keys)
    * [获取链接2](https://www.baidu.com/s?wd=%E8%8E%B7%E5%8F%96%20openai%20api_key)
    """

    def __init__(self,
                 model,
                 *,
                 api_key: str = None,
                 base_url: str = None,  # base_url 参数用于修改基础URL
                 timeout=None,
                 max_retries=None,
                 http_client=None,
                 msg_max_count: int = None,
                 **kwargs,
                 ):
        """
        :param kwargs: 主要是供OpenAI初始化时使用的参数
        """
        super().__init__(model, api_key, base_url, timeout, max_retries, http_client, msg_max_count, **kwargs)

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
        """
        [API 变更]
        现在的 query 本质上是 request 的包装，统一走 LiteLLM。
        :param conts: 用户输入的内容
        :param response_json: 是否强制 JSON 模式
        :param kwargs: 可以传递临时 model, api_key 等
        """
        # 1. 准备参数
        if response_json:
            kwargs['response_format'] = {"type": "json_object"}

        # 2. 如果有内容，直接复用 request (它会自动处理 _resolve_connection_params)
        if conts:
            # 如果是复杂结构，先解析
            # 注意：SkkChat.request 接收 text 或 Multimodal_Part，
            # 但这里我们利用 messages 参数绕过 prepare_request 的简单封装
            msg = self.parse_conts_to_message(conts, role='user')
            # 提取 content
            content_payload = msg['content']

            # 这里稍微特殊处理：如果 content 是 list (多模态)，我们直接传给 request 可能会被当成 text
            # 最简单的办法是利用 kwargs['messages'] 传递这次的临时消息，而不是通过 text 参数
            # 但为了保持一致性，我们手动构造一次

            # 方法 A: 借用 request 接口
            # 如果 content_payload 是字符串，直接传
            if isinstance(content_payload, str):
                return self.request(content_payload, **kwargs)
            else:
                # 如果是 list (多模态), request 接收 *speeches
                # 但 parse_cont_to_conts 返回的是 dict list，request 接收的是 Multimodal_Part 或 dict
                # 我们可以把它们解包传给 request
                # self.request(None, *content_payload, **kwargs)
                # 注意 request 内部逻辑：content = get_multimodal_content(text, *speeches)
                return self.request(None, *content_payload, **kwargs)

        else:
            # 仅使用历史记录请求 (比如 add_message 之后)
            # 传入空字符串触发 request 逻辑
            return self.request("", **kwargs)

    def query_image(self, prompt, save_file=None, **kwargs):
        """ 图片生成，底层使用 dalle 接口 """
        # 自动映射参数到 dalle 接口
        image_urls = self.dalle(prompt, return_format='url', **kwargs)

        # 记录到对话历史 (模仿旧版逻辑)
        self.add_message_from_conts(prompt, 'user')
        self.add_message_from_conts([{'image_url': url} for url in image_urls], 'assistant')

        if save_file and image_urls:
            self._download_file(image_urls[0], save_file)

        return image_urls[0] if image_urls else None

    def _download_file(self, url, fpath):
        resp = requests.get(url)
        Path(fpath).write_bytes(resp.content)


class MaqueChat:
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


def __3_信息摘要():
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


def __4_dify工作流():
    pass


class DifyChat:
    def __init__(self, app_key, *,
                 conversation_id=None,
                 user=None,
                 base_url=None):
        if not app_key.startswith('app-'):
            # 从环境变量读取服务映射key
            accounts = XlHosts.find_service(os.getenv('XL_DIFY_HOST'), 'dify')['accounts']
            app_key = accounts.get(app_key, app_key)

        self.app_key = app_key
        self.base_url = base_url or link_to_host_service(os.getenv('XL_DIFY_HOST'), 'dify')

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
        loguru_logger.info(query)
        loguru_logger.info(resp_json)

        return resp_json['answer']


def __5_其他各种提示词():
    pass


if __name__ == '__main__':
    from xlproject.code4101 import *

    chat = Chat('ollama/qwen3-vl:4b')
    # chat = Chat('ollama/deepseek-r1:7b')

    # 此时 response 通常会包含 <think>...</think> 在 content 里
    # 配合我上一条回答里的 _extract_content_and_think 正则清洗功能，完美提取！
    # print(chat.request('9.11和9.8哪个大？'))

    print(chat.query([{'image_file': r"C:\Users\kzche\Desktop\1.png"}, '这张图是什么内容']))

    print(chat.query('显卡名称是？'))

    # chat = Chat('ollama/qwen3-vl:4b')
    # res = chat.request('自然数50的下一个整数是多少')
    # print('answer1: ', res)
    # print('think: ', res.think)
    # print('answer2: ', chat.request('再之后是多少'))

    # chat = Chat('deepseek/deepseek-chat')
    # print(chat.request('你是谁'))

    # chat = Chat('aihubmix/gpt-5-mini')
    # print(chat.request('你是谁'))
