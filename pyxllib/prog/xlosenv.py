import os
import json
import base64

from pyxllib.text.newbie import add_quote


class XlOsEnv:
    """ pyxllib库自带的一套环境变量数据解析类

    会将json的字符串值，或者普通str，存储到环境变量中

    环境变量也可以用来实现全局变量的信息传递，虽然不太建议这样做

    >> XlOsEnv.persist_set('TP10_ACCOUNT',
                           {'server': '172.16.250.250', 'port': 22, 'user': 'ckz', 'passwd': '123456'},
                           True)
    >> print(XlOsEnv.get('TP10_ACCOUNT'), True)  # 展示存储的账号信息
    eyJzZXJ2ZXIiOiAiMTcyLjE2LjE3MC4xMzQiLCAicG9ydCI6IDIyLCAidXNlciI6ICJjaGVua3VuemUiLCAicGFzc3dkIjogImNvZGV4bHByIn0=
    >> XlOsEnv.unset('TP10_ACCOUNT')
    """

    @classmethod
    def get(cls, name, *, decoding=False):
        """ 获取环境变量值

        :param name: 环境变量名
        :param decoding: 是否需要先进行base64解码
        :return:
            返回json解析后的数据
            或者普通的字符串值
        """
        value = os.getenv(name, None)
        if value is None:
            return value

        if decoding:
            value = base64.b64decode(value.encode())

        try:
            return json.loads(value)
        except json.decoder.JSONDecodeError:
            return value.decode()

    @classmethod
    def set(cls, name, value, encoding=False):
        """ 临时改变环境变量

        :param name: 环境变量名
        :param value: 要存储的值
        :param encoding: 是否将内容转成base64后，再存储环境变量
            防止一些密码信息，明文写出来太容易泄露
            不过这个策略也很容易被破解；只防君子，难防小人

            当然，谁看到这有闲情功夫的话，可以考虑做一套更复杂的加密系统
            并且encoding支持多种不同的解加密策略，这样单看环境变量值就很难破译了
        :return: str, 最终存储的字符串内容
        """
        # 1 打包
        if isinstance(value, str):
            value = add_quote(value)
        else:
            value = json.dumps(value)

        # 2 编码
        if encoding:
            value = base64.b64encode(value.encode()).decode()

        # 3 存储到环境变量
        os.environ[name] = value

        return value

    @classmethod
    def persist_set(cls, name, value, encoding=False, *, cfgfile=None):
        """ python里默认是改不了系统变量的，需要使用一些特殊手段
        https://stackoverflow.com/questions/17657686/is-it-possible-to-set-an-environment-variable-from-python-permanently/17657905

        :param cfgfile: 在linux系统时，可以使用该参数
            默认是把环境变量写入 ~/.bashrc，可以考虑写到
            TODO 有这个设想，但很不好实现，不是很关键的功能，所以还未开发

        """
        # 写入环境变量这里是有点小麻烦的，要考虑unix和windows不同平台，以及怎么持久化存储的问题，这里直接调用一个三方库来解决
        from envariable import setenv

        value = cls.set(name, value, encoding)
        if value[0] == value[-1] == '"':
            value = '\\' + value + '\\'
        setenv(name, value)

        return value

    @classmethod
    def unset(cls, name):
        """ 删除环境变量 """
        from envariable import unsetenv
        unsetenv(name)
