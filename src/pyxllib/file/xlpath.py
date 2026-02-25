#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/02/24

import json
import pathlib
import tempfile

from pyxllib.prog.lazyimport import lazy_import

try:
    import ujson
except ModuleNotFoundError:
    ujson = lazy_import('ujson')

try:
    import charset_normalizer
except ModuleNotFoundError:
    charset_normalizer = lazy_import('charset_normalizer')


def get_encoding(content, default='utf-8'):
    """ 检测内容编码
    :param content: bytes 或 Path 或 str(文件路径)
    :param default: 检测失败时的默认编码
    """
    if isinstance(content, (str, pathlib.Path)):
        try:
            content = pathlib.Path(content).read_bytes()
        except Exception:
            return default
            
    if not isinstance(content, bytes):
        return default
        
    try:
        result = charset_normalizer.from_bytes(content).best()
        if result:
            return result.encoding
    except Exception:
        pass
    return default


class XlPath(type(pathlib.Path())):
    """ 继承自pathlib.Path的扩展路径类，提供更多便利的文件操作方法 """
    
    @classmethod
    def safe_init(cls, path):
        """ 安全初始化，如果路径无效返回None """
        try:
            return cls(path)
        except Exception:
            return None

    @classmethod
    def tempdir(cls):
        return cls(tempfile.gettempdir())

    def read_text(self, encoding=None, errors='strict', return_mode=False):
        """ 读取文本文件，支持自动检测编码 """
        if not encoding:
            try:
                content = pathlib.Path(self).read_bytes()
                result = charset_normalizer.from_bytes(content).best()
                if result:
                    encoding = result.encoding
                    s = result.output().decode(encoding)
                else:
                    encoding = 'utf-8'
                    s = pathlib.Path(self).read_text(encoding=encoding, errors=errors)
            except Exception:
                encoding = 'utf-8'
                s = pathlib.Path(self).read_text(encoding=encoding, errors=errors)
        else:
            s = pathlib.Path(self).read_text(encoding=encoding, errors=errors)
        
        if '\r' in s:
             s = s.replace('\r\n', '\n')
             
        if return_mode:
            return s, encoding
        return s

    def write_text(self, data, encoding='utf-8', **kwargs):
        """ 写入文本文件，自动创建父目录 """
        if not self.parent.exists():
            self.parent.mkdir(parents=True, exist_ok=True)
        return pathlib.Path(self).write_text(str(data), encoding=encoding, **kwargs)

    def read_json(self, encoding=None, **kwargs):
        """ 读取JSON文件 """
        s = self.read_text(encoding=encoding)
        try:
            return ujson.loads(s)
        except ValueError:
            return json.loads(s)

    def write_json(self, data, encoding='utf-8', ensure_ascii=False, indent=None, **kwargs):
        """ 写入JSON文件 """
        if not self.parent.exists():
            self.parent.mkdir(parents=True, exist_ok=True)
        with self.open('w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent, **kwargs)

    def read_jsonl(self, encoding='utf-8'):
        """ 读取JSONL文件 """
        data = []
        if self.is_file():
            with self.open('r', encoding=encoding) as f:
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        return data

    def write_jsonl(self, data, encoding='utf-8', ensure_ascii=False, default=None):
        """ 写入JSONL文件 """
        if not self.parent.exists():
             self.parent.mkdir(parents=True, exist_ok=True)
        with self.open('w', encoding=encoding) as f:
            for x in data:
                f.write(json.dumps(x, ensure_ascii=ensure_ascii, default=default) + '\n')

    def glob_files(self, pattern='*'):
        """ 获取所有文件 """
        for p in self.glob(pattern):
            if p.is_file():
                yield p

    def get_total_lines(self, encoding='utf-8', skip_blank=False):
        """ 统计行数 """
        count = 0
        with self.open('r', encoding=encoding) as f:
            for line in f:
                if skip_blank and not line.strip():
                    continue
                count += 1
        return count

    def yield_line(self, batch_size=None, encoding='utf-8'):
        """ 生成行 """
        from itertools import islice
        with self.open('r', encoding=encoding) as f:
            if batch_size is None:
                yield from (line.rstrip('\n') for line in f)
            else:
                while True:
                    batch = list(islice(f, batch_size))
                    if not batch:
                        break
                    yield [line.rstrip('\n') for line in batch]

    def split_to_dir(self, lines_per_file, dst_dir=None, encoding='utf-8'):
        """ 拆分文件 """
        if dst_dir is None:
            dst_dir = self.parent / self.stem
        else:
            dst_dir = XlPath(dst_dir)
        
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        outfile = None
        file_idx = 0
        line_idx = 0
        suffix = self.suffix
        
        with self.open('r', encoding=encoding) as f:
            for line in f:
                if line_idx % lines_per_file == 0:
                    if outfile: outfile.close()
                    outfile_path = dst_dir / f"{self.stem}_{file_idx:04d}{suffix}"
                    outfile = outfile_path.open('w', encoding='utf-8')
                    file_idx += 1
                outfile.write(line)
                line_idx += 1
        if outfile: outfile.close()


def cache_file(file, generator, encoding='utf-8'):
    """ 简单的文件缓存机制 """
    f = XlPath(file)
    if f.exists():
        return f.read_text(encoding=encoding)
    else:
        content = generator()
        f.write_text(str(content), encoding=encoding)
        return content


class GetEtag:
    """ 七牛原有etag功能基础上做封装 """

    @classmethod
    def from_bytes(cls, _bytes):
        import io
        import qiniu
        return qiniu.utils.etag_stream(io.BytesIO(_bytes))

    @classmethod
    def from_text(cls, text):
        import io
        import qiniu
        _bytes = text.encode('utf8')
        return qiniu.utils.etag_stream(io.BytesIO(_bytes))

    @classmethod
    def from_file(cls, file):
        import qiniu
        return qiniu.etag(file)

    @classmethod
    def from_url(cls, url):
        import requests
        return cls(requests.get(url).content)


def get_etag(arg):
    """ 七牛原有etag功能基础上做封装

    :param arg: 支持bytes二进制、文件、url地址

    只跟文件内容有关，跟文件创建、修改日期没关系
    如果读取文件后再处理etag，要尤其小心 '\r\n' 的问题！
    文件里如果是\r\n，我的File.read会变成\n，所以按文件取etag和read的内容算etag会不一样。
    """
    try:
        import qiniu
    except ImportError:
        raise ImportError("get_etag depends on 'qiniu' library. Please install it via 'pip install qiniu'.")

    import io
    import requests
    from pyxllib.prog.pupil import is_url, is_file

    if isinstance(arg, bytes):  # 二进制数据
        return qiniu.utils.etag_stream(io.BytesIO(arg))
    elif is_file(arg):  # 输入是一个文件
        return qiniu.etag(arg)
    elif is_url(arg):  # 输入是一个网页上的数据源
        return get_etag(requests.get(arg).content)
    elif isinstance(arg, str):  # 明文字符串转二进制
        return get_etag(arg.encode('utf8'))
    else:
        raise TypeError('不识别的数据类型')


class StreamJsonlWriter:
    """ 流式存储，主要用于存储文本化、jsonl格式数据 """

    def __init__(self, file_path, batch_size=2000, *,
                 delete_origin_file=False, json_default=str):
        self.file_path = XlPath(file_path)
        self.cache_text_lines = []
        self.batch_size = batch_size
        self.total_lines = 0

        self.delete_origin_file = delete_origin_file
        self.json_default = json_default

    def append_line(self, line):
        self.append_lines([line])

    def append_lines(self, data):
        """
        :param list data: 添加一组数据
        """
        import json
        for x in data:
            if isinstance(x, str):
                self.cache_text_lines.append(x)
            else:
                self.cache_text_lines.append(json.dumps(x, ensure_ascii=False,
                                                        default=self.json_default))
        if len(self.cache_text_lines) >= self.batch_size:
            self.flush()

    def flush(self):
        """ 刷新，将当前缓存写入文件 """
        if self.cache_text_lines:
            if self.total_lines == 0 and self.delete_origin_file:  # 第一次写入时，删除旧缓存文件
                self.file_path.delete()

            self.total_lines += len(self.cache_text_lines)
            self.file_path.parent.mkdir(exist_ok=True, parents=True)
            with open(self.file_path, 'a', encoding='utf8') as f:
                f.write('\n'.join(self.cache_text_lines) + '\n')
            self.cache_text_lines = []


# 兼容旧代码的别名
File = XlPath
Dir = XlPath
