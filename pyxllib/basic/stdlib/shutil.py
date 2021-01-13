#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/01/13 15:25

import os
import shutil


def unpack_zipfile(filename, extract_dir):
    """ 为了修复zipfile底层的中文解压乱码问题，修改了shutil._UNPACK_FORMATS的底层功能
    """
    from pyxllib.basic.stdlib import zipfile
    from pyxllib.basic import Dir, File
    from shutil import ReadError

    zip = zipfile.ZipFile(filename)
    if not zipfile.is_zipfile(filename):
        raise ReadError("%s is not a zip file" % filename)
    try:
        for info in zip.infolist():
            name = info.filename

            # don't extract absolute paths or ones with .. in them
            if name.startswith('/') or '..' in name:
                continue

            target = os.path.join(extract_dir, *name.split('/'))
            if not target:
                continue

            Dir(File(target).parent).ensure_dir()
            if not name.endswith('/'):
                # file
                data = zip.read(info.filename)
                f = open(target, 'wb')
                try:
                    f.write(data)
                finally:
                    f.close()
                    del data
    finally:
        zip.close()


# 解决unzip中文乱码问题
shutil._UNPACK_FORMATS['zip'] = (['.zip'], unpack_zipfile, [], "ZIP file")
