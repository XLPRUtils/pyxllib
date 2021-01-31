#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/01/24

"""
坚果云 同步配置：关闭某些类型的文件同步
"""

from pyxllib.basic import *

s = r"""# 坚果云自定义忽略同步规则
# 
# 警告:
# 该文件用来自定义坚果云的忽略同步规则, 文件编码必须为 UTF-8.
# 由于用户修改了自定义忽略规则导致坚果云无法正常工作的问题由用户自己负责.
#
# 说明:
# 每行一条规则. 以 # 开头的行将被忽略.
# 每条规则必须以英文半角句号开头, 否则会忽略
# 无效的规则会被忽略, 并在坚果云日志文件中提示.
#
# 注意事项:
# 忽略规则仅影响文件, 不影响文件夹
# 忽略规则会且仅会在 *上传* 时生效
# 每次修改需要重启客户端才能生效.
# 该规则文件不会自动同步到其他设备.
#
# 例子:
# 忽略所有扩展名为 *.bak 的文件, 规则为 .bak

# disabled/blacklist/whitelist 三选一
# disabled 禁用该功能
# blacklist 黑名单模式, 列出的文件类型都不会被上传
# whitelist 白名单模式, 只上传列出的文件类型
mode=blacklist

# 规则开始
.bak
.aux
.log
.out
.gz
.pyc
.gz(busy)
.bbl
.db
.synctex
.synctex(busy)
.dvi
.toc
.iml
"""


# 180706：增加过滤dvi文件
# TODO 181031：增加对linux环境的支持？

# synctex是winedt才有的，texstudio没有
# C:\Users\chen\AppData\Roaming\Nutstore


def main():
    root = Dir('Nutstore', os.getenv('APPDATA'))

    for d in root.select('*').subdirs():
        if d.name == 'db' or d.name.startswith('db_'):
            # 遍历所有db开头的目录，将下面的customExtRules.cfg都替换了
            f = File('customExtRules.cfg', d)
            print(f)
            f.write(s, encoding='utf8', if_exists='backup')


if __name__ == '__main__':
    main()
