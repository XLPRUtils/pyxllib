#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/04 09:14


from pyxllib.xlcv import *


def test_labeldata():
    # os.chdir(r'D:\home\datasets\RealEstate2020\temp')
    from pyxllib.data.label import LabelmeData

    ld = LabelmeData(r'D:\home\datasets\RealEstate2020\temp')
    # ld.to_labelattrs()

    return ld


@deprecated(reason='test', action='once')
def test_deprecated():
    """ 什么鬼，action="once"怎么也会重复这么多次！ """
    pass


@deprecated(reason='test', action='once')
def test_deprecated2():
    """ 什么鬼，action="once"怎么也会重复这么多次！ """
    pass


def main():
    for i in range(2):
        test_deprecated()
        test_deprecated2()

    test_deprecated()
    test_deprecated2()


if __name__ == '__main__':
    with TicToc(__name__):
        # ld = test_labeldata()
        # cocogt = ld.to_cocogt()
        # warnings.simplefilter(action, category)

        main()

        pass
