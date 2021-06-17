#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/04 09:14


from pyxllib.xlai import *


def test_labeldata():
    os.chdir(r'D:\home\datasets\RealEstate2020\temp')

    # 1 初始化
    ld = LabelmeData(r'D:\home\datasets\RealEstate2020\temp')
    # 升级字典
    # ld.to_labelattrs()

    # 2 转coco
    gt_dict = ld.to_coco_gt_dict()
    # pprint.pprint(gt_dict)

    # 3 转表格
    # ld.to_excel('test.xlsx')

    # 4 coco对象
    cgd = CocoGtData(gt_dict)

    # 5 coco对象再转回labelme，礼尚往来
    ld2 = cgd.to_labelme(r'D:\home\datasets\RealEstate2020\temp2')
    ld2.writes()

    # 6 ld2再转回coco，梅开二度
    gt_dict = ld2.to_coco_gt_dict()
    pprint.pprint(gt_dict)


if __name__ == '__main__':
    with TicToc(__name__):
        # test_labeldata()

        pass
