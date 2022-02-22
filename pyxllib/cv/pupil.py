#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/10/07 09:06

import re

class RgbFormatter:
    """ Format Color """
    def __init__(self, r=0, g=0, b=0):
        """ 标准的RGB颜色表达方式，数值范围在0~255 """
        self.r, self.g, self.b = r, g, b

    def __repr__(self):
        return f'({self.r}, {self.g}, {self.b})'

    def mixtures(self, rgbs, ratios=1):
        """ 混入另一部分颜色，产生一个新的颜色

        :param rgbs: 新的颜色
        :param ratios: 原颜色权重是1，其他颜色默认权重也是1

        >>> RgbFormatter(255, 0, 0).mixtures([RgbFormatter(0, 255, 0), RgbFormatter(0, 0, 255)])
        (85, 85, 85)
        """
        # 1 非列表结构全部转列表
        if isinstance(rgbs, RgbFormatter):
            rgbs = [rgbs]
        if not isinstance(ratios, (list, tuple)):
            ratios = [ratios]

        # 2 添加当前自身颜色
        rgbs.append(self)
        ratios.append(1)
        # ratios数与rgbs数相同
        if len(ratios) < len(rgbs):
            ratios += [1] * (len(rgbs) - len(ratios))

        # 3 权重计算
        sum_ratio = sum(ratios)
        r = sum([rgb.r * ratio for rgb, ratio in zip(rgbs, ratios)]) / sum_ratio
        g = sum([rgb.g * ratio for rgb, ratio in zip(rgbs, ratios)]) / sum_ratio
        b = sum([rgb.b * ratio for rgb, ratio in zip(rgbs, ratios)]) / sum_ratio
        return RgbFormatter(round(r), round(g), round(b))

    def light(self, ratio=1):
        """ 把颜色变浅

        :param ratio: 原值权重是1，这里可以设置白色的权重
        :return:

        >>> RgbFormatter(0, 0, 0).light()
        (128, 128, 128)
        """
        return self.mixtures(RgbFormatter(255, 255, 255), ratio)

    @staticmethod
    def from_rgb_int255(r=0, g=0, b=0):
        return RgbFormatter(r, g, b)

    def to_tuple(self):
        return self.r, self.g, self.b

    @staticmethod
    def from_vba_value(v):
        """ VBA使用一个Long类型存储颜色，像素(b,g,r) = b*65536 + g*256 + r

        >>> RgbFormatter().from_vba_value(11278293)
        (213, 23, 172)

        # 有时候用宏生成代码，可能会产生负值
        >>> RgbFormatter().from_vba_value(-11260549)
        (123, 45, 84)
        >>> RgbFormatter().from_vba_value(-16776961)
        (255, 0, 0)
        """
        if v < 0: v += 256 ** 3
        rgb = [v % 256, (v // 256) % 256, (v // 65536)]
        return RgbFormatter(*rgb)

    def to_vba_value(self, negative=False):
        """ 转vba的颜色代号值

        :param negative: 默认返回正数，可以设置获得负数表达的数值

        >>> RgbFormatter(123, 45, 84).to_vba_value()
        5516667
        >>> RgbFormatter(123, 45, 84).to_vba_value(negative=True)
        -11260549
        """
        v = self.r + self.g * 256 + self.b * 65536
        if negative: v -= 256 ** 3
        return v

    @property
    def vba_value(self):
        return self.to_vba_value()

    @staticmethod
    def from_hex(s):
        """
        >>> RgbFormatter.from_hex('#7B2D54')
        (123, 45, 84)
        >>> RgbFormatter.from_hex('7B2D54')
        (123, 45, 84)
        """
        m = re.search(r'(\w\w)(\w\w)(\w\w)', s)
        if not m: raise ValueError(f'16进制颜色格式有误: {s}')
        return RgbFormatter(*[int(x, 16) for x in m.groups()])

    def to_hex(self, lower=False):
        """
        :param lower: 默认返回大写，可以设置返回小写

        >>> RgbFormatter(123, 45, 84).to_hex()
        '#7B2D54'
        """
        pattern = '#{:02X}{:02X}{:02X}'
        if lower: pattern = pattern.replace('X', 'x')
        return pattern.format(self.r, self.g, self.b)

    @property
    def hex(self):
        return self.to_hex()

    @staticmethod
    def from_percentage(r, g, b):
        """ 主要用于svg等场景会用的百分率的格式
        输入的每个值是0~1内的百分率浮点数
        """
        return RgbFormatter(*[round(255 * x) for x in (r, g, b)])

    def to_percentage(self):
        """
        >>> RgbFormatter.from_percentage(0.5, 0.7, 0.9).to_percentage()
        [0.5019607843137255, 0.6980392156862745, 0.9019607843137255]
        """
        return [x / 255 for x in (self.r, self.g, self.b)]

    @property
    def percentage(self):
        return self.to_percentage()
