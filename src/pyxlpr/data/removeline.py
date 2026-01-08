#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/09/15 21:00


from pyxllib.xlcv import *


class RemoveLine:
    def __init__(self, filename):
        self.bgr_img = xlcv.read(filename)
        self.gray_img = cv2.cvtColor(self.bgr_img, cv2.COLOR_BGR2GRAY)
        self.binary_img = cv2.adaptiveThreshold(self.gray_img, 255, 0, 1, 11, 3)

    def debug(self):
        """ 同run，只是debug模式，会输出中间结果
        """
        bgr_img = self.bgr_img
        xlcv.show(bgr_img, '0 src')
        dprint(bgr_img.shape)

        lines = self.detect_lines()
        xlcv.show(xlcv.lines(bgr_img, lines, [0, 0, 255]), '1 hough')
        lines = self.refine_lines(lines)
        xlcv.show(xlcv.lines(bgr_img, lines, [0, 0, 255]), '2 expand')
        dprint(lines.shape)

        dst = self.remove_lines(lines)
        xlcv.show(dst, 'result')

    def run(self):
        """
        :return: 获得去掉线后的图片
        """
        lines = self.detect_lines()
        if lines.any():
            lines = self.refine_lines2(lines)
            return self.remove_lines(lines)
        else:
            return self.bgr_img

    def detect_lines(self, rho=1, theta=np.pi / 180, threshold=80, min_line_length=50, max_line_gap=30):
        lines = cv2.HoughLinesP(self.binary_img, rho, theta, threshold, min_line_length, max_line_gap)
        # 不知道为什么返回值第2维会多一个1，把它删了~
        lines = np.array([]) if lines is None else lines.reshape(-1, 4)
        return lines

    def refine_lines(self, lines):
        im = self.binary_img
        n, m = im.shape

        def f(v):
            """ 辅助函数：四舍五入取整 """
            return int(round(v))

        def expand(x, y, dx, dy):
            """ 从(x,y)开始，按dx,dy逐步遍历，直到遇到im二值图为0的位置 """
            while True:
                j, i = f(x + dx), f(y + dy)
                if 0 <= j < m and 0 <= i < n and im[i][j]:
                    x, y = x + dx, y + dy
                else:
                    return f(x), f(y)

        new_lines = []
        for line in lines:
            # 1 确保x1是左边，x2是右边
            x1, y1, x2, y2 = line
            if x2 < x1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            # 2 计算斜率，分两类处理
            if abs(x1 - x2) > abs(y1 - y2):
                dx, dy = 1, (y1 - y2) / (x1 - x2)
            else:
                dx, dy = (x1 - x2) / (y1 - y2), 1
            # 3 向左右各自延展
            x1, y1 = expand(x1, y1, -dx, -dy)
            x2, y2 = expand(x1, y1, dx, dy)
            new_lines.append((x1, y1, x2, y2))
        return np.array(new_lines)

    def refine_lines2(self, lines):
        """ 标准的延展会有些问题，会误删很多斜的笔划

        这个版本的删除，会只删除比较水平的情况，宁愿少删，但不误删
        """
        im = self.binary_img
        n, m = im.shape

        def f(v):
            """ 辅助函数：四舍五入取整 """
            return int(round(v))

        def expand(x, y, dx, dy):
            """ 从(x,y)开始，按dx,dy逐步遍历，直到遇到im二值图为0的位置 """
            while True:
                j, i = f(x + dx), f(y + dy)
                if 0 <= j < m and 0 <= i < n and im[i][j]:
                    x, y = x + dx, y + dy
                else:
                    return f(x), f(y)

        new_lines = []
        for line in lines:
            # 1 确保x1是左边，x2是右边
            x1, y1, x2, y2 = line
            if x2 < x1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            # 2 计算斜率，分两类处理
            dx, dy = 1, (y1 - y2) / (x1 - x2 + sys.float_info.epsilon)
            if abs(dy) > math.tan(math.pi / 180 * 10):
                # 倾斜度必须在10度以内（偏水平的线）
                continue
            # 3 向左右各自延展
            x1, y1 = expand(x1, y1, -dx, -dy)
            x2, y2 = expand(x1, y1, dx, dy)
            new_lines.append((x1, y1, x2, y2))
        return np.array(new_lines)

    def remove_lines(self, lines, box_size=(7, 3), num=5):
        """
        :param lines:
        :param box_size: 所有尺寸，默认是7*3的矩阵
            7*3矩阵中，如果有超出4个前景点，则恢复该背景点为原始像素值
        :param num: 矩阵至少需要的点数量
        :return:
        """
        # 1 计算填充用的默认背景色
        src = self.bgr_img
        if not lines.any(): return src
        bg_color = xlcv.bg_color(src, binary_img=self.binary_img)
        # dprint(bg_color)

        # 2 删笔画的mask矩阵
        mask = xlcv.lines(np.zeros(src.shape[:2]), lines, [255], 2)
        tmp = xlcv.lines(src, lines, bg_color, 2)  # 在原图画上背景颜色的线，产生抹除效果
        dst = np.array(tmp)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

        # 3 恢复断开的笔划
        top, left = box_size[0] // 2, box_size[1] // 2
        bottom, right = box_size[0] - top, box_size[1] - left
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                if not mask[i, j]: continue
                arr = tmp[max(i - top, 0):i + bottom, max(j - left, 0):j + right].reshape(-1)
                if sum(arr < 128) >= num:
                    dst[i, j] = src[i, j]
        # xlcv.show(tmp)
        return dst


def test_removeline(path):
    """ 批量处理，检查去线功能效果 """

    def func(p1, p2):
        dst = RemoveLine(str(p1)).run()
        cv2.imwrite(str(p2), dst)

    d1, d2 = Dir(path), Dir(path + '+')
    d2.ensure_dir()
    d1_state = d1.select('*.png')
    d1_state.subs = d1_state.subs
    d1_state.procpaths(func, ref_dir=d2, pinterval=1000)


if __name__ == '__main__':
    TicToc.process_time(f'{dformat()}启动准备共用时')
    tictoc = TicToc(__file__)
    os.chdir(r'D:\RealEstate2020')

    # RemoveLine(r'5_label+\agreement_label1_text2\handwriting\C100001245519-003.png').debug()
    test_removeline(r'5_label+\agreement_label1_text2\handwriting')
    test_removeline(r'5_label+\agreement_label1_text2\printed')  # 65680张图，704秒

    # cv2.waitKey(0)
    tictoc.toc()
