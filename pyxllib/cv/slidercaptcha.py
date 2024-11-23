import os
import cv2
import numpy as np
from xlproject.code4101 import XlPath

from pyxllib.file.specialist import download_file
from pyxllib.prog.pupil import is_url


class SliderCaptchaLocator:
    """ 滑动验证码的位置定位

    由于缺口一般是一个黑底图，整体颜色是偏暗的，可以利用这个特性来定位列所在位置
    """

    def __init__(self, background_image_path, slider_image_path=None):
        """
        :param background_image_path: 背景主图片
        :param slider_image_path: 滑动条图片
        """
        # 支持输入是url的情况
        if is_url(background_image_path):
            background_image_path = download_file(background_image_path, temp=True, ext='.png')
        if is_url(slider_image_path):
            slider_image_path = download_file(slider_image_path, temp=True, ext='.png')
        # 初始化图片等数据
        self.image_path = background_image_path
        self.image = cv2.imread(self.image_path)
        self.darkest_rows = None
        self.column_averages = None
        self.enhanced_averages = None
        self.radius = 24
        if slider_image_path is not None:
            # 先读取图片
            template_image = cv2.imread(slider_image_path)
            # 取图片宽度的40%作为radius
            self.radius = int(template_image.shape[1] * 0.4)

    def select_darkest_part_rows(self, fraction=0.1):
        """
        选择最暗的部分行，供下游计算列的平均暗度

        :param float fraction: 最暗行的比例
        :return: 最暗行的图像部分
        """
        if self.darkest_rows is None:
            average_row_values = np.mean(self.image, axis=(1, 2))
            threshold_index = int(len(average_row_values) * fraction)
            darkest_indices = np.argsort(average_row_values)[:threshold_index]
            self.darkest_rows = self.image[darkest_indices]
        return self.darkest_rows

    def calculate_column_averages(self):
        """
        计算列的平均强度。

        :return: 列的平均强度
        """
        if self.column_averages is None:
            rows = self.select_darkest_part_rows()
            column_values = np.mean(rows, axis=0)
            column_intensity = np.mean(column_values, axis=1)
            min_val, max_val = np.min(column_intensity), np.max(column_intensity)
            self.column_averages = ((column_intensity - min_val) / (max_val - min_val) * 255).astype(
                np.uint8) if max_val != min_val else np.zeros_like(column_intensity)
        return self.column_averages

    def enhance_column_averages(self):
        """
        增强列平均值，利用验证码的黑度图的对称性，进行一个增强，使得其他非验证码的零散黑条变白

        :param int radius: 对比半径，这个建议设成模版图宽度的40%~50%
        :return: 增强后的列平均值
        """
        if self.enhanced_averages is None:
            averages = self.calculate_column_averages()
            length = len(averages)
            enhanced = np.zeros_like(averages)
            for i in range(length):
                max_value = averages[i]
                for j in range(1, self.radius // 2 + 1):
                    if i - j >= 0 and i + j < length:
                        max_value = max(max_value, abs(int(averages[i - j]) - int(averages[i + j])))
                enhanced[i] = max_value
            self.enhanced_averages = enhanced
        return self.enhanced_averages

    def find_captcha_position(self, threshold=50):
        """
        查找验证码的位置。

        :param int threshold: 达到多少像素后，开始计算中位数
        :return: 验证码的列中心位置
        """
        intensities = self.enhance_column_averages()
        indices = []
        for intensity in range(256):
            new_indices = np.where(intensities == intensity)[0]
            if len(indices) + len(new_indices) > threshold:
                indices.extend(new_indices[:threshold - len(indices)])
                break
            indices.extend(new_indices)
        return int(np.median(indices))

    def draw_red_line(self, position, line_thickness=3):
        """
        在图像上绘制红线。

        :param int position: 红线位置
        :param int line_thickness: 线宽
        :return: 绘制红线后的图像
        """
        gray_image = np.tile(self.calculate_column_averages(), (100, 1)).astype(np.uint8)
        color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        cv2.line(color_image, (position, 0), (position, gray_image.shape[0]), (0, 0, 255), line_thickness)
        return color_image

    def debug(self):
        position = self.find_captcha_position()
        final_image = self.draw_red_line(position)
        cv2.imshow('Image with Red Line', final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    os.chdir(XlPath.desktop())
    locator = SliderCaptchaLocator('cap_union_new_getcapbysig.png')
    position = locator.find_captcha_position()
    final_image = locator.draw_red_line(position)
    cv2.imshow('Image with Red Line', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
