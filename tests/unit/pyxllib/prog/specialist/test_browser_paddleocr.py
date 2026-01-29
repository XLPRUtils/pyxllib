#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from paddleocr import PaddleOCR
from pyxllib.prog.specialist.browser import inspect_object

from pyxllib.xl import run_once

def test_paddleocr_rendering():
    # 1. 初始化 PaddleOCR
    # lang="ch" 表示中英文混合，device="gpu" 使用 GPU 加速
    print("正在初始化 PaddleOCR...")
    ocr = PaddleOCR(lang="ch", device="gpu")
    
    # 2. 图片路径
    img_path = r"D:\home\chenkunze\data\m2508凡修\mainwin\世界.jpg"
    
    if not os.path.exists(img_path):
        print(f"错误：找不到测试图片 {img_path}")
        return

    # 3. 进行预测
    print(f"正在识别图片：{img_path}")
    result = ocr.predict(img_path)
    
    if not result or not result[0]:
        print("未检测到文本内容。")
        return

    # 4. 获取第一个结果对象进行渲染展示
    res = result[0]
    print(f"识别完成，正在调用 browser 展示结果对象 (类型: {type(res)})...")
    
    # 使用 browser 模式展示
    inspect_object(res, mode="browser")

if __name__ == "__main__":
    test_paddleocr_rendering()
