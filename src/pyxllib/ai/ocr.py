#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/08/14 22:00

"""
ocr.py - PaddleOCR 封装工具

基于 PaddleOCR 封装的通用 OCR 识别工具。
主要优化了模型加载（单例/配置化管理）、输入预处理（统一格式）、输出标准化（LabelMe 格式转换）等流程。

特别注意：
PaddleOCR v3.x 相比旧的v2.x接口已变更：
1. 原 `ocr.ocr(img)` 接口不要使用，改用 `ocr.predict(img)`。
2. 返回结果结构变更，`predict` 返回的是 `OCRResult` 对象列表，包含更丰富的字段。

每一张图返回结果的字段说明 (fields_explanation)：
    "input_path": 输入图片的绝对路径。
    "page_index": 对于多页文档（如PDF），表示当前页码；普通图片通常为 null。
    "model_settings": 模型运行时的配置参数。
    "dt_polys": 检测到的文本行多边形坐标列表 (4点, 顺时针)。
    "text_det_params": 文本检测超参数。
    "text_type": 文本类型，通常为 'general'。
    "textline_orientation_angles": 文本行旋转角度。
    "text_rec_score_thresh": 识别分数阈值。
    "return_word_box": 是否返回单字坐标框。
    "rec_texts": 识别出的文本内容列表。
    "rec_scores": 每个文本的置信度分数。
    "rec_polys": 识别阶段确定的多边形坐标。
    "rec_boxes": 轴向外接矩形框坐标 [xmin, ymin, xmax, ymax]。

Usage::

    from pyxllib.ai.ocr import ocr_text

    # 1. 简单识别
    res = ocr_text('test.jpg')
    print(res['rec_texts'])

    # 2. 批量识别
    results = ocr_text(['1.jpg', '2.jpg'])
    for res in results:
        print(res['rec_texts'])
"""

import cv2
import time
import tempfile
import json
from pathlib import Path
import os

from loguru import logger
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR

from pyxllib.file.font import get_chinese_font_path
from pyxllib.prog.ctor_proxy import ConstructorProxy


# 基础版配置，一般都够用了
ConstructorProxy(PaddleOCR, "basic").config(
    lang="ch",
    device="gpu",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
# 需要使用的时候，这样就能获取，而且默认就是全局单例了
# ocr = ConstructorProxy(PaddleOCR, 'basic').get()

# 带全量子模型的版本
ConstructorProxy(PaddleOCR, "full").config(
    lang="ch",
    device="gpu",
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True,
)

# 极低阈值配置，用于尽可能捕获所有疑似文本区域（包含大量噪声，常用于 UI 自动化中的候选区域提取）
ConstructorProxy(PaddleOCR, "high_recall").config(
    lang="ch",
    device="gpu",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    det_db_thresh=0.01,
    det_db_box_thresh=0.01,
    det_db_unclip_ratio=1.5,
)


def generate_test_image(text="你好，PaddleOCR！", output_path=None):
    """生成一张带文本的测试图片"""
    if output_path is None:
        # 在临时目录下生成文件
        output_path = Path(tempfile.gettempdir()) / f"test_ocr_{int(time.time() * 1000)}.jpg"
    else:
        output_path = Path(output_path)

    # 创建一张白色背景的图片
    width, height = 400, 200
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 加载中文字体
    font_path = get_chinese_font_path()
    if font_path:
        font = ImageFont.truetype(font_path, 30)
    else:
        # 如果找不到字体，则使用默认字体（可能不支持中文）
        font = ImageFont.load_default()

    # 在图片上绘制文字
    draw.text((50, 50), text, fill=(0, 0, 0), font=font)
    draw.text((50, 100), "PaddleOCR Test", fill=(255, 0, 0), font=font)

    image.save(str(output_path))
    logger.info(f"测试图片已生成: {output_path}")
    return output_path


def _preprocess_ocr_input(img):
    """
    OCR 输入数据预处理子函数。
    将 Path/str, PIL.Image, numpy(gray/bgra/bgr) 统一转为 RGB 格式的 numpy 数组。
    """
    if isinstance(img, (str, Path)):
        # 既然是文件路径，通常直接传路径给 PaddleOCR 也是支持的，
        # 但为了统一行为，这里也可以选择转成 str 返回，或者读取为 array。
        # 原逻辑倾向于转为 str 让 Paddle 内部处理，或者在这里读出来。
        # 为了最大兼容性，我们保持原逻辑：如果是路径，转 str；
        # *但* 如果 PaddleOCR 内部读取失败，通常建议在这里用 cv2 读好传进去。
        # 这里维持原逻辑：Path -> str
        return str(img)

    elif isinstance(img, Image.Image):  # PIL.Image 对象，需要转为np类型
        # convert('RGB') 确保丢弃 Alpha 通道并转为 RGB
        return np.array(img.convert("RGB"))

    elif isinstance(img, np.ndarray):
        # 处理 cv2 读取的各种 numpy 数组情况
        if img.ndim == 2:  # 灰度图
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3:
            channels = img.shape[2]
            if channels == 4:  # BGRA -> RGB
                return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif channels == 3:  # BGR -> RGB
                # 假设输入是 cv2 默认的 BGR 顺序
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 其他情况（如已经是符合要求的 numpy array 或二进制流等），原样返回
    return img


def ocr_text(img, *, model="basic", **kwargs):
    """PaddleOCR 文本识别接口

    :param str|Path|Image.Image|np.ndarray|list img: 待识别的图片
        - str/Path: 图片路径
        - Image.Image: PIL Image对象
        - np.ndarray: OpenCV读取的numpy数组
        - list: 以上类型的列表，表示批量处理
    :param str|dict model: 模型配置
        - str: 预设的模型名，如 "basic", "full"
        - dict: 自定义初始化参数字典
    :param kwargs: 传递给 ocr_instance.predict() 的其他参数
    :return dict|list: 识别结果
        - 单图: 返回 OCRResult 字典
        - 批量: 返回 OCRResult 字典列表

    >>> res = ocr_text('test.jpg')  # doctest: +SKIP
    >>> print(res['rec_texts'])  # doctest: +SKIP
    ['text1', 'text2']
    """
    # 1. 获取 OCR 实例
    if isinstance(model, str):
        ocr_instance = ConstructorProxy(PaddleOCR, model).get()
    else:
        ocr_instance = ConstructorProxy(PaddleOCR).config(**model).get()

    # 2. 数据预处理（批量/单张处理）
    if isinstance(img, (list, tuple)):
        formatted_img = [_preprocess_ocr_input(i) for i in img]
        is_batch = True
    else:
        formatted_img = _preprocess_ocr_input(img)
        is_batch = False

    # 3. 调用识别
    # PaddleOCR 3.4+ 使用 predict 方法，支持批量处理
    results = ocr_instance.predict(formatted_img, **kwargs)

    # 4. 结果处理
    if not is_batch:
        return results[0] if results else []

    return results


def ocr_to_labelme(ocr_result, shape_type="polygon", image_path=None, label_fields=None, flags=None):
    """将 PaddleOCR 的识别结果转换为 LabelMe 格式

    :param dict|object ocr_result: PaddleOCR predict 返回的单图结果对象或字典
    :param str shape_type: 标注形状类型
        - 'polygon': 多边形 (默认)
        - 'rectangle': 矩形
    :param str image_path: 图片路径，默认优先尝试从结果中获取
    :param list label_fields: 需要包含在 label 中的字段列表
        - None: (默认) 只保存文本内容
        - list: 如 ['text', 'score']，会打包成 JSON 字符串
    :param dict flags: LabelMe 的全局 flags
    :return dict: LabelMe 格式的字典数据

    todo 可以考虑集成到res的成员接口？以及增加保存功能，保存到指定dir/stem，自动生成对应的img和json
    todo 可以简化image_path的配置？
    """
    if hasattr(ocr_result, "json"):
        res = ocr_result.json["res"]
    else:
        res = ocr_result

    # 1. 基础信息
    img_path = image_path or res.get("input_path", "unknown.jpg")
    img_name = Path(img_path).name

    # 尝试获取图片宽高
    height = 0
    width = 0
    if image_path and Path(image_path).exists():
        with Image.open(image_path) as img:
            width, height = img.size

    shapes = []
    texts = res.get("rec_texts", [])
    scores = res.get("rec_scores", [])
    angles = res.get("textline_orientation_angles", [])

    # 2. 准备 label 生成逻辑
    def get_label(idx):
        if not label_fields:
            return texts[idx] if idx < len(texts) else ""

        data = {}
        for field in label_fields:
            if field == "text" and idx < len(texts):
                data["text"] = texts[idx]
            elif field == "score" and idx < len(scores):
                data["score"] = round(float(scores[idx]), 4)
            elif field == "textline_orientation_angles" and idx < len(angles):
                data["angle"] = angles[idx]
            elif field in res:  # 其他可能的字段
                val = res[field]
                if isinstance(val, list) and idx < len(val):
                    data[field] = val[idx]

        if len(data) == 1 and "text" in data:
            return data["text"]
        return json.dumps(data, ensure_ascii=False)

    # 3. 根据 shape_type 选择坐标来源
    if shape_type == "rectangle":
        # 使用 rec_boxes: [xmin, ymin, xmax, ymax]
        boxes = res.get("rec_boxes", [])
        for i, box in enumerate(boxes):
            shapes.append(
                {
                    "label": get_label(i),
                    "points": [[box[0], box[1]], [box[2], box[3]]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {"score": round(float(scores[i]), 4)} if i < len(scores) else {},
                }
            )
    else:
        # 默认使用多边形 dt_polys
        polys = res.get("dt_polys", [])
        for i, poly in enumerate(polys):
            shapes.append(
                {
                    "label": get_label(i),
                    "points": poly,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {"score": round(float(scores[i]), 4)} if i < len(scores) else {},
                }
            )

    return {
        "version": "5.0.1",
        "flags": flags or {},
        "shapes": shapes,
        "imagePath": img_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def test_paddleocr(
    device="gpu",
    lang="ch",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=None,
):
    """测试 PaddleOCR 功能是否正常，生成图片并展示识别结果

    :param device: 运行设备，默认为 gpu
    :param lang: 识别语言，默认为 ch
    :param use_doc_orientation_classify: 是否使用文档方向分类
    :param use_doc_unwarping: 是否使用文档去矫正
    :param use_textline_orientation: 是否使用文本行方向检测
    :param enable_mkldnn: 是否开启 mkldnn 加速，cpu 模式下默认关闭（为了避免 crash）
    """
    # 1. 初始化模型
    t0 = time.time()

    # cpu模式下，默认关闭mkldnn，否则可能会报错
    # NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]
    if device == "cpu" and enable_mkldnn is None:
        enable_mkldnn = False

    kwargs = {}
    if enable_mkldnn is not None:
        kwargs["enable_mkldnn"] = enable_mkldnn

    ocr = PaddleOCR(
        lang=lang,
        device=device,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
        **kwargs,
    )
    t1 = time.time()
    logger.info(f"PaddleOCR 初始化耗时: {t1 - t0:.4f}s")

    # 2. 生成测试图片
    img_path = generate_test_image()

    # 3. 运行 OCR
    t2 = time.time()
    # PaddleOCR 3.4+ 使用 predict 接口
    res_list = ocr.predict(str(img_path))
    t3 = time.time()
    logger.info(f"PaddleOCR 运行耗时: {t3 - t2:.4f}s")

    if not res_list:
        logger.warning("未识别到任何结果")
        return

    res = res_list[0]

    # 4. 导出可视化结果
    vis_path = img_path.with_name(img_path.stem + "_vis.jpg")
    try:
        # 使用 PaddleOCR 3.x 内置的 save_to_img 方法
        res.save_to_img(vis_path)
        logger.info(f"可视化结果已保存: {vis_path}")

        # 5. 打开查看
        try:
            Image.open(vis_path).show()
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"保存或打开图片失败: {e}")


if __name__ == "__main__":
    test_paddleocr("gpu")
