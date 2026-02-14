#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
guichat.py - UI Visual Localization Assistant / UI 视觉定位助手
基于多模态大模型和 OCR 技术，提供 UI 界面元素的智能定位功能。

核心功能：
1. 集成 OCR 文本识别，辅助增强定位精度。
2. 自动构建包含视觉上下文和任务描述的 Prompt。
3. 详细的调试记录 (Debug Log)，可视化定位结果。

使用建议：
提问目标的时候，建议详细些把目标、需求描述清楚。
例如，“点击右上角的关闭按钮”比简单的“关闭”效果更好；
“找到包含‘确认’文字的蓝色按钮”比“确认按钮”更精确。

Usage:
>>> from pyxllib.autogui.guichat import GUIChat
>>> chat = GUIChat(debug=False)
>>> # chat.query_rect("screenshot.png", "点击右上角的设置图标")
"""

import os
import re
import sys
import time
import json
import datetime
import tempfile
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from PIL import Image, ImageDraw
from loguru import logger

from pyxllib.ai.chat import Chat
from pyxllib.ai.ocr import ocr_text
from pyxllib.text.document import Document


class GUIChat(Chat):
    """
    专门用于 UI 视觉定位的 Chat 类。
    集成了 OCR 增强、自动 Prompt 构建和调试记录功能。
    """

    def __init__(self, model_name: str = "ollama/qwen3-vl:8b-instruct", debug: bool = True, **kwargs):
        """
        :param model_name: 模型名称
        :param debug: 是否开启调试记录 (Document)
        :param kwargs: 传给 Chat 的其他参数
        """
        super().__init__(model_name, **kwargs)
        self.debug = debug
        self.doc = None
        if self.debug:
            self.doc = Document(title="GUIChat Debug Log", toc=3)
        self.turn_count = 0  # 轮次计数器

        self.ocr_config = "high_recall"

        # 系统提示词模板
        self.system_prompt_template = (
            "你是一个智能 UI 视觉定位助手。\n"
            "你的任务是根据用户的描述，在给定的 UI 界面截图中找到目标元素，并输出其边界框 (Bounding Box)。\n\n"
            "### 辅助信息 (OCR Context)\n"
            "我会提供 OCR 识别到的文本和坐标信息。请注意：\n"
            "1. **OCR 可能有误**：OCR 可能会把 '大地图' 误识别成 '天地图'，或者漏掉某些字符。请结合视觉特征和语义进行模糊匹配。\n"
            "2. **非文本元素**：如果用户寻找的是图标、按钮等非文本元素，OCR 提供的附近文本可以作为定位锚点 (Anchor)。\n"
            "3. **坐标系**：OCR 数据和我要求的输出坐标均使用 **0-1000 的归一化坐标** ([x_min, y_min, x_max, y_max])。\n\n"
            "### 输出格式\n"
            "请在分析思考后，严格按照以下格式输出目标位置：\n"
            "Thought: <你的分析过程，解释你是如何定位到目标的>\n"
            "Box: [x1, y1, x2, y2]\n"
            "如果是查找多个目标，请输出多行 Box: ...\n"
        )

        if self.debug and self.doc:
            self.doc.add_header("System Prompt", level=2)
            self.doc.add_text(f"{self.system_prompt_template}")

        self._last_image_path = None
        # print("DEBUG: initialized _last_image_path")

    def _run_ocr(self, img: Image.Image) -> Optional[Any]:
        """运行 OCR 并返回结果对象

        :param img: PIL Image 对象
        :return: OCR 结果对象或 None
        """
        try:
            # 确保传入的是 PIL Image
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")

            logger.info(f"Running OCR with config: {self.ocr_config}")
            res = ocr_text(img, model=self.ocr_config)
            return res
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            if self.debug and self.doc:
                self.doc.add_text(f"**OCR Failed**: {e}")
            return None

    def _format_ocr_info(self, ocr_res: Any, w: int, h: int) -> str:
        """将 OCR 结果格式化为 Prompt 文本 (归一化坐标)

        :param ocr_res: OCR 结果对象
        :param w: 图片宽度
        :param h: 图片高度
        :return: 格式化后的 Prompt 文本
        """
        if not ocr_res or "rec_boxes" not in ocr_res.json["res"]:
            return "OCR: 未检测到任何内容。"

        boxes = ocr_res.json["res"]["rec_boxes"]
        texts = ocr_res.json["res"]["rec_texts"]

        lines = ["### OCR 检测结果 (Text: [x1, y1, x2, y2] normalized 0-1000)"]

        for box, text in zip(boxes, texts):
            # box is [x1, y1, x2, y2] (pixel)
            x1, y1, x2, y2 = box
            # Normalize
            nx1 = int(x1 / w * 1000)
            ny1 = int(y1 / h * 1000)
            nx2 = int(x2 / w * 1000)
            ny2 = int(y2 / h * 1000)

            clean_text = text.replace("\n", " ").strip()
            lines.append(f"- {clean_text}: [{nx1}, {ny1}, {nx2}, {ny2}]")

        return "\n".join(lines)

    def _extract_boxes(self, text: str) -> List[List[float]]:
        """从模型回复中提取所有 Box

        :param text: 模型回复文本
        :return: Box 列表 [[x1, y1, x2, y2], ...]
        """
        # 匹配 Box: [1, 2, 3, 4] 或 Box: 1, 2, 3, 4
        pattern = r"Box:\s*[\[\(]?\s*([\d\.]+)[,\s]+([\d\.]+)[,\s]+([\d\.]+)[,\s]+([\d\.]+)\s*[\]\)]?"
        matches = re.findall(pattern, text)
        boxes = []
        for m in matches:
            try:
                boxes.append([float(x) for x in m])
            except ValueError:
                pass
        return boxes

    def _denormalize_box(self, box: List[float], w: int, h: int) -> List[int]:
        """归一化坐标 -> 像素坐标

        :param box: 归一化坐标 [x1, y1, x2, y2]
        :param w: 图片宽度
        :param h: 图片高度
        :return: 像素坐标 [x1, y1, x2, y2]
        """
        return [
            int(box[0] / 1000 * w),
            int(box[1] / 1000 * h),
            int(box[2] / 1000 * w),
            int(box[3] / 1000 * h),
        ]

    def _log_interaction(
        self,
        title: str,
        img: Image.Image,
        prompt_text: str,
        response_text: str,
        result_boxes: Optional[List[List[int]]] = None,
        ocr_res: Optional[Any] = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None
    ):
        """记录交互过程到 Document"""
        if not self.doc:
            return

        self.turn_count += 1
        self.doc.add_header(f"Round {self.turn_count}: {title.split(':')[0]}", level=2)

        # 1. Image
        self.doc.add_image(img, title="Input Image", thumbnail_pixels=200_000)

        # Time formatting
        t_fmt = "%H:%M:%S"
        st_str = start_time.strftime(t_fmt) if start_time else "N/A"
        et_str = end_time.strftime(t_fmt) if end_time else datetime.datetime.now().strftime(t_fmt)

        # 2. Prompt (显示完整内容)
        self.doc.add_header(f"Input ({st_str})", level=3)
        self.doc.add_text(prompt_text)

        # 3. Response
        self.doc.add_header(f"Response ({et_str})", level=3)
        self.doc.add_text(response_text)

        # 4. Visualization (如果有结果)
        if result_boxes:
            vis_img = None
            
            # 尝试使用 OCR 结果进行可视化底图生成
            if ocr_res and hasattr(ocr_res, 'save_to_img'):
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    vis_path = tmp.name
                
                try:
                    ocr_res.save_to_img(vis_path)
                    # 必须 copy 一份，否则 close 后文件被删可能导致 lazy load 失败
                    with Image.open(vis_path) as tmp_img:
                        vis_img = tmp_img.copy().convert("RGB")
                except Exception as e:
                    logger.warning(f"Failed to generate OCR visualization: {e}")
                finally:
                    if os.path.exists(vis_path):
                        try:
                            os.remove(vis_path)
                        except OSError:
                            pass

            if vis_img is None:
                vis_img = img.copy()

            draw = ImageDraw.Draw(vis_img)
            w, h = img.size
            vw, vh = vis_img.size
            
            # 判断是否为双栏布局（OCR可视化通常是原图+识别结果图，宽度约为原图2倍）
            # 允许一定的误差
            is_double_width = (vw > w * 1.5)

            for i, box in enumerate(result_boxes):
                # box is pixel coords [x1, y1, x2, y2]
                # Draw on left side (original)
                draw.rectangle(box, outline="red", width=5)
                draw.text((box[0], box[1]), str(i + 1), fill="red")
                
                # Draw on right side if double width
                if is_double_width:
                    # 假设右侧是简单的水平拼接，偏移量为 w
                    # 注意：PaddleOCR 的 save_to_img 实现通常是 hstack
                    box_right = [box[0] + w, box[1], box[2] + w, box[3]]
                    draw.rectangle(box_right, outline="red", width=5)
                    draw.text((box_right[0], box_right[1]), str(i + 1), fill="red")

            self.doc.add_header("Visualized Result", level=3)
            self.doc.add_image(vis_img, title=f"Found {len(result_boxes)} targets")

    def query(self, parts: Union[str, List[Any]], **kwargs) -> Any:
        """覆盖父类 query，增加调试记录。
        注意：这个方法主要用于通用对话，不一定返回 Box。

        :param parts: 输入内容，可以是字符串或列表
        :return: 模型响应
        """
        start_time = datetime.datetime.now()

        if self.debug and self.doc:
            self.turn_count += 1
            t_fmt = "%H:%M:%S"
            st_str = start_time.strftime(t_fmt)

            self.doc.add_header(f"Round {self.turn_count}: Generic Query", level=2)

            self.doc.add_header(f"Input ({st_str})", level=3)
            if isinstance(parts, list):
                for i, p in enumerate(parts):
                    if isinstance(p, dict):
                        if "image_file" in p:
                            img_file = str(p['image_file'])
                            self.doc.add_text(f"**[Part {i}] Image**: {img_file}")
                        elif not p:
                            continue
                        else:
                            self.doc.add_text(f"**[Part {i}] Text**:\n{p}")
                    else:
                        self.doc.add_text(f"**[Part {i}] Text**:\n{p}")
            else:
                self.doc.add_text(f"**Content**:\n{parts}")

        response = super().query(parts, **kwargs)
        end_time = datetime.datetime.now()

        if self.debug and self.doc:
            t_fmt = "%H:%M:%S"
            et_str = end_time.strftime(t_fmt)
            self.doc.add_header(f"Response ({et_str})", level=3)
            self.doc.add_text(f"{response}")

        return response

    def query_rects(self, image_input: Union[str, Path, Image.Image], target_description: str, **kwargs) -> List[List[int]]:
        """获取所有匹配目标的坐标列表。

        :param image_input: 图片路径或 PIL Image 对象
        :param target_description: 目标描述
        :return: 像素坐标列表 [[x1, y1, x2, y2], ...]
        """
        # 1. 准备图片
        if isinstance(image_input, (str, Path)):
            img_path = str(image_input)
            img = Image.open(img_path).convert("RGB")
        else:
            img = image_input.convert("RGB")
            img_path = "In-Memory Image"

        w, h = img.size

        # 2. 运行 OCR
        ocr_res = self._run_ocr(img)
        ocr_text_info = self._format_ocr_info(ocr_res, w, h)

        # 3. 构建 Prompt
        system_prompt = self.system_prompt_template
        user_prompt = f"Target Description: {target_description}\n\n{ocr_text_info}"

        # 构造 pyxllib Chat 接受的格式
        temp_img_path = None
        actual_img_path = img_path
        
        # 如果是内存图片，需要保存临时文件给 Chat
        if not isinstance(image_input, (str, Path)):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img.save(tmp.name)
                temp_img_path = tmp.name
            actual_img_path = temp_img_path

        prompt_parts = [
            system_prompt,  # 将 System Prompt 作为第一部分文本
            {"image_file": actual_img_path},
            user_prompt,
        ]

        # 调用父类 query
        try:
            # 强制使用 temperature=0
            start_time = datetime.datetime.now()
            response = super().query(prompt_parts, temperature=0, **kwargs)
            end_time = datetime.datetime.now()
        finally:
            # 清理临时文件
            if temp_img_path and os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
                except OSError:
                    pass

        # 4. 解析结果
        norm_boxes = self._extract_boxes(str(response))
        pixel_boxes = [self._denormalize_box(b, w, h) for b in norm_boxes]

        # 5. 记录调试
        self._log_interaction(
            f"Query Rects: {target_description}",
            img,
            user_prompt,
            str(response),
            pixel_boxes,
            ocr_res=ocr_res,
            start_time=start_time,
            end_time=end_time,
        )

        return pixel_boxes

    def query_rect(self, image_input: Union[str, Path, Image.Image], target_description: str, **kwargs) -> Optional[List[int]]:
        """获取单个目标（第一个匹配项）。

        :param image_input: 图片路径或 PIL Image
        :param target_description: 目标描述
        :return: 坐标 [x1, y1, x2, y2] 或 None
        """
        boxes = self.query_rects(image_input, target_description, **kwargs)
        if boxes:
            return boxes[0]
        return None

    def browse(self):
        """查看调试日志"""
        if self.doc:
            return self.doc.browse()
        else:
            print("Debug mode is off.")


def main(image=r"C:\Users\kzche\Desktop\2.png", target="点击关闭弹窗", model="ollama/qwen3-vl:8b-instruct"):
    """测试 GUIChat 功能。

    :param image: 图片路径
    :param target: 目标描述
    :param model: 模型名称

    todo 怎么避开本地测试图，可以自动生成？
    """
    if os.path.exists(image):
        print(f"Initializing GUIChat with model: {model}")
        bot = GUIChat(model_name=model, debug=True)

        print(f"Querying: {target}")
        rect = bot.query_rect(image, target)

        print(f"Result: {rect}")

        # 顺便测试一下通用 query
        print("\nTesting generic query...")
        # res = bot.query([{"image_file": image}, "这张图里主要包含了什么内容？请简要描述。"])
        # 第二轮直接提问即可，不需要再次传入图片，Chat 类会自动带上历史上下文
        res = bot.query("这张图里主要包含了什么内容？请简要描述。")
        print(f"Generic Response: {res}")

        # 打开调试报告
        bot.browse()
    else:
        print(f"Image not found: {image}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
