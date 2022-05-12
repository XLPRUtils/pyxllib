# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pyxllib.prog.pupil import check_install_package

# 没有paddle的时候，默认安装
check_install_package('paddle', 'paddlepaddle')
# 可能会遇到这个问题
# https://blog.csdn.net/qq_47997583/article/details/122430776
# pip install opencv-python-headless==4.1.2.30
# 其他依赖库
check_install_package('pyclipper')
check_install_package('imgaug')
check_install_package('lmdb')

import os
import sys

__dir__ = os.path.dirname(__file__)

import paddle

sys.path.append(os.path.join(__dir__, ''))

import cv2
import logging
import numpy as np
from pathlib import Path
import json

from pyxlpr.ppocr.tools.infer import predict_system
from pyxlpr.ppocr.utils.logging import get_logger

logger = get_logger()
from pyxlpr.ppocr.utils.utility import check_and_read_gif, get_image_file_list
from pyxlpr.ppocr.utils.network import maybe_download, download_with_progressbar, is_link, confirm_model_dir_url
from pyxlpr.ppocr.tools.infer.utility import draw_ocr, str2bool, check_gpu
from pyxlpr.ppstructure.utility import init_args, draw_structure_result
from pyxlpr.ppstructure.predict_system import OCRSystem, save_structure_res

from tqdm import tqdm
from pyxllib.xl import run_once, XlPath, Timer
from pyxllib.xlcv import xlcv, xlpil
from pyxllib.algo.geo import rect_bounds

__all__ = [
    'PaddleOCR', 'PPStructure', 'draw_ocr', 'draw_structure_result',
    'save_structure_res', 'download_with_progressbar'
]

SUPPORT_DET_MODEL = ['DB']
VERSION = '2.4'
SUPPORT_REC_MODEL = ['CRNN']
BASE_DIR = os.path.expanduser("~/.paddleocr/")

DEFAULT_OCR_MODEL_VERSION = 'PP-OCR'
DEFAULT_STRUCTURE_MODEL_VERSION = 'STRUCTURE'
MODEL_URLS = {
    'OCR': {
        'PP-OCRv2': {
            'det': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar',
                },
            },
            'rec': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar',
                    'dict_path': './ppocr/utils/ppocr_keys_v1.txt'
                }
            }
        },
        DEFAULT_OCR_MODEL_VERSION: {
            'det': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar',
                },
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_ppocr_mobile_v2.0_det_infer.tar',
                },
                'structure': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar'
                }
            },
            'rec': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/ppocr_keys_v1.txt'
                },
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/en_dict.txt'
                },
                'french': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/french_dict.txt'
                },
                'german': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/german_dict.txt'
                },
                'korean': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/korean_dict.txt'
                },
                'japan': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/japan_dict.txt'
                },
                'chinese_cht': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/chinese_cht_dict.txt'
                },
                'ta': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/ta_dict.txt'
                },
                'te': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/te_dict.txt'
                },
                'ka': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/ka_dict.txt'
                },
                'latin': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/latin_dict.txt'
                },
                'arabic': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/arabic_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/arabic_dict.txt'
                },
                'cyrillic': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/cyrillic_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/cyrillic_dict.txt'
                },
                'devanagari': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/devanagari_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/devanagari_dict.txt'
                },
                'structure': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar',
                    'dict_path': 'ppocr/utils/dict/table_dict.txt'
                }
            },
            'cls': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar',
                }
            },
        }
    },
    'STRUCTURE': {
        DEFAULT_STRUCTURE_MODEL_VERSION: {
            'table': {
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar',
                    'dict_path': 'ppocr/utils/dict/table_structure_dict.txt'
                }
            }
        }
    }
}


def parse_args(mMain=True):
    import argparse
    parser = init_args()
    parser.add_help = mMain
    parser.add_argument("--lang", type=str, default='ch')
    parser.add_argument("--det", type=str2bool, default=True)
    parser.add_argument("--rec", type=str2bool, default=True)
    parser.add_argument("--type", type=str, default='ocr')
    parser.add_argument(
        "--ocr_version",
        type=str,
        default='PP-OCRv2',
        help='OCR Model version, the current model support list is as follows: '
             '1. PP-OCRv2 Support Chinese detection and recognition model. '
             '2. PP-OCR support Chinese detection, recognition and direction classifier and multilingual recognition model.'
    )
    parser.add_argument(
        "--structure_version",
        type=str,
        default='STRUCTURE',
        help='Model version, the current model support list is as follows:'
             ' 1. STRUCTURE Support en table structure model.')

    for action in parser._actions:
        if action.dest in ['rec_char_dict_path', 'table_char_dict_path']:
            action.default = None
    if mMain:
        return parser.parse_args()
    else:
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        return argparse.Namespace(**inference_args_dict)


def parse_lang(lang):
    latin_lang = [
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
        'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
        'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
        'sw', 'tl', 'tr', 'uz', 'vi'
    ]
    arabic_lang = ['ar', 'fa', 'ug', 'ur']
    cyrillic_lang = [
        'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
        'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
    ]
    devanagari_lang = [
        'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
        'sa', 'bgc'
    ]
    if lang in latin_lang:
        lang = "latin"
    elif lang in arabic_lang:
        lang = "arabic"
    elif lang in cyrillic_lang:
        lang = "cyrillic"
    elif lang in devanagari_lang:
        lang = "devanagari"
    assert lang in MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION][
        'rec'], 'param lang must in {}, but got {}'.format(
        MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION]['rec'].keys(), lang)
    if lang == "ch":
        det_lang = "ch"
    elif lang == 'structure':
        det_lang = 'structure'
    else:
        det_lang = "en"
    return lang, det_lang


def get_model_config(type, version, model_type, lang):
    if type == 'OCR':
        DEFAULT_MODEL_VERSION = DEFAULT_OCR_MODEL_VERSION
    elif type == 'STRUCTURE':
        DEFAULT_MODEL_VERSION = DEFAULT_STRUCTURE_MODEL_VERSION
    else:
        raise NotImplementedError
    model_urls = MODEL_URLS[type]
    if version not in model_urls:
        logger.warning('version {} not in {}, auto switch to version {}'.format(
            version, model_urls.keys(), DEFAULT_MODEL_VERSION))
        version = DEFAULT_MODEL_VERSION
    if model_type not in model_urls[version]:
        if model_type in model_urls[DEFAULT_MODEL_VERSION]:
            logger.warning(
                'version {} not support {} models, auto switch to version {}'.
                    format(version, model_type, DEFAULT_MODEL_VERSION))
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error('{} models is not support, we only support {}'.format(
                model_type, model_urls[DEFAULT_MODEL_VERSION].keys()))
            sys.exit(-1)
    if lang not in model_urls[version][model_type]:
        if lang in model_urls[DEFAULT_MODEL_VERSION][model_type]:
            logger.warning(
                'lang {} is not support in {}, auto switch to version {}'.
                    format(lang, version, DEFAULT_MODEL_VERSION))
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error(
                'lang {} is not support, we only support {} for {} models'.
                    format(lang, model_urls[DEFAULT_MODEL_VERSION][model_type].keys(
                ), model_type))
            sys.exit(-1)
    return model_urls[version][model_type][lang]


class PaddleOCR(predict_system.TextSystem):
    def __init__(self, **kwargs):
        """
        paddleocr package
        args:
            **kwargs: other params show in paddleocr --help
        """
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)
        params.use_gpu = check_gpu(params.use_gpu)

        if not params.show_log:
            logger.setLevel(logging.INFO)
        self.use_angle_cls = params.use_angle_cls
        lang, det_lang = parse_lang(params.lang)

        # init model dir
        det_model_config = get_model_config('OCR', params.ocr_version, 'det',
                                            det_lang)
        params.det_model_dir, det_url = confirm_model_dir_url(
            params.det_model_dir,
            os.path.join(BASE_DIR, VERSION, 'ocr', 'det', det_lang),
            det_model_config['url'])
        rec_model_config = get_model_config('OCR', params.ocr_version, 'rec',
                                            lang)
        params.rec_model_dir, rec_url = confirm_model_dir_url(
            params.rec_model_dir,
            os.path.join(BASE_DIR, VERSION, 'ocr', 'rec', lang),
            rec_model_config['url'])
        cls_model_config = get_model_config('OCR', params.ocr_version, 'cls',
                                            'ch')
        params.cls_model_dir, cls_url = confirm_model_dir_url(
            params.cls_model_dir,
            os.path.join(BASE_DIR, VERSION, 'ocr', 'cls'),
            cls_model_config['url'])
        # download model
        maybe_download(params.det_model_dir, det_url)
        maybe_download(params.rec_model_dir, rec_url)
        maybe_download(params.cls_model_dir, cls_url)

        if params.det_algorithm not in SUPPORT_DET_MODEL:
            logger.error('det_algorithm must in {}'.format(SUPPORT_DET_MODEL))
            sys.exit(0)
        if params.rec_algorithm not in SUPPORT_REC_MODEL:
            logger.error('rec_algorithm must in {}'.format(SUPPORT_REC_MODEL))
            sys.exit(0)

        if params.rec_char_dict_path is None:
            params.rec_char_dict_path = str(
                Path(__file__).parent / rec_model_config['dict_path'])

        print(params)
        # init det_model and rec_model
        super().__init__(params)

    @classmethod
    @run_once('ignore,str')
    def build_ppocr(cls, use_angle_cls=True, lang="ch", show_log=False, **kwargs):
        """ 这个识别模型大概要占用850M显存

        指定的 det_model_dir、rec_model_dir 不存在时，会自动下载最新模型放到指定目录里
        """
        # 1 路径类变量自动转str类型，注意None的要跳过
        for k, v in kwargs.items():
            if k.endswith('_dir') or k.endswith('_path'):
                if v is not None:
                    kwargs[k] = str(kwargs[k])
        # 2 构建一个ppocr对象
        ppocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, show_log=show_log, **kwargs)
        # 3 识别一张空图，预初始化，使得后面的计时更准确
        ppocr.ocr(np.zeros([320, 320, 3], dtype='uint8'))
        return ppocr

    def __1_ocr(self):
        """ 识别功能 """
        pass

    def ocr(self, img, det=True, rec=True, cls=True):
        """
        ocr with paddleocr
        args：
            img: img for ocr, support ndarray, img_path and list or ndarray
            det: use text detection or not, if false, only rec will be exec. default is True
            rec: use text recognition or not, if false, only det will be exec. default is True

        使用示例：
        lines = ppocr.ocr(str(imfile))
        for line in lines:
            pts, [text, score] = line
        """
        img = xlcv.read(img, 0)

        assert isinstance(img, (np.ndarray, list, str))
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        if cls == True and self.use_angle_cls == False:
            logger.warning(
                'Since the angle classifier is not initialized, the angle classifier will not be uesd during the forward process'
            )

        if isinstance(img, str):
            # download net image
            if img.startswith('http'):
                download_with_progressbar(img, 'tmp.jpg')
                img = 'tmp.jpg'
            image_file = img
            img, flag = check_and_read_gif(image_file)
            if not flag:
                with open(image_file, 'rb') as f:
                    np_arr = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                return None
        if isinstance(img, np.ndarray) and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if det and rec:
            dt_boxes, rec_res = self.__call__(img, cls)
            return [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
        elif det and not rec:
            dt_boxes, elapse = self.text_detector(img)
            if dt_boxes is None:
                return None
            return [box.tolist() for box in dt_boxes]
        else:
            if not isinstance(img, list):
                img = [img]
            if self.use_angle_cls and cls:
                img, cls_res, elapse = self.text_classifier(img)
                if not rec:
                    return cls_res
            rec_res, elapse = self.text_recognizer(img)
            return rec_res

    def ocr2texts(self, img, sort_textline=False):
        """ 识别后，只返回文本清单

        :param sort_textline: 是否按文本行的几何关系重新排序
        """
        from pyxlpr.data.imtextline import TextlineShape
        lines = self.ocr(img)
        if sort_textline:
            lines.sort(key=lambda x: TextlineShape(x[0]))
        return [x[1][0] for x in lines]

    def rec_singleline(self, im, cls=False):
        """ 只识别一行文本 """
        lines = self.ocr(im, det=False, cls=cls)
        text = ' '.join([line[0] for line in lines])
        score = round(float(sum([line[1] for line in lines])) / len(lines), 4)
        return text, score

    def __2_view(self):
        """ 可视化、生成标注文件（也可以用于自动标注） """
        pass

    def ocr2img(self, img, det=True, cls=True):
        """ 识别并返回结果
        返回np.ndarray类型

        TODO 带det、rec参数的不同可视化效果还没写
        """
        result = self.ocr(img, cls=cls)
        for i in range(len(result)):
            result[i][1] = [result[i][1][0], float(result[i][1][1])]

        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [round(float(line[1][1]), 4) for line in result]

        image = xlpil.read(img).convert('RGB')
        im_show = draw_ocr(image, boxes, txts, scores)
        return im_show

    def ocr2labelme(self, root, det=False, rec=False):
        """ 对root目录下的所有图片，自动进行文字检测、识别

        :param root:
            输入如果是目录，会对目录里所有图片进行处理
            如果是图片，则只会对单张图进行处理，返回对应的json格式数据
        :param det: det和rec默认都关闭，没有功能效果，需要明确指定，防止意外覆盖已有的检测结果
            True，会重置检测结果
            False，直接使用已有的检测结果
            2，将检测结果转为矩形
        :param rec:
            True，检测后获得识别结果
            False，不执行识别，一般是只需要检测的场合
        """
        from pyxlpr.data.labelme import LabelmeDict

        # 1 工具函数
        def det_ocr(f):
            """ 使用程序完整生成一套标注数据 """
            data = LabelmeDict.gen_data(f)
            lines = self.ocr(str(f))
            for line in lines:
                pts, [text, score] = line
                pts = [[int(p[0]), int(p[1])] for p in pts]  # 转整数
                if det == 2:
                    pts = rect_bounds(pts)
                sp = LabelmeDict.gen_shape({'text': text, 'score': round(float(score), 4)}, pts)
                data['shapes'].append(sp)
            f.with_suffix('.json').write_json(data)

        def det_(f):
            """ 只检测不识别，这个一般没太必要，既然检测了，就一起识别了 """
            data = LabelmeDict.gen_data(f)
            lines = self.ocr(str(f), rec=False)
            for pts in lines:
                pts = [[int(p[0]), int(p[1])] for p in pts]  # 转整数
                if det == 2:
                    pts = rect_bounds(pts)
                sp = LabelmeDict.gen_shape('', pts)

                data['shapes'].append(sp)
            f.with_suffix('.json').write_json(data)

        def ocr(f):
            """ 只识别不检测。常用于手动调整框后，再自动识别一遍文本内容 """
            # 没有对应json文件不处理
            f2 = f.with_suffix('.json')
            if not f2.is_file():
                return

            # 读取已有检测数据，只更新识别结果
            image = xlcv.read(f)
            data = f2.read_json()
            for sp in data['shapes']:
                pts = LabelmeDict.to_quad_pts(sp)
                im = xlcv.get_sub(image, pts)
                lines = self.ocr(im, det=False)
                text = ' '.join([line[0] for line in lines])
                score = sum([line[1] for line in lines]) / len(lines)
                sp['label'] = json.dumps({'text': text, 'score': round(float(score), 4)}, ensure_ascii=False)
            f2.write_json(data)

        # 2 遍历文件批量处理
        root = XlPath(root)
        images = list(root.rglob_images('*'))
        for f in tqdm(images):
            if det and rec:
                det_ocr(f)
            elif det and not rec:
                det_(f)
            elif not det and rec:
                ocr(f)

    def __3_has_label_dataset(self):
        """ 有标注的数据的相关处理功能
        比如可以测算指标分数
        """
        pass

    def det_metric(self, dataset, *, print_mode=False):
        """  计算检测的分数和速度

        :param dataset:
            输入一个dataset，一般是用build_dataset生成的，可以把各种类型的数据统一为一个标准结构
            也可以自定义输入，只要可遍历，每个元素有polys标注，和image图片的二进制数据就行
            但一般还是建议走ppocr框架，里面有很多内置好的数据解析功能，能省很多重复工作
        :param print_mode: 是否输出运行速度信息
        """
        from pyxlpr.ppocr.metrics.eval_det_iou import DetectionIoUEvaluator

        # 1 对所有数据图片进行推断，并计时
        timer = Timer('总共耗时')
        gts, preds = [], []
        for x in dataset:
            # 注意这里图片已经先读入二进制数据了，所以会比实际部署中输入路径的方式快一个读取二进制数据的时间
            timer.start()
            # 要去掉难样本
            # trick: 没有文本的图在处理中，dataset会自动过滤掉，为了避免被过滤，会加一个w=h=1的难样本框
            gts.append([poly for poly, tag in zip(x['polys'], x['ignore_tags']) if not tag])
            img = xlcv.read_from_buffer(x['image'])
            lines = self.ocr(img, rec=False)
            preds.append(lines)
            timer.stop()

        if print_mode:
            timer.report()

        # 2 精度测评
        metric = DetectionIoUEvaluator.eval(gts, preds)
        metric['total_frame'] = len(timer.data)
        metric['fps'] = metric['total_frame'] / sum(timer.data)
        return metric

    def rec_metric_labelme(self, root, *, cls=False, bc=False, print_mode=True, ignores=None):
        """
        :param bc: 是否打开bcompare比较所有识别错误的内容
        :param dict ignores: 不处理的特殊标记
            比如 ignores={'content_class': '其它类'}
        """
        from pyxllib.debug.specialist import bcompare
        from pyxlpr.ppocr.metrics.rec_metric import RecMetric
        from pyxllib.prog.pupil import DictTool

        # 1 读取检测标注、调用self进行检测
        timer1, timer2 = Timer('读图速度'), Timer('总共耗时')
        # 有json文件才算有标注，空图最好也能对应一份空shapes的json文件才会进行判断
        files = list(XlPath(root).rglob_files('*.json'))
        tags, gts, preds = [], [], []
        for f in tqdm(files):
            data = f.read_json()

            timer1.start()
            img = xlcv.read(f.parent / data['imagePath'])
            timer1.stop()

            for i, sp in enumerate(data['shapes']):
                attr = DictTool.json_loads(sp['label'], 'text')


                # if ignores:
                #     for k, v in ignores.items():
                #         if attr.get(k) == v:
                #             continue

                tags.append(f'{f.stem}_{i:03}')  # 这里并不是要真的生成图片，所以有一定重复没有关系
                subimg = xlcv.get_sub(img, sp['points'])
                timer2.start()
                text, score = self.rec_singleline(subimg, cls=cls)
                timer2.stop()
                preds.append(text)
                gts.append(attr['text'])

        # 2 精度测评及测速
        metrics = RecMetric.eval(preds, gts)
        for k, v in metrics.items():
            metrics[k] = round(v, 4)
        if print_mode:
            timer1.report()
            timer2.report()
            print(metrics)
            print('fps={:.2f}'.format(1 / np.mean(timer2.data)))

        # 3 bc可视化比较
        if bc:
            left = '\n'.join([f'{t}\t{x}' for t, x, y in zip(tags, gts, preds) if x != y])
            right = '\n'.join([f'{t}\t{y}' for t, x, y in zip(tags, gts, preds) if x != y])
            bcompare(left, right)

    def rec_bc_labelme(self, root):
        """ 用bc可视化显示识别模型效果

        :param root: labelme格式数据所在的根目录
        """


build_paddleocr = PaddleOCR.build_ppocr


class PPStructure(OCRSystem):
    def __init__(self, **kwargs):
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)
        params.use_gpu = check_gpu(params.use_gpu)

        if not params.show_log:
            logger.setLevel(logging.INFO)
        lang, det_lang = parse_lang(params.lang)

        # init model dir
        det_model_config = get_model_config('OCR', params.ocr_version, 'det',
                                            det_lang)
        params.det_model_dir, det_url = confirm_model_dir_url(
            params.det_model_dir,
            os.path.join(BASE_DIR, VERSION, 'ocr', 'det', det_lang),
            det_model_config['url'])
        rec_model_config = get_model_config('OCR', params.ocr_version, 'rec',
                                            lang)
        params.rec_model_dir, rec_url = confirm_model_dir_url(
            params.rec_model_dir,
            os.path.join(BASE_DIR, VERSION, 'ocr', 'rec', lang),
            rec_model_config['url'])
        table_model_config = get_model_config(
            'STRUCTURE', params.structure_version, 'table', 'en')
        params.table_model_dir, table_url = confirm_model_dir_url(
            params.table_model_dir,
            os.path.join(BASE_DIR, VERSION, 'ocr', 'table'),
            table_model_config['url'])
        # download model
        maybe_download(params.det_model_dir, det_url)
        maybe_download(params.rec_model_dir, rec_url)
        maybe_download(params.table_model_dir, table_url)

        if params.rec_char_dict_path is None:
            params.rec_char_dict_path = str(
                Path(__file__).parent / rec_model_config['dict_path'])
        if params.table_char_dict_path is None:
            params.table_char_dict_path = str(
                Path(__file__).parent / table_model_config['dict_path'])

        print(params)
        super().__init__(params)

    def __call__(self, img):
        if isinstance(img, str):
            # download net image
            if img.startswith('http'):
                download_with_progressbar(img, 'tmp.jpg')
                img = 'tmp.jpg'
            image_file = img
            img, flag = check_and_read_gif(image_file)
            if not flag:
                with open(image_file, 'rb') as f:
                    np_arr = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                return None
        if isinstance(img, np.ndarray) and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        res = super().__call__(img)
        return res


def main():
    # for cmd
    args = parse_args(mMain=True)
    image_dir = args.image_dir
    if is_link(image_dir):
        download_with_progressbar(image_dir, 'tmp.jpg')
        image_file_list = ['tmp.jpg']
    else:
        image_file_list = get_image_file_list(args.image_dir)
    if len(image_file_list) == 0:
        logger.error('no images find in {}'.format(args.image_dir))
        return
    if args.type == 'ocr':
        engine = PaddleOCR(**(args.__dict__))
    elif args.type == 'structure':
        engine = PPStructure(**(args.__dict__))
    else:
        raise NotImplementedError

    for img_path in image_file_list:
        img_name = os.path.basename(img_path).split('.')[0]
        logger.info('{}{}{}'.format('*' * 10, img_path, '*' * 10))
        if args.type == 'ocr':
            result = engine.ocr(img_path,
                                det=args.det,
                                rec=args.rec,
                                cls=args.use_angle_cls)
            if result is not None:
                for line in result:
                    logger.info(line)
        elif args.type == 'structure':
            result = engine(img_path)
            save_structure_res(result, args.output, img_name)

            for item in result:
                item.pop('img')
                logger.info(item)
