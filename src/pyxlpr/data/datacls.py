#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/05/19 16:28

""" 相关数据格式类 """

from pathlib import Path
from pyxlpr.ai import *
from pyxlpr.data import *

# from pyxlpr.data.imtextline import TextlineShape

__1_zcdata = """
"""


class ZcTextGt:
    def __init__(self, root, data=None, *, imdir='images', parts=None):
        """
        :param root: 数据根目录
        :param imdir: 所在图片子目录
        :param parts: ['test.txt', 'train.txt'] 等分块标记

        备注：具体文本数据好像不太适合直接存到内存里，就先不存了。
            但是这个类至少可以把相关功能整合在一起，不零散。
        """
        self.root = Dir(root)
        self.imdir = Dir(imdir, self.root)
        self.parts = parts or []

        if data is None:
            pass
        self.data = data

    def writes_from_coco(self, gt_dict, *, prt=False):
        """ coco标注 --> 智财偏好的文本标注格式

        :param gt_dict: coco 的 gt 字典

        TODO gt_dict可能是过量版，增设筛选规则？
        """
        items = list(CocoData(gt_dict).group_gt())
        for img, anns in tqdm(items, desc='生成ZcTextGt的txt标注文件', disable=not prt):
            content = []
            for ann in anns:
                ltrb = xywh2ltrb(ann['bbox'])
                ltrb = ','.join([str(int(v)) for v in ltrb])
                content.append('\t'.join([ltrb, ann['label']]))
            File(img['file_name'], self.imdir, suffix='.txt').write('\n'.join(content))

    def writes(self, *, max_workers=8, prt=False):
        """ 重新写入txt文件 """

        def write(x):
            file, data = x
            if file:  # 如果文件存在，要遵循原有的编码规则
                with open(str(file), 'rb') as f:
                    bstr = f.read()
                encoding = get_encoding(bstr)
                file.write(data, encoding=encoding, if_exists='replace')
            else:  # 否则直接写入
                file.write(data)

        mtqdm(write, self.data, desc='写入labelme json数据', max_workers=max_workers, disable=not prt)


class ZcKvGt(ZcTextGt):

    def writes_from_coco(self, gt_dict, *, prt=False):
        """ coco标注 -> 智财偏好的，带类别的文本标注格式 """
        items = list(CocoData(gt_dict).group_gt())
        for img, anns in tqdm(items, '生成ZcKvGt的txt标注文件', disable=not prt):
            content = []
            for ann in anns:
                ltrb = xywh2ltrb(ann['bbox'])
                ltrb = ','.join([str(int(v)) for v in ltrb])
                cat_id = ann['category_id']
                cat = (cat_id + 1) // 2
                if cat == 5:
                    cat = 0
                    kv = -1
                else:
                    kv = (cat - 1) % 2
                content.append('\t'.join([ltrb, str(cat), str(kv), ann['label']]))
            File(img['file_name'], self.imdir, suffix='.txt').write('\n'.join(content))


class ZcKvDtOld:
    """ 旧版本的解析器 """

    def __init__(self, data=None):
        """
        :param data:
                  {filepath1: [{'logo': True, 'gt': 'address', 'pr': 'address', 'lb': 'LOT 3'},
                              {...},
                              ...]
                  filepath2: ...,
                  ...
                  }
        """
        self.data = data

    @classmethod
    def init_from_zc_txt(cls, file, *, reverse_annos=False):
        """ 从文件解析出字典结构数据

        :return: {filepath1: [{'logo': True, 'gt': 'address', 'pr': 'address', 'lb': 'LOT 3'},
                              {...},
                              ...]
                  filepath2: ...,
                  ...
                  }
        """
        content = File(file).read()
        content = re.sub(r'(.+?)(\n\n)([^\n]+\n)', r'\3\1\2', content, flags=re.DOTALL)
        parts = ContentPartSpliter.multi_blank_lines(content)

        data = dict()
        for pt in parts:
            lines = pt.splitlines()
            filepath = lines[0]
            annos = []
            for line in lines[1:]:  # 第1行是文件名，删掉
                m = re.match(r'(.+?), GT: (.+?), PR: (.+?), LB: (.+)', line)
                logo, gt, pr, lb = m.groups()
                annos.append({'logo': logo == 'True', 'gt': gt, 'pr': pr, 'lb': lb})
            if reverse_annos:
                annos = list(reversed(annos))
            data[filepath] = annos

        return cls(data)

    def to_coco_dt(self, gt_dict, *, printf=True):
        """ 配合gt标注文件，做更精细的zc结果解析
        """
        cat2id = {c['name']: c['id'] for c in gt_dict['categories']}
        gt_annos = {x[0]['file_name']: x for x in CocoData(gt_dict).group_gt()}
        dt_list = []
        for file, dt_annos in self.data.items():
            file_name = pathlib.Path(file).name
            image, annos = gt_annos[file_name]

            # 如果gt的annos比dt_annos少，需要扩充下，默认按最后一个gt的an填充
            n, m = len(annos), len(dt_annos)
            assert n == m  # 智财如果不带box出来，是要强制框数量相同的！否则box怎么一一对应？
            if n < m:
                annos += [annos[-1]] * (m - n)

            # TODO 有些不是按顺序匹配的，增加按文本匹配的功能

            for line, an in zip(dt_annos, annos):
                gt, pr, lb = an['gt'], an['pr'], an['lb']
                if printf and lb != an['label']:
                    print(file_name, lb, '<=>', an['label'])  # gt和dt的框没对应上，最好检查下问题
                # 附加值是识别错误的类别，没有则代表识别正确
                dt_list.append(
                    {'image_id': an['image_id'], 'category_id': cat2id[pr],
                     'bbox': an['bbox'], 'score': 1, 'label': lb})
        return dt_list


class ZcKvDt(ZcKvDtOld):
    """ 智财预测结果文件的通用解析类

    这里是210510周一16:24新版的结果，文件顺序头写对了，并且增加了cs每个结果的置信度
    """

    @classmethod
    def init_from_zc_txt(cls, file, *, reverse_annos=False):
        """ 从文件解析出字典结构数据

        有时候可能没有对应的 coco gt，则可以用这个直接把文件解析为内存数据处理

        :param reverse_annos: 是否对每个图片的标注结果，进行顺序反转
        :return: {filepath1: [{'logo': True, 'gt': 'address', 'pr': 'address', 'lb': 'LOT 3', 'cs': 1.0},
                              {...},
                              ...]
                  filepath2: ...,
                  ...
                  }
        """
        content = File(file).read()
        parts = ContentPartSpliter.multi_blank_lines(content)

        data = dict()
        for pt in parts:
            lines = pt.splitlines()
            filepath = lines[0]
            annos = []
            for line in lines[1:]:  # 第1行是文件名，删掉
                m = re.match(r'(.+?), GT: (.+?), PR: (.+?), LB: (.+?), CS: (.+)', line)
                logo, gt, pr, lb, cs = m.groups()
                annos.append({'logo': logo == 'True', 'gt': gt, 'pr': pr, 'lb': lb, 'cs': float(cs)})
            if reverse_annos:
                annos = list(reversed(annos))
            data[filepath] = annos

        return cls(data)

    def to_coco_dt(self, gt_dict, *, prt=True):
        """ 转coco的dt格式

        :param gt_dict: 需要有参照的gt文件，才能知道图片id，以及补充box位置信息
        """
        cat2id = {c['name']: c['id'] for c in gt_dict['categories']}
        gt_annos = {x[0]['file_name']: x for x in CocoData(gt_dict).group_gt()}
        dt_list = []
        for file, zc_annos in self.data.items():
            file_name = pathlib.Path(file).name
            image, im_annos = gt_annos[file_name]

            # 如果gt的annos比dt_annos少，需要扩充下，默认按最后一个gt的an填充
            n, m = len(im_annos), len(zc_annos)
            if n == m:
                # assert n == m  # 智财如果不带box出来，是要强制框数量相同的！否则box怎么一一对应？
                # if n < m:
                #     annos += [annos[-1]] * (m - n)
                # 要以顺序整体有依据，文本内容匹配为辅的策略配对
                im_annos.sort(key=lambda x: x['label'])
                zc_annos.sort(key=lambda x: x['lb'])
                dt_annos = []

                for a, b in zip(zc_annos, im_annos):
                    gt, pr, lb, cs = a['gt'], a['pr'], a['lb'], a['cs']
                    if prt and lb != b['label']:
                        # gt和dt的框没对应上，最好检查下问题
                        warn = ' '.join([file_name, lb, '<=>', b['label']])
                        dprint(warn)
                    # 附加值是识别错误的类别，没有则代表识别正确
                    dt_annos.append(
                        {'image_id': b['image_id'], 'category_id': cat2id[pr],
                         'bbox': b['bbox'], 'score': cs, 'label': lb})

                # 已经协调确定了空间顺序，但以防万一，可以再按空间排序下给到下游任务
                #   为了效率，也可以确保gt的annos有序，操作时annos顺序不动
                #   这里是前面为了匹配，已经把排序搞乱了，这里是必须要重排
                dt_annos.sort(key=lambda x: TextlineShape(xywh2ltrb(x['bbox'])))  # 几何重排
                dt_list += dt_annos
            else:  # 否则不匹配用不匹配的玩法
                raise NotImplementedError

        return dt_list


__2_other = """
"""


class SroieTextData:
    """ sroie task1、task2 的标注数据

    72,25,326,25,326,64,72,64,TAN WOON YANN
    50,82,440,82,440,121,50,121,BOOK TA .K(TAMAN DAYA) SDN BND
    205,121,285,121,285,139,205,139,789417-W
    """

    def __init__(self, root, part=None):
        """
        :param root: 'data/task1+2'
            images，目录下有987份jpg图片、txt标注
            test.txt，361张jpg图片清单
            train.txt，626张jpg图片清单
        :param part:
            'test', 只记录test的361张图书
            'train' 等同理
        """
        pass

    def to_coco(self):
        pass


class SroieClsData:
    """ sroie task3 关键信息提取的标注 """
    pass


class BaiduOcrRes:
    def __init__(self, data=None):
        """
        :param data: dict (key 文件名 -> value 识别结果dict)
            {'words_result':
                [ {'words': 'xxxx', 'location': {'top': 130, 'left': ~, 'width': ~, 'height': ~}}, ... ]
             'log_id': 136768...
             'words_result_num': 39
            }
        """
        self.data = data

    @classmethod
    def init_from_hxl_txt(cls, file):
        """ 按训龙给的txt格式来初始化

        数据格式：
            一共22499行，大概是所有agreement，但可能有些空白图所以没结果
            每行第一段是png完整路径，第二段是百度api返回的json读取为dict后直接print的结果
        """
        lines = File(file).read().splitlines()
        data = dict()
        for line in tqdm(lines, '解析txt中每张图对应的字典识别数据'):
            if line == '': continue
            # 切分路径和json数据
            # imfile, dictdata = line.split(maxsplit=1)
            imfile, dictdata = re.split(r'\s+', line, maxsplit=1)
            if 'debug' in str(imfile): continue  # 有65张debug_的图片不要（后记：服务器上多余的65个debug图已删）
            data[imfile] = eval(dictdata)

        return cls(data)

    def check(self, imdir):
        """ 检查json数据的一些问题 """
        with TicToc('缺失的文件'):
            files = Path(str(imdir)).glob('*.jpg')
            files1 = {f.stem for f in files}
            files2 = {Path(f).stem for f in self.data.keys()}
            files3 = {Path(f).stem for f in self.data.keys() if ('debug' in str(f))}
            print(f'缺失{len(files1 - files2)}个文件的识别结果')
            print(f'多出{len(files2 - files1)}个文件的识别结果')
            print(f'{len(files3)}个debug文件')
            sys.stderr.flush()

        with TicToc('check errors'):
            ct = Counter()
            for k, v in self.data.items():
                if 'error_code' in v:
                    ct[v['error_code']] += 1
                    print(k, v)
            print(ct.most_common())

    def to_coco_gt(self, images, *, start_dt_id=0):
        """ 转成coco格式

        :param images: coco gt格式的参考images （可以train、val各生成一个返回）
            TODO 可以扩展支持输入图片所在目录的形式初始化的方法
        :param start_dt_id: annotation起始编号
        :return: coco gt格式的字典
        """
        # 辅助数组
        image_files = {Path(x['file_name']).stem: x['id'] for x in images}
        # 遍历判断要处理的文件
        annotations = []
        for k, v in tqdm(self.data.items(), '解析出每张图片识别结果对应的annotations'):
            stem = Path(k).stem
            if stem not in image_files or 'error_code' in v:
                continue
            image_id = image_files[stem]
            for item in v['words_result']:
                loc = item['location']
                bbox = [loc['left'], loc['top'], loc['width'], loc['height']]
                start_dt_id += 1
                an = CocoGtData.gen_annotation(id=start_dt_id, bbox=bbox, image_id=image_id, label=item['words'])
                annotations.append(an)
        return {'images': images,
                'annotations': annotations,
                'categories': CocoGtData.gen_categories(['text'])}


if __name__ == '__main__':
    os.chdir('D:/home/datasets/textGroup/SROIE2019+/data/task3_testcrop')
    with TicToc(__name__):
        ld = LabelmeDataset.init_from_coco(r'test', 'data_crop.json')
        ld.writes()
        # print(ld)
