#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/06/02

"""
扩展了些自己的openpyxl工具
"""

from pyxllib.prog.pupil import check_install_package

check_install_package('openpyxl')
check_install_package('premailer')
check_install_package('xlrd2')
check_install_package('yattag')

import re

import openpyxl
from openpyxl import Workbook
from openpyxl.cell.cell import MergedCell
from openpyxl.styles import Font
from openpyxl.utils.cell import get_column_letter
import pandas as pd

from pyxllib.prog.newbie import RunOnlyOnce
from pyxllib.prog.pupil import EnchantBase, EnchantCvt
from pyxllib.algo.specialist import product
from pyxllib.debug.pupil import dprint
from pyxllib.debug.specialist import browser


def excel_addr(n, m) -> str:
    r"""数字索引转excel地址索引

    :param n: 行号，可以输入字符串形式的数字
    :param m: 列号，同上可以输入str的数字
    :return:

    >>> excel_addr(2, 3)
    'C2'
    """
    return f'{get_column_letter(int(m))}{n}'


class EnchantCell(EnchantBase):

    @classmethod
    @RunOnlyOnce
    def enchant(cls):
        names = cls.check_enchant_names([openpyxl.cell.cell.Cell, openpyxl.cell.cell.MergedCell])
        cls._enchant(openpyxl.cell.cell.Cell, names)
        cls._enchant(openpyxl.cell.cell.MergedCell, names)

    @staticmethod
    def in_range(cell):
        """判断一个单元格所在的合并单元格
        >> in_range(ws['C1'])
        <openpyxl.worksheet.cell_range.CellRange> A1:D3
        """
        ws = cell.parent
        for rng in ws.merged_cells.ranges:
            if cell.coordinate in rng:
                break
        else:  # 如果找不到则返回原值
            rng = cell
        return rng

    @staticmethod
    def mcell(cell):
        """返回“有效单元格”，即如果输入的是一个合并单元格，会返回该合并单元格左上角的单元格
        修改左上角单元格的值才是可行、有意义的

        因为跟合并单元格有关，所以 以m前缀 merge
        """
        if isinstance(cell, MergedCell):
            ws = cell.parent
            xy = cell.in_range().top[0]
            return ws[excel_addr(*xy)]
        else:
            return cell

    @staticmethod
    def celltype(cell):
        """
        :param cell: 一个单元格
        :return: 单元格类型
            0：普通单元格
            1：合并单元格其他衍生位置
            2：合并单元格的左上角的位置

        TODO 这个函数还是可以看看能不能有更好的实现、提速
        """
        if isinstance(cell, MergedCell):
            return 1
        elif isinstance(cell.offset(1, 0), MergedCell) or isinstance(cell.offset(0, 1), MergedCell):
            # 这里只能判断可能是合并单元格，具体是不是合并单元格，还要
            rng = cell.in_range()
            return 2 if hasattr(rng, 'size') else 0
        else:
            return 0

    @staticmethod
    def isnone(cell):
        """ 是普通单元格且值为None

        注意合并单元格的衍生单元格不为None
        """
        celltype = cell.celltype()
        return celltype == 0 and cell.value is None

    @staticmethod
    def copy_cell_format(cell, dst_cell):
        """ 单元格全格式复制，需要事先指定好新旧单元格的物理位置
        参考：https://stackoverflow.com/questions/23332259/copy-cell-style-openpyxl
        """
        from copy import copy
        if cell.has_style:
            dst_cell.font = copy(cell.font)  # 字体
            dst_cell.border = copy(cell.border)  # 表格线
            dst_cell.fill = copy(cell.fill)  # 填充色
            dst_cell.number_format = copy(cell.number_format)  # 数字格式
            dst_cell.protection = copy(cell.protection)  # 保护？
            dst_cell.alignment = copy(cell.alignment)  # 对齐格式
            # dst_cell.style = cell.style
        # if cell.comment:
        # 这个会引发AttributeError。。。
        #       vml = fromstring(self.workbook.vba_archive.read(ws.legacy_drawing))
        #   AttributeError: 'NoneType' object has no attribute 'read'
        # dst_cell.comment = copy(cell.comment)
        # 就算开了keep_vba可以强制写入了，打开的时候文件可能还是会错

    @staticmethod
    def copy_cell(cell, dst_cell):
        """ 单元格全格式、包括值的整体复制
        """
        dst_cell.value = cell.value
        cell.copy_cell_format(dst_cell)

    @staticmethod
    def down(cell):
        """ 输入一个单元格，向下移动一格
        注意其跟offset的区别，如果cell是合并单元格，会跳过自身的衍生单元格

        注意这里移动跟excel中操作也不太一样，设计的更加"原子化"，可以多配合cell.mcell功能使用。
        详见：【腾讯文档】cell移动机制说明 https://docs.qq.com/doc/DUkRUaFhlb3l4UG1P
        """
        r, c = cell.row, cell.column
        if cell.celltype():
            rng = cell.in_range()
            r = rng.max_row
        return cell.parent.cell(r + 1, c)

    @staticmethod
    def right(cell):
        r, c = cell.row, cell.column
        if cell.celltype():
            rng = cell.in_range()
            c = rng.max_col
        return cell.parent.cell(r, c + 1)

    @staticmethod
    def up(cell):
        r, c = cell.row, cell.column
        if cell.celltype():
            rng = cell.in_range()
            r = rng.min_row
        return cell.parent.cell(max(r - 1, 1), c)

    @staticmethod
    def left(cell):
        r, c = cell.row, cell.column
        if cell.celltype():
            rng = cell.in_range()
            r = rng.min_col
        return cell.parent.cell(r, max(c - 1, 1))


EnchantCell.enchant()


class EnchantWorksheet(EnchantBase):
    """ 扩展标准的Workshhet功能 """

    @classmethod
    @RunOnlyOnce
    def enchant(cls):
        names = cls.check_enchant_names([openpyxl.worksheet.worksheet.Worksheet], white_list=['_cells_by_row'])
        cls._enchant(openpyxl.worksheet.worksheet.Worksheet, names)

    @staticmethod
    def copy_worksheet(_self, dst_ws):
        """跨工作薄时复制表格内容的功能
        openpyxl自带的Workbook.copy_worksheet没法跨工作薄复制，很坑

        src_ws.copy_worksheet(dst_ws)
        """
        # 1 取每个单元格的值
        for row in _self:
            for cell in row:
                try:
                    cell.copy_cell(dst_ws[cell.coordinate])
                except AttributeError:
                    pass
        # 2 合并单元格的处理
        for rng in _self.merged_cells.ranges:
            dst_ws.merge_cells(rng.ref)
        # 3 其他表格属性的复制
        # 这个从excel读取过来的时候，是不准的，例如D3可能因为关闭时停留窗口的原因误跑到D103
        # dprint(origin_ws.freeze_panes)
        # target_ws.freeze_panes = origin_ws.freeze_panes

    @staticmethod
    def _cells_by_row(_self, min_col, min_row, max_col, max_row, values_only=False):
        """openpyxl的这个迭代器，遇到合并单元格会有bug
        所以我把它重新设计一下~~
        """
        for row in range(min_row, max_row + 1):
            cells = (_self.cell(row=row, column=column) for column in range(min_col, max_col + 1))
            if values_only:
                # yield tuple(cell.value for cell in cells)  # 原代码
                yield tuple(getattr(cell, 'value', None) for cell in cells)
            else:
                yield tuple(cells)

    @staticmethod
    def search(_self, pattern, min_row=None, max_row=None, min_col=None, max_col=None, order=None, direction=0):
        """ 查找满足pattern正则表达式的单元格

        :param pattern: 正则匹配式，可以输入re.complier对象
            会将每个单元格的值转成str，然后进行字符串规则search匹配
            支持多层嵌套 ['模块一', '属性1']
        :param direction: 只有在 pattern 为数组的时候有用
            pattern有多组时，会嵌套找单元格
            每计算出一个条件后，默认取该单元格下方的子区间 axis=0
            如果不是下方，而是右方，可以设置为1
            还有奇葩的上方、左方，可以分别设置为2、3
        :param order: 默认None，也就是 [1, 2] 的效果，规律详见product接口

        >> wb = openpyxl.load_workbook(filename='2020寒假教材各地区数量统计最新2020.1.1.xlsx')
        >> ws = Worksheet(wb['预算总表'])
        >> ws.search('年段')
        <Cell '预算总表'.B2>
        """
        # 1 定界
        x1, x2 = max(min_row or 1, 1), min(max_row or _self.max_row, _self.max_row)
        y1, y2 = max(min_col or 1, 1), min(max_col or _self.max_column, _self.max_column)

        # 2 遍历
        if isinstance(pattern, (list, tuple)):
            cel = None
            for p in pattern:
                cel = _self.search(p, x1, x2, y1, y2, order)
                if cel:
                    # up, down, left, right 找到的单元格四边界
                    l, u, r, d = getattr(cel.in_range(), 'bounds', (cel.column, cel.row, cel.column, cel.row))
                    if direction == 0:
                        x1, y1, y2 = max(x1, d), max(y1, l), min(y2, r)
                    elif direction == 1:
                        x1, x2, y1 = max(x1, u), min(x2, d), max(y1, r)
                    elif direction == 2:
                        x2, y1, y2 = min(x2, d), max(y1, l), min(y2, r)
                    elif direction == 3:
                        x1, x2, y2 = max(x1, u), min(x2, d), min(y2, l)
                    else:
                        raise ValueError(f'direction参数值错误{direction}')
                else:
                    return None
            return cel
        else:
            if isinstance(pattern, str): pattern = re.compile(pattern)
            for x, y in product(range(x1, x2 + 1), range(y1, y2 + 1), order=order):
                cell = _self.cell(x, y)
                if cell.celltype() == 1: continue  # 过滤掉合并单元格位置
                if pattern.search(str(cell.value)): return cell  # 返回满足条件的第一个值

    findcel = search

    @staticmethod
    def findrow(_self, pattern, *args, **kwargs):
        cel = _self.findcel(pattern, *args, **kwargs)
        return cel.row if cel else 0

    @staticmethod
    def findcol(_self, pattern, *args, **kwargs):
        cel = _self.findcel(pattern, *args, **kwargs)
        return cel.column if cel else 0

    @staticmethod
    def browser(_self):
        """注意，这里会去除掉合并单元格"""
        browser(pd.DataFrame(_self.values))

    @staticmethod
    def select_columns(_self, columns, column_name='searchkey'):
        r""" 获取表中columns属性列的值，返回dataframe数据类型

        :param columns: 搜索列名使用正则re.search字符串匹配查找
            可以单列：'attr1'，找到列头后，会一直往后取到最后一个非空值
            也可以多列： ['attr1', 'attr2', 'attr3']
                会结合多个列标题定位，数据从最大的起始行号开始取，
                （TODO 截止到最末非空值所在行  未实现，先用openpyxl自带的max_row判断，不过这个有时会判断过大）
            遇到合并单元格，会寻找其母单元格的值填充
        :param column_name: 返回的df。列名
            origin，原始的列名
            searchkey，搜索时用的查找名
        """
        if not isinstance(columns, (list, tuple)):
            columns = [columns]

        # 1 找到所有标题位置，定位起始行
        cels, names, start_line = [], [], -1
        for search_name in columns:
            cel = _self.findcel(search_name)
            if cel:
                cels.append(cel)
                if column_name == 'searchkey':
                    names.append(str(search_name))
                elif column_name == 'origin':
                    if isinstance(search_name, (list, tuple)) and len(search_name) > 1:
                        names.append('/'.join(list(search_name[:-1]) + [str(cel.value)]))
                    else:
                        names.append(str(cel.value))
                else:
                    raise ValueError(f'{column_name}')
                start_line = max(start_line, cel.down().row)
            else:
                dprint(search_name)  # 找不到指定列

        # 2 获得每列的数据
        datas = {}
        for k, cel in enumerate(cels):
            if cel:
                col = cel.column
                li = []
                for i in range(start_line, _self.max_row + 1):
                    v = _self.cell(i, col).mcell().value  # 注意合并单元格的取值
                    li.append(v)
                datas[names[k]] = li
            else:
                # 如果没找到列，设一个空列
                datas[names[k]] = [None] * (_self.max_row + 1 - start_line)
        df = pd.DataFrame(datas)

        # 3 去除所有空行数据
        df.dropna(how='all', inplace=True)

        return df

    @staticmethod
    def copy_range(_self, cell_range, rows=0, cols=0):
        """ 同表格内的 range 复制操作

        Copy a cell range by the number of rows and/or columns:
        down if rows > 0 and up if rows < 0
        right if cols > 0 and left if cols < 0
        Existing cells will be overwritten.
        Formulae and references will not be updated.
        """
        from openpyxl.worksheet.cell_range import CellRange
        from itertools import product
        # 1 预处理
        if isinstance(cell_range, str):
            cell_range = CellRange(cell_range)
        if not isinstance(cell_range, CellRange):
            raise ValueError("Only CellRange objects can be copied")
        if not rows and not cols:
            return
        min_col, min_row, max_col, max_row = cell_range.bounds
        # 2 注意拷贝顺序跟移动方向是有关系的，要防止被误覆盖，复制了新的值，而非原始值
        r = sorted(range(min_row, max_row + 1), reverse=rows > 0)
        c = sorted(range(min_col, max_col + 1), reverse=cols > 0)
        for row, column in product(r, c):
            _self.cell(row, column).copy_cell(_self.cell(row + rows, column + cols))

    @staticmethod
    def reindex_columns(_self, orders):
        """ 重新排列表格的列顺序

        >> ws.reindex_columns('I,J,A,,,G,B,C,D,F,E,H,,,K'.split(','))

        TODO 支持含合并单元格的整体移动？
        """
        from openpyxl.utils.cell import column_index_from_string
        max_row, max_column = _self.max_row, _self.max_column
        for j, col in enumerate(orders, 1):
            if not col: continue
            _self.copy_range(f'{col}1:{col}{max_row}', cols=max_column + j - column_index_from_string(col))
        _self.delete_cols(1, max_column)

    @staticmethod
    def to_html(ws, *, border=1,
                style='border-collapse:collapse; text-indent:0; margin:0 auto;') -> str:
        r"""
        .from_latex(r'''\begin{tabular}{|c|c|c|c|}
                      \hline
                      1 & 2 & & 4\\
                      \hline
                      \end{tabular}''').to_html())

        ==>

        <table border="1" style="border-collapse: collapse;">
          <tr>
            <td style="text-align:center">
              1
            </td>
            <td style="text-align:center">
              2
            </td>
            <td style="text-align:center"></td>
            <td style="text-align:center">
              4
            </td>
          </tr>
        </table>
        """
        from yattag import Doc, indent

        doc, tag, text = Doc().tagtext()
        tag_attrs = [('border', border), ('style', style)]
        # if self.data_tex:  # 原来做latex的时候有的一个属性
        #     tag_attrs.append(('data-tex', self.data_tex))

        with tag('table', *tag_attrs):
            # dprint(ws.max_row, ws.max_column)
            cols = ws.max_column
            for i in range(1, ws.max_row + 1):
                # TODO 这样跳过其实也不太好，有时候可能就是想创建一个空内容的表格
                for j in range(1, cols + 1):
                    if not ws.cell(i, j).isnone():
                        break
                else:  # 如果没有内容，跳过该行
                    continue

                with tag('tr'):
                    for j in range(1, cols + 1):
                        # ① 判断单元格类型
                        cell = ws.cell(i, j)
                        celltype = cell.celltype()
                        if celltype == 1:  # 合并单元格的衍生单元格
                            continue
                        elif cell.isnone():
                            continue
                        # ② 对齐等格式控制
                        params = {}
                        if celltype == 2:  # 合并单元格的左上角
                            rng = cell.in_range()
                            params['rowspan'] = rng.size['rows']
                            params['colspan'] = rng.size['columns']
                        if cell.alignment.horizontal:
                            params['style'] = 'text-align:' + cell.alignment.horizontal
                        # if cell.alignment.vertical:
                        #     params['valign'] = cell.alignment.vertical
                        with tag('td', **params):
                            v = str(cell.value)
                            # if not v: v = '&nbsp;'  # 200424周五15:40，空值直接上平台表格会被折叠，就加了个空格
                            doc.asis(v, )  # 不能用text，html特殊字符不用逃逸
        # res = indent(doc.getvalue(), indent_text=True)  # 美化输出模式。但是这句在某些场景会有bug。
        res = doc.getvalue()  # 紧凑模式

        return res

    @staticmethod
    def init_from_latex(ws, content):
        """ 注意没有取名为from_latex，因为ws是事先创建好的，这里只是能输入latex代码进行初始化而已 """
        from openpyxl.styles import Border, Alignment, Side

        from pyxllib.text.pupil import grp_bracket
        from pyxllib.text.latex import TexTabular

        BRACE2 = grp_bracket(2, inner=True)
        BRACE5 = grp_bracket(5, inner=True)

        # 暂时统一边框线的样式 borders。不做细化解析
        double = Side(border_style='thin', color='000000')

        # 处理表头
        data_tex = re.search(r'\\begin{tabular}\s*(?:\[.*?\])?\s*' + BRACE5, content).group(1)
        col_pos = TexTabular.parse_align(data_tex)  # 每列的格式控制
        # dprint(self.data_tex, col_pos)
        total_col = len(col_pos)
        # 删除头尾标记
        s = re.sub(r'\\begin{tabular}(?:\[.*?\])?' + BRACE5, '', re.sub(r'\\end{tabular}', '', content))
        row, col = 1, 1

        # 先用简单不严谨的规则确定用全网格，还是无网格
        # if '\\hline' not in s and '\\midrule' not in s:
        #     border = 0

        # 用 \\ 分割处理每一行
        for line in re.split(r'\\\\(?!{)', s)[:-1]:
            # dprint(line)
            # 1 处理当前行的所有列元素
            cur_line = line
            # dprint(line)
            # 清除特殊格式数据
            cur_line = re.sub(r'\\cmidrule' + BRACE2, '', cur_line)
            cur_line = re.sub(r'\\cline' + BRACE2, '', cur_line)
            for t in (r'\midrule', r'\toprule', r'\bottomrule', r'\hline', '\n'):
                cur_line = cur_line.replace(t, '')

            # 遍历每列
            # dprint(cur_line)
            for item in cur_line.strip().split('&'):
                item = item.strip()
                # dprint(item)
                cur_loc = excel_addr(row, col)
                # dprint(row, col)

                if 'multicolumn' in item:
                    size, align, text = TexTabular.parse_multicolumn(item)
                    align = TexTabular.parse_align(align) if align else col_pos[col - 1]  # 如果没有写对齐，用默认列的格式
                    n, m = size
                    # 左右对齐，默认是left
                    align = {'l': 'left', 'c': 'center', 'r': 'right'}.get(align, 'left')
                    cell = ws[cur_loc].mcell()
                    if cell.value:
                        cell.value += '\n' + text
                    else:
                        cell.value = text
                    ws[cur_loc].alignment = Alignment(horizontal=align, vertical='center')
                    merge_loc = excel_addr(row + n - 1, col + m - 1)
                    ws.merge_cells(f'{cur_loc}:{merge_loc}')
                    col += m
                elif 'multirow' in item:
                    n, bigstructs, width, fixup, text = TexTabular.parse_multirow(item, brace_text_only=False)
                    try:
                        ws[cur_loc] = text
                    except AttributeError:
                        # 遇到合并单元格重叠问题，就修改旧的合并单元格，然后添加新单元格
                        # 例如原来 A1:A3 是一个合并单元格，现在要独立一个A3，则原来的部分重置为A1:A2
                        rng = ws[cur_loc].in_range()
                        ws.unmerge_cells(rng.coord)  # 解除旧的合并单元格
                        ws.merge_cells(re.sub(r'\d+$', f'{row - 1}', rng.coord))
                        ws[cur_loc] = text
                    align = {'l': 'left', 'c': 'center', 'r': 'right'}.get(col_pos[col - 1], 'left')
                    ws[cur_loc].alignment = Alignment(horizontal=align, vertical='center')
                    # dprint(item, row, n)
                    merge_loc = excel_addr(row + n - 1, col)
                    ws.merge_cells(f'{cur_loc}:{merge_loc}')
                    col += 1
                else:
                    if ws[cur_loc].celltype() == 0:
                        ws[cur_loc].value = item
                        # dprint(item, col_pos, col)
                        align = {'l': 'left', 'c': 'center', 'r': 'right'}.get(col_pos[col - 1], 'left')
                        ws[cur_loc].alignment = Alignment(horizontal=align, vertical='center')
                    col += 1

            # 2 其他border等格式控制
            if r'\midrule' in line or r'\toprule' in line or r'\bottomrule' in line or r'\hline' in line:
                # 该行画整条线
                loc_1 = excel_addr(row, 1)
                loc_2 = excel_addr(row, total_col)
                comb_loc = f'{loc_1}:{loc_2}'
                for cell in ws[comb_loc][0]:
                    cell.border = Border(top=double)
            if r'\cmidrule' in line:
                for match in re.findall(r'\\cmidrule{([0-9]+)-([0-9]+)}', line):
                    loc_1 = excel_addr(row, match[0])
                    loc_2 = excel_addr(row, match[1])
                    comb_loc = f'{loc_1}:{loc_2}'
                    for cell in ws[comb_loc][0]:
                        cell.border = Border(top=double)
            if r'\cline' in line:
                for match in re.findall(r'\\cline{([0-9]+)-([0-9]+)}', line):
                    loc_1 = excel_addr(row, match[0])
                    loc_2 = excel_addr(row, match[1])
                    comb_loc = f'{loc_1}:{loc_2}'
                    for cell in ws[comb_loc][0]:
                        cell.border = Border(top=double)
            row, col = row + 1, 1

    @staticmethod
    def to_latex(ws):
        from pyxllib.text.latex import TexTabular

        li = []
        n, m = ws.max_row, ws.max_column
        format_count = [''] * m  # 记录每一列中各种对齐格式出现的次数
        merge_count = [0] * m  # 每列累积被合并行数，用于计算cline

        li.append('\\hline')
        for i in range(1, n + 1):
            if ws.cell(i, 1).isnone(): continue
            line = []
            j = 1
            while j < m + 1:
                cell = ws.cell(i, j)
                celltype = cell.celltype()
                if celltype == 0:  # 普通单元格
                    line.append(str(cell.value))
                elif celltype == 1:  # 合并单元格的衍生单元格
                    mcell = cell.mcell()  # 找出其母单元格
                    if mcell.column == cell.column:
                        columns = mcell.in_range().size['columns']
                        if columns > 1:
                            line.append(f'\\multicolumn{{{columns}}}{{|c|}}{{}}')  # 加一个空的multicolumn
                        else:
                            line.append('')  # 加一个空值
                elif celltype == 2:  # 合并单元格的左上角
                    rng = cell.in_range()
                    v = cell.value
                    rows, columns = rng.size['rows'], rng.size['columns']
                    if rows > 1:  # 有合并行
                        v = f'\\multirow{{{rows}}}*{{{v}}}'
                        for k in range(j, j + columns): merge_count[k - 1] = rows - 1
                    if columns > 1:  # 有合并列
                        # horizontal取值有情况有
                        # {‘center’, ‘centerContinuous’, ‘fill’, ‘left’, ‘justify’, ‘distributed’, ‘right’, ‘general’}
                        # 所以如果不是left、center、right，改成默认c
                        align = cell.alignment.horizontal[0]
                        if align not in 'lcr': align = 'c'
                        v = f'\\multicolumn{{{columns}}}{{|{align}|}}{{{v}}}'
                    line.append(str(v))
                    j += columns - 1
                if cell.alignment.horizontal:
                    format_count[j - 1] += cell.alignment.horizontal[0]
                j += 1
            li.append(' & '.join(line) + r'\\ ' + TexTabular.create_cline(merge_count))
            merge_count = [(x - 1 if x > 0 else x) for x in merge_count]
        li.append('\\end{tabular}\n')
        head = '\\begin{tabular}' + TexTabular.create_formats(format_count)
        li = [head] + li  # 开头其实可以最后加，在遍历中先确认每列用到最多的格式情况

        return '\n'.join(li)


EnchantWorksheet.enchant()


class EnchantWorkbook(EnchantBase):
    @classmethod
    @RunOnlyOnce
    def enchant(cls):
        names = cls.check_enchant_names([openpyxl.Workbook])
        cls_names = {'from_html'}
        cls._enchant(openpyxl.Workbook, cls_names, EnchantCvt.staticmethod2classmethod)
        cls._enchant(openpyxl.Workbook, names - cls_names)

    @staticmethod
    def adjust_sheets(wb, new_sheetnames):
        """ 按照 new_sheetnames 的清单重新调整sheets
            在清单里的按顺序罗列
            不在清单里的表格删除
            不能出现wb原本没有的表格名
        """
        for name in set(wb.sheetnames) - set(new_sheetnames):
            # 最好调用标准的remove接口删除sheet
            #   不然虽然能表面上看也能删除sheet，但会有命名空间的一些冗余信息留下
            wb.remove(wb[name])
        wb._sheets = [wb[name] for name in new_sheetnames]
        return wb

    @staticmethod
    def from_html(content):
        from pyxllib.stdlib.tablepyxl.tablepyxl import document_to_workbook
        # 支持多 <table> 结构
        return document_to_workbook(content)

    @staticmethod
    def from_latex(content):
        """
        参考：kun0zhou，https://github.com/kun-zhou/latex2excel/blob/master/latex2excel.py
        """
        from openpyxl import Workbook

        # 可以处理多个表格
        wb = Workbook()
        for idx, s in enumerate(re.findall(r'(\\begin{tabular}.*?\\end{tabular})', content, flags=re.DOTALL), start=1):
            if idx == 1:
                ws = wb.active
                ws.title = 'Table 1'
            else:
                ws = wb.create_sheet(title=f'Table {idx}')
            ws.init_from_latex(s)

        return wb

    @staticmethod
    def to_html(wb) -> str:
        li = []
        for ws in wb.worksheets:
            li.append(ws.to_html())
        return '\n\n'.join(li)

    @staticmethod
    def to_latex(wb):
        li = []
        for ws in wb.worksheets:
            li.append(ws.to_latex())
        return '\n'.join(li)


EnchantWorkbook.enchant()


def demo_openpyxl():
    # 一、新建一个工作薄
    from openpyxl import Workbook
    wb = Workbook()

    # 取一个工作表
    ws = wb.active  # wb['Sheet']，取已知名称、下标的表格，excel不区分大小写，这里索引区分大小写

    # 1 索引单元格的两种方法，及可以用.value获取值
    ws['A2'] = '123'
    dprint(ws.cell(2, 1).value)  # 123

    # 2 合并单元格
    ws.merge_cells('A1:C2')
    dprint(ws['A1'].value)  # None，会把原来A2的内容清除

    # print(ws['A2'].value)  # AttributeError: 'MergedCell' object has no attribute 'value'

    # ws.unmerge_cells('A1:A3')  # ValueError: list.remove(x): x not in list，必须标记完整的合并单元格区域，否则会报错
    ws['A1'].value = '模块一'
    ws['A3'].value = '属性1'
    ws['B3'].value = '属性2'
    ws['C3'].value = '属性3'

    ws.merge_cells('D1:E2')
    ws['D1'].value = '模块二'
    ws['D3'].value = '属性1'
    ws['E3'].value = '属性2'

    dprint(ws['A1'].offset(1, 0).coordinate)  # A2
    dprint(ws['A1'].down().coordinate)  # A3

    # 3 设置单元格样式、格式
    from openpyxl.comments import Comment
    cell = ws['A3']
    cell.font = Font(name='Courier', size=36)
    cell.comment = Comment(text="A comment", author="Author's Name")

    styles = [['Number formats', 'Comma', 'Comma [0]', 'Currency', 'Currency [0]', 'Percent'],
              ['Informative', 'Calculation', 'Total', 'Note', 'Warning Text', 'Explanatory Text'],
              ['Text styles', 'Title', 'Headline 1', 'Headline 2', 'Headline 3', 'Headline 4', 'Hyperlink',
               'Followed Hyperlink', 'Linked Cell'],
              ['Comparisons', 'Input', 'Output', 'Check Cell', 'Good', 'Bad', 'Neutral'],
              ['Highlights', 'Accent1', '20 % - Accent1', '40 % - Accent1', '60 % - Accent1', 'Accent2', 'Accent3',
               'Accent4', 'Accent5', 'Accent6', 'Pandas']]
    for i, name in enumerate(styles, start=4):
        ws.cell(i, 1, name[0])
        for j, v in enumerate(name[1:], start=2):
            ws.cell(i, j, v)
            ws.cell(i, j).style = v

    # 二、测试一些功能
    dprint(ws.search('模块二').coordinate)  # D1
    dprint(ws.search(['模块二', '属性1']).coordinate)  # D3

    dprint(ws.findcol(['模块一', '属性1'], direction=1))  # 0

    wb.save("demo_openpyxl.xlsx")
