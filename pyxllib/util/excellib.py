#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2019/12/11 14:14

"""
这里的代码抄自tablepyxl

然后末尾再扩展了些自己的openpyxl工具
"""

from code4101py.util.debuglib import *

try:
    import openpyxl
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'premailer'])
    subprocess.run(['pip3', 'install', 'openpyxl'])
    import openpyxl


____tablepyxl_style = """
tablepyxl.style的代码
"""


# This is where we handle translating css styles into openpyxl styles
# and cascading those from parent to child in the dom.

from openpyxl.cell import cell
from openpyxl.styles import Font, Alignment, PatternFill, NamedStyle, Border, Side, Color
from openpyxl.styles.fills import FILL_SOLID
from openpyxl.styles.numbers import FORMAT_CURRENCY_USD_SIMPLE, FORMAT_PERCENTAGE
from openpyxl.styles.colors import BLACK


from lxml import html
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from premailer import Premailer


import openpyxl.worksheet.worksheet

FORMAT_DATE_MMDDYYYY = 'mm/dd/yyyy'


def colormap(color):
    """
    Convenience for looking up known colors
    """
    cmap = {'black': BLACK}
    return cmap.get(color, color)


def style_string_to_dict(style):
    """
    Convert css style string to a python dictionary
    """
    def clean_split(string, delim):
        return (s.strip() for s in string.split(delim))
    styles = [clean_split(s, ":") for s in style.split(";") if ":" in s]
    return dict(styles)


def get_side(style, name):
    return {'border_style': style.get('border-{}-style'.format(name)),
            'color': colormap(style.get('border-{}-color'.format(name)))}


known_styles = {}


def style_dict_to_named_style(style_dict, number_format=None):
    """
    Change css style (stored in a python dictionary) to openpyxl NamedStyle
    """

    style_and_format_string = str({
        'style_dict': style_dict,
        'parent': style_dict.parent,
        'number_format': number_format,
    })

    if style_and_format_string not in known_styles:
        # Font
        font = Font(bold=style_dict.get('font-weight') == 'bold',
                    color=style_dict.get_color('color', None),
                    size=style_dict.get('font-size'))

        # Alignment
        vertical = style_dict.get('vertical-align', 'center')
        if vertical not in {'bottom', 'justify', 'distributed', 'top', 'center'}: vertical = 'center'
        alignment = Alignment(horizontal=style_dict.get('text-align', 'general'),
                              vertical=vertical,
                              wrap_text=style_dict.get('white-space', 'nowrap') == 'normal')

        # Fill
        bg_color = style_dict.get_color('background-color')
        fg_color = style_dict.get_color('foreground-color', Color())
        fill_type = style_dict.get('fill-type')
        if bg_color and bg_color != 'transparent':
            fill = PatternFill(fill_type=fill_type or FILL_SOLID,
                               start_color=bg_color,
                               end_color=fg_color)
        else:
            fill = PatternFill()

        # Border
        border = Border(left=Side(**get_side(style_dict, 'left')),
                        right=Side(**get_side(style_dict, 'right')),
                        top=Side(**get_side(style_dict, 'top')),
                        bottom=Side(**get_side(style_dict, 'bottom')),
                        diagonal=Side(**get_side(style_dict, 'diagonal')),
                        diagonal_direction=None,
                        outline=Side(**get_side(style_dict, 'outline')),
                        vertical=None,
                        horizontal=None)

        name = 'Style {}'.format(len(known_styles) + 1)

        pyxl_style = NamedStyle(name=name, font=font, fill=fill, alignment=alignment, border=border,
                                number_format=number_format)

        known_styles[style_and_format_string] = pyxl_style

    return known_styles[style_and_format_string]


class StyleDict(dict):
    """
    It's like a dictionary, but it looks for items in the parent dictionary
    """
    def __init__(self, *args, **kwargs):
        self.parent = kwargs.pop('parent', None)
        super(StyleDict, self).__init__(*args, **kwargs)

    def __getitem__(self, item):
        if item in self:
            return super(StyleDict, self).__getitem__(item)
        elif self.parent:
            return self.parent[item]
        else:
            raise KeyError('{} not found'.format(item))

    def __hash__(self):
        return hash(tuple([(k, self.get(k)) for k in self._keys()]))

    # Yielding the keys avoids creating unnecessary data structures
    # and happily works with both python2 and python3 where the
    # .keys() method is a dictionary_view in python3 and a list in python2.
    def _keys(self):
        yielded = set()
        for k in self.keys():
            yielded.add(k)
            yield k
        if self.parent:
            for k in self.parent._keys():
                if k not in yielded:
                    yielded.add(k)
                    yield k

    def get(self, k, d=None):
        try:
            return self[k]
        except KeyError:
            return d

    def get_color(self, k, d=None):
        """
        Strip leading # off colors if necessary
        """
        color = self.get(k, d)
        if hasattr(color, 'startswith') and color.startswith('#'):
            color = color[1:]
            if len(color) == 3:  # Premailers reduces colors like #00ff00 to #0f0, openpyxl doesn't like that
                color = ''.join(2 * c for c in color)
        return color


class Element(object):
    """
    Our base class for representing an html element along with a cascading style.
    The element is created along with a parent so that the StyleDict that we store
    can point to the parent's StyleDict.
    """
    def __init__(self, element, parent=None):
        self.element = element
        self.number_format = None
        parent_style = parent.style_dict if parent else None
        self.style_dict = StyleDict(style_string_to_dict(element.get('style', '')), parent=parent_style)
        self._style_cache = None

    def style(self):
        """
        Turn the css styles for this element into an openpyxl NamedStyle.
        """
        if not self._style_cache:
            self._style_cache = style_dict_to_named_style(self.style_dict, number_format=self.number_format)
        return self._style_cache

    def get_dimension(self, dimension_key):
        """
        Extracts the dimension from the style dict of the Element and returns it as a float.
        """
        dimension = self.style_dict.get(dimension_key)
        if dimension:
            if dimension[-2:] in ['px', 'em', 'pt', 'in', 'cm']:
                dimension = dimension[:-2]
            dimension = float(dimension)
        return dimension


class Table(Element):
    """
    The concrete implementations of Elements are semantically named for the types of elements we are interested in.
    This defines a very concrete tree structure for html tables that we expect to deal with. I prefer this compared to
    allowing Element to have an arbitrary number of children and dealing with an abstract element tree.

    """
    def __init__(self, table):
        """
        takes an html table object (from lxml)
        """
        super(Table, self).__init__(table)
        table_head = table.find('thead')
        self.head = TableHead(table_head, parent=self) if table_head is not None else None
        table_body = table.find('tbody')
        self.body = TableBody(table_body if table_body is not None else table, parent=self)


class TableHead(Element):
    """
    This class maps to the `<th>` element of the html table.
    """
    def __init__(self, head, parent=None):
        super(TableHead, self).__init__(head, parent=parent)
        self.rows = [TableRow(tr, parent=self) for tr in head.findall('tr')]


class TableBody(Element):
    """
    This class maps to the `<tbody>` element of the html table.
    """
    def __init__(self, body, parent=None):
        super(TableBody, self).__init__(body, parent=parent)
        self.rows = [TableRow(tr, parent=self) for tr in body.findall('tr')]


class TableRow(Element):
    """
    This class maps to the `<tr>` element of the html table.
    """
    def __init__(self, tr, parent=None):
        super(TableRow, self).__init__(tr, parent=parent)
        self.cells = [TableCell(cell, parent=self) for cell in tr.findall('th') + tr.findall('td')]


def element_to_string(el):
    return _element_to_string(el).strip()


def _element_to_string(el):
    string = ''

    for x in el.iterchildren():
        # 表格里的内容保持不变
        # string += '\n' + _element_to_string(x)
        string += html.tostring(x, encoding='unicode', with_tail=False)

    text = el.text.strip() if el.text else ''
    tail = el.tail.strip() if el.tail else ''

    return text + string + '\n' + tail


class TableCell(Element):
    """
    This class maps to the `<td>` element of the html table.
    """
    CELL_TYPES = {'TYPE_STRING', 'TYPE_FORMULA', 'TYPE_NUMERIC', 'TYPE_BOOL', 'TYPE_CURRENCY', 'TYPE_PERCENTAGE',
                  'TYPE_NULL', 'TYPE_INLINE', 'TYPE_ERROR', 'TYPE_FORMULA_CACHE_STRING', 'TYPE_INTEGER'}

    def __init__(self, cell, parent=None):
        super(TableCell, self).__init__(cell, parent=parent)
        self.value = element_to_string(cell)
        self.number_format = self.get_number_format()

    def data_type(self):
        cell_types = self.CELL_TYPES & set(self.element.get('class', '').split())
        if cell_types:
            if 'TYPE_FORMULA' in cell_types:
                # Make sure TYPE_FORMULA takes precedence over the other classes in the set.
                cell_type = 'TYPE_FORMULA'
            elif cell_types & {'TYPE_CURRENCY', 'TYPE_INTEGER', 'TYPE_PERCENTAGE'}:
                cell_type = 'TYPE_NUMERIC'
            else:
                cell_type = cell_types.pop()
        else:
            cell_type = 'TYPE_STRING'
        return getattr(cell, cell_type)

    def get_number_format(self):
        if 'TYPE_CURRENCY' in self.element.get('class', '').split():
            return FORMAT_CURRENCY_USD_SIMPLE
        if 'TYPE_INTEGER' in self.element.get('class', '').split():
            return '#,##0'
        if 'TYPE_PERCENTAGE' in self.element.get('class', '').split():
            return FORMAT_PERCENTAGE
        if 'TYPE_DATE' in self.element.get('class', '').split():
            return FORMAT_DATE_MMDDYYYY
        if self.data_type() == cell.TYPE_NUMERIC:
            try:
                int(self.value)
            except ValueError:
                return '#,##0.##'
            else:
                return '#,##0'

    def format(self, cell):
        cell.style = self.style()
        data_type = self.data_type()
        if data_type:
            cell.data_type = data_type


____tablepyxl = """
tablepyxl.tablepyxl
"""


def string_to_int(s):
    if s.isdigit():
        return int(s)
    return 0


def get_Tables(doc):
    tree = html.fromstring(doc)
    comments = tree.xpath('//comment()')
    for comment in comments:
        comment.drop_tag()
    return [Table(table) for table in tree.xpath('//table')]


def write_rows(worksheet, elem, row, column=1):
    """
    Writes every tr child element of elem to a row in the worksheet

    returns the next row after all rows are written
    """
    from openpyxl.cell.cell import MergedCell

    initial_column = column
    for table_row in elem.rows:
        for table_cell in table_row.cells:
            cell = worksheet.cell(row=row, column=column)
            while isinstance(cell, MergedCell):
                column += 1
                cell = worksheet.cell(row=row, column=column)

            colspan = string_to_int(table_cell.element.get("colspan", "1"))
            rowspan = string_to_int(table_cell.element.get("rowspan", "1"))
            if rowspan > 1 or colspan > 1:
                worksheet.merge_cells(start_row=row, start_column=column,
                                      end_row=row + rowspan - 1, end_column=column + colspan - 1)

            cell.value = table_cell.value
            table_cell.format(cell)
            min_width = table_cell.get_dimension('min-width')
            max_width = table_cell.get_dimension('max-width')

            if colspan == 1:
                # Initially, when iterating for the first time through the loop, the width of all the cells is None.
                # As we start filling in contents, the initial width of the cell (which can be retrieved by:
                # worksheet.column_dimensions[get_column_letter(column)].width) is equal to the width of the previous
                # cell in the same column (i.e. width of A2 = width of A1)
                width = max(worksheet.column_dimensions[get_column_letter(column)].width or 0, len(table_cell.value) + 2)
                if max_width and width > max_width:
                    width = max_width
                elif min_width and width < min_width:
                    width = min_width
                worksheet.column_dimensions[get_column_letter(column)].width = width
            column += colspan
        row += 1
        column = initial_column
    return row


def table_to_sheet(table, wb):
    """
    Takes a table and workbook and writes the table to a new sheet.
    The sheet title will be the same as the table attribute name.
    """
    ws = wb.create_sheet(title=table.element.get('name'))
    insert_table(table, ws, 1, 1)


def document_to_workbook(doc, wb=None, base_url=None):
    """
    Takes a string representation of an html document and writes one sheet for
    every table in the document.

    The workbook is returned
    """
    if not wb:
        wb = Workbook()
        wb.remove(wb.active)

    # Premailer 是一个第三方库，能把html中css标注的样式，展开到body每个blocks中
    inline_styles_doc = Premailer(doc, base_url=base_url, remove_classes=False).transform()
    # tablepyxl 库作者写的html转Table对象
    tables = get_Tables(inline_styles_doc)

    for table in tables:
        table_to_sheet(table, wb)

    return wb


def document_to_xl(doc, filename, base_url=None):
    """
    Takes a string representation of an html document and writes one sheet for
    every table in the document. The workbook is written out to a file called filename
    """
    wb = document_to_workbook(doc, base_url=base_url)
    wb.save(filename)


def insert_table(table, worksheet, column, row):
    if table.head:
        row = write_rows(worksheet, table.head, row, column)
    if table.body:
        row = write_rows(worksheet, table.body, row, column)


def insert_table_at_cell(table, cell):
    """
    Inserts a table at the location of an openpyxl Cell object.
    """
    ws = cell.parent
    column, row = cell.column, cell.row
    insert_table(table, ws, column, row)


____mycode = """
自己写的扩展
"""


class Openpyxl:
    """
    对openpyxl库做一些基本的功能拓展
    """

    @staticmethod
    def address(n, m) -> str:
        r"""数字索引转excel地址索引
        :param n: 行号，可以输入字符串形式的数字
        :param m: 列号，同上可以输入str的数字
        :return:

        >>> Openpyxl.address(2, 3)
        'C2'
        """
        from openpyxl.utils.cell import get_column_letter
        return f'{get_column_letter(int(m))}{n}'

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
        from openpyxl.cell.cell import MergedCell
        if isinstance(cell, MergedCell):
            ws = cell.parent
            xy = Openpyxl.in_range(cell).top[0]
            return ws[Openpyxl.address(*xy)]
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
        from openpyxl.cell.cell import MergedCell
        if isinstance(cell, MergedCell):
            return 1
        elif isinstance(cell.offset(1, 0), MergedCell) or isinstance(cell.offset(0, 1), MergedCell):
            # 这里只能判断可能是合并单元格，具体是不是合并单元格，还要
            rng = Openpyxl.in_range(cell)
            return 2 if hasattr(rng, 'size') else 0
        else:
            return 0

    @staticmethod
    def isnone(cell):
        """是普通单元格且值为None
        注意合并单元格的衍生单元格不为None
        """
        celltype = Openpyxl.celltype(cell)
        return celltype == 0 and cell.value is None

    @staticmethod
    def copy_cell_format(cell, new_cell):
        """ 单元格全格式复制，需要事先指定好新旧单元格的物理位置
        参考：https://stackoverflow.com/questions/23332259/copy-cell-style-openpyxl
        """
        from copy import copy
        if cell.has_style:
            new_cell.font = copy(cell.font)  # 字体
            new_cell.border = copy(cell.border)  # 表格线
            new_cell.fill = copy(cell.fill)  # 填充色
            new_cell.number_format = copy(cell.number_format)  # 数字格式
            new_cell.protection = copy(cell.protection)  # 保护？
            new_cell.alignment = copy(cell.alignment)  # 对齐格式
            # new_cell.style = cell.style
        # if cell.comment:
            # 这个会引发AttributeError。。。
            #       vml = fromstring(self.workbook.vba_archive.read(ws.legacy_drawing))
            #   AttributeError: 'NoneType' object has no attribute 'read'
            # new_cell.comment = copy(cell.comment)
            # 就算开了keep_vba可以强制写入了，打开的时候文件可能还是会错

    @staticmethod
    def copy_cell(cell, new_cell):
        """ 单元格全格式、包括值的整体复制
        """
        new_cell.value = cell.value
        Openpyxl.copy_cell_format(cell, new_cell)

    @classmethod
    def down(cls, cell):
        """输入一个单元格，向下移动一格
        注意其跟offset的区别，如果cell是合并单元格，会跳过自身的衍生单元格
        """
        if cls.celltype(cell):  # 合并单元格
            rng = cls.in_range(cell)
            return cell.parent.cell(rng.max_row+1, cell.column)
        else:
            return cell.offset(1, 0)

    @classmethod
    def right(cls, cell):
        if cls.celltype(cell):
            rng = cls.in_range(cell)
            return cell.parent.cell(cell.row, rng.max_row+1)
        else:
            return cell.offset(0, 1)

    @classmethod
    def up(cls, cell):
        if cls.celltype(cell):
            rng = cls.in_range(cell)
            return cell.parent.cell(rng.min_row-1, cell.column)
        else:
            return cell.offset(-1, 0)

    @classmethod
    def left(cls, cell):
        if cls.celltype(cell):
            rng = cls.in_range(cell)
            return cell.parent.cell(cell.row, rng.min_row-1)
        else:
            return cell.offset(0, -1)

    @staticmethod
    def copy_worksheet(origin_ws, target_ws):
        """跨工作薄时复制表格内容的功能
        openpyxl自带的Workbook.copy_worksheet没法跨工作薄复制，很坑
        """
        # 1、取每个单元格的值
        for row in origin_ws:
            for cell in row:
                try:
                    Openpyxl.copy_cell(cell, target_ws[cell.coordinate])
                except AttributeError:
                    pass
        # 2、合并单元格的处理
        for rng in origin_ws.merged_cells.ranges:
            target_ws.merge_cells(rng.ref)
        # 3、其他表格属性的复制
        # 这个从excel读取过来的时候，是不准的，例如D3可能因为关闭时停留窗口的原因误跑到D103
        # dprint(origin_ws.freeze_panes)
        # target_ws.freeze_panes = origin_ws.freeze_panes


def product(*iterables, order=None, repeat=1):
    """ 对 itertools 的product扩展orders参数的更高级的product迭代器
    :param order: 假设iterables有n=3个迭代器，则默认 orders=[1, 2, 3] （起始编号1）
        即标准的product，是按顺序对每个迭代器进行重置、遍历的
        但是我扩展的这个接口，允许调整每个维度的更新顺序
        例如设置为 [-2, 1, 3]，表示先对第2维降序，然后按第1、3维的方式排序获得各个坐标点
        注：可以只输入[-2]，默认会自动补充维[1, 3]

    for x in product('ab', 'cd', 'ef', order=[3, -2, 1]):
        print(x)

    ['a', 'd', 'e']
    ['b', 'd', 'e']
    ['a', 'c', 'e']
    ['b', 'c', 'e']
    ['a', 'd', 'f']
    ['b', 'd', 'f']
    ['a', 'c', 'f']
    ['b', 'c', 'f']

    TODO 我在想numpy这么牛逼，会不会有等价的功能接口可以实现，我不用重复造轮子？
    """
    import itertools, numpy

    # 一、标准调用方式
    if order is None:
        for x in itertools.product(*iterables, repeat=repeat):
            yield x
        return

    # 二、输入orders参数的调用方式
    # 1、补全orders参数长度
    n = len(iterables)
    for i in range(1, n + 1):
        if not (i in order or -i in order):
            order.append(i)
    if len(order) != n: return ValueError(f'orders参数值有问题 {order}')

    # 2、生成新的迭代器组
    new_iterables = [(iterables[i-1] if i > 0 else reversed(iterables[-i-1])) for i in order]
    idx = numpy.argsort([abs(i) - 1 for i in order])
    for y in itertools.product(*new_iterables, repeat=repeat):
        yield [y[i] for i in idx]


class Worksheet(openpyxl.worksheet.worksheet.Worksheet):
    """ 扩展标准的Workshhet功能
    >> wb = openpyxl.load_workbook(filename='高中数学知识树匹配终稿.xlsx', data_only=True)
    >> ws1 = Worksheet(wb['main'])
    >> ws2 = Worksheet(wb['导出'])
    """

    def __init__(self, ws):
        self.__dict__ = ws.__dict__

    def _cells_by_row(self, min_col, min_row, max_col, max_row, values_only=False):
        """openpyxl的这个迭代器，遇到合并单元格会有bug
        所以我把它重新设计一下~~
        """
        for row in range(min_row, max_row + 1):
            cells = (self.cell(row=row, column=column) for column in range(min_col, max_col + 1))
            if values_only:
                # yield tuple(cell.value for cell in cells)  # 原代码
                yield tuple(getattr(cell, 'value', None) for cell in cells)
            else:
                yield tuple(cells)

    def search(self, pattern, min_row=None, max_row=None, min_col=None, max_col=None, order=None, direction=0):
        """查找满足pattern正则表达式的单元格

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
        # 1、定界
        x1, x2 = max(min_row or 1, 1), min(max_row or self.max_row, self.max_row)
        y1, y2 = max(min_col or 1, 1), min(max_col or self.max_column, self.max_column)

        # 2、遍历
        if isinstance(pattern, (list, tuple)):
            cel = None
            for p in pattern:
                cel = self.search(p, x1, x2, y1, y2, order)
                if cel:
                    # up, down, left, right 找到的单元格四边界
                    l, u, r, d = getattr(Openpyxl.in_range(cel), 'bounds', (cel.column, cel.row, cel.column, cel.row))
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
                cell = self.cell(x, y)
                if Openpyxl.celltype(cell) == 1: continue  # 过滤掉合并单元格位置
                if pattern.search(str(cell.value)): return cell  # 返回满足条件的第一个值

    findcel = search

    def findrow(self, pattern, *args, **kwargs):
        cel = self.findcel(pattern, *args, **kwargs)
        return cel.row if cel else 0

    def findcol(self, pattern, *args, **kwargs):
        cel = self.findcel(pattern, *args, **kwargs)
        return cel.column if cel else 0

    def chrome(self):
        """注意，这里会去除掉合并单元格"""
        chrome(pd.DataFrame(self.values))

    def select_columns(self, columns, column_name='searchkey'):
        r"""获取表中columns属性列的值，返回dataframe数据类型

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

        # 1、找到所有标题位置，定位起始行
        cels, names, start_line = [], [], -1
        for search_name in columns:
            cel = self.findcel(search_name)
            if cel:
                cels.append(cel)
                if column_name=='searchkey':
                    names.append(str(search_name))
                elif column_name=='origin':
                    if isinstance(search_name, (list, tuple)) and len(search_name) > 1:
                        names.append('/'.join(list(search_name[:-1]) + [str(cel.value)]))
                    else:
                        names.append(str(cel.value))
                else:
                    raise ValueError(f'{column_name}')
                start_line = max(start_line, Openpyxl.down(cel).row)
            else:
                dprint(search_name)  # 找不到指定列

        # 2、获得每列的数据
        datas = {}
        for k, cel in enumerate(cels):
            if cel:
                col = cel.column
                li = []
                for i in range(start_line, self.max_row+1):
                    v = Openpyxl.mcell(self.cell(i, col)).value  # 注意合并单元格的取值
                    li.append(v)
                datas[names[k]] = li
            else:
                # 如果没找到列，设一个空列
                datas[names[k]] = [None]*(self.max_row + 1 - start_line)
        df = pd.DataFrame(datas)

        # 3、去除所有空行数据
        df.dropna(how='all', inplace=True)

        return df

    def copy_range(self, cell_range, rows=0, cols=0):
        """ 同表格内的 range 复制操作
        Copy a cell range by the number of rows and/or columns:
        down if rows > 0 and up if rows < 0
        right if cols > 0 and left if cols < 0
        Existing cells will be overwritten.
        Formulae and references will not be updated.
        """
        from openpyxl.worksheet.cell_range import CellRange
        from itertools import product
        # 1、预处理
        if isinstance(cell_range, str):
            cell_range = CellRange(cell_range)
        if not isinstance(cell_range, CellRange):
            raise ValueError("Only CellRange objects can be copied")
        if not rows and not cols:
            return
        min_col, min_row, max_col, max_row = cell_range.bounds
        # 2、注意拷贝顺序跟移动方向是有关系的，要防止被误覆盖，复制了新的值，而非原始值
        r = sorted(range(min_row, max_row+1), reverse=rows > 0)
        c = sorted(range(min_col, max_col+1), reverse=cols > 0)
        for row, column in product(r, c):
            Openpyxl.copy_cell(self.cell(row, column), self.cell(row + rows, column+cols))

    def reindex_columns(self, orders):
        """ 重新排列表格的列顺序
        >> ws.reindex_columns('I,J,A,,,G,B,C,D,F,E,H,,,K'.split(','))

        TODO 支持含合并单元格的整体移动？
        """
        from openpyxl.utils.cell import column_index_from_string
        max_row, max_column = self.max_row, self.max_column
        for j, col in enumerate(orders, 1):
            if not col: continue
            self.copy_range(f'{col}1:{col}{max_row}', cols=max_column+j-column_index_from_string(col))
        self.delete_cols(1, max_column)


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


def demo_openpyxl():
    # 一、新建一个工作薄
    from openpyxl import Workbook
    wb = Workbook()

    # 取一个工作表
    ws = wb.active  # wb['Sheet']，取已知名称、下标的表格，excel不区分大小写，这里索引区分大小写

    # 1、索引单元格的两种方法，及可以用.value获取值
    ws['A2'] = '123'
    dprint(ws.cell(2, 1).value)  # 123

    # 2、合并单元格
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
    dprint(Openpyxl.down(ws['A1']).coordinate)  # A3

    # 3、设置单元格样式、格式
    from openpyxl.comments import Comment
    cell = ws['A3']
    cell.font = Font(name='Courier', size=36)
    cell.comment = Comment(text="A comment", author="Author's Name")
    from openpyxl.styles.colors import RED

    styles = [['Number formats', 'Comma', 'Comma [0]', 'Currency', 'Currency [0]', 'Percent'],
              ['Informative', 'Calculation', 'Total', 'Note', 'Warning Text', 'Explanatory Text'],
              ['Text styles', 'Title', 'Headline 1', 'Headline 2', 'Headline 3', 'Headline 4', 'Hyperlink', 'Followed Hyperlink', 'Linked Cell'],
              ['Comparisons', 'Input', 'Output', 'Check Cell', 'Good', 'Bad', 'Neutral'],
              ['Highlights', 'Accent1', '20 % - Accent1', '40 % - Accent1', '60 % - Accent1', 'Accent2', 'Accent3', 'Accent4', 'Accent5', 'Accent6', 'Pandas']]
    for i, name in enumerate(styles, start=4):
        ws.cell(i, 1, name[0])
        for j, v in enumerate(name[1:], start=2):
            ws.cell(i, j, v)
            ws.cell(i, j).style = v

    # 二、测试一些功能
    ws = Worksheet(ws)

    dprint(ws.search('模块二').coordinate)  # D1
    dprint(ws.search(['模块二', '属性1']).coordinate)  # D3

    dprint(ws.findcol(['模块一', '属性1'], direction=1))  # 0

    wb.save("demo_openpyxl.xlsx")
