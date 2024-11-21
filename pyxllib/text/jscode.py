#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽，梁奕本（js去注释部分）
# @Email  : 877362867@qq.com, https://lyeebn.gitee.io/technology-shop/HeyBoss.html
# @Date   : 2023/10/20

from collections import Counter
import re
import textwrap
import os

from jinja2 import Template

try:
    import jsbeautifier
except ModuleNotFoundError:
    pass

from pyxllib.file.specialist import XlPath
from pyxllib.prog.cachetools import xlcache


def __1_删注释功能():
    """
    用编译语法解析方式分析，清理JS中的注释，支持嵌套

    Usage:
    1、A simple function
    from pyxllib.text.jscode import dropJScomment
    dropJScomment(jsSourceCodeAsString)

    2、Object
    from pyxllib.text.jscode import JSParser
    js = JSParser(jsSourceCode)
    jsc = js.clearComment()
    """


def 删注释周围留空(c, arr):  # 应急
    arr.reverse()
    for i in arr:
        # 先用这个快速处理，有空再优化： https://blog.csdn.net/cooco369/article/details/82994932
        # c = c[:i].rstrip() + '%' + c[i:].lstrip()  # debug，定位
        有回车 = False
        l = r = 上一回车处 = i
        r = i + 1
        len_c = len(c)
        while len_c > l > 0:
            l -= 1
            ci = c[l]
            if ci == '\n':  # \r已统一为\n
                有回车 = True
            elif ci not in '\t \v':  # TODO 中文空格行不行
                break
        while r < len_c:
            ci = c[r]
            if ci in '\n\r':  # \r已统一为\n
                有回车 = True
                上一回车处 = r  # 保持缩进，但有 Bug 灵异，难道是 ?
            elif ci not in '\t \v':  # TODO 中文空格行不行
                break
            r += 1
            # print(r)
        c = c[:l + 1] + ('\n' if 有回车 else '') + c[上一回车处:]  # 有必要多留个空格吗
        # 已知 BUG：连续注释后的缩进无法保持，但不影响 JS 代码逻辑就是了
    return c


def 回溯区分正则除号(c):
    # 此函数参数需要提前处理：高偶合，非内聚
    # 原理：除号是二元运算符，其前面必有一个量：数值（可以是变量名或字符串字面量），这不太好穷举
    # 而正则为一字面量，前面可能必为某种运算符 = + ，或特殊符号：& | 逻辑? 括号 ( , 参数，[ { 对象 : ，或 ; 语句结束符，回车。有个坑：折行要注意。
    # 摆脱对正则的依赖，变成无第三无依赖
    # True 为 正则， False 为除号
    i = len(c)
    while i > 0:
        i -= 1
        ci = c[i]
        if ci in '\t \v':
            continue  # 暂时无法给出结论
        elif ci in '\n\r':  # 折行，暂时无法给出结论，其实可以与如上合并，区别可能是回车前可能是已省略的分号
            # return 回溯区分正则除号(c, i=i)  # 经验证，js 中 转义回车仅限于字符串内，运算符后是可折行的， \ 反而语法错误
            continue  # 经考虑，还是不必递归了
        elif ci in '{[(=,;?:&|!*~%^':  # todo ++ --  ，注：~ 为非预期操作，但因类型转换语法允许 https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Guide/Expressions_and_operators
            return True  # 正则
        elif ci == '+':  # todo ++ --
            i -= 1
            if i > 0 and c[i] == '+': return False  # 除号 # ++： / 前 为变量 为除尘
            return True  # 正则
        elif ci == '-':  # 要么  变量--，要反报错
            i -= 1
            if i > 0 and c[i] == '-': return False  # 除号 # ++： / 前 为变量 为除尘
            return True  # raise BaseException('/ 前单 -')  # 实测因类型转换，两边转数值类型，最多是 NaN，而不至于报错
        else:
            return False  # 除号，可能是变量，什么的
    return True  # 正则


class JSParser:
    def 普通引号(self, 号='"'):  # 号：开始结束定界符，下同
        while self.indexPointer < self.jsCodeLength:  # 正常会提前 return 见下
            self.indexPointer += 1
            si = self.jsSourceCode[self.indexPointer]
            self.jsWithoutComment += si  # 这应该是以下所有情况包括 else 都要的
            if si == 号:
                return
            elif si == '\\':  # 转义，提前吃掉， \n'"` 等
                # 即  r += '\\'
                self.indexPointer += 1
                if self.jsSourceCode[
                    self.indexPointer + 1] == 号:  # 超标报错正好，对应其语法错误，TODO 以后完善，实测 JS，  如果在引号中回车前没\反是其语法错误；就当源文件语法吧
                    pass
                self.jsWithoutComment += self.jsSourceCode[self.indexPointer]  # 转义后的字符好像都是要吃掉的，不用 if，待深入思考
            elif si == '\n':
                raise BaseException('原稿语法有误》缺右字符串定界：\a' + 号)

    def 反引号(self, 号='`'):
        while self.indexPointer < self.jsCodeLength:  # 正常会提前 return 见下
            self.indexPointer += 1
            si = self.jsSourceCode[self.indexPointer]
            self.jsWithoutComment += si  # 这应该是以下所有情况包括 else 都要的
            if si == 号:
                return
            elif si == '\\':
                # 即  r += '\\'
                self.indexPointer += 1
                if self.jsSourceCode[
                    self.indexPointer + 1] == 号:  # 超标报错正好，对应其语法错误，TODO 以后完善，实测 JS，  如果在引号中回车前没\反是其语法错误；就当源文件语法吧
                    pass
                self.jsWithoutComment += self.jsSourceCode[self.indexPointer]  # 转义后的字符好像都是要吃掉的，不用 if，待深入思考
            elif si == '$' and self.jsSourceCode[
                self.indexPointer + 1] == '{':  # 重要区别 。经实验 ${  后必有 } 否则报错：g = `2${2+3 7`
                self.indexPointer += 1
                self.jsWithoutComment += '{'
                self.反引号嵌套表达式允许多行注释()
            pass

    def 反引号嵌套表达式允许多行注释(self):  # 适合钻牛角尖，  TODO 用 堆栈 结构练手？  这好像可以 递归 main 了，差个 } return 吧
        while self.indexPointer < self.jsCodeLength:  # 正常会提前 return 见下
            self.indexPointer += 1
            si = self.jsSourceCode[self.indexPointer]
            if si == '}':
                self.jsWithoutComment += si
                return
            elif si in '"\'':  # 吃撑了的嵌套骚操作  si == '"' or si == "'"
                self.jsWithoutComment += si
                self.普通引号(号=si)
            elif si == "`":  # `，还能嵌套 String.raw` 算了，不玩了
                self.jsWithoutComment += si
                self.反引号()
            elif si == '/':  # 以下这一块逻辑 同 main ？
                self.反斜线然后呢()
                # if s[idx+1] == '*':  # 注意这里，嵌套了注释！超标正好报错，有效的 JS 代码 这里不会超标
                #     idx += 1
                #     多行注释()
                # elif s[idx+1] == '/':
                #     idx += 1
                #     单行注释()
                # else:
                #     r += s[idx]
            else:
                self.jsWithoutComment += si

    def 反斜线然后呢(self):  # 共用于【main】与 【反引号嵌套表达式允许多行注释】中的表达式
        偷看 = self.jsSourceCode[self.indexPointer + 1]
        if 偷看 == '/':
            self.单行注释()
            self.jsWithoutComment += self.注释占位
        elif 偷看 == '*':
            self.jsWithoutComment += self.注释占位
            self.indexPointer += 1
            self.多行注释()
        elif 回溯区分正则除号(
                self.jsWithoutComment):  # bool(re.search('[\n\r(\[\{=+,;&?|]([ \t]*|\\[\n\r])*[ \t]*$', self.jsWithoutComment)):  # def 区分正则或除号(): ←
            self.jsWithoutComment += '/'
            self.正则()
        else:
            self.jsWithoutComment += '/'
            # 除号不必特殊处理吧()
        # TODO，重要【严重】有没有可能是除号：q = 1 / 2 + /2/ // 除号、正则 、注释并存
        # 如果是除号，那之前应该有数值  \d)，不好判断，考虑到 JS 有些隐性的类型转换，如：'6' / 3
        # 如果是正则，往前回溯应该必有 = 或 + ，也可能在函数作为参数 （甚至可能还有变态 通过 \ 加回车，在正则前折行） re 如下

    def 源反引号(self, 号='`'):
        return self.反引号(号=号)  # TODO 先借用，也就差个 \ ，其它有什么区别待考虑
        pass

    def 单行注释(self):
        # while (s[idx] != '\n' or s[idx] != '\r') and idx < 长 :
        while (self.jsSourceCode[self.indexPointer] not in '\n\r') and self.indexPointer < self.jsCodeLength:
            self.indexPointer += 1
        self.注释location.append(len(self.jsWithoutComment))
        self.jsWithoutComment += '\n'  # 补回车于删注释点后，不能反了

    def 多行注释(self):
        self.indexPointer += 1  # 要吗？加过没
        while not (self.jsSourceCode[self.indexPointer] == '*' and self.jsSourceCode[
            self.indexPointer + 1] == '/'): self.indexPointer += 1
        self.indexPointer += 1
        # r += '\n'
        self.注释location.append(len(self.jsWithoutComment))

    def 正则(self):  # 正常的 JS 代码中 正则中无回车符
        while True:  # not (jsSourceCode[self.indexPointer + 1] == '/')
            self.indexPointer += 1  # TODO 放哪里
            si = self.jsSourceCode[self.indexPointer]
            self.jsWithoutComment += si
            if si == '/':
                return
            elif si in '\n\r':
                raise BaseException('正则还能折行？你是哪个老师教的')
            elif self.jsSourceCode[self.indexPointer] == '\\':
                self.indexPointer += 1  # TODO 放哪里
                self.jsWithoutComment += self.jsSourceCode[self.indexPointer]

    def __init__(self, jsSourceCode):
        # 就传源码吧，文件打开让用户自己去做，不然还得判断是文件名还是字符串，还得判断文件是否存在，还得依赖 os 包
        self.jsSourceCode = jsSourceCode.replace('\r', '\n') + '\n            '  # 防超标，为啥 .strip() 会异常
        self.jsCodeLength = len(jsSourceCode)
        self.indexPointer = -1  # 游标指针
        self.注释占位 = ''  # 生僻占位？以便后期删其前 \s
        self.注释location = []  # 记录当时 len(self.jsWithoutComment)，改天写吧
        self.jsWithoutComment = False  # init 先执行，所以不能执行调此类的其它函数，全写到一个这个函数里也麻烦，缩进了两级，如果要用方法可能重复执行，故用缓存法，

    def clearComment(self):  # 难道这个函数要在放类外吗（以便 init 能调用）
        if self.jsWithoutComment: return self.jsWithoutComment  # 已经求值过就用缓存，避免如下代码重复执行。这里一个变量两用：初始 bool 类型 False ，之后存清注释的str结果，类型改变，如果要让 GPT 改为 C++ 可能要注意一下，除了这里，别的地文都没有动态类型
        # 仅第一次计算，
        self.jsWithoutComment = ''
        while self.indexPointer < self.jsCodeLength:  # 要注意在哪里加 1 ，统一在处理开头处加吧，让当前处理的指标与相应字符初始一致，特殊情况再特殊加
            self.indexPointer += 1
            si = self.jsSourceCode[self.indexPointer]
            if si == '/':  # 正则、单/多行注释、除号，idea：注释前可能有多余的 \s，删注释时，可以留个 unicode 记号，到时用正则删
                self.反斜线然后呢()
                continue  # 其后的情况都要 r += si， TODO 合并
            elif si in '"\'':  # '  "        →1 si == '"' or si == "'"
                self.jsWithoutComment += si
                self.普通引号(号=si)
            elif si == "`":  # `
                self.jsWithoutComment += si
                self.反引号()
            elif si == "S" and self.jsCodeLength - self.indexPointer > 11 and self.jsSourceCode[
                                                                              self.indexPointer:self.indexPointer + 11] == 'String.raw`':  # String.raw` 元字符串
                self.jsWithoutComment += si
                self.indexPointer += 10;
                self.jsWithoutComment += 'tring.raw`'  # 分两步，以便上面那个能和其它情景合并
                self.源反引号()
            elif si == "\n":  # 压缩连续回车（空行）寻找一回车位置，这可能出现 BUG 吗？慎重，危险，不能出现于字符串，注释等中
                tmp = self.indexPointer
                while tmp < self.jsCodeLength:
                    tmp += 1
                    if self.jsSourceCode[tmp] == '\n':
                        self.indexPointer = tmp
                    elif self.jsSourceCode[tmp] in '\t ':
                        break
                    else:
                        break
                self.jsWithoutComment += si
            else:
                self.jsWithoutComment += si
            # 格式化字符串好像没什么特殊的
            pass
        self.jsWithoutComment = 删注释周围留空(self.jsWithoutComment, self.注释location).strip()
        # print(self.注释location)
        return self.jsWithoutComment


def remove_js_comments(jsSourceCode):  # 对外接口，将本来用得两行代码封装为一行
    js = JSParser(jsSourceCode)
    return js.clearComment()


def __2_类js的as处理功能():
    pass


airscript_head = r"""
// 0 基础组件代码（可以放在功能代码之前，也能放在最后面）

// 根据提供的 pattern 在 range 中寻找 cell
// 如果没有提供 range，默认在 ActiveSheet.UsedRange 中寻找
function findCell(pattern, range = ActiveSheet.UsedRange) {
    const cell = range.Find(pattern, range, xlValues, xlWhole)
    return cell
}

function levenshteinDistance(a, b) {
    const matrix = [];

    let i;
    for (i = 0; i <= b.length; i++) {
        matrix[i] = [i];
    }

    let j;
    for (j = 0; j <= a.length; j++) {
        matrix[0][j] = j;
    }

    for (i = 1; i <= b.length; i++) {
        for (j = 1; j <= a.length; j++) {
            if (b.charAt(i - 1) === a.charAt(j - 1)) {
                matrix[i][j] = matrix[i - 1][j - 1];
            } else {
                matrix[i][j] = Math.min(matrix[i - 1][j - 1] + 1, Math.min(matrix[i][j - 1] + 1, matrix[i - 1][j] + 1));
            }
        }
    }

    return matrix[b.length][a.length];
}

// 根据提供的 pattern 在 range 中寻找 column
// 如果没有提供 range，默认在 ActiveSheet.UsedRange 中寻找
function findColumn(pattern, range = ActiveSheet.UsedRange) {
    let cell = findCell(pattern, range);  // 首先尝试精确匹配
    if (!cell) {  // 如果精确匹配失败，尝试模糊匹配
        let minDistance = Infinity;
        let minDistanceColumn;
        for (let i = 1; i <= range.Columns.Count; i++) {
            let columnName = range.Cells(1, i).Value;
            let distance = levenshteinDistance(pattern, columnName);
            if (distance < minDistance) {
                minDistance = distance;
                minDistanceColumn = i;
            }
        }
        return minDistanceColumn;
    }
    if (cell) { return cell.Column }
}

// 根据提供的 pattern 在 range 中寻找 row
// 如果没有提供 range，默认在 ActiveSheet.UsedRange 中寻找
function findRow(pattern, range = ActiveSheet.UsedRange) {
    const cell = findCell(pattern, range)
    if (cell) { return cell.Row }
}

// 判断一个 cells 集合是否为空
function isEmpty(cells) {
    for (let i = 1; i <= cells.Count; i++) {
        if (cells.Item(i).Text) {
            return false;
        }
    }
    return true;
}

// 获取实际使用的区域
function getUsedRange(maxRows = 500, maxColumns = 100, startFromA1 = true) {
    /* 允许通过"表格上下文"信息，调整这里数据行的上限500行，或者列上限100列
        注意，如果分析预设的表格数据在这个限定参数内可以不改
        只有表格未知，或者明确数据量超过设置时，需要重新调整这里的参数
        调整的时候千万不要故意凑的刚刚好，可以设置一定的冗余区间
        比如数据说有4101条，那么这里阈值设置为5000也是可以的，比较保险。
    */

    // 默认获得的区间，有可能是有冗余的空行，所以还要进一步优化
    let usedRange = ActiveSheet.UsedRange;

    let lastRow = Math.min(usedRange.Rows.Count, maxRows);
    let lastColumn = Math.min(usedRange.Columns.Count, maxColumns);

    let firstRow = 1;
    let firstColumn = 1;

    // 找到最后一个非空行
    for (; lastRow >= firstRow; lastRow--) {
        if (!isEmpty(usedRange.Rows(lastRow).Cells)) {
            break;
        }
    }

    // 找到最后一个非空列
    for (; lastColumn >= firstColumn; lastColumn--) {
        if (!isEmpty(usedRange.Columns(lastColumn).Cells)) {
            break;
        }
    }

    // 如果表格不是从"A1"开始，找到第一个非空行和非空列
    if (!startFromA1) {
        for (; firstRow <= lastRow; firstRow++) {
            if (!isEmpty(usedRange.Rows(firstRow).Cells)) {
                break;
            }
        }

        for (; firstColumn <= lastColumn; firstColumn++) {
            if (!isEmpty(usedRange.Columns(firstColumn).Cells)) {
                break;
            }
        }
    }

    // 创建一个新的 Range 对象，它只包含非空的行和列
    let newUsedRange = ActiveSheet.Range(
        usedRange.Cells(firstRow, firstColumn),
        usedRange.Cells(lastRow, lastColumn)
    );

    return newUsedRange;  // 返回新的实际数据区域
}

// 将 Excel 日期转换为 JavaScript 日期
function xlDateToJSDate(xlDate) {
    return new Date((xlDate - 25569) * 24 * 3600 * 1000);
}

// 判断日期是否在本周
function isCurrentWeek(date) {
    const today = new Date();
    today.setHours(0, 0, 0, 0);  // 把时间设为午夜以准确地比较日期
    const firstDayOfWeek = new Date(today.setDate(today.getDate() - today.getDay()));
    const lastDayOfWeek = new Date(today.setDate(today.getDate() - today.getDay() + 6));
    return date >= firstDayOfWeek && date <= lastDayOfWeek;
}

// 判断日期是否在当前月份
function isCurrentMonth(date) {
    const currentDate = new Date();
    currentDate.setHours(0, 0, 0, 0);  // 把时间设为午夜stdcode以准确地比较日期
    return date.getMonth() === currentDate.getMonth() && date.getFullYear() === currentDate.getFullYear();
}

// 判断日期是否在下周
function isNextWeek(date) {
  const today = new Date();
  today.setHours(0, 0, 0, 0);  // 把时间设为午夜以准确地比较日期
  const nextWeek = new Date(today.getFullYear(), today.getMonth(), today.getDate() + 7);
  return date > today && date <= nextWeek;
}

// 判断日期是否在下个月
function isNextMonth(date) {
    const today = new Date();
    today.setHours(0, 0, 0, 0);  // 把时间设为午夜以准确地比较日期
    const nextMonth = new Date(today.getFullYear(), today.getMonth() + 1, 1);
    const endDateOfNextMonth = new Date(today.getFullYear(), today.getMonth() + 2, 0);
    return date >= nextMonth && date <= endDateOfNextMonth;
}
""".strip()


@xlcache()
def get_airscript_head2(definitions=False):
    s = (XlPath(__file__).parent / 'airscript.js').read_text().strip()
    vars = {
        'JSA_POST_HOST_URL': os.getenv('JSA_POST_HOST_URL', 'https://xmutpriu.com'),
        'JSA_POST_TOKEN': os.getenv('JSA_POST_TOKEN', ''),
        'JSA_POST_DEFAULT_HOST': os.getenv('JSA_POST_DEFAULT_HOST', 'senseserver3'),
    }
    content = Template(s).render(vars)
    if not definitions:
        return content
    return extract_definitions_with_comments(content + '\n')


class AirScriptCodeFixer:
    @classmethod
    def fix_colors(cls, code_text):
        # 1 一些错误的颜色设置方法
        if re.search(r'(?<!\.)\b(Color.\w+)\b', code_text):
            return 0, code_text

        # 2 不能像vba那样，直接对颜色设置一个数值
        match = re.search(r'\.Color\s*=\s*(\d+)', code_text)
        if match:
            color_number = int(match.group(1))
            red = color_number % 256
            green = (color_number // 256) % 256
            blue = (color_number // 256 // 256) % 256
            rgb_format = f'RGB({red}, {green}, {blue})'
            code_text = code_text[:match.start(1)] + rgb_format + code_text[match.end(1):]

        # 3 一些错误的颜色设置方法，进行修正
        configs = {
            '红色': 'RGB(255, 0, 0)',
            '黄色': 'RGB(255, 255, 0)',
            '绿色': 'RGB(0, 255, 0)',
            '蓝色': 'RGB(0, 0, 255)',
            '灰色': 'RGB(128, 128, 128)',
            'red': 'RGB(255, 0, 0)',
            'yellow': 'RGB(255, 255, 0)',
            'green': 'RGB(0, 255, 0)',
            'blue': 'RGB(0, 0, 255)',
            'black': 'RGB(0, 0, 0)',
            'gray': 'RGB(128, 128, 128)',
            'grey': 'RGB(128, 128, 128)',
            'purple': 'RGB(128, 0, 128)',
            'pink': 'RGB(255, 192, 203)',
            'orange': 'RGB(255, 128, 0)',
        }

        def replace_color_fmt(m):
            t1, t2 = m.groups()
            t2 = t2.strip('"\'').lower()
            if t2 in configs:
                return f'{t1}{configs[t2]}'
            elif m2 := re.search(r'[a-fA-F0-9]{6}', t2):
                res = f'{t1}RGB({int(m2.group(0)[:2], 16)}, ' \
                      f'{int(m2.group(0)[2:4], 16)}, ' \
                      f'{int(m2.group(0)[4:], 16)})'
                return res
            return t1 + m.group(2)

        text = re.sub(r'''(\bColor\s*=\s*)(['"].+?['"])''', replace_color_fmt, code_text)

        # 4 经过优化仍无法修正的颜色问题
        if re.search(r'''\bColor\s*=\s*['"]''', text):
            # global count_target
            # ms = re.findall(r'''\bColor\s*=\s*(['"].+)''', text)
            # for m in ms:
            #     count_target[m] += 1
            return 0, text

        return 1, text

    @classmethod
    def fix_miscellaneous(cls, code_text):
        """ 修复其他各种杂项问题 """
        text = code_text

        # Cannot convert a Symbol value to a string, 一般是对Excel对象使用'+='运算报错
        text = re.sub(r'(\s+)((?:.+)Value2?)\s+(?:\+=)\s+(.+)', r'\1\2 = \2 + \3', text)  # 531条

        # 各种错误的接口调用形式
        text = text.replace('.Range.Find(', '.Find(')  # 8条

        # sort接口问题
        text = re.sub(r'(\.Sort\(.*?,\s+)(-1|0|false)\)', r'\g<1>2)', text)  # 328条

        # 做数据有效性的时候，有时候会有重复的引号嵌套
        text = re.sub(r'''(Formula\d:\s*')"(.+?)"''', r'\1\2', text)

        # 230907周四19:56，枚举值不用放在字符串中
        text = re.sub(r'''(['"`])(xlCellTypeVisible)\1''', r'\2', text)

        # 231106周一18:42，range的使用规范性
        text = re.sub(r'Range\(("|\')([A-Z]+|\d+)("|\')\)', r'Range(\1\2:\2\1)', text)

        return 1, text

    @classmethod
    def delete_error_record(cls, code_text):
        return 1, code_text

    @classmethod
    def check_assistant_content(cls, code_text):
        text = code_text

        global count_target
        pieces = re.findall(r'[a-zA-Z_\d\.]+\.Columns', text)
        count_target += Counter([x.strip() for x in pieces])

        # Columns前一般用ActiveSheet就行了

        return 1, text

    @classmethod
    def simplify_advtools(cls, code_text):
        """ 移除高级工具函数代码，用其他更简洁的方式取代 """
        text = code_text
        text = text.replace('getUsedRange()', 'ActiveSheet.UsedRange')
        text = re.sub(r'''findCell\(((['"]).+?\2)(, [a-zA-Z]+)?\)''',
                      r'ActiveSheet.UsedRange.Find(\1)', text)
        text = re.sub(r'''findColumn\(((['"]).+?\2)(, [a-zA-Z]+)?\)''',
                      r'ActiveSheet.UsedRange.Find(\1).Column', text)
        text = re.sub(r'''findRow\(((['"]).+?\2(, [a-zA-Z]+)?)\)''',
                      r'ActiveSheet.UsedRange.Find(\1).Row', text)

        return 1, text

    @classmethod
    def simplify_code(cls, code_text, indent=4):
        """ 代码简化，去掉一些冗余写法

        包括代码美化，默认缩进是4，但在训练阶段，建议默认缩进是2，
        """
        # 1 代码精简
        code_text = re.sub(r'Application\.(WorksheetFunction|ActiveWorkbook|ActiveSheet|Sheets|Range|Workbook)', r'\1',
                           code_text)
        code_text = re.sub(r'Workbook\.(Sheets)', r'\1', code_text)
        code_text = re.sub(r'ActiveSheet\.(Range|Rows|Columns|Cells)', r'\1', code_text)
        code_text = re.sub(r'(\w+)\.(Row|Column)\s*\+\s*\1\.\2s\.Count\s*-\s*1', r'\1.\2End', code_text)
        code_text = re.sub(r'\bvar\b', 'let', code_text)
        code_text = code_text.replace('Sheets.Item(', 'Sheets(')
        code_text = re.sub(r'Application.Enum.\w+.(\w+)', r'\1', code_text)

        # 2 代码美化
        opts = jsbeautifier.default_options()
        opts.indent_size = indent
        code_text = jsbeautifier.beautify(code_text, opts)

        return 1, code_text.strip()

    @classmethod
    def simplify_code2(cls, code_text, indent=4):
        """ 有些规则可能在标注数据中想留着，但训练的时候想删除，则可以调用这个进一步级别的简化 """
        _, code_text = cls.simplify_code(code_text, indent)
        return code_text

    @classmethod
    def fix_stdcode(cls, code_text):
        """ 更智能的，缺什么组件才补什么组件 """
        # 1 检查依赖补充
        text = code_text
        _, text = cls.simplify_advtools(text)

        defined_vars = set(re.findall(r'(?:<=^|\b)(?:var|let|const|function)\s+(\w+)(?:\s+|\()', text))
        used_vars = set(re.findall(r'(?<!\.)\b(\w+)\b', text))

        # 2 提取js中的函数
        def extract_functions(code_string):
            pattern = r"(function\s+(\w+).+?^\})"
            matches = re.findall(pattern, code_string, re.MULTILINE | re.DOTALL)
            return {name: func for func, name in matches}

        js_funcs = extract_functions(airscript_head)

        # 3 补充缺失的定义
        pre_additional_code = []
        for name, code in {'xlDateToJSDate': '',
                           'isCurrentWeek': '',
                           'isCurrentMonth': '',
                           'isNextWeek': '',
                           'isNextMonth': '',
                           'usedRange': 'const usedRange = ActiveSheet.UsedRange;',
                           'headerRows': 'const headerRows = usedRange.Rows("1:1");',
                           'firstDataRow': 'const firstDataRow = headerRows.RowEnd + 1;',
                           'lastDataRow': 'const lastRow = usedRange.RowEnd;',
                           }.items():
            if name in used_vars and name not in defined_vars:
                if name in js_funcs:
                    code = js_funcs[name]
                if code:
                    pre_additional_code.append(code)
                    used_vars.remove(name)
                else:  # 有未定义就使用的变量，这条数据不要了
                    return 0, text
        else:
            # 还得再检查一波是不是有叫'xxxColumn'的变量未定义被使用
            logo = True
            for name in used_vars:
                if name.endswith('Column') and name not in defined_vars:
                    logo = False
                    break
            if logo and pre_additional_code:
                text = '\n'.join(pre_additional_code) + '\n' + text
            return 1, text

    @classmethod
    def pre_proc(cls, code_text):
        code_text = re.sub(r'^\\n', '', code_text, flags=re.MULTILINE)
        return 1, code_text

    @classmethod
    def fix_loc_head(cls, code_text):
        """ 修复定位头 """
        m1 = re.search(r'//\s*1([\.\s]+)定位', code_text)
        m2 = re.search(r'//\s*2([\.\s]+)业务功能', code_text)
        if not m1 and m2:
            code_text = '// 1' + m2.group(1) + '定位\n' + code_text
        return 1, code_text

    @classmethod
    def remove_stdcode(cls, code_text):
        """ 删除开头固定的组件头代码 """
        code_text = re.sub(r'(.*?)(//\s*1[\.\s]+定位)', r'\2', code_text, flags=re.DOTALL)
        code_text = re.sub(r'// 0 基础组件代码（可以放在功能代码之前，也能放在最后面）.+?$', '', code_text,
                           flags=re.DOTALL)
        return 1, code_text

    @classmethod
    def fix_texts(cls, code_text):
        """ 修复文本中出现的关键词，描述 """
        s = code_text
        s = s.replace('<表格结构信息描述>', '表格摘要')
        s = s.replace('<孩子:表格摘要>', '表格摘要')
        return 1, s

    @classmethod
    def fix_base(cls, code_text):
        text = code_text
        for func in [
            cls.simplify_code,
            cls.fix_colors,
            cls.fix_miscellaneous,
            cls.advanced_remove_comments_regex,
        ]:
            status, text = func(text)
            if not status:
                return status, text
        return status, text

    @classmethod
    def fix_base2(cls, code_text):
        text = code_text
        for func in [
            cls.simplify_code,
            cls.fix_colors,
            cls.fix_miscellaneous,
        ]:
            status, text = func(text)
            if not status:
                return status, text
        return status, text

    @classmethod
    def fix_all(cls, code_text):
        old_text = code_text
        text = code_text
        for func in [
            cls.simplify_code,
            cls.fix_colors,
            cls.fix_miscellaneous,
            cls.fix_stdcode,
            # cls.advanced_remove_comments_regex,
        ]:
            status, text = func(text)
            if not status:
                return status, text
        # if text != old_text:
        #     bcompare(old_text, text)
        #     dprint()
        return status, text

    @classmethod
    def format_hanging_indent(cls, text):
        r""" 优化悬挂缩进的文本排版

        :param str text: 输入文本
        :return str: 优化后的文本

        >>> AirScriptCodeFixer.format_hanging_indent('const usedRange = getUsedRange();\\n        const headerRows = usedRange.Rows(\'1:1\');')
        'const usedRange = getUsedRange();\\nconst headerRows = usedRange.Rows(\'1:1\');'
        """
        lines = text.strip().split('\n')  # 去掉前后空行并分割成行
        first_line = lines.pop(0)  # 取出第1行
        remaining_text = '\n'.join(lines)  # 剩余行合并为一个字符串
        dedented_text = textwrap.dedent(remaining_text)  # 对剩余行进行反缩进处理

        return 1, first_line + '\n' + dedented_text  # 将处理后的剩余行和第1行拼接回去

    @classmethod
    def remove_comments_regex(cls, js_code):
        """ 这个代码功能并不严谨，只是一个临时快速方案 """
        js_code = re.sub(r'^\s*/\*.*?\*/\n?', '', js_code, flags=re.DOTALL | re.MULTILINE)
        js_code = re.sub(r'^\s*//.*\n?', '', js_code, flags=re.MULTILINE)

        # Removing multi-line comments
        js_code = re.sub(r'\s*/\*.*?\*/', '', js_code, flags=re.DOTALL)
        # Removing single-line comments
        js_code = re.sub(r'\s*//.*', '', js_code)
        return js_code

    @classmethod
    def advanced_remove_comments_regex(cls, js_code):
        # Regex to match strings, either single or double quoted
        string_pattern = r'(?:"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\')'

        # Combined regex pattern to match strings or single/multi-line comments
        pattern = r'|'.join([
            string_pattern,  # match strings first to avoid removing content inside them
            r'\/\/[^\n]*',  # single line comments
            r'\/\*.*?\*\/'  # multi-line comments
        ])

        def replacer(match):
            # If the matched text is a string, return it unchanged
            if match.group(0).startswith(('"', "'")):
                return match.group(0)
            # Otherwise, it's a comment, so return an empty string
            return ''

        # Use re.sub with the replacer function
        return 1, re.sub(pattern, replacer, js_code, flags=re.DOTALL)

    @classmethod
    def remove_js_comment(cls, js_code):
        try:
            js_code2 = remove_js_comments(js_code)
        except BaseException as e:
            js_code2 = cls.remove_comments_regex(js_code)
        return 1, js_code2


def __3_js代码结构解析():
    pass


def extract_definitions_with_comments(js_code):
    """ 找出、切分每段函数的定义(函数开头的注释)

    这里是用正则实现的版本，强制要求函数结束的时候用的是单行}结尾
        如果实现中间内容也会出现这种单行}结尾，可以想写特殊手段规避开
    """
    pattern = r"""
    (                                  # 开始捕获注释部分
        (?:(?:/\*[^*]*\*+(?:[^/*][^*]*\*+)*/)|(?://[^\n]*))\s*  # 匹配注释
    )*
    (                                  # 开始捕获声明部分
        \b(?:var|let|const|function)\b\s+    # 匹配声明关键字
    )
    (\w+)                               # 捕获变量名或函数名
    \s*(?:\(.*?\))?\s*\{                # 匹配函数参数列表后跟'{'
    [\s\S]+?                            # 非贪婪匹配所有字符
    (?<=\n\}\n)                         # 确认'}'出现在单独一行
    """
    matches = re.finditer(pattern, js_code, re.VERBOSE)
    definitions = {}
    for match in matches:
        identifier = match.group(3).strip()  # 根据正则表达式的修改，更新捕获组的索引
        full_definition = match.group(0).strip()
        definitions[identifier] = full_definition
    return definitions


def find_identifiers_in_code(code):
    """ 正则实现的找标识符的版本

    用基于esprima的语法树实现的方式，遇到不是那么标准的代码的时候，太多问题和局限了
    还会多此一举过滤掉注释部分等
    """
    return set(re.findall(r'\b(\w+)\b', code))


def find_direct_dependencies(definitions):
    """
    查找每个定义中的直接依赖关系。
    使用 esprima 提取代码中的标识符，并与定义列表求交集。

    :param definitions: 要输入一组数据是因为只检查这一组内的命名空间的东西
    """
    keys = set(definitions.keys())
    dependencies = {key: [] for key in definitions}

    for key, code in definitions.items():
        identifiers = find_identifiers_in_code(code)
        direct_deps = identifiers.intersection(keys)
        dependencies[key] = list(direct_deps - {key})  # 排除自身

    return dependencies


def assemble_dependencies_from_jstools(cur_code, jstools=None, place_tail=False):
    """
    根据输入的 cur_code ，从预设的jstools工具代码库中自动提取所有相关依赖定义

    :param str cur_node: 当前代码
    :param str jstools: 依赖的工具代码
    :param bool place_tail: 把工具代码放在末尾
        放在末尾的目的，是类似jsa那样的场景能在开头直接看到关键的业务代码逻辑

        一般大部分工具函数都是可以放在末尾的
        但是要注意也有个别特殊的实现，是以定义变量的模式来使用的，则不能放倒末尾

    """
    # 1 获得工具代码
    # wps场景支持全局return处理，但这个在编译器里会报错，可以先暴力删掉，不影响我这里的相关处理逻辑
    identifiers_in_input = find_identifiers_in_code(cur_code)

    if jstools is None:
        definitions = get_airscript_head2(True)
    else:
        definitions = extract_definitions_with_comments(jstools)

    # 2 找到所有使用到的符号
    # 初始化结果列表，并按照 definitions 的顺序存储
    visited = set()
    dependencies = find_direct_dependencies(definitions)

    def resolve_dependencies(identifier):
        """递归解决依赖，确保按照 definitions 的顺序添加"""
        if identifier in visited:
            return
        visited.add(identifier)
        for dep in dependencies[identifier]:
            resolve_dependencies(dep)

    # 从输入代码的标识符开始，递归查找依赖
    for identifier in set(definitions.keys()).intersection(identifiers_in_input):
        resolve_dependencies(identifier)

    # 3 拼接代码
    required_code = [definitions[identifier] for identifier in definitions if identifier in visited]
    if place_tail:
        required_code.insert(0, '\n\n// 以下是工具代码')
        required_code.insert(0, cur_code)
    else:
        required_code.append(cur_code)

    return "\n\n".join(required_code)


if __name__ == '__main__':
    pass
