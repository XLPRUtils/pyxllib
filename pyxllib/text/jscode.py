#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Sponsor 甲方: 泽少
# @Author : 本少：https://lyeebn.gitee.io/technology-shop/HeyBoss.html
# @Date   : 2023/09/27

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
        r=i+1
        len_c = len(c)
        while len_c > l >0:
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
        c = c[:l+1] + ('\n' if 有回车 else '') + c[上一回车处:]  # 有必要多留个空格吗
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


class JSParser():
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
            elif si == '$' and self.jsSourceCode[self.indexPointer + 1] == '{':  # 重要区别 。经实验 ${  后必有 } 否则报错：g = `2${2+3 7`
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
        elif 回溯区分正则除号(self.jsWithoutComment): #bool(re.search('[\n\r(\[\{=+,;&?|]([ \t]*|\\[\n\r])*[ \t]*$', self.jsWithoutComment)):  # def 区分正则或除号(): ←
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
            elif si == "S" and self.jsCodeLength - self.indexPointer > 11 and self.jsSourceCode[ self.indexPointer:self.indexPointer + 11] == 'String.raw`':  # String.raw` 元字符串
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


if __name__ == '__main__':
    pass
