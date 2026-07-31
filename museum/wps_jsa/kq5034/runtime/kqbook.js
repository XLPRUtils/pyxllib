function main() {
    // 1 这里填上要支持的api、智能匹配函数接口清单
    const funcsMap = {
        // 基础工具
        findCol,
        writeArrToSheet,
        locateTableRange,
        sqlSelect,
        insertNewDataWithHeaders,
        设置哈希前景色,
        设置哈希背景色,
        // 课程总表
        美化考勤总表,
        // 课次数据
        检索课程链接,
        // 订单操作
        检查已返款,
        执行退款,
        releaseMutexLock,
        // jsa工具
        点我开始转换,
        JSA_AI提示词如下,
        // 新增以下两个供Py调用的接口
        api_release_lock,
    }

    // 2 api：py-jsa脚本令牌模式永远是最高匹配优先级
    if (Context.argv.funcName) return funcsMap[Context.argv.funcName](...(Context.argv.args || []))

    // 3 自定义：也可以注释掉3里的执行部分，下面这里手动写要执行的函数
    // return sanitizeForJSON()

    // 4 智能匹配，优先级：可以在第1个字符串自定义要运行的功能 > 选中单元格指定函数名 > 选中单元格所在第1列是触发函数名
    let funcName = '' || Selection.Cells(1, 1).Value2
    if (!funcsMap[funcName]) funcName = ActiveSheet.Cells(Selection.Row, 1)
    if (funcsMap[funcName]) return funcsMap[funcName]()
}

function __1_课程总表() {
}

function 优化哈希颜色() {
    const [ur, rows, cols] = locateTableRange('课程', 4)
    for (let i = rows.start; i <= rows.end; i++) {
        const color1 = textHashToColor(ur.Cells(i, cols['考勤负责人']).Text)
        ur.Cells(i, cols['考勤负责人']).Font.Color = color1

        const color2 = textHashToColor(ur.Cells(i, cols['课程类型']).Text)
        ur.Cells(i, cols['课程类型']).Font.Color = color2
        ur.Cells(i, cols['课程名称']).Font.Color = color2
    }
}

// 已完成的课程标注灰色
function 设置进度颜色() {
    const [ur, rows, cols] = locateTableRange('课程', 4)
    for (let i = rows.start; i <= rows.end; i++) {
        const val = ur.Cells(i, cols['考勤实际完成结点']).Text
        if (val) {
            // 整行标注灰色242
            ur.Rows(i).Interior.Color = RGB(242, 242, 242)
        } else {
            // 整行取消颜色
            ur.Rows(i).Interior.Color = -1
        }
    }
}

function 美化考勤总表() {
    优化哈希颜色()
    设置进度颜色()
}


function __3_jsa工具代码() {
}

function 点我开始转换() {
    let place_tail = true
    if (Range('E3').Value2 == '前面') {
        place_tail = false
    }
    const dataForPy = { 'code': Range('A4').Value2, 'place_tail': place_tail }
    const pyScript = `
import os
import json
from pyxllib.text.jscode import assemble_dependencies_from_jstools

os.environ['JSA_PYTHON_HOST'] = 'https://example.invalid'
os.environ['JSA_PYTHON_HOST'] = 'https://example.invalid'

def main():
    data = json.loads(r"""${JSON.stringify(dataForPy)}""")
    return {'code': assemble_dependencies_from_jstools(data['code'], place_tail=data['place_tail'])}
`

    const data = runIsolatedPyScript(pyScript, 'codepc_mi15')
    Range('C4').Value2 = data['code'] + '\n'
    console.log(data)
}


function JSA_AI提示词如下() {
    const pyScript = `
from xlproject.code4101 import xlhome_path
file = xlhome_path('slns/pyxllib/pyxllib/text/jsa_ai_prompt.md')
return {'content': file.read_text()}
`
    const jsonData = runIsolatedPyScript(pyScript, 'codepc_mi15')
    Selection.Cells(1, 1).Offset(1, 0).Value2 = jsonData['content']
}

function __4_课次数据() {
}

function 检索课程链接() {
    const [ur, rows, cols, tools] = locateTableRange2('课次数据', 4)
    // 程序运行状态标记
    const taskStatusCel = findCel('检索课程链接', ur).Offset(1, 0)
    const status = setMutexLock(taskStatusCel)

    // 确保当前只有一个实例在运行
    if (status) {
        const pyScript = `
import json
import pandas as pd

from pyxllib.ext.wpsapi import WpsOnlineBook
from xlsln.kq5034.kqmain import KqTools


def main():
    # 1 从jsa触发，来运行py代码
    kq = KqTools()
    kq.xe2.switch_shop('5034山中薪')  # 目前只有店铺1有爬虫

    links = kq.xe2.search_lesson_links(
        "${tools.findargcel('检索课程名称：').Text}",
        "${tools.findargcel('直播状态：').Text}",
        ${tools.findargcel('检索数量上限：').Value2}  # 这里到生产环境要把引号去掉，是数值不是字符串类型
    )

    wb = WpsOnlineBook('<FILE_ID>', '<SCRIPT_ID>')

    # 2 使用生成器逐条获取数据，并写回表格
    for row in links:
        # 2.1 其他字段信息
        row['lesson_id2'] = row['lesson_id']
        row['shop_id'] = 1  # 目前爬虫也只支持店铺1
        del row['lesson_id']

        # 2.2 加入"回放数据"
        row2 = kq.xe2.get_leeson_playback_settings(row['lesson_id2'])
        row.update(row2)

        # 2.3 将单条数据写入wps表格
        # wb.insert_jsonl([row], '课次数据', 2, 4)

        df = pd.DataFrame([row])  # 将当前行转为DataFrame再转dict更方便通用处理
        data = df.to_dict(orient='split')
        del data['index']
        data_json = json.dumps(data, ensure_ascii=False)
        wb.run_func('insertNewDataWithHeaders', data_json, 2, 4, '课次数据')

    # 3 数据插入完成后，清掉正在运行的标记
    # wb.run_airscript("findCel('检索课程链接', Sheets('课次数据').UsedRange).Offset(1, 0).Value2 = ''")
    wb.run_func('api_release_lock', '课次数据', '检索课程链接', '', 1, 0)
    kq.xe2.tab.close()

    return {'status': 'ok'}
`
        const res = runIsolatedPyScript({ script: pyScript, long_task: true }, 'codepc_mi15')
        // taskStatusCel.Value2 += '，taskId=' + res['task_id']
    }
}

function __5_课次更新回放时间() {
}

function __6_订单操作() {
}

function 检查已返款() {
    // 1 定位
    const taskStatusCel = findCel('程序状态：', '订单操作').Offset(1, 0)
    const status = setMutexLock(taskStatusCel)

    // 2 异步执行
    if (status) {
        const pyScript = `
from kq5034 import KqTools
KqTools().kqbook_检查已返款()
`
        runIsolatedPyScript({ script: pyScript, long_task: true }, 'codepc_mi15')
    }
}

function 执行退款() {
    // 1 定位
    const taskStatusCel = findCel('程序状态：', '订单操作').Offset(1, 0)
    const status = setMutexLock(taskStatusCel)

    // 2 异步执行
    if (status) {
        const pyScript = `
from kq5034 import KqTools
KqTools().kqbook_执行退款()
`
        runIsolatedPyScript({ script: pyScript, long_task: true }, 'codepc_mi15')
    }
}

function __x_工具代码() {
}


function __x_工具代码() {
}

// 安全转换为数字类型
function safeToNumber(value) {
    if (value === null || value === undefined || value === '') return NaN
    const num = Number(value)
    return isNaN(num) ? NaN : num
}

// 获取数组中的有效数字
function getValidNumbers(array) {
    return array
        .map(safeToNumber)
        .filter(num => !isNaN(num))
}

// 获取数组中的最大值
function getMaxValue(array) {
    const validNumbers = getValidNumbers(array)
    return validNumbers.length > 0 ? Math.max(...validNumbers) : undefined
}

function levenshteinDistance(a, b) {
    const matrix = []

    let i
    for (i = 0; i <= b.length; i++) matrix[i] = [i]

    let j
    for (j = 0; j <= a.length; j++) matrix[0][j] = j

    for (i = 1; i <= b.length; i++) {
        for (j = 1; j <= a.length; j++) {
            if (b.charAt(i - 1) === a.charAt(j - 1)) {
                matrix[i][j] = matrix[i - 1][j - 1]
            } else {
                matrix[i][j] = Math.min(matrix[i - 1][j - 1] + 1, Math.min(matrix[i][j - 1] + 1, matrix[i - 1][j] + 1))
            }
        }
    }

    return matrix[b.length][a.length]
}

function levenshteinSimilarity(a, b) {
    return (1 - levenshteinDistance(a, b) / Math.max(a.length, b.length)).toFixed(4)
}

/**
 * @description 找到与目标字符串匹配度最高的前K个结果。
 * @param {string} target 目标字符串。
 * @param {Array|Function} candidates 候选集合，有以下三种格式：
 *     1. 字符串数组，例如 ["abc", "def"]。
 *     2. Excel范围对象，例如 Range("A1:A10")，会提取范围中的文本内容。
 *     3. 格式化数组，例如 [ [obj1, str1], [obj2, str2], ... ]，
 *        其中 obj 为原始对象，str 为用于匹配的字符串。
 * @param {number} k 返回的匹配结果数量。
 * @returns {Array} 包含 [obj, text, sim] 的数组，表示对象、文本及匹配度。
 */
function findTopKMatches(target, candidates, k) {
    let stdCands
    if (Array.isArray(candidates)) {
        stdCands = typeof candidates[0] === "string" ? candidates.map((s, i) => [i, s]) : candidates
    } else if (typeof candidates === "function") {
        stdCands = []
        for (let i = 1; i <= candidates.Rows.Count; i++) {
            const cel = candidates.Cells(i, 1)
            stdCands.push([cel, cel.Text])
        }
    } else {
        throw new Error("Unsupported format.")
    }

    const results = stdCands
        .map(([obj, str]) => [obj, str, parseFloat(levenshteinSimilarity(target, str))])
        .sort((a, b) => b[2] - a[2])
    return k ? results.slice(0, k) : results
}

/**
 * 设置互斥锁，主要用在使用js-py的场景，确保触发的py程序只有一个
 * @param {Range} cel 单元格对象
 * @returns {string} 如果已经有锁，返回undefined；否则返回新设置的锁
 */
function setMutexLock(cel) {
    if (cel.Text.startsWith('已启动程序: ')) return
    cel.Value2 = '已启动程序: ' + formatLocalDatetime(new Date())
    return cel.Text
}

/** 
 * 使用js-py同步的时候，一般是js端释放锁，异步则是py端释放锁
 * @param {Range|string} celOrSheetName 单元格对象或工作表名
 * @param {string} argName 要解锁的参数名
 * @param {number} rowOffset 相对于argCel的行偏移量，默认参数值在参数名下面
 * @param {number} colOffset 相对于argCel的列偏移量
 */
function releaseMutexLock(celOrSheetName, argName, rowOffset=1, colOffset=0) {
    let cel
    if (typeof celOrSheetName === 'string') {
        cel = findCel(argName, celOrSheetName)
        if (cel === undefined) throw new Error(`未找到参数名${argName}`)
        cel = cel.Offset(rowOffset, colOffset)
    } else {
        cel = celOrSheetName
    }
    cel.Value2 = '待机中，可启动程序'
}

/**
 * Find的参数很多：https://airsheet.wps.cn/docs/apiV2/excel/workbook/Range/%E6%96%B9%E6%B3%95/Find%20%E6%96%B9%E6%B3%95.html
 * 但个人感觉比较可能需要配置到的就lookAt，如果有其他特殊定位需求，可以自己使用类似原理.Find找到行列就好
 * @param what 要查找的内容
 * @param ur 查找区域，默认当前表格UsedRange
 * @param lookAt 可以选xlWhole（单元格内容=what）或xlPart（单元格内容包含了what）
 * @return 找到的单元格
 * todo 还没思考如果匹配情况不唯一怎么处理，目前都是返回第1个匹配项。因为这我自己重名场景不多，真遇到也可以手动约束ur范围后适当解决该问题。
 * todo 支持多行表头的嵌套定位？比如料理一级标题下的二级标题合计定位方式：['料理', '合计']，可以区别于另一个"合计": ['酒水', '合计']
 */
function findCel(what, ur = ActiveSheet.UsedRange, lookAt = xlWhole) {
    if (typeof ur === 'string') ur = ur.includes(':') ? Range(ur) : Sheets(ur).UsedRange
    return ur.Find(what, undefined, undefined, lookAt)
}

function findRow(what, ur = ActiveSheet.UsedRange, lookAt = xlWhole) {
    const cel = findCel(what, ur, lookAt)
    if (cel) return cel.Row
}

function findCol(what, ur = ActiveSheet.UsedRange, lookAt = xlWhole) {
    let cel = findCel(what, ur, lookAt)
    if (cel) return cel.Column
}

// 判断 cells 集合是否全空
function isEmpty(cels) {
    for (let i = 1; i <= cels.Count; i++) if (cels.Item(i).Text) return false
    return true
}

// 获取ws实际使用的区域：会裁剪掉四周没有数据的空白区域（之前被使用过的区域或设置过格式等操作，默认ur会得到空白区域干扰数据范围定位）
function getUsedRange(ws = ActiveSheet) {
    // 1 定位默认的UsedRange
    if (typeof ws === 'string') ws = Sheets(ws)
    let ur = ws.UsedRange
    let firstRow = 1, firstCol = 1, lastRow = ur.Rows.Count, lastCol = ur.Columns.Count

    // 2 裁剪四周
    // todo 待官方支持TRIMRANGE后可能有更简洁的解决方案。期望官方底层不是这样暴力检索，应该有更高效的解决方式

    // 找到最后一个非空行
    for (; lastRow >= firstRow; lastRow--) if (!isEmpty(ur.Rows(lastRow).Cells)) break
    // 最后一个非空列
    for (; lastCol >= firstCol; lastCol--) if (!isEmpty(ur.Columns(lastCol).Cells)) break
    // 第一个非空行
    for (; firstRow <= lastRow; firstRow++) if (!isEmpty(ur.Rows(firstRow).Cells)) break
    // 第一个非空列
    for (; firstCol <= lastCol; firstCol++) if (!isEmpty(ur.Columns(firstCol).Cells)) break

    // 3 创建一个新的 Range 对象，它只包含非空的行和列
    return ws.Range(ur.Cells(firstRow, firstCol), ur.Cells(lastRow, lastCol))
}

/**
 * 表格结构化定位工具
 * @param ws 输入表格名，或表格对象
 * @param dataRow 输入两个值的数组，第1个值标记(不含表头的)数据起始行，第2个值标记数据结束行。
 *  只输入单数值，未传入第2个参数时，默认以0填充，例如：4 -> [4, 0]
 *  起始行标记：
 *      0，智能检测。如果cols有给入字段名，以找到的第1个字段的下一行作为起始行。否则默认设置为ur的第2行。
 *      正整数，人工精确指定数据起始行（输入的是整张表格的绝对行号）
 *      '料理'等精确的字段名标记，以找到的单元格下一行作为数据起始行
 *      负数，比如-2，表示基于第2列（B列），使用.End(xlDown)机制找到第1条有数据的行的下一行作为数据起始行
 *  结束行标记：
 *      0，智能检测。以getUsedRange的最后一行为准。
 *      正整数，人工精确指定数据结束行（有时候数据实际可能有100行，可以只写10，实现少量部分样本的功能测试）
 *      '料理'等精确的字段名标记，同负数模式，以找到的所在列，配合.End(xlUp)确定最后一行有数据的位置
 *      负数，比如-3，表示基于第3列（C列），使用.End(xlUp)对这列的最后一行数据位置做判定，作为数据最后一行的标记
 * @param fields 后续要使用到的相关字段数据，使用as2.0版本的时候，该参数可以不输入，会在使用中动态检索
 * @return [ur, rows, cols]
 *      ur，表格实际的UsedRange
 *      rows是字典，rows.start、rows.end分别存储了数据的起止行
 *      cols也是字典，存储了个字段名对应的所在列编号，比如cols['料理']
 *      注：返回的行、列，都是相对ur的位置，所以可以类似这样 ur.Cells(rows.start, cols[x]) 取到第1条数据在x字段的值
 */
function as1_locateTableRange(ws, dataRow = [0, 0], fields = []) {
    // 1 初步确定数据区域getUsedRange范围
    const ur = getUsedRange(ws)
    ws = ur.Worksheet
    // dataRow可以输入单个数值
    if (typeof dataRow === 'number') dataRow = [dataRow, 0]
    let rows = {
        start: dataRow[0] === 0 ? ur.Row + 1 : dataRow[0],
        end: dataRow[1] === 0 ? ur.Row + ur.Rows.Count - 1 : dataRow[1]
    }

    // 2 获取列名对应的列号
    let cols = {}
    fields.forEach(colName => {
        const col = findCol(colName, ur)
        if (col) {
            cols[colName] = col
            // 如果此时rows.start还未确定，则以该单元格的下一行作为数据起始行
            if (rows.start === 0) rows.start = findRow(colName, ur) + 1 || 0  // 有可能会找不到，则保持0
        }
    })

    // 3 定位行号
    if (typeof rows.start === 'string') rows.start = findRow(rows.start, ur) + 1
    if (rows.start < 0) {
        const col = -rows.start
        rows.start = 2
        if (isEmpty(ws.Cells(1, col))) rows.start = ws.Cells(1, col).End(xlDown).Row + 1
    }

    if (typeof rows.end === 'string') rows.end = -findCol(rows.end, ur)
    if (rows.end < 0) {
        const cel = ws.Cells(ws.Rows.Count, -rows.end)
        rows.end = cel.Row
        if (isEmpty(cel)) rows.end = cel.End(xlUp).Row
    }

    // 4 转成ur里的相对行号
    rows.start -= ur.Row - 1
    rows.end -= ur.Row - 1
    for (const colName in cols) cols[colName] -= ur.Column - 1

    return [ur, rows, cols]
}

function locateTableRange(ws, dataRow = [0, 0], fields = []) {
    // 1 先获得基础版本的结果
    let [ur, rows, cols] = as1_locateTableRange(ws, dataRow, fields)

    // 2 使用 Proxy 实现动态查找未配置的字段（该功能仅AirScript2.0可用，1.0请使用as1_locateTableRange接口）
    cols = new Proxy(cols, {
        get(target, prop) {
            if (prop in target) {
                return target[prop] // 已配置字段，直接返回
            } else {
                const dynamicCol = findCol(prop, ur) // 动态查找
                if (dynamicCol) {
                    target[prop] = dynamicCol // 缓存动态找到的列
                    return dynamicCol
                }
            }
        }
    })

    // cols支持在使用中动态自增字段
    return [ur, rows, cols]
}

/**
 * 表格结构化定位工具的增强版本，在locateTableRange基础上增加了tools简化一些常用操作
 * tools增加的工具详见内部实现的子函数注释
 * todo 250109周四14:07 这套实现并不太好，过渡封装了，后续还是研究下怎么做出ur.Cells我感觉更好。
 */
function locateTableRange2(ws, dataRow = [0, 0], fields = []) {
    let [ur, rows, cols] = locateTableRange(ws, dataRow, fields)

    class TableTools {
        constructor(ur, rows, cols) {
            this.ur = ur
            this.rows = rows
            this.cols = cols
        }

        getcel(row, colName) {
            return this.ur.Cells(row, this.cols[colName])
        }

        /**
         * 获取指定行和列名的单元格值
         * @param {number} row 行号
         * @param {string} colName 列名
         * @return {any} 单元格的Value2值
         */
        getval(row, colName) {
            return this.ur.Cells(row, this.cols[colName]).Value2
        }

        gettext(row, colName) {
            return this.ur.Cells(row, this.cols[colName]).Text
        }

        /**
         * 查找参数名对应的单元格
         * @param {string} argName 参数名
         * @param {string} direction 查找方向，'down' 表示下方，'right' 表示右侧，默认为 'down'
         * @return {any} 单元格对象
         */
        findargcel(argName, direction = 'down') {
            const cel = findCel(argName, this.ur)
            if (!cel) {
                // 如果未找到参数名，返回 undefined
                return undefined
            }

            let targetCell
            if (direction === 'down') {
                // 查找下方单元格
                targetCell = cel.Offset(1, 0)
            } else if (direction === 'right') {
                // 查找右侧单元格
                targetCell = cel.Offset(0, 1)
            } else {
                // 如果方向不正确，抛出错误
                throw new Error(`未知的方向参数: ${direction}`)
            }

            // 返回目标单元格的值
            return targetCell
        }
    }

    let tools = new TableTools(ur, rows, cols)
    return [ur, rows, cols, tools]
}

/**
 * 打包sheet下多个字段fields的数据
 * @param ws 表格名或表格对象
 * @param fields 要打包的字段名或列号，支持字段名称或整数，明确指定某列的位置
 * @param dataRow 数据起始行，默认[0, 0]
 * @param filterEmptyRows 是否过滤空行，默认true
 * @param useTextFormat 是否根据单元格格式返回Text格式，默认true
 * @return {'名称': [x1, x2, ...], '标签': [y1, y2, ...]}
 */
// 使用示例：sqlSelect('料理', ['名称', '标签']
// todo fields能否不输入，默认获取所有字段数据（此时需要给出表头所在行）
// todo 多级表头类的数据怎么处理？
// todo 支持一定的筛选功能？避免表格太大时要传输的数据过多。
function sqlSelect(ws, fields, dataRow = [0, 0], filterEmptyRows = true, useTextFormat = true) {
    // 1 确定数据范围和字段列号映射
    const [ur, rows, cols] = locateTableRange(ws, dataRow, fields)

    // 2 初始化字段数据和格式映射
    const fieldsData = fields.reduce((dataMap, field) => {
        dataMap[field] = []
        return dataMap
    }, {})

    const formatMap = {}
    if (useTextFormat) {
        Object.entries(cols).forEach(([field, col]) => {
            let firstCell = ur.Cells(rows.start, col)
            let format = firstCell.NumberFormat
            formatMap[field] = format !== 'G/通用格式' ? 'Text' : 'Value2'
        })
    }

    // 3 遍历数据行填充字段数据
    for (let row = rows.start; row <= rows.end; row++) {
        if (filterEmptyRows) {
            const isEmptyRow = Object.values(cols).every(col => ur.Cells(row, col).Value2 === undefined)
            if (isEmptyRow) continue; // 跳过空行
        }

        // 填充每个字段的数据
        Object.entries(cols).forEach(([field, col]) => {
            let cell = ur.Cells(row, col)
            if (useTextFormat) {
                fieldsData[field].push(formatMap[field] === 'Text' ? cell.Text : cell.Value2)
            } else {
                fieldsData[field].push(cell.Value2)
            }
        })
    }

    // 4 返回结果
    return fieldsData
}

function writeArrToSheet(arr, startCel) {
    // 1 startCel可以输入字符串，且注意这样是可以附带表格位置信息的 'Sheet1!A1'
    // 如果是字符串，转Range对象
    if (typeof startCel === 'string') startCel = Range(startCel)

    // 2 遍历数组，将每行的数据写入 Excel
    for (let i = 0; i < arr.length; i++) {
        const row = arr[i]
        // 如果当前行存在，则遍历该行的元素
        if (Array.isArray(row)) {
            for (let j = 0; j < row.length; j++) {
                startCel.Offset(i, j).Value2 = row[j]
            }
        }
    }
}

/**
 * 插入新行并复制格式，兼容jsa1.0和2.0，并可选择格式复制方向
 * @param {number} dataStartRow - 数据起始行
 * @param {number} insertCount - 需要插入的行数
 * @param {string} direction - 复制格式的方向，支持 'xlUp'（往上复制，表示基于下面一行的格式） 或 'xlDown'（与xlUp相反）
 * @param {object} ws - 工作表对象，默认为ActiveSheet
 */
function insertRowsWithFormat(dataStartRow, insertCount, ws = ActiveSheet, direction = 'xlUp') {
    if (insertCount <= 0) return
    const insertRange = `${dataStartRow}:${dataStartRow + insertCount - 1}`

    if (ws.Rows.RowEnd) {  // jsa1.0
        ws.Rows(insertRange).Insert()
        ws.Rows(insertRange).ClearContents()  // 1.0有可能会出现插入的不是空行，还顺带拷贝了数据~
        if (direction === 'xlUp') {
            ws.Rows(dataStartRow + insertCount).Copy()
            ws.Rows(insertRange).PasteSpecial(xlPasteFormats)
        }
    } else {
        // 2.0的insert才能传参。1.0或默认不传参相当于是xlDown的效果，指新插入的行是拷贝的上面一行的格式。
        direction = direction === 'xlUp' ? xlUp : xlDown
        ws.Rows(insertRange).Insert(direction)
    }
}


function insertNewDataWithHeaders(jsonData, headerRow = 1, dataStartRow = 2, ws = ActiveSheet, direction = 'xlUp') {
    // 1 预处理 index，将其合并到 columns 和 data
    jsonData = typeof jsonData === 'string' ? JSON.parse(jsonData) : jsonData
    ws = typeof ws === 'string' ? Sheets(ws) : ws

    let columns = jsonData.columns || []
    let data = jsonData.data || []
    if (jsonData.index) {
        columns = ['index', ...columns]
        data = jsonData.index.map((idx, i) => [idx, ...data[i]])
    }

    // 2 处理可能出现的新字段
    // 获取现有的表头
    let existingHeaders = []
    const usedRange = ws.UsedRange;
    for (let col = usedRange.Column; col <= usedRange.Column + usedRange.Columns.Count - 1; col++) {
        existingHeaders.push(ws.Cells(headerRow, col).Value2)
    }

    // 计算新增的字段
    const newHeaders = columns.filter(column => !existingHeaders.includes(column))
    const allHeaders = [...existingHeaders, ...newHeaders]

    // 如果有新字段，扩展表头
    if (newHeaders.length > 0) {
        for (let j = 0; j < allHeaders.length; j++) {
            ws.Cells(headerRow, usedRange.Column + j).Value2 = allHeaders[j]
        }
    }

    // 构建插入数据的映射关系
    const headerIndexMap = {}
    for (let j = 0; j < allHeaders.length; j++) {
        headerIndexMap[allHeaders[j]] = usedRange.Column + j
    }

    // 3 插入新行
    insertRowsWithFormat(dataStartRow, data.length, ws, direction)
    for (let i = 0; i < data.length; i++) {
        const rowData = data[i]
        for (let j = 0; j < columns.length; j++) {
            const colName = columns[j]
            const colIdx = headerIndexMap[colName]
            ws.Cells(dataStartRow + i, colIdx).Value2 = rowData[j]
        }
    }
}

// 服务器路径，url
const JSA_POST_HOST_URL = 'https://code4101.com'

// 要处理的目标主机
const JSA_POST_DEFAULT_HOST = 'codepc_mi15'

// 请求的header格式，以及对应的token
const JSA_HTTP_HEADERS = {
    'Authorization': '<REDACTED>',
}

// 每次都是独立环境状态，运行较长时间任务，返回代码中return的字典数据
function runIsolatedPyScript(script, host = JSA_POST_DEFAULT_HOST) {
    const url = `${JSA_POST_HOST_URL}/${host}/common/run_isolated_py`
    // 判断 script 的类型: script可以只输入py代码，也可以输入配置好的整个字典数据
    const payload = typeof script === 'string' ? { script } : script
    const resp = HTTP.post(url, payload, { headers: JSA_HTTP_HEADERS })
    return resp.json()
}

// 格式化输出本地时间
function formatLocalDatetime(date = new Date()) {
    function pad(num) {  // 补齐两位数
        return num < 10 ? '0' + num : num
    }

    let year = date.getFullYear()
    let month = pad(date.getMonth() + 1)  // getMonth() 返回的月份是从0开始的
    let day = pad(date.getDate())
    let hour = pad(date.getHours())
    let minute = pad(date.getMinutes())
    let second = pad(date.getSeconds())
    // 我自己的日期风格，/格式一定是本地时间，-格式则本地和utc0时间都有可能
    return `${year}/${month}/${day} ${hour}:${minute}:${second}`
}

/**
 * @description 计算文本的哈希颜色
 * @param {string} text 要计算颜色的文本
 * @param {number} light 0 不改变颜色，0~1，增加亮度权重，-1~0，增加暗度权重
 * 参考资料：cel.Font.Color 设置前景色，cel.Interior.Color 设置背景色
 */
function textHashToColor(text, light = 0) {
    // 1 计算哈希值
    const hash = Array.from(text).reduce((acc, char) =>
        ((acc << 5) - acc + char.charCodeAt(0)) | 0, 0)

    if (hash === 0) return -1

    // 2 提取RGB组件
    let [r, g, b] = [
        (hash & 0xFF0000) >> 16,
        (hash & 0x00FF00) >> 8,
        hash & 0x0000FF
    ]

    // 3 调整亮度
    if (light !== 0) {
        const adjust = (value) => light > 0
            ? (1 - light) * value + light * 255
            : (1 + light) * value;

        [r, g, b] = [r, g, b].map(adjust).map(Math.round)
    }

    return RGB(r, g, b)
}

function 设置哈希前景色(rng, light = 0) {
    rng = rng || Selection
    for (let i = 1; i <= rng.Rows.Count; i++) {
        for (let j = 1; j <= rng.Columns.Count; j++) {
            let cell = rng.Cells(i, j)
            cell.Font.Color = textHashToColor(cell.Text, light)
        }
    }
}

function 设置哈希背景色(rng, light = 0) {
    rng = rng || Selection
    for (let i = 1; i <= rng.Rows.Count; i++) {
        for (let j = 1; j <= rng.Columns.Count; j++) {
            let cell = rng.Cells(i, j)
            cell.Interior.Color = textHashToColor(cell.Text, light)
        }
    }
}

// 将数据转换为可标准化为json的格式
function sanitizeForJSON(data, depth = 1) {
    const type = typeof data

    // 1 基本类型直接返回
    if (data === undefined) return null
    if (data === null || type === 'number' || type === 'string' || type === 'boolean') return data

    // 2 处理数组
    if (Array.isArray(data)) return data.map(sanitizeForJSON)

    // 3 通过关键词判定特殊类型
    // 判定所用的key要尽量冷门，避免和普通字典的有效key冲突歧义。一般可以挑一个名字最长的，实在不行的时候也可以复合检查多个key。
    if (data.hasOwnProperty('FillAcrossSheets')) return 'Sheets'
    else if (data.hasOwnProperty('EnableFormatConditionsCalculation')) return `Sheets('${data.Name}')`
    // Cells也算Range类型
    else if (data.hasOwnProperty('CalculateRowMajorOrder')) return `Range('${data.Address(false, false)}')`

    // 4 处理对象
    const result = {}
    let isEmpty = true

    for (const key in data) {
        if (data.hasOwnProperty(key)) {
            result[key] = depth > 0 ? sanitizeForJSON(data[key], depth - 1) : ''
            isEmpty = false
        }
    }

    return isEmpty ? type : result
}

/**
 * 为单元格添加或删除超链接
 * @param {Object} cel - 单元格对象
 * @param {string} [link] - 超链接地址（可选）。如果不提供，则删除超链接。
 * @param {string} [text] - 显示文本（可选）。默认使用单元格当前内容，如果为空则显示链接地址。
 */
function setHyperlink(cel, link, text, screenTip) {
    // 必须清空，否则如果在旧的url基础上操作，实测会有bug
    cel.Hyperlinks.Delete()
    if (link) {
        // 如果文本参数未提供，则默认使用单元格当前内容
        // 如果单元格内容为空，则设置为 undefined，这样会默认显示链接地址
        const displayText = text !== undefined ? text : (cel.Value2 || undefined)
        // 各位置参数意思：目标单元格，主链接(文件内引用可能会用到)，次链接，悬浮提示文本，展示文本
        // 不过我在wps表格上没测出screenTip效果
        cel.Hyperlinks.Add(cel, link, undefined, screenTip, displayText)
    }
}

/**
 * API接口：释放锁 / 修改指定单元格的值
 * 用于 Python 任务完成后回调修改状态
 * @param {string} sheetName 表格名称
 * @param {string} anchorText 定位锚点文本 (如 "检索课程链接")
 * @param {string} value 要写入的值 (空字符串表示解锁)
 * @param {number} rowOffset 行偏移
 * @param {number} colOffset 列偏移
 */
function api_release_lock(sheetName, anchorText, value, rowOffset = 1, colOffset = 0) {
    const ws = Sheets(sheetName);
    const cel = findCel(anchorText, ws.UsedRange);
    if (cel) {
        cel.Offset(rowOffset, colOffset).Value2 = value;
        return "lock released";
    }
    return "anchor not found";
}

function __y_main() {

}

const res = sanitizeForJSON(main())
console.log(res)
// noinspection JSAnnotator
return res
