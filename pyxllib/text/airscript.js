function __1_算法工具() {

}

function levenshteinDistance(a, b) {
    const matrix = []

    let i
    for (i = 0; i <= b.length; i++) matrix[i] = [i];

    let j
    for (j = 0; j <= a.length; j++) matrix[0][j] = j;

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
        stdCands = typeof candidates[0] === "string"
            ? candidates.map((s, i) => [i, s])
            : candidates
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

function __2_定位工具() {

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
    for (let i = 1; i <= cels.Count; i++)
        if (cels.Item(i).Text)
            return false
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
    for (; lastRow >= firstRow; lastRow--)
        if (!isEmpty(ur.Rows(lastRow).Cells))
            break
    // 最后一个非空列
    for (; lastCol >= firstCol; lastCol--)
        if (!isEmpty(ur.Columns(lastCol).Cells))
            break
    // 第一个非空行
    for (; firstRow <= lastRow; firstRow++)
        if (!isEmpty(ur.Rows(firstRow).Cells))
            break
    // 第一个非空列
    for (; firstCol <= lastCol; firstCol++)
        if (!isEmpty(ur.Columns(firstCol).Cells))
            break

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
 */
function locateTableRange2(ws, dataRow = [0, 0], fields = []) {
    let [ur, rows, cols] = locateTableRange(ws, dataRow, fields)

    class TableTools {
        constructor(ur, rows, cols) {
            this.ur = ur
            this.rows = rows
            this.cols = cols
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


function __3_json数据导入导出() {

}

// 打包sheet下多个字段fields的数据
// 使用示例：packTableDataFields('料理', ['名称', '标签']
//  fields的参数支持字段名称或整数，明确指定某列的位置
// 返回格式：{'名称': [x1, x2, ...], '标签': [y1, y2, ...]}
// todo fields能否不输入，默认获取所有字段数据（此时需要给出表头所在行）
// todo 多级表头类的数据怎么处理？
function packTableDataFields(ws, fields, dataRow = [0, 0], filterEmptyRows = true) {
    // 1 确定数据范围和字段列号映射
    const [ur, rows, cols] = locateTableRange(ws, dataRow, fields)

    // 2 初始化字段格式数据
    const fieldsData = fields.reduce((dataMap, field) => {
        dataMap[field] = []
        return dataMap
    }, {})

    // 3 遍历数据行填充字段数据
    for (let row = rows.start; row <= rows.end; row++) {
        if (filterEmptyRows) {
            const isEmptyRow = Object.values(cols).every(col => ur.Cells(row, col).Value2 === undefined)
            if (isEmptyRow) continue; // 跳过空行
        }

        // 填充每个字段的数据
        Object.entries(cols).forEach(([field, col]) => {
            fieldsData[field].push(ur.Cells(row, col).Value2)
        })
    }

    // 4 返回结果
    return fieldsData
}

// 和packTableDataFields仅差在返回的数据格式上，这个版本的返回值是主流的jsonl格式
// 返回格式：list[dict]， [{'名称': x1, '标签': y1}, {'名称': x2, '标签': y2}, ...]
function packTableDataList(ws, fields, dataRow, filterEmptyRows = true) {
    const fieldsData = packTableDataFields(ws, fields, dataRow, filterEmptyRows)
    const rowCount = fieldsData[fields[0]].length
    const listData = []

    // 将紧凑格式转换为列表字典格式
    for (let i = 0; i < rowCount; i++) {
        const rowDict = {}
        fields.forEach(field => {
            rowDict[field] = fieldsData[field][i]
        })
        listData.push(rowDict)
    }
    return listData
}

function clearSheetData(headerRow = 1, dataStartRow = 2, ws = ActiveSheet) {
    let headerStartRow, headerEndRow, dataEndRow

    // 检查 headerRow 参数，-1 表示不处理表头
    if (headerRow === -1) {
        headerStartRow = headerEndRow = null
    } else if (typeof headerRow === 'number') {
        headerStartRow = headerEndRow = headerRow
    } else if (Array.isArray(header)) {
        [headerStartRow, headerEndRow] = headerRow
    }

    // 检查 dataStartRow 参数，-1 表示不处理数据区域
    if (dataStartRow === -1) {
        dataStartRow = dataEndRow = null
    } else if (typeof dataStartRow === 'number') {
        let usedRange = ws.UsedRange
        dataEndRow = usedRange.Row + usedRange.Rows.Count - 1
    } else if (Array.isArray(dataStartRow)) {
        [dataStartRow, dataEndRow] = dataStartRow
    }

    // 清空表头区域（保留格式），若未设置为 -1
    if (headerStartRow !== null && headerEndRow !== null) {
        ws.Rows(headerStartRow + ':' + headerEndRow).ClearContents()
    }

    // 删除数据区域（不保留格式），若未设置为 -1
    if (dataStartRow !== null && dataEndRow !== null) {
        ws.Rows(dataStartRow + ':' + dataEndRow).Clear()
    }
}

// 将py里df.to_dict(orient='split')的数据格式写入ws
// 这个数据一般有3个属性：index, columns, data
function writeDfSplitDictToSheet(jsonData, headerRow = 1, dataStartRow = 2, ws = ActiveSheet) {
    let columns = jsonData.columns || []
    let data = jsonData.data || []

    // 若存在 index，则将其添加为第一列
    if (jsonData.index) {
        columns = ['index', ...columns]
        data = jsonData.index.map((idx, i) => [idx, ...data[i]])
    }

    const startCol = ws.UsedRange.Column

    // 写入表头
    if (headerRow > 0) {
        for (let j = 0; j < columns.length; j++) {
            ws.Cells(headerRow, startCol + j).Value2 = columns[j]
        }
    }

    // 写入数据内容
    for (let i = 0; i < data.length; i++) {
        for (let j = 0; j < data[i].length; j++) {
            ws.Cells(dataStartRow + i, startCol + j).Value2 = data[i][j]
        }
    }
}


function writeArrToSheet(arr, startCel) {
    // 遍历数组，将每行的数据写入 Excel
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
 * @param {string} direction - 复制格式的方向，支持 'xlUp' 或 'xlDown'
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
        ws.Rows(insertRange).Insert(direction)
    }
}


function insertNewDataWithHeaders(jsonData, headerRow = 1, dataStartRow = 2, ws = ActiveSheet) {
    // 1 预处理 index，将其合并到 columns 和 data
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
    insertRowsWithFormat(dataStartRow, data.length, ws)
    for (let i = 0; i < data.length; i++) {
        const rowData = data[i]
        for (let j = 0; j < columns.length; j++) {
            const colName = columns[j]
            const colIdx = headerIndexMap[colName]
            ws.Cells(dataStartRow + i, colIdx).Value2 = rowData[j]
        }
    }
}

function __4_py服务工具箱() {

}

// 保留环境状态，运行短小任务，返回代码中print输出的内容
function runPyScript(script, query = '', host = '{{JSA_POST_DEFAULT_HOST}}') {
    const url = `{{JSA_POST_HOST_URL}}/${host}/common/run_py`
    const resp = HTTP.post(url, {query, script}, {
        headers: {
            'Authorization': 'Bearer {{JSA_POST_TOKEN}}',
            'Content-Type': 'application/json'
        }
    })
    return resp.json().output
}

// 每次都是独立环境状态，运行较长时间任务，返回代码中return的字典数据
function runIsolatedPyScript(script, host = '{{JSA_POST_DEFAULT_HOST}}') {
    const url = `{{JSA_POST_HOST_URL}}/${host}/common/run_isolated_py`
    // 判断 script 的类型: script可以只输入py代码，也可以输入配置好的整个字典数据
    const payload = typeof script === 'string' ? {script} : script
    const resp = HTTP.post(url, payload, {
        headers: {
            'Authorization': 'Bearer {{JSA_POST_TOKEN}}',
            'Content-Type': 'application/json'
        }
    })
    return resp.json()
}

function getPyTaskResult(taskId, retries = 1, host = '{{JSA_POST_DEFAULT_HOST}}', delay = 5000) {
    const url = `{{JSA_POST_HOST_URL}}/${host}/common/get_task_result/${taskId}`
    for (let attempt = 0; attempt < retries; attempt++) {
        const resp = HTTP.get(url, {
            headers: {
                'Authorization': 'Bearer {{JSA_POST_TOKEN}}',
                'Content-Type': 'application/json'
            }
        });
        const jsonResp = resp.json()
        if ('__get_task_result_error__' in jsonResp) {
            console.log(`第 ${attempt + 1} 次获取任务结果失败，请稍后再试...`)
            if (attempt < retries - 1) {
                timeSleep(delay)
            }
        } else {
            return jsonResp
        }
    }
}

function __5_日期处理() {
    /*
    为了理解js相关的日期处理原理，有时区和时间戳两个关键点要明白：
    1、js中的Date存储的不是"年/月/日"这样简单的数值，而是有带"时区"标记的，是一个时间点的"精确指定"，而不是"数值描述"。
        说人话，意思就是我们输入的任何时间，都是要精确转换对应到utc0时区的时间上的。
            console.log看到的就是utc版的时间。
        但是我们使用.getHours()等方法的时候，又会自动转换到本地时间的数值。
        理解这套逻辑，就好理解核心的存储、转换逻辑，不容易混乱和出错了。

        Date的初始化还很智能，使用'2024-11'等标准的ISO格式时，默认输入的就是utc时间。
        而使用'2024/11'等斜杠模式的时候，默认输入的就是本地时间。
    2、js的时间戳(timestamp)是1970-01-01到当前的毫秒数，excel的时间戳是1900-01-01到当前的天数(小数部分表示时分秒)
        二者是不一样的，差25569天，所以需要一套转换操作逻辑
        但如果是要把excel的日期转成js，如果没有太高的精度要求，
            且恰好用'2024/11'的斜杠格式表示本地时间的话，有简单的初始化方法 new Date(cell.Text)

    也有人整理的资料：https://bbs.wps.cn/topic/17094

    具体的excelDateToJSDate、jsDateToExcelDate是gpt给我实现的，我也搞不太懂，但测试正确就行了~
    */

    // 输入utc时间点
    console.log(new Date("2024-11-20"))
    const d = new Date();  // 当前时间
    console.log(d)

    // 输入本地时间
    console.log(new Date("2024/11/20"))

    // 看到的是本地时间的"小时"
    console.log(d.getHours())
}

// 将Excel日期时间戳转为JS日期, adjustTimezone是是否默认按照本地时间处理
function excelDateToJSDate(excelDate, adjustTimezone = true) {
    const excelEpochStart = new Date(Date.UTC(1899, 11, 30))
    const jsDate = new Date(excelEpochStart.getTime() + excelDate * 86400000)
    if (adjustTimezone) {
        const timezoneOffset = jsDate.getTimezoneOffset() * 60000
        return new Date(jsDate.getTime() + timezoneOffset)
    }
    return jsDate
}

// 将JS日期转为excel时间戳
function jsDateToExcelDate(jsDate = new Date(), adjustTimezone = true) {
    let adjustedDate = jsDate
    if (adjustTimezone) {
        const timezoneOffset = jsDate.getTimezoneOffset() * 60000
        adjustedDate = new Date(jsDate.getTime() - timezoneOffset)
    }
    const excelEpochStart = new Date(Date.UTC(1899, 11, 30))
    const diffInMs = adjustedDate.getTime() - excelEpochStart.getTime()
    return diffInMs / 86400000
}

// 判断日期是否在本周
function isCurrentWeek(date) {
    const today = new Date()
    today.setHours(0, 0, 0, 0);  // 把时间设为午夜以准确地比较日期
    const firstDayOfWeek = new Date(today.setDate(today.getDate() - today.getDay()))
    const lastDayOfWeek = new Date(today.setDate(today.getDate() - today.getDay() + 6))
    return date >= firstDayOfWeek && date <= lastDayOfWeek
}

// 判断日期是否在当前月份
function isCurrentMonth(date) {
    const currentDate = new Date()
    currentDate.setHours(0, 0, 0, 0)  // 把时间设为午夜stdcode以准确地比较日期
    return date.getMonth() === currentDate.getMonth() && date.getFullYear() === currentDate.getFullYear()
}

// 判断日期是否在下周
function isNextWeek(date) {
    const today = new Date()
    today.setHours(0, 0, 0, 0)  // 把时间设为午夜以准确地比较日期
    const nextWeek = new Date(today.getFullYear(), today.getMonth(), today.getDate() + 7)
    return date > today && date <= nextWeek
}

// 判断日期是否在下个月
function isNextMonth(date) {
    const today = new Date()
    today.setHours(0, 0, 0, 0)  // 把时间设为午夜以准确地比较日期
    const nextMonth = new Date(today.getFullYear(), today.getMonth() + 1, 1)
    const endDateOfNextMonth = new Date(today.getFullYear(), today.getMonth() + 2, 0)
    return date >= nextMonth && date <= endDateOfNextMonth
}

// 正常应该下面这样写就行了。但是AirScript1.0运行这个会报错，AirScript2.0可以运行但没效果。就用了笨办法来做。
// setTimeout(function () {}, milliseconds);
function timeSleep(milliseconds) {
    const startTime = new Date()
    while (new Date() - startTime < milliseconds) {
    }
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

function __6_其他() {

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


function __7_考勤() {
    // 个人的考勤业务定制化功能
}

function highlightCourseProgress(refundDict, cell) {
    let color, refundAmount

    // 1.1 当堂完成
    let cellValue = cell.Value2 + ''
    if (cellValue.includes('当堂')) {
        color = [0, 255, 0]    // 绿色
        refundAmount = refundDict['当堂']
    }

    // 1.2 有返款的回放完成
    if (refundAmount === undefined) {
        // 遍历refundDict中的关键词
        for (const keyword in refundDict) {
            if (cellValue.includes(keyword)) {
                color = [255, 255, 0]    // 黄色
                refundAmount = refundDict[keyword]
                break
            }
        }
    }

    // 1.3 无返款的回放完成
    if (refundAmount === undefined && cellValue.includes('回放')) {
        color = [128, 128, 128]    // 灰色
        refundAmount = 0
    }

    // 1.4 未完成
    if (refundAmount === undefined) {
        color = [255, 255, 255]    // 白色
        refundAmount = 0
    }

    // 2 返回结果

    // 提取百分比
    let percentageRegex = /\d*%/g  // 全局标志 'g'
    let matches = Array.isArray(cellValue.match(percentageRegex)) ? cellValue.match(percentageRegex) : []
    let weight = parseFloat(matches.pop()) || 0 // 最后一个匹配结果

    // 颜色淡化
    for (let i = 0; i < 3; i++)
        color[i] = (color[i] * weight + 255 * 100) / (weight + 100)

    // 设置颜色
    cell.Interior.Color = RGB(color[0], color[1], color[2])

    // 返回返款金额
    return refundAmount
}

// 分析回放规则文本，从中提取结构化的字典解释
function parseRefundRules(text) {
    const match = text.match(/"\d+(\/\d+)*"/)
    const refundDict = {}

    if (match) {
        const values = match[0].slice(1, -1).split('/') // 去掉引号，然后分割
        // 遍历数字并构建键名
        values.forEach((value, index) => {
            if (parseInt(value) === 0) return // 如果数字是0，终止添加
            const key = index === 0 ? "当堂" : `第${index}天`
            refundDict[key] = parseInt(value)
        })
    }
    return refundDict
}

function __x_main() {
    // 我常用的jsa脚本触发运行的模式
}

// 1 这里填上要支持的函数接口清单
const funcsMap = {packTableDataFields}
// 2 支持三种触发方式，及优先级：py-jsa调用 > 选中单元格指定函数名 > 选中单元格所在第1列是触发函数名
let funcName = Context.argv.funcName || Selection.Cells(1, 1).Value2 || ''
if (!funcsMap[funcName]) funcName = ActiveSheet.Cells(Selection.Row, 1)
// 3 如果找得到函数则运行
if (funcsMap[funcName]) return funcsMap[funcName](...Context.argv.args)
// 4 也可以注释掉3，下面写自己手动调试要运行的代码
