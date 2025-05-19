// 我个人一般使用的启动函数
function __main__() {
    // 1 这里填上api、智能匹配要支持的函数接口清单
    const funcsMap = {
        findCol,
        writeArrToSheet,
        locateTableRange,
        packTableDataFields,
    }

    // 2 api：py-jsa脚本令牌模式永远是最高匹配优先级
    if (Context.argv.funcName) return funcsMap[Context.argv.funcName](...Context.argv.args)

    // 3 自定义：有需要可以打开这部分代码，手动设计要执行的功能，并使用return结束不用进入第4部分
    // return sanitizeForJSON()

    // 4 智能匹配，优先级：可以在第1个字符串自定义要运行的功能 > 选中单元格指定函数名 > 选中单元格所在第1列是触发函数名
    let funcName = '' || Selection.Cells(1, 1).Value2
    if (!funcsMap[funcName]) funcName = ActiveSheet.Cells(Selection.Row, 1)
    if (funcsMap[funcName]) return funcsMap[funcName]()
}

function __0_prog() {

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

function __1_定位工具() {

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


function __2_json数据导入导出() {

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
// 使用示例：packTableDataFields('料理', ['名称', '标签']
// todo fields能否不输入，默认获取所有字段数据（此时需要给出表头所在行）
// todo 多级表头类的数据怎么处理？
// todo 支持一定的筛选功能？避免表格太大时要传输的数据过多。
function packTableDataFields(ws, fields, dataRow = [0, 0], filterEmptyRows = true, useTextFormat = true) {
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

function __3_py服务工具() {

}

// 服务器路径，url
const JSA_POST_HOST_URL = '{{JSA_POST_HOST_URL}}'
// 要处理的目标主机
const JSA_POST_DEFAULT_HOST = '{{JSA_POST_DEFAULT_HOST}}'
// 请求的header格式，以及对应的token
const JSA_HTTP_HEADERS = {
    'Authorization': 'Bearer {{JSA_POST_TOKEN}}', 'Content-Type': 'application/json'
}

// 保留环境状态，运行短小任务，返回代码中print输出的内容
function runPyScript(script, query = '', host = JSA_POST_DEFAULT_HOST) {
    const url = `${JSA_POST_HOST_URL}/${host}/common/run_py`
    const resp = HTTP.post(url, {query, script}, {headers: JSA_HTTP_HEADERS})
    return resp.json().output
}

// 每次都是独立环境状态，运行较长时间任务，返回代码中return的字典数据
function runIsolatedPyScript(script, host = JSA_POST_DEFAULT_HOST) {
    const url = `${JSA_POST_HOST_URL}/${host}/common/run_isolated_py`
    // 判断 script 的类型: script可以只输入py代码，也可以输入配置好的整个字典数据
    const payload = typeof script === 'string' ? {script} : script
    const resp = HTTP.post(url, payload, {headers: JSA_HTTP_HEADERS})
    return resp.json()
}

function getPyTaskResult(taskId, retries = 1, host = JSA_POST_DEFAULT_HOST, delay = 5000) {
    const url = `${JSA_POST_HOST_URL}/${host}/common/get_task_result/${taskId}`
    for (let attempt = 0; attempt < retries; attempt++) {
        const resp = HTTP.get(url, {headers: JSA_HTTP_HEADERS})
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

function __4_日期处理() {
    /*
    文档：https://www.yuque.com/xlpr/pyxllib/zdgppdtls3a15nhg
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

function __5_其他() {

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

// 将条件格式的应用范围从部分行扩展到整列（手动增删表格过程中，可能会破坏原本比如L:L条件格式范围为L1:L100，这里可以批量调整变回L:L）
function extendFormatConditionsToFullColumns(ws) {
    const formatConditions = ws.UsedRange.FormatConditions
    for (let i = 1; i <= formatConditions.Count; i++) {
        const condition = formatConditions.Item(i)
        const addr = condition.AppliesTo.Address()
        // 使用正则匹配类似 $J$2:$J$1048576 的格式，变成$J:$J
        const match = addr.match(/^\$([A-Z]+)\$\d+:\$\1\$\d+$/)
        if (match) {
            // match[1] 是捕获的列字母
            const colLetter = match[1]
            condition.ModifyAppliesToRange(st.Range(`${colLetter}:${colLetter}`))
        }
    }
}


function 测试中_新建一条条件格式() {
    const ws = ActiveSheet
    const rng = ws.UsedRange

    const formatConditions = rng.FormatConditions
    // B列重复值高亮
    const newCondition = formatConditions.Add(st.XlFormatConditionType.xlCellValue, st.XlFormatConditionOperator.xlEqual, '=B2')

}

function __5_资源描述计算() {
    // 我自己原创的一套资源描述机制、应用
}

// 简单的高精度计算类
class SimpleDecimal {
    constructor(value) {
        this.value = typeof value === 'string' ? value : String(value)
    }

    plus(other) {
        const otherValue = other instanceof SimpleDecimal ? other.value : String(other)
        const result = (parseFloat(this.value) + parseFloat(otherValue)).toFixed(10)
        return new SimpleDecimal(parseFloat(result))
    }

    minus(other) {
        const otherValue = other instanceof SimpleDecimal ? other.value : String(other);
        const result = (parseFloat(this.value) - parseFloat(otherValue)).toFixed(10)
        return new SimpleDecimal(parseFloat(result))
    }

    isZero() {
        return Math.abs(parseFloat(this.value)) < 1e-10
    }

    isPositive() {
        return parseFloat(this.value) > 0
    }

    toString() {
        return this.value
    }
}

/**
 * 解析物品描述字符串，返回物品及其数量的字典 (使用高精度计算)
 * @param desc 物品描述，如"招募令1,三品元气宝箱1.5,灵石120,玄晶120" (支持小数数量)
 * @returns {Object} 返回格式 {"招募令": SimpleDecimal(1), "三品元气宝箱": SimpleDecimal(1.5), ...}
 */
function 解析物资清单(desc) {
    if (!desc) return {}

    return desc.split(',').reduce((items, item) => {
        // 跳过空项
        if (!item.trim()) return items

        const m = item.match(/^(.+?)([+-]?\d+\.?\d*)?$/)
        if (!m) {
            const itemName = item.trim()
            items[itemName] = items[itemName]
                ? items[itemName].plus(1)
                : new SimpleDecimal(1)
        } else {
            const itemName = m[1].trim()
            const countStr = m[2] ? m[2].trim() : '1'
            const count = new SimpleDecimal(countStr)
            items[itemName] = items[itemName]
                ? items[itemName].plus(count)
                : new SimpleDecimal(count)
        }
        return items
    }, {})
}

/**
 * 资源清单减法操作，支持高精度计算
 * @param desc1 第一个资源清单描述
 * @param desc2 第二个资源清单描述 (要减去的)
 * @returns {string} 减法结果的字符串表示
 * 示例：'a1,b2,c3'-'b1,c1'='a,b,c2'
 */
function 物资清单相减(desc1, desc2) {
    const items1 = 解析物资清单(desc1)
    const items2 = 解析物资清单(desc2)
    const result = {}

    // 添加第一个清单的项目
    for (const item in items1) {
        result[item] = items1[item] // 已经是 SimpleDecimal 实例
    }

    // 减去第二个清单的项目
    for (const item in items2) {
        if (result[item]) {
            result[item] = result[item].minus(items2[item])
        } else {
            // 如果第一个清单没有此项，则添加负值
            result[item] = new SimpleDecimal(0).minus(items2[item])
        }
    }

    // 构建结果字符串，过滤掉数量为0的项目
    return Object.entries(result)
        .filter(([_, count]) => !count.isZero())
        .map(([item, count]) => {
            // 如果数量等于1，则省略数量显示
            if (count.toString() === '1') {
                return item
            } else {
                return `${item}${count}`
            }
        })
        .join(',')
}

/**
 * 资源清单加法操作，支持高精度计算
 * @param desc1 第一个资源清单描述
 * @param desc2 第二个资源清单描述 (要加上的)
 * @returns {string} 加法结果的字符串表示
 * 示例：'a,b2,c3'+'b,c'='a,b3,c4'
 */
function 物资清单相加(desc1, desc2) {
    const items1 = 解析物资清单(desc1)
    const items2 = 解析物资清单(desc2)
    const result = {}

    // 添加第一个清单的项目
    for (const item in items1) {
        result[item] = items1[item] // 已经是 SimpleDecimal 实例
    }

    // 加上第二个清单的项目
    for (const item in items2) {
        if (result[item]) {
            result[item] = result[item].plus(items2[item])
        } else {
            // 如果第一个清单没有此项，则直接添加
            result[item] = items2[item]
        }
    }

    // 构建结果字符串，过滤掉数量为0的项目
    return Object.entries(result)
        .filter(([_, count]) => !count.isZero())
        .map(([item, count]) => {
            // 如果数量等于1，则省略数量显示
            if (count.toString() === '1') {
                return item
            } else {
                return `${item}${count}`
            }
        })
        .join(',')
}
