function __1_定位工具() {

}

// 根据提供的 pattern 在 range 中寻找 cell
// 如果没有提供 range，默认在 ActiveSheet.UsedRange 中寻找
function findCell(pattern, range = ActiveSheet.UsedRange) {
    // return range.Find(pattern, range, xlValues, xlWhole);  // 241119周二21:43，1.0突然就不兼容这么用了
    return range.Find(pattern);
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

// 兼容1.0版本的常规使用方式
function findColumn(pattern, range = ActiveSheet.UsedRange) {
    let cell = findCell(pattern, range);
    let columnIndex;
    if (!cell) {
        let minDistance = Infinity;
        for (let i = 1; i <= range.Columns.Count; i++) {
            const columnName = range.Cells(1, i).Value;
            const distance = levenshteinDistance(pattern, columnName);
            if (distance < minDistance) {
                minDistance = distance;
                columnIndex = i;
            }
        }
    } else {
        columnIndex = cell.Column;
    }
    return columnIndex;
}


// 2.0版本里支持缓存模式的查询
const findColumn2 = Object.assign(
    function (pattern, range = ActiveSheet.UsedRange, cache = true) {
        // 定义内部缓存
        const sheetName = range.Parent.Name;
        const rangeAddress = range.Address;
        const cacheKey = `${sheetName}-${rangeAddress}-${pattern}`;

        // 检查缓存命中
        if (cache && findColumn2._cache[cacheKey] !== undefined) {
            return findColumn2._cache[cacheKey];
        }

        // 查找列逻辑
        let cell = findCell(pattern, range);  // 精确匹配
        let columnIndex;

        if (!cell) {  // 模糊匹配
            let minDistance = Infinity;
            for (let i = 1; i <= range.Columns.Count; i++) {
                const columnName = range.Cells(1, i).Value;
                const distance = levenshteinDistance(pattern, columnName);
                if (distance < minDistance) {
                    minDistance = distance;
                    columnIndex = i;
                }
            }
        } else {
            columnIndex = cell.Column;
        }

        // 若启用缓存，存入缓存
        if (cache) {
            findColumn2._cache[cacheKey] = columnIndex;
        }

        return columnIndex;
    },
    {
        // 缓存对象
        _cache: {},

        // 清除缓存方法
        clearCache() {
            this._cache = {};
        }
    }
);


// 判断一个 cells 集合是否为空
function isEmpty(cells) {
    for (let i = 1; i <= cells.Count; i++) {
        if (cells.Item(i).Text) {
            return false;
        }
    }
    return true;
}

// 根据提供的 pattern 在 range 中寻找 row
// 如果没有提供 range，默认在 ActiveSheet.UsedRange 中寻找
function findRow(pattern, range = ActiveSheet.UsedRange) {
    const cell = findCell(pattern, range);
    if (cell) {
        return cell.Row;
    }
}

// 获取实际使用的区域
function getUsedRange(sheet = ActiveSheet, maxRows = 500, maxColumns = 100, startFromA1 = true) {
    /* 允许通过"表格上下文"信息，调整这里数据行的上限500行，或者列上限100列
        注意，如果分析预设的表格数据在这个限定参数内可以不改
        只有表格未知，或者明确数据量超过设置时，需要重新调整这里的参数
        调整的时候千万不要故意凑的刚刚好，可以设置一定的冗余区间
        比如数据说有4101条，那么这里阈值设置为5000也是可以的，比较保险。
    */

    // 默认获得的区间，有可能是有冗余的空行，所以还要进一步优化
    let usedRange = sheet.UsedRange;

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
    let newUsedRange = sheet.Range(
        usedRange.Cells(firstRow, firstColumn),
        usedRange.Cells(lastRow, lastColumn)
    );

    return newUsedRange;  // 返回新的实际数据区域
}

function __2_json数据导入导出() {

}

// 打包sheet下多个字段fields的数据
// 使用示例：packTableDataFields('料理', ['名称', '标签'], 4)
//  fields的参数支持字段名称或整数，明确指定某列的位置
// 返回格式：{'名称': [x1, x2, ...], '标签': [y1, y2, ...]}
function packTableDataFields(sheet, fields, dataStartRow, dataEndRow, filterEmptyRows = true) {
    // 1 数据范围
    let st = sheet;
    if (typeof sheet === 'string') {
        st = Sheets(sheet);
    }
    const urg = getUsedRange(st);

    // 2 初始化字段列映射表
    const fieldColMap = fields.reduce((map, field) => {
        if (typeof field === 'string') {
            map[field] = findColumn(field, urg);
            if (dataStartRow === undefined) {
                // 默认按找到的第1个字段名的下一行为数据起始行
                dataStartRow = findRow(field) + 1;
            }
        } else if (typeof field === 'number') {
            // 注意，整数作为键会自动转为str，但使用的时候整数索引也会自动转str索引能对上。
            //  主要是了解这原理注意重名覆盖问题。
            map[field] = field;
        }
        return map;
    }, {});
    if (dataStartRow === undefined) {
        dataStartRow = 2;  // 数据默认从第2行开始
    }

    // 3 构建字段格式数据
    // 先建好空数组
    const fieldsData = fields.reduce((dataMap, field) => {
        dataMap[field] = [];
        return dataMap;
    }, {});

    // 遍历数据行
    dataEndRow = dataEndRow || (urg.Row + urg.Rows.Count - 1);
    for (let row = dataStartRow; row <= dataEndRow; row++) {
        // 如果需要过滤空行
        if (filterEmptyRows) {
            const isEmptyRow = Object.values(fieldColMap).every(col => st.Cells(row, col).Value2 === undefined);
            if (isEmptyRow) continue; // 跳过空行
        }

        // 填充字段数据
        Object.entries(fieldColMap).forEach(([field, col]) => {
            fieldsData[field].push(st.Cells(row, col).Value2);
        });
    }

    return fieldsData;
}

// 和packTableDataFields仅差在返回的数据格式上，这个版本的返回值是通常使用更多的jsonl格式
// 返回格式：list[dict]， [{'名称': x1, '标签': y1}, {'名称': x2, '标签': y2}, ...]
function packTableDataList(sheet, fields, dataStartRow, dataEndRow) {
    const fieldsData = packTableDataFields(sheet, fields, dataStartRow, dataEndRow);
    const rowCount = fieldsData[fields[0]].length;
    const listData = [];

    // 将紧凑格式转换为列表字典格式
    for (let i = 0; i < rowCount; i++) {
        const rowDict = {};
        fields.forEach(field => {
            rowDict[field] = fieldsData[field][i];
        });
        listData.push(rowDict);
    }
    return listData;
}

function clearSheetData(headerRow = 1, dataStartRow = 2, sheet = ActiveSheet) {
    let headerStartRow, headerEndRow, dataEndRow;

    // 检查 headerRow 参数，-1 表示不处理表头
    if (headerRow === -1) {
        headerStartRow = headerEndRow = null;
    } else if (typeof headerRow === 'number') {
        headerStartRow = headerEndRow = headerRow;
    } else if (Array.isArray(header)) {
        [headerStartRow, headerEndRow] = headerRow;
    }

    // 检查 dataStartRow 参数，-1 表示不处理数据区域
    if (dataStartRow === -1) {
        dataStartRow = dataEndRow = null;
    } else if (typeof dataStartRow === 'number') {
        let usedRange = sheet.UsedRange;
        dataEndRow = usedRange.Row + usedRange.Rows.Count - 1;
    } else if (Array.isArray(dataStartRow)) {
        [dataStartRow, dataEndRow] = dataStartRow;
    }

    // 清空表头区域（保留格式），若未设置为 -1
    if (headerStartRow !== null && headerEndRow !== null) {
        sheet.Rows(headerStartRow + ':' + headerEndRow).ClearContents();
    }

    // 删除数据区域（不保留格式），若未设置为 -1
    if (dataStartRow !== null && dataEndRow !== null) {
        sheet.Rows(dataStartRow + ':' + dataEndRow).Clear();
    }
}

// 将py里df.to_dict(orient='split')的数据格式写入sheet
// 这个数据一般有3个属性：index, columns, data
function writeDfSplitDictToSheet(jsonData, headerRow = 1, dataStartRow = 2, sheet = ActiveSheet) {
    let columns = jsonData.columns || [];
    let data = jsonData.data || [];

    // 若存在 index，则将其添加为第一列
    if (jsonData.index) {
        columns = ['index', ...columns];
        data = jsonData.index.map((idx, i) => [idx, ...data[i]]);
    }

    const startCol = sheet.UsedRange.Column;

    // 写入表头
    if (headerRow > 0) {
        for (let j = 0; j < columns.length; j++) {
            sheet.Cells(headerRow, startCol + j).Value2 = columns[j];
        }
    }

    // 写入数据内容
    for (let i = 0; i < data.length; i++) {
        for (let j = 0; j < data[i].length; j++) {
            sheet.Cells(dataStartRow + i, startCol + j).Value2 = data[i][j];
        }
    }
}


function writeArrToSheet(arr, startCel) {
    // 遍历数组，将每行的数据写入 Excel
    for (let i = 0; i < arr.length; i++) {
        const row = arr[i];
        // 如果当前行存在，则遍历该行的元素
        if (Array.isArray(row)) {
            for (let j = 0; j < row.length; j++) {
                startCel.Offset(i, j).Value2 = row[j];
            }
        }
    }
}

function insertNewDataWithHeaders(jsonData, headerRow = 1, dataStartRow = 2, sheet = ActiveSheet) {
    // 1 预处理 index，将其合并到 columns 和 data
    let columns = jsonData.columns || [];
    let data = jsonData.data || [];
    if (jsonData.index) {
        columns = ['index', ...columns];
        data = jsonData.index.map((idx, i) => [idx, ...data[i]]);
    }

    // 2 处理可能出现的新字段
    // 获取现有的表头
    let existingHeaders = [];
    const usedRange = sheet.UsedRange;
    for (let col = usedRange.Column; col <= usedRange.Column + usedRange.Columns.Count - 1; col++) {
        existingHeaders.push(sheet.Cells(headerRow, col).Value2);
    }

    // 计算新增的字段
    const newHeaders = columns.filter(column => !existingHeaders.includes(column));
    const allHeaders = [...existingHeaders, ...newHeaders];

    // 如果有新字段，扩展表头
    if (newHeaders.length > 0) {
        for (let j = 0; j < allHeaders.length; j++) {
            sheet.Cells(headerRow, usedRange.Column + j).Value2 = allHeaders[j];
        }
    }

    // 构建插入数据的映射关系
    const headerIndexMap = {};
    for (let j = 0; j < allHeaders.length; j++) {
        headerIndexMap[allHeaders[j]] = usedRange.Column + j;
    }

    // 3 插入新行
    const newDataRows = data.length;
    const insertEndRow = dataStartRow + newDataRows - 1;
    if (insertEndRow >= dataStartRow) {
        sheet.Rows(`${dataStartRow}:${insertEndRow}`).Insert(xlUp);
    }

    for (let i = 0; i < data.length; i++) {
        const rowData = data[i];
        for (let j = 0; j < columns.length; j++) {
            const columnName = columns[j];
            const columnIndex = headerIndexMap[columnName];
            sheet.Cells(dataStartRow + i, columnIndex).Value2 = rowData[j];
        }
    }
}


function __3_py服务工具箱() {

}

function runPyScript(script, query = '', host = '{{JSA_POST_DEFAULT_HOST}}') {
    const url = `{{JSA_POST_HOST_URL}}/${host}/common/run_py`;
    const resp = HTTP.post(url, {query, script}, {
        headers: {
            'Authorization': 'Bearer {{JSA_POST_TOKEN}}',
            'Content-Type': 'application/json'
        }
    });
    return resp.json().output;
}

function runIsolatedPyScript(script, host = '{{JSA_POST_DEFAULT_HOST}}') {
    const url = `{{JSA_POST_HOST_URL}}/${host}/common/run_isolated_py`;
    // 判断 script 的类型: script可以只输入py代码，也可以输入配置好的整个字典数据
    const payload = typeof script === 'string' ? {script} : script;
    const resp = HTTP.post(url, payload, {
        headers: {
            'Authorization': 'Bearer {{JSA_POST_TOKEN}}',
            'Content-Type': 'application/json'
        }
    });
    return resp.json();
}


function getPyTaskResult(taskId, retries = 1, host = '{{JSA_POST_DEFAULT_HOST}}', delay = 5000) {
    const url = `{{JSA_POST_HOST_URL}}/${host}/common/get_task_result/${taskId}`;
    for (let attempt = 0; attempt < retries; attempt++) {
        const resp = HTTP.get(url, {
            headers: {
                'Authorization': 'Bearer {{JSA_POST_TOKEN}}',
                'Content-Type': 'application/json'
            }
        });
        const jsonResp = resp.json();
        if ('__get_task_result_error__' in jsonResp) {
            console.log(`第 ${attempt + 1} 次获取任务结果失败，请稍后再试...`);
            if (attempt < retries - 1) {
                timeSleep(delay);
            }
        } else {
            return jsonResp;
        }
    }
}


function __4_日期处理() {
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
    const excelEpochStart = new Date(Date.UTC(1899, 11, 30));
    const jsDate = new Date(excelEpochStart.getTime() + excelDate * 86400000);
    if (adjustTimezone) {
        const timezoneOffset = jsDate.getTimezoneOffset() * 60000;
        return new Date(jsDate.getTime() + timezoneOffset);
    }
    return jsDate;
}

// 将JS日期转为excel时间戳
function jsDateToExcelDate(jsDate = new Date(), adjustTimezone = true) {
    let adjustedDate = jsDate;
    if (adjustTimezone) {
        const timezoneOffset = jsDate.getTimezoneOffset() * 60000;
        adjustedDate = new Date(jsDate.getTime() - timezoneOffset);
    }
    const excelEpochStart = new Date(Date.UTC(1899, 11, 30));
    const diffInMs = adjustedDate.getTime() - excelEpochStart.getTime();
    return diffInMs / 86400000;
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


// 正常应该下面这样写就行了。但是jsa1.0运行这个会报错，jsa2.0可以运行但没效果。就用了笨办法来做。
// setTimeout(function () {}, milliseconds);
function timeSleep(milliseconds) {
    const startTime = new Date();
    while (new Date() - startTime < milliseconds) {
    }
}
