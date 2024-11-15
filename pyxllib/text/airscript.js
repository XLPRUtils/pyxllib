function __1_定位工具() {

}

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
    const cell = findCell(pattern, range)
    if (cell) {
        return cell.Row
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

function clearSheetData(header = 1, dataStartRow = 2, sheet = ActiveSheet) {
    let headerStartRow, headerEndRow, dataEndRow;

    // 检查 header 参数，-1 表示不处理表头
    if (header === -1) {
        headerStartRow = headerEndRow = null;
    } else if (typeof header === 'number') {
        headerStartRow = headerEndRow = header;
    } else if (Array.isArray(header)) {
        [headerStartRow, headerEndRow] = header;
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

function writeJsonToSheet(jsonData, headerRow = 1, dataStartRow = 2, sheet = ActiveSheet) {
    let columns = jsonData.columns || [];
    let data = jsonData.data || [];

    // 若存在 index，则将其添加为第一列
    if (jsonData.index) {
        columns = ['index', ...columns];
        data = jsonData.index.map((idx, i) => [idx, ...data[i]]);
    }

    const startCol = sheet.UsedRange.Column;

    // 写入表头
    for (let j = 0; j < columns.length; j++) {
        sheet.Cells(headerRow, startCol + j).Value2 = columns[j];
    }

    // 写入数据内容
    for (let i = 0; i < data.length; i++) {
        for (let j = 0; j < data[i].length; j++) {
            sheet.Cells(dataStartRow + i, startCol + j).Value2 = data[i][j];
        }
    }
}


function writeArrToSheet(arr, cel) {
    // 遍历数组，将每行的数据写入 Excel
    for (let i = 0; i < arr.length; i++) {
        const row = arr[i];
        // 如果当前行存在，则遍历该行的元素
        if (Array.isArray(row)) {
            for (let j = 0; j < row.length; j++) {
                cel.Offset(i, j).Value2 = row[j];
            }
        }
    }
}


// 将选中sheetName，及指定字段fields的数据，打包成list[dict]格式
// 使用示例：tableDataToCompactJSON('酒水', ['名称', '标签'], 4)
function tableDataToJSON(sheetName, fields, startRow = null, endRow = null) {
    const sheet = Sheets(sheetName);
    const urg = getUsedRange(sheet);

    // 获取起始和结束行，支持默认算法
    const dataStartRow = startRow || (urg.Find(fields[0]).Row + 1); // 默认为字段列的下一行
    const dataEndRow = endRow || (urg.Row + urg.Rows.Count - 1); // 默认为表格的最后一行

    // 查找各字段对应的列位置
    const fieldColumns = fields.reduce((acc, field) => {
        acc[field] = findColumn(field, urg);
        return acc;
    }, {});

    // 遍历每一行数据并提取指定字段
    const jsonData = [];
    for (let i = dataStartRow; i <= dataEndRow; i++) {
        const rowData = {};
        fields.forEach(field => {
            // 注意：提取单元格的文本内容。注意如果单元格是undefined，好像这个field是自动过滤掉不保存的，下游要注意特殊处理
            rowData[field] = sheet.Cells(i, fieldColumns[field]).Value2;
        });
        jsonData.push(rowData); // 将行数据加入 JSON 数组
    }
    return jsonData;
}

// 返回格式{field1: [x1, x2, ...], field2: [y1, y2, ...]}
function tableDataToCompactJSON(sheetName, fields, startRow = null, endRow = null) {
    const sheet = Sheets(sheetName);
    const urg = getUsedRange(sheet);

    // 获取起始和结束行，默认使用字段列下一行至数据区域最后一行
    const dataStartRow = startRow || (urg.Find(fields[0]).Row + 1);
    const dataEndRow = endRow || (urg.Row + urg.Rows.Count - 1);

    // 查找各字段对应的列位置
    const fieldColumns = fields.reduce((acc, field) => {
        acc[field] = findColumn(field, urg);
        return acc;
    }, {});

    // 创建紧凑的JSON结构
    const compactJSON = fields.reduce((acc, field) => {
        acc[field] = []; // 初始化每个字段的列表
        return acc;
    }, {});

    // 遍历每一行数据并填充到紧凑结构中
    for (let i = dataStartRow; i <= dataEndRow; i++) {
        fields.forEach(field => {
            compactJSON[field].push(sheet.Cells(i, fieldColumns[field]).Value2);
        });
    }
    return compactJSON;
}


function __3_py服务工具箱() {

}

// 调用python后端服务
function runPyScript(script, query = '', host = '{{JSA_POST_DEFAULT_HOST}}') {
    const url = `{{JSA_POST_HOST_URL}}/${host}/common/run_py`;

    try {
        // 使用 HTTP 模块发送 POST 请求
        const response = HTTP.post(url, {query, script}, {
            headers: {
                'Authorization': 'Bearer {{JSA_POST_TOKEN}}',
                'Content-Type': 'application/json'
            }
        });

        // 获取并解析响应
        const output = response.json().output;
        return output;
    } catch (error) {
        return {result: `\n执行报错：\n${error.message}`};
    }
}

function runIsolatedPyScript(script, host = '{{JSA_POST_DEFAULT_HOST}}') {
    const url = `{{JSA_POST_HOST_URL}}/${host}/common/run_isolated_py`;
    try {
        // 判断 script 的类型: script可以只输入py代码，也可以输入配置好的整个字典数据
        const payload = typeof script === 'string' ? {script} : script;
        const response = HTTP.post(url, payload, {
            headers: {
                'Authorization': 'Bearer {{JSA_POST_TOKEN}}',
                'Content-Type': 'application/json'
            }
        });
        return response.json();
    } catch (error) {
        return {result: `\n执行报错：\n${error.message}`};
    }
}


function getPyTaskResult(taskId, retries = 1, host = '{{JSA_POST_DEFAULT_HOST}}', delay = 5000) {
    const url = `{{JSA_POST_HOST_URL}}/${host}/common/get_task_result/${taskId}`;

    for (let attempt = 0; attempt < retries; attempt++) {
        try {
            // 发送 GET 请求获取任务结果
            const response = HTTP.get(url, {
                headers: {
                    'Authorization': 'Bearer {{JSA_POST_TOKEN}}',
                    'Content-Type': 'application/json'
                }
            });

            const result = response.json();

            // 如果任务完成，则返回结果
            if (!result.error) {
                return result;
            }

            console.log(`任务尚未完成，重试第 ${attempt + 1} 次...`);
        } catch (error) {
            console.error(`获取任务结果时出错: ${error.message}`);
        }

        // 如果需要重试，则等待一段时间
        if (attempt < retries - 1) {
            Sleep(delay);
        }
    }

    // 超过重试次数仍未成功
    return {error: "任务超时未完成"};
}


function __4_日期处理() {

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
