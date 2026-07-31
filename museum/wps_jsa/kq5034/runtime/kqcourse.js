let 课程标准名 = Application.ActiveWorkbook.Name
    .replace(/\.[^.]+$/, '')  // 去掉文件名后缀
    .replace(/^\d{2}(\d{6})/, 'd$1')  // 文件名开头'2025'改为'd25'
    .replace(/\./g, '点')  // 剩下的'.'改成'点'字
    .replace(/,/g, '')  // 删除所有英文逗号

function main() {
    // 1 这里填上要支持的api、智能匹配函数接口清单
    const funcsMap = {
        findCol,
        writeArrToSheet,
        locateTableRange,
        sqlSelect,
        releaseMutexLock,
        自动填充考勤表日期,
        批量优化条件格式,
        数字加前缀,
        更新订单匹配,
        更新用户匹配,
        getStatus,
        setStatus,
        get禅宗周次,
        step3_计算视频应返款_觉观念住,
        step3_计算视频应返款_梵呗初阶,
        step3_计算视频应返款_梵呗增益,
        step3_计算视频应返款_禅宗,
        step3_计算视频应返款_念住闯关,
        修正五阶打卡应返款公式,
        step5_更新已返款,
        更新修订
    }

    // 2 api：py-jsa脚本令牌模式永远是最高匹配优先级
    if (Context.argv.funcName) return funcsMap[Context.argv.funcName](...(Context.argv.args || []))

    // 3 自定义：也可以注释掉3里的执行部分，下面这里手动写要执行的函数
    // return 自动填充考勤表日期()
    // return 批量优化条件格式()
    // return 设置课次超链接()

    // return 更新修订()

    // 4 智能匹配，优先级：可以在第1个字符串自定义要运行的功能 > 选中单元格指定函数名 > 选中单元格所在第1列是触发函数名
    let funcName = '' || Selection.Cells(1, 1).Value2
    if (!funcsMap[funcName]) funcName = ActiveSheet.Cells(Selection.Row, 1)
    if (funcsMap[funcName]) return funcsMap[funcName]()
}

function __1_配置表格() {

}

/**
 * 计算结束日期
 * @param {string} strStart - 开始日期，格式为"yyyy年m月d日"
 * @param {number} intervalDays - 间隔天数
 * @return {string} 结束日期，格式为"yyyy年m月d日"
 */
function _calculateEndDate(strStart, intervalDays) {
    // 解析开始日期
    const [year, month, day] = strStart.match(/(\d{4})年(\d{1,2})月(\d{1,2})日/).slice(1).map(Number)

    const startDate = new Date(year, month - 1, day)
    const endDate = new Date(startDate.getTime() + intervalDays * 24 * 60 * 60 * 1000)

    return `${endDate.getFullYear()}年${endDate.getMonth() + 1}月${endDate.getDate()}日`
}

/**
 * 生成日期范围数组
 * @param {string} strStart - 开始日期，格式为"yyyy年m月d日"
 * @param {string} strEnd - 结束日期，格式为"yyyy年m月d日"
 * @param {number} n - 生成的日期范围数量
 * @return {Array} 日期范围数组，每个元素格式为"yyyy年m月d日~yyyy年m月d日"
 */
function _generateDateRanges(strStart, strEnd, n) {
    // 解析输入的日期字符串
    const [startYear, startMonth, startDay] = strStart.match(/(\d{4})年(\d{1,2})月(\d{1,2})日/).slice(1).map(Number)
    const [endYear, endMonth, endDay] = strEnd.match(/(\d{4})年(\d{1,2})月(\d{1,2})日/).slice(1).map(Number)

    // 获取日期差值
    const startDate = new Date(startYear, startMonth - 1, startDay)
    const endDate = new Date(endYear, endMonth - 1, endDay)
    const deltaDays = (endDate - startDate) / (1000 * 60 * 60 * 24)

    // 使用数组来存储日期范围，保持顺序
    const dateRanges = []

    for (let i = 0; i < n; i++) {
        // 生成新的起始和结束日期
        const newStartDate = new Date(startDate.getTime() + i * 24 * 60 * 60 * 1000)
        const newEndDate = new Date(newStartDate.getTime() + deltaDays * 24 * 60 * 60 * 1000)

        // 格式化日期为输出格式
        const formattedStart = `${newStartDate.getMonth() + 1}月${newStartDate.getDate()}日`
        const formattedEnd = `${newEndDate.getMonth() + 1}月${newEndDate.getDate()}日`

        // 添加到数组中
        dateRanges.push(`${formattedStart}~${formattedEnd}`)
    }

    // 返回生成的日期范围
    return dateRanges
}


/**
 * 填充考勤表的日期范围
 * @param {string} strStart - 开始日期，格式为"m月d日"
 * @param {number} intervalDays - 间隔天数
 * @param {number} [n] - 课程数量，可选参数
 */
function _填充考勤表日期范围(strStart, intervalDays, n) {
    // 定位表格范围，使用默认值
    const [ur, rows, cols] = locateTableRange('考勤表', 4)

    // 查找第1课的单元格位置
    const firstLessonCell = findCel('第01课', ur, xlPart)

    if (!firstLessonCell) {
        console.log('未找到第1课单元格')
        return
    }

    // 如果n未提供，则使用表格的列数作为课程数量
    if (n === undefined) {
        n = ur.Columns.Count
    }

    // 计算结束日期
    const strEnd = _calculateEndDate(strStart, intervalDays)

    // 生成日期范围
    const dateRanges = _generateDateRanges(strStart, strEnd, n)

    // 填充日期范围到表格
    for (let i = 0; i < n; i++) {
        ur.Cells(firstLessonCell.Row - 1, firstLessonCell.Column + i).Value2 = dateRanges[i]
    }
}

function _formatDateString(str) {
    const regex = /d(\d{6})/; // 匹配d后接6位数字的部分
    const match = str.match(regex);

    if (!match) return ''; // 如果未匹配到，返回空字符串或进行其他处理

    const datePart = match[1]; // 提取6位数字
    const year = `20${datePart.substr(0, 2)}`; // 组合年份
    const month = parseInt(datePart.substr(2, 2), 10); // 解析月份并去除前导零
    const day = parseInt(datePart.substr(4, 2), 10); // 解析日期并去除前导零

    return `${year}年${month}月${day}日`;
}

/**
 * 自动填充考勤表日期
 * @param {string} strStart - 开始日期，格式为"yyyy年m月d日"
 */
function 自动填充考勤表日期(strStart) {
    strStart = strStart || _formatDateString(课程标准名)

    let intervalDays, lessonCount

    if (课程标准名.includes('觉观')) {
        intervalDays = 5 - 1
        lessonCount = 21
    } else if (课程标准名.includes('念住')) {
        intervalDays = 7 - 1
        lessonCount = 21
    } else if (课程标准名.includes('梵呗初阶')) {
        intervalDays = 5
        lessonCount = 11
    } else if (课程标准名.includes('梵呗增益')) {
        // todo 梵呗增益最后两课还要手动调整成无回放模式
        intervalDays = 5
        lessonCount = 22
    } else {
        console.error('无法识别工作簿类型，请检查工作簿名称')
        return
    }

    // 解析开始日期
    const [year, month, day] = strStart.match(/(\d{4})年(\d{1,2})月(\d{1,2})日/).slice(1).map(Number)

    // 调用_填充考勤表日期范围函数
    _填充考勤表日期范围(strStart, intervalDays, lessonCount)

    if (课程标准名.includes('觉观') || 课程标准名.includes('念住')) {
        const [ur, rows, cols] = locateTableRange('考勤表', 4)
        const firstLessonCell = findCel('第01课', ur, xlPart)

        if (firstLessonCell) {
            // 计算第22课的日期
            const lesson22Date = new Date(year, month - 1, day + 21)
            // 格式化第22课日期
            const formattedDate22 = `${lesson22Date.getFullYear()}/${lesson22Date.getMonth() + 1}/${lesson22Date.getDate()}`

            // 填写第22课日期
            ur.Cells(firstLessonCell.Row - 1, firstLessonCell.Column + 21).Value2 = formattedDate22

            // 如果是念住，还要填写第23课日期
            if (课程标准名.includes('念住')) {
                // 计算第23课的日期
                const lesson23Date = new Date(year, month - 1, day + 22)
                // 格式化第23课日期
                const formattedDate23 = `${lesson23Date.getFullYear()}/${lesson23Date.getMonth() + 1}/${lesson23Date.getDate()}`

                // 填写第23课日期
                ur.Cells(firstLessonCell.Row - 1, firstLessonCell.Column + 22).Value2 = formattedDate23
            }
        } else {
            console.log('未找到第1课单元格，无法填充日期')
        }
    }
}


function 批量优化条件格式() {
    const sheetNames = ['考勤表', '报名表']
    for (const sheetName of sheetNames) {
        const sheet = Sheets(sheetName)
        extendFormatConditionsToFullColumns(sheet)
        // todo 设置超链接后单元格底色会没掉，需要再补充设置
    }
}

function 设置课次超链接() {
    // 1 从py后台取到名称、链接数据
    let pyScript = `
from xlsln.kq5034.kqmain import 获取课程链接
return 获取课程链接('${课程标准名}')
`
    const jsonData = runIsolatedPyScript(pyScript)

    // 2 定位到"返款配置"后面一列，开始设置名称、链接
    // 用findCol找出第2行'返款配置'所在列，加1就是打开开始配置列
    let ws = Sheets('考勤表')
    let startCol = findCol('返款配置', ws.Rows(2),) + 1

    // 遍历clockins，设置名称
    for (let i = 0; i < jsonData.clockins.length; i++) {
        let name = jsonData.clockins[i].name
        let url = jsonData.clockins[i].url
        setHyperlink(ws.Cells(2, startCol + i), url, name)
    }

    // 3 继续往右列配置，接下来就是课程的名称和链接
    for (let i = 0; i < jsonData.lessons.length; i++) {
        let name = jsonData.lessons[i].name
        let url = jsonData.lessons[i].url

        // 如果name包含'='，需要只显示第一个=右边的内容，比如 '第1周=佛教概观1-1' -> '佛教概观1-1'
        const equalIndex = name.indexOf('=');
        if (equalIndex !== -1) {
            name = name.slice(equalIndex + 1);
        }

        setHyperlink(ws.Cells(2, startCol + jsonData.clockins.length + i), url, name)
    }
}


function __2_匹配数据() {

}


function 数字加前缀() {
    const [ur, rows, cols] = locateTableRange('报名表', 4)

    for (let name of ['手机号', '微信支付订单号']) {
        const j = cols[name]
        for (let i = rows.start; i <= rows.end; i++) {
            let val = ur.Cells(i, j).Text
            // 如果val以单引号开头，则去掉单引号
            if (val.startsWith("'")) val = val.slice(1)

            // 根据模式执行操作
            if (name === '微信支付订单号') {
                if (!/^`/.test(val)) ur.Cells(i, j).Value2 = '`' + val
            } else {
                ur.Cells(i, j).Value2 = "'" + val
            }
        }
    }
}

function 更新订单匹配() {
    // 1 定位
    const taskStatusCel = findCel('商户订单号', '报名表').Offset(1, 0)
    const status = setMutexLock(taskStatusCel)

    // 2 异步执行
    if (status) {
        const pyScript = `
from xlsln.kq5034.courses.${课程标准名} import 考勤课程
kq = 考勤课程()
kq.更新订单匹配()
`
        runIsolatedPyScript({script: pyScript, long_task: true}, 'codepc_mi15')
    }
}

function 更新用户匹配() {
    // 1 定位
    const taskStatusCel = findCel('商户订单号', '报名表').Offset(1, 0)
    const status = setMutexLock(taskStatusCel)

    // 2 异步执行
    if (status) {
        const pyScript = `
from xlsln.kq5034.courses.${课程标准名} import 考勤课程
kq = 考勤课程()
kq.更新用户匹配()
`
        runIsolatedPyScript({script: pyScript, long_task: true}, 'codepc_mi15')
    }
}

function __3_日常考勤() {

}

function getStatus() {
    return findCel('返款配置', Sheets('考勤表').UsedRange).Offset(1, 0).Value2
}


function setStatus(status, offsetDays = 0) {
    // 1 格式化时间
    let date = new Date()  // 默认当前时间
    if (offsetDays) date.setDate(date.getDate() + Number(offsetDays))

    let tag = `最近运行更新时间：\n${formatLocalDatetime(date)}`
    tag += `,${status}`

    // 2 在指定位置填写状态
    findCel('返款配置', Sheets('考勤表').UsedRange).Offset(1, 0).Value2 = tag
    return tag
}

function get禅宗周次() {
    return findCel('当前应返款', Sheets('考勤表').Rows(2)).Offset(1, 0).Value2
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
    refundDict['回放'] = 0  // 其他未明确标记的带'回放'字眼的一律返款金额为0
    return refundDict
}

function highlightCourseProgress(refundDict, cell) {
    let color, refundAmount, text
    text = cell.Text

    // 1 找到redundDict字典中最大值
    const sortedEntries = Object.entries(refundDict).sort((a, b) => b[1] - a[1])  // 确保按照值从大到小排序
    const maxRefund = sortedEntries[0][1] || 0
    const secondRefund = sortedEntries.length > 1 ? sortedEntries[1][1] : maxRefund

    // 2 遍历refundDict中的所有key，判断cell的文本值是否包含了对应key，存储对应的refundAmount值
    for (const [key, value] of sortedEntries) {
        if (text.includes(key)) {
            refundAmount = value
            break
        }
    }

    // 3 根据refundAmout设置基础颜色：与maxRefund相等设置绿色，正值设为黄色，0值设为灰色，undefined设为白色
    if (refundAmount === maxRefund) {
        color = [0, 255, 0]  // 绿色
    } else if (refundAmount > 0) {
        color = [255, 255, 0]  // 黄色
        // 黄色情况下，要根据refundAmount权重，淡化颜色
        color[2] = (1 - refundAmount / secondRefund) * 128
    } else if (refundAmount === 0) {
        color = [128, 128, 128]  // 灰色
    } else {
        color = [255, 255, 255]  // 白色
    }

    // 4 根据完成进度再进行一轮颜色渲染
    // 提取百分比
    const weight = parseFloat(text.match(/\d*%/)?.pop()) || 100;

    // 颜色淡化
    for (let i = 0; i < 3; i++) color[i] = (color[i] * weight + 255 * 100) / (weight + 100)

    // 设置颜色
    cell.Interior.Color = RGB(color[0], color[1], color[2])

    // 5 返回返款金额
    return refundAmount || 0
}


function highlightZenStageProgress(cell) {
    const text = cell.Text || ''
    if (text.includes('准时完成')) {
        cell.Interior.Color = RGB(0, 255, 0)
        return true
    }
    if (text.includes('延') && text.includes('完成')) {
        cell.Interior.Color = RGB(192, 192, 192)
        return false
    }

    // 修道班未完整看完时只有观察进度，没有局部返款，也不显示进度底色。
    cell.Interior.ColorIndex = -4142  // xlColorIndexNone
    return false
}


function step3_计算视频应返款_觉观念住(start_row, end_row) {
    const [ur, rows, cols] = locateTableRange('考勤表', 4, ['视频应返款'])
    // 如果start_row, end_row未传入，则默认使用rows.start, rows.end
    if (!start_row) start_row = rows.start
    if (!end_row) end_row = rows.end

    const refundDict = parseRefundRules(findCel('视频应返款', ur).Offset(1, 0).Text)
    cols['第01课'] = findCol('第01课', ur.Rows(2), xlPart)
    cols['第21课'] = findCol('第21课', ur.Rows(2), xlPart)
    for (let i = start_row; i <= end_row; i++) {
        let totalRefund = 0
        for (let j = cols['第01课']; j <= cols['第21课']; j++)
            totalRefund += highlightCourseProgress(refundDict, ur.Cells(i, j))
        ur.Cells(i, cols['视频应返款']).Value2 = totalRefund
    }
}

function step3_计算视频应返款_梵呗初阶(start_row, end_row) {
    const [ur, rows, cols] = locateTableRange('考勤表', 4, ['视频应返款'])
    if (!start_row) start_row = rows.start
    if (!end_row) end_row = rows.end

    const refundDict = parseRefundRules(findCel('视频应返款', ur).Offset(1, 0).Text)
    cols['第01课'] = findCol('第01课', ur.Rows(2), xlPart)
    cols['第11课'] = findCol('第11课', ur.Rows(2), xlPart)
    for (let i = start_row; i <= end_row; i++) {
        let totalRefund = 0
        for (let j = cols['第01课']; j <= cols['第11课']; j++)
            totalRefund += highlightCourseProgress(refundDict, ur.Cells(i, j))
        ur.Cells(i, cols['视频应返款']).Value2 = totalRefund
    }
}


function step3_计算视频应返款_梵呗增益(start_row, end_row) {
    const [ur, rows, cols] = locateTableRange('考勤表', 4, ['视频应返款'])
    if (!start_row) start_row = rows.start
    if (!end_row) end_row = rows.end

    const refundDict = parseRefundRules(findCel('视频应返款', ur).Offset(1, 0).Text)
    cols['第01课'] = findCol('第01课', ur.Rows(2), xlPart)
    cols['第22课'] = findCol('第22课', ur.Rows(2), xlPart)
    for (let i = start_row; i <= end_row; i++) {
        let totalRefund = 0
        for (let j = cols['第01课']; j <= cols['第22课']; j++)
            totalRefund += highlightCourseProgress(refundDict, ur.Cells(i, j))
        ur.Cells(i, cols['视频应返款']).Value2 = totalRefund
    }
}

function step3_计算视频应返款_禅宗系列(start_row, end_row, price, 起始课程名, 终止课程名) {
    // 暂时为 禅宗4阶 专门定制的，之后有其他禅宗课记得要更精细的划分
    const [ur, rows, cols] = locateTableRange('考勤表', 4, ['视频应返款'])
    if (!start_row) start_row = rows.start
    if (!end_row) end_row = rows.end

    cols['start'] = findCol(起始课程名, ur.Rows(2), xlPart)
    cols['end'] = findCol(终止课程名, ur.Rows(2), xlPart)
    for (let i = start_row; i <= end_row; i++) {
        let totalRefund = 0
        for (let j = cols['start']; j <= cols['end']; j++) {
            if (highlightZenStageProgress(ur.Cells(i, j))) {
                let coursePrice = price; // 默认使用传入的price

                // 1 为了兼容禅宗4.5阶，阿毗达磨系列是一个课拆成多个小课，返款额度需要特殊处理
                // 在第2行的值如果前缀是'阿毗达摩概论'，且第3行的值是一个非空整数值，则以这个整数作为price
                let courseName = ur.Cells(2, j).Text;
                if (courseName.indexOf('阿毗达磨概论') === 0) {
                    // 获取第3行的显示文本和实际值
                    let cellText = ur.Cells(3, j).Text;
                    let specialPrice = ur.Cells(3, j).Value2;

                    // 修改后的判断逻辑：
                    // 1. 文本中不能包含'日' (排除类似 "10月24日" 的日期格式)
                    // 2. 实际值必须是数字且大于0
                    if (cellText.indexOf('日') === -1 && typeof specialPrice === 'number' && specialPrice > 0) {
                        coursePrice = specialPrice;
                    }
                }

                // 2 否则其他一般情况处理
                totalRefund += coursePrice;
            }
        }
        ur.Cells(i, cols['视频应返款']).Value2 = totalRefund
    }
}


function step3_计算视频应返款_禅宗(start_row, end_row) {
    if (课程标准名.includes('一阶')) {
        step3_计算视频应返款_禅宗系列(start_row, end_row, 27, '佛教概观1', '神经心理学4')
    } else if (课程标准名.includes('二三阶')) {
        step3_计算视频应返款_禅宗系列(start_row, end_row, 25, '人天福德门1', '禅宗学导讲7')
    } else if (课程标准名.includes('二阶')) {
        step3_计算视频应返款_禅宗系列(start_row, end_row, 20, '佛教概观5', '文化心理学3')
    } else if (课程标准名.includes('三阶')) {
        step3_计算视频应返款_禅宗系列(start_row, end_row, 18, '义理堂6', '中国佛教史7')
    } else if (课程标准名.includes('四阶')) {
        step3_计算视频应返款_禅宗系列(start_row, end_row, 15, '阿含经导读1', '基础止观39')
    }  else if (课程标准名.includes('4点5阶')) {
        step3_计算视频应返款_禅宗系列(start_row, end_row, 17, '佛教概观1', '佛教概观4+')
    } else if (课程标准名.includes('五阶')) {
        step3_计算视频应返款_禅宗系列(start_row, end_row, 15, '佛教史1', '法蕴论智分别法心分别')
    }

}

/**
 * 五阶共学、共修各最多计 16 次，每次 5 元，打卡返款合计封顶 160 元。
 * 公式在 AirScript 内部构造，避免通过 Value2 或外部参数把公式写成文本。
 */
function 修正五阶打卡应返款公式() {
    const [ur, rows, cols] = locateTableRange(
        '考勤表',
        4,
        ['打卡应返款', '共学打卡', '共修打卡'],
    )
    const noteCell = findCel('打卡应返款', ur).Offset(1, 0)
    noteCell.Value2 = '全期封顶：共学16次×5元 + 共修16周×5元，打卡最多返160元。'

    const samples = []
    for (let i = rows.start; i <= rows.end; i++) {
        const cell = ur.Cells(i, cols['打卡应返款'])
        const row = cell.Row
        cell.NumberFormat = 'G/通用格式'
        cell.Formula = `=MIN(P${row},16)*5+MIN(Q${row},16)*5`
        if (i === rows.start || i === rows.end || i === Math.floor((rows.start + rows.end) / 2)) {
            samples.push({row, formula: cell.Formula, value: cell.Value2})
        }
    }
    return {updated: rows.end - rows.start + 1, samples}
}

function step3_计算视频应返款_念住闯关(start_row, end_row) {
    const [ur, rows, cols] = locateTableRange('考勤表', 4, ['视频应返款'])
    if (!start_row) start_row = rows.start
    if (!end_row) end_row = rows.end

    cols['第01课'] = findCol('第01课', ur.Rows(2), xlPart)
    cols['第17届答疑'] = findCol('第17届答疑', ur.Rows(2), xlPart)
    for (let i = start_row; i <= end_row; i++) {
        let totalRefund = 0
        let score = 0
        for (let j = cols['第01课']; j <= cols['第17届答疑']; j++) {
            if (j >= cols['第01课'] + 11) { // 从第12课开始处理课次和答疑的打包
                let refund1 = highlightCourseProgress({'3遍': 3, '2遍': 2, '1遍': 1}, ur.Cells(i, j))
                let refund2 = highlightCourseProgress({'遍': 1}, ur.Cells(i, j + 1))
                // totalRefund += refund1  // 250301周六11:11修改，答疑不影响返款
                // 250609周一修改，只要完成一遍就返款全部20元
                if (refund1 > 0) {
                    totalRefund += 20
                    score += refund1 - 1
                }
                j++ // 跳过下一列的答疑
            } else {
                let refund1 = highlightCourseProgress({'3遍': 3, '2遍': 2, '1遍': 1}, ur.Cells(i, j))
                if (refund1 > 0) totalRefund += 20
            }
        }
        ur.Cells(i, cols['视频应返款']).Value2 = totalRefund
        ur.Cells(i, cols['优秀学员评分']).Value2 = score
    }
}


function step5_更新已返款() {
    const [ur, rows, cols] = locateTableRange('考勤表', 4, ['已返款', '当前应返款', '返款配置'])
    for (let i = rows.start; i <= rows.end; i++) {
        const v1 = ur.Cells(i, cols['当前应返款']).Value2
        const v2 = ur.Cells(i, cols['返款配置']).Text
        if (v1 > 0 && v2 !== '') ur.Cells(i, cols['已返款']).Value2 += v1
    }
}


function __4_修复工具() {

}

function 修订单条考勤数据(cel, userIdCol=6) {
    // 1 获取参数信息

    // 获取单元格所在的行和列
    const r = cel.Row
    const c = cel.Column

    // 找到同行'用户ID'对应的值
    const ws = cel.Worksheet
    // const userIdCol = findCol('用户ID', ws.Rows(2))  // 不能用find，否则会覆盖上一次find查找'修订*'的配置
    const userId = ws.Cells(r, userIdCol).Value2
    
    // 还有同列对应的第1、2行单元格的值(表头)
    // 但是第1行还比较特别，如果是合并单元格，要正确取到合并单元格开头的值
    
    // 获取第1行对应列的单元格
    const header1Cell = ws.Cells(1, c)
    
    // 处理第1行合并单元格的情况，获取合并单元格的起始单元格
    let header1Value = header1Cell.Value2
    if (header1Cell.MergeCells) header1Value = header1Cell.MergeArea.Cells(1, 1).Value2
    
    // 获取第2行对应列的单元格值
    const header2Value = ws.Cells(2, c).Value2

    // 2 执行修订函数
    let status = 0  // 课程完成状态标记
    // 正则提取c.Text中的数值，如果确实有，则覆盖默认的status的值，否则不要处理
    const match = cel.Text.match(/\d+/)
    if (match) status = parseInt(match[0], 10)
    return runIsolatedPyScript(`
from xlsln.kq5034.courses.${课程标准名} import 考勤课程
kq = 考勤课程()
lesson_name = kq.kqdb.find_lesson_name(['${课程标准名}', '${header1Value}', '${header2Value}'])
kq.kqdb.patch_lesson_data(lesson_name, '${userId}', ${status})
`
    )
}

function 检查处理修订数据() {
    const ws = Sheets('考勤表')
    const ur = ws.UsedRange
    const userIdCol = findCol('用户ID', ws.Rows(2))
    let foundCell = ur.Find('修订', undefined, undefined, xlPart)
    let firstFound = foundCell || null

    // 循环查找所有以'修订'开头的单元格
    while (foundCell) {
        // console.log(foundCell.Address())
        修订单条考勤数据(foundCell, userIdCol)

        // 继续查找下一个
        let nextFound = ur.FindNext(foundCell)        
        // 如果回到了第一个找到的单元格，说明已经找完所有匹配项
        if (nextFound && nextFound.Address() === firstFound.Address()) {
            break
        }
        foundCell = nextFound
    }
}

function 更新修订() {
    // 1 需要先检查是否有需要'修订'的异常数据
    检查处理修订数据()

    // 2 然后再更新数据
    return runIsolatedPyScript({
        script: `
from xlsln.kq5034.courses.${课程标准名} import 考勤课程
kq = 考勤课程()
kq.更新修订()
return {'res': '更新完成'}
        `, long_task: true
    })
}


function __x_工具代码() {
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
function releaseMutexLock(celOrSheetName, argName, rowOffset = 1, colOffset = 0) {
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

// 服务器路径，url
const JSA_POST_HOST_URL = 'https://code4101.com'

// 要处理的目标主机
const JSA_POST_DEFAULT_HOST = 'codepc_mi15'

// 请求的header格式，以及对应的token
const JSA_HTTP_HEADERS = {
    'Authorization': '<REDACTED>',
    'Content-Type': 'application/json'
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
 * @param {string} [screenTip] - 悬浮提示文本（未实测出效果）
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
function extendFormatConditionsToFullColumns(sheet) {
    const formatConditions = sheet.UsedRange.FormatConditions
    for (let i = 1; i <= formatConditions.Count; i++) {
        const condition = formatConditions.Item(i)
        const addr = condition.AppliesTo.Address()
        // 使用正则匹配类似 $J$2:$J$1048576 的格式，变成$J:$J
        // const match = addr.match(/^\$([A-Z]+)\$\d+:\$\1\$\d+$/)
        // $A$2:$B$1048576 的格式，变成$A:$B
        const match = addr.match(/^\$([A-Z]+)\$\d+:\$([A-Z]+)\$\d+$/)
        if (match) {
            // match[1] 是捕获的列字母
            const colLetter = match[1]
            condition.ModifyAppliesToRange(sheet.Range(`${colLetter}:${colLetter}`))
        }
    }
}

function __y_main() {

}

const res = sanitizeForJSON(main())
console.log(res)
// noinspection JSAnnotator
return res
