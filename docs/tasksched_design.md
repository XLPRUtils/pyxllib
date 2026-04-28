# 持久化协作式任务调度器设计方案

## 1. 背景与目标

现有自动化脚本常见两类场景：

- 考勤、课程、账单等批处理脚本：按固定时间运行若干函数，失败时通常直接抛错停止。
- 手游、GUI、爬虫等长流程自动化：任务可能运行很久，需要在步骤之间让出控制权，并允许前置任务临时插队。

传统 `while True + sleep + if due` 可以快速解决单个脚本，但当任务需要持久化、动态计算下次时间、失败后稍后重试、生成器任务让出控制权、前置任务插队时，业务代码会被调度细节污染。

本方案目标是抽出一个高度独立的调度内核：

- 调度器只认识 `callable`、时间、状态文件、运行实例和错误策略。
- 业务函数自己处理业务逻辑，例如 AnLib、OCR、WPS、微信、课程、账号、页面导航。
- 任务定义、时间策略、运行实例、状态持久化、日志观测互相解耦。

## 2. 核心概念

### 2.1 Task 与 Job

`task` 是配置期的任务定义，描述“这个函数如何被调度”。

`job` 是运行期的一次任务实例，描述“这个 task 当前正在跑的这一次”。

两者必须分开：

- `task` 可以长期存在，并持久化 `next_run_at`、`last_error` 等状态。
- `job` 只在任务运行中存在，可以是普通函数调用，也可以是挂起中的生成器。

### 2.2 状态文件

通用调度器的核心输入是持久化状态文件路径：

```python
scheduler = TaskScheduler(state_path="scheduler_state.json")
```

具体项目可以在领域层提供便捷封装，例如凡修传入业务目录后自动使用 `root / "status.json"`。但这个规则不属于通用调度器。

### 2.3 next_run_at

调度器统一围绕 `next_run_at` 工作：

- `next_run_at <= now`：task 可以创建或恢复 job。
- `next_run_at > now`：task 暂不运行。

`task(default_next_time=...)`、`daily`、`every`、`retry`、`ctx.next_time` 本质上都是在设置下一次 `next_run_at`。

## 3. API 设计

### 3.1 最小使用示例

```python
from pyxllib.prog.tasksched import TaskScheduler

scheduler = TaskScheduler(state_path="data/kq5034_scheduler.json")

scheduler.task(每日早晨例常2).daily("07:30")
scheduler.task(每日晚上例常).daily("22:30")

scheduler.run_forever(tick_seconds=1)
```

`task(func)` 直接注册函数。默认任务名来自函数名，`name` 同时作为展示名和持久化状态键。

需要手动命名时：

```python
scheduler.task(func, name="自定义展示名")
scheduler.task(func, default_next_time="2026-04-26 05:00:00")
scheduler.task(func, default_next_time=lambda s: s.next_time("05:00"))
```

同一个 `name` 不能重复注册。同一个函数如果要重复添加成多个 task，必须显式传不同 `name`；重复使用默认函数名会报错：

```python
scheduler.task(游历, name="羊驼_游历").daily("05:00")
scheduler.task(游历, name="驼二_游历").daily("05:00")
```

### 3.2 首次 next_run_at

注册 task 时，如果状态文件里没有这个 task 的 `next_run_at`，调度器默认写入当前时间：

```python
scheduler.task(日常灵泉).daily("20:29")
```

这表示首次启动时先给 task 一次检查机会。窗口期、场景、CD 是否真的允许执行业务，由 task 自己判断：

```python
def 日常灵泉(ctx):
    if not is_within_time("20:29", "20:35"):
        return ctx.next_time("20:29")

    yield from 执行业务()
    return ctx.next_time("20:29")
```

如果某个 task 首次注册时不想立即检查，可以用 `default_next_time` 显式指定初始时间：

```python
scheduler.task(每日早晨例常2, default_next_time=lambda s: s.next_time("07:30")).daily("07:30")
```

`default_next_time` 只在状态缺失时生效；状态文件里已有 `next_run_at` 时不会覆盖。它可以是具体时间、时间字符串，或接收 scheduler 的 callable。

### 3.3 函数签名

调度器支持两种任务函数：

```python
def 普通任务():
    ...

def 需要上下文的任务(ctx):
    ...
    return ctx.next_time(seconds=60)
```

规则：

- 0 个必填参数：调用 `func()`。
- 1 个必填参数：调用 `func(ctx)`。
- 其他签名：报错，要求业务方自行包一层函数。

### 3.4 时间策略动作链

`task(...)` 只描述任务本身；时间触发逻辑通过动作链配置。

```python
scheduler.task(日常助手).daily("00:00", "05:00", "12:00", "18:00")
scheduler.task(托管重连).every(minutes=5)
scheduler.task(长流程任务).daily("05:00").timeout(minutes=20)
```

`daily(*anchors)` 是多个每日锚点的封装：

```python
min(next_time("00:00"), next_time("05:00"), next_time("12:00"), next_time("18:00"))
```

`every(...)` 表示固定间隔运行。

`timeout(...)` 表示单个 job 的最长允许运行时间。它不改变 task 的触发时间，只保护已经启动的运行实例。

### 3.5 动态下次时间

业务运行后才能知道下次时间时，任务函数通过 `ctx.next_time(...)` 写回调度结果。

```python
def 仙府寻访仙侣(ctx):
    yield from 执行业务流程()

    text = 读取界面倒计时文本()
    cd_seconds = 解析倒计时秒数(text)
    return ctx.next_time(seconds=cd_seconds)
```

`ctx.next_time` 的语义应与现有 `next_time` 工具函数对齐：

```python
ctx.next_time("05:00")
ctx.next_time("05:00", "12:00", "18:00", "00:00")
ctx.next_time(seconds=60)
ctx.next_time("05:00", minutes=10)
```

多个锚点时取最近的未来时间。

### 3.6 生命周期钩子

调度器还支持少量和业务无关的生命周期钩子，用于表达“调度循环阶段变化”：

```python
scheduler.on_start(准备运行环境)
scheduler.on_wakeup(空闲后恢复运行前检查环境)
```

钩子函数支持两种签名：

```python
def 准备运行环境():
    ...

def 空闲后恢复运行前检查环境(scheduler):
    ...
```

语义：

- `on_start`：每次 `run_forever(...)` 启动时执行一次。
- `on_wakeup`：调度器从空闲等待状态重新发现 ready task，并准备执行下一批任务前执行一次。

这类钩子不参与 `next_run_at`，也不应该被当成普通业务 task 持久化。它适合做环境校准，例如桌面窗口排版、工作目录确认、外部资源轻量探活。具体校准逻辑仍属于业务层；调度器只负责提供生命周期入口。

## 4. 运行策略

### 4.1 排序规则

调度器每次拿回控制权后重新扫描任务。

推荐排序规则：

1. `priority` 越大越靠前，默认 `100`。
2. `priority` 相同，按 task 注册顺序。

注册顺序应显式保存为递增 `order`，不依赖容器实现细节。

### 4.2 插队、重置、恢复

`task` 支持三个运行策略参数：

```python
scheduler.task(
    func,
    priority=100,
    触发时可插队=True,
    触发时重置下游=False,
    被上游插队后续跑=True,
    仅活跃时运行=False,
)
```

含义：

- `触发时可插队`：自己满足运行条件时，是否可以打断当前正在运行的下游 job，先执行自己。
- `触发时重置下游`：自己开始运行时，是否清掉排序在自己后面的 active jobs。
- `被上游插队后续跑`：自己被前置 task 插队后，之后是否恢复原 job；如果为 `False`，下次从头创建 job。
- `仅活跃时运行`：自己是辅助守护任务，不负责唤醒空闲调度器；只有存在普通 task 到期或普通 job 运行中时才参与调度。

当下游 `job2` 正在运行，上游 `task1` 触发并开始执行时：

```python
should_reset_job2 = (
    task1.触发时重置下游
    or not job2.task.被上游插队后续跑
)
```

典型配置：

```python
scheduler.task(托管重连).every(minutes=5)
# 默认可插队，不重置下游，下游之后继续跑。

scheduler.task(回到世界, 触发时重置下游=True).daily("05:00")
# 一旦开始，清掉下游运行现场。

scheduler.task(日常副本, 被上游插队后续跑=False).daily("05:00")
# 自己被打断后不恢复，重新开始。

scheduler.task(卡死检测, 仅活跃时运行=True).every(minutes=1)
# 只有业务任务运行期间检查；没有业务任务时不靠它空转唤醒。
```

### 4.3 普通函数与生成器函数

普通函数一次调用到结束：

```python
result = func(ctx)
```

生成器函数每次只推进一步：

```python
result = next(active_generator)
```

每个 `yield` 都表示“释放控制权给调度器”。调度器拿回控制权后重新扫描 task，检查是否有更前置的任务需要插队。

生成器结束时，通过 `StopIteration.value` 读取 `return` 值，并解释其中的下一次调度时间。

## 5. 错误、超时与重试

默认错误策略必须保守：没有配置 `retry(...)` 的 task，报错后保存错误状态并抛出异常，中断调度器。

```python
scheduler.task(每日早晨例常2).daily("07:30")
# 报错：记录错误并中断调度器。
```

只有显式配置了 `retry(...)` 的 task，才进入守护式重试：

```python
scheduler.task(托管重连).every(minutes=5).retry(seconds=10)
scheduler.task(日常助手).daily("00:00", "05:00", "12:00", "18:00").retry(minutes=1)
```

`retry(...)` 不表示“连续尝试 N 次”，而表示：

> 本次 job 失败后，多久以后重新进入可运行状态。

失败处理流程：

1. 销毁当前 job，生成器现场不再保留。
2. 记录 `last_error_at`、`last_error`、`error_count`。
3. 设置 `next_run_at = now + retry_interval`。
4. 调度器继续运行其他 task。

这样错误处理复用原有时间机制，不需要额外 retry 队列，也不会连续重跑导致系统资源消耗或不可逆污染。

### 5.1 超时

`timeout(...)` 是 job 级运行保护：

```python
scheduler.task(日常助手).daily("00:00", "05:00", "12:00", "18:00").timeout(minutes=20)
scheduler.task(托管重连).every(minutes=5).timeout(seconds=30).retry(seconds=10)
```

语义：

- job 从创建开始计时。
- 如果 job 运行时间超过 timeout，调度器把它视为一次错误。
- 超时后的处理完全复用错误处理逻辑：未配置 `retry(...)` 时中断调度器；配置了 `retry(...)` 时销毁当前 job，记录错误，并设置下一次重试时间。

第一版只承诺协作式超时：

- 生成器任务在每次 `yield` 后检查是否超时。
- 普通同步函数如果长时间阻塞在函数内部，同线程调度器无法安全强杀，只能等函数返回或抛错后再发现超时。

如果需要硬超时，应由业务方把任务封装到子进程或专门的可终止执行器中。通用调度器核心不负责强杀线程或回滚业务副作用。

## 6. 日志与观测

`run_forever` 不承载业务调度策略，只负责循环运行、空闲等待和观测控制。

推荐接口：

```python
scheduler = TaskScheduler(
    state_path="scheduler_state.json",
    log_path="scheduler.log",  # 可选
    trace=0,
)

scheduler.run_forever(tick_seconds=1, idle_ratio=0.8, max_idle_seconds=None)
```

`trace` 是框架观测级别，避免与 Python logging 的日志级别混淆：

- `trace=0`：框架本身不主动输出运行日志，业务函数自行记录。
- `trace=1`：job 级日志，包括开始、结束、失败、插队、恢复、重置下游、下次运行时间。
- `trace=2`：tick/step 级日志，包括每次 tick 时间、选中的 task/job、yield/return 所在文件名和代码行号。

`log_path=None` 时只输出控制台；设置 `log_path` 时控制台和文件同时输出。

## 7. 空闲等待

当没有 ready task，也没有 active job 时，调度器进入空闲等待。第一版采用凡修中已经验证过的“按下一次任务剩余时间的 80% 休眠”的思路：

```python
sleep_seconds = min(max(remaining_seconds * idle_ratio, tick_seconds), remaining_seconds)
```

含义：

- 离下一个 task 很远时，不需要每秒空转。
- 不直接睡到目标点，而是睡 80%，给时钟误差、外部状态变化、状态文件调整留出余量。
- 不会睡过最近的 `next_run_at`。
- 可以用 `max_idle_seconds` 限制最长空闲休眠时间。

如果没有任何可用的 `next_run_at`，退回 `tick_seconds`。

## 8. 状态结构草案

状态文件建议按 task name 存储：

```json
{
  "tasks": {
    "每日早晨例常2": {
      "name": "每日早晨例常2",
      "next_run_at": "2026-04-26 07:30:00",
      "last_run_at": "2026-04-25 07:30:00",
      "last_success_at": "2026-04-25 07:35:10",
      "last_error_at": null,
      "last_error": null,
      "error_count": 0
    }
  }
}
```

active job 是否持久化要谨慎。第一版可以只在内存中保留生成器现场；进程重启后未完成 job 统一重建。后续如果需要跨进程恢复，应由业务层设计显式状态机，而不是试图序列化 Python 生成器。

## 9. 调度循环草案

核心循环：

```python
while True:
    now = current_time()

    ready_tasks = find_tasks(next_run_at <= now)
    active_jobs = current_active_jobs()

    candidate = select_first_by_priority_and_order(ready_tasks, active_jobs)
    if candidate is None:
        sleep(calc_idle_sleep_seconds())
        continue

    if candidate is a ready task and not active:
        apply_interrupt_and_invalidation(candidate)
        job = create_job(candidate)
    else:
        job = candidate

    if job_timeout_exceeded(job):
        handle_error(job, TimeoutError(...))
        continue

    try:
        result = step_job(job)
    except Exception as err:
        handle_error(job, err)
        continue

    if job yielded:
        continue

    if job finished:
        update_next_run_at(job, result)
        destroy_job(job)
```

每次 `yield` 或 `return` 后都重新进入扫描阶段，保证前置任务可以及时插队。

## 10. 设计边界

调度器负责：

- task 注册与排序。
- 时间策略与 `next_run_at`。
- job 创建、挂起、恢复、销毁。
- 状态持久化。
- 错误记录与显式 retry。
- 框架级日志和 trace。

调度器不负责：

- AnLib、OCR、WPS、微信、浏览器、课程、手游、账号等业务语义。
- 业务前置动作，例如“回到世界主页”。
- 业务流程内部如何拆步骤。
- 跨进程恢复 Python 生成器现场。

需要前置动作时，业务方自行封装 callable：

```python
def 日常助手(ctx):
    yield from 到达世界主页()
    yield from 执行日常助手()
    return ctx.next_time("00:00", "05:00", "12:00", "18:00")
```

## 11. 与现有调度工具的关系

`APScheduler` 擅长标准时间触发、后台 job 管理和 cron 表达式。本方案关注的是另一类问题：

- 持久化业务状态文件。
- 单线程协作式长流程。
- 生成器任务分步让出控制权。
- 上游任务插队与下游 job 重置。
- 任务运行后动态写回下一次时间。

因此它不应被设计成 APScheduler 的简单封装，而应作为一个小型持久化调度内核。后续如有需要，可以在外层提供 APScheduler 适配器，但核心模型不依赖 APScheduler。

## 12. 实施建议

第一阶段建议只实现最小闭环：

1. `TaskScheduler(state_path, log_path=None, trace=0)`。
2. `scheduler.task(func, name=None, default_next_time=None, priority=100, 触发时可插队=True, 触发时重置下游=False, 被上游插队后续跑=True)`。
3. `.daily(*anchors)`、`.every(...)`、`.timeout(...)`、`.retry(...)`。
4. `ctx.next_time(...)`。
5. 普通函数和生成器函数的统一执行。
6. `run_forever(tick_seconds=1, idle_ratio=0.8, max_idle_seconds=None)`。
7. JSON 状态文件读写。

第二阶段再补：

- `run_once()`、`run_until_idle()`。
- 任务状态查询。
- 更完整的 trace=2 源码位置记录。
- failed/paused 任务管理。
- 测试用 fake clock。

第一版不要引入配置文件、复杂插件系统或业务适配层。先保持代码驱动的 API，让通用调度器边界稳定后，再按真实使用痛点扩展。
