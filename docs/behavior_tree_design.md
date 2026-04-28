# 持久化协作式行为树设计方案

## 1. 目标

`pyxllib.prog.behavior_tree` 提供一个业务无关的行为树内核，面向 GUI 自动化、手游脚本、考勤批处理、爬虫等需要“反复 tick、可 yield、可持久化时间状态”的场景。

框架只理解这些通用概念：

- `Node`：行为树节点。
- `Status`：`SUCCESS`、`FAILURE`、`RUNNING`、`SKIP`。
- `Action`：包装普通函数或生成器函数。
- `Sequence`、`Selector`、`MemorySelector` 等组合节点。
- `Daily`、`Every`、`DynamicTime`、`Window` 等时间节点。
- `Retry`、`Timeout` 等通用保护节点。
- `BehaviorTreeRunner`：运行、保存状态、空闲等待。

账号、课程、微信、WPS、AnLib、OCR、世界主页、图片识别等都属于业务层，不进入通用行为树框架。

## 2. 最小示例

```python
from pyxllib.prog.behavior_tree import Action, BehaviorTreeRunner, Daily, Root, Sequence

tree = Root(
    Action(每日早晨例常).daily("07:30")
)

runner = BehaviorTreeRunner(tree, "scheduler_state.json")
runner.run_forever(tick_seconds=1)
```

`Action` 支持两类函数：

```python
def task():
    ...

def task(ctx):
    ...
    return ctx.next_time(seconds=60)
```

如果函数返回生成器，每次 `yield` 都会释放控制权，让行为树下一轮重新遍历。

## 3. 持久化默认值

默认持久化策略按节点语义区分：

- `Daily(..., persist=True)`：默认持久化，因为每日任务通常需要跨进程记住下一次时间。
- `Every(..., persist=False)`：默认只保存在内存，适合守护检查。
- `Once(..., persist=False)`：默认只在当前进程内执行一次。
- `MemorySelector(..., persist=False)`：默认只记住当前进程内的运行分支。
- 其他节点默认不持久化，需要跨进程保存时显式传 `persist=True`。

持久化文件结构由 `BehaviorTreeRunner` 维护：

```json
{
  "nodes": {
    "Root/ReactiveSelector/羊驼/日常_助手": {
      "next_run_at": "2026-04-26 12:00:00"
    }
  },
  "blackboard": {}
}
```

业务项目可以把自己的状态也放在同一个 JSON 文件中；行为树只使用 `nodes` 和 `blackboard` 两个保留区。

## 4. 时间节点

`Daily` 是多个每日锚点的封装：

```python
Action(日常助手).daily("05:00", "12:00", "18:00", "00:00")
```

首次没有状态时，默认立即给节点一次检查机会。这避免给每种触发机制设计“逆 next_time”。如果首次不想补跑，或需要忽略旧状态重置调度，可以用 `start`：

```python
Action(task).daily("05:00", start="run")        # 默认：有状态用状态；无状态立刻跑
Action(task).daily("05:00", start="next")       # 有状态用状态；无状态等下一次
Action(task).daily("05:00", start="reset-run")  # 忽略旧状态；立刻跑
Action(task).daily("05:00", start="reset-next") # 忽略旧状态；等下一次
```

临时调试时可以用 `enabled=False` 让节点本次进程不参与调度。关闭后，节点不会运行、不会初始化状态，也不会参与 `IdleUntilNextWake` 的下一次唤醒计算：

```python
Action(task).daily("05:00", start="next", enabled=False)
```

也可以传入更底层的 `default_next_time` 指定初始时间：

```python
Action(task).daily("05:00", default_next_time="2026-04-27 05:00:00")
```

`DynamicTime` 用于业务运行结束后才能确定下一次时间的场景：

```python
DynamicTime(Action(仙府寻访仙侣), fallback_seconds=1800)
```

业务函数可以用 `ctx.next_time(...)` 或在业务封装里写入 `ctx.next_run_at` 来决定下一次唤醒。

如果业务项目需要把行为树的下一次时间同步到自己的兼容状态字段，可以使用 `on_schedule`：

```python
Action(task).daily("05:00", on_schedule=lambda ctx, run_at: sync_legacy_task_time(run_at))
```

`on_schedule` 不参与调度决策，只是时间节点写入 `next_run_at` 后的同步回调。

如果业务已经有自己的持久化字段，也可以显式让时间节点只保留内存态：

```python
Action(task).daily("05:00", persist=False, on_schedule=sync_time)
```

## 5. 空闲等待

`IdleUntilNextWake` 是普通节点，不是 runner 的特殊回调：

```python
Root(
    ReactiveSelector(
        业务子树,
        Sequence(
            IdleUntilNextWake(ratio=0.8),
            Action(桌面排版),
        ),
    )
)
```

它会按下一次可唤醒时间的剩余时长睡眠一部分，默认采用 80% 策略，并且不会睡过最近的下一次唤醒时间。

## 6. 守护服务

`WithServices` 用来表达“业务子树活跃或到期时，顺便运行守护检查”：

```python
WithServices(
    业务子树,
    Every(minutes=5, child=Action(托管重连)),
    Every(minutes=1, child=Action(卡死检测)),
)
```

守护服务的 `Every` 不负责唤醒空闲树。没有业务任务到期、也没有业务节点正在运行时，守护服务不会导致程序持续空转。

## 7. 错误、重试与超时

默认策略是保守的：未显式配置 `Retry` 时，异常会被记录到 runner 状态并继续抛出，中断调度器。

显式配置 `Retry` 后，异常会转换为下一次重试时间：

```python
Action(日常助手).daily("05:00").timeout(minutes=20).retry(minutes=1)
```

`Retry` 表示“失败后多久再进入可运行状态”，不是连续重试 N 次。

`Timeout` 是协作式超时：生成器节点在下一次 tick 时检查是否超时；普通同步函数如果长时间不返回，框架不能安全强杀，只能等函数释放控制权后处理。

## 8. Fanxiu 业务树形态

Fanxiu 的业务层应该只组合通用节点和业务函数：

```python
Root(
    ReactiveSelector(
        Once(Action(layout_scenario_1)),
        WithServices(
            MemorySelector(
                Sequence(
                    Action(lambda: 设置当前账号("羊驼")),
                    MemorySelector(
                        Daily("12:29", Window("12:29", "12:35", Action(日常_魔祖))),
                        Daily("05:00", "12:00", "18:00", "00:00", Action(日常_助手)),
                        DynamicTime(Action(仙府_寻访仙侣)),
                    ),
                ),
                Sequence(
                    Action(lambda: 设置当前账号("驼二")),
                    MemorySelector(...),
                ),
            ),
            Every(minutes=5, child=Action(托管重连)),
            Every(minutes=1, child=Action(卡死检测)),
        ),
        Sequence(
            IdleUntilNextWake(ratio=0.8),
            Action(layout_scenario_1),
        ),
    )
)
```

账号只是业务树的一种分支组织方式，不是通用行为树节点。多账号、课程更新、微信通知等都应以业务函数或业务封装组合进通用节点。

