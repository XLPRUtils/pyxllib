# autogui 浮动模板定位案例

本文用 CodeYun 凡修 `#204 小助手清单` 的真实场景解释 `pyxllib.autogui` 里“标注模板”的使用方式。

## 1. 场景

手游里的助手清单是一个滚动列表。首屏里 `神物园助手` 的 `执行` 按钮位置固定，但滚动一屏后出现的 `仙府资源小助手` 位于不同 y 坐标，按钮文案也从 `执行` 变成 `领取`。

如果直接记录绝对坐标，脚本只能在某一屏成立；只要列表滚动、条目增减或任务状态变化，点击就会落到错误控件。

## 2. 模板模型

更稳定的模型是：

```text
View(#204 小助手清单)
  Shape(滚动窗口)
  Shape(神物园助手)
    Shape(标题)
    Shape(执行)
  Shape(仙府资源小助手)
    Shape(标题)
    Shape(领取)
```

其中 `神物园助手`、`仙府资源小助手` 是条目模板。模板不要求它们在当前截图的同一绝对位置出现，只要求条目内部结构相同或相近。

运行时流程：

1. 在 `滚动窗口` 内 OCR 搜索目标标题。
2. 得到当前可见条目的标题中心 `actual_title_center`。
3. 从静态模板计算 `template_button_center - template_title_center`。
4. 点击 `actual_title_center + offset`。

公式：

```text
actual_target_center =
  actual_anchor_center + (template_target_center - template_anchor_center)
```

## 3. pyxllib 中已有的基础设施

旧 `pyxllib.autogui.anlib` 里已经有这套思路的早期形态：

- `AnShape / AnView / AnRegion` 表达父子 shape、截图、OCR、点击和拖拽。
- `AnView.create_sub_view(sub_view_loc, anchor_shape_loc, det_shape)` 会把静态子视图按动态锚点平移，注释里说明它用于滚动窗口中元素的动态定位。
- `ActivityWatchParams.activity_ignore_regions` 可在比较画面变化时忽略局部区域，和滚动签名排除遮挡区域是同一种设计需求。

新版轻量基础类型在 `pyxllib.autogui.behavior_tree` 重新导出：

- `View`：静态帧。
- `Shape`：标注框，支持 `children()`、`descendants()`、`box()`。
- `ActionPlanner`：把归一化 shape 坐标转为帧内像素坐标，提供 `shape_center()` 和滚动拖拽点计算。
- `ShapeMatchPlanner`：统一图像/OCR 匹配角色。
- `Runtime`：提供截图、匹配、点击、滚动加载等运行期能力。

这说明“模板定位”不应该散落在业务代码里。业务层只应负责 OCR 找到当前条目锚点，几何计算可以抽象成 pyxllib 的通用工具；新版 API 可以理解为把旧 `create_sub_view` 的经验迁移到 `View/Shape/ActionPlanner` 这组轻量模型上。

## 4. 建议补充的通用函数

后续可以在 `pyxllib.autogui` 中补一个纯几何工具，输入静态 `View`/父 shape/子 shape 和运行时锚点，输出目标点击点：

```python
def resolve_child_by_floating_anchor(
    view,
    *,
    parent_title: str,
    anchor_child_title: str,
    target_child_title: str,
    observed_anchor_center: tuple[float, float],
) -> tuple[float, float]:
    ...
```

该函数不做 OCR、不截图、不点击，只做三件事：

- 找到父 shape。
- 找到锚点子 shape 和目标子 shape。
- 用相对偏移把目标子 shape 平移到当前观测位置。

业务 Runtime 可在它外层处理滚动搜索、结果闭环和异常诊断。

## 5. 经验规则

- 场景身份和滚动内容要分离；动态列表内容不能作为场景锚点。
- 父子 shape 命名应利用层级消除歧义，子 shape 保持通用名，例如 `执行`、`领取`、`标题`。
- 点击必须有闭环结果：明确结果页、无事项提示、回到原页超时保底或失败诊断。
- `no_popup` 是无可见反馈，不是业务成功。
- 动作追踪应保存点击前截图和带标记的点击点，方便回放真实落点。
