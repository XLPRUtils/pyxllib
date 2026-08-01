# KQ5034 课程数据 PG 退役档案

> 退役日期：2026-07-31

考勤运行态的视频与打卡配置、采集结果已经迁入每门课程自己的 CodeYun 工作簿：

- `视频配置`
- `视频数据`
- `打卡配置`
- `打卡数据`

因此下列四张 PG 表退出运行链，只保留一段观察期，不立即删除：

| 原表名 | 观察期表名 |
| --- | --- |
| `lesson_table` | `lesson_table_retired_20260731` |
| `lesson_data_table` | `lesson_data_table_retired_20260731` |
| `clockin_table` | `clockin_table_retired_20260731` |
| `clockin_data_table` | `clockin_data_table_retired_20260731` |

仍在运行态使用的 PG 表：

- `user_table`：报名学员与小鹅通用户匹配
- `weipay_table`：支付订单查询

## 架构边界

- `xlproject/src/xlsln/kq5034/kqmain.py` 只调度各课程脚本，不再执行全局视频/打卡 Step 1。
- 每门课程从自己的 CodeYun 配置 sheet 判断是否到期，再由 mi15 采集并写回 mf。
- 同一资源 URL 在 mf 使用 3 小时共享缓存；空结果和失败不写成可复用成功缓存。
- CodeYun 不保留旧 PG 回退逻辑。课程 sheet 缺失时应明确失败，不能读取观察期表继续运行。
- 历史 `KqDb` / `KqTools` 课程数据方法仅用于理解旧实现，不属于当前考勤运行接口。

## 回滚

观察期若确认仍有遗漏依赖，可在确认目标原表名尚未被重新创建后，用同一事务恢复：

```sql
BEGIN;
ALTER TABLE public.lesson_table_retired_20260731 RENAME TO lesson_table;
ALTER TABLE public.lesson_data_table_retired_20260731 RENAME TO lesson_data_table;
ALTER TABLE public.clockin_table_retired_20260731 RENAME TO clockin_table;
ALTER TABLE public.clockin_data_table_retired_20260731 RENAME TO clockin_data_table;
COMMIT;
```

不要创建同名兼容视图。兼容视图会掩盖遗漏依赖，失去观察期改名的验证价值。

## 最终删除条件

至少观察一个完整考勤运行周期，确认：

1. 未结课课程的 Step 1–3 正常；
2. 新课程模板可从内部配置 sheet 生成和校验表头；
3. 日志没有 `relation ... does not exist`；
4. 用户匹配与订单查询正常；
5. 没有人工脚本仍调用四张观察期表。

满足后再单独评审删除表及其序列、索引，不在本次退役中自动删除。
