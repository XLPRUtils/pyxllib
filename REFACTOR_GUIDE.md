# pyxllib 重构指南：移除 File 类 (XlPath)

## 1. 背景与目标 (Background & Goals)

**背景**: `pyxllib` 是一个历史悠久的工具库，其中 `File` 类 (即 `XlPath`) 是对标准库 `pathlib.Path` 的深度封装。随着 Python 标准库的完善和项目架构的演进，`File` 类的过度封装导致了维护成本增加，且与标准库生态存在隔阂。

**目标**: 
1.  **移除 `File` 类**: 将 `pyxllib`、`pyxlpr` 和 `xlproject` 中所有对 `File` (或 `XlPath`) 的依赖，替换为 Python 标准库的 `pathlib.Path`。
2.  **功能解耦**: 将 `File` 类中实用的扩展方法（如自动编码检测、JSONL 读写等）提取为独立的工具函数。
3.  **沉淀经验**: 在移除旧代码前，总结其设计思想和优秀实践，形成文档。

## 2. 涉及范围 (Scope)

重构将产生连锁反应，涉及以下三个仓库/目录：
1.  **`pyxllib`** (`d:\home\chenkunze\slns\pyxllib\src\pyxllib`): 核心库，`File` 类的定义处。
2.  **`pyxlpr`** (`d:\home\chenkunze\slns\pyxllib\src\pyxlpr`): 下游依赖，大量使用了 `pyxllib`。
3.  **`xlproject`** (`d:\home\chenkunze\slns\xlproject`): 业务项目，依赖前两者。

## 3. 迁移策略 (Migration Strategy)

由于依赖链较长，必须分步进行。

### Phase 1: 准备工作 (Preparation)
1.  **创建工具模块**: 在 `pyxllib.file` 下创建新的工具模块（如 `utils.py` 或 `io.py`），将 `XlPath` 中的实用逻辑（`read_json`, `write_jsonl`, `split_to_dir` 等）移植过去，改为接受 `pathlib.Path` 或 `str` 作为参数的函数。
2.  **保持兼容**: 在 `XlPath` 移除前，可以先让其内部调用这些新函数，确保逻辑一致。

### Phase 2: `pyxllib` 内部重构 (Internal Refactoring)
1.  **扫描**: 查找 `pyxllib` 内部所有使用 `File` 或 `XlPath` 的地方。
2.  **替换**:
    *   将 `File(path)` 替换为 `pathlib.Path(path)`。
    *   将 `file.read_json()` 替换为 `pyxllib.file.utils.read_json(path)` 或 `json.loads(path.read_text(encoding='utf-8'))`。
    *   **注意**: 仔细处理编码问题。`XlPath.read_text` 有自动编码检测，而 `pathlib.Path.read_text` 默认无此功能。如果确需自动检测，请使用新封装的工具函数。
3.  **验证**: 运行 `pyxllib` 的单元测试，确保无回归。

### Phase 3: 下游项目重构 (Downstream Refactoring)
1.  **更新依赖**: 确保下游项目使用修改后的 `pyxllib`。
2.  **批量替换**: 对 `pyxlpr` 和 `xlproject` 重复 Phase 2 的替换步骤。

### Phase 4: 清理与文档 (Cleanup & Documentation)
1.  **移除代码**: 正式删除 `XlPath` 类定义及 `File` 别名。
2.  **撰写总结**: 在 `docs/legacy_file_design.md` (或新建 `docs/refactor/file_class_retrospective.md`) 中，详细记录 `File` 类的设计亮点（如自动编码处理、链式调用便利性）以及为何要移除它（如继承标准库带来的复杂性、非标准接口的维护负担）。

## 4. 技术细节与映射 (Technical Details & Mapping)

### 类定义
*   **Old**: `from pyxllib.file.specialist import File` / `XlPath`
*   **New**: `from pathlib import Path`

### 常用方法映射

| 方法 | 旧实现 (`XlPath`) | 新实现建议 (Standard Lib / New Utils) |
| :--- | :--- | :--- |
| **初始化** | `File('path/to/file')` | `Path('path/to/file')` |
| **读文本** | `f.read_text()` (自动检测编码) | `p.read_text(encoding='utf-8')` (默认) <br> 或 `pyxllib.file.utils.read_text(p, auto_encoding=True)` |
| **写文本** | `f.write_text(s)` (自动建目录) | `p.parent.mkdir(parents=True, exist_ok=True)` <br> `p.write_text(s, encoding='utf-8')` |
| **读JSON** | `f.read_json()` | `json.loads(p.read_text(encoding='utf-8'))` <br> 或 `pyxllib.data.jsonlib.read_json(p)` |
| **写JSON** | `f.write_json(data)` | `p.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')` <br> 或 `pyxllib.data.jsonlib.write_json(p, data)` |
| **读JSONL**| `f.read_jsonl()` | `pyxllib.data.jsonlib.read_jsonl(p)` (建议迁移到 jsonlib) |
| **写JSONL**| `f.write_jsonl(data)` | `pyxllib.data.jsonlib.write_jsonl(p, data)` |
| **行数** | `f.get_total_lines()` | 手动实现或使用工具函数 |
| **遍历** | `f.glob_files('*')` | `[p for p in path.glob('*') if p.is_file()]` |

### 关键差异提示
1.  **自动创建目录**: `XlPath` 的写操作（`write_text`, `write_json` 等）会自动创建父目录。`pathlib.Path` 不会，必须手动调用 `p.parent.mkdir(parents=True, exist_ok=True)`。**这是最容易遗漏的变更点！**
2.  **编码**: `XlPath` 依赖 `charset_normalizer`。如果业务逻辑强依赖自动编码检测，必须保留该逻辑的工具函数版本。

## 5. 指令 (Instructions for AI)

当 AI 接收到重构任务时：
1.  **Read Context**: 仔细阅读本指南，确认当前处于哪个阶段（Phase）。
2.  **Check Scope**: 确认要修改的文件范围。
3.  **Implement Safely**:
    *   如果是写入操作，务必检查是否添加了 `mkdir` 逻辑。
    *   如果是读取操作，确认编码参数是否正确。
4.  **Update Docs**: 任务完成后，检查是否需要更新相关文档。

---
**附录**: `XlPath` 原始代码片段参考
(请参考 `pyxllib/file/specialist/xlpath.py` 文件内容)
