# WPS / JSA 技术博物馆

这里保存已经退出生产运行的 WPS 在线表格与 JSA（AirScript）资料，供历史追溯、设计研究和未来其他项目人工参考。

## 边界

- `museum/` 不属于 Python 源码包，不参与导入、构建、测试发现或部署。
- CodeYun 考勤系统不得读取这里的任何文件。
- 这里不提供兼容回退；现行系统失败时应修复现行实现。
- 历史模板中的授权信息、文件身份和脚本身份必须脱敏。
- 若其他项目确实需要维护旧系统，应显式使用 `pyxllib.legacy.wps_jsa`，不得从 `pyxllib.api` 等公共入口隐式取得。

## 内容

- `kq5034/`：旧考勤系统的课程脚本、表格脚本、共享运行时快照、研究材料和旧接手文档。
- `docs/generic_jsa_ai_prompt.md`：通用 JSA 使用说明的历史版本。

通用 Python 调用能力保存在：

```python
from pyxllib.legacy.wps_jsa import WpsOnlineBook, load_airscript_template
```

该接口是明确的历史入口，不对新功能作稳定性承诺。
