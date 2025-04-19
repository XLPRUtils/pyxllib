# 1 安装

推荐使用uv环境管理工具，使用uv时，以下pip要改为"uv pip"
，更多uv工具使用技巧见：[uv环境管理工具](https://www.yuque.com/xlpr/pyxllib/uv)。

```shell
# 这样可以安装纯净版pyxllib源码，不附带任何其他三方库依赖。
# 在很清楚自己仅需要什么小功能组件，做简单任务，或者为了打包精简的exe时很有用。
pip install pyxllib

# 常规情况下建议使用basic版基础依赖，目前basic还不算很轻量，项目在不断迭代优化中。
pip install pyxllib[basic]

# 有需要再额外补充一些组件的写法，比如autogui是windows上ui自动化操作相关系列功能
pip install pyxllib[basic,autogui]

# 不怕重依赖，清楚自己在干什么的情况下，可以使用下述配置
pip install pyxllib[advance]
pip install pyxllib[full]
```

pyxllib支持的依赖分组情况，请直接看项目的pyproject.toml配置文件。

大家在使用中有发现依赖使用不流畅，不舒服，反人性的地方，欢迎狠狠来怼我。

# 2 文档

本库的完整介绍文档在语雀: https://www.yuque.com/xlpr/pyxllib
