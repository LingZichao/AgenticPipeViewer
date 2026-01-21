# AgenticPipeViewer

---

TODO:

* localFlush

主要短板：条件语言与信号解析的健壮性（边沿/去抖/时序语义）、复杂依赖与回溯能力、性能与大规模 FSDB 处理、结果可追溯/可验证性、GUI/交互与调试体验、版本化配置与回归稳定性

---

我思考了一个方案：Condition不再是dataclass， 它有自己的方法，这样直接执行exec()即可，fsdb_analyzer只是搞定fsdb_builder初始化，但具体的数据访问调用都交给下面的yaml_builder和Condition处理 ​README.md 12-18​ 。整个项目的依赖是

Struct：
                yaml_validator
                      |
                      V
fsdb_builder --> yaml_builder -> fsdb_analyzer -> view
    |                 ^               ^
    V                 |               |
cond_builder _________|_______________|