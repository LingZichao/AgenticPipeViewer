# AgenticPipeViewer

---

### Quick Test
```bash
    python3 view.py -c tests/ifu_dep.yaml
```

TODO:

* localFlush

主要短板：条件语言与信号解析的健壮性（边沿/去抖/时序语义）、复杂依赖与回溯能力、性能与大规模 FSDB 处理、结果可追溯/可验证性、GUI/交互与调试体验、版本化配置与回归稳定性

---

Struct：
                yaml_validator
                      |
                      V
fsdb_builder --> yaml_builder -> fsdb_analyzer -> view
    |                 ^               ^
    V                 |               |
cond_builder _________|_______________|