# FSDB Analyzer 使用指南

基于事件依赖的 FSDB 信号追踪工具。

## 快速开始

```bash
python3 fsdb_analyzer.py -c ifu_dep.yaml
```

---

## 核心概念

### 两种任务模式

| 模式 | 触发条件 | 行为 |
|------|----------|------|
| **Trigger** | 无 `dependsOn` | 全局扫描，每个匹配创建一条 **trace** |
| **Trace** | 有 `dependsOn` | 从上游 trace 的时间点开始搜索，可产生多个 **fork** |

### Trace 与 Fork

```
Trigger Task (biu_read):
  trace_0: time=100, data="AABBCCDD"
  trace_1: time=200, data="11223344"
      ↓
Trace Task (ifu2idu):
  trace_0 → fork[0]: inst0 匹配 "DD"
          → fork[1]: inst1 匹配 "CC"
          → fork[2]: inst2 匹配 "BB"
          → fork[3]: inst3 匹配 "AA"
  trace_1 → fork[0]: inst0 匹配 "44"
          → ...
```

- **trace_id**: 从 Trigger 任务继承，贯穿整条依赖链
- **fork_path**: 记录从根到当前节点的 fork 路径，如 `[0, 2, 1]`

---

## YAML 配置结构

```yaml
fsdbFile: /path/to/waveform.fsdb
globalClock: tb.clk
scope: tb.top.module          # 全局信号前缀（可选）

output:
  directory: reports          # 输出目录
  verbose: true               # 详细日志
  timeout: 10000000           # 搜索时间窗口（FSDB 时间单位）

tasks:
  - id: task_id               # 必填，唯一标识
    name: "描述"              # 可选，显示名称
    dependsOn: upstream_id    # 可选，声明依赖
    matchMode: all            # 可选: first, all, unique_per_var
    maxMatch: 4               # 可选，每个上游 trace 最多匹配次数
    condition:
      - "signal == 1'b1"
    capture:
      - signal_name
    logging:
      - "Log message {signal_name}"
```

---

## 示例解析：IFU 指令追踪

### 场景

BIU 读取 128-bit 数据，拆分为 4 条 32-bit 指令，追踪每条指令进入 IDU 的时刻。

### 配置

```yaml
tasks:
  # Task 1: Trigger - 捕获 BIU 读取事件
  - id: biu_read
    condition:
      - "biu_ifu_rd_data_vld == 1'b1"
    capture:
      - biu_ifu_rd_data           # 128-bit 数据

  # Task 2: Trace - 追踪指令分发
  - id: ifu2idu
    dependsOn: biu_read           # 依赖 biu_read
    maxMatch: 4                   # 每次读取最多 4 条指令
    condition:
      - "ifu_idu_ib_inst{idx}_vld == 1'b1"
      - "&& ifu_idu_ib_inst{idx}_data[31:0] <@ $dep.biu_read.biu_ifu_rd_data.$split(4)"
    capture:
      - ifu_idu_ib_inst{idx}_data
```

### 执行流程

```
1. biu_read 全局扫描，找到 213 个 BIU 读取事件
   → 创建 trace_0, trace_1, ..., trace_212

2. ifu2idu 对每个 trace 独立搜索:
   trace_0 (data=0x000013b70221813300100137f14021f3):
     $split(4) → [f14021f3, 00100137, 02218133, 000013b7]
     搜索 ifu_idu_ib_inst{0,1,2,3}_data 是否包含这些值
     → fork[0]: idx=0, data=f14021f3
     → fork[1]: idx=1, data=00100137
     → fork[2]: idx=2, data=02218133
     → fork[3]: idx=3, data=000013b7
```

---

## 条件语法

### 信号引用

| 语法 | 说明 |
|------|------|
| `signal` | 使用全局 scope 解析 |
| `$mod.signal` | 显式使用 scope 前缀 |
| `signal[31:0]` | 位切片 |
| `signal{idx}` | 模式变量，自动展开匹配 |

### 依赖引用

```yaml
$dep.task_id.signal_name           # 引用上游捕获的信号值
$dep.task_id.signal.$split(N)      # 将值拆分为 N 等份
```

### 运算符

| 运算符 | 说明 |
|--------|------|
| `==`, `!=`, `<`, `>` | 比较 |
| `&&`, `\|\|` | 逻辑运算 |
| `&`, `\|`, `^` | 位运算 |
| `<@` | 包含检查（值是否在集合中） |
| `.$split(N)` | 拆分信号为 N 部分 |

### Verilog 字面量

```yaml
32'hDEADBEEF    # 32位十六进制
8'b10101010    # 8位二进制
16'd1234       # 16位十进制
1'b1           # 单bit
```

---

## 匹配模式 (matchMode)

| 模式 | 行为 | 适用场景 |
|------|------|----------|
| `first` | 找到第一个匹配就停止 | 1:1 简单依赖 |
| `all` (默认) | 捕获所有匹配 | 流水线复用场景 |
| `unique_per_var` | 每个模式变量值只匹配一次 | 无复用的分发场景 |

### maxMatch

限制每个上游 trace 的最大匹配数，防止错位匹配：

```yaml
maxMatch: 4   # 每个 BIU 读取最多产生 4 条指令
```

---

## 输出结构

每个匹配的 row_data 包含：

```python
{
    "time": 123,              # FSDB 时间索引
    "trace_id": 0,            # 从根 Trigger 继承
    "fork_path": [0, 2],      # 完整 fork 路径
    "fork_id": 2,             # 当前层 fork 索引
    "vars": {"idx": "1"},     # 匹配的模式变量
    "capd": {                 # 捕获的信号值
        "signal_name": "DEADBEEF"
    }
}
```

---

## 多级依赖链

```yaml
tasks:
  - id: A          # Trigger: 创建 trace
  - id: B          # Trace A: 每个 trace 可产生多个 fork
    dependsOn: A
  - id: C          # Trace B: 每个 fork 继续产生 fork
    dependsOn: B
```

执行结果：
```
A: trace_id=0,1,2  fork_path=[]
B: trace_id=0      fork_path=[0], [1], [2]
   trace_id=1      fork_path=[0]
C: trace_id=0      fork_path=[0,0], [0,1], [1,0], [2,0]
   trace_id=1      fork_path=[0,0]
```

**关键**: `trace_id` 始终从根继承，`fork_path` 唯一标识每条执行路径。

---

## Trace 生命周期追踪

### 概述

新增的 trace 生命周期追踪功能将每个 trace 的所有 fork 分支展开成独立的线性路径,清晰展示从触发到完成的完整执行链。

### 输出格式

执行分析后,会自动生成两种输出:

1. **控制台输出**: 在任务执行完成后打印 Trace Lifecycle Report
2. **文件输出**: 导出到 `{output_dir}/trace_lifecycle.txt`

### 线性路径展开

所有的 fork 分支都会被展开成独立的完整路径。例如:

```
原始 trace 结构:
  biu_read (trigger)
    ├─> ifu2idu (fork 0) -> idu_exec
    ├─> ifu2idu (fork 1) -> idu_exec
    └─> ifu2idu (fork 2) -> idu_exec

展开后的线性路径:
  Path #1: biu_read → ifu2idu (fork 0) → idu_exec
  Path #2: biu_read → ifu2idu (fork 1) → idu_exec
  Path #3: biu_read → ifu2idu (fork 2) → idu_exec
```

### 示例输出

```
======================================================================
Trace Lifecycle Report (Linear Paths)
======================================================================

Path #1 (Trace 0, Branch 0):
  ● [biu_read] time=100
     LOG: BIU read data=0xAABBCCDD
  → [ifu2idu] time=105 (idx=0)
     LOG: IDU inst0=0xDD
  ◆ [idu_exec] time=110
     LOG: Execute inst=0xDD

Path #2 (Trace 0, Branch 1):
  ● [biu_read] time=100
     LOG: BIU read data=0xAABBCCDD
  → [ifu2idu] time=106 (idx=1)
     LOG: IDU inst1=0xCC
  ◆ [idu_exec] time=111
     LOG: Execute inst=0xCC

Path #3 (Trace 1, Branch 0):
  ● [biu_read] time=200
     LOG: BIU read data=0x11223344
  → [ifu2idu] time=205 (idx=0)
     LOG: IDU inst0=0x44
  ◆ [idu_exec] time=210
     LOG: Execute inst=0x44
```

### 解读

- **Path #X** - 全局路径编号(连续递增)
- **(Trace Y, Branch Z)** - 原始 trace ID 和该 trace 内的分支编号
- `●` - 路径起点(Trigger 事件)
- `→` - 路径中间步骤
- `◆` - 路径终点(叶子节点)
- **(var=value)** - 模式变量的匹配值
- `LOG:` - 配置的 logging 消息

每条路径都是完整且独立的,从 trigger 开始到某个终点结束,没有分支嵌套。

### 详细模式

启用 `verbose: true` 时,文件输出中会包含每个事件捕获的所有信号值:

```
Path #1 (Trace 0, Branch 0):
  ● [biu_read] time=100
     LOG: BIU read data=0xAABBCCDD
     Captured signals:
       tb.biu.biu_ifu_rd_data = AABBCCDD
       tb.biu.biu_ifu_rd_grnt = 1
  → [ifu2idu] time=105 (idx=0)
     LOG: IDU inst0=0xDD
     Captured signals:
       tb.ifu.ifu_idu_ib_inst0_data = DD
       tb.ifu.ifu_idu_ib_inst0_vld = 1
  ◆ [idu_exec] time=110
     Captured signals:
       tb.idu.exec_result = ABCD1234
```

### 使用场景

1. **调试复杂依赖链**: 每条路径都是独立完整的,容易跟踪特定执行序列
2. **性能分析**: 分析每条路径从触发到完成的时间跨度
3. **覆盖率验证**: 统计路径数量,确认所有预期分支都被覆盖
4. **报告生成**: 线性路径格式直观清晰,适合作为验证报告
5. **流水线分析**: 观察不同指令/数据在流水线中的完整路径

### 配置

Trace 生命周期追踪默认启用,无需额外配置。输出位置由 `output.directory` 控制:

```yaml
output:
  directory: reports        # trace_lifecycle.txt 会生成在这里
  verbose: true             # 控制是否输出详细的信号值
```
