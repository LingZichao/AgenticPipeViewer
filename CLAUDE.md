# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgenticPipeViewer is an FSDB (Fast Signal Database) signal analyzer for hardware verification. It enables complex signal tracing and condition evaluation across RTL simulation waveforms, with support for task dependencies and pattern matching.

## Architecture

### Core Components

**[fsdb_analyzer.py](fsdb_analyzer.py)** - Main entry point and orchestrator
- Loads YAML configuration via `YamlBuilder`
- Initializes `FsdbBuilder` for FSDB file access
- Builds execution order based on task dependencies
- Executes tasks in two modes:
  - **Trigger mode**: Tasks without dependencies, match conditions globally across all time
  - **Trace mode**: Tasks with dependencies, search forward from upstream task matches
- Manages runtime data sharing between tasks

**[yaml_builder.py](yaml_builder.py)** - Configuration parser and validator
- Validates YAML structure with detailed error messages including line numbers
- Converts task dictionaries to `Task` dataclass objects
- Resolves signal paths with scope support (`$mod` references)
- Detects circular dependencies and builds topological execution order
- Collects all signals-of-interest (SOI) from tasks (capture + condition signals)
- Exports dependency graphs via graphviz

**[condition_builder.py](condition_builder.py)** - Condition compilation and evaluation
- Compiles string-based conditions into executable `Condition` objects
- Implements `ExpressionEvaluator` using Python AST for safe expression evaluation
- Supports custom operators:
  - `$split(n)`: Splits wide signal into n equal parts, returns `SignalGroup`
  - `<@`: "contains" operator for checking if value exists in `SignalGroup`
- Handles pattern conditions with `{variable}` placeholders
  - Pre-computes all possible variable values at build time
  - Tests each candidate value at runtime, requires unique match
- Supports `$dep.task_id.signal` references to upstream task data
- Converts Verilog literals (e.g., `32'hDEAD`, `4'b1010`) to integers

**[fsdb_builder.py](fsdb_builder.py)** - FSDB interface wrapper
- Wraps external tools: `fsdbdebug` (hierarchy) and `fsdbreport` (signal dump)
- Expands `{*}` wildcard patterns to actual signal names
- Handles signal bit ranges (e.g., `signal[127:0]`)
- Caches dumped signal values for efficient access
- Converts dot notation (`tb.module.signal`) to FSDB slash format (`/tb/module/signal`)

**[utils.py](utils.py)** - Shared utilities
- `resolve_signal_path()`: Resolves signal paths with scope (handles `$mod` prefix)
- `split_signal()`: Splits hex signal value into equal-width parts
- `verilog_to_int()`: Converts Verilog literals to Python integers
- `match_signal_with_bitwidth()`: Matches signal names with optional bit ranges

### Data Flow

1. **Initialization**: `FsdbAnalyzer.__init__()`
   - Load YAML config → `YamlBuilder.load_config()` validates structure
   - Initialize `FsdbBuilder` with FSDB file path
   - Resolve config → `YamlBuilder.resolve_config()` creates `Task` objects

2. **Signal Collection**: `FsdbAnalyzer.run()`
   - Collect all signals-of-interest → `YamlBuilder.collect_raw_signals()`
     - From `capture` fields (with `{var}` → `{*}`)
     - From `condition` expressions → `ConditionBuilder.collect_signals()`
   - Expand patterns and dump signals → `FsdbBuilder.dump_signals()`

3. **Condition Building**:
   - For each task → `ConditionBuilder.build(task, fsdb_builder)`
   - Pattern conditions: Pre-compute candidate values by expanding signals with `expand_raw_signals()` and extracting variables via regex
   - Creates `Condition` object with compiled evaluator function

4. **Task Execution** (topological order):
   - **Trigger mode** (`_trace_trigger`): Load all needed signals, evaluate condition for each time index
   - **Trace mode** (`_trace_depends`): For each upstream match, search forward from that time
   - Store matched rows to `runtime_data[task.id]` for downstream tasks

## YAML Configuration Format

### Required Fields
```yaml
fsdbFile: path/to/simulation.fsdb
globalClock: tb.clk
scope: tb.top.module  # Optional global scope

output:
  directory: reports_dir
  verbose: false

tasks:
  - id: task1  # Required unique identifier
    condition: "signal == 1'b1"
    capture:
      - signal_name
```

### Task Fields

- **id** (required): Unique task identifier
- **name** (optional): Human-readable task name
- **scope** (optional): Task-specific scope, overrides global scope
- **dependsOn** (optional): Task ID(s) this task depends on (string or list)
- **condition** (required): Condition expression (string or list of strings)
- **capture** (required): List of signals to capture when condition matches
- **logging** (optional): F-string template for logging matched rows

### Condition Syntax

Conditions are Python-like expressions with extensions:

**Signal References**:
- `signal_name` - Resolved with task/global scope
- `$mod.signal` - Explicitly uses scope prefix
- `signal[31:0]` - Bit slice
- `signal{var}` - Pattern with variable (must match exactly one value)

**Operators**:
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `&&` (and), `||` (or), `not`
- Bitwise: `&`, `|`, `^`, `<<`, `>>`
- Custom: `<@` (contains in SignalGroup), `.$split(n)` (split signal)

**Dependencies**:
- `$dep.task_id.signal_name` - Reference captured signal from upstream task

**Verilog Literals**:
- `32'hDEADBEEF`, `8'b10101010`, `16'd1234`

### Pattern Matching

Patterns use `{variable}` placeholders:
```yaml
condition: "signal{idx}_valid == 1 && signal{idx}_data != 0"
capture:
  - signal{idx}_data  # Expands using matched variable value
```

The analyzer pre-computes all possible values for `{idx}`, then tests each at runtime.

### Task Dependencies

Tasks with `dependsOn` execute in trace mode:
```yaml
- id: upstream
  condition: "req_valid == 1"
  capture: [req_addr]

- id: downstream
  dependsOn: upstream
  condition: "$dep.upstream.req_addr == resp_addr"
  capture: [resp_data]
```

For each match in `upstream`, search forward from that time in `downstream`.

## Common Commands

### Running Analysis
```bash
python3 fsdb_analyzer.py -c config.yaml
```

### Development

The codebase uses Python 3 with minimal external dependencies:
- `pyyaml` - YAML parsing
- `graphviz` (optional) - Dependency graph visualization

External tools required (EDA toolchain):
- `fsdbdebug` - Signal hierarchy extraction
- `fsdbreport` - Signal value dumping

These tools must be in PATH (typically from Verdi/nWave installation).

## Key Design Patterns

### Two-Phase Signal Resolution

1. **Collection phase**: Collect raw signals with `{*}` wildcards
2. **Expansion phase**: Expand patterns against FSDB signal list

This minimizes FSDB queries by batching signal dumps.

### Condition Compilation

Conditions compile once at build time, execute many times:
- `ConditionBuilder.build()` → `Condition` with evaluator function
- `ConditionBuilder.exec(condition, runtime_data)` → boolean

Pattern conditions pre-compute candidate values for efficiency.

### AST-Based Expression Evaluation

Uses Python `ast` module for safe expression evaluation:
- Parse condition string → AST
- Walk AST, evaluate nodes with custom visitor
- Resolve signal names → values from `runtime_data["signal_values"]`

This avoids unsafe `eval()` while supporting complex expressions.

### Runtime Data Structure

```python
runtime_data = {
    "signal_values": {signal_name: hex_value},  # Current time slice
    "upstream_row": {time: int, capd: {signal: value}},  # For $dep refs
    "upstream_data": {capd: [signal_list]},  # Available upstream signals
    "vars": {var_name: matched_value},  # Pattern variable bindings
}
```

## Important Conventions

- **Signal paths**: Use dot notation internally (`tb.module.signal`), convert to slash format for FSDB tools
- **Signal values**: Stored as hex strings without '0x' prefix; convert to int for evaluation
- **Bit ranges**: Signal names may have trailing `[msb:lsb]`; normalize by removing for cache keys
- **Pattern normalization**: `{variable}` → `{*}` for wildcard matching
- **Error context**: Include line numbers from YAML in validation errors

## Example Workflow

For a task with pattern and dependency:
```yaml
- id: fetch
  condition: "icache_line{bank}_valid == 1"
  capture: [icache_line{bank}_data]

- id: decode
  dependsOn: fetch
  condition: "inst_data <@ $dep.fetch.icache_line{bank}_data.$split(4)"
  capture: [inst_opcode]
```

Execution:
1. Collect signals: `icache_line{*}_valid`, `icache_line{*}_data`, `inst_data`, `inst_opcode`
2. Expand and dump all signals from FSDB
3. Build `fetch` condition: Pre-compute all bank indices (0, 1, 2, 3...)
4. Execute `fetch`: For each time, test each bank, require unique match
5. Store `fetch` results with captured `icache_line{bank}_data`
6. Build `decode` condition: Split upstream data into 4 parts
7. Execute `decode`: For each fetch match, search forward until `inst_data` is in split group
