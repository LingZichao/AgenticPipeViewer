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
- Implements global flush mechanism:
  - Compiles globalFlush condition using `build_raw()` (no Task object required)
  - Pre-computes flush boundaries via `_compute_flush_boundaries()`
  - Uses linear scan (`_get_flush_region()`) to determine time point's region
  - Terminates traces crossing flush boundaries in `_trace_depends()`
- Executes tasks in two modes:
  - **Trigger mode**: Tasks without dependencies, match conditions globally across all time
  - **Trace mode**: Tasks with dependencies, search forward from upstream task matches (respects flush boundaries)
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
- Provides two build methods:
  - `build(task, fsdb_builder)`: Build from Task object (internally calls `build_raw()`)
  - `build_raw(raw_condition, scope, fsdb_builder)`: Direct compilation without Task object (used for globalFlush)
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
     - From `globalFlush.condition` (if present)
   - Expand patterns and dump signals → `FsdbBuilder.dump_signals()`

3. **Condition Building**:
   - For each task → `ConditionBuilder.build(task, fsdb_builder)`
   - Pattern conditions: Pre-compute candidate values by expanding signals with `expand_raw_signals()` and extracting variables via regex
   - Creates `Condition` object with compiled evaluator function
   - For globalFlush → `ConditionBuilder.build_raw(raw_condition, scope, fsdb_builder)`
     - No Task object needed, direct compilation from raw expression

4. **Global Flush Boundary Computation** (if globalFlush defined):
   - Evaluate flush condition for all time points → `_compute_flush_boundaries()`
   - Store flush timestamps in `self.flush_boundaries` list
   - Partition timeline into regions separated by flush events

5. **Task Execution** (topological order):
   - **Trigger mode** (`_trace_trigger`): Load all needed signals, evaluate condition for each time index
   - **Trace mode** (`_trace_depends`): For each upstream match, search forward from that time
     - Check flush region at start: `start_region = _get_flush_region(start_time)`
     - For each subsequent time point, verify `current_region == start_region`
     - Terminate trace immediately if region boundary is crossed
   - Store matched rows to `runtime_data[task.id]` for downstream tasks

## YAML Configuration Format

### Required Fields
```yaml
fsdbFile: path/to/simulation.fsdb
globalClock: tb.clk
scope: tb.top.module  # Optional global scope

# Optional: Global pipeline flush condition
globalFlush:
  condition:
    - rtu_ifu_flush == 1'b1
    - rtu_ifu_xx_expt_vld == 1'b1

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

### Global Flush

Global flush terminates all in-flight traces when pipeline-wide flush events occur (e.g., exceptions, branch mispredictions, explicit flushes).

**Configuration**:
```yaml
globalFlush:
  condition:
    - rtu_ifu_flush == 1'b1           # Explicit flush from RTU
    - rtu_ifu_xx_expt_vld == 1'b1     # Exception occurs
```

**Behavior**:
- The analyzer pre-computes all flush boundary time points where the condition is satisfied
- Timeline is partitioned into regions separated by flush boundaries
- Any trace that attempts to cross a flush boundary is terminated
- This applies to ALL tasks, regardless of dependencies

**Implementation Details**:
- Flush condition compilation uses `ConditionBuilder.build_raw()` (no Task object needed)
- Boundaries are computed via `_compute_flush_boundaries()` after signals are dumped
- Region lookup uses linear scan via `_get_flush_region(time)`
- Cross-region detection in `_trace_depends()` terminates traces immediately

**Local Flush vs Global Flush**:
- **Global Flush**: Defined in YAML root, affects ALL tasks pipeline-wide (handled in code)
- **Local Flush**: Partial pipeline flushes (e.g., branch prediction) handled via task mutual exclusion in YAML
  ```yaml
  # Example: Local flush via mutual exclusion
  - id: path_predict_correct
    condition: "ipctrl_branch_mistaken == 1'b0"

  - id: path_predict_wrong
    condition: "ipctrl_branch_mistaken == 1'b1"
  ```

**Time-Axis Partitioning Example**:
```
Time:     0  10  20  30  40  50  60  70  80
          |---|---|---|---|---|---|---|---|
Flush:         ^           ^
Region:   [ 0  ][ 1       ][ 2           ]

Trace starting at T=15:
  - Can search forward to T=39 (same region)
  - Terminated at T=40 (crosses flush boundary)
```

## Common Commands

### Running Analysis
```bash
# Standard analysis
python3 fsdb_analyzer.py -c config.yaml

# Debug mode: limit to first N trigger matches (useful for testing/debugging)
python3 fsdb_analyzer.py -c config.yaml --debug-num 1

# Generate dependency graph only (no FSDB analysis)
python3 fsdb_analyzer.py -c config.yaml --deps-only
```

### Command Line Options

- `-c, --config FILE`: YAML configuration file path (required)
- `--debug-num N`: Limit number of trigger matches for debugging (default: 0 = unlimited)
  - Only affects trigger tasks (tasks without dependencies)
  - Useful for quickly testing trace logic with minimal data
  - Example: `--debug-num 1` matches only the first trigger, creating a single trace
- `--deps-only`: Generate dependency graph and exit without FSDB analysis

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
- Resolve signal names → values from `Condition.signal_map` using current `time`

This avoids unsafe `eval()` while supporting complex expressions.

### Runtime Data Structure

```python
runtime_data = {
    "time": int,  # Current time index
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

## Trace Lifecycle Tracking

The analyzer tracks the complete lifecycle of each trace and expands all fork branches into independent linear paths.

### Data Structure

- `trace_lifecycle`: Dict mapping trace_id to list of events
- Each event contains:
  - `type`: "trigger" (starts trace) or "match" (downstream match)
  - `task_id`, `task_name`: Task that generated the event
  - `time`, `fork_path`, `fork_id`: Execution context
  - `vars`: Matched pattern variables
  - `capd`: Captured signal values
  - `log_msg`: Optional formatted log message

### Linear Path Expansion

The `_build_linear_paths()` method converts trace events into linear paths:
1. Build event map from fork_path to event
2. Find all leaf nodes (paths that are not prefixes of other paths)
3. For each leaf, construct complete path from root (trigger) to leaf
4. Returns list of paths, where each path is a complete event sequence

This expansion ensures every fork branch becomes an independent, traceable path.

### Output

After all tasks complete, the analyzer:
1. Prints trace lifecycle report to console via `_print_trace_lifecycle()`
2. Exports to `{output_dir}/trace_lifecycle.txt` via `_export_trace_lifecycle()`

Format shows linear paths with global numbering:
- Path #X (Trace Y, Branch Z)
- Each path uses symbols: ● (start), → (middle), ◆ (end)
- No nested indentation - each path is a flat sequence

### Implementation

- `_record_trace_event()`: Records each match event with full context
- Called from `_trace_trigger()` for trigger events
- Called from `_trace_depends()` for all match modes (first/all/unique_per_var)
- Log messages stored in event data rather than printed immediately
- `_build_linear_paths()`: Expands fork tree into independent linear paths

## Duplicate Match Detection

The analyzer detects when the same waveform row (time point) is matched multiple times **within the same task**, which may indicate that a condition is satisfied by multiple upstream traces or forks reaching the same downstream event.

### Data Structure

- `matched_rows_tracker`: Dict mapping `(task_id, time)` tuple to list of match records
- Only tracks duplicates within the same task (cross-task matches are ignored)
- Each match record contains:
  - `task_id`, `task_name`: Task that matched this row
  - `trace_id`: Trace identifier (which upstream trigger)
  - `fork_path`: Fork path at time of match

### Behavior

**Real-time Warning**: When a duplicate match is detected within the same task, an immediate warning is printed to console:
```
[WARN] Row duplicate match detected in task 'decode' at time=42!
  Current match: trace=1, path=[0, 2]
  Previous match(es) in same task:
    - trace=1, path=[0, 1]
    - trace=2, path=[0]
  This may indicate condition matched multiple times from different upstream traces/forks.
```

**Summary Report**: At the end of analysis, a summary is displayed:
```
======================================================================
[WARN] Duplicate Match Detection Summary (Within-Task)
======================================================================
Total rows with duplicate matches within same task: 5
See trace_lifecycle.txt for details.
======================================================================
```

**Detailed Export**: The `trace_lifecycle.txt` file includes a "Duplicate Match Summary (Within-Task Only)" section:
```
======================================================================
Duplicate Match Summary (Within-Task Only)
======================================================================
Total rows with duplicate matches within same task: 5

Details:

Task 'decode':
  Time=42: 3 matches
    - trace=1, path=[0, 1]
    - trace=1, path=[0, 2]
    - trace=2, path=[0]
  Time=58: 2 matches
    - trace=3, path=[0]
    - trace=4, path=[0]
```

### Implementation

- `_check_duplicate_match()`: Checks for previous matches at the same time point within the same task
  - Uses `(task_id, time)` as key to isolate duplicates per task
  - Called from `_trace_trigger()` for all trigger matches
  - Called from `_trace_depends()` for all trace matches (all match modes)
  - Prints warning immediately when duplicate detected
  - Updates `matched_rows_tracker` with current match record

### Use Cases

Within-task duplicate matches can indicate:
1. **Fork Convergence**: Multiple forks from the same upstream trace converge on the same downstream event
2. **Multi-Trace Convergence**: Different upstream triggers (traces) lead to matching the same downstream event
3. **Condition Over-Matching**: The condition is too broad and matches the same event multiple times

**Example Scenario**:
- Upstream task creates 2 traces (trace 0, trace 1)
- Each trace forks into multiple paths
- Downstream task's condition matches at time=42 from:
  - trace 0, fork 1
  - trace 0, fork 2
  - trace 1, fork 0

This detection helps identify potential issues where a downstream event is being counted multiple times, which may or may not be intended behavior.

## Example Workflow

For a task with pattern and dependency:
```yaml
- id: fetch
  condition: "icache_line{bank}_valid == 1"
  capture: [icache_line{bank}_data]
  logging: "Fetch bank={bank} data={icache_line{bank}_data}"

- id: decode
  dependsOn: fetch
  condition: "inst_data <@ $dep.fetch.icache_line{bank}_data.$split(4)"
  capture: [inst_opcode]
  logging: "Decode opcode={inst_opcode}"
```

Execution:
1. Collect signals: `icache_line{*}_valid`, `icache_line{*}_data`, `inst_data`, `inst_opcode`
2. Expand and dump all signals from FSDB
3. Build `fetch` condition: Pre-compute all bank indices (0, 1, 2, 3...)
4. Execute `fetch`: For each time, test each bank, require unique match
   - Record trigger event for each match with trace_id
5. Store `fetch` results with captured `icache_line{bank}_data`
6. Build `decode` condition: Split upstream data into 4 parts
7. Execute `decode`: For each fetch match, search forward until `inst_data` is in split group
   - Record match events linked to upstream trace_id
8. Print trace lifecycle report showing complete path for each trace
