#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from yaml_builder import YamlBuilder, Task
from fsdb_builder import FsdbBuilder
from condition_builder import ConditionBuilder, Condition
from utils import resolve_signal_path, normalize_signal_name


class FsdbAnalyzer:
    """Advanced FSDB signal analyzer with complex condition support"""

    def __init__(self, config_path: str, deps_only: bool = False) -> None:
        """Initialize analyzer from config file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"[ERROR] Config file not found: {config_path}")

        if config_file.suffix not in [".yaml", ".yml"]:
            print(f"[WARN] Config file extension is not .yaml or .yml: {config_path}")

        self.config_path: str = config_path
        self.deps_only = deps_only

        # Step 1: Load basic config to get FSDB file path
        self.yaml_builder = YamlBuilder()
        raw_config = self.yaml_builder.load_config(config_path)

        # Step 2: Extract config parameters
        self.verbose: bool = raw_config["output"]["verbose"]
        self.timeout: int = raw_config["output"]["timeout"]
        self.fsdb_file = Path(raw_config["fsdbFile"])
        self.output_dir = Path(raw_config["output"]["directory"])

        # Step 3: Initialize ConditionBuilder before resolving config
        self.cond_builder = ConditionBuilder()

        # Step 4: Initialize FsdbBuilder only if not in deps-only mode
        if not deps_only:
            self.fsdb_builder = FsdbBuilder(self.fsdb_file, self.output_dir, self.verbose)
        else:
            # In deps-only mode, skip FSDB initialization
            self.fsdb_builder = None

        # Step 5: Resolve config (convert dict tasks to Task objects)
        self.config: Dict[str, Any] = self.yaml_builder.resolve_config(raw_config)

        self.clock_signal: str = self.config["globalClock"]
        self.global_scope: str = self.config["scope"]
        self.runtime_data: Dict[str, Any] = {}

        # Global flush support (will be compiled after signals are dumped)
        self.gflush_condition: Optional[Condition] = None
        self.flush_boundaries: List[int] = []

        # Trace lifecycle tracking
        self.trace_lifecycle: Dict[int, List[Dict[str, Any]]] = {}  # trace_id -> list of events

    def _record_trace_event(
        self,
        trace_id: int,
        task: Task,
        row_data: Dict[str, Any],
        event_type: str = "match",
    ) -> None:
        """Record a trace lifecycle event

        Args:
            trace_id: Trace identifier
            task: Task that generated this event
            row_data: Row data with time, fork_path, captured values
            event_type: Type of event ("trigger", "match", "timeout", "complete")
        """
        if trace_id not in self.trace_lifecycle:
            self.trace_lifecycle[trace_id] = []

        event = {
            "type": event_type,
            "task_id": task.id,
            "task_name": task.name or task.id,
            "time": row_data.get("time", -1),
            "fork_path": row_data.get("fork_path", []),
            "fork_id": row_data.get("fork_id", -1),
            "vars": row_data.get("vars", {}),
            "capd": row_data.get("capd", {}),
        }

        self.trace_lifecycle[trace_id].append(event)

    def _expand_templates(
        self, templates: List[str], vars: Dict[str, str], scope: str
    ) -> List[str]:
        """Expand signal templates with resolved variables after condition evaluation"""
        signals = []
        for tmpl in templates:
            if isinstance(tmpl, str) and "{" in tmpl:
                sig = tmpl
                for var_name, var_val in vars.items():
                    sig = sig.replace(f"{{{var_name}}}", var_val)
                sig = resolve_signal_path(sig, scope)
                signals.append(sig)
            else:
                signals.append(tmpl)
        return signals

    def _trace_trigger(self, task: Task) -> List[Dict[str, Any]]:
        """Trigger mode: match conditions globally, each match starts new trace"""
        templates = task.capture
        print(f"Evaluating condition for {len(templates)} signal(s)")

        cond = task.condition
        if cond is None:
            raise RuntimeError(f"[ERROR] Condition not built for task '{task.id}'")

        # Collect all signals needed (from condition + capture templates)
        raw_signal_patterns = set()

        # Add condition signals (already normalized with {*} wildcards)
        raw_signal_patterns.update(cond.signals)

        # Expand capture templates with {*} wildcards
        for tmpl in templates:
            if "{" in tmpl:
                wildcard = re.sub(r'\{[^}]+\}', '{*}', tmpl)
                raw_signal_patterns.add(wildcard)
            else:
                raw_signal_patterns.add(tmpl)

        # Expand {*} patterns to actual signal names
        all_signal_names = self.fsdb_builder.expand_raw_signals(list(raw_signal_patterns))

        # Load all signals from FSDB cache (already dumped in run())
        # Note: Cache uses normalized names (without bit ranges)
        signal_data = {}
        for sig in all_signal_names:
            # Normalize signal name by removing bit range
            normalized_sig = normalize_signal_name(sig)
            try:
                signal_data[normalized_sig] = self.fsdb_builder.get_signal(normalized_sig)
            except RuntimeError:
                print(f"[WARN] Signal {normalized_sig} not in cache, skipping")

        if not signal_data:
            print("[WARN] No signals loaded for evaluation")
            return []

        max_len = max(len(vals) for vals in signal_data.values())

        # Evaluate condition for each time point
        matched_rows = []
        trace_id = 0  # Track number of matches

        for row_idx in range(max_len):
            try:
                # Build signal_values for this time point
                signal_values = {}
                for sig, vals in signal_data.items():
                    signal_values[sig] = vals[row_idx] if row_idx < len(vals) else '0'

                runtime_data = {
                    "signal_values": signal_values,
                    "upstream_row": {},
                    "upstream_data": {},
                    "vars": {},
                }

                if self.cond_builder.exec(cond, runtime_data):
                    # Condition matched! Extract pattern variables
                    vars = runtime_data.get("vars", {})

                    # Expand capture templates with matched variables
                    signals = self._expand_templates(templates, vars, task.scope or "")

                    # Build row_data with trace_id and fork_path (root has empty fork_path)
                    row_data = {"time": row_idx, "trace_id": trace_id, "fork_path": [], "capd": {}}
                    for sig in signals:
                        # Normalize signal name to match cache keys
                        normalized_sig = normalize_signal_name(sig)
                        if normalized_sig in signal_data:
                            vals = signal_data[normalized_sig]
                            # Use original signal name (with scope) as key in capd
                            row_data["capd"][sig] = (
                                vals[row_idx] if row_idx < len(vals) else "0"
                            )
                        # Signal not found in signal_data - skip silently

                    matched_rows.append(row_data)

                    # Record trace lifecycle event (trigger starts a new trace)
                    self._record_trace_event(trace_id, task, row_data, event_type="trigger")

                    trace_id += 1  # Increment trace_id for each match

                    # Log if configured (but don't print immediately - save for trace lifecycle)
                    if task.logging:
                        log_msg = self.yaml_builder.format_log(
                            task.logging, row_data, signals, row_idx
                        )
                        # Store log message in trace event
                        if trace_id - 1 in self.trace_lifecycle:
                            self.trace_lifecycle[trace_id - 1][-1]["log_msg"] = log_msg
            except Exception as e:
                print(f"[WARN] Error evaluating condition at row {row_idx}: {e}")

        return matched_rows

    def _trace_depends(self, task: Task) -> List[Dict[str, Any]]:
        """Trace mode: match from upstream with trace fork support

        Supports three matching modes via task.match_mode:
        - "first": Original behavior, capture first match and break
        - "all": Capture ALL matches in time window (default)
        - "unique_per_var": One match per unique pattern variable combination
        """
        templates = task.capture
        depends = task.deps
        dep_id = depends[0] if depends else None
        match_mode = task.match_mode  # "first", "all", "unique_per_var"
        max_match = task.max_match  # 0 = unlimited

        if dep_id not in self.runtime_data:
            raise ValueError(
                f"[ERROR] Upstream dependent task '{dep_id}' not found in task_data"
            )

        upstream_data = self.runtime_data[dep_id]
        upstream_rows = upstream_data["rows"]

        max_match_info = f", maxMatch={max_match}" if max_match > 0 else ""
        print(
            f"Tracing from upstream '{dep_id}' ({len(upstream_rows)} rows) with matchMode='{match_mode}'{max_match_info}"
        )
        if self.verbose and upstream_rows:
            print(
                f"  First upstream row: time={upstream_rows[0]['time']}, signals={list(upstream_rows[0]['capd'].keys())}"
            )

        cond = task.condition
        if cond is None:
            raise RuntimeError(f"[ERROR] Condition not built for task '{task.id}'")

        # Collect all signals needed (from condition + capture templates)
        raw_signal_patterns = set()

        # Add condition signals (already normalized with {*} wildcards)
        raw_signal_patterns.update(cond.signals)

        # Expand capture templates with {*} wildcards
        for tmpl in templates:
            if "{" in tmpl:
                wildcard = re.sub(r'\{[^}]+\}', '{*}', tmpl)
                raw_signal_patterns.add(wildcard)
            else:
                raw_signal_patterns.add(tmpl)

        # Expand {*} patterns to actual signal names
        all_signal_names = self.fsdb_builder.expand_raw_signals(list(raw_signal_patterns))

        # Load all signals
        # Note: Cache uses normalized names (without bit ranges)
        signal_data = {}
        for sig in all_signal_names:
            # Normalize signal name by removing bit range
            normalized_sig = normalize_signal_name(sig)
            try:
                signal_data[normalized_sig] = self.fsdb_builder.get_signal(normalized_sig)
            except RuntimeError:
                print(f"[WARN] Signal {normalized_sig} not found in cache, skipping")

        if not signal_data:
            print("[WARN] No signals loaded for condition evaluation")
            return []

        # Get timeout from config and timestamps from FSDB
        timeout = self.timeout
        timestamps = self.fsdb_builder.timestamps

        if not timestamps:
            print("[WARN] No timestamps available from FSDB")
            return []

        log_format = task.logging

        # For each upstream row, search forward with time window and fork support
        matched_rows = []

        for upstream_row in upstream_rows:
            # Inherit trace_id from upstream (fixes trace lineage)
            upstream_trace_id = upstream_row["trace_id"]
            upstream_fork_path = upstream_row.get("fork_path", [])
            start_row_idx = upstream_row["time"]  # Row index
            start_time = timestamps[start_row_idx]  # Actual FSDB timestamp

            if self.verbose:
                print(f"  Trace {upstream_trace_id} (path={upstream_fork_path}): searching from row {start_row_idx} (time={start_time})")

            fork_id = 0
            seen_vars: Set[Tuple[Tuple[str, str], ...]] = set()  # For unique_per_var mode

            # Record starting flush region
            start_region = self._get_flush_region(start_time) if self.flush_boundaries else -1

            for row_idx in range(start_row_idx, len(timestamps)):
                current_time = timestamps[row_idx]

                # Check flush region boundary
                if self.flush_boundaries:
                    current_region = self._get_flush_region(current_time)
                    if current_region != start_region:
                        if self.verbose:
                            print(f"    [T={start_time}] Trace terminated at T={current_time}: "
                                  f"crossed flush boundary (region {start_region}→{current_region})")
                        break

                # Check time window limit
                if current_time > start_time + timeout:
                    if self.verbose:
                        print(f"    Timeout at row {row_idx} (time={current_time}), {fork_id} forks found")
                    break

                # Check max_match limit (per upstream trigger)
                if max_match > 0 and fork_id >= max_match:
                    if self.verbose:
                        print(f"    Max match limit ({max_match}) reached at row {row_idx}, stopping")
                    break

                try:
                    # Prepare signal values for this row
                    signal_values = {}
                    for sig, vals in signal_data.items():
                        signal_values[sig] = vals[row_idx] if row_idx < len(vals) else '0'

                    runtime_data = {
                        "signal_values": signal_values,
                        "upstream_row": upstream_row,
                        "upstream_data": upstream_data,
                        "vars": {},
                    }

                    condition_result = self.cond_builder.exec(cond, runtime_data)
                    if condition_result:
                        # Check for multiple matched variables (pattern condition with multiple valid values)
                        all_matched = runtime_data.get("_all_matched_vars", {})
                        single_vars = runtime_data.get("vars", {})

                        # Determine the list of variable combinations to process
                        if all_matched:
                            # Multiple matches at this time step - iterate through all
                            var_name = list(all_matched.keys())[0]
                            all_var_values = all_matched[var_name]
                        else:
                            # Single match
                            var_name = list(single_vars.keys())[0] if single_vars else None
                            all_var_values = [single_vars.get(var_name)] if var_name else [None]

                        # Process each matched variable value
                        for var_val in all_var_values:
                            matched_vars = {var_name: var_val} if var_name and var_val else {}

                            # Apply match mode filtering
                            if match_mode == "first":
                                # Original behavior: capture first match, break
                                row_data = self._build_row_data(
                                    row_idx, upstream_trace_id, fork_id, upstream_fork_path,
                                    matched_vars, templates, signal_data, task
                                )
                                matched_rows.append(row_data)

                                # Record trace lifecycle event
                                self._record_trace_event(upstream_trace_id, task, row_data, event_type="match")

                                if self.verbose:
                                    print(f"    Fork {fork_id}: match at row {row_idx} (time={current_time})")

                                if log_format:
                                    signals = self._expand_templates(templates, matched_vars, task.scope or "")
                                    log_msg = self.yaml_builder.format_log(log_format, row_data, signals, row_idx)
                                    # Store log message in trace event
                                    if upstream_trace_id in self.trace_lifecycle:
                                        self.trace_lifecycle[upstream_trace_id][-1]["log_msg"] = log_msg

                                break  # Stop after first match

                            elif match_mode == "unique_per_var":
                                # One match per unique pattern variable combination
                                var_key = tuple(sorted(matched_vars.items()))
                                if var_key in seen_vars:
                                    continue  # Skip duplicate var combinations
                                seen_vars.add(var_key)

                                row_data = self._build_row_data(
                                    row_idx, upstream_trace_id, fork_id, upstream_fork_path,
                                    matched_vars, templates, signal_data, task
                                )
                                matched_rows.append(row_data)

                                # Record trace lifecycle event
                                self._record_trace_event(upstream_trace_id, task, row_data, event_type="match")

                                if self.verbose:
                                    print(f"    Fork {fork_id}: match at row {row_idx} (vars={matched_vars})")

                                if log_format:
                                    signals = self._expand_templates(templates, matched_vars, task.scope or "")
                                    log_msg = self.yaml_builder.format_log(log_format, row_data, signals, row_idx)
                                    # Store log message in trace event
                                    if upstream_trace_id in self.trace_lifecycle:
                                        self.trace_lifecycle[upstream_trace_id][-1]["log_msg"] = log_msg

                                fork_id += 1

                            else:  # match_mode == "all" (default)
                                # Capture ALL matches in time window
                                row_data = self._build_row_data(
                                    row_idx, upstream_trace_id, fork_id, upstream_fork_path,
                                    matched_vars, templates, signal_data, task
                                )
                                matched_rows.append(row_data)

                                # Record trace lifecycle event
                                self._record_trace_event(upstream_trace_id, task, row_data, event_type="match")

                                if self.verbose:
                                    print(f"    Fork {fork_id}: match at row {row_idx} (vars={matched_vars})")

                                if log_format:
                                    signals = self._expand_templates(templates, matched_vars, task.scope or "")
                                    log_msg = self.yaml_builder.format_log(log_format, row_data, signals, row_idx)
                                    # Store log message in trace event
                                    if upstream_trace_id in self.trace_lifecycle:
                                        self.trace_lifecycle[upstream_trace_id][-1]["log_msg"] = log_msg

                                fork_id += 1

                        # For "first" mode, break out of the outer time loop too
                        if match_mode == "first" and matched_rows:
                            break

                except Exception as e:
                    if self.verbose:
                        print(f"    [Error] at row {row_idx}: {e}")
                    continue

            if fork_id == 0:
                print(f"[WARN] No match found for trace {upstream_trace_id} (path={upstream_fork_path}) at time {start_time}")

        return matched_rows

    def _build_row_data(
        self,
        row_idx: int,
        trace_id: int,
        fork_id: int,
        upstream_fork_path: List[int],
        vars: Dict[str, str],
        templates: List[str],
        signal_data: Dict[str, List[str]],
        task: Task,
    ) -> Dict[str, Any]:
        """Build row_data with fork support

        Args:
            row_idx: Current row index in signal data
            trace_id: Upstream trace identifier (inherited from root)
            fork_id: Fork index within this trace
            upstream_fork_path: Fork path from upstream row
            vars: Matched pattern variables
            templates: Capture signal templates
            signal_data: Signal cache data
            task: Task configuration

        Returns:
            Row data dictionary with trace info and captured values
        """
        # Expand capture templates with matched variables
        signals = self._expand_templates(templates, vars, task.scope or "")

        row_data: Dict[str, Any] = {
            "time": row_idx,
            "trace_id": trace_id,
            "fork_path": upstream_fork_path + [fork_id],
            "fork_id": fork_id,
            "vars": vars.copy(),
            "capd": {},
        }

        for sig in signals:
            # Normalize signal name to match cache keys
            normalized_sig = normalize_signal_name(sig)
            if normalized_sig in signal_data:
                vals = signal_data[normalized_sig]
                # Use original signal name (with scope) as key in capd
                row_data["capd"][sig] = vals[row_idx] if row_idx < len(vals) else "0"

        return row_data

    def _export_trace_lifecycle(self, output_file: Path) -> None:
        """Export trace lifecycle to file as linear paths

        Args:
            output_file: Path to output file
        """
        if not self.trace_lifecycle:
            return

        try:
            with open(output_file, "w") as f:
                f.write("=" * 70 + "\n")
                f.write("Trace Lifecycle Report (Linear Paths)\n")
                f.write("=" * 70 + "\n\n")

                path_counter = 0
                for trace_id in sorted(self.trace_lifecycle.keys()):
                    events = self.trace_lifecycle[trace_id]
                    if not events:
                        continue

                    # Build linear paths from trace events
                    linear_paths = self._build_linear_paths(events)

                    # Write each path
                    for path_idx, path in enumerate(linear_paths):
                        path_counter += 1
                        f.write(f"Path #{path_counter} (Trace {trace_id}, Branch {path_idx}):\n")

                        for step_idx, event in enumerate(path):
                            task_name = event["task_name"]
                            time = event["time"]
                            log_msg = event.get("log_msg", "")

                            # Use simple arrow notation for linear sequence
                            if step_idx == 0:
                                symbol = "●"  # Start
                            elif step_idx == len(path) - 1:
                                symbol = "◆"  # End
                            else:
                                symbol = "→"  # Middle

                            f.write(f"  {symbol} [{task_name}] time={time}")

                            # Add variables if present
                            if event.get("vars"):
                                vars_str = ", ".join(f"{k}={v}" for k, v in event["vars"].items())
                                f.write(f" ({vars_str})")

                            f.write("\n")

                            # Write log message if available
                            if log_msg:
                                f.write(f"     LOG: {log_msg}\n")

                            # Write captured signals in verbose mode
                            if self.verbose and event["capd"]:
                                f.write(f"     Captured signals:\n")
                                for sig, val in event["capd"].items():
                                    f.write(f"       {sig} = {val}\n")

                        f.write("\n")  # Empty line between paths

            print(f"[INFO] Trace lifecycle report exported to: {output_file}")
        except Exception as e:
            print(f"[WARN] Failed to export trace lifecycle: {e}")

    def _build_linear_paths(self, events: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Build linear paths from trace events by expanding forks

        Each path is a complete chain from trigger to a leaf node.

        Args:
            events: List of events for a single trace_id

        Returns:
            List of paths, where each path is a list of events
        """
        if not events:
            return []

        # Build a tree structure: fork_path -> event
        event_map: Dict[tuple, Dict[str, Any]] = {}
        for event in events:
            fork_path_tuple = tuple(event["fork_path"])
            event_map[fork_path_tuple] = event

        # Find all leaf nodes (nodes that are not prefixes of other nodes)
        all_paths = set(event_map.keys())
        leaf_paths = []

        for path in all_paths:
            is_leaf = True
            for other_path in all_paths:
                if other_path != path and len(other_path) > len(path):
                    # Check if path is a prefix of other_path
                    if other_path[:len(path)] == path:
                        is_leaf = False
                        break
            if is_leaf:
                leaf_paths.append(path)

        # Build linear paths from root to each leaf
        linear_paths = []
        root_event = events[0]  # Trigger event always first

        for leaf_path in leaf_paths:
            path_events = [root_event]  # Start with trigger

            # Build path by following fork_path indices
            for depth in range(len(leaf_path)):
                prefix = leaf_path[:depth + 1]
                if prefix in event_map:
                    path_events.append(event_map[prefix])

            linear_paths.append(path_events)

        # If no downstream events, return just the trigger
        if not linear_paths:
            linear_paths = [[root_event]]

        return linear_paths

    def _print_trace_lifecycle(self) -> None:
        """Print trace lifecycle as linear paths (expand all forks)"""
        if not self.trace_lifecycle:
            return

        print(f"\n{'=' * 70}")
        print("Trace Lifecycle Report (Linear Paths)")
        print(f"{'=' * 70}\n")

        path_counter = 0
        for trace_id in sorted(self.trace_lifecycle.keys()):
            events = self.trace_lifecycle[trace_id]
            if not events:
                continue

            # Build linear paths from trace events
            linear_paths = self._build_linear_paths(events)

            # Print each path
            for path_idx, path in enumerate(linear_paths):
                path_counter += 1
                print(f"Path #{path_counter} (Trace {trace_id}, Branch {path_idx}):")

                for step_idx, event in enumerate(path):
                    task_name = event["task_name"]
                    time = event["time"]
                    log_msg = event.get("log_msg", "")

                    # Use simple arrow notation for linear sequence
                    if step_idx == 0:
                        symbol = "●"  # Start
                    elif step_idx == len(path) - 1:
                        symbol = "◆"  # End
                    else:
                        symbol = "→"  # Middle

                    print(f"  {symbol} [{task_name}] time={time}", end="")

                    # Add variables if present
                    if event.get("vars"):
                        vars_str = ", ".join(f"{k}={v}" for k, v in event["vars"].items())
                        print(f" ({vars_str})", end="")

                    print()  # New line

                    # Print log message if available
                    if log_msg:
                        print(f"     LOG: {log_msg}")

                    # Print captured signals in verbose mode
                    if self.verbose and event["capd"]:
                        capd_preview = {k: v for k, v in list(event["capd"].items())[:3]}
                        print(f"     Captured: {capd_preview}")

                print()  # Empty line between paths

    def _capture_task(self, task: Task) -> str:
        """Execute capture mode task"""
        # Detect trace mode using pre-analyzed flag
        if task.deps:
            # Trace: match from upstream
            matched_rows = self._trace_depends(task)
        else:
            # Trigger: match all time (Start point)
            matched_rows = self._trace_trigger(task)

        # Store to memory
        self.runtime_data[task.id] = {
            "rows": matched_rows,
            "capd": task.capture,  # All captured signals are available for reference
        }

        print(f"Matched {len(matched_rows)} rows")
        print(f"Result: {len(matched_rows)} rows in memory\n")
        return f"[Memory] {len(matched_rows)} rows"

    def _compute_flush_boundaries(self) -> None:
        """Compute all global flush boundary time points"""
        print("[INFO] Computing global flush boundaries...")

        timestamps = self.fsdb_builder.timestamps

        for row_idx in range(len(timestamps)):
            # Get signal values for this time point
            signal_values = {}
            for signal in self.fsdb_builder.signal_cache:
                norm_signal = normalize_signal_name(signal)
                values = self.fsdb_builder.signal_cache[signal]
                signal_values[norm_signal] = values[row_idx] if row_idx < len(values) else "0"

            runtime_data = {"signal_values": signal_values}

            # Check if flush condition is satisfied
            if self.cond_builder.exec(self.gflush_condition, runtime_data):
                flush_time = timestamps[row_idx]
                self.flush_boundaries.append(flush_time)

                if self.verbose:
                    # Output which signals triggered flush
                    flush_signals = [
                        f"{sig}={val}"
                        for sig, val in signal_values.items()
                        if val not in ("0", "*0")
                    ]
                    print(f"  Flush @ T={flush_time}: {', '.join(flush_signals[:5])}")

        print(f"[INFO] Found {len(self.flush_boundaries)} flush boundaries")

    def _get_flush_region(self, time: int) -> int:
        """Return the flush region number for a given time point

        Regions are numbered starting from 0. Each flush boundary increments the region.
        """
        region = 0
        for boundary in self.flush_boundaries:
            if boundary <= time:
                region += 1
            else:
                break
        return region

    def run(self) -> None:
        """Execute all configured analysis tasks"""

        # Build execution order based on dependencies (always needed for graph export)
        tasks : List[Task] = self.config.get("tasks", [])
        task_order = self.yaml_builder.build_exec_order()

        # Early exit for deps-only mode after graph is generated
        if self.deps_only:
            print("[INFO] Dependency graph generated. Exiting deps-only mode without FSDB analysis.")
            return
        if not self.fsdb_builder:
            raise RuntimeError("[ERROR] FsdbBuilder not initialized in deps-only mode")

        # Collect all signals-of-interest from all tasks using YamlBuilder
        soi = self.yaml_builder.collect_raw_signals(self.global_scope)
        self.fsdb_builder.dump_signals(soi)

        # Build all conditions after signals are dumped, TODO
        for task in tasks:
            if task.condition is None:
                task.condition = self.cond_builder.build(task, self.fsdb_builder)

        # Build globalFlush condition after signals are dumped
        if "globalFlush" in self.config:
            flush_config = self.config["globalFlush"]

            # Build condition directly from raw expression
            self.gflush_condition = self.cond_builder.build_raw(
                raw_condition=flush_config["condition"],
                scope=self.global_scope,
                fsdb_builder=self.fsdb_builder
            )
            print("[INFO] Global flush condition compiled")

            # Compute flush boundaries
            self._compute_flush_boundaries()

        print(f"\n{'=' * 70}")
        print(f"[INFO] FSDB Analyzer - Collected {len(tasks)} task(s)")
        print(f"{'=' * 70}")
        print(f"[INFO] FSDB file: {self.fsdb_file}")
        print(f"[INFO] Clock signal: {self.clock_signal}")
        print(f"[INFO] Output directory: {self.output_dir}")
        print(f"[INFO] Verbose mode: {'yes' if self.verbose else 'no'}")
        print(f"{'=' * 70}\n")

        results = []
        for exec_idx, task_idx in enumerate(task_order, 1):
            task = tasks[task_idx]
            # Use name for display, fallback to id
            task_name = task.name or task.id or f"Task {exec_idx}"

            print(f"\n[Task {exec_idx}/{len(tasks)}] {task_name}")
            if task.deps:
                print(f"  Depends on: {', '.join(task.deps)}")
            print(f"{'-' * 70}")

            try:
                result = self._capture_task(task)
                results.append((task_name, result))
            except Exception as e:
                print(f"Task failed: {e}")
                results.append((task_name, f"ERROR: {e}"))

        # Summary
        print(f"\n{'=' * 70}")
        print("Summary:")
        print(f"{'=' * 70}")
        for name, result in results:
            print(f"  {name}: {result}")
        print()

        # Export trace lifecycle to file (no console output)
        trace_report_file = self.output_dir / "trace_lifecycle.txt"
        self._export_trace_lifecycle(trace_report_file)


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced FSDB Signal Analyzer with Complex Conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with advanced configuration
  %(prog)s -c ifu_analysis_advanced.yaml
        """,
    )

    parser.add_argument(
        "-c", "--config", required=True, 
        help="YAML configuration file path"
    )

    parser.add_argument(
        "--deps-only",
        action="store_true",
        help="Only generate dependency graph and exit (skip FSDB analysis)",
    )

    args = parser.parse_args()

    analyzer = FsdbAnalyzer(args.config, deps_only=args.deps_only)
    analyzer.run()


if __name__ == "__main__":
    main()
