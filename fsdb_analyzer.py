#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Any
from yaml_builder import YamlBuilder, Task
from fsdb_builder import FsdbBuilder
from condition_builder import ConditionBuilder
from utils import resolve_signal_path, normalize_signal_name


class FsdbAnalyzer:
    """Advanced FSDB signal analyzer with complex condition support"""

    def __init__(self, config_path: str) -> None:
        """Initialize analyzer from config file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"[ERROR] Config file not found: {config_path}")

        if config_file.suffix not in [".yaml", ".yml"]:
            print(f"[WARN] Config file extension is not .yaml or .yml: {config_path}")

        self.config_path: str = config_path

        # Step 1: Load basic config to get FSDB file path
        self.yaml_builder = YamlBuilder()
        raw_config = self.yaml_builder.load_config(config_path)

        # Step 2: Initialize FsdbBuilder and get signal hierarchy
        self.verbose: bool = raw_config["output"]["verbose"]
        self.timeout: int = raw_config["output"]["timeout"]
        self.fsdb_file = Path(raw_config["fsdbFile"])
        self.output_dir = Path(raw_config["output"]["directory"])
        self.fsdb_builder = FsdbBuilder(self.fsdb_file, self.output_dir, self.verbose)

        # Get all available signals from FSDB
        # all_signals = self.fsdb_builder.get_signals_index()

        # Step 3: Pass signal info to YamlBuilder for validation and resolution
        self.config: dict[str, Any] = self.yaml_builder.resolve_config(raw_config)

        self.clock_signal: str = self.config["clockSignal"]
        self.global_scope: str = self.config["scope"]
        self.runtime_data: dict[str, Any] = {}
        self.cond_builder = ConditionBuilder()

    def _expand_templates(
        self, templates: list[str], vars: dict[str, str], scope: str
    ) -> list[str]:
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

    def _trace_trigger(self, task: Task) -> list[dict[str, Any]]:
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

                    # Build row_data with trace_id
                    row_data = {"time": row_idx, "trace_id": trace_id, "capd": {}}
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
                    trace_id += 1  # Increment trace_id for each match

                    # Log if configured
                    if task.logging:
                        log_msg = self.yaml_builder.format_log(
                            task.logging, row_data, signals, row_idx
                        )
                        print(f"  [LOG] {log_msg}")
            except Exception as e:
                print(f"[WARN] Error evaluating condition at row {row_idx}: {e}")

        return matched_rows

    def _trace_depends(self, task: Task) -> list[dict[str, Any]]:
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

        if dep_id not in self.runtime_data:
            raise ValueError(
                f"[ERROR] Upstream dependent task '{dep_id}' not found in task_data"
            )

        upstream_data = self.runtime_data[dep_id]
        upstream_rows = upstream_data["rows"]

        print(
            f"Tracing from upstream '{dep_id}' ({len(upstream_rows)} rows) with matchMode='{match_mode}'"
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

        for trace_id, upstream_row in enumerate(upstream_rows):
            start_row_idx = upstream_row["time"]  # Row index
            start_time = timestamps[start_row_idx]  # Actual FSDB timestamp

            if self.verbose:
                print(f"  Trace {trace_id}: searching from row {start_row_idx} (time={start_time})")

            fork_id = 0
            seen_vars: set[tuple[tuple[str, str], ...]] = set()  # For unique_per_var mode

            for row_idx in range(start_row_idx, len(timestamps)):
                current_time = timestamps[row_idx]

                # Check time window limit
                if current_time > start_time + timeout:
                    if self.verbose:
                        print(f"    Timeout at row {row_idx} (time={current_time}), {fork_id} forks found")
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
                                    row_idx, trace_id, fork_id, matched_vars,
                                    templates, signal_data, task
                                )
                                matched_rows.append(row_data)

                                if self.verbose:
                                    print(f"    Fork {fork_id}: match at row {row_idx} (time={current_time})")

                                if log_format:
                                    signals = self._expand_templates(templates, matched_vars, task.scope or "")
                                    log_msg = self.yaml_builder.format_log(log_format, row_data, signals, row_idx)
                                    print(f"    [LOG] {log_msg}")

                                break  # Stop after first match

                            elif match_mode == "unique_per_var":
                                # One match per unique pattern variable combination
                                var_key = tuple(sorted(matched_vars.items()))
                                if var_key in seen_vars:
                                    continue  # Skip duplicate var combinations
                                seen_vars.add(var_key)

                                row_data = self._build_row_data(
                                    row_idx, trace_id, fork_id, matched_vars,
                                    templates, signal_data, task
                                )
                                matched_rows.append(row_data)

                                if self.verbose:
                                    print(f"    Fork {fork_id}: match at row {row_idx} (vars={matched_vars})")

                                if log_format:
                                    signals = self._expand_templates(templates, matched_vars, task.scope or "")
                                    log_msg = self.yaml_builder.format_log(log_format, row_data, signals, row_idx)
                                    print(f"    [LOG] {log_msg}")

                                fork_id += 1

                            else:  # match_mode == "all" (default)
                                # Capture ALL matches in time window
                                row_data = self._build_row_data(
                                    row_idx, trace_id, fork_id, matched_vars,
                                    templates, signal_data, task
                                )
                                matched_rows.append(row_data)

                                if self.verbose:
                                    print(f"    Fork {fork_id}: match at row {row_idx} (vars={matched_vars})")

                                if log_format:
                                    signals = self._expand_templates(templates, matched_vars, task.scope or "")
                                    log_msg = self.yaml_builder.format_log(log_format, row_data, signals, row_idx)
                                    print(f"    [LOG] {log_msg}")

                                fork_id += 1

                        # For "first" mode, break out of the outer time loop too
                        if match_mode == "first" and matched_rows:
                            break

                except Exception as e:
                    if self.verbose:
                        print(f"    [Error] at row {row_idx}: {e}")
                    continue

            if fork_id == 0:
                print(f"[WARN] No match found for trace {trace_id} at time {start_time}")

        return matched_rows

    def _build_row_data(
        self,
        row_idx: int,
        trace_id: int,
        fork_id: int,
        vars: dict[str, str],
        templates: list[str],
        signal_data: dict[str, list[str]],
        task: Task,
    ) -> dict[str, Any]:
        """Build row_data with fork support

        Args:
            row_idx: Current row index in signal data
            trace_id: Upstream trace identifier
            fork_id: Fork index within this trace
            vars: Matched pattern variables
            templates: Capture signal templates
            signal_data: Signal cache data
            task: Task configuration

        Returns:
            Row data dictionary with trace info and captured values
        """
        # Expand capture templates with matched variables
        signals = self._expand_templates(templates, vars, task.scope or "")

        row_data: dict[str, Any] = {
            "time": row_idx,
            "trace_id": trace_id,
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

    def run(self) -> None:
        """Execute all configured analysis tasks"""

        # Build execution order based on dependencies
        tasks : list[Task] = self.config.get("tasks", [])
        task_order = self.yaml_builder.build_exec_order()
        # Collect all signals-of-interest from all tasks using YamlBuilder
        soi = self.yaml_builder.collect_raw_signals(self.global_scope)
        self.fsdb_builder.dump_signals(soi)

        # Build all conditions after signals are dumped, TODO
        for task in tasks:
            if task.condition is None:
                task.condition = self.cond_builder.build(task, self.fsdb_builder)

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

    args = parser.parse_args()

    analyzer = FsdbAnalyzer(args.config)
    analyzer.run()


if __name__ == "__main__":
    main()
