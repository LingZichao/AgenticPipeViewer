#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Any
from yaml_builder import YamlBuilder, Task
from fsdb_builder import FsdbBuilder
from condition_builder import ConditionBuilder
from utils import resolve_signal_path


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

        # Collect all signals needed (from condition + capture templates)
        all_signal_names = set()

        # Add condition signals (already normalized with {*} wildcards)
        all_signal_names.update(task.condition.signals)

        # Expand capture templates with {*} wildcards
        for tmpl in templates:
            if "{" in tmpl:
                wildcard = re.sub(r'\{[^}]+\}', '{*}', tmpl)
                all_signal_names.add(wildcard)
            else:
                all_signal_names.add(tmpl)

        # Load all signals from FSDB cache (already dumped in run())
        signal_data = {}
        for sig in all_signal_names:
            try:
                signal_data[sig] = self.fsdb_builder.get_signal(sig)
            except RuntimeError:
                print(f"[WARN] Signal {sig} not in cache, skipping")

        if not signal_data:
            print("[WARN] No signals loaded for evaluation")
            return []

        max_len = max(len(vals) for vals in signal_data.values())

        # Evaluate condition for each time point
        matched_rows = []
        for trace_id, row_idx in enumerate(range(max_len)):
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

                if self.cond_builder.exec(task.condition, runtime_data):
                    # Condition matched! Extract pattern variables
                    vars = runtime_data.get("vars", {})

                    # Expand capture templates with matched variables
                    signals = self._expand_templates(templates, vars, task.scope or "")

                    # Build row_data with trace_id
                    row_data = {"time": row_idx, "trace_id": trace_id, "capd": {}}
                    for sig in signals:
                        if sig in signal_data:
                            vals = signal_data[sig]
                            row_data["capd"][sig] = (
                                vals[row_idx] if row_idx < len(vals) else "0"
                            )

                    matched_rows.append(row_data)

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
        """Trace mode: match from upstream dependent task"""
        templates = task.capture
        depends = task.deps
        dep_id = depends[0] if depends else None

        if dep_id not in self.runtime_data:
            raise ValueError(
                f"[ERROR] Upstream dependent task '{dep_id}' not found in task_data"
            )

        upstream_data = self.runtime_data[dep_id]
        upstream_rows = upstream_data["rows"]

        print(
            f"Tracing from upstream dependent task '{dep_id}' with {len(upstream_rows)} rows"
        )
        if self.verbose and upstream_rows:
            print(
                f"  First upstream row: time={upstream_rows[0]['time']}, signals={list(upstream_rows[0]['capd'].keys())}"
            )

        cond = task.condition

        # Collect all signals needed: from capture and from condition
        all_signal_names = set()
        pattern_mappings = {}

        # From capture
        for sig in templates:
            if "{" in sig:
                # Convert {variable} to {*} for expansion
                wildcard_pattern = re.sub(r'\{[^}]+\}', '{*}', sig)
                actual_signals = self.fsdb_builder.expand_raw_signals([wildcard_pattern])
                pattern_mappings[sig] = actual_signals
                all_signal_names.update(actual_signals)
            else:
                all_signal_names.add(sig)

        # From condition - extract pattern signals
        if task.raw_condition and "{" in task.raw_condition:
            patterns = re.findall(r"[\w.]+\{[\w]+\}[\w.]*", task.raw_condition)
            # Convert {variable} to {*} for expansion
            wildcard_patterns = [re.sub(r'\{[^}]+\}', '{*}', p) for p in patterns]
            expanded = self.fsdb_builder.expand_raw_signals(wildcard_patterns)
            all_signal_names.update(expanded)

        # Load all signals
        signal_data = {}
        for sig in all_signal_names:
            try:
                signal_data[sig] = self.fsdb_builder.get_signal(sig)
            except RuntimeError:
                print(f"[WARN] Signal {sig} not found in cache, skipping")

        if not signal_data:
            print("[WARN] No signals loaded for condition evaluation")
            return []

        max_len = max(len(vals) for vals in signal_data.values())
        log_format = task.logging

        # For each upstream row, search forward
        matched_rows = []
        for trace_id, upstream_row in enumerate(upstream_rows):
            start_time = upstream_row["time"]

            if self.verbose:
                print(f"  Searching from time {start_time}... (trace_id={trace_id})")

            # Search forward from start_time
            match_found = False
            for row_idx in range(start_time, max_len):
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
                    if self.cond_builder.exec(cond, runtime_data):
                        row_data = {"time": row_idx, "trace_id": trace_id, "capd": {}}
                        for sig in templates:
                            vals = signal_data[sig]
                            row_data["capd"][sig] = (
                                vals[row_idx] if row_idx < len(vals) else "0"
                            )
                        matched_rows.append(row_data)
                        match_found = True
                        if self.verbose:
                            print(f"    Found match at time {row_idx}")
                        if log_format:
                            log_msg = self.yaml_builder.format_log(
                                log_format, row_data, templates, row_idx
                            )
                            print(f"  [LOG] {log_msg}")
                        break
                except Exception as e:
                    if self.verbose and row_idx == start_time:
                        print(f"    [Error] at time {row_idx}: {e}")
                    continue

            if not match_found:
                print(f"[WARN] No match found for upstream row at time {start_time}")

        return matched_rows

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
        print(f"[DEBUG] Collected {len(soi)} raw signals")
        for sig in soi:
            print(f"  - {sig}")
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
