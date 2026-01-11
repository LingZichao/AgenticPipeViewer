#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any
from yaml_builder import YamlBuilder
from fsdb_builder import FsdbBuilder
from condition_builder import ConditionBuilder


class FsdbAnalyzer:
    """Advanced FSDB signal analyzer with complex condition support"""

    def __init__(self, config_path: str) -> None:
        """Initialize analyzer from config file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"[ERROR] Config file not found: {config_path}")

        if config_file.suffix not in ['.yaml', '.yml']:
            print(f"[WARN] Config file extension is not .yaml or .yml: {config_path}")

        self.config_path: str = config_path

        # Step 1: Load basic config to get FSDB file path
        self.yaml_builder: YamlBuilder = YamlBuilder()
        basic_config = self.yaml_builder.load_config(config_path)

        # Step 2: Initialize FsdbBuilder and get signal hierarchy
        self.fsdb_file: Path = Path(basic_config['fsdbFile'])
        self.output_dir: Path = Path(basic_config['output']['directory'])
        self.verbose: bool = basic_config['output']['verbose']
        self.fsdb_builder: FsdbBuilder = FsdbBuilder(self.fsdb_file,
                                        self.output_dir,
                                        self.verbose)

        # Get all available signals from FSDB
        all_signals = self.fsdb_builder.get_all_signals()

        # Step 3: Pass signal info to YamlBuilder for validation and resolution
        self.config: dict[str, Any] = self.yaml_builder.resolve_signals(all_signals, basic_config)

        self.clock_signal: str = self.config['clockSignal']
        self.global_scope: str = self.config['scope']
        self.runtime_data: dict[str, Any] = {}
        self.cond_builder: ConditionBuilder = ConditionBuilder(self.fsdb_builder, self.yaml_builder)

    def _trace_trigger(self, task: dict[str, Any],
                       capture_signals: list[str], 
                       condition: str) -> list[dict[str, Any]]:
        """Normal mode: match all rows globally"""
        print(f"Evaluating condition for {len(capture_signals)} signal(s)")

        # Build condition once
        cond = self.cond_builder.build(condition, task, self.global_scope)

        signal_data = {}
        for sig in capture_signals:
            if '{' not in sig:
                signal_data[sig] = self.fsdb_builder.dump_signal(sig)

        max_len = max(len(vals) for vals in signal_data.values()) if signal_data else 0
        log_format = task.get('logging')

        # Evaluate condition for each row
        matched_rows = []
        for row_idx in range(max_len):
            try:
                runtime_data = {'row_idx': row_idx, 'upstream_row': {}, 'upstream_data': {}}
                if self.cond_builder.exec(cond, runtime_data):
                    captured_vars = task.get('_captured_vars', {})

                    # Resolve capture signals with captured variables
                    actual_capture_signals = []
                    for sig in capture_signals:
                        if isinstance(sig, str) and '{' in sig:
                            actual_sig = sig
                            for var_name, var_val in captured_vars.items():
                                actual_sig = actual_sig.replace(f'{{{var_name}}}', var_val)
                            actual_sig = self.yaml_builder._resolve_signal_path(actual_sig, task.get('scope', ''), self.global_scope)
                            actual_capture_signals.append(actual_sig)
                        else:
                            actual_capture_signals.append(sig)

                    row_data = {'time': row_idx, 'signals': {}}
                    for sig in actual_capture_signals:
                        if sig in signal_data:
                            vals = signal_data[sig]
                            row_data['signals'][sig] = vals[row_idx] if row_idx < len(vals) else '0'
                        else:
                            try:
                                vals = self.fsdb_builder.dump_signal(sig)
                                row_data['signals'][sig] = vals[row_idx] if row_idx < len(vals) else '0'
                            except (ValueError, RuntimeError, KeyError):
                                row_data['signals'][sig] = '0'

                    matched_rows.append(row_data)

                    if log_format:
                        log_msg = self.yaml_builder.format_log_message(log_format, row_data, actual_capture_signals, row_idx)
                        print(f"  [LOG] {log_msg}")
            except Exception as e:
                print(f"[WARN] Error evaluating condition at row {row_idx}: {e}")

        return matched_rows

    def _trace_depends(self, task: dict[str, Any],
                             capture_signals: list[str],
                             condition: str) -> list[dict[str, Any]]:
        """Trace mode: match from upstream dependent task"""
        depends = task.get('dependsOn', [])
        dep_id = depends[0] if depends else None

        if dep_id not in self.runtime_data:
            raise ValueError(f"[ERROR] Upstream dependent task '{dep_id}' not found in task_data")

        upstream_data = self.runtime_data[dep_id]
        upstream_rows = upstream_data['rows']

        print(f"Tracing from upstream dependent task '{dep_id}' with {len(upstream_rows)} rows")
        if self.verbose and upstream_rows:
            print(f"  First upstream row: time={upstream_rows[0]['time']}, signals={list(upstream_rows[0]['signals'].keys())}")

        # Build condition once
        cond = self.cond_builder.build(condition, task, self.global_scope)

        signal_data = {}
        for sig in capture_signals:
            signal_data[sig] = self.fsdb_builder.dump_signal(sig)
        max_len = max(len(vals) for vals in signal_data.values()) if signal_data else 0

        log_format = task.get('logging')

        # For each upstream row, search forward
        matched_rows = []
        for trace_id, upstream_row in enumerate(upstream_rows):
            start_time = upstream_row['time']

            if self.verbose:
                print(f"  Searching from time {start_time}... (trace_id={trace_id})")

            # Search forward from start_time
            match_found = False
            for row_idx in range(start_time, max_len):
                try:
                    runtime_data = {'row_idx': row_idx, 'upstream_row': upstream_row, 'upstream_data': upstream_data}
                    if self.cond_builder.exec(cond, runtime_data):
                        row_data = {'time': row_idx, 'trace_id': trace_id, 'signals': {}}
                        for sig in capture_signals:
                            vals = signal_data[sig]
                            row_data['signals'][sig] = vals[row_idx] if row_idx < len(vals) else '0'
                        matched_rows.append(row_data)
                        match_found = True
                        if self.verbose:
                            print(f"    Found match at time {row_idx}")
                        if log_format:
                            log_msg = self.yaml_builder.format_log_message(log_format, row_data, capture_signals, row_idx)
                            print(f"  [LOG] {log_msg}")
                        break
                except Exception as e:
                    if self.verbose and row_idx == start_time:
                        print(f"    Error at time {row_idx}: {e}")
                    continue

            if not match_found:
                print(f"[WARN] No match found for upstream row at time {start_time}")

        return matched_rows

    def _capture_task(self, task: dict[str, Any], task_id: str) -> str:
        """Execute capture mode task"""
        condition = task.get('condition','')

        # Use pre-resolved capture signals from YamlBuilder
        capture_signals = task.get('resolved_capture', [])

        # Resolve $dep references in capture signals
        capture_signals = self.yaml_builder.resolve_dep_references(capture_signals, task_id, self.runtime_data)

        # Detect trace mode using pre-analyzed flag
        depends = task.get('dependsOn')
        has_dep = task.get('has_dep_in_condition', False)

        if depends and has_dep:
            # Trace: match from upstream
            matched_rows = self._trace_depends(task, capture_signals, condition)
        else:
            # Trigger: match all time (Start point)
            matched_rows = self._trace_trigger(task, capture_signals, condition)

        # Store to memory
        self.runtime_data[task_id] = {
            'rows': matched_rows,
            'signals': capture_signals  # All captured signals are available for reference
        }
        # print(matched_rows)
        # print(capture_signals)

        print(f"Matched {len(matched_rows)} rows")

        print(f"Result: {len(matched_rows)} rows in memory\n")
        return f"[Memory] {len(matched_rows)} rows"

   
    def run(self) -> None:
        """Execute all configured analysis tasks"""
        tasks = self.config.get('tasks', [])
        # Build execution order based on dependencies
        task_exec_order = self.yaml_builder.build_exec_order()

        print(f"\n{'='*70}")
        print(f"[INFO] FSDB Analyzer - Collected {len(tasks)} task(s)")
        print(f"{'='*70}")
        print(f"[INFO] FSDB file: {self.fsdb_file}")
        print(f"[INFO] Clock signal: {self.clock_signal}")
        print(f"[INFO] Output directory: {self.output_dir}")
        print(f"[INFO] Verbose mode: {'yes' if self.verbose else 'no'}")
        print(f"{'='*70}\n")

        # Collect all signals of interest from all tasks using YamlBuilder
        soi = self.yaml_builder.collect_signals(self.global_scope)
        self.fsdb_builder.dump_all_signals(soi)

        results = []
        for exec_idx, task_idx in enumerate(task_exec_order, 1):
            task = tasks[task_idx]
            task_id = task.get('id', f'task_{task_idx}')
            # Use name for display, fallback to id
            task_name = task.get('name') or task.get('id') or f'Task {exec_idx}'

            print(f"\n[Task {exec_idx}/{len(tasks)}] {task_name}")
            if task.get('dependsOn'):
                print(f"  Depends on: {', '.join(task['dependsOn'])}")
            print(f"{'-'*70}")

            try:
                result = self._capture_task(task, task_id)
                results.append((task_name, result))
            except Exception as e:
                print(f"Task failed: {e}")
                results.append((task_name, f"ERROR: {e}"))
        
        # Summary
        print(f"\n{'='*70}")
        print("Summary:")
        print(f"{'='*70}")
        for name, result in results:
            print(f"  {name}: {result}")
        print()

def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Advanced FSDB Signal Analyzer with Complex Conditions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run with advanced configuration
  %(prog)s -c ifu_analysis_advanced.yaml
        '''
    )
    
    parser.add_argument(
        '-c', '--config',
        required=True,
        help='YAML configuration file path'
    )
    
    args = parser.parse_args()
    
    analyzer = FsdbAnalyzer(args.config)
    analyzer.run()


if __name__ == '__main__':
    main()
