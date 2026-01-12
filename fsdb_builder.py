#!/usr/bin/env python3
import subprocess
import os
import re
from pathlib import Path
from typing import List, Dict
from utils import resolve_signal_path


class FsdbBuilder:
    """FSDB file parser using external tools"""

    def __init__(self, fsdb_file: Path, output_dir: Path, verbose: bool = False) -> None:
        self.fsdb_file: Path = fsdb_file
        self.verbose: bool = verbose
        self.output_dir: Path = output_dir
        self.signal_cache: Dict[str, List[str]] = {}
        self.signal_widths: Dict[str, tuple[int, int]] = {}
        self.all_signals_list: List[str] = []

    def to_fsdb_path(self, signal: str) -> str:
        """Convert signal path from dot notation to FSDB format (slash)"""
        if signal.startswith('/'):
            return signal
        return '/' + signal.replace('.', '/')

    def get_signals_index(self) -> List[str]:
        """Get all signals from FSDB using fsdbdebug"""
        if self.all_signals_list:
            return self.all_signals_list

        cmd = ['fsdbdebug', '-hier_tree', str(self.fsdb_file.absolute())]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get signal hierarchy: {result.stderr}")

        signals = []
        for line in result.stdout.split('\n'):
            line = line.strip()
            if not line.startswith('Var:'):
                continue

            # Extract signal name and width from format: Var: type name l:left r:right ...
            parts = line.split()
            if len(parts) < 4:
                continue
            sig_name = parts[2]
            # Find l: and r: values
            left, right = None, None
            for part in parts:
                if part.startswith('l:'):
                    left = int(part[2:])
                elif part.startswith('r:'):
                    right = int(part[2:])

            if left is not None and right is not None:
                self.signal_widths[sig_name] = (left, right)
            signals.append(sig_name)

        self.all_signals_list = signals
        return signals

    def get_signal(self, signal: str) -> List[str]:
        """Get cached signal values"""
        if signal in self.signal_cache:
            return self.signal_cache[signal]
        raise RuntimeError(f"Signal {signal} not found in cache. Call dump_signals first.")

    def dump_signals(self, signals: List[str]) -> None:
        """Dump all signals at once using single fsdbreport call"""
        if not signals:
            return

        fsdb_paths = [self.to_fsdb_path(sig) for sig in signals]

        tmp_file = self.output_dir / '.fsdb_dump_tmp.txt'

        try:
            cmd = ['fsdbreport', str(self.fsdb_file.absolute()), '-of', 'h', '-w', '1024', '-s'] + fsdb_paths + ['-o', tmp_file]

            print(f"[INFO] Dumping {len(signals)} signal(s) from FSDB...")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to dump signals: {result.stderr}")

            with open(tmp_file, 'r') as f:
                lines = f.readlines()

            if self.verbose and self.output_dir:
                output_file = self.output_dir / 'fsdb_dump.txt'
                with open(output_file, 'w') as f:
                    f.writelines(lines)

            header_idx = -1
            for i, line in enumerate(lines):
                if 'Time' in line:
                    header_idx = i
                    break

            if header_idx == -1:
                raise RuntimeError("Cannot find header in fsdbreport output")

            for sig in signals:
                self.signal_cache[sig] = []

            data_start = header_idx + 2

            for line in lines[data_start:]:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue

                for idx, val in enumerate(parts[1:]):
                    if idx < len(signals):
                        self.signal_cache[signals[idx]].append(val)

        finally:
            if not self.verbose and os.path.exists(tmp_file):
                os.unlink(tmp_file)

    def find_matching_signals(self, pattern: str, task_scope: str, global_scope: str) -> List[tuple[str, Dict[str, str]]]:
        """Find signals matching a pattern with {variable} placeholders"""
        pattern = resolve_signal_path(pattern, task_scope, global_scope)
        var_pattern = re.findall(r"\{(\w+)\}", pattern)
        if not var_pattern:
            return [(pattern, {})]

        regex_pattern = "^" + re.escape(pattern)
        for var in var_pattern:
            regex_pattern = regex_pattern.replace(re.escape(f"{{{var}}}"), r"(\d+)")
        regex_pattern += "$"

        matches = []
        for sig in self.all_signals_list:
            sig_dot = sig.replace("/", ".").lstrip(".")
            match = re.match(regex_pattern, sig_dot)
            if match:
                captured = {var: match.group(i + 1) for i, var in enumerate(var_pattern)}
                matches.append((sig_dot, captured))
        return matches
