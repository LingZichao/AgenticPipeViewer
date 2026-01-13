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
        self.signal_cache: Dict[str, List[str]] = {}  # Normalized name -> values
        self.signal_widths: Dict[str, tuple[int, int]] = {}
        self.all_signals_list: List[str] = []
        self.signal_name_map: Dict[str, str] = {}  # Normalized name -> FSDB name with bitwidth

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

        # fsdbdebug outputs to stderr, not stdout
        output = result.stderr if result.stderr else result.stdout
        signals = []
        for line in output.split('\n'):
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

    def expand_raw_signals(self, raw_signals: List[str]) -> List[str]:
        """Expand raw signals containing {*} patterns to actual signal names

        Note: This method handles two types of bit ranges:
        1. FSDB signal bit width: e.g., signal[127:0] in FSDB index
        2. Pattern with bit range: e.g., signal{*}[31:0] (bit range is NOT expanded)

        Args:
            raw_signals: List of signal patterns, may contain {*} wildcards

        Returns:
            List of actual signal names with patterns expanded
        """
        all_signals = self.get_signals_index()
        expanded = []

        for sig in raw_signals:
            if "{*}" in sig:
                # Convert pattern to regex: signal{*} -> signal[a-zA-Z0-9_$]+
                # Match any valid Verilog identifier characters
                regex_pattern = "^" + re.escape(sig).replace(r"\{\*\}", r"[a-zA-Z0-9_$]+")
                regex_pattern += r"(?:\[\d+:\d+\])?$"  # Allow optional bit range

                matches = []
                for fsdb_sig in all_signals:
                    sig_dot = fsdb_sig.replace("/", ".").lstrip(".")
                    if re.match(regex_pattern, sig_dot):
                        matches.append(sig_dot)
                        expanded.append(sig_dot)
            else:
                # For non-pattern signals, try to match with bit range
                matched = False
                for fsdb_sig in all_signals:
                    sig_dot = fsdb_sig.replace("/", ".").lstrip(".")
                    # Exact match or match with bit range
                    if sig_dot == sig or sig_dot.startswith(sig + "["):
                        expanded.append(sig_dot)
                        matched = True
                        break
                if not matched:
                    expanded.append(sig)

        return expanded

    def dump_signals(self, signals: List[str]) -> None:
        """Dump all signals at once using single fsdbreport call"""
        if not signals:
            return

        # Expand patterns with {*} - this already handles bit ranges
        matched_signals = self.expand_raw_signals(signals)

        print(f"[DEBUG] Total signals after expansion: {len(matched_signals)}")
        for sig in matched_signals:
            print(f"  - {sig}")

        if not matched_signals:
            print("[WARN] No signals found in FSDB")
            return

        # Build mapping: normalized name (without bitwidth) -> FSDB name (with bitwidth)
        for sig in matched_signals:
            normalized = re.sub(r'\[\d+:\d+\]$', '', sig)  # Remove trailing [msb:lsb]
            self.signal_name_map[normalized] = sig

        fsdb_paths = [self.to_fsdb_path(sig) for sig in matched_signals]

        tmp_file = self.output_dir / '.fsdb_dump_tmp.txt'

        try:
            cmd = ['fsdbreport', str(self.fsdb_file.absolute()), '-of', 'h', '-w', '1024', '-s'] + fsdb_paths + ['-o', tmp_file]

            print(f"[INFO] Dumping {len(matched_signals)} signal(s) from FSDB...")
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

            # Initialize cache for original signal names (without bit ranges)
            for sig in matched_signals:
                normalized = re.sub(r'\[\d+:\d+\]$', '', sig)
                self.signal_cache[normalized] = []

            data_start = header_idx + 2

            for line in lines[data_start:]:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue

                for idx, val in enumerate(parts[1:]):
                    if idx < len(matched_signals):
                        normalized = re.sub(r'\[\d+:\d+\]$', '', matched_signals[idx])
                        self.signal_cache[normalized].append(val)

        finally:
            if not self.verbose and os.path.exists(tmp_file):
                os.unlink(tmp_file)

