#!/usr/bin/env python3
import subprocess
import re
import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from .utils import Signal, resolve_signal_path


class FsdbBuilder:
    """FSDB file parser using external tools"""

    def __init__(self, fsdb_file: Path, output_dir: Path, verbose: bool = False) -> None:
        self.fsdb_file: Path  = fsdb_file
        self.output_dir: Path = output_dir
        self.verbose: bool    = verbose

        self.signals: Dict[str, Signal] = {}  # Normalized name -> Signal object
        self.timestamps: List[int] = []  # FSDB timestamps in 100fs units
        self._signals_list: List[str] = [] # All signals name in FSDB
        self._signals_vidcode_map: Dict[str, int] = {}  # signal_name -> vidcode

    def get_signals_index(self) -> Dict[str, int]:
        """Get all signals from FSDB, returns normalized_name -> vidcode mapping"""
        if self._signals_list:
            return {sig: self._signals_vidcode_map.get(sig, -1) for sig in self._signals_list}

        # If signals list is empty, build it from fsdbdebug
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

            # Extract signal name and vidcode
            parts = line.split()
            if len(parts) < 4:
                continue
            # Normalize FSDB slash-separated path to dot notation
            sig_name = parts[2].replace("/", ".").lstrip(".")

            # Find vidcode (last integer before 1B)
            vidcode = None
            for i, part in enumerate(parts):
                if i > 3 and part.isdigit() and i + 1 < len(parts) and parts[i + 1] in ('1B', '0'):
                    vidcode = int(part)
                    break

            if vidcode is not None:
                self._signals_vidcode_map[sig_name] = vidcode

            signals.append(sig_name)

        self._signals_list = signals
        return {sig: self._signals_vidcode_map.get(sig, -1) for sig in signals}

    def get_signal(self, signal: str) -> List[str]:
        """Get single cached signal values"""
        if signal in self.signals:
            return self.signals[signal].values
        raise RuntimeError(f"Signal {signal} not found in cache. Call dump_signals first.")

    def expand_pattern(self, raw_signals: List[str]) -> List[str]:
        """Expand patterns and resolve bit-ranges for a list of signals"""
        all_sigs = self.get_signals_index().keys()
        expanded = []
        for sig in raw_signals:
            if "{*}" in sig:
                # Convert pattern to regex: signal{*} -> signal[a-zA-Z0-9_$]+
                pattern = "^" + re.escape(sig).replace(r"\{\*\}", r"[a-zA-Z0-9_$]+") + r"(?:\[\d+:\d+\])?$"
                expanded.extend([s for s in all_sigs if re.match(pattern, s)])
            else:
                # Match exact name or name with bit range (e.g., sig -> sig[31:0])
                match = next((s for s in all_sigs if s == sig or s.startswith(sig + "[")), sig)
                expanded.append(match)
        return expanded

    def resolve_pattern(self, pattern: str, scope: str = "") -> Tuple[List[str], List[str]]:
        """Resolve a pattern with variables to matched signal names and variable values
        
        Example: "ifu{idx}.vld" -> (['tb.ifu0.vld', 'tb.ifu1.vld'], ['0', '1'])
        """
        # Step 1: Resolve pattern with scope, then convert {variable} to {*}
        resolved_pattern = resolve_signal_path(pattern, scope)
        var_match = re.search(r'\{(\w+)\}', resolved_pattern)
        if not var_match:
            # No pattern variable, just expand as normal
            expanded = self.expand_pattern([resolved_pattern])
            return expanded, []

        var_name = var_match.group(1)
        wildcard_pattern = re.sub(r'\{[^}]+\}', '{*}', resolved_pattern)
        expanded_signals = self.expand_pattern([wildcard_pattern])

        # Step 2: Build regex to extract variable value from actual signal names
        # Escape the pattern but keep the variable part as a group
        extract_regex = re.escape(resolved_pattern).replace(
            re.escape(f'{{{var_name}}}'), r'([a-zA-Z0-9_$]+)'
        )
        # Allow optional bit range at the end
        extract_regex = f'^{extract_regex}(?:\\[\\d+:\\d+\\])?$'

        # Step 3: Extract variable values and signals
        matched_signals = []
        possible_vals = set()
        for sig in expanded_signals:
            match = re.match(extract_regex, sig)
            if match:
                matched_signals.append(sig)
                possible_vals.add(match.group(1))

        return matched_signals, sorted(list(possible_vals))

    def dump_signals(self, signals: List[str]) -> None:
        """Dump all signals-of-interest using fsdbdebug -vc -vidcode"""
        if not signals:
            print("[WARN] No signals provided, skipping FSDB dump")
            return

        vidcode_map = self.get_signals_index()
        self.signals = {}
        matched_signals = []
        missing_signals = []

        for raw_sig in signals:
            # Expand pattern signals or resolve bit-ranges for direct signals
            for matched_name in self.expand_pattern([raw_sig]):
                normalized = Signal.normalize(matched_name)
                if normalized not in self.signals:
                    vidcode = vidcode_map.get(matched_name, -1)
                    if vidcode == -1:
                        missing_signals.append(matched_name)
                        continue

                    self.signals[normalized] = Signal(
                        raw_name=matched_name,
                        vidcode=vidcode,
                        values=[]
                    )
                    matched_signals.append(normalized)

        if not matched_signals:
            print("[WARN] No signals found in FSDB")
            return

        if missing_signals:
            error_msg = "[ERROR] The following signals do not exist in FSDB:\n"
            for sig in missing_signals:
                error_msg += f"  - {sig}\n"
            error_msg += "\nPlease check your YAML configuration for typos or incorrect signal names."
            raise RuntimeError(error_msg)

        print(f"[INFO] Dumping {len(matched_signals)} signal(s) from FSDB using fsdbdebug...")

        # Extract value changes for each signal
        time_set = set()
        self.timestamps = []
        all_vc_data: Dict[str, Dict[int, str]] = {}  # normalized_sig -> {time_idx: value}

        # pass: extract value changes
        # Calculate workers: ~16 signals per thread, capped at 32
        num_workers = min(32, max(1, (len(matched_signals) + 15) // 16))
        print(f"[INFO] Using {num_workers} threads for parallel dumping ({len(matched_signals)} signals)...")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_norm = {executor.submit(self._dump_single_signal, norm): norm for norm in matched_signals}
            for future in concurrent.futures.as_completed(future_to_norm):
                normalized, vc_data, local_time_set = future.result()
                if vc_data:
                    all_vc_data[normalized] = vc_data
                    time_set.update(local_time_set)

        # Build unified time series
        self.timestamps = sorted(list(time_set))

        # Fill Signal objects with values at each timestamp (forward-fill)
        for normalized in matched_signals:
            signal_obj = self.signals[normalized]
            vc_data = all_vc_data.get(normalized, {})
            signal_obj.set_waveform(self.timestamps, vc_data)

        # Write verbose output if enabled
        if self.verbose and self.output_dir:
            self._write_verbose_output(matched_signals)

    def _dump_single_signal(self, normalized: str):
        """Internal helper for parallel signal dumping"""
        signal_obj = self.signals[normalized]
        cmd = ['fsdbdebug', '-vc', '-vidcode', str(signal_obj.vidcode), str(self.fsdb_file.absolute())]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, timeout=120)

        if result.returncode != 0:
            print(f"[WARN] Failed to dump signal {normalized}: {result.stderr}")
            return normalized, {}, set()

        output = result.stderr if result.stderr else result.stdout
        vc_data = {}
        local_time_set = set()

        for line in output.split('\n'):
            if 'vc:' not in line or 'xtag:' not in line or 'val:' not in line:
                continue

            match = re.search(r'xtag:\s*\(\s*\d+\s+(\d+)\)\s+val:\s*([01xzXZ]+)', line)
            if match:
                time_val = int(match.group(1))
                binary_val = match.group(2)
                hex_val = Signal.binary_to_hex(binary_val, signal_obj.bit_width)
                vc_data[time_val] = hex_val
                local_time_set.add(time_val)

        return normalized, vc_data, local_time_set

    def _write_verbose_output(self, signals: List[str]) -> None:
        """Write signal dump to file in verbose mode"""
        output_file = self.output_dir / 'fsdb_dump.txt'
        try:
            with open(output_file, 'w') as f:
                # Write header
                header = "Time".ljust(15)
                for norm_name in signals:
                    sig_name = norm_name.split('.')[-1].split('/')[-1]
                    header += sig_name[:20].ljust(22)
                f.write(header + '\n')
                f.write('-' * len(header) + '\n')

                # Write data rows
                for idx, time in enumerate(self.timestamps):
                    row = str(time).ljust(15)
                    for norm_name in signals:
                        if norm_name in self.signals:
                            row += self.signals[norm_name].get_value(idx)[:20].ljust(22)
                        else:
                            row += '0'.ljust(22)
                    f.write(row + '\n')

            print(f"[INFO] Verbose dump written to: {output_file}")
        except Exception as e:
            print(f"[WARN] Failed to write verbose output: {e}")

    # Removed _binary_to_hex as it's now in Signal class

