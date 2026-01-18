#!/usr/bin/env python3
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Tuple
from utils import resolve_signal_path, normalize_signal_name


class FsdbBuilder:
    """FSDB file parser using external tools"""

    def __init__(self, fsdb_file: Path, output_dir: Path, verbose: bool = False) -> None:
        self.fsdb_file: Path = fsdb_file
        self.verbose: bool = verbose
        self.output_dir: Path = output_dir
        self.signal_cache: Dict[str, List[str]] = {}  # Normalized name -> values
        self.signal_widths: Dict[str, Tuple[int, int]] = {}
        self.all_signals_list: List[str] = []
        self.signal_name_map: Dict[str, str] = {}  # Normalized name -> FSDB name with bitwidth
        self.timestamps: List[int] = []  # FSDB timestamps in 100fs units
        self._signal_vidcode_map: Dict[str, int] = {}  # signal_name -> vidcode

    def to_fsdb_path(self, signal: str) -> str:
        """Convert signal path from dot notation to FSDB format (slash)"""
        if signal.startswith('/'):
            return signal
        return '/' + signal.replace('.', '/')

    def get_signals_index(self) -> Dict[str, int]:
        """Get all signals from FSDB using fsdbdebug, returns signal_name -> vidcode mapping"""
        if self.all_signals_list:
            return {sig: self._signal_vidcode_map.get(sig, -1) for sig in self.all_signals_list}

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

            # Extract signal name, width, and vidcode from format:
            # Var: type name[bits] l:left r:right ... vidcode ...
            parts = line.split()
            if len(parts) < 4:
                continue
            sig_name = parts[2]

            # Find l:, r: values and vidcode (last integer before 1B)
            left, right, vidcode = None, None, None
            for i, part in enumerate(parts):
                if part.startswith('l:'):
                    left = int(part[2:])
                elif part.startswith('r:'):
                    right = int(part[2:])
                elif i > 3 and part.isdigit() and i + 1 < len(parts) and parts[i + 1] in ('1B', '0'):
                    vidcode = int(part)

            if left is not None and right is not None:
                self.signal_widths[sig_name] = (left, right)
            if vidcode is not None:
                self._signal_vidcode_map[sig_name] = vidcode

            signals.append(sig_name)

        self.all_signals_list = signals
        return {sig: self._signal_vidcode_map.get(sig, -1) for sig in signals}

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
        """Dump all signals using fsdbdebug -vc -vidcode"""
        if not signals:
            return

        # Expand patterns with {*} - this already handles bit ranges
        matched_signals = self.expand_raw_signals(signals)

        if not matched_signals:
            print("[WARN] No signals found in FSDB")
            return

        # Build mapping: normalized name (without bitwidth) -> FSDB name (with bitwidth)
        for sig in matched_signals:
            normalized = normalize_signal_name(sig)
            self.signal_name_map[normalized] = sig

        # Get vidcode mapping
        vidcode_map = self.get_signals_index()

        print(f"[INFO] Dumping {len(matched_signals)} signal(s) from FSDB using fsdbdebug...")

        # Initialize cache and timestamps
        for sig in matched_signals:
            normalized = normalize_signal_name(sig)
            self.signal_cache[normalized] = []
        self.timestamps = []

        # Extract value changes for each signal
        all_vc_data: Dict[str, Dict[int, str]] = {}  # normalized_sig -> {time_idx: value}
        time_set = set()

        # First pass: check all signals have vidcode
        missing_signals = []
        for sig in matched_signals:
            vidcode = vidcode_map.get(sig, -1)
            if vidcode == -1:
                missing_signals.append(sig)

        if missing_signals:
            error_msg = "[ERROR] The following signals do not exist in FSDB:\n"
            for sig in missing_signals:
                error_msg += f"  - {sig}\n"
            error_msg += "\nPlease check your YAML configuration for typos or incorrect signal names."
            raise RuntimeError(error_msg)

        # Second pass: extract value changes
        for sig in matched_signals:
            normalized = normalize_signal_name(sig)
            vidcode = vidcode_map.get(sig, -1)

            # Get bit width for this signal
            left, right = self.signal_widths.get(sig, (0, 0))
            bit_width = abs(left - right) + 1

            cmd = ['fsdbdebug', '-vc', '-vidcode', str(vidcode), str(self.fsdb_file.absolute())]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True, timeout=120)

            if result.returncode != 0:
                print(f"[WARN] Failed to dump signal {sig}: {result.stderr}")
                continue

            # Parse output: "N vc: xtag: (0 time)  val: binary_value"
            output = result.stderr if result.stderr else result.stdout
            vc_data = {}

            for line in output.split('\n'):
                if 'vc:' not in line or 'xtag:' not in line or 'val:' not in line:
                    continue

                # Parse: "N vc: xtag: (0 time)  val: binary_value"
                match = re.search(r'xtag:\s*\(\s*\d+\s+(\d+)\)\s+val:\s*([01xzXZ]+)', line)
                if match:
                    time_val = int(match.group(1))
                    binary_val = match.group(2)
                    # Convert binary to hex with proper padding
                    hex_val = self._binary_to_hex(binary_val, bit_width)
                    vc_data[time_val] = hex_val
                    time_set.add(time_val)

            all_vc_data[normalized] = vc_data

        # Build unified time series
        self.timestamps = sorted(list(time_set))

        # Fill signal cache with values at each timestamp (forward-fill)
        for sig in matched_signals:
            normalized = normalize_signal_name(sig)
            vc_data = all_vc_data.get(normalized, {})

            # Get bit width for proper initial value padding
            left, right = self.signal_widths.get(sig, (0, 0))
            bit_width = abs(left - right) + 1
            hex_chars = (bit_width + 3) // 4  # Round up to nearest hex char
            current_val = '0' * hex_chars if hex_chars > 0 else '0'  # Default padded initial value

            for time in self.timestamps:
                if time in vc_data:
                    current_val = vc_data[time]
                self.signal_cache[normalized].append(current_val)

        # Write verbose output if enabled
        if self.verbose and self.output_dir:
            self._write_verbose_output(matched_signals)

    def _write_verbose_output(self, signals: List[str]) -> None:
        """Write signal dump to file in verbose mode"""
        output_file = self.output_dir / 'fsdb_dump.txt'
        try:
            with open(output_file, 'w') as f:
                # Write header
                header = "Time".ljust(15)
                for sig in signals:
                    sig_name = sig.split('.')[-1].split('/')[-1]
                    header += sig_name[:20].ljust(22)
                f.write(header + '\n')
                f.write('-' * len(header) + '\n')

                # Write data rows
                for idx, time in enumerate(self.timestamps):
                    row = str(time).ljust(15)
                    for sig in signals:
                        normalized = normalize_signal_name(sig)
                        if normalized in self.signal_cache:
                            val = self.signal_cache[normalized][idx] if idx < len(self.signal_cache[normalized]) else '0'
                            row += val[:20].ljust(22)
                        else:
                            row += '0'.ljust(22)
                    f.write(row + '\n')

            print(f"[INFO] Verbose dump written to: {output_file}")
        except Exception as e:
            print(f"[WARN] Failed to write verbose output: {e}")

    def _binary_to_hex(self, binary_str: str, bit_width: int = 0) -> str:
        """Convert binary string to hex string (without 0x prefix)

        Args:
            binary_str: Binary string from FSDB
            bit_width: Signal bit width for proper padding (0 = no padding)

        Returns:
            Hex string with proper zero padding based on bit width
        """
        # Calculate expected hex character count from bit width
        expected_hex_chars = (bit_width + 3) // 4 if bit_width > 0 else 0

        # Handle x/z values
        if 'x' in binary_str.lower() or 'z' in binary_str.lower():
            # Return as-is for x/z values with padding marker
            result = '*' + binary_str[:8] if len(binary_str) > 8 else '*' + binary_str
            # Pad if needed
            if expected_hex_chars > 0 and len(result) - 1 < expected_hex_chars:
                result = '*' + binary_str.ljust(expected_hex_chars, 'x')
            return result

        try:
            # Pad to multiple of 4 bits
            padding = (4 - len(binary_str) % 4) % 4
            binary_str = '0' * padding + binary_str
            # Convert to hex
            hex_val = hex(int(binary_str, 2))[2:]  # Remove '0x'
            hex_val = hex_val.upper()

            # Pad to expected length based on bit width
            if expected_hex_chars > 0:
                hex_val = hex_val.zfill(expected_hex_chars)

            return hex_val
        except ValueError:
            # Return padded zero on error
            return '0' * expected_hex_chars if expected_hex_chars > 0 else '0'

