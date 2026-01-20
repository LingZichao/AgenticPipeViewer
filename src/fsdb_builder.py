#!/usr/bin/env python3
import subprocess
import re
from pathlib import Path
from typing import List, Dict
from utils import Signal


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

    def to_fsdb_path(self, signal: str) -> str:
        """Convert signal path from dot notation to FSDB format (slash)"""
        if signal.startswith('/'):
            return signal
        return '/' + signal.replace('.', '/')

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
        """Get cached signal values"""
        if signal in self.signals:
            return self.signals[signal].values
        raise RuntimeError(f"Signal {signal} not found in cache. Call dump_signals first.")

    def expand_raw_signals(self, raw_signals: List[str]) -> List[str]:
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

    def dump_signals(self, signals: List[str]) -> None:
        """Dump all signals-of-interest using fsdbdebug -vc -vidcode"""
        if not signals:
            print("[WARN] No signals provided, skipping FSDB dump")
            return

        vidcode_map = self.get_signals_index()
        self.signals = {}
        matched_signals = []

        for raw_sig in signals:
            # Expand pattern signals or resolve bit-ranges for direct signals
            for matched_name in self.expand_raw_signals([raw_sig]):
                normalized = Signal.normalize(matched_name)
                if normalized not in self.signals:
                    vidcode = vidcode_map.get(matched_name, -1)
                    self.signals[normalized] = Signal(
                        raw_name=matched_name,
                        vidcode=vidcode,
                        values=[]
                    )
                    matched_signals.append(matched_name)

        if not matched_signals:
            print("[WARN] No signals found in FSDB")
            return

        print(f"[INFO] Dumping {len(matched_signals)} signal(s) from FSDB using fsdbdebug...")

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
            normalized = Signal.normalize(sig)
            signal_obj = self.signals[normalized]
            
            cmd = ['fsdbdebug', '-vc', '-vidcode', str(signal_obj.vidcode), str(self.fsdb_file.absolute())]
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
                    hex_val = Signal.binary_to_hex(binary_val, signal_obj.bit_width)
                    vc_data[time_val] = hex_val
                    time_set.add(time_val)

            all_vc_data[normalized] = vc_data

        # Build unified time series
        self.timestamps = sorted(list(time_set))

        # Fill Signal objects with values at each timestamp (forward-fill)
        for sig in matched_signals:
            normalized = Signal.normalize(sig)
            signal_obj = self.signals[normalized]
            vc_data = all_vc_data.get(normalized, {})
            signal_obj.set_waveform(self.timestamps, vc_data)

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
                    normalized = Signal.normalize(sig)
                    sig_name = normalized.split('.')[-1].split('/')[-1]
                    header += sig_name[:20].ljust(22)
                f.write(header + '\n')
                f.write('-' * len(header) + '\n')

                # Write data rows
                for idx, time in enumerate(self.timestamps):
                    row = str(time).ljust(15)
                    for sig in signals:
                        normalized = Signal.normalize(sig)
                        if normalized in self.signals:
                            row += self.signals[normalized].get_value(idx)[:20].ljust(22)
                        else:
                            row += '0'.ljust(22)
                    f.write(row + '\n')

            print(f"[INFO] Verbose dump written to: {output_file}")
        except Exception as e:
            print(f"[WARN] Failed to write verbose output: {e}")

    # Removed _binary_to_hex as it's now in Signal class

