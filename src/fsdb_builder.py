#!/usr/bin/env python3
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base import Signal, resolve_signal_path


class FsdbBuilder:
    """FSDB parser backed exclusively by wavekit/Verdi NPI."""

    def __init__(self, fsdb_file: Path, output_dir: Path, verbose: bool = False) -> None:
        self.fsdb_file: Path = fsdb_file
        self.output_dir: Path = output_dir
        self.verbose: bool = verbose

        self.timestamps: List[int] = []
        self._signals: Dict[str, Signal] = {}
        self._signals_list: List[str] = []
        self._reader: Any = None

    def _open_reader(self) -> Any:
        """Open the FSDB through wavekit, or fail with actionable setup help."""
        if self._reader is not None:
            return self._reader

        try:
            import wavekit
            from wavekit import FsdbReader
        except Exception as exc:
            raise RuntimeError(
                "wavekit is required for FSDB access. Install it in the project "
                "environment with `.venv/bin/pip3 install wavekit`."
            ) from exc

        try:
            if not wavekit.has_fsdb_support():
                raise RuntimeError("wavekit reports FSDB support is unavailable")
            self._reader = FsdbReader(str(self.fsdb_file.absolute()))
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize wavekit FSDB runtime. Configure Verdi NPI with "
                "VERDI_HOME, WAVEKIT_NPI_LIB, or LD_LIBRARY_PATH, then retry."
            ) from exc

        return self._reader

    @staticmethod
    def _with_range_suffix(signal: Any) -> str:
        """Return an APV-compatible dotted signal path, including vector ranges."""
        full_name = signal.full_name
        if re.search(r"\[\d+(?::\d+)?\]$", full_name):
            return full_name

        if signal.range is not None:
            high, low = signal.range
            return f"{full_name}[{high}:{low}]"

        if signal.width and signal.width > 1:
            return f"{full_name}[{signal.width - 1}:0]"

        return full_name

    def _collect_signal_paths(self, scope: Any) -> List[str]:
        paths: List[str] = []

        for sig in scope.signal_list:
            try:
                members = sig.member_list
            except Exception:
                members = None

            if members:
                for member in members:
                    paths.extend(self._collect_member_paths(member))
            else:
                paths.append(self._with_range_suffix(sig))

        for child in scope.child_scope_list:
            paths.extend(self._collect_signal_paths(child))

        return paths

    def _collect_member_paths(self, signal: Any) -> List[str]:
        try:
            members = signal.member_list
        except Exception:
            members = None

        if not members:
            return [self._with_range_suffix(signal)]

        paths: List[str] = []
        for member in members:
            paths.extend(self._collect_member_paths(member))
        return paths

    def get_signals_index(self) -> Dict[str, int]:
        """Get all FSDB signal paths in APV's dotted-name format."""
        if not self._signals_list:
            reader = self._open_reader()
            signals: List[str] = []
            for top_scope in reader.top_scope_list():
                signals.extend(self._collect_signal_paths(top_scope))
            self._signals_list = sorted(set(signals))

        # Compatibility map: callers only need membership and keys.
        return {sig: -1 for sig in self._signals_list}

    def get_signal(self, signal: str) -> Signal:
        """Get single cached Signal object by normalized name."""
        if signal in self._signals:
            return self._signals[signal]
        raise RuntimeError(f"Signal {signal} not found in cache. Call dump_signals first.")

    def expand_pattern(self, raw_signals: List[str]) -> List[str]:
        """Expand patterns and resolve bit-ranges for a list of signals."""
        all_sigs = self.get_signals_index().keys()
        expanded = []
        for sig in raw_signals:
            if "{*}" in sig:
                pattern = "^" + re.escape(sig).replace(r"\{\*\}", r"[a-zA-Z0-9_$]+") + r"(?:\[\d+:\d+\])?$"
                expanded.extend([s for s in all_sigs if re.match(pattern, s)])
            else:
                match = next((s for s in all_sigs if s == sig or s.startswith(sig + "[")), sig)
                expanded.append(match)
        return expanded

    def resolve_pattern(self, pattern: str, scope: str = "") -> Tuple[List[str], List[str]]:
        """Resolve a pattern with variables to matched signal names and values."""
        resolved_pattern = resolve_signal_path(pattern, scope)
        var_match = re.search(r'\{(\w+)\}', resolved_pattern)
        if not var_match:
            expanded = self.expand_pattern([resolved_pattern])
            return expanded, []

        var_name = var_match.group(1)
        wildcard_pattern = re.sub(r'\{[^}]+\}', '{*}', resolved_pattern)
        expanded_signals = self.expand_pattern([wildcard_pattern])

        extract_regex = re.escape(resolved_pattern).replace(
            re.escape(f'{{{var_name}}}'), r'([a-zA-Z0-9_$]+)'
        )
        extract_regex = f'^{extract_regex}(?:\\[\\d+:\\d+\\])?$'

        matched_signals = []
        possible_vals = set()
        for sig in expanded_signals:
            match = re.match(extract_regex, sig)
            if match:
                matched_signals.append(sig)
                possible_vals.add(match.group(1))

        return matched_signals, sorted(list(possible_vals))

    def dump_signals(self, signals: List[str]) -> None:
        """Dump all signals-of-interest using wavekit/Verdi NPI."""
        if not signals:
            print("[WARN] No signals provided, skipping FSDB dump")
            return

        signal_index = self.get_signals_index()
        self._signals = {}
        matched_signals = []
        missing_signals = []

        for raw_sig in signals:
            for matched_name in self.expand_pattern([raw_sig]):
                normalized = Signal.normalize(matched_name)
                if normalized in self._signals:
                    continue

                if matched_name not in signal_index:
                    missing_signals.append(matched_name)
                    continue

                sig = Signal(raw_name=matched_name)
                self._signals[normalized] = sig
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

        print(f"[INFO] Dumping {len(matched_signals)} signal(s) from FSDB using wavekit...")

        time_set = set()
        self.timestamps = []
        all_vc_data: Dict[str, Dict[int, str]] = {}

        for normalized in matched_signals:
            normalized, vc_data, local_time_set = self._dump_single_signal(normalized)
            if vc_data:
                all_vc_data[normalized] = vc_data
                time_set.update(local_time_set)

        self.timestamps = sorted(list(time_set))

        for normalized in matched_signals:
            signal_obj = self._signals[normalized]
            vc_data = all_vc_data.get(normalized, {})
            signal_obj.set_waveform(self.timestamps, vc_data)

        if self.verbose and self.output_dir:
            self._write_verbose_output(matched_signals)

    @staticmethod
    def _value_to_hex(value: Any, bit_width: int) -> str:
        hex_chars = max(1, (bit_width + 3) // 4)
        return f"{int(value):0{hex_chars}X}"

    def _dump_single_signal(self, normalized: str):
        """Load one signal's value changes through wavekit/Verdi NPI."""
        reader = self._open_reader()
        signal_obj = self._signals[normalized]

        try:
            npi_signal = reader.file_handle.get_signal(signal_obj.raw_name)
            value_changes = reader.file_handle.load_value_change(
                npi_signal,
                begin_time=0,
                end_time=reader.file_handle.max_time(),
                xz_value=0,
            )
        except Exception as exc:
            print(f"[WARN] Failed to dump signal {normalized}: {exc}")
            return normalized, {}, set()

        bit_width = npi_signal.width()
        vc_data = {}
        local_time_set = set()

        for time_val, value in value_changes:
            time_int = int(time_val)
            vc_data[time_int] = self._value_to_hex(value, bit_width)
            local_time_set.add(time_int)

        return normalized, vc_data, local_time_set

    def _write_verbose_output(self, signals: List[str]) -> None:
        """Write signal dump to file in verbose mode."""
        output_file = self.output_dir / 'fsdb_dump.txt'
        try:
            with open(output_file, 'w') as f:
                header = "Time".ljust(15)
                for norm_name in signals:
                    sig_name = norm_name.split('.')[-1].split('/')[-1]
                    header += sig_name[:20].ljust(22)
                f.write(header + '\n')
                f.write('-' * len(header) + '\n')

                for idx, time in enumerate(self.timestamps):
                    row = str(time).ljust(15)
                    for norm_name in signals:
                        if norm_name in self._signals:
                            row += self._signals[norm_name].get_value(idx)[:20].ljust(22)
                        else:
                            row += '0'.ljust(22)
                    f.write(row + '\n')

            print(f"[INFO] Verbose dump written to: {output_file}")
        except Exception as e:
            print(f"[WARN] Failed to write verbose output: {e}")
