#!/usr/bin/env python3
"""Utility functions shared across modules"""


import re
from dataclasses import dataclass, field
from typing import List, Union, Sequence, Optional, Dict


@dataclass
class Signal:
    """Represents a signal with its metadata and captured values"""
    raw_name: str  # Original name from FSDB (may include bitwidth)
    name: str = field(init=False)  # Normalized name without bit range
    scope: str = ""
    msb: int = 0
    lsb: int = 0
    vidcode: int = -1
    values: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize normalized name and other metadata after construction"""
        self.name = self.normalize(self.raw_name)
        
        # Parse MSB/LSB from raw_name if not already set
        if self.msb == 0 and self.lsb == 0:
            match = re.search(r'\[(\d+):(\d+)\]$', self.raw_name)
            if match:
                self.msb = int(match.group(1))
                self.lsb = int(match.group(2))

    @staticmethod
    def normalize(name: str) -> str:
        """Remove bit range from signal name (e.g., 'sig[127:0]' -> 'sig')"""
        return re.sub(r'\[\d+:\d+\]$', '', name)

    @property
    def fsdb_path(self) -> str:
        """Convert dot notation to FSDB slash format"""
        # Use raw_name for FSDB operations
        if self.raw_name.startswith('/'):
            return self.raw_name
        return '/' + self.raw_name.replace('.', '/')

    @property
    def bit_width(self) -> int:
        """Calculate bit width from MSB and LSB"""
        return abs(self.msb - self.lsb) + 1

    def get_value(self, time_idx: int) -> str:
        """Get hex value string at specific time index"""
        if time_idx < len(self.values):
            return self.values[time_idx]
        return "0"

    def get_int_value(self, time_idx: int) -> int:
        """Get integer value at specific time index"""
        val_str = self.get_value(time_idx)
        if val_str in ["x", "z", "X", "Z"] or val_str.startswith("*"):
            return 0
        if not val_str.startswith("0x") and not val_str.startswith("0X"):
            val_str = "0x" + val_str
        return int(val_str, 16)

    def set_waveform(self, timestamps: List[int], vc_data: Dict[int, str]):
        """Fill waveform values with forward-fill logic based on global timestamps"""
        self.values = []
        bit_width = self.bit_width
        hex_chars = (bit_width + 3) // 4
        current_val = '0' * hex_chars if hex_chars > 0 else '0'

        for time in timestamps:
            if time in vc_data:
                current_val = vc_data[time]
            self.values.append(current_val)

    @staticmethod
    def binary_to_hex(binary_str: str, bit_width: int = 0) -> str:
        """Convert binary string to hex string (without 0x prefix)"""
        expected_hex_chars = (bit_width + 3) // 4 if bit_width > 0 else 0

        if 'x' in binary_str.lower() or 'z' in binary_str.lower():
            result = '*' + binary_str[:8] if len(binary_str) > 8 else '*' + binary_str
            if expected_hex_chars > 0 and len(result) - 1 < expected_hex_chars:
                result = '*' + binary_str.ljust(expected_hex_chars, 'x')
            return result

        try:
            padding = (4 - len(binary_str) % 4) % 4
            binary_str = '0' * padding + binary_str
            hex_val = hex(int(binary_str, 2))[2:].upper()
            if expected_hex_chars > 0:
                hex_val = hex_val.zfill(expected_hex_chars)
            return hex_val
        except ValueError:
            return '0' * expected_hex_chars if expected_hex_chars > 0 else '0'


@dataclass
class SignalGroup:
    """Represents a group of signal values or pattern variable captures

    Used for:
    - Signal values from $split() operations (values are ints)
    - Pattern variable captures like {idx} (values are strings)
    """
    values: List[Union[int, str]]

    def __init__(self, values: Sequence[Union[int, str]]):
        """Initialize with sequence that can be List[int], List[str], or List[int|str]"""
        self.values = list(values)

    def contains(self, value: Union[int, str]) -> bool:
        """Check if value is in this group"""
        return value in self.values

    def filter(self, predicate) -> "SignalGroup":
        """Filter values by predicate"""
        return SignalGroup(values=[v for v in self.values if predicate(v)])

    def unique(self) -> Union[int, str]:
        """Get unique value, raise if not exactly one"""
        if len(self.values) == 1:
            return self.values[0]
        elif len(self.values) == 0:
            raise ValueError("SignalGroup is empty")
        else:
            raise ValueError(f"SignalGroup has multiple values: {self.values}")


def resolve_signal_path(signal: str, scope: str) -> str:
    """Resolve signal path with scope support

    Args:
        signal: Signal name to resolve
        scope: Resolved scope for this task

    Returns:
        Resolved signal path
    """
    if not isinstance(signal, str):
        return signal

    if signal.startswith("$mod."):
        if not scope:
            raise ValueError(f"[ERROR] $mod used but no scope defined: {signal}")
        signal = signal.replace("$mod.", scope + ".", 1)
    elif signal == "$mod":
        if not scope:
            raise ValueError("[ERROR] $mod used but no scope defined")
        return scope
    elif scope and not signal.startswith("tb.") and not signal.startswith("/"):
        signal = scope + "." + signal

    return signal


def split_signal(signal_val: str, num_parts: int, bit_width: int = 0) -> List[int]:
    """Split a wide signal value into equal parts

    Args:
        signal_val: Hex signal value (with or without 0x prefix)
        num_parts: Number of parts to split into
        bit_width: Optional explicit bit width. If 0, infer from string length.

    Returns:
        List of integer values for each part (LSB first)
    """
    if signal_val.startswith('0x') or signal_val.startswith('0X'):
        signal_val = signal_val[2:]
    val_int = int(signal_val, 16)

    # If bit_width not specified, infer from string length
    if bit_width == 0:
        total_bits = len(signal_val) * 4
    else:
        total_bits = bit_width

    bits_per_part = total_bits // num_parts
    mask = (1 << bits_per_part) - 1
    return [(val_int >> (i * bits_per_part)) & mask for i in range(num_parts)]

def normalize_signal_name(signal: str) -> str:
    """Remove bit range from signal name for cache lookup
    
    Deprecated: Use Signal.normalize(signal) instead.
    """
    return Signal.normalize(signal)


def verilog_to_int(match) -> str:
    parts = match.group(0).split("'")
    base_map = {'b': 2, 'o': 8, 'd': 10, 'h': 16}
    base = base_map.get(parts[1][0].lower(), 10)
    return str(int(parts[1][1:].replace('_', ''), base))

def match_signal_with_bitwidth(signal: str, available_signals: List[str]) -> str:
    """Match a signal name with its FSDB representation (may include bit range)

    Args:
        signal: Signal name to match
        available_signals: List of available signals from FSDB

    Returns:
        Matched signal name (with bit range if present), or original signal if no match
    """
    # Try exact match first
    if signal in available_signals:
        return signal

    # Try to find signal with bit range (e.g., sig[127:0])
    pattern = re.escape(signal) + r"(\[\d+:\d+\])?"
    for fsdb_sig in available_signals:
        if re.match(f"^{pattern}$", fsdb_sig):
            return fsdb_sig

    return signal