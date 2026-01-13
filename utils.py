#!/usr/bin/env python3
"""Utility functions shared across modules"""


import re
from dataclasses import dataclass
from typing import List, Union, Sequence


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


def split_signal(signal_val: str, num_parts: int) -> list[int]:
    """Split a wide signal value into equal parts

    Args:
        signal_val: Hex signal value (with or without 0x prefix)
        num_parts: Number of parts to split into

    Returns:
        List of integer values for each part
    """
    if signal_val.startswith('0x') or signal_val.startswith('0X'):
        signal_val = signal_val[2:]
    val_int = int(signal_val, 16)
    total_bits = len(signal_val) * 4
    bits_per_part = total_bits // num_parts
    mask = (1 << bits_per_part) - 1
    return [(val_int >> (i * bits_per_part)) & mask for i in range(num_parts)]

def verilog_to_int(match: re.Match[str]) -> str:
    parts = match.group(0).split("'")
    base_map = {'b': 2, 'o': 8, 'd': 10, 'h': 16}
    base = base_map.get(parts[1][0].lower(), 10)
    return str(int(parts[1][1:].replace('_', ''), base))

def match_signal_with_bitwidth(signal: str, available_signals: list[str]) -> str:
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