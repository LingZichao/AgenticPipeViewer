#!/usr/bin/env python3
"""Utility functions shared across modules"""


import re


def resolve_signal_path(signal: str, task_scope: str, global_scope: str) -> str:
    """Resolve signal path with scope support

    Args:
        signal: Signal name to resolve
        task_scope: Task-specific scope
        global_scope: Global scope

    Returns:
        Resolved signal path
    """
    if not isinstance(signal, str):
        return signal

    scope = task_scope if task_scope else global_scope

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