#!/usr/bin/env python3
"""Utility functions shared across modules"""


import re
from typing import List, Union, Sequence, Optional, Dict, Any

class Signal:
    """Represents a signal with its metadata and captured values"""
    raw_name: str  # Original name from FSDB (may include bitwidth)
    scope: str
    name: str  # Normalized name without bit range
    msb: int
    lsb: int
    vidcode: int
    values: List[str]

    def __init__(self, raw_name: str, scope: str = ""):
        """Initialize Signal from raw name and scope
        
        Args:
            raw_name: Original signal name (may include bit range like [127:0])
            scope: Signal scope path
        """
        self.raw_name = raw_name
        self.scope = scope
        self.name = self.normalize(raw_name)
        self.msb = 0
        self.lsb = 0
        self.vidcode = -1
        self.values = []
        
        # Parse MSB/LSB from raw_name
        match = re.search(r'\[(\d+):(\d+)\]$', self.raw_name)
        if match:
            self.msb = int(match.group(1))
            self.lsb = int(match.group(2))

    def is_pattern(self) -> bool:
        """Check if this is a pattern signal (False for base Signal)"""
        return False
    
    def get_template(self) -> str:
        """Get template string for expansion (returns raw_name for regular Signal)"""
        return self.raw_name

    def resolve(self, var_bindings: Dict[str, Any]) -> str:
        """Resolve signal path (no expansion for regular Signal)"""
        return resolve_signal_path(self.raw_name, self.scope)

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


class PatternSignal:
    """Pattern signal factory that generates Signal instances

    Unlike Signal which stores actual waveform data, PatternSignal serves as a 
    template factory for generating specific Signal instances based on variable bindings.

    Key responsibilities:
    - Parse template patterns with variables (e.g., "signal{idx}_data")
    - Generate wildcard patterns for FSDB matching (e.g., "signal{*}_data")
    - Track candidate variable values discovered from FSDB
    - Create specific Signal instances through template expansion

    Lifecycle:
    1. Created from template string (e.g., "signal{idx}_data")
    2. Normalized to wildcard pattern (e.g., "signal{*}_data")
    3. Resolved against FSDB to find candidate values (e.g., ["0", "1", "2"])
    4. Expanded at runtime to generate specific Signal instances

    Example:
        >>> ps = PatternSignal(template="icache_line{bank}_valid")
        >>> ps.variable_name  # "bank"
        >>> ps.wildcard_pattern  # "icache_line{*}_valid"
        >>> ps.set_candidates(["0", "1", "2", "3"])
        >>> signal_instance = ps.generate_signal({"bank": "2"})  # Creates Signal("icache_line2_valid")
    """

    def __init__(self, template: str, scope: str = ""):
        """Initialize PatternSignal from template string

        Args:
            template: Signal template with {var} pattern (e.g., "signal{idx}_data")
            scope: Signal scope
        """
        # Extract variable name
        var_match = re.search(r'\{(\w+)\}', template)
        if not var_match:
            raise ValueError(f"No pattern variable found in template: {template}")

        self.template = template
        self.scope = scope
        self.variable_name = var_match.group(1)
        self.wildcard_pattern = re.sub(r'\{[^}]+\}', '{*}', template)

        # Store pattern-specific fields
        self.candidates: List[str] = []
        self.resolved_signals: List[Signal] = []  # Actual Signal objects after FSDB expansion

    def is_pattern(self) -> bool:
        """Check if this is a pattern signal (always True for PatternSignal)"""
        return True
    
    def get_wildcard_name(self) -> str:
        """Get the wildcard pattern name for FSDB matching"""
        return self.wildcard_pattern
    
    def get_template(self) -> str:
        """Get template string for expansion"""
        return self.template
    
    def get_raw_name(self) -> str:
        """Get raw name (alias for wildcard_pattern for compatibility)"""
        return self.wildcard_pattern
    
    @property
    def raw_name(self) -> str:
        """Property alias for get_raw_name() to maintain compatibility"""
        return self.get_raw_name()

    def has_variable(self) -> bool:
        """Check if this is a pattern signal (always True for PatternSignal)"""
        return True

    def set_candidates(self, candidate_values: List[str], signal_objects: Optional[List[Signal]] = None):
        """Set candidate values and their corresponding Signal objects

        Args:
            candidate_values: List of possible values for pattern variable
            signal_objects: List of actual Signal objects after FSDB expansion
        """
        self.candidates = sorted(candidate_values)
        if signal_objects:
            self.resolved_signals = signal_objects

    def generate_signal(self, var_bindings: Dict[str, str]) -> Signal:
        """Generate a specific Signal instance from template with variable bindings

        Args:
            var_bindings: Dictionary mapping variable names to values

        Returns:
            New Signal instance with expanded name
        """
        expanded_name = self.resolve(var_bindings)
        return Signal(raw_name=expanded_name, scope=self.scope)
    
    def get_resolved_signal(self, var_bindings: Dict[str, str]) -> Optional[Signal]:
        """Get the actual Signal object for the expanded template from resolved cache

        Args:
            var_bindings: Dictionary mapping variable names to values

        Returns:
            Signal object matching the expanded template, or None if not found
        """
        expanded_name = self.resolve(var_bindings)
        for sig in self.resolved_signals:
            if sig.raw_name == expanded_name or sig.name == Signal.normalize(expanded_name):
                return sig
        return None

    def resolve(self, var_bindings: Dict[str, str]) -> str:
        """Resolve signal path by expanding template with variable bindings

        Args:
            var_bindings: Dictionary mapping variable names to values

        Returns:
            Expanded and scope-resolved signal name
        """
        sig = self.template
        # Apply all available variable bindings
        for var_name, var_val in var_bindings.items():
            sig = sig.replace(f"{{{var_name}}}", str(var_val))
        
        # Ensure path is fully resolved with scope
        return resolve_signal_path(sig, self.scope)

    def expand(self, var_bindings: Dict[str, str]) -> str:
        """Alias for resolve() for backward compatibility"""
        return self.resolve(var_bindings)

    def expand_all(self) -> List[str]:
        """Expand template with all candidate values

        Returns:
            List of all possible signal names

        Example:
            >>> ps = PatternSignal("signal{idx}_data")
            >>> ps.set_candidates(["0", "1", "2"])
            >>> ps.expand_all()
            ["signal0_data", "signal1_data", "signal2_data"]
        """
        return [
            self.template.replace(f"{{{self.variable_name}}}", val)
            for val in self.candidates
        ]

    def get_candidate_group(self) -> SignalGroup:
        """Get candidates as a SignalGroup for filtering/testing

        Returns:
            SignalGroup containing all candidate values
        """
        return SignalGroup(values=self.candidates)

    def validate_value(self, value: str) -> bool:
        """Check if a value is a valid candidate

        Args:
            value: Candidate value to check

        Returns:
            True if value is in candidates list
        """
        return value in self.candidates

    def __repr__(self) -> str:
        """String representation for debugging"""
        cand_preview = self.candidates[:3] if len(self.candidates) <= 3 else self.candidates[:3] + ["..."]
        return (
            f"PatternSignal(template='{self.template}', "
            f"var='{self.variable_name}', "
            f"candidates={cand_preview})"
        )


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