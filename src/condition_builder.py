#!/usr/bin/env python3
import re
import ast
from typing import Any, Callable, Dict, List, Optional, Tuple, Set, TYPE_CHECKING
from dataclasses import dataclass
from utils import resolve_signal_path, verilog_to_int, split_signal, SignalGroup

if TYPE_CHECKING:
    from yaml_builder import Task
    from fsdb_builder import FsdbBuilder


@dataclass
class Condition:
    """Compiled condition that can be executed with runtime data"""

    raw_expr: str  # Original expression string
    norm_expr: str  # Normalized expression (Python-compatible)
    evaluator: Callable[[Dict[str, Any]], bool]  # Evaluation function
    signals: List[str]  # Signals needed (normalized names with {*})
    has_pattern: bool  # Whether contains pattern matching {var}
    pattern_var: str = ""  # Pattern variable name if has_pattern


class ExpressionEvaluator(ast.NodeVisitor):
    """AST-based expression evaluator with custom operator support"""

    def __init__(self, scope: str, runtime_data: Dict[str, Any]):
        self.scope = scope
        self.signal_values = runtime_data.get("signal_values", {})
        self.upstream_row = runtime_data.get("upstream_row", {})
        self.upstream_data = runtime_data.get("upstream_data", {})

    def eval(self, expr: str) -> Any:
        """Evaluate expression, handling custom operators"""
        # Normalize logical operators: && -> and, || -> or
        normalized = expr.replace("&&", " and ").replace("||", " or ")

        # Normalize: Verilog literals and $dep references
        # Use double underscore __ as separator to avoid conflicts with underscores in names
        normalized = re.sub(r"\d+'[bhdoBHDO][0-9a-fA-F_]+", verilog_to_int, normalized)
        for match in re.finditer(r"\$dep\.(\w+)\.(\w+)", normalized):
            normalized = normalized.replace(
                match.group(0), f"_dep__{match.group(1)}__{match.group(2)}"
            )

        # Transform $split() to _split() function call
        # Match: expr.$split(n) -> _split(expr, n)
        # Support: signal.$split(4), signal[31:0].$split(4), _dep_task_sig.$split(4)
        normalized = re.sub(
            r"([\w_]+(?:\[\d+:\d+\])?)\.\$split\(", r"_split(\1, ", normalized
        )

        # Transform <@ to a comparison operator that Python can parse
        # We'll use 'in' operator, ensuring proper spacing
        normalized = normalized.replace("<@", " in ")

        tree = ast.parse(normalized, mode="eval")
        return self.visit(tree.body)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left, right = self.visit(node.left), self.visit(node.right)
        ops = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.Mod: lambda: left % right,
            ast.BitAnd: lambda: left & right,
            ast.BitOr: lambda: left | right,
            ast.BitXor: lambda: left ^ right,
            ast.LShift: lambda: left << right,
            ast.RShift: lambda: left >> right,
        }
        return ops[type(node.op)]()

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        ops = {
            ast.Not: lambda: not operand,
            ast.Invert: lambda: ~operand,
            ast.UAdd: lambda: +operand,
            ast.USub: lambda: -operand,
        }
        return ops[type(node.op)]()

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            return all(self.visit(v) for v in node.values)
        return any(self.visit(v) for v in node.values)

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if isinstance(op, ast.In):
                # Handle <@ operator (converted to 'in')
                if isinstance(right, SignalGroup):
                    if not isinstance(left, int):
                        raise ValueError(
                            f"Left operand of <@ must be int, got {type(left)}"
                        )
                    result = right.contains(left)
                    if not result:
                        return False
                else:
                    if left != right:
                        return False
            else:
                ops = {
                    ast.Eq: lambda: left == right,
                    ast.NotEq: lambda: left != right,
                    ast.Lt: lambda: left < right,
                    ast.LtE: lambda: left <= right,
                    ast.Gt: lambda: left > right,
                    ast.GtE: lambda: left >= right,
                }
                if not ops[type(op)]():
                    return False
            left = right
        return True

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Call(self, node: ast.Call) -> Any:
        """Handle function calls, including custom operators like _split() and _contains()"""
        if isinstance(node.func, ast.Name):
            if node.func.id == "_split":
                # _split(signal, num_parts) -> SignalGroup[int]
                if len(node.args) != 2:
                    raise ValueError("_split() requires exactly 2 arguments")
                # Reset bit width tracking before visiting the signal argument
                self._last_dep_bitwidth = 0
                signal_val = self.visit(node.args[0])
                num_parts = self.visit(node.args[1])
                # Use stored bit width if available (from $dep reference)
                bit_width = getattr(self, '_last_dep_bitwidth', 0)
                # split_signal returns List[int], so this is type-safe
                int_values: List[int] = split_signal(hex(signal_val), num_parts, bit_width)
                result = SignalGroup(values=int_values)
                return result
            elif node.func.id == "_contains":
                # _contains(value, group) -> bool
                if len(node.args) != 2:
                    raise ValueError("_contains() requires exactly 2 arguments")
                left_val = self.visit(node.args[0])
                right_val = self.visit(node.args[1])
                # Right side should be a SignalGroup
                if isinstance(right_val, SignalGroup):
                    return right_val.contains(left_val)
                # If not a group, treat as single value comparison
                return left_val == right_val
        raise ValueError(f"Unsupported function call: {ast.unparse(node)}")

    def visit_Name(self, node: ast.Name) -> Any:
        name = node.id
        if name.startswith("_dep__"):
            # Format: _dep__taskid__signalname (using __ as separator)
            parts = name.split("__")
            if len(parts) >= 3:
                # parts[0] = "_dep", parts[1] = taskid, parts[2] = signalname
                signal_name = parts[2]
            else:
                raise ValueError(f"Invalid $dep format: {name}")
            for sig in self.upstream_data.get("capd", []):
                if (
                    sig.endswith("." + signal_name)
                    or sig.endswith("/" + signal_name)
                    or sig == signal_name
                ):
                    val_str = self.upstream_row["capd"].get(sig, "0")
                    # Store the original hex string length for bit width calculation
                    # This is used by $split to preserve leading zeros
                    clean_str = val_str[2:] if val_str.startswith("0x") else val_str
                    val_int = int("0x" + clean_str, 16)
                    # Store bit width as metadata (hack: use tuple to pass both)
                    # The caller will extract the int value, but _split can use this
                    self._last_dep_bitwidth = len(clean_str) * 4
                    return val_int
            raise ValueError(f"Signal '{signal_name}' not found in upstream")

        signal = resolve_signal_path(name, self.scope)
        val_str = self.signal_values.get(signal, "0")
        if val_str in ["x", "z", "X", "Z"] or val_str.startswith("*"):
            return 0
        return int("0x" + val_str if not val_str.startswith("0x") else val_str, 16)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        signal_name = self._get_name(node.value)
        signal = resolve_signal_path(signal_name, self.scope)
        val_str = self.signal_values.get(signal, "0")
        val_int = (
            0
            if val_str in ["x", "z", "X", "Z"] or val_str.startswith("*")
            else int("0x" + val_str if not val_str.startswith("0x") else val_str, 16)
        )

        if isinstance(node.slice, ast.Slice):
            # Python parses [31:0] as lower=31, upper=0
            # But Verilog convention is [high:low], so we need to handle this
            # Python slice: [start:stop] -> lower=start, upper=stop
            # Verilog slice: [msb:lsb] -> we want high=msb, low=lsb
            msb = self.visit(node.slice.lower) if node.slice.lower else None  # Left of colon
            lsb = self.visit(node.slice.upper) if node.slice.upper else None  # Right of colon
            if msb is not None and lsb is not None:
                # Ensure msb >= lsb (Verilog convention)
                high, low = (msb, lsb) if msb >= lsb else (lsb, msb)
                return (val_int >> low) & ((1 << (high - low + 1)) - 1)
            return val_int
        else:
            index = self.visit(node.slice)
            return (val_int >> index) & 1

    def _get_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        raise ValueError(f"Cannot extract name from {node}")


class ConditionParser:
    """Unified condition expression parser with preprocessing and signal collection"""

    def __init__(self, scope: str):
        self.scope = scope
        self.signals: Set[str] = set()
        self.has_pattern = False
        self.pattern_var = ""
        self.pattern_signals: List[str] = []

    def parse(self, expr: str) -> Tuple[str, List[str], bool, str]:
        """
        Parse and preprocess expression, collecting all metadata.

        Returns: (normalized_expr, signals, has_pattern, pattern_var)
        """
        # Normalize input: convert list to string
        if isinstance(expr, list):
            expr = " ".join(line.strip() for line in expr if line.strip())
        elif not isinstance(expr, str):
            expr = str(expr)
        
        # Step 1: Extract pattern signals (before normalization)
        self._extract_patterns(expr)

        # Step 2: Normalize syntax to Python-compatible form
        normalized = self._normalize_syntax(expr)

        # Step 3: Parse AST and collect signals
        self._collect_signals_from_ast(normalized)

        return normalized, list(self.signals), self.has_pattern, self.pattern_var

    def _extract_patterns(self, expr: str) -> None:
        """Extract and resolve pattern signals like signal{var}"""
        patterns = re.findall(r"[\w.]+\{[\w]+\}[\w.]*", expr)

        if patterns:
            self.has_pattern = True
            self.pattern_signals = patterns

            # Extract variable name(s)
            var_names = set()
            for pattern in patterns:
                var_names.update(re.findall(r"\{(\w+)\}", pattern))

            if len(var_names) == 1:
                self.pattern_var = list(var_names)[0]
            elif len(var_names) > 1:
                raise ValueError(
                    f"Multiple pattern variables not supported: {var_names}"
                )

            # Add pattern signals with {*} wildcard
            for pattern in patterns:
                normalized_pattern = re.sub(r"\{[^}]+\}", "{*}", pattern)
                resolved = resolve_signal_path(normalized_pattern, self.scope)
                self.signals.add(resolved)

    def _normalize_syntax(self, expr: str) -> str:
        """Normalize custom syntax to Python-compatible AST"""
        # Convert logical operators
        normalized = expr.replace("&&", " and ").replace("||", " or ")
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # Remove Verilog literals (replace with 0 for AST parsing)
        normalized = re.sub(r"\d+'[bhdoBHDO][0-9a-fA-F_]+", "0", normalized)

        # Normalize $dep references to _dep__ format (using __ as separator)
        for match in re.finditer(r"\$dep\.(\w+)\.(\w+)", normalized):
            normalized = normalized.replace(
                match.group(0), f"_dep__{match.group(1)}__{match.group(2)}"
            )

        # Transform $split() to _split() function call
        normalized = re.sub(
            r"([\w_]+(?:\[\d+:\d+\])?)\.\$split\(", r"_split(\1, ", normalized
        )

        # Transform <@ to 'in' operator
        normalized = normalized.replace("<@", " in ")

        # Remove pattern signals from normalized expression (for AST parsing)
        for pattern in self.pattern_signals:
            normalized = re.sub(re.escape(pattern), "0", normalized)

        return normalized

    def _collect_signals_from_ast(self, expr: str) -> None:
        """Collect signals by walking the AST"""
        try:
            tree = ast.parse(expr, mode="eval")
            self._visit_ast(tree.body)
        except SyntaxError:
            # Invalid syntax, will be caught during evaluation
            pass

    def _visit_ast(self, node: ast.AST) -> None:
        """Recursively visit AST nodes to collect signals"""
        if isinstance(node, ast.Name):
            self._collect_name(node)
        elif isinstance(node, ast.Attribute):
            self._collect_attribute(node)
        elif isinstance(node, ast.Subscript):
            self._collect_subscript(node)
        elif isinstance(node, ast.Call):
            self._collect_call(node)
        else:
            # Recursively visit children for other node types
            for child in ast.iter_child_nodes(node):
                self._visit_ast(child)

    def _collect_name(self, node: ast.Name) -> None:
        """Collect signal from Name node"""
        name = node.id
        # Skip Python keywords and internal names (starting with _)
        if name not in ["True", "False", "None"] and not name.startswith("_"):
            try:
                resolved = resolve_signal_path(name, self.scope)
                self.signals.add(resolved)
            except (ValueError, RuntimeError):
                pass

    def _collect_attribute(self, node: ast.Attribute) -> None:
        """Collect signal from Attribute node (e.g., module.signal)"""
        signal_name = self._get_name(node)
        if signal_name and not signal_name.startswith("_"):
            try:
                resolved = resolve_signal_path(signal_name, self.scope)
                self.signals.add(resolved)
            except (ValueError, RuntimeError):
                pass

    def _collect_subscript(self, node: ast.Subscript) -> None:
        """Collect signal from Subscript node (e.g., signal[31:0])"""
        signal_name = self._get_name(node.value)
        if signal_name and not signal_name.startswith("_"):
            try:
                resolved = resolve_signal_path(signal_name, self.scope)
                self.signals.add(resolved)
            except (ValueError, RuntimeError):
                pass

    def _collect_call(self, node: ast.Call) -> None:
        """Handle function calls like _split(signal, n)"""
        if isinstance(node.func, ast.Name) and node.func.id == "_split":
            # For _split(signal, n), collect the signal argument
            if len(node.args) >= 1:
                self._visit_ast(node.args[0])

    def _get_name(self, node: ast.AST) -> str:
        """Extract signal name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return ""


class ConditionBuilder:
    """Build and execute conditions with runtime data"""

    @staticmethod
    def collect_signals(condition_str: str, scope: str) -> List[str]:
        """Extract signal identifiers from condition expression (static method)"""
        parser = ConditionParser(scope)
        _, signals, _, _ = parser.parse(condition_str)
        return signals

    def build(self, task: "Task", fsdb_builder: "FsdbBuilder") -> Condition:
        """Build a condition from task"""
        scope = task.scope or ""
        raw_expr = task.raw_condition
        return self.build_raw(raw_expr, scope, fsdb_builder)

    def build_raw(
        self,
        raw_condition: str,
        scope: str,
        fsdb_builder: "FsdbBuilder"
    ) -> Condition:
        """Build condition from raw expression (no Task object required)

        Args:
            raw_condition: Raw condition expression string
            scope: Signal scope for resolution
            fsdb_builder: FSDB builder for signal expansion

        Returns:
            Compiled Condition object
        """
        # Parse and preprocess expression using unified parser
        parser = ConditionParser(scope)
        norm_expr, signals, has_pattern, pattern_var = parser.parse(raw_condition)
        
        # Normalize raw_condition for storage (convert list to string if needed)
        if isinstance(raw_condition, list):
            raw_condition_str = " ".join(line.strip() for line in raw_condition if line.strip())
        else:
            raw_condition_str = str(raw_condition)

        # Pre-compute pattern candidates if needed
        pattern_candidates = None
        if has_pattern:
            pattern_candidates = self._compute_pattern_candidates(
                parser.pattern_signals, pattern_var, scope, fsdb_builder
            )

        # Build unified evaluator
        evaluator = self._build_evaluator(
            scope,
            raw_condition_str,
            has_pattern,
            pattern_var,
            parser.pattern_signals if has_pattern else None,
            pattern_candidates,
        )

        return Condition(
            raw_expr=raw_condition_str,
            norm_expr=norm_expr,
            evaluator=evaluator,
            signals=signals,
            has_pattern=has_pattern,
            pattern_var=pattern_var,
        )

    def _compute_pattern_candidates(
        self,
        patterns: List[str],
        var_name: str,
        scope: str,
        fsdb_builder: "FsdbBuilder",
    ) -> SignalGroup:
        """Pre-compute all possible pattern variable values from FSDB

        Uses expand_raw_signals() to get signal names, then extracts variable values
        using regex pattern matching.
        """
        possible_vals = set()

        for pattern in patterns:
            # Step 1: Resolve pattern with scope, then convert {variable} to {*}
            resolved_pattern = resolve_signal_path(pattern, scope)
            wildcard_pattern = re.sub(r'\{[^}]+\}', '{*}', resolved_pattern)
            expanded_signals = fsdb_builder.expand_raw_signals([wildcard_pattern])

            # Step 2: Build regex to extract variable value from actual signal names
            extract_regex = re.escape(resolved_pattern).replace(
                re.escape(f'{{{var_name}}}'), r'([a-zA-Z0-9_$]+)'
            )
            # Allow optional bit range at the end
            extract_regex = f'^{extract_regex}(?:\\[\\d+:\\d+\\])?$'

            # Step 3: Extract variable values from matched signals
            for sig in expanded_signals:
                match = re.match(extract_regex, sig)
                if match:
                    possible_vals.add(match.group(1))
        return SignalGroup(values=list(possible_vals))

    def _build_evaluator(
        self,
        scope: str,
        raw_expr: str,
        has_pattern: bool,
        pattern_var: str,
        patterns: Optional[List[str]] = None,
        candidates: Optional[SignalGroup] = None,
    ) -> Callable[[Dict[str, Any]], bool]:
        """Build unified evaluator for both simple and pattern conditions"""

        if not has_pattern:
            # Simple condition: direct evaluation
            def evaluator(runtime_data: Dict[str, Any]) -> bool:
                expr_eval = ExpressionEvaluator(scope, runtime_data)
                return bool(expr_eval.eval(raw_expr))

            return evaluator

        # Pattern condition: test each candidate value
        # Assert patterns and candidates are not None for pattern conditions
        assert patterns is not None, "patterns required for pattern conditions"
        assert candidates is not None, "candidates required for pattern conditions"

        def evaluator(runtime_data: Dict[str, Any]) -> bool:
            def test_value(val: str) -> bool:
                # Substitute pattern variable with candidate value
                test_expr = raw_expr
                for pattern in patterns:
                    test_expr = test_expr.replace(
                        pattern, pattern.replace(f"{{{pattern_var}}}", val)
                    )
                try:
                    expr_eval = ExpressionEvaluator(scope, runtime_data)
                    result = bool(expr_eval.eval(test_expr))
                    return result
                except (ValueError, RuntimeError, KeyError):
                    return False

            # Filter candidates to find matches
            matched_group = candidates.filter(test_value)

            if len(matched_group.values) == 0:
                return False
            elif len(matched_group.values) == 1:
                # Unique match - store the matched variable
                runtime_data["vars"] = {pattern_var: matched_group.values[0]}
                return True
            else:
                # Multiple matches - store ALL matched variables for fork handling
                # The caller (_trace_depends) will iterate through these
                runtime_data["vars"] = {pattern_var: matched_group.values[0]}  # First match
                runtime_data["_all_matched_vars"] = {
                    pattern_var: matched_group.values  # All matches
                }
                return True  # Return True, let caller handle multiple forks

        return evaluator

    def exec(self, condition: Condition, runtime_data: Dict[str, Any]) -> bool:
        """Execute condition with runtime data"""
        try:
            return condition.evaluator(runtime_data)
        except Exception as e:
            raise ValueError(
                f"[ERROR] Failed to evaluate condition '{condition.raw_expr}': {e}"
            )

