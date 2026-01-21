#!/usr/bin/env python3
import re
import ast
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from .utils import resolve_signal_path, verilog_to_int, split_signal, SignalGroup


class Condition:
    """Compiled condition that can be executed with runtime data
    
    Supports two-phase initialization for pattern conditions:
    1. Build phase: Parse structure without expanding patterns
    2. Activate phase: Expand pattern candidates and rebuild evaluator
    """

    def __init__(
        self,
        raw_expr: str,
        norm_expr: str,
        evaluator: Callable[[Dict[str, Any]], bool],
        signals: List[str],
        has_pattern: bool,
        pattern_var: str = ""
    ):
        self.raw_expr = raw_expr
        self.norm_expr = norm_expr
        self.evaluator = evaluator
        self.signals = signals
        self.has_pattern = has_pattern
        self.pattern_var = pattern_var
        
        # Two-phase initialization support
        self._activated = not has_pattern  # Non-pattern conditions are always activated
        self.scope: str = ""  # Set by builder for pattern conditions
        self._pattern_signals: List[str] = []  # Pattern signal templates

    def activate(self, pattern_resolver: Callable[[str], Tuple[List[str], List[str]]]) -> None:
        """Activate pattern condition by expanding candidates and rebuilding evaluator
        
        Args:
            pattern_resolver: Callback to resolve pattern (pattern -> (signals, candidates))
            
        Raises:
            RuntimeError: If called on non-pattern condition or already activated
        """
        if not self.has_pattern:
            return  # Non-pattern conditions don't need activation
        
        if self._activated:
            return  # Already activated
        
        # Expand pattern candidates
        possible_vals = set()
        for pattern in self._pattern_signals:
            _, candidates = pattern_resolver(pattern)
            possible_vals.update(candidates)
        
        pattern_candidates = SignalGroup(values=list(possible_vals))
        
        # Rebuild evaluator with candidates
        from .cond_builder import ConditionBuilder
        builder = ConditionBuilder()
        self.evaluator = builder._build_evaluator(
            self.scope,
            self.raw_expr,
            self.has_pattern,
            self.pattern_var,
            self._pattern_signals,
            pattern_candidates
        )
        
        self._activated = True

    def exec(self, runtime_data: Dict[str, Any]) -> bool:
        """Execute this condition with runtime data

        Args:
            runtime_data: Dictionary containing:
                - signal_values: Current time slice signal values
                - signal_metadata: Signal objects with bit width info
                - upstream_row: Dependency chain data (optional)
                - upstream_data: Complete upstream results (optional)
                - vars: Pattern variable bindings (modified by evaluator)

        Returns:
            True if condition is satisfied, False otherwise

        Raises:
            ValueError: If evaluation fails with detailed error context
            RuntimeError: If pattern condition not activated before exec
        """
        if self.has_pattern and not self._activated:
            raise RuntimeError(
                f"[ERROR] Pattern condition '{self.raw_expr}' not activated. "
                "Call activate(pattern_resolver) before exec()."
            )
        
        try:
            return self.evaluator(runtime_data)
        except Exception as e:
            raise ValueError(
                f"[ERROR] Failed to evaluate condition '{self.raw_expr}': {e}"
            )


class ExpressionEvaluator(ast.NodeVisitor):
    """AST-based expression evaluator with custom operator support"""

    def __init__(self, scope: str, runtime_data: Dict[str, Any]):
        self.scope = scope
        self.signal_values = runtime_data.get("signal_values", {})
        self.signal_metadata = runtime_data.get("signal_metadata", {})
        self.upstream_row = runtime_data.get("upstream_row", {})
        self.upstream_data = runtime_data.get("upstream_data", {})
        self._last_bitwidth = 0

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

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        signal_name = self._get_name(node)
        signal = resolve_signal_path(signal_name, self.scope)
        
        # Track bit width for $split
        if signal in self.signal_metadata:
            self._last_bitwidth = self.signal_metadata[signal].bit_width
            
        val_str = self.signal_values.get(signal, "0")
        if val_str in ["x", "z", "X", "Z"] or val_str.startswith("*"):
            return 0
        return int("0x" + val_str if not val_str.startswith("0x") else val_str, 16)

    def visit_Call(self, node: ast.Call) -> Any:
        """Handle function calls, including custom operators like _split() and _contains()"""
        if isinstance(node.func, ast.Name):
            if node.func.id == "_split":
                # _split(signal, num_parts) -> SignalGroup[int]
                if len(node.args) != 2:
                    raise ValueError("_split() requires exactly 2 arguments")
                # Reset bit width tracking before visiting the signal argument
                self._last_bitwidth = 0
                signal_val = self.visit(node.args[0])
                num_parts = self.visit(node.args[1])
                # Use tracked bit width (from visit_Attribute or visit_Name)
                bit_width = self._last_bitwidth
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
                target_task_id = parts[1]
                signal_name = parts[2]
            else:
                raise ValueError(f"Invalid $dep format: {name}")

            # Get the full dependency chain from upstream_row
            # dep_chain is a dict: {task_id: {signal_name: value}}
            dep_chain = self.upstream_row.get("dep_chain", {})

            # Try to find the signal in the target task's captured data
            # First check if target_task_id is in the dependency chain
            if target_task_id not in dep_chain:
                raise ValueError(
                    f"Task '{target_task_id}' not found in dependency chain. "
                    f"Available tasks: {list(dep_chain.keys())}"
                )

            # Get the captured data from the target task
            target_capd = dep_chain[target_task_id]

            # Search for the signal in the target task's captured signals
            for sig in target_capd.keys():
                if (
                    sig.endswith("." + signal_name)
                    or sig.endswith("/" + signal_name)
                    or sig == signal_name
                ):
                    val_str = target_capd.get(sig, "0")
                    # Store the original hex string length for bit width calculation
                    # This is used by $split to preserve leading zeros
                    clean_str = val_str[2:] if val_str.startswith("0x") else val_str
                    self._last_bitwidth = len(clean_str) * 4
                    return int("0x" + clean_str, 16)

            raise ValueError(
                f"Signal '{signal_name}' not found in task '{target_task_id}'. "
                f"Available signals: {list(target_capd.keys())}"
            )

        signal = resolve_signal_path(name, self.scope)
        
        # Track bit width for $split
        if signal in self.signal_metadata:
            self._last_bitwidth = self.signal_metadata[signal].bit_width
            
        val_str = self.signal_values.get(signal, "0")
        if val_str in ["x", "z", "X", "Z"] or val_str.startswith("*"):
            return 0
        return int("0x" + val_str if not val_str.startswith("0x") else val_str, 16)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        signal_name = self._get_name(node.value)
        if isinstance(node.value, ast.Name) and signal_name.startswith("_dep__"):
            parts = signal_name.split("__")
            if len(parts) >= 3:
                target_task_id = parts[1]
                dep_signal_name = "__".join(parts[2:])
            else:
                raise ValueError(f"Invalid $dep format: {signal_name}")

            dep_chain = self.upstream_row.get("dep_chain", {})
            if target_task_id not in dep_chain:
                raise ValueError(
                    f"Task '{target_task_id}' not found in dependency chain. "
                    f"Available tasks: {list(dep_chain.keys())}"
                )

            target_capd = dep_chain[target_task_id]
            for sig in target_capd.keys():
                if (
                    sig.endswith("." + dep_signal_name)
                    or sig.endswith("/" + dep_signal_name)
                    or sig == dep_signal_name
                ):
                    val_str = target_capd.get(sig, "0")
                    clean_str = val_str[2:] if val_str.startswith("0x") else val_str
                    val_int = int("0x" + clean_str, 16)
                    self._last_bitwidth = len(clean_str) * 4
                    break
            else:
                raise ValueError(
                    f"Signal '{dep_signal_name}' not found in task '{target_task_id}'. "
                    f"Available signals: {list(target_capd.keys())}"
                )
        else:
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

    def build_raw(
        self,
        raw_condition: str,
        scope: str
    ) -> Condition:
        """Build condition from raw expression (two-phase initialization)
        
        Args:
            raw_condition: Raw condition expression string
            scope: Signal scope for resolution
            
        Returns:
            Condition object (pattern conditions need activate() before exec())
            
        Note:
            For pattern conditions, call condition.activate(pattern_resolver)
            after FSDB signals are dumped.
        """
        # Parse and preprocess expression using unified parser
        parser = ConditionParser(scope)
        norm_expr, signals, has_pattern, pattern_var = parser.parse(raw_condition)
        
        # Normalize raw_condition for storage (convert list to string if needed)
        if isinstance(raw_condition, list):
            raw_condition_str = " ".join(line.strip() for line in raw_condition if line.strip())
        else:
            raw_condition_str = str(raw_condition)

        if has_pattern:
            # Pattern condition: Create placeholder evaluator for later activation
            def placeholder_evaluator(runtime_data: Dict[str, Any]) -> bool:
                raise RuntimeError(
                    f"Pattern condition '{raw_condition_str}' not activated. "
                    "Call activate(pattern_resolver) before exec()."
                )
            
            condition = Condition(
                raw_condition_str,
                norm_expr,
                placeholder_evaluator,
                signals,
                has_pattern=True,
                pattern_var=pattern_var
            )
            # Store metadata for later activation
            condition.scope = scope
            condition._pattern_signals = parser.pattern_signals
            return condition
        else:
            # Non-pattern condition: Build complete evaluator immediately
            evaluator = self._build_evaluator(
                scope,
                raw_condition_str,
                has_pattern=False,
                pattern_var="",
                patterns=None,
                candidates=None
            )
            return Condition(
                raw_condition_str,
                norm_expr,
                evaluator,
                signals,
                has_pattern=False,
                pattern_var=""
            )

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
