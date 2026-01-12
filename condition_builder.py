#!/usr/bin/env python3
import re
import ast
from typing import Any, Callable, List, TYPE_CHECKING
from dataclasses import dataclass
from utils import resolve_signal_path, verilog_to_int, split_signal

if TYPE_CHECKING:
    from yaml_builder import Task
    from fsdb_builder import FsdbBuilder


@dataclass
class Condition:
    """Compiled condition that can be executed with runtime data"""

    expr: str
    evaluator: Callable[[dict[str, Any]], bool]


@dataclass
class SignalGroup:
    """Represents a group of signal values with operations"""

    values: List[int]

    def contains(self, value: int) -> bool:
        """Check if value is in this group"""
        return value in self.values

    def filter(self, predicate) -> "SignalGroup":
        """Filter values by predicate"""
        return SignalGroup(values=[v for v in self.values if predicate(v)])

    def unique(self) -> int:
        """Get unique value, raise if not exactly one"""
        if len(self.values) == 1:
            return self.values[0]
        elif len(self.values) == 0:
            raise ValueError("SignalGroup is empty")
        else:
            raise ValueError(f"SignalGroup has multiple values: {self.values}")


class ExpressionEvaluator(ast.NodeVisitor):
    """AST-based expression evaluator with custom operator support"""

    def __init__(
        self, task_scope: str, global_scope: str, runtime_data: dict[str, Any]
    ):
        self.task_scope = task_scope
        self.global_scope = global_scope
        self.signal_values = runtime_data.get("signal_values", {})
        self.upstream_row  = runtime_data.get("upstream_row", {})
        self.upstream_data = runtime_data.get("upstream_data", {})

    def eval(self, expr: str) -> Any:
        """Evaluate expression, handling custom operators"""
        # Normalize: Verilog literals and $dep references
        normalized = re.sub(r"\d+'[bhdoBHDO][0-9a-fA-F_]+", verilog_to_int, expr)
        for match in re.finditer(r"\$dep\.(\w+)\.(\w+)", normalized):
            normalized = normalized.replace(
                match.group(0), f"_dep_{match.group(1)}_{match.group(2)}"
            )

        # Transform $split() to _split() function call
        # Match: expr.$split(n) -> _split(expr, n)
        # Support: signal.$split(4), signal[31:0].$split(4), _dep_task_sig.$split(4)
        normalized = re.sub(
            r"([\w_]+(?:\[\d+:\d+\])?)\.\$split\(", r"_split(\1, ", normalized
        )

        # Transform <@ to _contains() function call
        # Match: left <@ right -> _contains(left, right)
        # Use a more robust approach: find <@ and wrap the entire expression
        if "<@" in normalized:
            # Replace <@ with a unique marker first
            normalized = normalized.replace("<@", " __CONTAINS_OP__ ")
            # Then wrap with _contains()
            parts = normalized.split(" __CONTAINS_OP__ ")
            if len(parts) == 2:
                normalized = f"_contains({parts[0].strip()}, {parts[1].strip()})"

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
                # _split(signal, num_parts) -> SignalGroup
                if len(node.args) != 2:
                    raise ValueError("_split() requires exactly 2 arguments")
                signal_val = self.visit(node.args[0])
                num_parts = self.visit(node.args[1])
                return SignalGroup(values=split_signal(hex(signal_val), num_parts))
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
        if name.startswith("_dep_"):
            parts = name.split("_", 3)
            signal_name = parts[3]
            for sig in self.upstream_data.get("capd", []):
                if (
                    sig.endswith("." + signal_name)
                    or sig.endswith("/" + signal_name)
                    or sig == signal_name
                ):
                    val_str = self.upstream_row["capd"].get(sig, "0")
                    return (
                        int(val_str, 16)
                        if val_str.startswith("0x")
                        else int("0x" + val_str, 16)
                    )
            raise ValueError(f"Signal '{signal_name}' not found in upstream")

        signal = resolve_signal_path(name, self.task_scope, self.global_scope)
        val_str = self.signal_values.get(signal, "0")
        if val_str in ["x", "z", "X", "Z"] or val_str.startswith("*"):
            return 0
        return int("0x" + val_str if not val_str.startswith("0x") else val_str, 16)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        signal_name = self._get_name(node.value)
        signal = resolve_signal_path(signal_name, self.task_scope, self.global_scope)
        val_str = self.signal_values.get(signal, "0")
        val_int = (
            0
            if val_str in ["x", "z", "X", "Z"] or val_str.startswith("*")
            else int("0x" + val_str if not val_str.startswith("0x") else val_str, 16)
        )

        if isinstance(node.slice, ast.Slice):
            high = self.visit(node.slice.upper) if node.slice.upper else None
            low = self.visit(node.slice.lower) if node.slice.lower else None
            if high is not None and low is not None:
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


class ConditionBuilder:
    """Build and execute conditions with runtime data"""

    def build(
        self, task: "Task", global_scope: str, fsdb_builder: "FsdbBuilder"
    ) -> Condition:
        """Build a condition from task"""
        task_scope = task.scope or ""
        condition_str = task.raw_condition
        has_pattern = "{" in condition_str and "}" in condition_str

        if has_pattern:
            patterns = re.findall(r"[\w.]+\{[\w]+\}[\w.\[\]:]*", condition_str)
            var_names = set()
            for p in patterns:
                var_names.update(re.findall(r"\{(\w+)\}", p))
            if len(var_names) != 1:
                raise ValueError(
                    f"Currently only one variable supported, found: {var_names}"
                )
            var_name = list(var_names)[0]
            evaluator = self._build_pattern_evaluator(
                patterns,
                var_name,
                condition_str,
                task_scope,
                global_scope,
                fsdb_builder,
            )
        else:
            evaluator = self._build_simple_evaluator(
                condition_str, task_scope, global_scope
            )

        return Condition(expr=condition_str, evaluator=evaluator)

    def _build_pattern_evaluator(
        self,
        patterns: list[str],
        var_name: str,
        condition_str: str,
        task_scope: str,
        global_scope: str,
        fsdb_builder: "FsdbBuilder",
    ) -> Callable[[dict[str, Any]], bool]:
        """Build evaluator for pattern conditions"""

        def evaluator(runtime_data: dict[str, Any]) -> bool:
            possible_vals = set()
            for pattern in patterns:
                for _, captured in fsdb_builder.find_matching_signals(
                    pattern, task_scope, global_scope
                ):
                    if var_name in captured:
                        possible_vals.add(captured[var_name])

            candidates = SignalGroup(values=list(possible_vals))

            def test_value(val: str) -> bool:
                test_cond = condition_str
                for pattern in patterns:
                    test_cond = test_cond.replace(
                        pattern, pattern.replace(f"{{{var_name}}}", val)
                    )
                try:
                    expr_eval = ExpressionEvaluator(
                        task_scope, global_scope, runtime_data
                    )
                    return bool(expr_eval.eval(test_cond))
                except (ValueError, RuntimeError, KeyError):
                    return False

            matched_group = candidates.filter(test_value)

            try:
                matched_val = matched_group.unique()
                runtime_data["vars"] = {var_name: matched_val}
                return True
            except ValueError:
                if len(matched_group.values) == 0:
                    return False
                raise ValueError(
                    f"Ambiguous match: {matched_group.values} for '{var_name}'"
                )

        return evaluator

    def _build_simple_evaluator(
        self, condition_str: str, task_scope: str, global_scope: str
    ) -> Callable[[dict[str, Any]], bool]:
        """Build evaluator for simple conditions"""

        def evaluator(runtime_data: dict[str, Any]) -> bool:
            expr_eval = ExpressionEvaluator(task_scope, global_scope, runtime_data)
            return bool(expr_eval.eval(condition_str))

        return evaluator

    def exec(self, condition: Condition, runtime_data: dict[str, Any]) -> bool:
        """Execute condition with runtime data"""
        try:
            return condition.evaluator(runtime_data)
        except Exception as e:
            raise ValueError(f"Failed to evaluate condition '{condition.expr}': {e}")
