#!/usr/bin/env python3
import re
from typing import Any, Callable, List, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from yaml_builder import Task


@dataclass
class Condition:
    """Compiled condition that can be executed with runtime data"""
    expr: str
    evaluator: Callable[[dict[str, Any]], bool]


class ConditionBuilder:
    """Build and execute conditions with runtime data"""

    def __init__(self, fsdb_builder: Any, yaml_builder: Any) -> None:
        self.fsdb_builder = fsdb_builder
        self.yaml_builder = yaml_builder

    def build(self, condition_str: str, task: 'Task', global_scope: str) -> Condition:
        """Build a condition from string"""
        task_scope = task.scope or ''

        if '{' in condition_str and '}' in condition_str:
            evaluator = self._build_pattern_evaluator(condition_str, task, task_scope, global_scope)
        elif '<@' in condition_str:
            evaluator = self._build_containment_evaluator(condition_str, task, task_scope, global_scope)
        else:
            evaluator = self._build_simple_evaluator(condition_str, task_scope, global_scope)

        return Condition(expr=condition_str, evaluator=evaluator)

    def exec(self, condition: Condition, runtime_data: dict[str, Any]) -> bool:
        """Execute condition with runtime data"""
        try:
            return condition.evaluator(runtime_data)
        except Exception as e:
            raise ValueError(f"Failed to evaluate condition '{condition.expr}': {e}")

    def _build_simple_evaluator(self, condition_str: str, task_scope: str,
                                global_scope: str) -> Callable[[dict[str, Any]], bool]:
        """Build evaluator for simple conditions"""
        def verilog_to_int(match: re.Match[str]) -> str:
            parts = match.group(0).split("'")
            base_map = {'b': 2, 'o': 8, 'd': 10, 'h': 16}
            base = base_map.get(parts[1][0].lower(), 10)
            return str(int(parts[1][1:].replace('_', ''), base))

        normalized = re.sub(r"\d+'[bhdoBHDO][0-9a-fA-F_]+", verilog_to_int, condition_str)
        signal_names = [t for t in re.findall(r'\b[a-zA-Z_]\w*\b', normalized)
                       if t not in ['and', 'or', 'not', 'True', 'False'] and f'$dep.{t}' not in condition_str]

        def evaluator(runtime_data: dict[str, Any]) -> bool:
            row_idx = runtime_data['row_idx']
            upstream_row = runtime_data.get('upstream_row', {})
            upstream_data = runtime_data.get('upstream_data', {})
            context: dict[str, Any] = {}
            expr = normalized

            for token in signal_names:
                try:
                    signal = self.yaml_builder._resolve_signal_path(token, task_scope, global_scope)
                    values = self.fsdb_builder.dump_signal(signal)
                    val = values[row_idx] if row_idx < len(values) else '0'
                    if val in ['x', 'z', 'X', 'Z'] or val.startswith('*'):
                        context[token] = 0
                    else:
                        context[token] = int('0x' + val if not val.startswith('0x') else val, 16)
                except (ValueError, RuntimeError, KeyError):
                    pass

            if upstream_row and upstream_data:
                for match in re.finditer(r'\$dep\.(\w+)\.(\w+)', expr):
                    dep_task_id, signal_name = match.groups()
                    for sig in upstream_data.get('signals', []):
                        if sig.endswith('.' + signal_name) or sig.endswith('/' + signal_name) or sig == signal_name:
                            val = upstream_row['signals'].get(sig, '0')
                            var_name = f'dep_{dep_task_id}_{signal_name}'
                            context[var_name] = int(val, 16) if val.startswith('0x') else int('0x' + val, 16)
                            expr = expr.replace(f'$dep.{dep_task_id}.{signal_name}', var_name)
                            break

            return bool(eval(expr, {"__builtins__": {}}, context))

        return evaluator

    def _build_pattern_evaluator(self, condition_str: str, task: 'Task',
                                 task_scope: str, global_scope: str) -> Callable[[dict[str, Any]], bool]:
        """Build evaluator for pattern matching conditions"""
        patterns = re.findall(r'[\w.]+\{[\w]+\}[\w.\[\]:]*', condition_str)

        def evaluator(runtime_data: dict[str, Any]) -> bool:
            if not patterns:
                return self._build_simple_evaluator(condition_str, task_scope, global_scope)(runtime_data)

            all_matches = {}
            for pattern in patterns:
                matches = self._find_matching_signals(pattern, task_scope, global_scope)
                all_matches[pattern] = matches

            var_names = set()
            for pattern in patterns:
                var_names.update(re.findall(r'\{(\w+)\}', pattern))

            if len(var_names) != 1:
                raise ValueError(f"Pattern matching currently supports only one variable, found: {var_names}")

            var_name = list(var_names)[0]
            possible_values = set()
            for pattern, matches in all_matches.items():
                for sig, captured in matches:
                    if var_name in captured:
                        possible_values.add(captured[var_name])

            matched_values = []
            for val in possible_values:
                test_condition = condition_str
                for pattern in patterns:
                    test_condition = test_condition.replace(pattern, pattern.replace(f'{{{var_name}}}', val))
                try:
                    if self._build_simple_evaluator(test_condition, task_scope, global_scope)(runtime_data):
                        matched_values.append(val)
                except (ValueError, RuntimeError, KeyError):
                    pass

            if len(matched_values) == 0:
                return False
            elif len(matched_values) == 1:
                task.metadata['_captured_vars'] = {var_name: matched_values[0]}
                return True
            else:
                raise ValueError(f"Ambiguous pattern match: multiple values matched {matched_values} for variable '{var_name}'")

        return evaluator

    def _build_containment_evaluator(self, condition_str: str, task: 'Task',
                                    task_scope: str, global_scope: str) -> Callable[[dict[str, Any]], bool]:
        """Build evaluator for containment operator <@"""
        parts = condition_str.split('<@')
        if len(parts) != 2:
            raise ValueError(f"Invalid <@ expression: {condition_str}")

        left_expr = parts[0].strip()
        right_expr = parts[1].strip()

        if '.$split(' not in right_expr:
            raise ValueError(f"Right side of <@ must use $split(): {right_expr}")

        split_match = re.match(r'(.+)\.\$split\((\d+)\)', right_expr)
        if not split_match:
            raise ValueError(f"Invalid $split syntax: {right_expr}")

        signal_expr = split_match.group(1)
        num_parts = int(split_match.group(2))

        def evaluator(runtime_data: dict[str, Any]) -> bool:
            left_val = self._eval_signal_expr(left_expr, task_scope, global_scope, runtime_data)
            right_val = self._eval_signal_expr(signal_expr, task_scope, global_scope, runtime_data)
            parts_list = self._split_signal(hex(right_val), num_parts)
            return left_val in parts_list

        return evaluator

    def _find_matching_signals(self, pattern: str, task_scope: str, global_scope: str) -> List[tuple[str, dict[str, str]]]:
        """Find signals matching a pattern with {variable} placeholders"""
        pattern = self.yaml_builder._resolve_signal_path(pattern, task_scope, global_scope)
        var_pattern = re.findall(r'\{(\w+)\}', pattern)
        if not var_pattern:
            return [(pattern, {})]

        regex_pattern = '^' + re.escape(pattern)
        for var in var_pattern:
            regex_pattern = regex_pattern.replace(re.escape(f'{{{var}}}'), r'(\d+)')
        regex_pattern += '$'

        all_signals = self.fsdb_builder.get_all_signals()
        matches = []
        for sig in all_signals:
            sig_dot = sig.replace('/', '.').lstrip('.')
            match = re.match(regex_pattern, sig_dot)
            if match:
                captured = {var: match.group(i + 1) for i, var in enumerate(var_pattern)}
                matches.append((sig_dot, captured))
        return matches

    def _eval_signal_expr(self, expr: str, task_scope: str, global_scope: str, runtime_data: dict[str, Any]) -> int:
        """Evaluate a signal expression and return its integer value"""
        row_idx = runtime_data['row_idx']
        upstream_row = runtime_data.get('upstream_row', {})
        upstream_data = runtime_data.get('upstream_data', {})

        if expr.startswith('$dep.'):
            dep_match = re.match(r'\$dep\.(\w+)\.(\w+)(?:\[[\d:]+\])?', expr)
            if not dep_match:
                raise ValueError(f"Invalid $dep reference: {expr}")
            dep_task_id, signal_name = dep_match.groups()
            for sig in upstream_data.get('signals', []):
                if sig.endswith('.' + signal_name) or sig.endswith('/' + signal_name) or sig == signal_name:
                    val_str = upstream_row['signals'].get(sig, '0')
                    return int(val_str, 16) if val_str.startswith('0x') else int('0x' + val_str, 16)
            raise ValueError(f"Signal '{signal_name}' not found in upstream task '{dep_task_id}'")

        bit_range_match = re.match(r'(.+)\[(\d+):(\d+)\]', expr)
        if bit_range_match:
            signal_name, high_bit, low_bit = bit_range_match.group(1), int(bit_range_match.group(2)), int(bit_range_match.group(3))
        else:
            signal_name, high_bit, low_bit = expr, None, None

        signal = self.yaml_builder._resolve_signal_path(signal_name, task_scope, global_scope)
        values = self.fsdb_builder.dump_signal(signal)
        val_str = values[row_idx] if row_idx < len(values) else '0'

        if val_str in ['x', 'z', 'X', 'Z'] or val_str.startswith('*'):
            val_int = 0
        else:
            val_int = int('0x' + val_str if not val_str.startswith('0x') else val_str, 16)

        if high_bit is not None and low_bit is not None:
            val_int = (val_int >> low_bit) & ((1 << (high_bit - low_bit + 1)) - 1)

        return val_int

    def _split_signal(self, signal_val: str, num_parts: int) -> list[int]:
        """Split a wide signal value into equal parts"""
        if signal_val.startswith('0x') or signal_val.startswith('0X'):
            signal_val = signal_val[2:]
        val_int = int(signal_val, 16)
        total_bits = len(signal_val) * 4
        bits_per_part = total_bits // num_parts
        mask = (1 << bits_per_part) - 1
        return [(val_int >> (i * bits_per_part)) & mask for i in range(num_parts)]
