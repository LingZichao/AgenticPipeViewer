#!/usr/bin/env python3
import argparse
import subprocess
import yaml
import ast
import re
from pathlib import Path
from typing  import List, Union, Any


class FsdbAnalyzer:
    """Advanced FSDB signal analyzer with complex condition support"""

    def __init__(self, config_path: str, need_verbose: bool = False):
        """Initialize analyzer from config file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"[ERROR] Config file not found: {config_path}")

        if config_file.suffix not in ['.yaml', '.yml']:
            print(f"[WARN] Config file extension is not .yaml or .yml: {config_path}")

        self.verbose      = need_verbose
        self.config_path  = config_path
        self.line_map     = {}  # Initialize before _load_config
        self.config       = self._load_config(config_path)

        self.fsdb_file    = Path(self.config['fsdbFile'])
        self.output_dir   = Path(self.config['output']['directory'])
        self.clock_signal = self.config['clockSignal']
        self.global_scope = self.config['scope']
        self.verbose      = self.config['output']['verbose']
        self.task_data    = {}  # Store captured data from tasks for downstream use
        self.task_execution_order = []  # Topologically sorted task order
        self.signal_cache = {}  # Cache for dumped signal data
        self.all_signals_list = None  # Cache for all signals in FSDB
    
    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration with validation"""
        # First load with line number tracking
        try:
            with open(config_path, 'r') as f:
                # Load with composer to get line numbers
                self._extract_line_numbers(config_path)
                
            # Then load normally for actual data
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

        except yaml.YAMLError as e:
            # Extract line number from YAML error if available
            line_info = ""
            problem_mark = getattr(e, 'problem_mark', None)
            if problem_mark is not None:
                line = getattr(problem_mark, 'line', None)
                if line is not None:
                    line_info = f" at line {line + 1}"
            raise ValueError(f"[ERROR] YAML syntax error{line_info}: {e}")
        
        if 'fsdbFile' not in config:
            line_info = self._get_line_info('fsdbFile')
            raise ValueError(f"[ERROR] Missing required field 'fsdbFile' {line_info}")
        
        # Validate FSDB file exists and has correct extension
        fsdb_path = Path(config['fsdbFile'])
        if not fsdb_path.exists():
            raise FileNotFoundError(f"[ERROR] FSDB file not found: {fsdb_path}")
        
        if not str(fsdb_path).endswith('.fsdb'):
            print(f"[WARN] Target FSDB file extension is not .fsdb: {fsdb_path}")
        
        if 'tasks' not in config or not config['tasks']:
            line_info = self._get_line_info('tasks')
            raise ValueError(f"[ERROR] Missing 'tasks' field or task list is empty {line_info}")
        
        # Set default values for optional fields
        config.setdefault('clock_signal', 'clk')
        config.setdefault('scope', '')
        
        if 'output' not in config:
            config['output'] = {}
        config['output'].setdefault('directory', 'temp_reports')
        config['output'].setdefault('verbose', False)
        config['output'].setdefault('dependency_graph', 'deps.png')  # None or filename
        
        # Create output directory
        output_dir = Path(config['output']['directory'])
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to create output directory {output_dir}: {e}")
        
        # Validate each task configuration
        for idx, task in enumerate(config['tasks'], 1):
            self._validate_task(task, idx)
        
        # Validate task dependencies
        self._validate_dependencies(config['tasks'])
        
        return config
    
    def _validate_task(self, task: dict, task_num: int):
        """Validate task configuration"""
        line_info = self._get_line_info(f'tasks[{task_num-1}]')

        # Get task identifier for error messages (prefer id, fallback to name or task number)
        task_identifier = task.get('id') or task.get('name') or f'Task {task_num}'

        # Helper function for non-empty string validation
        def require_nonempty(value: Any, field_name: str) -> None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"[ERROR] Task '{task_identifier}' '{field_name}' must be non-empty string{line_info}")
        
        # Validate name field (optional)
        if 'name' in task:
            require_nonempty(task['name'], "name")

        # Validate id field (Required)
        if 'id' in task:
            task_id = task['id']
            require_nonempty(task_id, 'id')
            if not task_id.replace('_', '').replace('-', '').isalnum():
                raise ValueError(f"[ERROR] Task '{task_identifier}' id '{task_id}' contains invalid characters{line_info}")

        # Validate scope field (optional)
        if 'scope' in task:
            require_nonempty(task['scope'], 'scope')

        # Validate depends field
        if 'dependsOn' in task:
            depends = task['dependsOn']
            if isinstance(depends, str):
                require_nonempty(depends, 'dependsOn')

            elif isinstance(depends, list):
                if not depends:
                    raise ValueError(f"[ERROR] Task '{task_identifier}' 'dependsOn' list cannot be empty{line_info}")
                for dep_id in depends:
                    if not isinstance(dep_id, str) or not dep_id.strip():
                        raise ValueError(f"[ERROR] Task '{task_identifier}' 'dependsOn' contains invalid dependency{line_info}")
            else:
                raise ValueError(f"[ERROR] Task '{task_identifier}' 'dependsOn' must be string or list{line_info}")

        # Check required fields
        if 'condition' not in task:
            raise ValueError(f"[ERROR] Task '{task_identifier}' missing 'condition' field{line_info}")

        # Validate condition structure
        condition = task['condition']
        if isinstance(condition, str):
            require_nonempty(condition, 'condition')
        elif isinstance(condition, list):
            if not condition:
                raise ValueError(f"[ERROR] Task '{task_identifier}' 'condition' list cannot be empty{line_info}")
            for line in condition:
                if not isinstance(line, str) or not line.strip():
                    raise ValueError(f"[ERROR] Task '{task_identifier}' 'condition' contains invalid line{line_info}")
        else:
            raise ValueError(f"[ERROR] Task '{task_identifier}' 'condition' must be string or list of strings{line_info}")

        # Validate capture fields
        capture = task.get('capture', [])
        if not capture:
            print(f"[WARN] Task '{task_identifier}' has no signals specified in 'capture' field{line_info}")
        elif not isinstance(capture, list):
            raise ValueError(f"[ERROR] Task '{task_identifier}' 'capture' field must be a list{line_info}")
    
    def _validate_dependencies(self, tasks: list):
        """Validate task dependency graph and detect cycles"""
        # Helper to normalize depends to list
        def get_dep_list(depends: Union[str, list, None]) -> list:
            if not depends:
                return []
            return [depends] if isinstance(depends, str) else depends
        
        # Build task ID map
        task_map = {}
        for idx, task in enumerate(tasks):
            task_id = task.get('id')
            if not task_id:
                continue
            if task_id in task_map:
                raise ValueError(f"[ERROR] Duplicate task id '{task_id}' found")
            task_map[task_id] = idx
        
        # Validate all dependencies exist
        for idx, task in enumerate(tasks):
            task_display_name = task.get('name') or task.get('id') or f'Task {idx+1}'
            for dep_id in get_dep_list(task.get('dependsOn')):
                if dep_id not in task_map:
                    raise ValueError(f"[ERROR] Task '{task_display_name}' depends on non-existent task '{dep_id}'")
        
        # Detect cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_idx: int, path: list) -> bool:
            """DFS to detect cycles"""
            visited.add(task_idx)
            rec_stack.add(task_idx)
            path.append(tasks[task_idx].get('id', f'task_{task_idx}'))
            
            for dep_id in get_dep_list(tasks[task_idx].get('dependsOn')):
                dep_idx = task_map[dep_id]
                if dep_idx not in visited:
                    if has_cycle(dep_idx, path):
                        return True
                elif dep_idx in rec_stack:
                    # Cycle detected
                    cycle_start = path.index(dep_id)
                    cycle = ' -> '.join(path[cycle_start:] + [dep_id])
                    raise ValueError(f"[ERROR] Circular dependency detected: {cycle}")
            
            path.pop()
            rec_stack.remove(task_idx)
            return False
        
        # Check all tasks
        for idx in range(len(tasks)):
            if idx not in visited:
                has_cycle(idx, [])

    def _extract_line_numbers(self, config_path: str):
        """Extract line numbers for all keys in YAML file"""
        try:
            with open(config_path, 'r') as f:
                lines = f.readlines()
                
            # Simple line number extraction based on key patterns
            for line_num, line in enumerate(lines, 1):
                stripped = line.lstrip()
                if stripped and not stripped.startswith('#') and ':' in stripped:
                    # Extract key name
                    key = stripped.split(':')[0].strip()
                    if key.startswith('- '):
                        # List item
                        continue
                    # indent_level = len(line) - len(stripped)
                    self.line_map[key] = line_num
        except Exception:
            # If extraction fails, just continue without line numbers
            pass
    
    def _get_line_info(self, key_path: str) -> str:
        """Get line information for error messages"""
        # Try to find line number for the key
        # Key path format: 'tasks[0].condition' -> look for 'condition'
        parts = key_path.replace('[', '.').replace(']', '').split('.')
        
        # Try to find any matching key
        for part in reversed(parts):
            if part.isdigit():
                continue
            if part in self.line_map:
                return f" (line {self.line_map[part]})"
        
        return ""
    
    def _to_fsdb_path(self, signal: str) -> str:
        """Convert signal path from dot notation to FSDB format (slash)"""
        # Handle already converted paths
        if signal.startswith('/'):
            return signal
        # Convert dot notation to slash
        return '/' + signal.replace('.', '/')
    
    def _resolve_signal_path(self, signal: str, task_scope: str = '') -> str:
        """Resolve signal path with scope support
        
        Rules:
        1. $mod -> replace with current scope
        2. Relative path (no 'tb.' prefix, no '/') -> add scope prefix
        3. Absolute path (starts with 'tb.' or '/') -> keep as-is
        
        Args:
            signal: Signal path to resolve
            task_scope: Task-level scope (overrides global scope)
        
        Returns:
            Resolved signal path
        """
        if not isinstance(signal, str):
            return signal
        
        # Determine effective scope (task > global)
        scope = task_scope if task_scope else self.global_scope
        
        # Handle $mod reference
        if signal.startswith('$mod.'):
            if not scope:
                raise ValueError(f"[ERROR] $mod used but no scope defined: {signal}")
            # Replace $mod with scope
            signal = signal.replace('$mod.', scope + '.', 1)
        elif signal == '$mod':
            if not scope:
                raise ValueError("[ERROR] $mod used but no scope defined")
            return scope
        # Handle relative paths (add scope prefix)
        elif scope and not signal.startswith('tb.') and not signal.startswith('/'):
            # Relative path: prepend scope
            signal = scope + '.' + signal
        # Absolute paths (tb.* or /*) remain unchanged
        
        return signal
    
    def _resolve_dep_references(self, value: Any, task_id: str) -> Any:
        """Resolve $dep.task_id.signal references to actual values"""
        if isinstance(value, str):
            # Check for $dep reference pattern
            if value.startswith('$dep.'):
                # Parse: $dep.task_id.var_name
                parts = value.split('.', 2)
                if len(parts) != 3:
                    raise ValueError(f"[ERROR] Invalid $dep reference: {value}. Format: $dep.task_id.var_name")
                
                _, dep_task_id, var_name = parts
                
                # Check if dependency data exists
                if dep_task_id not in self.task_data:
                    raise ValueError(f"[ERROR] Task '{task_id}' references non-existent or not-yet-executed task '{dep_task_id}'")
                
                task_exports = self.task_data[dep_task_id]
                if var_name not in task_exports:
                    available = ', '.join(task_exports.keys())
                    raise ValueError(
                        f"[ERROR] Task '{task_id}' references non-existent variable '{var_name}' "
                        f"from task '{dep_task_id}'. Available: {available}"
                    )
                
                # Return the signal path
                return task_exports[var_name]
            return value
        elif isinstance(value, list):
            # Recursively resolve in lists
            return [self._resolve_dep_references(item, task_id) for item in value]
        elif isinstance(value, dict):
            # Recursively resolve in dicts
            return {k: self._resolve_dep_references(v, task_id) for k, v in value.items()}
        else:
            return value

    def _build_execution_order(self) -> List[int]:
        """Build topologically sorted task execution order"""
        tasks = self.config.get('tasks', [])
        
        # Helper to normalize depends to list
        def get_dep_list(depends: Union[str, list, None]) -> list:
            if not depends:
                return []
            return [depends] if isinstance(depends, str) else depends
        
        # Build task ID to index map
        task_map = {task['id']: idx for idx, task in enumerate(tasks) if task.get('id')}
        
        # Build adjacency list (dependency graph)
        in_degree = [0] * len(tasks)
        adj_list = [[] for _ in range(len(tasks))]
        
        for idx, task in enumerate(tasks):
            for dep_id in get_dep_list(task.get('dependsOn')):
                dep_idx = task_map[dep_id]
                adj_list[dep_idx].append(idx)
                in_degree[idx] += 1
        
        # Kahn's algorithm for topological sort
        queue = [i for i in range(len(tasks)) if in_degree[i] == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(tasks):
            raise RuntimeError("[ERROR] Failed to build execution order - circular dependency")
        
        return result
    
    def _export_dependency_graph(self, output_file: str) -> None:
        """Export task dependency graph to image file"""
        try:
            from graphviz import Digraph
        except ImportError:
            print("[WARN] graphviz not installed. Skip dependency graph export.")
            print("[WARN] Install with: pip install graphviz")
            return
        
        tasks = self.config['tasks']
        
        # Helper to normalize depends to list
        def get_dep_list(depends: Union[str, list, None]) -> list:
            if not depends:
                return []
            return [depends] if isinstance(depends, str) else depends
        
        # Create graph
        dot = Digraph(comment='Task Dependencies')
        dot.attr(rankdir='LR')  # Left to right layout
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
        
        # Add nodes
        for idx, task in enumerate(tasks):
            task_id = task.get('id', f'task_{idx}')
            # Use name for display, fallback to id
            task_display_name = task.get('name') or task.get('id') or f'Task {idx+1}'

            # Add execution order if available
            if hasattr(self, 'task_execution_order') and idx in self.task_execution_order:
                exec_order = self.task_execution_order.index(idx) + 1
                label = f"[{exec_order}] {task_display_name}\\n({task_id})"
            else:
                label = f"{task_display_name}\\n({task_id})"

            dot.node(task_id, label)
        
        # Add edges (dependencies)
        for idx, task in enumerate(tasks):
            task_id = task.get('id', f'task_{idx}')
            for dep_id in get_dep_list(task.get('dependsOn')):
                dot.edge(dep_id, task_id)
        
        # Determine output format and path
        output_path = self.output_dir / output_file
        file_ext = output_path.suffix.lstrip('.')
        if not file_ext:
            file_ext = 'png'
            output_path = output_path.with_suffix('.png')
        
        # Render graph
        try:
            dot.render(str(output_path.with_suffix('')), format=file_ext, cleanup=True)
            print(f"[INFO] Dependency graph exported to: {output_path}")
        except Exception as e:
            print(f"[WARN] Failed to export dependency graph: {e}")
    
    def _dump_signal(self, signal: str) -> List[str]:
        """Dump all values of a signal from FSDB"""
        if signal in self.signal_cache:
            return self.signal_cache[signal]
        raise RuntimeError(f"Signal {signal} not found in cache. Call _dump_all_signals first.")

    def _get_all_signals(self) -> List[str]:
        """Get all signals from FSDB using fsdbdebug"""
        if self.all_signals_list is not None:
            return self.all_signals_list

        cmd = ['fsdbdebug', '-hier_tree', str(self.fsdb_file.absolute())]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get signal hierarchy: {result.stderr}")

        # Parse output to extract signal paths
        signals = []
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and '/' in line:
                # Extract signal path (format: /path/to/signal)
                if line.startswith('/'):
                    signals.append(line)

        self.all_signals_list = signals
        return signals

    def _find_matching_signals(self, pattern: str, scope: str = '') -> List[tuple]:
        """Find signals matching a pattern with {variable} placeholders

        Returns list of (signal_path, captured_vars_dict) tuples
        """
        # Resolve pattern with scope
        pattern = self._resolve_signal_path(pattern, scope)

        # Extract variable names from pattern
        var_pattern = re.findall(r'\{(\w+)\}', pattern)
        if not var_pattern:
            # No variables, just return the signal if it exists
            return [(pattern, {})]

        # Convert pattern to regex
        regex_pattern = re.escape(pattern)
        for var in var_pattern:
            regex_pattern = regex_pattern.replace(re.escape(f'{{{var}}}'), r'(\d+)')
        regex_pattern = '^' + regex_pattern + '$'

        # Get all signals and match
        all_signals = self._get_all_signals()
        matches = []

        for sig in all_signals:
            # Convert to dot notation for matching
            sig_dot = sig.replace('/', '.').lstrip('.')
            match = re.match(regex_pattern, sig_dot)
            if match:
                captured = {}
                for i, var in enumerate(var_pattern):
                    captured[var] = match.group(i + 1)
                matches.append((sig_dot, captured))

        return matches

    def _dump_all_signals(self, signals: list):
        """Dump all signals at once using single fsdbreport call"""
        if not signals:
            return

        # Convert to FSDB paths
        fsdb_paths = [self._to_fsdb_path(sig) for sig in signals]

        # Use temp file for output
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp_file = tmp.name

        try:
            # Build command with all signals (default hex, max width 1024)
            cmd = ['fsdbreport', str(self.fsdb_file.absolute()), '-of', 'h', '-w','1024', '-s'] + fsdb_paths + ['-o', tmp_file]

            print(f"[INFO] Dumping {len(signals)} signal(s) from FSDB...")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to dump signals: {result.stderr}")

            # Read output file
            with open(tmp_file, 'r') as f:
                lines = f.readlines()

            # Copy to verbose output if requested
            if self.verbose:
                output_file = self.output_dir / 'fsdb_dump.txt'
                with open(output_file, 'w') as f:
                    f.writelines(lines)

            # Parse table format output
            # Find header line (contains signal names)
            header_idx = -1
            for i, line in enumerate(lines):
                if 'Time' in line:
                    header_idx = i
                    break

            if header_idx == -1:
                raise RuntimeError("Cannot find header in fsdbreport output")

            # Initialize cache for all signals
            for sig in signals:
                self.signal_cache[sig] = []

            # Skip separator line
            data_start = header_idx + 2

            # Parse data lines
            for line in lines[data_start:]:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue

                # parts[0] is time, parts[1:] are signal values
                for idx, val in enumerate(parts[1:]):
                    if idx < len(signals):
                        self.signal_cache[signals[idx]].append(val)

        finally:
            # Clean up temp file
            import os
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)

    def _normalize_condition(self, condition: Union[str, list]) -> str:
        """Normalize condition to single Python expression string"""
        if isinstance(condition, str):
            return condition.strip()
        elif isinstance(condition, list):
            return ' '.join(line.strip() for line in condition if line.strip())
        return str(condition)

    def _eval_condition(self, condition_str: str, task: dict, task_id: str, row_idx: int,
                       upstream_row: dict, upstream_data: dict) -> bool:
        """Evaluate Python expression condition with pattern matching support"""
        task_scope = task.get('scope', '')

        # Check if condition contains pattern matching {var}
        if '{' in condition_str and '}' in condition_str:
            return self._eval_pattern_condition(condition_str, task, task_id, row_idx, upstream_row, upstream_data)

        # Handle <@ operator (containment check)
        if '<@' in condition_str:
            return self._eval_containment(condition_str, task, task_id, row_idx, upstream_row, upstream_data)

        # Convert Verilog literals to Python integers
        def verilog_to_int(match):
            width_base = match.group(0)
            parts = width_base.split("'")
            base_char = parts[1][0].lower()
            value_str = parts[1][1:].replace('_', '')

            base_map = {'b': 2, 'o': 8, 'd': 10, 'h': 16}
            base = base_map.get(base_char, 10)
            return str(int(value_str, base))

        condition_str = re.sub(r"\d+'[bhdoBHDO][0-9a-fA-F_]+", verilog_to_int, condition_str)

        # Build evaluation context with signal values
        context = {}

        # Extract all identifiers from expression
        for token in re.findall(r'\b[a-zA-Z_]\w*\b', condition_str):
            if token in ['and', 'or', 'not', 'True', 'False']:
                continue

            # Handle $dep references
            if condition_str.find(f'$dep.{token}') != -1:
                continue  # Skip, will be handled separately

            # Resolve as signal
            try:
                signal = self._resolve_signal_path(token, task_scope)
                values = self._dump_signal(signal)
                val = values[row_idx] if row_idx < len(values) else '0'
                # Handle special values (x, z, *, etc.)
                if val in ['x', 'z', 'X', 'Z'] or val.startswith('*'):
                    context[token] = 0
                else:
                    # Add 0x prefix if not present for hex values
                    if not val.startswith('0x') and not val.startswith('0X'):
                        val = '0x' + val
                    context[token] = int(val, 16)
            except (ValueError, RuntimeError, KeyError):
                pass

        # Handle $dep references if upstream context exists
        if upstream_row and upstream_data:
            dep_pattern = r'\$dep\.(\w+)\.(\w+)'
            for match in re.finditer(dep_pattern, condition_str):
                dep_task_id, signal_name = match.groups()
                upstream_signals = upstream_data.get('signals', [])
                for sig in upstream_signals:
                    if sig.endswith('.' + signal_name) or sig.endswith('/' + signal_name) or sig == signal_name:
                        val = upstream_row['signals'].get(sig, '0')
                        var_name = f'dep_{dep_task_id}_{signal_name}'
                        context[var_name] = int(val, 16)
                        condition_str = condition_str.replace(f'$dep.{dep_task_id}.{signal_name}', var_name)
                        break

        try:
            return bool(eval(condition_str, {"__builtins__": {}}, context))
        except Exception as e:
            raise ValueError(f"Failed to evaluate condition '{condition_str}': {e}")

    def _split_signal(self, signal_val: str, num_parts: int) -> list:
        """Split a wide signal value into equal parts

        Args:
            signal_val: Hex string value (with or without 0x prefix)
            num_parts: Number of parts to split into

        Returns:
            List of hex values, from LSB to MSB
        """
        # Remove 0x prefix if present
        if signal_val.startswith('0x') or signal_val.startswith('0X'):
            signal_val = signal_val[2:]

        # Convert to integer
        val_int = int(signal_val, 16)

        # Calculate bits per part
        total_bits = len(signal_val) * 4  # Each hex digit = 4 bits
        bits_per_part = total_bits // num_parts

        # Split into parts
        parts = []
        mask = (1 << bits_per_part) - 1
        for i in range(num_parts):
            part = (val_int >> (i * bits_per_part)) & mask
            parts.append(part)

        return parts

    def _eval_containment(self, condition_str: str, task: dict, task_id: str, row_idx: int,
                         upstream_row: dict, upstream_data: dict) -> bool:
        """Evaluate containment operator <@

        Format: left_expr <@ right_expr.$split(n)
        """
        # Split by <@ operator
        parts = condition_str.split('<@')
        if len(parts) != 2:
            raise ValueError(f"Invalid <@ expression: {condition_str}")

        left_expr = parts[0].strip()
        right_expr = parts[1].strip()

        # Evaluate left side (should be a signal with optional bit range)
        left_val = self._eval_signal_expr(left_expr, task, row_idx, upstream_row, upstream_data)

        # Evaluate right side (should be signal.$split(n))
        if '.$split(' not in right_expr:
            raise ValueError(f"Right side of <@ must use $split(): {right_expr}")

        # Parse $split expression
        split_match = re.match(r'(.+)\.\$split\((\d+)\)', right_expr)
        if not split_match:
            raise ValueError(f"Invalid $split syntax: {right_expr}")

        signal_expr = split_match.group(1)
        num_parts = int(split_match.group(2))

        # Get signal value
        right_val = self._eval_signal_expr(signal_expr, task, row_idx, upstream_row, upstream_data)

        # Split right value into parts
        parts_list = self._split_signal(hex(right_val), num_parts)

        # Check if left value is in parts
        return left_val in parts_list

    def _eval_signal_expr(self, expr: str, task: dict, row_idx: int,
                         upstream_row: dict, upstream_data: dict) -> int:
        """Evaluate a signal expression and return its integer value

        Handles:
        - Simple signal names
        - Signals with bit ranges [high:low]
        - $dep references
        """
        task_scope = task.get('scope', '')

        # Handle $dep references
        if expr.startswith('$dep.'):
            dep_match = re.match(r'\$dep\.(\w+)\.(\w+)(?:\[[\d:]+\])?', expr)
            if not dep_match:
                raise ValueError(f"Invalid $dep reference: {expr}")

            dep_task_id, signal_name = dep_match.groups()
            upstream_signals = upstream_data.get('signals', [])

            for sig in upstream_signals:
                if sig.endswith('.' + signal_name) or sig.endswith('/' + signal_name) or sig == signal_name:
                    val_str = upstream_row['signals'].get(sig, '0')
                    if val_str.startswith('0x') or val_str.startswith('0X'):
                        return int(val_str, 16)
                    else:
                        return int('0x' + val_str, 16)

            raise ValueError(f"Signal '{signal_name}' not found in upstream task '{dep_task_id}'")

        # Handle regular signals (with optional bit range)
        bit_range_match = re.match(r'(.+)\[(\d+):(\d+)\]', expr)
        if bit_range_match:
            signal_name = bit_range_match.group(1)
            high_bit = int(bit_range_match.group(2))
            low_bit = int(bit_range_match.group(3))
        else:
            signal_name = expr
            high_bit = None
            low_bit = None

        # Resolve signal path
        signal = self._resolve_signal_path(signal_name, task_scope)
        values = self._dump_signal(signal)
        val_str = values[row_idx] if row_idx < len(values) else '0'

        # Convert to integer
        if val_str in ['x', 'z', 'X', 'Z'] or val_str.startswith('*'):
            val_int = 0
        else:
            if not val_str.startswith('0x') and not val_str.startswith('0X'):
                val_str = '0x' + val_str
            val_int = int(val_str, 16)

        # Extract bit range if specified
        if high_bit is not None and low_bit is not None:
            mask = (1 << (high_bit - low_bit + 1)) - 1
            val_int = (val_int >> low_bit) & mask

        return val_int

    def _eval_pattern_condition(self, condition_str: str, task: dict, task_id: str, row_idx: int,
                                upstream_row: dict, upstream_data: dict) -> tuple:
        """Evaluate condition with pattern matching {var}

        Returns (matched, captured_vars) where captured_vars is dict of {var: value}
        """
        task_scope = task.get('scope', '')

        # Extract all patterns with {var}
        patterns = re.findall(r'[\w.]+\{[\w]+\}[\w.\[\]:]*', condition_str)
        if not patterns:
            return self._eval_condition(condition_str, task, task_id, row_idx, upstream_row, upstream_data)

        # For each pattern, find all matching signals
        all_matches = {}
        for pattern in patterns:
            matches = self._find_matching_signals(pattern, task_scope)
            all_matches[pattern] = matches

        # Try each combination of matched signals
        # For simplicity, assume all patterns use the same variable name
        var_names = set()
        for pattern in patterns:
            vars_in_pattern = re.findall(r'\{(\w+)\}', pattern)
            var_names.update(vars_in_pattern)

        if len(var_names) != 1:
            raise ValueError(f"Pattern matching currently supports only one variable, found: {var_names}")

        var_name = list(var_names)[0]

        # Collect all possible values for the variable
        possible_values = set()
        for pattern, matches in all_matches.items():
            for sig, captured in matches:
                if var_name in captured:
                    possible_values.add(captured[var_name])

        # Test each possible value
        matched_values = []
        for val in possible_values:
            # Substitute {var} with actual value in condition
            test_condition = condition_str
            for pattern in patterns:
                actual_signal = pattern.replace(f'{{{var_name}}}', val)
                test_condition = test_condition.replace(pattern, actual_signal)

            # Evaluate the substituted condition
            try:
                if self._eval_condition(test_condition, task, task_id, row_idx, upstream_row, upstream_data):
                    matched_values.append(val)
            except (ValueError, RuntimeError, KeyError):
                pass

        # Check uniqueness
        if len(matched_values) == 0:
            return False
        elif len(matched_values) == 1:
            # Store captured variable for later use
            if not hasattr(task, '_captured_vars'):
                task['_captured_vars'] = {}
            task['_captured_vars'][var_name] = matched_values[0]
            return True
        else:
            raise ValueError(f"Ambiguous pattern match: multiple values matched {matched_values} for variable '{var_name}'")

    def _has_dep_in_condition(self, condition: Union[str, list]) -> bool:
        """Check if condition contains $dep references"""
        condition_str = self._normalize_condition(condition)
        return '$dep.' in condition_str

    def _collect_signals_from_condition(self, condition: Union[str, list], task_scope: str, signals: set):
        """Extract signal identifiers from Python expression using AST"""
        condition_str = self._normalize_condition(condition)

        # Remove $dep references and Verilog literals before parsing
        condition_str = re.sub(r'\$dep\.\w+\.\w+', '0', condition_str)
        condition_str = re.sub(r"\d+'[bhdoBHDO][0-9a-fA-F_]+", '0', condition_str)  # Verilog literals

        try:
            tree = ast.parse(condition_str, mode='eval')
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    token = node.id
                    if token not in ['True', 'False', 'None']:
                        try:
                            resolved = self._resolve_signal_path(token, task_scope)
                            signals.add(resolved)
                        except (ValueError, RuntimeError):
                            pass
        except SyntaxError:
            # Fallback: extract identifiers using regex
            for token in re.findall(r'\b[a-zA-Z_]\w*\b', condition_str):
                if token not in ['and', 'or', 'not', 'True', 'False', 'None']:
                    try:
                        resolved = self._resolve_signal_path(token, task_scope)
                        signals.add(resolved)
                    except (ValueError, RuntimeError):
                        pass

    def _log_match(self, log_format: str, row_data: dict, capture_signals: list, row_idx: int):
        """Format and print log message for matched row"""
        # Build context for formatting
        context = {'__time__': row_idx}
        for sig in capture_signals:
            sig_name = sig.split('.')[-1].split('/')[-1]  # Get signal name without path
            val_str = row_data['signals'].get(sig, '0')
            # Handle special values
            if val_str in ['x', 'z', 'X', 'Z'] or val_str.startswith('*'):
                context[sig_name] = '0x0'
            else:
                # Add 0x prefix if not present for hex values
                if not val_str.startswith('0x') and not val_str.startswith('0X'):
                    val_str = '0x' + val_str
                context[sig_name] = val_str

        try:
            # Use eval with f-string for formatting
            log_msg = eval(f'f"""{log_format}"""', {}, context)
            print(f"  [LOG] {log_msg}")
        except Exception as e:
            print(f"  [LOG ERROR] Failed to format log: {e}")

    def _match_all_rows(self, task: dict, task_id: str, capture_signals: list, condition: Union[str, list]) -> list:
        """Normal mode: match all rows globally"""
        print(f"Evaluating condition for {len(capture_signals)} signal(s)")

        condition_str = self._normalize_condition(condition)

        # Resolve capture signals (handle {var} substitution)
        resolved_capture_signals = []
        for sig in capture_signals:
            if isinstance(sig, str) and '{' not in sig:
                resolved_capture_signals.append(sig)

        # If no resolved signals yet, we need to wait for pattern matching
        if not resolved_capture_signals:
            resolved_capture_signals = capture_signals

        signal_data = {}
        for sig in resolved_capture_signals:
            if '{' not in sig:  # Skip pattern signals for now
                signal_data[sig] = self._dump_signal(sig)

        max_len = max(len(vals) for vals in signal_data.values()) if signal_data else 0

        log_format = task.get('logging')

        # Evaluate condition for each row
        matched_rows = []
        for row_idx in range(max_len):
            try:
                if self._eval_condition(condition_str, task, task_id, row_idx, {}, {}):
                    # Get captured variables if any
                    captured_vars = task.get('_captured_vars', {})

                    # Resolve capture signals with captured variables
                    actual_capture_signals = []
                    for sig in capture_signals:
                        if isinstance(sig, str) and '{' in sig:
                            # Substitute {var} with actual value
                            actual_sig = sig
                            for var_name, var_val in captured_vars.items():
                                actual_sig = actual_sig.replace(f'{{{var_name}}}', var_val)
                            actual_sig = self._resolve_signal_path(actual_sig, task.get('scope', ''))
                            actual_capture_signals.append(actual_sig)
                        else:
                            actual_capture_signals.append(sig)

                    row_data = {'time': row_idx, 'signals': {}}
                    for sig in actual_capture_signals:
                        if sig in signal_data:
                            vals = signal_data[sig]
                            row_data['signals'][sig] = vals[row_idx] if row_idx < len(vals) else '0'
                        else:
                            # Need to dump this signal on-demand
                            try:
                                vals = self._dump_signal(sig)
                                row_data['signals'][sig] = vals[row_idx] if row_idx < len(vals) else '0'
                            except (ValueError, RuntimeError, KeyError):
                                row_data['signals'][sig] = '0'

                    matched_rows.append(row_data)

                    if log_format:
                        self._log_match(log_format, row_data, actual_capture_signals, row_idx)
            except Exception as e:
                print(f"[WARN] Error evaluating condition at row {row_idx}: {e}")

        return matched_rows

    def _trace_from_upstream(self, task: dict, task_id: str, capture_signals: list, condition: Union[str, list]) -> list:
        """Trace mode: match from upstream task"""
        depends = task.get('dependsOn', [])
        dep_id = depends if isinstance(depends, str) else depends[0]

        if dep_id not in self.task_data:
            raise ValueError(f"[ERROR] Upstream task '{dep_id}' not found in task_data")

        upstream_data = self.task_data[dep_id]
        upstream_rows = upstream_data['rows']

        print(f"Tracing from upstream task '{dep_id}' with {len(upstream_rows)} rows")
        if self.verbose and upstream_rows:
            print(f"  Upstream exports: {upstream_data.get('exports', {})}")
            print(f"  First upstream row: time={upstream_rows[0]['time']}, signals={list(upstream_rows[0]['signals'].keys())}")

        condition_str = self._normalize_condition(condition)

        signal_data = {}
        for sig in capture_signals:
            signal_data[sig] = self._dump_signal(sig)
        max_len = max(len(vals) for vals in signal_data.values()) if signal_data else 0

        log_format = task.get('logging')

        # For each upstream row, search forward
        matched_rows = []
        for upstream_row in upstream_rows:
            start_time = upstream_row['time']

            if self.verbose:
                print(f"  Searching from time {start_time}...")

            # Search forward from start_time
            match_found = False
            for row_idx in range(start_time, max_len):
                try:
                    if self._eval_condition(condition_str, task, task_id, row_idx, upstream_row, upstream_data):
                        row_data = {'time': row_idx, 'signals': {}}
                        for sig in capture_signals:
                            vals = signal_data[sig]
                            row_data['signals'][sig] = vals[row_idx] if row_idx < len(vals) else '0'
                        matched_rows.append(row_data)
                        match_found = True
                        if self.verbose:
                            print(f"    Found match at time {row_idx}")
                        if log_format:
                            self._log_match(log_format, row_data, capture_signals, row_idx)
                        break
                except Exception as e:
                    if self.verbose and row_idx == start_time:
                        print(f"    Error at time {row_idx}: {e}")
                    continue

            if not match_found:
                print(f"[WARN] No match found for upstream row at time {start_time}")

        return matched_rows

    def _write_output_file(self, matched_rows: list, capture_signals: list, out_path: Path):
        """Write matched rows to CSV file"""
        with open(out_path, 'w') as f:
            # Header
            f.write('__time__,' + ','.join(capture_signals) + '\n')
            # Data rows
            for row in matched_rows:
                row_data = [str(row['time'])]
                for sig in capture_signals:
                    row_data.append(row['signals'].get(sig, '0'))
                f.write(','.join(row_data) + '\n')

    def _capture_task(self, task: dict, task_id: str) -> str:
        """Execute capture mode task"""
        condition = task.get('condition')
        capture_signals = task.get('capture', [])

        if not condition:
            raise ValueError("[ERROR] Capture task missing 'condition' field")

        if not capture_signals:
            raise ValueError("[ERROR] Capture task must specify signal list in 'capture' field")

        if not isinstance(capture_signals, list):
            raise ValueError(f"[ERROR] 'capture' field must be a list, got: {type(capture_signals).__name__}")

        # Resolve $dep references in capture signals
        capture_signals = self._resolve_dep_references(capture_signals, task_id)

        # Resolve signal paths with scope and handle {var} substitution
        task_scope = task.get('scope', '')
        resolved_signals = []
        for sig in capture_signals:
            if isinstance(sig, str):
                # Check if signal contains {var} pattern
                if '{' in sig and '}' in sig:
                    # Will be resolved per-row during matching
                    resolved_signals.append(sig)
                else:
                    sig = self._resolve_signal_path(sig, task_scope)
                    resolved_signals.append(sig)
            else:
                resolved_signals.append(sig)
        capture_signals = resolved_signals

        # Validate signal paths
        for idx, sig in enumerate(capture_signals, 1):
            if isinstance(sig, list):
                sig_str = ''.join(str(s) for s in sig)
            else:
                sig_str = str(sig)

            if not sig_str or sig_str.isspace():
                raise ValueError(f"[ERROR] capture[{idx-1}] signal is empty")

        # Get output configuration
        output_config = task.get('output', {})
        output_format = output_config.get('format', 'hex')

        valid_formats = ['hex', 'bin', 'dec']
        if output_format not in valid_formats:
            raise ValueError(f"[ERROR] Invalid output format: {output_format}. Valid: {', '.join(valid_formats)}")

        output_file = output_config.get('file')

        # Detect trace mode
        depends = task.get('dependsOn')
        if depends and self._has_dep_in_condition(condition):
            # Trace mode: match from upstream
            matched_rows = self._trace_from_upstream(task, task_id, capture_signals, condition)
        else:
            # Trigger or normal mode: global match
            matched_rows = self._match_all_rows(task, task_id, capture_signals, condition)

        # Store to memory
        self.task_data[task_id] = {
            'rows': matched_rows,
            'signals': capture_signals  # All captured signals are available for reference
        }

        print(f"Matched {len(matched_rows)} rows")

        # Write output file only if configured
        if output_file:
            out_path = self.output_dir / output_file
            self._write_output_file(matched_rows, capture_signals, out_path)
            print(f"Result: Saved to {out_path}\n")
            return str(out_path)
        else:
            print(f"Result: {len(matched_rows)} rows in memory\n")
            return f"[Memory] {len(matched_rows)} rows"

   
    def run(self):
        """Execute all configured analysis tasks"""
        tasks = self.config.get('tasks', [])
        if not tasks:
            print("[WARN] No tasks configured")
            return

        # Build execution order based on dependencies
        self.task_execution_order = self._build_execution_order()

        # Export dependency graph if configured
        dep_graph_file = self.config['output'].get('dependency_graph')
        if dep_graph_file:
            self._export_dependency_graph(dep_graph_file)

        print(f"\n{'='*70}")
        print(f"[INFO] FSDB Analyzer - Running {len(tasks)} task(s)")
        print(f"{'='*70}")
        print(f"[INFO] FSDB file: {self.fsdb_file}")
        print(f"[INFO] Clock signal: {self.clock_signal}")
        print(f"[INFO] Output directory: {self.output_dir}")
        print(f"[INFO] Verbose mode: {'yes' if self.verbose else 'no'}")
        print(f"{'='*70}\n")

        # Collect all signals from all tasks (capture + condition)
        all_signals = set()
        for task in tasks:
            task_scope = task.get('scope', '')
            # Collect from capture
            for sig in task.get('capture', []):
                if isinstance(sig, str):
                    resolved = self._resolve_signal_path(sig, task_scope)
                    all_signals.add(resolved)
            # Collect from condition
            condition = task.get('condition')
            if condition:
                self._collect_signals_from_condition(condition, task_scope, all_signals)

        # Dump all signals at once
        if all_signals:
            self._dump_all_signals(list(all_signals))

        results = []
        for exec_idx, task_idx in enumerate(self.task_execution_order, 1):
            task = tasks[task_idx]
            task_id = task.get('id', f'task_{task_idx}')
            # Use name for display, fallback to id
            task_display_name = task.get('name') or task.get('id') or f'Task {exec_idx}'

            print(f"\n[Task {exec_idx}/{len(tasks)}] {task_display_name}")
            if task.get('id'):
                print(f"  ID: {task['id']}")
            if task.get('dependsOn'):
                deps = task['dependsOn'] if isinstance(task['dependsOn'], list) else [task['dependsOn']]
                print(f"  Depends on: {', '.join(deps)}")
            print(f"{'-'*70}")

            try:
                result = self._capture_task(task, task_id)
                results.append((task_display_name, result))
            except Exception as e:
                print(f"Task failed: {e}")
                import traceback
                traceback.print_exc()
                results.append((task_display_name, f"ERROR: {e}"))
        
        # Summary
        print(f"\n{'='*70}")
        print("Summary:")
        print(f"{'='*70}")
        for name, result in results:
            print(f"  {name}: {result}")
        print()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Advanced FSDB Signal Analyzer with Complex Conditions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run with advanced configuration
  %(prog)s -c ifu_analysis_advanced.yaml
  
  # Keep intermediate reports for debugging
  %(prog)s -c config.yaml --verbose
        '''
    )
    
    parser.add_argument(
        '-c', '--config',
        required=True,
        help='YAML configuration file path'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Keep intermediate report files'
    )
    
    args = parser.parse_args()
    
    analyzer = FsdbAnalyzer(args.config, args.verbose)
    analyzer.run()


if __name__ == '__main__':
    main()
