#!/usr/bin/env python3
import yaml
import re
from pathlib import Path
from typing import Union, Any, List, Optional
from dataclasses import dataclass, field
from utils import resolve_signal_path
from condition_builder import Condition

@dataclass
class Task:
    """Task configuration data structure"""

    id: str
    capture: List[str]
    raw_condition: str
    has_dep_in_condition: bool
    has_pattern_in_capture: bool

    name: Optional[str] = None
    scope: Optional[str] = None
    deps: List[str] = field(default_factory=list)
    logging: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[Condition] = None  # Built Condition object, set later
    match_mode: str = "all"  # Matching mode: "first", "all", "unique_per_var"
    max_match: int = 0  # Maximum matches per upstream trigger (0 = unlimited)

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], global_scope: str, yaml_builder: "YamlBuilder"
    ) -> "Task":
        """Create Task from dictionary with all preprocessing"""
        task_scope = data.get("scope", "")
        # Calculate final scope: task scope overrides global scope
        final_scope = task_scope if task_scope else global_scope

        raw_capture = data.get("capture", [])
        raw_condition = data.get("condition", "")

        # Resolve capture signals to final form
        capture = yaml_builder.resolve_capture_signals(raw_capture, final_scope)

        # Analyze condition and capture
        has_dep = "$dep." in raw_condition
        has_pattern = any("{" in str(sig) and "}" in str(sig) for sig in raw_capture)

        task_id = data.get("id")
        if not task_id or not isinstance(task_id, str):
            raise ValueError("[ERROR] Task 'id' field is required and must be a string")

        # Validate match_mode
        match_mode = data.get("matchMode", "all")
        valid_modes = ("first", "all", "unique_per_var")
        if match_mode not in valid_modes:
            raise ValueError(
                f"[ERROR] Task '{task_id}' has invalid matchMode '{match_mode}'. "
                f"Valid options: {', '.join(valid_modes)}"
            )

        # Validate max_match
        max_match = data.get("maxMatch", 0)
        if not isinstance(max_match, int) or max_match < 0:
            raise ValueError(
                f"[ERROR] Task '{task_id}' has invalid maxMatch '{max_match}'. "
                f"Must be a non-negative integer (0 = unlimited)"
            )

        return cls(
            id=task_id,
            raw_condition=raw_condition,
            capture=capture,
            has_dep_in_condition=has_dep,
            has_pattern_in_capture=has_pattern,
            name=data.get("name"),
            scope=final_scope,
            deps=data.get("dependsOn", []),
            logging=data.get("logging"),
            match_mode=match_mode,
            max_match=max_match,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Task to dictionary"""
        result: Dict[str, Any] = {
            "capture": self.capture,
            "dependsOn": self.deps,
            "condition": self.raw_condition,
        }
        if self.id:
            result["id"] = self.id
        if self.name:
            result["name"] = self.name
        if self.scope:
            result["scope"] = self.scope
        if self.logging:
            result["logging"] = self.logging
        return result


class YamlBuilder:
    """YAML configuration loader and validator"""

    def __init__(self) -> None:
        self.line_map: Dict[str, int] = {}
        self.config: Optional[dict[str, Any]] = None
        self.output_dir: Optional[Path] = None
        self._tasks_resolved: bool = False

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration with validation"""
        self._extract_line_numbers(config_path)

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            line_info = ""
            problem_mark = getattr(e, "problem_mark", None)
            if problem_mark is not None:
                line = getattr(problem_mark, "line", None)
                if line is not None:
                    line_info = f" at line {line + 1}"
            raise ValueError(f"[ERROR] YAML syntax error{line_info}: {e}")

        self._validate_config(config, config_path)
        return config

    def _validate_config(self, config: Dict[str, Any], config_path: str) -> None:
        """Validate configuration structure"""
        if "fsdbFile" not in config:
            line_info = self._get_line_info("fsdbFile")
            raise ValueError(f"[ERROR] Missing required field 'fsdbFile' {line_info}")

        fsdb_path = Path(config["fsdbFile"])
        if not fsdb_path.exists():
            raise FileNotFoundError(f"[ERROR] FSDB file not found: {fsdb_path}")

        if not str(fsdb_path).endswith(".fsdb"):
            print(f"[WARN] Target FSDB file extension is not .fsdb: {fsdb_path}")

        if "tasks" not in config or not config["tasks"]:
            line_info = self._get_line_info("tasks")
            raise ValueError(
                f"[ERROR] Missing 'tasks' field or task list is empty {line_info}"
            )

        config.setdefault("globalClock", "clk")
        config.setdefault("scope", "")

        if "output" not in config:
            config["output"] = {}
        config["output"].setdefault("directory", "temp_reports")
        config["output"].setdefault("verbose", False)
        config["output"].setdefault("dependency_graph", "deps.png")
        config["output"].setdefault("timeout", 100)  # Default 100 time units

        # Validate timeout
        if not isinstance(config["output"]["timeout"], int) or config["output"]["timeout"] <= 0:
            raise ValueError("[ERROR] output.timeout must be a positive integer")

        self.output_dir = Path(config["output"]["directory"])
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(
                f"[ERROR] Failed to create output directory {self.output_dir}: {e}"
            )

        for idx, task in enumerate(config["tasks"], 1):
            self._validate_task(task, idx)
            # Normalize condition to single string
            if "condition" in task:
                task["condition"] = self._normalize_condition(task["condition"])
            # Normalize logging to single string
            if "logging" in task:
                task["logging"] = self._normalize_logging(task["logging"])
            # Normalize dependsOn to list
            if "dependsOn" in task:
                depends = task["dependsOn"]
                if isinstance(depends, str):
                    task["dependsOn"] = [depends]

        self._validate_dependencies(config["tasks"])
        self.config = config

    def resolve_config(
        self, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve config by converting dict tasks to Task objects

        Args:
            config: Raw config dictionary
            cond_builder: Optional ConditionBuilder instance
            all_signals: Optional list of all available signals from FSDB
        """
        self.config = config

        # Convert dict tasks to Task objects
        global_scope = config.get("scope", "")
        task_objects: List[Task] = []
        for task_dict in config.get("tasks", []):
            task_obj = Task.from_dict(task_dict, global_scope, self)
            task_objects.append(task_obj)

        config["tasks"] = task_objects
        self._tasks_resolved = True
        return config

    def collect_raw_signals(self, global_scope: str) -> List[str]:
        """Collect all raw signals-of-interest from all tasks (capture + condition)

        Signals may contain wildcard patterns like {*} which will be expanded by fsdb_builder.

        Args:
            global_scope: Global scope for signal resolution

        Returns:
            List of all raw signals (may contain {*} patterns)
        """
        if not self.config:
            raise RuntimeError("Config not loaded. Call load_config first.")

        from condition_builder import ConditionBuilder

        all_signals: set[str] = set()
        tasks: List[Union[dict[str, Any], Task]] = self.config.get("tasks", [])

        for task in tasks:
            if isinstance(task, Task):
                # Collect capture signals
                for sig in task.capture:
                    if isinstance(sig, str):
                        normalized_sig = re.sub(r'\{[^}]+\}', '{*}', sig)
                        all_signals.add(normalized_sig)

                # Collect condition signals using ConditionBuilder
                if task.raw_condition:
                    condition_str = task.raw_condition.replace("&&", " and ").replace("||", " or ")
                    condition_str = re.sub(r'\s+', ' ', condition_str).strip()
                    cond_signals = ConditionBuilder.collect_signals(condition_str, task.scope or "")
                    all_signals.update(cond_signals)
            else:
                raise RuntimeError("Tasks must be resolved to Task objects before collecting signals")

        return list(all_signals)

    def _resolve_signal_path(self, signal: str, scope: str) -> str:
        """Resolve signal path with scope support (delegates to utils)"""
        return resolve_signal_path(signal, scope)

    def _normalize_condition(self, condition: Union[str, list[str]]) -> str:
        """Normalize condition to single Python expression string"""
        if isinstance(condition, str):
            return condition.strip()
        elif isinstance(condition, list):
            return " ".join(line.strip() for line in condition if line.strip())
        return str(condition)

    def _normalize_logging(self, logging: Union[str, list[str]]) -> str:
        """Normalize logging to single format string"""
        if isinstance(logging, str):
            return logging.strip()
        elif isinstance(logging, list):
            return " ".join(line.strip() for line in logging if line.strip())
        return str(logging)

    def _validate_task(self, task: Dict[str, Any], task_num: int) -> None:
        """Validate task configuration"""
        line_info = self._get_line_info(f"tasks[{task_num - 1}]")

        def require_nonempty(value: Any, field_name: str) -> None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"[ERROR] Task {task_num} '{field_name}' must be non-empty string{line_info}"
                )

        # id is required
        if "id" not in task:
            raise ValueError(
                f"[ERROR] Task {task_num} missing required field 'id'{line_info}"
            )

        task_id = task["id"]
        require_nonempty(task_id, "id")
        if not task_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"[ERROR] Task '{task_id}' id contains invalid characters{line_info}"
            )

        task_identifier = task_id

        if "name" in task:
            require_nonempty(task["name"], "name")

        if "scope" in task:
            require_nonempty(task["scope"], "scope")

        if "dependsOn" in task:
            depends = task["dependsOn"]
            if isinstance(depends, str):
                require_nonempty(depends, "dependsOn")
            elif isinstance(depends, list):
                if not depends:
                    raise ValueError(
                        f"[ERROR] Task '{task_identifier}' 'dependsOn' list cannot be empty{line_info}"
                    )
                for dep_id in depends:
                    if not isinstance(dep_id, str) or not dep_id.strip():
                        raise ValueError(
                            f"[ERROR] Task '{task_identifier}' 'dependsOn' contains invalid dependency{line_info}"
                        )
            else:
                raise ValueError(
                    f"[ERROR] Task '{task_identifier}' 'dependsOn' must be string or list{line_info}"
                )

        if "condition" not in task:
            raise ValueError(
                f"[ERROR] Task '{task_identifier}' missing 'condition' field{line_info}"
            )

        condition = task["condition"]
        if isinstance(condition, str):
            require_nonempty(condition, "condition")
        elif isinstance(condition, list):
            if not condition:
                raise ValueError(
                    f"[ERROR] Task '{task_identifier}' 'condition' list cannot be empty{line_info}"
                )
            for line in condition:
                if not isinstance(line, str) or not line.strip():
                    raise ValueError(
                        f"[ERROR] Task '{task_identifier}' 'condition' contains invalid line{line_info}"
                    )
        else:
            raise ValueError(
                f"[ERROR] Task '{task_identifier}' 'condition' must be string or list of strings{line_info}"
            )

        capture = task.get("capture", [])
        if not capture:
            print(
                f"[WARN] Task '{task_identifier}' has no signals specified in 'capture' field{line_info}"
            )
        elif not isinstance(capture, list):
            raise ValueError(
                f"[ERROR] Task '{task_identifier}' 'capture' field must be a list{line_info}"
            )
        else:
            for idx, sig in enumerate(capture):
                if isinstance(sig, str):
                    if not sig or sig.isspace():
                        raise ValueError(
                            f"[ERROR] Task '{task_identifier}' capture[{idx}] signal is empty{line_info}"
                        )

    def _validate_dependencies(self, tasks: List[dict[str, Any]]) -> None:
        """Validate task dependency graph and detect cycles"""
        task_map: Dict[str, int] = {}
        for idx, task in enumerate(tasks):
            task_id = task.get("id")
            if not task_id:
                continue
            if task_id in task_map:
                raise ValueError(f"[ERROR] Duplicate task id '{task_id}' found")
            task_map[task_id] = idx

        for idx, task in enumerate(tasks):
            task_display_name = task.get("name") or task.get("id") or f"Task {idx + 1}"
            for dep_id in task.get("dependsOn", []):
                if dep_id not in task_map:
                    raise ValueError(
                        f"[ERROR] Task '{task_display_name}' depends on non-existent task '{dep_id}'"
                    )

        visited: set[int] = set()
        rec_stack: set[int] = set()

        def has_cycle(task_idx: int, path: List[str]) -> bool:
            visited.add(task_idx)
            rec_stack.add(task_idx)
            path.append(tasks[task_idx].get("id", f"task_{task_idx}"))

            for dep_id in tasks[task_idx].get("dependsOn", []):
                dep_idx = task_map[dep_id]
                if dep_idx not in visited:
                    if has_cycle(dep_idx, path):
                        return True
                elif dep_idx in rec_stack:
                    cycle_start = path.index(dep_id)
                    cycle = " -> ".join(path[cycle_start:] + [dep_id])
                    raise ValueError(f"[ERROR] Circular dependency detected: {cycle}")

            path.pop()
            rec_stack.remove(task_idx)
            return False

        for idx in range(len(tasks)):
            if idx not in visited:
                has_cycle(idx, [])

    def _extract_line_numbers(self, config_path: str) -> None:
        """Extract line numbers for all keys in YAML file"""
        try:
            with open(config_path, "r") as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                stripped = line.lstrip()
                if stripped and not stripped.startswith("#") and ":" in stripped:
                    key = stripped.split(":")[0].strip()
                    if key.startswith("- "):
                        continue
                    self.line_map[key] = line_num
        except Exception:
            pass

    def _get_line_info(self, key_path: str) -> str:
        """Get line information for error messages"""
        parts = key_path.replace("[", ".").replace("]", "").split(".")

        for part in reversed(parts):
            if part.isdigit():
                continue
            if part in self.line_map:
                return f" (line {self.line_map[part]})"

        return ""

    def build_exec_order(self) -> List[int]:
        """Build topologically sorted task execution order and export dependency graph"""
        if not self.config:
            raise RuntimeError("Config not loaded. Call load_config first.")

        tasks: List[Task] = self.config.get("tasks", [])

        task_map = {task.id: idx for idx, task in enumerate(tasks) if task.id}

        in_degree = [0] * len(tasks)
        adj_list = [[] for _ in range(len(tasks))]

        for idx, task in enumerate(tasks):
            for dep_id in task.deps:
                dep_idx = task_map[dep_id]
                adj_list[dep_idx].append(idx)
                in_degree[idx] += 1

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
            raise RuntimeError(
                "[ERROR] Failed to build execution order - circular dependency"
            )

        # Export dependency graph if configured
        dep_graph_file = self.config["output"].get("dependency_graph")
        if dep_graph_file and self.output_dir:
            self.export_deps_graph(dep_graph_file, self.output_dir, result)

        return result

    def export_deps_graph(
        self, output_file: str, output_dir: Path, task_execution_order: List[int]
    ) -> None:
        """Export task dependency graph to image file"""
        # Always compute output path first
        output_path = output_dir / output_file
        file_ext = output_path.suffix.lstrip(".")
        if not file_ext:
            file_ext = "png"
            output_path = output_path.with_suffix(".png")

        if not self.config:
            raise RuntimeError("Config not loaded. Call load_config first.")

        # Pure-Python NetworkX + Matplotlib implementation (no system 'dot' needed)
        self._export_deps_graph_matplotlib(output_path, task_execution_order)

    def _export_deps_graph_matplotlib(
        self, output_path: Path, task_execution_order: List[int]
    ) -> None:
        """Pure-Python fallback using NetworkX + Matplotlib.

        Produces PNG/PDF/SVG without requiring Graphviz 'dot'."""
        try:
            import matplotlib
            matplotlib.use("Agg")  # Headless backend for servers/CI
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as e:
            print(
                f"[WARN] Matplotlib/NetworkX not installed ({e}). Skip dependency graph export."
            )
            print("[WARN] Install with: pip install networkx matplotlib")
            return

        if not self.config:
            raise RuntimeError("Config not loaded. Call load_config first.")

        tasks: List[Task] = self.config["tasks"]
        G = nx.DiGraph()

        # Build nodes with labels (include execution order when available)
        for idx, task in enumerate(tasks):
            task_id = task.id or f"task_{idx}"
            task_display_name = task.name or task.id or f"Task {idx + 1}"
            if task_execution_order and idx in task_execution_order:
                exec_order = task_execution_order.index(idx) + 1
                label = f"[{exec_order}] {task_display_name}\n({task_id})"
            else:
                label = f"{task_display_name}\n({task_id})"
            G.add_node(task_id, label=label)

        # Build edges from dependencies
        for task in tasks:
            src_id = task.id or f"task_{tasks.index(task)}"
            for dep_id in task.deps:
                G.add_edge(dep_id, src_id)

        # Layout: deterministic left-to-right DAG layers without Graphviz
        if task_execution_order:
            topo_nodes = [
                tasks[i].id or f"task_{i}"
                for i in task_execution_order
                if (tasks[i].id or f"task_{i}") in G
            ]
        else:
            topo_nodes = list(nx.topological_sort(G))

        level: Dict[str, int] = {n: 0 for n in G.nodes}
        for node in topo_nodes:
            preds = list(G.predecessors(node))
            if preds:
                level[node] = max(level[p] + 1 for p in preds)

        order_rank = {node: idx for idx, node in enumerate(topo_nodes)}
        levels: Dict[int, list[str]] = {}
        for node, lvl in level.items():
            levels.setdefault(lvl, []).append(node)

        pos: Dict[str, tuple[float, float]] = {}
        x_spacing, y_spacing = 3.0, 1.6
        for lvl in sorted(levels.keys()):
            nodes = sorted(levels[lvl], key=lambda n: order_rank.get(n, 0))
            for idx, node in enumerate(nodes):
                pos[node] = (lvl * x_spacing, -idx * y_spacing)

        # Draw nodes and edges
        fig, ax = plt.subplots(figsize=(max(6, len(G.nodes) * 0.8), max(4, len(G.nodes) * 0.6)))
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=1800,
            node_color="#ADD8E6",
            node_shape="s",
            ax=ax,
        )
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=16, width=1.8, ax=ax)

        # Labels
        labels = {n: G.nodes[n].get("label", n) for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)

        ax.set_axis_off()

        # Save figure based on extension
        ext = output_path.suffix.lstrip(".").lower()
        if ext not in {"png", "pdf", "svg"}:
            # Default to PNG if unsupported extension
            output_path = output_path.with_suffix(".png")

        try:
            plt.tight_layout()
            fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] Dependency graph exported (matplotlib) to: {output_path}")
        except Exception as e:
            plt.close(fig)
            print(f"[WARN] Failed to export dependency graph via matplotlib: {e}")

    def format_log(
        self,
        log_format: str,
        row_data: Dict[str, Any],
        capture_signals: List[str],
        row_idx: int,
    ) -> str:
        """Format log message for matched row

        Args:
            log_format: Format string for the log message
            row_data: Row data containing signal values
            capture_signals: List of captured signal names
            row_idx: Current row index (time)

        Returns:
            Formatted log message string
        """
        context: Dict[str, Any] = {"__time__": row_idx}
        for sig in capture_signals:
            sig_name = sig.split(".")[-1].split("/")[-1]
            val_str = row_data["capd"].get(sig, "0")
            if val_str in ["x", "z", "X", "Z"] or val_str.startswith("*"):
                context[sig_name] = "0x0"
            else:
                if not val_str.startswith("0x") and not val_str.startswith("0X"):
                    val_str = "0x" + val_str
                context[sig_name] = val_str

        try:
            return eval(f'f"""{log_format}"""', {}, context)
        except Exception as e:
            return f"[LOG ERROR] Failed to format log: {e}"

    def resolve_capture_signals(
        self, capture_signals: List[Any], scope: str
    ) -> List[str]:
        """Resolve capture signal paths with scope support

        Args:
            capture_signals: List of signal names to capture
            scope: Resolved scope for this task

        Returns:
            List of resolved signal paths (patterns preserved)
        """
        resolved_signals: List[str] = []
        for sig in capture_signals:
            if isinstance(sig, str):
                # Always resolve with scope, even if it contains {var} pattern
                resolved_sig = self._resolve_signal_path(sig, scope)
                resolved_signals.append(resolved_sig)
            else:
                resolved_signals.append(str(sig))
        return resolved_signals

    def resolve_dep_references(
        self, value: Any, task_id: str, task_data: Dict[str, Any]
    ) -> Any:
        """Resolve $dep.task_id.signal references to actual values

        Args:
            value: Value to resolve (can be str, list, dict, or other)
            task_id: Current task ID
            task_data: Dictionary of task execution data

        Returns:
            Resolved value with $dep references replaced
        """
        if isinstance(value, str):
            if value.startswith("$dep."):
                parts = value.split(".", 2)
                if len(parts) != 3:
                    raise ValueError(
                        f"[ERROR] Invalid $dep reference: {value}. Format: $dep.task_id.var_name"
                    )

                _, dep_task_id, var_name = parts

                if dep_task_id not in task_data:
                    raise ValueError(
                        f"[ERROR] Task '{task_id}' references non-existent or not-yet-executed task '{dep_task_id}'"
                    )

                task_exports = task_data[dep_task_id]
                if var_name not in task_exports:
                    available = ", ".join(task_exports.keys())
                    raise ValueError(
                        f"[ERROR] Task '{task_id}' references non-existent variable '{var_name}' "
                        f"from task '{dep_task_id}'. Available: {available}"
                    )

                return task_exports[var_name]
            return value
        elif isinstance(value, list):
            return [
                self.resolve_dep_references(item, task_id, task_data) for item in value
            ]
        elif isinstance(value, dict):
            return {
                k: self.resolve_dep_references(v, task_id, task_data)
                for k, v in value.items()
            }
        else:
            return value
