#!/usr/bin/env python3
import yaml
import re
from pathlib import Path
from typing import Union, Any, List, Optional, Dict, Tuple, Set
from dataclasses import dataclass, field
from .utils import resolve_signal_path
from .cond_builder import Condition, ConditionBuilder
from .yaml_validator import YamlValidator
from .fsdb_builder import FsdbBuilder

@dataclass
class Task:
    """Task configuration data structure"""

    id: str
    capture: List[str]
    raw_condition: str

    name: Optional[str] = None
    scope: Optional[str] = None
    deps: List[str] = field(default_factory=list)
    logging: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[Condition] = None  # Built Condition object, set later
    match_mode: str = "all"  # Matching mode: "first", "all", "unique_per_var"
    max_match: int = 0  # Maximum matches per upstream trigger (0 = unlimited)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], global_scope: str) -> "Task":
        """Create Task from dictionary with all preprocessing"""
        task_scope = data.get("scope", "")
        # Calculate final scope: task scope overrides global scope
        final_scope = task_scope if task_scope else global_scope

        raw_capture = data.get("capture", [])
        raw_condition = data.get("condition", "")

        # Resolve capture signals with scope
        capture = [resolve_signal_path(sig, final_scope) for sig in raw_capture]

        return cls(
            id=data.get("id",""),
            raw_condition=raw_condition,
            capture=capture,
            name=data.get("name"),
            scope=final_scope,
            deps=data.get("dependsOn", []),
            logging=data.get("logging"),
            match_mode=data.get("matchMode", "all"),
            max_match=data.get("maxMatch", 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Task to dictionary"""
        result: Dict[str, Any] = {
            "id": self.id,
            "capture": self.capture,
            "dependsOn": self.deps,
            "condition": self.raw_condition,
        }
        if self.name:
            result["name"] = self.name
        if self.scope:
            result["scope"] = self.scope
        if self.logging:
            result["logging"] = self.logging
        return result


class YamlBuilder:
    """YAML configuration loader and resolver"""
    config: Dict[str, Any]

    def __init__(self) -> None:
        self.output_dir: Optional[Path] = None

        self._validator = YamlValidator()
        self._tasks_resolved: bool = False
        self._fsdb_builder: Optional[FsdbBuilder] = None

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration with validation"""
        self._validator.extract_line_numbers(config_path)

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

        self._validator.validate_config(config)
        self.output_dir = Path(config["output"]["directory"])
        self.config = config

        # Extract FSDB parameters and create builder
        fsdb_file = Path(config["fsdbFile"])
        output_dir = Path(config["output"]["directory"])
        verbose = config["output"]["verbose"]

        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize FSDB builder
        self._fsdb_builder = FsdbBuilder(fsdb_file, output_dir, verbose)

        return config

    def get_fsdb_builder(self) -> FsdbBuilder:
        """Get the FsdbBuilder instance

        Returns:
            FsdbBuilder instance created during config loading

        Raises:
            RuntimeError: If called before load_config()
        """
        if self._fsdb_builder is None:
            raise RuntimeError(
                "[ERROR] FsdbBuilder not initialized. Call load_config() first."
            )
        return self._fsdb_builder

    @property
    def fsdb_builder(self) -> FsdbBuilder:
        """Convenience property to access FsdbBuilder"""
        return self.get_fsdb_builder()

    def resolve_config(
        self,
        cond_builder: Optional[ConditionBuilder] = None,
        dump_signals: bool = True
    ) -> Tuple[Dict[str, Any], Optional[Condition]]:
        """Resolve config by converting dict tasks to Task objects and building conditions

        Args:
            cond_builder: ConditionBuilder instance for compiling conditions
            dump_signals: If True, collect SOI and dump signals from FSDB (default: True)
                         Set to False for deps-only mode

        Returns:
            Tuple of (resolved config dictionary, globalFlush condition if exists)
        """
        config = self.config
        global_scope = config.get("scope", "")

        # Convert dict tasks to Task objects
        task_objects: List[Task] = []
        for task_dict in config.get("tasks", []):
            task_obj = Task.from_dict(task_dict, global_scope)
            task_objects.append(task_obj)

        config["tasks"] = task_objects
        self._tasks_resolved = True

        # Collect and dump signals if requested
        if dump_signals and self._fsdb_builder:
            soi = self.collect_raw_signals(global_scope)
            self._fsdb_builder.dump_signals(soi)

        # Build conditions if cond_builder provided
        gflush_condition: Optional[Condition] = None
        if cond_builder:
            # Define pattern resolver callback
            def pattern_resolver(pattern: str) -> Tuple[List[str], List[str]]:
                if self._fsdb_builder:
                    return self._fsdb_builder.resolve_pattern(pattern, global_scope)
                else:
                    # In deps-only mode, provide a dummy expansion
                    from .utils import resolve_signal_path
                    import re
                    resolved = resolve_signal_path(pattern, global_scope)
                    wildcard = re.sub(r'\{[^}]+\}', '{*}', resolved)
                    return [wildcard], []

            # Build all task conditions
            for task in task_objects:
                if task.condition is None:
                    task.condition = cond_builder.build(task, pattern_resolver)

            # Build globalFlush condition
            if "globalFlush" in config:
                flush_config = config["globalFlush"]
                gflush_condition = cond_builder.build_raw(
                    raw_condition=flush_config["condition"],
                    scope=global_scope,
                    pattern_resolver=pattern_resolver
                )

        return config, gflush_condition

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

        all_signals: Set[str] = set()
        tasks: List[Union[Dict[str, Any], Task]] = self.config.get("tasks", [])

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

        # Collect globalFlush condition signals
        if "globalFlush" in self.config:
            flush_condition = self._normalize(self.config["globalFlush"]["condition"])
            flush_signals = ConditionBuilder.collect_signals(
                flush_condition,
                self.config.get("scope", "")
            )
            all_signals.update(flush_signals)

        return list(all_signals)

    def _normalize(self, value: Union[str, List[str]]) -> str:
        """Normalize string or list of strings to single space-joined string"""
        if isinstance(value, str):
            return value.strip()
        elif isinstance(value, list):
            return " ".join(line.strip() for line in value if line.strip())
        return str(value)

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
        levels: Dict[int, List[str]] = {}
        for node, lvl in level.items():
            levels.setdefault(lvl, []).append(node)

        pos: Dict[str, Tuple[float, float]] = {}
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

        Supports format specifiers for base conversion:
            {signal:x} - hex format (lowercase)
            {signal:X} - hex format (uppercase)
            {signal:b} - binary format
            {signal:d} - decimal format
            {signal:o} - octal format

        Signal names in log format can use full paths with dots and brackets:
            {x_ct_ifu_top.ifu_idu_ib_inst0_data[31:0]}
            {x_ct_ifu_top.ifu_idu_ib_inst0_data[31:0]:x}
        """
        from .utils import Signal

        context: Dict[str, Any] = {"__time__": row_idx}

        # Find all placeholders with format specifiers in the log format
        # Pattern matches {name} or {name:spec} where name can include [msb:lsb] bit ranges
        # Pattern explanation:
        # - Signal ref can contain: normal chars, dots, or [msb:lsb] bit ranges
        # - Format spec is optional :x/:X/:b/:d/:o at the end
        placeholder_pattern = re.compile(r'\{((?:[^}:\[\]]+|\[\d+:\d+\])+)(?::([xXbdo]))?\}')

        # Collect all signal references from log_format and their format specs
        # Key: normalized signal name (last component without bit range)
        # Value: (original_ref, needs_int_conversion)
        signal_refs: Dict[str, Tuple[str, bool]] = {}

        for match in placeholder_pattern.finditer(log_format):
            signal_ref = match.group(1)
            format_spec = match.group(2)

            if signal_ref == "__time__":
                continue

            # Get the last component and normalize it
            last_component = signal_ref.split(".")[-1].split("/")[-1]
            normalized = Signal.normalize(last_component)
            needs_int = format_spec in ("x", "X", "b", "d", "o")

            # If same signal appears with different format specs, prefer int conversion
            if normalized in signal_refs:
                _, existing_needs_int = signal_refs[normalized]
                needs_int = needs_int or existing_needs_int

            signal_refs[normalized] = (signal_ref, needs_int)

        # Build context using normalized signal names
        for sig in capture_signals:
            last_component = sig.split(".")[-1].split("/")[-1]
            normalized = Signal.normalize(last_component)

            val_str = row_data["capd"].get(sig, "0")

            # Handle undefined values
            if val_str in ["x", "z", "X", "Z"] or val_str.startswith("*"):
                val_str = "0"

            # Check if this signal needs int conversion
            needs_int = signal_refs.get(normalized, (None, False))[1]

            if needs_int:
                # Parse hex string to integer
                if val_str.startswith("0x") or val_str.startswith("0X"):
                    context[normalized] = int(val_str, 16)
                else:
                    context[normalized] = int(val_str, 16) if val_str else 0
            else:
                # Keep as string with 0x prefix for default display
                if not val_str.startswith("0x") and not val_str.startswith("0X"):
                    val_str = "0x" + val_str
                context[normalized] = val_str

        # Replace signal references in log_format with normalized names
        def replace_placeholder(match: re.Match) -> str:
            signal_ref = match.group(1)
            format_spec = match.group(2)

            if signal_ref == "__time__":
                return match.group(0)

            last_component = signal_ref.split(".")[-1].split("/")[-1]
            normalized = Signal.normalize(last_component)

            if format_spec:
                return "{" + normalized + ":" + format_spec + "}"
            else:
                return "{" + normalized + "}"

        safe_log_format = placeholder_pattern.sub(replace_placeholder, log_format)

        try:
            return eval(f'f"""{safe_log_format}"""', {}, context)
        except Exception as e:
            return f"[LOG ERROR] Failed to format log: {e}"

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
