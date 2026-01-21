#!/usr/bin/env python3
"""YAML configuration validator for AgenticPipeViewer

This module handles validation of YAML configuration structure, including:
- Required fields validation
- Type checking
- Dependency graph validation (circular dependency detection)
- Line number tracking for error reporting
"""

from pathlib import Path
from typing import Any, Dict, List, Set


class YamlValidator:
    """Validator for YAML configuration structure and semantics"""

    def __init__(self) -> None:
        self.line_map: Dict[str, int] = {}

    def extract_line_numbers(self, config_path: str) -> None:
        """Extract line numbers for all keys in YAML file for error reporting

        Args:
            config_path: Path to YAML configuration file
        """
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

    def get_line_info(self, key_path: str) -> str:
        """Get line information for error messages

        Args:
            key_path: Dot-separated key path (e.g., "tasks[0].condition")

        Returns:
            Formatted line info string like " (line 42)" or empty string if not found
        """
        parts = key_path.replace("[", ".").replace("]", "").split(".")

        for part in reversed(parts):
            if part.isdigit():
                continue
            if part in self.line_map:
                return f" (line {self.line_map[part]})"

        return ""

    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate complete configuration structure

        Args:
            config: Parsed YAML configuration dictionary

        Raises:
            ValueError: If validation fails
            FileNotFoundError: If FSDB file not found
        """
        self._validate_fsdb_file(config)
        self._validate_tasks_exist(config)
        self._set_defaults(config)
        self._validate_global_flush(config)
        self._validate_output_config(config)
        self._validate_all_tasks(config)
        self._validate_dependency_graph(config["tasks"])

    def _validate_fsdb_file(self, config: Dict[str, Any]) -> None:
        """Validate fsdbFile field"""
        if "fsdbFile" not in config:
            line_info = self.get_line_info("fsdbFile")
            raise ValueError(f"[ERROR] Missing required field 'fsdbFile' {line_info}")

        fsdb_path = Path(config["fsdbFile"])
        if not fsdb_path.exists():
            raise FileNotFoundError(f"[ERROR] FSDB file not found: {fsdb_path}")

        if not str(fsdb_path).endswith(".fsdb"):
            print(f"[WARN] Target FSDB file extension is not .fsdb: {fsdb_path}")

    def _validate_tasks_exist(self, config: Dict[str, Any]) -> None:
        """Validate tasks field exists and is non-empty"""
        if "tasks" not in config or not config["tasks"]:
            line_info = self.get_line_info("tasks")
            raise ValueError(
                f"[ERROR] Missing 'tasks' field or task list is empty {line_info}"
            )

    def _set_defaults(self, config: Dict[str, Any]) -> None:
        """Set default values for optional fields"""
        config.setdefault("globalClock", "clk")
        config.setdefault("scope", "")

        if "output" not in config:
            config["output"] = {}
        config["output"].setdefault("directory", "temp_reports")
        config["output"].setdefault("verbose", False)
        config["output"].setdefault("dependency_graph", "deps.png")
        config["output"].setdefault("timeout", 100)

    def _validate_global_flush(self, config: Dict[str, Any]) -> None:
        """Validate globalFlush configuration (optional)"""
        if "globalFlush" not in config:
            return

        flush_config = config["globalFlush"]
        if not isinstance(flush_config, dict):
            raise ValueError("[ERROR] globalFlush must be a dict")

        if "condition" not in flush_config:
            raise ValueError("[ERROR] globalFlush.condition is required")

        if not isinstance(flush_config["condition"], list):
            raise ValueError("[ERROR] globalFlush.condition must be a list")

    def _validate_output_config(self, config: Dict[str, Any]) -> None:
        """Validate output configuration and create output directory"""
        # Validate timeout
        if not isinstance(config["output"]["timeout"], int) or config["output"]["timeout"] <= 0:
            raise ValueError("[ERROR] output.timeout must be a positive integer")

        # Create output directory
        output_dir = Path(config["output"]["directory"])
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(
                f"[ERROR] Failed to create output directory {output_dir}: {e}"
            )

    def _validate_all_tasks(self, config: Dict[str, Any]) -> None:
        """Validate all tasks and normalize their fields"""
        for idx, task in enumerate(config["tasks"], 1):
            self._validate_task(task, idx)
            self._normalize_task_fields(task)

    def _normalize_task_fields(self, task: Dict[str, Any]) -> None:
        """Normalize task fields to standard format

        - Converts condition and logging to single string
        - Converts dependsOn to list
        """
        # Normalize condition and logging to single string
        if "condition" in task:
            task["condition"] = self._normalize_value(task["condition"])
        if "logging" in task:
            task["logging"] = self._normalize_value(task["logging"])

        # Normalize dependsOn to list
        if "dependsOn" in task:
            depends = task["dependsOn"]
            if isinstance(depends, str):
                task["dependsOn"] = [depends]

    @staticmethod
    def _normalize_value(value: Any) -> str:
        """Normalize string or list of strings to single space-joined string

        Args:
            value: Value to normalize (str, list, or other)

        Returns:
            Normalized string
        """
        if isinstance(value, str):
            return value.strip()
        elif isinstance(value, list):
            return " ".join(line.strip() for line in value if line.strip())
        return str(value)

    def _validate_task(self, task: Dict[str, Any], task_num: int) -> None:
        """Validate single task configuration

        Args:
            task: Task dictionary
            task_num: Task number (1-based) for error messages

        Raises:
            ValueError: If validation fails
        """
        line_info = self.get_line_info(f"tasks[{task_num - 1}]")

        def require_nonempty(value: Any, field_name: str) -> None:
            """Helper to check field is non-empty string"""
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"[ERROR] Task {task_num} '{field_name}' must be non-empty string{line_info}"
                )

        # Validate required fields
        self._validate_task_id(task, task_num, line_info, require_nonempty)

        task_id = task["id"]

        # Validate optional fields
        if "name" in task:
            require_nonempty(task["name"], "name")

        if "scope" in task:
            require_nonempty(task["scope"], "scope")

        self._validate_task_depends_on(task, task_id, line_info, require_nonempty)
        self._validate_task_condition(task, task_id, line_info, require_nonempty)
        self._validate_task_capture(task, task_id, line_info)
        self._validate_task_match_mode(task, task_id, line_info)
        self._validate_task_max_match(task, task_id, line_info)

    def _validate_task_id(
        self, task: Dict[str, Any], task_num: int, line_info: str, require_nonempty
    ) -> None:
        """Validate task id field"""
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

    def _validate_task_depends_on(
        self, task: Dict[str, Any], task_id: str, line_info: str, require_nonempty
    ) -> None:
        """Validate task dependsOn field"""
        if "dependsOn" not in task:
            return

        depends = task["dependsOn"]
        if isinstance(depends, str):
            require_nonempty(depends, "dependsOn")
        elif isinstance(depends, list):
            if not depends:
                raise ValueError(
                    f"[ERROR] Task '{task_id}' 'dependsOn' list cannot be empty{line_info}"
                )
            for dep_id in depends:
                if not isinstance(dep_id, str) or not dep_id.strip():
                    raise ValueError(
                        f"[ERROR] Task '{task_id}' 'dependsOn' contains invalid dependency{line_info}"
                    )
        else:
            raise ValueError(
                f"[ERROR] Task '{task_id}' 'dependsOn' must be string or list{line_info}"
            )

    def _validate_task_condition(
        self, task: Dict[str, Any], task_id: str, line_info: str, require_nonempty
    ) -> None:
        """Validate task condition field"""
        if "condition" not in task:
            raise ValueError(
                f"[ERROR] Task '{task_id}' missing 'condition' field{line_info}"
            )

        condition = task["condition"]
        if isinstance(condition, str):
            require_nonempty(condition, "condition")
        elif isinstance(condition, list):
            if not condition:
                raise ValueError(
                    f"[ERROR] Task '{task_id}' 'condition' list cannot be empty{line_info}"
                )
            for line in condition:
                if not isinstance(line, str) or not line.strip():
                    raise ValueError(
                        f"[ERROR] Task '{task_id}' 'condition' contains invalid line{line_info}"
                    )
        else:
            raise ValueError(
                f"[ERROR] Task '{task_id}' 'condition' must be string or list of strings{line_info}"
            )

    def _validate_task_capture(
        self, task: Dict[str, Any], task_id: str, line_info: str
    ) -> None:
        """Validate task capture field"""
        capture = task.get("capture", [])
        if not capture:
            print(
                f"[WARN] Task '{task_id}' has no signals specified in 'capture' field{line_info}"
            )
        elif not isinstance(capture, list):
            raise ValueError(
                f"[ERROR] Task '{task_id}' 'capture' field must be a list{line_info}"
            )
        else:
            for idx, sig in enumerate(capture):
                if isinstance(sig, str):
                    if not sig or sig.isspace():
                        raise ValueError(
                            f"[ERROR] Task '{task_id}' capture[{idx}] signal is empty{line_info}"
                        )

    def _validate_task_match_mode(
        self, task: Dict[str, Any], task_id: str, line_info: str
    ) -> None:
        """Validate task matchMode field"""
        if "matchMode" not in task:
            return

        match_mode = task["matchMode"]
        valid_modes = ("first", "all", "unique_per_var")
        if match_mode not in valid_modes:
            raise ValueError(
                f"[ERROR] Task '{task_id}' has invalid matchMode '{match_mode}'{line_info}. "
                f"Valid options: {', '.join(valid_modes)}"
            )

    def _validate_task_max_match(
        self, task: Dict[str, Any], task_id: str, line_info: str
    ) -> None:
        """Validate task maxMatch field"""
        if "maxMatch" not in task:
            return

        max_match = task["maxMatch"]
        if not isinstance(max_match, int) or max_match < 0:
            raise ValueError(
                f"[ERROR] Task '{task_id}' has invalid maxMatch '{max_match}'{line_info}. "
                f"Must be a non-negative integer (0 = unlimited)"
            )

    def _validate_dependency_graph(self, tasks: List[Dict[str, Any]]) -> None:
        """Validate task dependency graph and detect cycles

        Args:
            tasks: List of task dictionaries

        Raises:
            ValueError: If duplicate task IDs found, non-existent dependencies referenced,
                       or circular dependencies detected
        """
        # Build task ID map and check for duplicates
        task_map: Dict[str, int] = {}
        for idx, task in enumerate(tasks):
            task_id = task.get("id")
            if not task_id:
                continue
            if task_id in task_map:
                raise ValueError(f"[ERROR] Duplicate task id '{task_id}' found")
            task_map[task_id] = idx

        # Validate all dependencies exist
        for idx, task in enumerate(tasks):
            task_display_name = task.get("name") or task.get("id") or f"Task {idx + 1}"
            for dep_id in task.get("dependsOn", []):
                if dep_id not in task_map:
                    raise ValueError(
                        f"[ERROR] Task '{task_display_name}' depends on non-existent task '{dep_id}'"
                    )

        # Detect circular dependencies using DFS
        visited: Set[int] = set()
        rec_stack: Set[int] = set()

        def has_cycle(task_idx: int, path: List[str]) -> bool:
            """DFS helper to detect cycles in dependency graph

            Args:
                task_idx: Current task index
                path: Current path of task IDs

            Returns:
                True if cycle detected (raises ValueError with cycle path)
            """
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
