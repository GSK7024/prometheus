"""
Data models for Prometheus AI Orchestrator
Contains enums, dataclasses, and Pydantic models
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from prometheus_modules.utils import normalize_path

logger = logging.getLogger(__name__)


class TaskType(Enum):
    SCAFFOLD = "SCAFFOLD"
    BLUEPRINT_FILE = "BLUEPRINT_FILE"
    TDD_IMPLEMENTATION = "TDD_IMPLEMENTATION"
    CODE_MODIFICATION = "CODE_MODIFICATION"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    ENVIRONMENT_CHECK = "ENVIRONMENT_CHECK"
    FILE_SYSTEM_REFACTORING = "FILE_SYSTEM_REFACTORING"
    DEPLOYMENT = "DEPLOYMENT"
    SECURITY_SCAN = "SECURITY_SCAN"
    PERFORMANCE_OPTIMIZATION = "PERFORMANCE_OPTIMIZATION"
    ML_MODEL_TRAINING = "ML_MODEL_TRAINING"
    DATA_PROCESSING = "DATA_PROCESSING"
    COGNITIVE_ANALYSIS = "COGNITIVE_ANALYSIS"
    SELF_EVOLUTION = "SELF_EVOLUTION"
    PLAN_EPICS = "PLAN_EPICS"
    SETUP_CI_PIPELINE = "SETUP_CI_PIPELINE"
    CREATE_DOCKERFILE = "CREATE_DOCKERFILE"
    USER_STORY_REFINEMENT = "USER_STORY_REFINEMENT"
    SPRINT_REVIEW = "SPRINT_REVIEW"
    GIT_COMMIT = "GIT_COMMIT"


class TaskStatus(Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DevelopmentStrategy(Enum):
    TDD = "tdd"
    AGILE = "agile"
    DEVOPS = "devops"
    COGNITIVE = "cognitive"


class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MODEL_3D = "3d_model"
    SENSOR_DATA = "sensor_data"


class CognitiveState(Enum):
    FOCUSED = "focused"
    EXPLORATORY = "exploratory"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    METACOGNITIVE = "metacognitive"
    EVOLUTIONARY = "evolutionary"  # New state


@dataclass
class Evidence:
    url: str
    title: str
    snippet: str
    hash: str
    score: float = 0.0


@dataclass
class PlanTask:
    id: str
    description: str
    rationale: str = ""
    dependencies: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    owner_persona: str = ""
    estimate_hours: float = 0.0


@dataclass
class PlanNode:
    id: str
    title: str
    objective: str
    tasks: List[PlanTask] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)


@dataclass
class PlanGraph:
    goal: str
    nodes: Dict[str, PlanNode] = field(default_factory=dict)
    root_id: str = "root"

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, indent=2)


@dataclass
class MethodBlueprint:
    method_name: str
    description: str
    parameters: List[Dict[str, str]] = field(default_factory=list)
    return_type: str = "Any"
    complexity: str = "O(1)"  # Time complexity
    exceptions: List[str] = field(default_factory=list)
    cognitive_complexity: float = 1.0  # 1.0 to 5.0 scale
    test_cases: List[str] = field(default_factory=list)  # New for TDD

    @classmethod
    def from_dict(cls, data: dict) -> Optional["MethodBlueprint"]:
        if (
            not isinstance(data, dict)
            or "method_name" not in data
            or "description" not in data
        ):
            return None
        known_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known_fields}
        return cls(**kwargs)


@dataclass
class ClassBlueprint:
    class_name: str
    description: str
    methods: List[MethodBlueprint] = field(default_factory=list)
    attributes: List[Dict[str, str]] = field(default_factory=list)
    inheritance: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    design_pattern: str = ""
    cognitive_load: float = 1.0  # 1.0 to 10.0 scale
    test_suite: str = ""  # New for TDD blueprint

    @classmethod
    def from_dict(cls, data: dict) -> Optional["ClassBlueprint"]:
        if (
            not isinstance(data, dict)
            or "class_name" not in data
            or "description" not in data
        ):
            return None

        method_blueprints = []
        if isinstance(data.get("methods"), list):
            for method_data in data.get("methods", []):
                method_obj = MethodBlueprint.from_dict(method_data)
                if method_obj:
                    method_blueprints.append(method_obj)

        known_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known_fields}
        kwargs["methods"] = method_blueprints
        return cls(**kwargs)


@dataclass
class FileBlueprint:
    filename: str
    description: str
    classes: List[ClassBlueprint] = field(default_factory=list)
    functions: List[MethodBlueprint] = field(default_factory=list)
    language: str = "python"
    framework: str = ""
    dependencies: List[str] = field(default_factory=list)
    cognitive_cohesion: float = 1.0  # 1.0 to 5.0 scale
    test_file: str = ""  # Paired test file for TDD

    @classmethod
    def from_dict(cls, data: dict) -> Optional["FileBlueprint"]:
        if (
            not isinstance(data, dict)
            or "filename" not in data
            or "description" not in data
        ):
            return None

        class_blueprints = []
        if isinstance(data.get("classes"), list):
            for class_data in data.get("classes", []):
                class_obj = ClassBlueprint.from_dict(class_data)
                if class_obj:
                    class_blueprints.append(class_obj)

        function_blueprints = []
        if isinstance(data.get("functions"), list):
            for func_data in data.get("functions", []):
                func_obj = MethodBlueprint.from_dict(func_data)
                if func_obj:
                    function_blueprints.append(func_obj)

        known_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known_fields}
        kwargs["classes"] = class_blueprints
        kwargs["functions"] = function_blueprints
        return cls(**kwargs)


@dataclass
class LivingBlueprint:
    root: List[FileBlueprint] = field(default_factory=list)
    architecture: str = "monolithic"
    deployment_target: str = "local"
    version: str = "1.0.0"
    cognitive_architecture: str = "standard"
    evolutionary_path: List[Dict] = field(default_factory=list)
    agile_backlog: List[Dict] = field(default_factory=list)  # New for Agile
    ci_pipeline: Dict = field(default_factory=dict)  # New for DevOps

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LivingBlueprint":
        """Safely and recursively constructs a LivingBlueprint from a dictionary, ignoring unknown fields."""
        if not isinstance(data, dict):
            logger.warning(
                f"Invalid blueprint data format. Expected a dict. Got: {type(data)}"
            )
            # Attempt to repair if the data is a list of files
            if isinstance(data, list):
                logger.info("Attempting to repair blueprint by wrapping list in root.")
                data = {"root": data}
            else:
                return cls()

        root_files = []
        if isinstance(data.get("root"), list):
            for file_data in data.get("root", []):
                file_obj = FileBlueprint.from_dict(file_data)
                if file_obj:
                    root_files.append(file_obj)
        else:
            logger.warning(
                f"Blueprint is missing 'root' list key. Data: {list(data.keys())}"
            )

        known_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known_fields}
        kwargs["root"] = root_files
        return cls(**kwargs)

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, indent=2)

    def get_file(self, filename: str) -> Optional[FileBlueprint]:
        normalized_filename = normalize_path(filename)
        for f in self.root:
            if normalize_path(f.filename) == normalized_filename:
                return f
        return None

    def add_or_update_file(self, file_blueprint: FileBlueprint):
        """Adds or updates a file blueprint in the root list."""
        filename = normalize_path(file_blueprint.filename)
        for i, f in enumerate(self.root):
            if normalize_path(f.filename) == filename:
                self.root[i] = file_blueprint
                return
        self.root.append(file_blueprint)


@dataclass
class LanguageContext:
    language_name: str
    file_extension: str
    test_command: str
    prerequisites: List[str] = field(default_factory=list)
    linter_command: Optional[str] = None
    package_manager: Optional[str] = None
    build_command: Optional[str] = None
    run_command: Optional[str] = None
    debug_command: Optional[str] = None
    cognitive_complexity: float = 1.0  # 1.0 to 5.0 scale
    coverage_tool: str = "coverage"  # New for TDD


@dataclass
class TDDFilePair:
    test_file_path: str
    test_file_code: str
    source_file_path: str
    source_file_code: str
    test_coverage: float = 0.0
    performance_benchmark: Optional[Dict[str, float]] = None
    cognitive_coverage: float = 0.0  # How well it covers cognitive aspects
    failing_tests: List[str] = field(default_factory=list)  # New for failure tracking


@dataclass
class FinalizationReport:
    readme_content: str
    requirements_content: str
    deployment_guide: str
    api_documentation: str
    troubleshooting_guide: str
    cognitive_architecture: str  # Documentation of the cognitive architecture used
    agile_retrospective: str = ""  # New for Agile
    devops_metrics: Dict = field(default_factory=dict)  # New for DevOps


@dataclass
class ProjectState:
    project_name: str
    goal: str
    current_blueprint: LivingBlueprint
    task_queue: List[Dict] = field(default_factory=list)
    completed_tasks: List[Dict] = field(default_factory=list)
    failed_tasks: List[Dict] = field(default_factory=list)
    cognitive_state: Dict = field(default_factory=dict)
    performance_metrics: Dict = field(default_factory=dict)
    session_data: Dict = field(default_factory=dict)
    agile_board: Dict = field(default_factory=dict)  # New for Agile
    devops_status: Dict = field(default_factory=dict)  # New for DevOps