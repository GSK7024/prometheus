"""
Configuration module for Prometheus AI Orchestrator
Contains Config class and related configuration utilities
"""

import os
import sys

# Try to import cryptography, fall back gracefully
try:
    from cryptography.fernet import Fernet
except ImportError:
    # Create a mock Fernet class for environments without cryptography
    class Fernet:
        @staticmethod
        def generate_key():
            return b'32-byte-secret-key-for-testing-only'

        def __init__(self, key):
            self.key = key

        def encrypt(self, data):
            return data

        def decrypt(self, data):
            return data


class Config:
    """Enhanced configuration with cognitive parameters"""

    def __init__(self):
        # Existing configuration
        self.env = os.getenv("ENVIRONMENT", "development")
        self.max_retries = 10 if self.env == "production" else 5
        self.backoff_factor = 2
        self.max_api_calls_per_minute = 100 if self.env == "production" else 60
        self.request_timeout = 900.0  # Increased timeout for large generation calls
        self.metamorphosis_threshold = 3
        self.max_debug_attempts = 5
        self.sandbox_base_path = "./projects"
        self.memory_dir = ".forge_memory"
        self.session_dir = ".forge_sessions"
        self.cache_dir = ".forge_cache"
        self.max_file_size_mb = 10
        self.max_project_size_gb = 5
        self.max_memory_usage_percent = 95
        self.max_cpu_usage_percent = 80

        # Cognitive parameters
        self.cognitive_dim = 768
        self.quantum_qubits = 4
        self.neuroplasticity_rate = 0.1
        self.memory_capacity = 10000
        self.attention_heads = 8

        # Advanced language support
        self.language_configs = {
            "python": {
                "file_extension": ".py",
                "test_command": "{python} -m pytest -xvs",
                "linter_command": "ruff check . --fix && ruff format . && mypy . --ignore-missing-imports",
                "package_manager": "pip",
                "prerequisites": ["python", "pip"],
                "framework_configs": {
                    "django": {"test_command": "python manage.py test"},
                    "flask": {"test_command": "pytest"},
                    "fastapi": {"test_command": "pytest"},
                },
            },
            "javascript": {
                "file_extension": ".js",
                "test_command": "npm test",
                "linter_command": "eslint . --fix && npx prettier --write .",
                "package_manager": "npm",
                "prerequisites": ["node", "npm"],
                "framework_configs": {
                    "react": {"test_command": "npm test -- --watchAll=false"},
                    "vue": {"test_command": "npm run test:unit"},
                    "angular": {"test_command": "ng test"},
                },
            },
            "rust": {
                "file_extension": ".rs",
                "test_command": "cargo test",
                "linter_command": "cargo clippy --fix",
                "package_manager": "cargo",
                "prerequisites": ["rustc", "cargo"],
                "framework_configs": {
                    "actix": {"test_command": "cargo test"},
                    "rocket": {"test_command": "cargo test"},
                },
            },
            "go": {
                "file_extension": ".go",
                "test_command": "go test ./...",
                "linter_command": "gofmt -w . && go vet ./...",
                "package_manager": "go",
                "prerequisites": ["go"],
                "framework_configs": {
                    "gin": {"test_command": "go test ./..."},
                    "echo": {"test_command": "go test ./..."},
                },
            },
            "java": {
                "file_extension": ".java",
                "test_command": "mvn test",
                "linter_command": "mvn checkstyle:checkstyle",
                "package_manager": "maven",
                "prerequisites": ["java", "maven"],
                "framework_configs": {
                    "spring": {"test_command": "mvn test"},
                    "quarkus": {"test_command": "mvn test"},
                },
            },
        }

        # Multi-modal capabilities
        self.supported_modalities = [
            "text",
            "image",
            "audio",
            "video",
            "3d_model",
            "sensor_data",
        ]
        self.max_image_size = (3840, 2160)  # 4K
        self.max_audio_duration = 600  # seconds
        self.max_video_duration = 300  # seconds

        # Cloud and deployment configurations
        self.cloud_providers = [
            "aws",
            "azure",
            "gcp",
            "digitalocean",
            "kubernetes",
            "edge",
        ]
        self.deployment_strategies = [
            "blue-green",
            "canary",
            "rolling",
            "recreate",
            "ai-optimized",
        ]

        # Security configurations
        self.encryption_key = os.getenv(
            "ENCRYPTION_KEY", Fernet.generate_key().decode()
        )
        self.ssl_verification = True if self.env == "production" else False

        # Performance monitoring
        self.monitoring_interval = 10  # seconds
        self.performance_thresholds = {
            "cpu": 80,
            "memory": 95,
            "disk": 90,
            "network_latency": 5000,  # ms
            "gpu": 85,
        }

        # Cognitive thresholds
        self.cognitive_thresholds = {
            "attention": 0.7,
            "certainty": 0.8,
            "novelty": 0.6,
            "relevance": 0.75,
        }

        # Strategy-specific configs
        # Use string keys here to avoid referencing enums before they're defined
        self.strategy_configs = {
            "tdd": {
                "test_coverage_threshold": 0.95,  # High coverage requirement
                "mock_external_calls": True,
                "refactor_after_test": True,
                "mutation_threshold": 0.6,
            },
            "agile": {
                "sprint_duration": 14,  # Days
                "story_points_scale": [1, 2, 3, 5, 8, 13],
                "velocity_tracking": True,
                "sprint_batch_size": 5,
            },
            "devops": {
                "ci_frequency": "push",
                "cd_strategy": "blue-green",
                "security_scan_every": 5,  # Commits
            },
            "cognitive": {
                "evolution_frequency": 5,  # Tasks
                "metacognition_depth": 3,
                "novelty_threshold": 0.7,
            },
        }

        # Performance toggles
        # Skip per-file detailed blueprints and go straight to code generation
        self.skip_detailed_blueprints = True
        # Diff guard toggles
        self.enable_diff_guard = True
        self.forbidden_diff_patterns = [
            r"AKIA[0-9A-Z]{16}",  # AWS Access Key ID
            r"-----BEGIN (?:RSA|DSA|EC) PRIVATE KEY-----",
            r"rm\s+-rf\s+/",  # dangerous destructive command
            r"curl\s+http[s]?://[^\s]+\s*\|\s*sh",  # curl pipe to shell
        ]
        # Personas
        self.max_personas = 1000
        # Deep-think planner
        self.use_deep_planner = True
        self.research_budgets = {
            "max_queries": 8,
            "max_results_per_query": 5,
            "fetch_timeout_s": 20,
            "concurrency": 5,
        }
        # Cost controls for Agile
        self.reviewer_enabled = True
        self.reviewer_every_n_tasks = 3
        self.reviewer_min_lines = 80
        self.skip_reviewer_for_small_changes = True
        self.max_same_file_tasks = 2
        self.reviewer_max_per_file_consecutive = 1

        # Language configs with enhanced TDD support
        self.language_configs["python"]["test_command"] = (
            "pytest --cov=src --cov-report=term-missing -v --tb=short"  # Enhanced with coverage
        )
        self.language_configs["python"]["linter_command"] = (
            "ruff check . --fix --select=E,F,W && ruff format . && mypy . --strict"
        )
        self.language_configs["python"]["prerequisites"] = [
            "python",
            "pip",
            "pytest",
            "pytest-cov",
            "ruff",
            "mypy",
        ]  # Added tools
        self.language_configs["python"]["framework_configs"]["django"][
            "test_command"
        ] = "pytest --cov"
        self.language_configs["python"]["framework_configs"]["flask"][
            "test_command"
        ] = "pytest --cov"
        self.language_configs["python"]["framework_configs"]["fastapi"][
            "test_command"
        ] = "pytest --cov"
        self.language_configs["python"]["tdd_template"] = (
            "def test_{method_name}(self):\n    # Arrange\n    # Act\n    # Assert\n    pass\n"  # Added TDD template
        )

        self.language_configs["javascript"]["test_command"] = (
            "npm test -- --coverage --watchAll=false"  # Added coverage
        )
        self.language_configs["javascript"]["linter_command"] = (
            "eslint . --fix && npx prettier --write ."
        )
        self.language_configs["javascript"]["prerequisites"] = [
            "node",
            "npm",
            "jest",
            "@jest/globals",
        ]  # Added Jest
        self.language_configs["javascript"]["framework_configs"]["react"][
            "test_command"
        ] = "npm test -- --coverage"
        self.language_configs["javascript"]["framework_configs"]["vue"][
            "test_command"
        ] = "npm run test:unit -- --coverage"
        self.language_configs["javascript"]["framework_configs"]["angular"][
            "test_command"
        ] = "ng test --code-coverage"

        # Similar enhancements for other languages...
        self.language_configs["rust"]["test_command"] = (
            "cargo test -- --test-threads=1"  # Parallelism control
        )
        self.language_configs["rust"]["prerequisites"].append(
            "cargo-nextest"
        )  # For faster tests

        self.language_configs["go"]["test_command"] = (
            "go test ./... -coverprofile=coverage.out"
        )
        self.language_configs["go"]["prerequisites"].append(
            "gotestsum"
        )  # For better test output

        self.language_configs["java"]["test_command"] = "mvn test -Djacoco.skip=false"
        self.language_configs["java"]["prerequisites"].append(
            "jacoco-maven-plugin"
        )  # Coverage

        # Multi-modal with better limits
        self.max_image_size = (7680, 4320)  # 8K support
        self.max_audio_duration = 1800
        self.max_video_duration = 600

        # Build/Run defaults for multi-language execution
        try:
            self.language_configs["python"]["run_command"] = (
                f"{sys.executable} -m pytest -q"
            )
        except Exception:
            pass
        self.language_configs["javascript"]["run_command"] = "npm start"
        self.language_configs["javascript"]["build_command"] = "npm run build"
        self.language_configs["rust"]["build_command"] = "cargo build --release"
        self.language_configs["rust"]["run_command"] = "cargo run"
        self.language_configs["go"]["build_command"] = "go build ./..."
        self.language_configs["go"]["run_command"] = "go run ."
        self.language_configs["java"]["build_command"] = "mvn -q -DskipTests package"
        self.language_configs["java"]["run_command"] = "mvn -q exec:java"

        # Cloud with GitHub integration
        self.cloud_providers.append("github")
        self.deployment_strategies.append("gitops")

        # Security enhanced
        self.security_scans = ["sast", "dast", "sca", "secrets"]  # Added scan types

        # Performance with GPU monitoring
        self.monitoring_interval = 5
        self.performance_thresholds["cpu"] = 70
        self.performance_thresholds["memory"] = 85
        self.performance_thresholds["disk"] = 80
        self.performance_thresholds["network_latency"] = 200
        self.performance_thresholds["gpu"] = 75

        # Cognitive thresholds tuned
        self.cognitive_thresholds["attention"] = 0.8
        self.cognitive_thresholds["certainty"] = 0.85
        self.cognitive_thresholds["novelty"] = 0.7
        self.cognitive_thresholds["relevance"] = 0.8

        # Enhanced limits
        self.max_file_size_mb = 50  # Increased limits
        self.max_project_size_gb = 10
        self.max_memory_usage_percent = 90
        self.max_cpu_usage_percent = 75

        self.max_retries = 15 if self.env == "production" else 8  # Increased retries
        self.backoff_factor = 1.5  # More aggressive backoff
        self.max_api_calls_per_minute = 200 if self.env == "production" else 120
        self.request_timeout = 600.0
        self.metamorphosis_threshold = 2  # Lowered for faster evolution
        self.max_debug_attempts = 8

        # Cognitive parameters enhanced
        self.cognitive_dim = 1024  # Increased for richer representations
        self.quantum_qubits = 8
        self.neuroplasticity_rate = 0.05  # Slower learning for stability
        self.memory_capacity = 50000
        self.attention_heads = 12


# Global config instance
config = Config()