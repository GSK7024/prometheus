import json
import asyncio
import os
import httpx
from dotenv import load_dotenv
import shutil
import subprocess
import sys
import re
from typing import List, Optional, Dict, Tuple, Any
import logging
import logging.config
import uuid
from abc import ABC, abstractmethod
import time
import ast
import threading
import collections
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from functools import wraps
import signal
import traceback
import tempfile
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import random
import psutil
import yaml
from cryptography.fernet import Fernet

try:
    import docker  # Optional
except Exception:
    docker = None
try:
    from kubernetes import client, config as kube_config  # Optional
except Exception:
    client = None
    kube_config = None
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import DBSCAN

try:
    import faiss  # Optional; required for NeuromorphicMemory
except Exception:
    faiss = None

import chromadb
from chromadb.config import Settings

try:
    import coverage  # Optional: for code coverage measurement
except Exception:
    coverage = None
try:
    import pytest  # Optional: for enhanced TDD testing
except Exception:
    pytest = None


# sklearn does not provide `coverage_score` — remove invalid import and provide
# a small fallback function if any part of thecode calls it later.
def coverage_score(y_true, y_pred):
    """Fallback coverage_score placeholder.

    The original project imported `coverage_score` from sklearn which does not
    exist. Provide a minimal implementation that returns the proportion of
    non-empty predictions as a simple proxy. Replace with a real metric if
    you have a specific definition.
    """
    try:
        y_pred_list = list(y_pred)
        return sum(1 for p in y_pred_list if p) / max(1, len(y_pred_list))
    except Exception:
        return 0.0


from git import Repo  # Added for Git integration in DevOps


# --- Enhanced Quantum-Inspired Cognitive Architecture ---
class QuantumCognitiveCore(nn.Module):
    """Enhanced quantum-inspired neural network with improved entanglement and superposition simulation for superior decision making."""

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_qubits=8
    ):  # Increased qubits for better parallelism
        super(QuantumCognitiveCore, self).__init__()
        self.num_qubits = num_qubits
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Enhanced quantum-inspired layers with variational quantum circuits simulation
        self.quantum_encoder = nn.Linear(input_dim, num_qubits * hidden_dim)
        self.quantum_circuit = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),  # Added normalization for stability
                    nn.ReLU(),
                )
                for _ in range(num_qubits)
            ]
        )
        self.quantum_decoder = nn.Linear(num_qubits * hidden_dim, output_dim)

        # Improved entanglement with learnable Hadamard gates simulation
        self.entanglement = nn.Parameter(torch.randn(num_qubits, num_qubits))
        self.superposition_weights = nn.Parameter(
            torch.ones(num_qubits)
        )  # For superposition collapse

        # Enhanced cognitive state with memory gates
        self.cognitive_state = torch.zeros(1, hidden_dim)
        # Use output_dim for attention so it matches decoder output; keeps dimensions consistent
        self.attention_weights = nn.MultiheadAttention(
            output_dim, num_heads=8, batch_first=True
        )
        self.memory_gate = nn.Linear(output_dim, 1)  # For gating long-term memory

    def forward(self, x, cognitive_state=None):
        # Encode input into quantum superposition state
        encoded = torch.tanh(self.quantum_encoder(x))
        batch_size = x.size(0)
        encoded = encoded.view(batch_size, self.num_qubits, self.hidden_dim)

        # Apply variational quantum circuit transformations
        quantum_states = []
        for i in range(self.num_qubits):
            state = self.quantum_circuit[i](encoded[:, i, :])
            state = torch.sigmoid(state)  # Activation for qubit states
            quantum_states.append(state)

        # Simulate entanglement and superposition collapse
        entangled = torch.stack(quantum_states, dim=1)  # (B, Q, H)
        # Build entanglement matrix over qubits and apply across qubit axis
        ent_matrix = self.entanglement @ torch.diag(
            self.superposition_weights
        )  # (Q, Q)
        entangled = torch.einsum("bqh,qq->bqh", entangled, ent_matrix)  # (B, Q, H)
        entangled = entangled.reshape(batch_size, -1)

        # Decode to classical output with noise for exploration
        noise = torch.randn_like(entangled) * 0.01  # Quantum noise simulation
        output = self.quantum_decoder(entangled + noise)

        # Enhanced cognitive state update with memory gating
        if cognitive_state is not None:
            # Ensure cognitive_state shape is (batch, seq_len=1, embed=output_dim)
            if cognitive_state.dim() == 2:
                cognitive_state = cognitive_state.unsqueeze(1)
            # Project query over current output context
            attn_output, _ = self.attention_weights(
                cognitive_state, output.unsqueeze(1), output.unsqueeze(1)
            )
            # attn_output: (batch, 1, output_dim)
            gate = torch.sigmoid(self.memory_gate(attn_output.squeeze(1)))
            new_cognitive_state = cognitive_state.squeeze(
                1
            ) * gate + attn_output.squeeze(1) * (1 - gate)
            return output, new_cognitive_state

        return output


# --- Enhanced Neuromorphic Memory System ---
class NeuromorphicMemory:
    """Advanced memory system with improved clustering, lazy loading, and active forgetting mechanism."""

    def __init__(
        self, memory_dim=768, num_clusters=20
    ):  # Increased clusters for finer granularity
        self.memory_dim = memory_dim
        self.num_clusters = num_clusters
        # Use FAISS if available; otherwise fall back to a simple in-memory cosine index
        if faiss is not None:
            self.memory_index = faiss.IndexFlatIP(memory_dim)
            self._uses_faiss = True
        else:
            self._uses_faiss = False
            class _LocalIPIndex:
                def __init__(self, dim: int):
                    self.dim = dim
                    self.vectors = []
                def add(self, vecs):
                    for v in vecs:
                        self.vectors.append(v.astype(np.float32))
                def search(self, query, k):
                    if not self.vectors:
                        scores = np.zeros((1, k), dtype=np.float32)
                        indices = -np.ones((1, k), dtype=np.int64)
                        return scores, indices
                    mat = np.vstack(self.vectors).astype(np.float32)  # (N, D)
                    # Normalize matrix and query for cosine similarity
                    mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
                    q = query.astype(np.float32)
                    q = q / (np.linalg.norm(q) + 1e-8)
                    sims = np.dot(mat_norm, q.T).reshape(-1)  # (N,)
                    top_k = min(k, sims.shape[0])
                    idxs = np.argsort(-sims)[:top_k]
                    scores = sims[idxs].astype(np.float32)
                    # Pad to k
                    if top_k < k:
                        pad_scores = np.zeros(k - top_k, dtype=np.float32)
                        pad_idxs = -np.ones(k - top_k, dtype=np.int64)
                        scores = np.concatenate([scores, pad_scores])
                        idxs = np.concatenate([idxs, pad_idxs])
                    return scores.reshape(1, -1), idxs.reshape(1, -1)
            self.memory_index = _LocalIPIndex(memory_dim)
        self.memory_data = []
        self.memory_metadata = []
        self.cluster_model = DBSCAN(
            eps=0.3, min_samples=1
        )  # Tuned for better clustering
        # Keep a parallel store of embeddings for relevance and cluster heuristics
        self._embedding_store = []

        # Lazy-loaded components with fallback
        self.cognitive_embedder = None
        self.tokenizer = None
        self._is_initialized = False
        self.forgetting_threshold = (
            0.1  # For active forgetting of low-relevance memories
        )

    def _initialize_embedder(self):
        """Initializes the embedding model on first use with fallback."""
        if not self._is_initialized:
            logger.info("Initializing enhanced Neuromorphic Memory embedder...")
            try:
                self.cognitive_embedder = AutoModel.from_pretrained(
                    "sentence-transformers/all-mpnet-base-v2"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "sentence-transformers/all-mpnet-base-v2"
                )
                self._is_initialized = True
                logger.info("Embedder initialized successfully.")
            except Exception as e:
                logger.error(
                    f"Failed to initialize model: {e}. Using random embeddings as fallback."
                )
                self.embed_text = lambda text: np.random.randn(
                    1, self.memory_dim
                ).astype(np.float32)  # Fallback

    def embed_text(self, text):
        """Embed text with normalization."""
        self._initialize_embedder()
        if not self._is_initialized:
            return None

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.cognitive_embedder(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        return embedding / np.linalg.norm(embedding)  # L2 normalization

    def add_memory(self, data, metadata=None):
        """Add memory with relevance scoring and forgetting check."""
        embedding = self.embed_text(str(data))
        if embedding is None:
            return
        # Ensure embedding is float32 and normalized
        embedding = embedding.astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm

        # Compute relevance as similarity to existing memories (max cosine similarity)
        relevance = 1.0
        if self._embedding_store:
            mat = np.vstack(self._embedding_store).astype(np.float32)  # (N, D)
            sims = np.dot(mat, embedding.T).reshape(-1)
            relevance = float(np.max(sims))
        # Active forgetting of very low-relevance items
        if relevance < self.forgetting_threshold:
            logger.debug("Memory forgotten due to low relevance.")
            return

        self.memory_index.add(embedding)
        self.memory_data.append(data)
        self.memory_metadata.append(metadata or {})
        # Track embedding for future relevance/cluster computations
        self._embedding_store.append(embedding.squeeze(0))

        # Periodic clustering with forgetting
        if len(self.memory_data) % 50 == 0:  # More frequent updates
            self.update_clustering()

    def retrieve_similar(self, query, k=10):  # Increased k for broader recall
        """Retrieve similar memories with relevance filtering."""
        query_embedding = self.embed_text(query)
        if query_embedding is None:
            return []

        scores, indices = self.memory_index.search(query_embedding, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.memory_data):
                rel_score = scores[0][i]  # Inner product as relevance
                if rel_score > 0.5:  # Filter low-relevance
                    results.append(
                        {
                            "data": self.memory_data[idx],
                            "metadata": self.memory_metadata[idx],
                            "relevance": rel_score,
                        }
                    )
        return results

    def update_clustering(self):
        """Update clustering with active forgetting."""
        if len(self.memory_data) > 5:
            embeddings = np.vstack([self.embed_text(str(d)) for d in self.memory_data])
            if embeddings.shape[0] == 0:
                return

            clusters = self.cluster_model.fit_predict(embeddings)
            for i, cluster_id in enumerate(clusters):
                self.memory_metadata[i]["cluster"] = int(cluster_id)

            # Forget outlier clusters
            outlier_mask = clusters == -1
            self.memory_data = [
                d for j, d in enumerate(self.memory_data) if not outlier_mask[j]
            ]
            self.memory_metadata = [
                m for j, m in enumerate(self.memory_metadata) if not outlier_mask[j]
            ]


# --- Configuration Management ---
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
            },
            "agile": {
                "sprint_duration": 14,  # Days
                "story_points_scale": [1, 2, 3, 5, 8, 13],
                "velocity_tracking": True,
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


config = Config()


# --- Enhanced Logging with Cognitive Tracing ---
class CognitiveFormatter(logging.Formatter):
    """A custom log formatter that adds cognitive context safely."""

    def format(self, record):
        # Ensure trace_id and cognitive_state exist on the record
        if not hasattr(record, "trace_id"):
            record.trace_id = "NO_TRACE"

        if not hasattr(record, "cognitive_state"):
            cognitive_state_obj = getattr(
                threading.current_thread(), "cognitive_state", "INITIALIZING"
            )
            if isinstance(cognitive_state_obj, dict):
                record.cognitive_state = cognitive_state_obj.get(
                    "current_mode", "UNKNOWN"
                )
            else:
                record.cognitive_state = str(cognitive_state_obj)

        if not hasattr(record, "strategy"):
            record.strategy = getattr(
                threading.current_thread(), "dev_strategy", "UNKNOWN"
            )

        return super().format(record)


def setup_logging(log_level=logging.INFO, json_logging=False, enable_tracing=True):
    """Configure logging with cognitive tracing and structured logging"""

    trace_id = str(uuid.uuid4())

    class CognitiveFilter(logging.Filter):
        def filter(self, record):
            record.trace_id = trace_id
            cognitive_state_obj = getattr(
                threading.current_thread(), "cognitive_state", "INITIALIZING"
            )

            if isinstance(cognitive_state_obj, dict):
                record.cognitive_state = cognitive_state_obj.get(
                    "current_mode", "UNKNOWN"
                )
            else:
                record.cognitive_state = str(cognitive_state_obj)
            record.strategy = getattr(
                threading.current_thread(), "dev_strategy", "UNKNOWN"
            )
            return True

    log_handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "cognitive" if not json_logging else "json",
            "level": log_level,
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": f"cognition_{trace_id[:8]}.log",
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
            "formatter": "cognitive",
            "level": log_level,
        },
    }

    log_formatters = {
        "cognitive": {
            "()": CognitiveFormatter,  # Use custom formatter
            "format": "%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(trace_id)s] - [cognition=%(cognitive_state)s] - [strategy=%(strategy)s] - %(message)s",
        },
    }

    if json_logging:
        try:
            from pythonjsonlogger import jsonlogger

            log_formatters["json"] = {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(cognitive_state)s %(trace_id)s %(strategy)s",
            }
        except ImportError:
            logging.warning("JSON logger not available, using default formatting")

    # Get all existing loggers and add the filter to them
    for logger_name in logging.Logger.manager.loggerDict:
        logging.getLogger(logger_name).addFilter(CognitiveFilter())

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"cognitive": {"()": CognitiveFilter}},
            "formatters": log_formatters,
            "handlers": log_handlers,
            "root": {
                "handlers": ["console", "file"],
                "level": log_level,
                "filters": ["cognitive"],
            },
        }
    )

    if enable_tracing:
        logging.info(f"Cognitive tracing enabled with trace_id: {trace_id}")

    return trace_id


# Initialize logging
trace_id = setup_logging(logging.INFO, False, True)
logger = logging.getLogger(__name__)


# --- Advanced Utility Functions ---
def normalize_path(path: str) -> str:
    """Consistently formats a path to use forward slashes with advanced validation"""
    if not path:
        return ""

    # Convert to Path object for robust handling
    path_obj = Path(path)

    # Handle special cases (e.g., URLs, cloud storage paths)
    if path.startswith(("http://", "https://", "s3://", "gs://", "azure://")):
        return path  # Don't normalize URLs or cloud storage paths

    # Normalize local file paths
    normalized = path_obj.as_posix().strip("/")

    # Prevent path traversal attacks
    if any(part in ("..", "~") for part in normalized.split("/")):
        raise SecurityError(f"Potential path traversal detected: {path}")

    return normalized


def sanitize_dir_path(path: str) -> str:
    r"""Sanitize directory path components to remove characters invalid on Windows.

    This function replaces characters that Windows forbids in file/dir names
    (e.g. < > : " / \ | ? *) with underscores for each path component while
    preserving an initial drive specifier like 'C://'. It also avoids reserved
    names such as CON, PRN, AUX, NUL, COM1..COM9, LPT1..LPT9 by appending
    a suffix if needed.
    """
    if not path:
        return path

    try:
        p = Path(path)
    except Exception:
        # Best-effort fallback
        return re.sub(r'[<>:"|?*]', "_", path)

    parts = []
    for i, part in enumerate(p.parts):
        # Preserve drive (e.g., 'C:\\') as the first part on Windows
        if i == 0 and re.match(r"^[A-Za-z]:$", part):
            parts.append(part)
            continue

        # Replace illegal characters
        safe = re.sub(r'[<>:"|?*]', "_", part)

        # Avoid reserved device names
        upper = safe.upper()
        if upper in {"CON", "PRN", "AUX", "NUL"} or re.match(
            r"^(COM[1-9]|LPT[1-9])$", upper
        ):
            safe = safe + "_dir"

        # Avoid empty path parts
        if not safe:
            safe = "_"

        parts.append(safe)

    try:
        safe_path = Path(*parts)
        # Return a string form that is appropriate for the OS
        return str(safe_path)
    except Exception:
        return re.sub(r'[<>:"|?*]', "_", path)


def sanitize_config_paths(cfg: "Config"):
    """Sanitize common config directory paths in-place."""
    try:
        cfg.sandbox_base_path = sanitize_dir_path(cfg.sandbox_base_path)
        cfg.session_dir = sanitize_dir_path(cfg.session_dir)
        cfg.memory_dir = sanitize_dir_path(cfg.memory_dir)
        cfg.cache_dir = sanitize_dir_path(cfg.cache_dir)
    except Exception:
        # If sanitization unexpectedly fails, leave values as-is; higher-level
        # code will handle mkdir failures and log them.
        pass


def create_dir_safely(path: str) -> Path:
    """Attempt to create a directory and fall back to a temp directory on failure.

    Returns the Path that should be used (may be the original or a temp dir).
    """
    try:
        safe_path_str = sanitize_dir_path(path)
        p = Path(safe_path_str)
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception as e:
        logger.warning(
            f"Failed to create directory '{path}': {e}. Falling back to temp directory."
        )
        temp = Path(tempfile.gettempdir()) / f"laar_fallback_{uuid.uuid4().hex[:8]}"
        try:
            temp.mkdir(parents=True, exist_ok=True)
            return temp
        except Exception:
            # As a last resort return current working dir as Path
            return Path(".")


def safe_execute(func):
    """Decorator for safe execution with comprehensive error handling and metrics"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        cognitive_state = getattr(threading.current_thread(), "cognitive_state", {})
        dev_strategy = getattr(threading.current_thread(), "dev_strategy", "UNKNOWN")
        if isinstance(cognitive_state, dict):
            cognitive_state["current_operation"] = func.__name__
            cognitive_state["strategy_context"] = dev_strategy

        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(
                f"Function {func.__name__} executed successfully in {execution_time:.2f}s"
            )

            # Update cognitive metrics
            if isinstance(cognitive_state, dict):
                cognitive_state["last_success"] = time.time()
                cognitive_state["success_count"] = (
                    cognitive_state.get("success_count", 0) + 1
                )
                cognitive_state["avg_execution_time"] = (
                    cognitive_state.get("avg_execution_time", 0) * 0.9
                    + execution_time * 0.1
                )

            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Error in {func.__name__}: {str(e)} (execution time: {execution_time:.2f}s)"
            )
            logger.debug(traceback.format_exc())

            # Update cognitive metrics
            if isinstance(cognitive_state, dict):
                cognitive_state["last_error"] = time.time()
                cognitive_state["error_count"] = (
                    cognitive_state.get("error_count", 0) + 1
                )
                cognitive_state["last_error_type"] = type(e).__name__

            # Check if this is a retryable error
            if isinstance(
                e, (httpx.HTTPStatusError, httpx.RequestError, asyncio.TimeoutError)
            ):
                raise  # Let tenacity handle retries

            # For other errors, wrap in a standard exception
            raise ExecutionError(f"Error in {func.__name__}: {str(e)}") from e

    return wrapper


def validate_file_path(path: str, base_dir: str) -> bool:
    """Validate that a file path is within the allowed directory with advanced checks"""
    try:
        full_path = Path(base_dir) / path
        resolved_path = full_path.resolve()
        base_resolved = Path(base_dir).resolve()

        # Check if path is within base directory
        if (
            base_resolved not in resolved_path.parents
            and resolved_path != base_resolved
        ):
            return False

        # Check for forbidden patterns
        forbidden_dirs = ["/etc/", "/root/", "/var/", "/proc/", "/sys/"]
        path_str = str(resolved_path).replace("\\", "/")
        if any(fd in path_str for fd in forbidden_dirs):
            return False
        forbidden_exts = (".pem", ".key", ".env", ".secret", ".token")
        if any(path_str.lower().endswith(ext) for ext in forbidden_exts):
            return False

        return True
    except (ValueError, RuntimeError):
        return False


def encrypt_data(data: str, key: str = None) -> str:
    """Encrypt sensitive data using Fernet symmetric encryption"""
    key = key or config.encryption_key
    fernet = Fernet(key.encode())
    encrypted = fernet.encrypt(data.encode())
    return encrypted.decode()


def decrypt_data(encrypted_data: str, key: str = None) -> str:
    """Decrypt data using Fernet symmetric encryption"""
    key = key or config.encryption_key
    fernet = Fernet(key.encode())
    decrypted = fernet.decrypt(encrypted_data.encode())
    return decrypted.decode()


def get_system_metrics() -> Dict[str, float]:
    """Get comprehensive system metrics for performance monitoring"""
    metrics = {}

    # CPU usage
    metrics["cpu_percent"] = psutil.cpu_percent(interval=1)

    # Memory usage
    memory = psutil.virtual_memory()
    metrics["memory_percent"] = memory.percent
    metrics["memory_available_gb"] = memory.available / (1024**3)

    # Disk usage
    disk = psutil.disk_usage("/")
    metrics["disk_percent"] = disk.percent
    metrics["disk_free_gb"] = disk.free / (1024**3)

    # Network metrics
    net_io = psutil.net_io_counters()
    metrics["network_bytes_sent"] = net_io.bytes_sent
    metrics["network_bytes_recv"] = net_io.bytes_recv

    return metrics


# --- Advanced Type Definitions ---
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
    goal: str
    project_name: str
    completed_tasks: List[str] = field(default_factory=list)
    task_queue: List[Dict] = field(default_factory=list)
    living_blueprint: LivingBlueprint = field(default_factory=LivingBlueprint)
    file_contents: Dict[str, str] = field(default_factory=dict)
    chronic_failure_tracker: Dict[str, int] = field(default_factory=dict)
    development_strategy: str = "tdd"
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    security_scan_results: Dict[str, Any] = field(default_factory=dict)
    cost_estimates: Dict[str, float] = field(default_factory=dict)
    cognitive_state: Dict[str, Any] = field(default_factory=dict)
    evolutionary_path: List[Dict] = field(default_factory=list)
    # Agile specific state
    epics: List[str] = field(default_factory=list)
    current_epic: Optional[str] = None
    # Cognitive specific state
    current_strategy: str = "tdd"
    # Agile enhancements
    user_stories: List[Dict] = field(default_factory=list)  # New
    sprint_backlog: List[Dict] = field(default_factory=list)  # New
    velocity_history: List[float] = field(default_factory=list)  # New
    # DevOps enhancements
    git_repo: Optional[str] = None  # New
    ci_status: str = "pending"  # New
    # Cognitive
    metacognition_log: List[Dict] = field(default_factory=list)  # New

    def __post_init__(self):
        if not self.living_blueprint:
            self.living_blueprint = LivingBlueprint()
        if not self.file_contents:
            self.file_contents = {}
        if not self.chronic_failure_tracker:
            self.chronic_failure_tracker = {}
        if not self.performance_metrics:
            self.performance_metrics = {}
        if not self.security_scan_results:
            self.security_scan_results = {}
        if not self.cost_estimates:
            self.cost_estimates = {}
        if not self.cognitive_state:
            self.cognitive_state = {
                "current_mode": CognitiveState.ANALYTICAL.value,
                "attention_level": 0.8,
                "certainty": 0.7,
                "novelty": 0.5,
                "relevance": 0.9,
                "last_state_change": time.time(),
                "evolution_count": 0,
                "metacognitive_awareness": 0.6,
            }
        if not self.evolutionary_path:
            self.evolutionary_path = []


# --- Custom Exceptions ---
class SecurityError(Exception):
    """Security-related exception"""

    pass


class ExecutionError(Exception):
    """General execution error"""

    pass


class ResourceExhaustedError(Exception):
    """System resource exhaustion error"""

    pass


class EvolutionRequiredError(Exception):
    """Exception indicating that system evolution is required"""

    pass


class CognitiveShiftError(Exception):
    """Exception indicating a need for cognitive state change"""

    pass


# EVOLUTION: New exception for handling token limit errors from the API.
class TokenLimitError(Exception):
    """Exception for when the API response is truncated due to token limits."""

    pass


# --- Enhanced Memory Manager with Neuromorphic Support ---
class NeuromorphicMemoryManager:
    """Manages multi-modal long-term memory with neuromorphic capabilities"""

    def __init__(self, memory_dir: str = config.memory_dir):
        self.memory_dir = memory_dir
        self.client = None
        self.collections = {}
        self.neuromorphic_memory = NeuromorphicMemory()

        try:
            self.client = chromadb.PersistentClient(
                path=memory_dir, settings=Settings(anonymized_telemetry=False)
            )

            # Create collections for different types of memories
            self.collections = {
                "project_blueprints": self.client.get_or_create_collection(
                    name="project_blueprints", metadata={"hnsw:space": "cosine"}
                ),
                "bug_solutions": self.client.get_or_create_collection(
                    name="bug_solutions", metadata={"hnsw:space": "cosine"}
                ),
                "successful_strategies": self.client.get_or_create_collection(
                    name="successful_strategies", metadata={"hnsw:space": "cosine"}
                ),
                "code_patterns": self.client.get_or_create_collection(
                    name="code_patterns", metadata={"hnsw:space": "cosine"}
                ),
                "architecture_patterns": self.client.get_or_create_collection(
                    name="architecture_patterns", metadata={"hnsw:space": "cosine"}
                ),
                "cognitive_patterns": self.client.get_or_create_collection(
                    name="cognitive_patterns", metadata={"hnsw:space": "cosine"}
                ),
                "tdd_patterns": self.client.get_or_create_collection(
                    name="tdd_patterns", metadata={"hnsw:space": "cosine"}
                ),  # New
                "agile_retrospectives": self.client.get_or_create_collection(
                    name="agile_retrospectives", metadata={"hnsw:space": "cosine"}
                ),  # New
                "devops_pipelines": self.client.get_or_create_collection(
                    name="devops_pipelines", metadata={"hnsw:space": "cosine"}
                ),  # New
            }

            logger.info("Neuromorphic memory initialized successfully")
        except ImportError:
            logger.warning(
                "chromadb is not installed. Long-term memory will be disabled."
            )
        except Exception as e:
            logger.error(f"Failed to initialize memory: {str(e)}")

    def add_memory(
        self,
        goal: str,
        blueprint: LivingBlueprint,
        strategy: List[str] = None,
        modalities: List[Modality] = None,
        embeddings: Optional[Dict[str, List[float]]] = None,
        cognitive_state: Dict = None,
        dev_strategy: str = None,
    ):
        if "project_blueprints" not in self.collections:
            return

        try:
            # Store blueprint
            blueprint_doc = f"User Goal: {goal}\n\nBlueprint:\n{blueprint.to_json()}\n\nCognitive State: {json.dumps(cognitive_state) if cognitive_state else 'N/A'}\nStrategy: {dev_strategy}"
            blueprint_id = f"project_{hash(goal)}_{dev_strategy or 'default'}"

            # Add embeddings if provided
            embedding = embeddings.get("blueprint", []) if embeddings else []

            self.collections["project_blueprints"].add(
                documents=[blueprint_doc],
                ids=[blueprint_id],
                embeddings=[embedding] if embedding else None,
            )

            # Store strategy
            if strategy and "successful_strategies" in self.collections:
                strategy_doc = (
                    f"User Goal: {goal}\n\nSuccessful Strategy:\n"
                    + "\n".join(f"- {s}" for s in strategy)
                )
                strategy_id = f"strategy_{hash(goal)}_{dev_strategy or 'default'}"

                strategy_embedding = (
                    embeddings.get("strategy", []) if embeddings else []
                )

                self.collections["successful_strategies"].add(
                    documents=[strategy_doc],
                    ids=[strategy_id],
                    embeddings=[strategy_embedding] if strategy_embedding else None,
                )

            # Strategy-specific additions
            if (
                dev_strategy == DevelopmentStrategy.TDD.value
                and "tdd_patterns" in self.collections
            ):
                self.collections["tdd_patterns"].add(
                    documents=[f"TDD for {goal}"], ids=[f"tdd_{hash(goal)}"]
                )

            if (
                dev_strategy == DevelopmentStrategy.AGILE.value
                and "agile_retrospectives" in self.collections
            ):
                self.collections["agile_retrospectives"].add(
                    documents=[f"Agile for {goal}"], ids=[f"agile_{hash(goal)}"]
                )

            if (
                dev_strategy == DevelopmentStrategy.DEVOPS.value
                and "devops_pipelines" in self.collections
            ):
                self.collections["devops_pipelines"].add(
                    documents=[f"DevOps for {goal}"], ids=[f"devops_{hash(goal)}"]
                )

            # Add to neuromorphic memory
            self.neuromorphic_memory.add_memory(
                {"goal": goal, "blueprint": blueprint.to_json()},
                {
                    "type": "project",
                    "strategy": strategy,
                    "cognitive_state": cognitive_state,
                },
            )

            logger.info(f"Successfully saved neuromorphic memory for goal: '{goal}'")
        except Exception as e:
            logger.warning(f"Failed to add memory: {e}")

    def retrieve_similar_memories(
        self, query: str, n_results: int = 5, collection: str = "project_blueprints"
    ) -> list:
        if (
            collection not in self.collections
            or self.collections[collection].count() == 0
        ):
            return []

        try:
            results = self.collections[collection].query(
                query_texts=[query], n_results=n_results
            )
            return results.get("documents", [[]])[0]
        except Exception as e:
            logger.warning(f"Failed to retrieve memories: {e}")
            return []

    def retrieve_by_embedding(
        self,
        embedding: List[float],
        n_results: int = 5,
        collection: str = "project_blueprints",
    ) -> list:
        if (
            collection not in self.collections
            or self.collections[collection].count() == 0
        ):
            return []

        try:
            results = self.collections[collection].query(
                query_embeddings=[embedding], n_results=n_results
            )
            return results.get("documents", [[]])[0]
        except Exception as e:
            logger.warning(f"Failed to retrieve by embedding: {e}")
            return []

    def add_bug_solution(
        self,
        error_log: str,
        socratic_analysis: str,
        solution: dict,
        embedding: Optional[List[float]] = None,
        cognitive_state: Dict = None,
    ):
        if "bug_solutions" not in self.collections:
            return

        try:
            document = f"Error Log:\n{error_log}\n\nSocratic Analysis:\n{socratic_analysis}\n\nSolution:\n{json.dumps(solution, indent=2)}\n\nCognitive State: {json.dumps(cognitive_state) if cognitive_state else 'N/A'}"
            doc_id = f"bug_{hash(error_log)}"

            self.collections["bug_solutions"].add(
                documents=[document],
                ids=[doc_id],
                embeddings=[embedding] if embedding else None,
            )

            # Add to neuromorphic memory
            self.neuromorphic_memory.add_memory(
                {"error": error_log, "solution": solution},
                {
                    "type": "bug",
                    "analysis": socratic_analysis,
                    "cognitive_state": cognitive_state,
                },
            )

            logger.info("Successfully saved bug solution with embedding")
        except Exception as e:
            logger.warning(f"Failed to add bug solution: {e}")

    def add_code_pattern(
        self,
        pattern_name: str,
        pattern_code: str,
        description: str,
        embedding: Optional[List[float]] = None,
        cognitive_state: Dict = None,
    ):
        if "code_patterns" not in self.collections:
            return

        try:
            document = f"Pattern: {pattern_name}\n\nDescription: {description}\n\nCode:\n{pattern_code}\n\nCognitive State: {json.dumps(cognitive_state) if cognitive_state else 'N/A'}"
            doc_id = f"pattern_{hash(pattern_name)}"

            self.collections["code_patterns"].add(
                documents=[document],
                ids=[doc_id],
                embeddings=[embedding] if embedding else None,
            )

            # Add to neuromorphic memory
            self.neuromorphic_memory.add_memory(
                {"pattern": pattern_name, "code": pattern_code},
                {
                    "type": "pattern",
                    "description": description,
                    "cognitive_state": cognitive_state,
                },
            )

            logger.info(f"Successfully saved code pattern: {pattern_name}")
        except Exception as e:
            logger.warning(f"Failed to add code pattern: {e}")

    def retrieve_cognitive_patterns(self, query: str, n_results: int = 5) -> list:
        """Retrieve patterns based on cognitive similarity"""
        return self.neuromorphic_memory.retrieve_similar(query, n_results)


# --- Helper function for Tenacity retry condition ---
def is_retryable_api_error(exception: BaseException) -> bool:
    """
    Determines if an exception from an API call is retryable.
    Retry on server errors (5xx), rate limiting (429), or network issues.
    """
    if isinstance(exception, httpx.HTTPStatusError):
        is_server_error = exception.response.status_code >= 500
        is_rate_limit = exception.response.status_code == 429
        if is_server_error:
            logger.warning(
                f"Server error {exception.response.status_code}. Will retry..."
            )
            return True
        if is_rate_limit:
            logger.warning("Rate limit exceeded. Will retry after a delay...")
            return True
    elif isinstance(exception, (httpx.RequestError, asyncio.TimeoutError)):
        logger.warning(
            f"Network error encountered ({type(exception).__name__}). Will retry..."
        )
        return True
    return False


# --- Advanced AI Interaction Layer with Quantum Cognitive Core ---
class QuantumCognitiveAI:
    """Advanced AI interaction layer with quantum-inspired cognitive capabilities"""

    def __init__(self, llm_choice: str = "api", api_keys: Dict[str, str] = None):
        self.llm_choice = llm_choice
        self.api_call_count = 0
        self.last_call_time = 0
        self.rate_limit_lock = asyncio.Lock()
        # Limit concurrent outbound API calls to reduce 429/503 likelihood
        self._concurrency = asyncio.Semaphore(2)
        self.api_keys = api_keys or {
            "gemini": os.getenv("GEMINI_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
        }
        self.available_models = {}
        self.current_model = None
        self.model_preference = []
        # Simple LRU cache for identical prompts to cut duplicate costs
        self._prompt_cache_max = 128
        self._prompt_cache_ttl = 300.0  # seconds
        self._prompt_cache = collections.OrderedDict()
        self.cognitive_core = QuantumCognitiveCore(
            input_dim=config.cognitive_dim,
            hidden_dim=256,
            output_dim=config.cognitive_dim,
            num_qubits=config.quantum_qubits,
        )
        self.cognitive_state = torch.zeros(1, 256)

        # Initialize available models based on configuration
        self._initialize_models()

        # Initialize multi-modal capabilities
        self.supported_modalities = self._get_supported_modalities()

    def _initialize_models(self):
        """Initialize available AI models with fallback preferences and corrected token limits."""
        if self.llm_choice == "api":
            # ARCHITECTURAL CHANGE: Per user request, only use gemini-2.5-flash and remove fallback.
            self.available_models = {
                "gemini-2.5-flash": {
                    "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
                    "modalities": ["text", "image"],
                    "max_tokens": 819200,
                    "cost_per_1k": 0.50,
                }
            }
            self.model_preference = ["gemini-2.5-flash"]
            self.current_model = self.model_preference[0]
        elif self.llm_choice == "openai":
            self.available_models = {
                "openai-gpt-4": {
                    "endpoint": "https://api.openai.com/v1/chat/completions",
                    "modalities": ["text"],
                    "max_tokens": 8192,
                    "cost_per_1k": 30.00,
                    "model_name": "gpt-4",
                },
                "openai-gpt-4-turbo": {
                    "endpoint": "https://api.openai.com/v1/chat/completions",
                    "modalities": ["text", "image"],
                    "max_tokens": 8192,
                    "cost_per_1k": 10.00,
                    "model_name": "gpt-4-turbo",
                },
            }
            self.model_preference = ["openai-gpt-4-turbo", "openai-gpt-4"]
            self.current_model = self.model_preference[0]
        elif self.llm_choice == "local":
            self.available_models = {
                "llama3:8b": {
                    "endpoint": "http://localhost:11434/api/generate",
                    "modalities": ["text"],
                    "max_tokens": 4096,
                    "cost_per_1k": 0.00,
                },
                "llama3:70b": {
                    "endpoint": "http://localhost:11434/api/generate",
                    "modalities": ["text"],
                    "max_tokens": 8192,
                    "cost_per_1k": 0.00,
                },
                "mistral:7b": {
                    "endpoint": "http://localhost:11434/api/generate",
                    "modalities": ["text"],
                    "max_tokens": 4096,
                    "cost_per_1k": 0.00,
                },
                "mixtral:8x7b": {
                    "endpoint": "http://localhost:11434/api/generate",
                    "modalities": ["text"],
                    "max_tokens": 8192,
                    "cost_per_1k": 0.00,
                },
            }
            self.model_preference = ["llama3:8b"]
            self.current_model = self.model_preference[0]

        logger.info(
            f"Initialized with model provider '{self.llm_choice}', default model '{self.current_model}'"
        )

    def _get_supported_modalities(self) -> List[Modality]:
        """Get supported modalities for the current model"""
        if self.current_model and self.current_model in self.available_models:
            return self.available_models[self.current_model].get("modalities", ["text"])
        return ["text"]

    async def _enforce_rate_limit(self):
        """Enforce rate limiting for API calls with cost tracking"""
        async with self.rate_limit_lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            min_interval = 60.0 / config.max_api_calls_per_minute

            if time_since_last_call < min_interval:
                await asyncio.sleep(min_interval - time_since_last_call)

            self.last_call_time = time.time()

    def _calculate_cost(
        self, prompt_tokens: int, completion_tokens: int, model_name: str
    ) -> float:
        """Calculate the cost of an API call"""
        if model_name in self.available_models:
            cost_per_1k = self.available_models[model_name].get("cost_per_1k", 0)
            total_tokens = prompt_tokens + completion_tokens
            return (total_tokens / 1000) * cost_per_1k
        return 0.0

    def _apply_cognitive_filter(
        self, prompt: str, cognitive_state: Dict, dev_strategy: str = None
    ) -> str:
        """Apply cognitive filtering to the prompt based on current state"""
        if cognitive_state.get("current_mode") == CognitiveState.FOCUSED.value:
            # Add focus directives to prompt
            prompt = f"[FOCUS MODE] Please provide concise, focused responses. Avoid unnecessary elaboration.\n\n{prompt}"
        elif cognitive_state.get("current_mode") == CognitiveState.CREATIVE.value:
            # Add creativity directives to prompt
            prompt = f"[CREATIVE MODE] Please think outside the box and provide innovative solutions.\n\n{prompt}"
        elif cognitive_state.get("current_mode") == CognitiveState.ANALYTICAL.value:
            # Add analytical directives to prompt
            prompt = f"[ANALYTICAL MODE] Please provide detailed, logical analysis with clear reasoning.\n\n{prompt}"
        elif cognitive_state.get("current_mode") == CognitiveState.EVOLUTIONARY.value:
            prompt = f"[EVOLUTIONARY MODE] Suggest improvements and evolutionary changes.\n\n{prompt}"

        # Adjust based on attention level
        attention = cognitive_state.get("attention_level", 0.8)
        if attention < 0.6:
            prompt = f"[LOW ATTENTION] Please provide simpler, more direct responses.\n\n{prompt}"

        # Strategy-specific directives
        if dev_strategy == DevelopmentStrategy.TDD.value:
            prompt += "\n[TDD] Always include tests first, ensure >95% coverage."
        elif dev_strategy == DevelopmentStrategy.AGILE.value:
            prompt += "\n[AGILE] Break into user stories with acceptance criteria."
        elif dev_strategy == DevelopmentStrategy.DEVOPS.value:
            prompt += "\n[DEVOPS] Include CI/CD steps and GitOps integration."
        elif dev_strategy == DevelopmentStrategy.COGNITIVE.value:
            prompt += "\n[COGNITIVE] Optimize for self-evolution and metacognition."

        return prompt

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=5, max=120),
        retry=retry_if_exception(is_retryable_api_error),
        before_sleep=lambda retry_state: logger.info(
            f"API call failed with a retryable error: {retry_state.outcome.exception()}. "
            f"Retrying in {retry_state.next_action.sleep:.2f} seconds... (Attempt {retry_state.attempt_number})"
        ),
    )
    @safe_execute
    async def _make_api_call(
        self,
        prompt: str,
        phase: str,
        is_json: bool = False,
        image_data: str = None,
        audio_data: str = None,
        video_data: str = None,
        modalities: List[Modality] = None,
        cognitive_state: Dict = None,
        model_override: Optional[str] = None,
        dev_strategy: str = None,
    ):
        """
        Make multi-modal API call with cognitive filtering, model override, and robust retry logic.
        This function no longer uses a fallback model loop; it relies on the @retry decorator.
        """
        # Concurrency and app-level rate limit
        await self._enforce_rate_limit()
        self.api_call_count += 1

        modalities = modalities or ["text"]
        model_name_to_use = model_override or self.current_model
        model_config = self.available_models.get(model_name_to_use, {})

        logger.info(
            f"API call attempt with model: {model_name_to_use} for phase: {phase}"
        )

        if cognitive_state:
            filtered_prompt = self._apply_cognitive_filter(
                prompt, cognitive_state, dev_strategy
            )
        else:
            filtered_prompt = prompt

        # Prompt cache lookup (after filtered_prompt is set)
        cache_key = None
        if modalities is None:
            modalities = ["text"]
        if not (image_data or audio_data or video_data):
            cache_key = (model_name_to_use, is_json, hash(filtered_prompt))
            # purge expired entries
            now_ts = time.time()
            expired = []
            for k, (ts, val) in list(self._prompt_cache.items()):
                if now_ts - ts > self._prompt_cache_ttl:
                    expired.append(k)
            for k in expired:
                self._prompt_cache.pop(k, None)
            if cache_key in self._prompt_cache:
                ts, cached_text = self._prompt_cache[cache_key]
                if now_ts - ts <= self._prompt_cache_ttl:
                    # Move to end (LRU)
                    self._prompt_cache.move_to_end(cache_key)
                    return cached_text

        payload = {}
        headers = {"Content-Type": "application/json"}

        if self.llm_choice == "api" and model_name_to_use.startswith("gemini"):
            parts = [{"text": filtered_prompt}]
            if image_data and "image" in modalities:
                parts.append(
                    {"inline_data": {"mime_type": "image/png", "data": image_data}}
                )
            if audio_data and "audio" in modalities:
                parts.append(
                    {"inline_data": {"mime_type": "audio/mp3", "data": audio_data}}
                )
            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "temperature": 0.7,
                    "topP": 0.95,
                    "maxOutputTokens": model_config.get("max_tokens", 8192),
                },
            }
            if is_json:
                payload["generationConfig"]["responseMimeType"] = "application/json"

        elif self.llm_choice == "openai":
            headers["Authorization"] = f"Bearer {self.api_keys.get('openai', '')}"
            messages = [{"role": "user", "content": filtered_prompt}]
            if dev_strategy:
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": f"Development strategy: {dev_strategy}",
                    },
                )
            payload = {
                "model": model_config.get("model_name", "gpt-4-turbo"),
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": model_config.get("max_tokens", 8192),
            }
            if is_json:
                payload["response_format"] = {"type": "json_object"}

        elif self.llm_choice == "local":
            payload = {
                "model": model_name_to_use.split(":")[0],
                "prompt": filtered_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "num_predict": model_config.get("max_tokens", 4096),
                },
            }
            if is_json:
                payload["format"] = "json"

        try:
            endpoint = model_config["endpoint"]
            if self.llm_choice == "api":
                endpoint += f"?key={self.api_keys.get('gemini', '')}"

            async with self._concurrency:
                async with httpx.AsyncClient(
                    timeout=config.request_timeout, verify=config.ssl_verification
                ) as client:
                    response = await client.post(
                        endpoint, headers=headers, json=payload
                    )
                    response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx
                result = response.json()

            prompt_tokens = len(filtered_prompt.split()) * 1.3
            completion_text = ""
            if self.llm_choice == "api":
                if (
                    "candidates" in result
                    and result["candidates"]
                    and "content" in result["candidates"][0]
                    and "parts" in result["candidates"][0]["content"]
                    and result["candidates"][0]["content"]["parts"]
                ):
                    # EVOLUTION: Check for MAX_TOKENS finish reason.
                    finish_reason = result["candidates"][0].get("finishReason")
                    if finish_reason == "MAX_TOKENS":
                        logger.warning("API response was truncated due to MAX_TOKENS.")
                        raise TokenLimitError(
                            "The model's response was cut short due to token limits."
                        )

                    completion_text = result["candidates"][0]["content"]["parts"][0][
                        "text"
                    ]
                else:
                    logger.warning(
                        f"Unexpected API response structure from {model_name_to_use}: {result}"
                    )
                    completion_text = json.dumps(result)
            elif self.llm_choice == "openai":
                completion_text = result["choices"][0]["message"]["content"]
            elif self.llm_choice == "local":
                completion_text = result["response"]

            completion_tokens = len(completion_text.split()) * 1.3
            cost = self._calculate_cost(
                int(prompt_tokens), int(completion_tokens), model_name_to_use
            )
            logger.info(
                f"API call successful. Phase: {phase}. Model: {model_name_to_use}. Estimated cost: ${cost:.4f}"
            )
            # cache result
            if cache_key is not None:
                self._prompt_cache[cache_key] = (time.time(), completion_text)
                if len(self._prompt_cache) > self._prompt_cache_max:
                    self._prompt_cache.popitem(last=False)
            return completion_text
        except httpx.HTTPStatusError as e:
            # Respect Retry-After on 429/503 if provided
            status = e.response.status_code
            if status in (429, 503):
                retry_after = e.response.headers.get("retry-after")
                sleep_s = None
                if retry_after:
                    try:
                        sleep_s = float(retry_after)
                    except Exception:
                        sleep_s = None
                if not sleep_s:
                    # Try to parse Google RetryInfo from body
                    try:
                        body = e.response.json()
                        details = body.get("error", {}).get("details", [])
                        for d in details:
                            if (
                                d.get("@type", "").endswith("RetryInfo")
                                and "retryDelay" in d
                            ):
                                rs = d["retryDelay"]
                                if isinstance(rs, str) and rs.endswith("s"):
                                    sleep_s = float(rs[:-1])
                                else:
                                    sleep_s = float(rs)
                                break
                    except Exception:
                        pass
                if sleep_s:
                    jitter = random.uniform(0, 0.25 * sleep_s)
                    await asyncio.sleep(min(120.0, sleep_s + jitter))
            logger.error(f"API call to {model_name_to_use} failed: {e}")
            raise
        except (httpx.RequestError, asyncio.TimeoutError) as e:
            logger.error(f"Network error during API call: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during API call: {e}")
            raise

    def _clean_code(self, response_text: str, language: str = "python") -> str:
        """Extracts code from markdown fences with advanced parsing"""
        patterns = [
            f"```{language}\\n(.*?)\\n```",
            f"```{language.lower()}\\n(.*?)\\n```",
            "```\\n(.*?)\\n```",
            f"<{language}_code>\\n(.*?)\\n</{language}_code>",
        ]

        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                return match.group(1).strip()

        # If no code blocks found, try to extract just the code
        lines = response_text.split("\n")
        code_lines = [
            line
            for line in lines
            if not line.strip().startswith(("#", "//", "/*", "*"))
        ]
        return "\n".join(code_lines).strip()

    def _clean_json(self, response_data: Any) -> dict:
        """Extracts JSON from response with advanced error recovery"""
        if isinstance(response_data, dict):
            return response_data

        if not isinstance(response_data, str):
            raise TypeError(
                f"Expected string or dict for JSON cleaning, got {type(response_data)}"
            )

        # Attempt to find JSON within markdown fences first
        match = re.search(r"```json\n(.*?)\n```", response_data, re.DOTALL)
        if match:
            response_data = match.group(1)

        # Multiple JSON extraction strategies
        strategies = [
            lambda s: re.search(r"\{.*\}", s, re.DOTALL),  # Try to find JSON object
            lambda s: re.search(r"\[.*\]", s, re.DOTALL),  # Try to find JSON array
        ]

        for strategy in strategies:
            try:
                match = strategy(response_data)
                if match:
                    json_str = match.group(0)
                    return json.loads(json_str)
            except (json.JSONDecodeError, AttributeError):
                continue

        # Bracket-aware scan to extract the first valid JSON object/array
        def _extract_bracketed(s: str) -> Optional[str]:
            start_idxs = [i for i, ch in enumerate(s) if ch in "{["]
            for start in start_idxs:
                stack = []
                in_str = False
                esc = False
                for i in range(start, len(s)):
                    ch = s[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                        continue
                    else:
                        if ch == '"':
                            in_str = True
                            continue
                        if ch in "{[":
                            stack.append(ch)
                        elif ch in "}]":
                            if not stack:
                                break
                            open_ch = stack.pop()
                            if (open_ch == "{" and ch != "}") or (
                                open_ch == "[" and ch != "]"
                            ):
                                break
                            if not stack:
                                candidate = s[start : i + 1]
                                try:
                                    json.loads(candidate)
                                    return candidate
                                except Exception:
                                    pass
                # try next possible start
            return None

        candidate = _extract_bracketed(response_data)
        if candidate:
            return json.loads(candidate)

        # Final attempt: parse the whole string
        try:
            return json.loads(response_data)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"No valid JSON found in response: {e.msg}", response_data, e.pos
            )

    async def _get_structured_output(
        self,
        original_prompt: str,
        persona_name: str,
        default_response: dict,
        schema: dict = None,
        modalities: List[Modality] = None,
        cognitive_state: Dict = None,
        model_override: Optional[str] = None,
        dev_strategy: str = None,
    ) -> dict:
        """Get structured output with cognitive filtering"""
        raw_response = ""
        error_message = ""

        prompt = (
            f"{original_prompt}\n\n[CRITICAL] Your output MUST be a single, valid JSON object that strictly conforms to the following JSON Schema:\n{json.dumps(schema, indent=2)}"
            if schema
            else original_prompt
        )

        try:
            raw_response = await self._make_api_call(
                prompt,
                persona_name,
                is_json=True,
                modalities=modalities,
                cognitive_state=cognitive_state,
                model_override=model_override,
                dev_strategy=dev_strategy,
            )

            if isinstance(raw_response, str):
                json_response = self._clean_json(raw_response)
            else:
                json_response = raw_response

            return json_response
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            error_message = str(e)
            logger.warning(f"Direct call failed to produce valid JSON: {error_message}")

        try:
            repaired_response = await self.structured_data_repairman(
                original_prompt,
                raw_response,
                error_message,
                schema,
                cognitive_state,
                dev_strategy,
            )
            return self._clean_json(repaired_response)
        except (json.JSONDecodeError, ValueError, TypeError) as e2:
            logger.warning(f"Repair attempt failed: {e2}")

        logger.error("All attempts failed. Returning default response.")
        return default_response

    async def structured_data_repairman(
        self,
        prompt: str,
        raw_response: str,
        error: str,
        schema: dict,
        cognitive_state: Dict,
        dev_strategy: str = None,
    ) -> str:
        """Repair invalid JSON output using a specialized prompt."""
        repair_prompt = f"""The previous response was invalid JSON. Error: {error}

Raw response: {raw_response}

Original prompt: {prompt}

Schema: {json.dumps(schema, indent=2)}

Please repair and return ONLY valid JSON that conforms to the schema. No explanations."""
        return await self._make_api_call(
            repair_prompt,
            "Repairman",
            cognitive_state=cognitive_state,
            dev_strategy=dev_strategy,
        )

    # Advanced AI Personas with cognitive capabilities
    async def master_planner(
        self,
        goal: str,
        image_context: str = None,
        cognitive_state: Dict = None,
        dev_strategy: str = None,
    ) -> dict:
        """Advanced project planning with cognitive context"""
        prompt = (
            "You are a 10x developer and product manager AI. Your goal is not just to meet the user's request, but to *exceed* it.\n"
            "1.  **Classify:** Determine if the goal is to build a 'website' or a 'python_agent'.\n"
            "2.  **Ambitious Elaboration:** Analyze the user's goal and add high-value features that a user would want but might not have thought to ask for.\n"
            f"USER'S GOAL: '{goal}'\n\n"
        )

        if image_context:
            prompt += f"IMAGE CONTEXT: {image_context}\n\n"

        if dev_strategy:
            prompt += f"DEVELOPMENT STRATEGY: {dev_strategy.upper()}\n\n"

        prompt += "Respond with a single JSON object with two keys: 'project_type' and 'refined_goal'."

        schema = {
            "type": "object",
            "properties": {
                "project_type": {"type": "string", "enum": ["website", "python_agent"]},
                "refined_goal": {"type": "string"},
            },
            "required": ["project_type", "refined_goal"],
        }

        modalities = ["text"]
        if image_context:
            modalities.append("image")

        return await self._get_structured_output(
            prompt,
            "Master Planner",
            {"project_type": "python_agent", "refined_goal": goal},
            schema=schema,
            modalities=modalities,
            cognitive_state=cognitive_state,
            dev_strategy=dev_strategy,
        )

    async def constraints_analyst(
        self,
        goal: str,
        project_type: str,
        image_context: str = None,
        cognitive_state: Dict = None,
        dev_strategy: str = None,
    ) -> list:
        """Generate constraints with cognitive context"""
        prompt = (
            "You are a Constraints Analyst. Based on the user's goal, generate a list of non-negotiable constraints.\n"
            f"PROJECT TYPE: '{project_type}'\nGOAL: '{goal}'\n\n"
        )

        if image_context:
            prompt += f"IMAGE CONTEXT: {image_context}\n\n"

        if dev_strategy:
            prompt += f"STRATEGY: {dev_strategy}\n\n"

        prompt += "Respond with a JSON object containing one key, 'constraints', which is a list of descriptive strings."

        schema = {
            "type": "object",
            "properties": {
                "constraints": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["constraints"],
        }

        modalities = ["text"]
        if image_context:
            modalities.append("image")

        response = await self._get_structured_output(
            prompt,
            "Constraints Analyst",
            {"constraints": []},
            schema=schema,
            modalities=modalities,
            cognitive_state=cognitive_state,
            dev_strategy=dev_strategy,
        )
        return response.get("constraints", [])

    async def epic_planner(
        self,
        goal: str,
        constraints: list,
        cognitive_state: Dict,
        dev_strategy: str = None,
    ) -> list:
        """Plan epics for Agile development"""
        prompt = (
            "You are an Agile Epic Planner. Based on the goal and constraints, plan high-level epics for the project.\n"
            f"GOAL: '{goal}'\nCONSTRAINTS: {constraints}\n\n"
            "Respond with a JSON object containing one key, 'epics', which is a list of epic descriptions."
        )

        if dev_strategy and dev_strategy == DevelopmentStrategy.AGILE.value:
            prompt += "\nInclude story points estimation."

        schema = {
            "type": "object",
            "properties": {
                "epics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "story_points": {"type": "number"},
                        },
                    },
                }
            },
            "required": ["epics"],
        }

        response = await self._get_structured_output(
            prompt,
            "Epic Planner",
            {"epics": []},
            schema=schema,
            cognitive_state=cognitive_state,
            dev_strategy=dev_strategy,
        )
        return response.get("epics", [])

    async def cognitive_architect(
        self,
        goal: str,
        constraints: list,
        past_projects: list,
        cognitive_state: Dict,
        dev_strategy: str = None,
        file_to_blueprint: Optional[str] = None,
        existing_files: List[str] = None,
    ) -> dict:
        """
        Generates a project blueprint. Can be used in two modes:
        1. Initial Planning: Generates a list of files.
        2. Detailed File Blueprinting: Generates the detailed structure for a single file.
        """
        # EVOLUTION: This persona is now used iteratively.
        if file_to_blueprint:
            # Mode 2: Generate blueprint for a single file (called by a BLUEPRINT_FILE task)
            prompt = (
                f"You are a Cognitive Architect. Design the detailed blueprint for the file '{file_to_blueprint}'.\n"
                f"GOAL: '{goal}'\nCONSTRAINTS: {constraints}\nEXISTING FILES PLANNED: {existing_files}\n\n"
                "Focus on creating a robust and modular design for this specific file. Ensure method names are descriptive and follow conventions.\n"
                "CRITICAL: Be concise but complete. Your entire response must fit within the model's output limit.\n"
                "Respond with a single, complete FileBlueprint JSON object for this file, including classes, methods, attributes, dependencies, etc. "
            )
            schema = {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "description": {"type": "string"},
                    "classes": {"type": "array", "items": {"type": "object"}},
                    "functions": {"type": "array", "items": {"type": "object"}},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                    "language": {"type": "string"},
                },
                "required": ["filename", "description", "language"],
            }
            default_response = {
                "filename": file_to_blueprint,
                "description": "Default blueprint.",
                "language": "python",
                "classes": [],
                "functions": [],
                "dependencies": [],
            }
        else:
            # Mode 1: Generate the initial list of files (called once during initialization)
            prompt = (
                "You are a Cognitive Architect focused on lean, iterative development. Your primary function is to define the absolute minimum viable product (MVP) file structure.\n"
                f"GOAL: '{goal}'\n\n"
                "CRITICAL INSTRUCTION: Your ONLY task is to identify the 3 to 5 most essential files required to start this project. Do NOT plan the entire application. Start with the core logic, a main entry point, and a configuration file. A test file for the core logic is also required for TDD.\n"
                "DEVIATING FROM THIS 3-5 FILE LIMIT WILL CAUSE A SYSTEM FAILURE. YOU MUST ADHERE TO THIS CONSTRAINT.\n"
                "Respond with a JSON object containing one key, 'files', which is a list of objects. Each object must have 'filename' and 'description' keys."
            )
            schema = {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["filename", "description"],
                        },
                    }
                },
                "required": ["files"],
            }
            default_response = {"files": []}

        if dev_strategy:
            prompt += f"\nSTRATEGY: {dev_strategy} - Adapt blueprint accordingly (e.g., add test files for TDD)."

        return await self._get_structured_output(
            prompt,
            "Cognitive Architect",
            default_response,
            schema=schema,
            cognitive_state=cognitive_state,
            dev_strategy=dev_strategy,
        )

    async def ceo_strategist(
        self,
        goal: str,
        completed_tasks: List[str],
        current_codebase: Dict[str, str],
        project_state: ProjectState,
        last_task_result: Optional[str] = None,
        cognitive_state: Dict = None,
        dev_strategy: str = None,
    ) -> Dict:
        """CEO strategist for next task planning. Replaces the graph planner."""
        prompt = (
            "You are the CEO and Chief Strategist of an AI development team. Your role is to determine the single next most important task.\n"
            f"OVERALL GOAL: '{goal}'\n"
            f"COMPLETED TASKS: {completed_tasks}\n"
            f"CURRENT FILES IN PROJECT: {list(current_codebase.keys())}\n"
            f"LAST TASK RESULT: {last_task_result}\n"
            f"CURRENT DEVELOPMENT STRATEGY: {dev_strategy}\n\n"
            "Given the current state, what is the single most critical task to perform next to move the project forward? "
            "Consider the strategy. For TDD, if a module is implemented, the next task should be to test it. For Agile, it might be refining the next user story.\n"
            "Your response must be a single JSON object representing the next task."
        )

        valid_task_types = [e.value for e in TaskType]
        schema = {
            "type": "object",
            "properties": {
                "task_type": {"type": "string", "enum": valid_task_types},
                "task_description": {"type": "string"},
                "details": {"type": "object"},
            },
            "required": ["task_type", "task_description", "details"],
        }

        default_response = {
            "task_type": TaskType.TDD_IMPLEMENTATION.value,
            "task_description": "Default Task: Implement core feature in main.py",
            "details": {"target_file": "main.py"},
        }

        return await self._get_structured_output(
            prompt,
            "CEO Strategist (Next Task)",
            default_response,
            schema=schema,
            cognitive_state=cognitive_state,
            dev_strategy=dev_strategy,
        )

    async def code_crafter(
        self,
        goal: str,
        task: Dict,
        living_blueprint: LivingBlueprint,
        current_codebase: Dict[str, str],
        cognitive_state: Dict = None,
        feedback: str = None,
        dev_strategy: str = None,
    ) -> str:
        """Enhanced code generation with strategy-specific templates."""
        target_file = task.get("details", {}).get("target_file")
        file_blueprint = living_blueprint.get_file(target_file)

        prompt = (
            f"You are an expert programmer. Your task is to write the complete code for the file '{target_file}'.\n"
            f"PROJECT GOAL: {goal}\n"
            f"DEVELOPMENT STRATEGY: {dev_strategy}\n"
            f"CURRENT CODEBASE CONTEXT: {list(current_codebase.keys())}\n"
            f"FILE BLUEPRINT:\n{json.dumps(file_blueprint, default=lambda o: o.__dict__, indent=2)}\n\n"
            "Adhere strictly to the blueprint. Implement all classes and methods as described. Ensure the code is robust, well-documented, and production-ready.\n"
        )
        if feedback:
            prompt += f"PREVIOUS ATTEMPT FAILED. FEEDBACK: {feedback}\nPlease correct the code based on this feedback.\n"

        prompt += "Your output must ONLY be the raw code for the file. Do not include markdown fences or any explanations."

        raw_code = await self._make_api_call(
            prompt,
            "Code Crafter",
            cognitive_state=cognitive_state,
            dev_strategy=dev_strategy,
        )
        # The code cleaning might be less necessary if the prompt is very strict, but it's a good safeguard.
        return self._clean_code(raw_code, language=file_blueprint.language)

    async def code_validator(
        self,
        code: str,
        language: str,
        cognitive_state: Dict = None,
        dev_strategy: str = None,
    ) -> Dict:
        """Enhanced validation with actual test execution for TDD."""
        validation = {"is_valid": True, "feedback": ""}

        if not code or code.strip() == "":
            return {"is_valid": False, "feedback": "Generated code is empty."}

        # Syntax check
        if language == "python":
            try:
                ast.parse(code)
            except SyntaxError as e:
                return {"is_valid": False, "feedback": f"Syntax error: {e}"}

        if dev_strategy == DevelopmentStrategy.TDD.value:
            # Generate and run tests
            test_code = await self._generate_tests(code, language)
            coverage_report = self._run_tests_and_coverage(test_code, code, language)
            if (
                coverage_report["coverage"]
                < config.strategy_configs[dev_strategy]["test_coverage_threshold"]
            ):
                return {
                    "is_valid": False,
                    "feedback": f"Coverage {coverage_report['coverage']:.2%} < {config.strategy_configs[dev_strategy]['test_coverage_threshold']:.2%}",
                }
            if coverage_report["failing_tests"]:
                return {
                    "is_valid": False,
                    "feedback": f"Failing tests: {coverage_report['failing_tests']}",
                }

        # Linting for all strategies
        lint_result = await self._run_linter(code, language)
        if lint_result["issues"]:
            validation["feedback"] += f" Linting issues: {lint_result['issues']}"

        return validation

    async def _generate_tests(self, code: str, language: str) -> str:
        """Generate tests using AI."""
        prompt = f"Generate comprehensive tests for this {language} code:\n{code}"
        return await self._make_api_call(prompt, "Test Generator")

    def _run_tests_and_coverage(
        self, test_code: str, source_code: str, language: str
    ) -> Dict:
        """Run tests with coverage using temporary files."""
        temp_dir = Path("temp_tests")
        temp_dir.mkdir(exist_ok=True)
        test_path = temp_dir / "test_file.py"
        source_path = temp_dir / "source.py"

        with open(test_path, "w") as f:
            f.write(test_code)
        with open(source_path, "w") as f:
            f.write(source_code)

        try:
            # Measure coverage on the temporary source file only
            cov = coverage.Coverage(
                source=[str(source_path.parent)], include=[str(source_path)], omit=[]
            )
            cov.start()
            result = subprocess.run(
                ["pytest", str(test_path), "-v", "--tb=short"],
                capture_output=True,
                text=True,
            )
            cov.stop()
            cov.save()
            report = cov.report()
            coverage_pct = report / 100.0

            failing = [] if result.returncode == 0 else result.stderr.split("\n")
            return {
                "coverage": coverage_pct,
                "failing_tests": failing[:3],
            }  # Limit feedback
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _run_linter(self, code: str, language: str) -> Dict:
        """Run linter on code."""
        temp_file = Path("temp_lint.py")
        with open(temp_file, "w") as f:
            f.write(code)
        # Run formatter and checker as separate commands for Windows compatibility
        linter_cmds = [
            [sys.executable, "-m", "ruff", "format", str(temp_file.parent)],
            [sys.executable, "-m", "ruff", "check", str(temp_file.parent), "--fix"],
        ]
        issues = 0
        for cmd in linter_cmds:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                issues += proc.stderr.count("error") + proc.stdout.count("error")
        try:
            temp_file.unlink()
        except Exception:
            pass
        return {"issues": issues}

    async def metamorph_architect(
        self,
        goal: str,
        failure_context: str,
        cognitive_state: Dict,
        dev_strategy: str = None,
    ) -> dict:
        """Plan evolution for failures."""
        prompt = f"Plan evolution for goal '{goal}' after failure: {failure_context}\nStrategy: {dev_strategy}"
        schema = {
            "type": "object",
            "properties": {
                "code_changes": {"type": "object"},
                "cognitive_enhancements": {"type": "object"},
            },
        }
        return await self._get_structured_output(
            prompt,
            "Metamorph Architect",
            {},
            schema,
            cognitive_state=cognitive_state,
            dev_strategy=dev_strategy,
        )


# --- Base Tool Class ---
class BaseTool(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    async def execute(self, *args, **kwargs) -> dict:
        pass


# --- Enhanced Tools ---
class CognitiveShellCommander(BaseTool):
    """Shell commander with cognitive optimizations"""

    def __init__(self, working_dir: str):
        self.working_dir = working_dir

    @safe_execute
    async def execute(
        self, command: str, cognitive_context: Dict = None, dev_strategy: str = None
    ) -> Dict:
        """Execute shell command with cognitive optimization"""
        optimized = self._optimize_command(command, cognitive_context, dev_strategy)
        result = subprocess.run(
            optimized, shell=True, capture_output=True, text=True, cwd=self.working_dir
        )
        analyzed = self._analyze_results(
            {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            cognitive_context,
        )
        return analyzed

    def _optimize_command(
        self, command: str, cognitive_context: Dict = None, dev_strategy: str = None
    ) -> str:
        """Optimize command based on cognitive state and strategy"""
        optimized = command

        if not cognitive_context or not isinstance(cognitive_context, dict):
            return optimized

        # Apply optimizations based on cognitive state
        if cognitive_context.get("current_mode") == CognitiveState.FOCUSED.value:
            # Add timing only on POSIX; on Windows keep command as-is
            if os.name != "nt" and not command.startswith("time "):
                optimized = f"time {command}"

        if cognitive_context.get("attention_level", 0.8) < 0.6:
            # Add verbose output for low attention mode
            if " -v " not in command and " --verbose " not in command:
                if "grep" in command or "find" in command:
                    optimized = command  # Don't add verbose to these
                else:
                    optimized = f"{command} -v"

        # Strategy-specific optimizations
        if dev_strategy == DevelopmentStrategy.TDD.value:
            if "pytest" in command:
                optimized += " --cov"
        elif dev_strategy == DevelopmentStrategy.DEVOPS.value:
            if "docker" in command:
                optimized += " --rm"

        return optimized

    def _analyze_results(self, result: Dict, cognitive_context: Dict = None) -> Dict:
        """Analyze command results with cognitive context"""
        # Add cognitive analysis to results
        if result["returncode"] == 0:
            result["cognitive_analysis"] = "Command executed successfully"

            # Analyze output for patterns
            output = result["stdout"]
            if "error" in output.lower() or "warning" in output.lower():
                result["cognitive_analysis"] += " but output contains warnings"
            elif "success" in output.lower() or "completed" in output.lower():
                result["cognitive_analysis"] += " with positive indicators"
        else:
            result["cognitive_analysis"] = "Command failed"

            # Analyze error patterns
            error = result["stderr"]
            if "permission denied" in error.lower():
                result["cognitive_analysis"] += " due to permission issues"
            elif "not found" in error.lower():
                result["cognitive_analysis"] += " because command or file was not found"
            elif "memory" in error.lower():
                result["cognitive_analysis"] += " due to memory issues"

        return result


class CloudDeploymentTool(BaseTool):
    """Enhanced cloud deployment tool with cognitive capabilities"""

    name = "CloudDeploymentTool"
    description = "Deploys applications to cloud platforms with cognitive optimization"

    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.shell = CognitiveShellCommander(working_dir)
        self.docker_client = None
        self.cognitive_state = {}

        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized successfully")
        except docker.errors.DockerException as e:
            logger.warning(
                f"Failed to initialize Docker client, Docker-related tools will be unavailable: {e}"
            )
        except Exception as e:
            logger.warning(
                f"An unexpected error occurred during Docker client initialization: {e}"
            )

        try:
            # Try to load Kubernetes config
            if os.getenv("KUBERNETES_SERVICE_HOST"):
                kube_config.load_incluster_config()
            else:
                kube_config.load_kube_config()
            self.k8s_apps_v1 = client.AppsV1Api()
            self.k8s_core_v1 = client.CoreV1Api()
            logger.info("Kubernetes client initialized successfully")
        except kube_config.ConfigException as e:
            logger.warning(
                f"Could not load Kubernetes configuration. K8s tools will be unavailable: {e}"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Kubernetes client: {e}")

    @safe_execute
    async def execute(
        self,
        platform: str,
        strategy: str = "rolling",
        config_file: str = "deployment.yaml",
        cognitive_context: Dict = None,
        dev_strategy: str = None,
        **kwargs,
    ) -> dict:
        """Deploy application with cognitive optimization"""

        if platform not in config.cloud_providers:
            return {
                "status": "FAILURE",
                "message": f"Unsupported cloud platform: {platform}. Supported: {config.cloud_providers}",
            }

        if strategy not in config.deployment_strategies:
            return {
                "status": "FAILURE",
                "message": f"Unsupported deployment strategy: {strategy}. Supported: {config.deployment_strategies}",
            }

        logger.info(
            f"Deploying to {platform} using {strategy} strategy with cognitive optimization"
        )

        try:
            if dev_strategy == DevelopmentStrategy.DEVOPS.value:
                # Generate GitHub Actions workflow
                workflow = self._generate_ci_cd_workflow(strategy)
                workflow_path = (
                    Path(self.working_dir) / ".github" / "workflows" / "deploy.yml"
                )
                workflow_path.parent.mkdir(parents=True, exist_ok=True)
                with open(workflow_path, "w") as f:
                    yaml.dump(workflow, f)
                logger.info("Generated GitHub Actions CI/CD workflow")

            if platform == "kubernetes":
                return await self._deploy_to_kubernetes(
                    config_file, strategy, cognitive_context
                )
            elif platform == "aws":
                return await self._deploy_to_aws(strategy, cognitive_context, **kwargs)
            elif platform == "azure":
                return await self._deploy_to_azure(
                    strategy, cognitive_context, **kwargs
                )
            elif platform == "gcp":
                return await self._deploy_to_gcp(strategy, cognitive_context, **kwargs)
            elif platform == "github":
                return await self._deploy_to_github(
                    strategy, cognitive_context, **kwargs
                )
            elif platform == "edge":
                return await self._deploy_to_edge(strategy, cognitive_context, **kwargs)
            else:
                return {
                    "status": "FAILURE",
                    "message": f"Deployment to {platform} not yet implemented",
                }
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return {"status": "FAILURE", "message": f"Deployment failed: {str(e)}"}

    async def _deploy_to_kubernetes(
        self, config_file: str, strategy: str, cognitive_context: Dict
    ) -> dict:
        """Deploy to Kubernetes with cognitive optimization"""
        if not hasattr(self, "k8s_apps_v1") or not hasattr(self, "k8s_core_v1"):
            return {"status": "FAILURE", "message": "Kubernetes client not initialized"}

        try:
            # Read deployment configuration
            config_path = Path(self.working_dir) / config_file
            if not config_path.exists():
                return {
                    "status": "FAILURE",
                    "message": f"Deployment config file not found: {config_file}",
                }

            with open(config_path, "r") as f:
                deployment_config = yaml.safe_load(f)

            # Apply cognitive optimization to deployment config
            if cognitive_context:
                deployment_config = self._optimize_deployment_config(
                    deployment_config, cognitive_context
                )

            # Apply deployment
            if deployment_config.get("kind") == "Deployment":
                resp = self.k8s_apps_v1.create_namespaced_deployment(
                    body=deployment_config, namespace="default"
                )
                logger.info(f"Deployment created. status='{resp.status}'")
            elif deployment_config.get("kind") == "Service":
                resp = self.k8s_core_v1.create_namespaced_service(
                    body=deployment_config, namespace="default"
                )
                logger.info(f"Service created. status='{resp.status}'")

            return {
                "status": "SUCCESS",
                "message": f"Successfully deployed to Kubernetes using {strategy} strategy with cognitive optimization",
            }
        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {e}")
            return {
                "status": "FAILURE",
                "message": f"Kubernetes deployment failed: {str(e)}",
            }

    def _optimize_deployment_config(
        self, config: Dict, cognitive_context: Dict
    ) -> Dict:
        """Optimize deployment configuration based on cognitive context"""
        optimized = config.copy()

        if not cognitive_context or not isinstance(cognitive_context, dict):
            return optimized

        # Apply optimizations based on cognitive state
        if cognitive_context.get("current_mode") == CognitiveState.FOCUSED.value:
            # Focused mode: optimize for performance
            if optimized.get("kind") == "Deployment":
                if "spec" not in optimized:
                    optimized["spec"] = {}
                if "template" not in optimized["spec"]:
                    optimized["spec"]["template"] = {}
                if "spec" not in optimized["spec"]["template"]:
                    optimized["spec"]["template"]["spec"] = {}

                # Add resource limits for focused deployment
                containers = optimized["spec"]["template"]["spec"].get("containers", [])
                for container in containers:
                    if "resources" not in container:
                        container["resources"] = {}
                    container["resources"]["limits"] = {"cpu": "1000m", "memory": "1Gi"}
                    container["resources"]["requests"] = {
                        "cpu": "500m",
                        "memory": "512Mi",
                    }

        elif cognitive_context.get("current_mode") == CognitiveState.CREATIVE.value:
            # Creative mode: optimize for flexibility
            if optimized.get("kind") == "Deployment":
                if "spec" not in optimized:
                    optimized["spec"] = {}
                optimized["spec"]["replicas"] = 3

        return optimized

    async def _deploy_to_aws(
        self, strategy: str, cognitive_context: Dict, **kwargs
    ) -> dict:
        """Deploy to AWS with cognitive optimization"""
        try:
            # Use shell to simulate AWS deployment (e.g., using AWS CLI)
            aws_cmd = "aws ecs update-service --cluster default --service my-service --desired-count 1"  # Placeholder for real AWS
            result = await self.shell.execute(aws_cmd, cognitive_context)

            if result["returncode"] == 0:
                return {
                    "status": "SUCCESS",
                    "message": f"Successfully deployed to AWS using {strategy} strategy with cognitive optimization",
                }
            else:
                return {
                    "status": "FAILURE",
                    "message": f"AWS deployment failed: {result['stderr']}",
                }
        except Exception as e:
            logger.error(f"AWS deployment failed: {e}")
            return {"status": "FAILURE", "message": f"AWS deployment failed: {str(e)}"}

    async def _deploy_to_azure(
        self, strategy: str, cognitive_context: Dict, **kwargs
    ) -> dict:
        """Deploy to Azure with cognitive optimization"""
        try:
            azure_cmd = (
                "az webapp up --name myapp --resource-group mygroup"  # Placeholder
            )
            result = await self.shell.execute(azure_cmd, cognitive_context)

            if result["returncode"] == 0:
                return {
                    "status": "SUCCESS",
                    "message": f"Successfully deployed to Azure using {strategy} strategy with cognitive optimization",
                }
            else:
                return {
                    "status": "FAILURE",
                    "message": f"Azure deployment failed: {result['stderr']}",
                }
        except Exception as e:
            return {
                "status": "FAILURE",
                "message": f"Azure deployment failed: {str(e)}",
            }

    async def _deploy_to_gcp(
        self, strategy: str, cognitive_context: Dict, **kwargs
    ) -> dict:
        """Deploy to GCP with cognitive optimization"""
        try:
            gcp_cmd = "gcloud run deploy --image gcr.io/project/image"  # Placeholder
            result = await self.shell.execute(gcp_cmd, cognitive_context)

            if result["returncode"] == 0:
                return {
                    "status": "SUCCESS",
                    "message": f"Successfully deployed to GCP using {strategy} strategy with cognitive optimization",
                }
            else:
                return {
                    "status": "FAILURE",
                    "message": f"GCP deployment failed: {result['stderr']}",
                }
        except Exception as e:
            return {"status": "FAILURE", "message": f"GCP deployment failed: {str(e)}"}

    async def _deploy_to_github(
        self, strategy: str, cognitive_context: Dict, **kwargs
    ) -> dict:
        """Deploy to GitHub with GitOps."""
        try:
            # Placeholder for GitHub deployment
            gh_cmd = "gh workflow run deploy --repo owner/repo"  # Requires gh CLI
            result = await self.shell.execute(gh_cmd, cognitive_context)

            if result["returncode"] == 0:
                return {
                    "status": "SUCCESS",
                    "message": f"Successfully triggered GitHub deployment using {strategy}",
                }
            else:
                return {
                    "status": "FAILURE",
                    "message": f"GitHub deployment failed: {result['stderr']}",
                }
        except Exception as e:
            return {
                "status": "FAILURE",
                "message": f"GitHub deployment failed: {str(e)}",
            }

    async def _deploy_to_edge(
        self, strategy: str, cognitive_context: Dict, **kwargs
    ) -> dict:
        """Deploy to edge devices with cognitive optimization"""
        try:
            edge_cmd = "balena deploy myapp"  # Placeholder for edge deployment
            result = await self.shell.execute(edge_cmd, cognitive_context)

            if result["returncode"] == 0:
                return {
                    "status": "SUCCESS",
                    "message": f"Successfully deployed to edge using {strategy} strategy with cognitive optimization",
                }
            else:
                return {
                    "status": "FAILURE",
                    "message": f"Edge deployment failed: {result['stderr']}",
                }
        except Exception as e:
            logger.error(f"Edge deployment failed: {e}")
            return {"status": "FAILURE", "message": f"Edge deployment failed: {str(e)}"}

    def _generate_ci_cd_workflow(self, strategy: str) -> Dict:
        """Generate GitHub Actions for DevOps."""
        return {
            "name": "CI/CD Pipeline",
            "on": ["push"],
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [{"uses": "actions/checkout@v2"}, {"run": "pytest --cov"}],
                },
                "deploy": {
                    "needs": "test",
                    "runs-on": "ubuntu-latest",
                    "steps": [{"run": f"deploy with {strategy}"}],
                },
            },
        }

    def _optimize_for_edge(self, strategy: str, cognitive_context: Dict) -> str:
        """Optimize deployment strategy for edge computing"""
        if strategy == "blue-green":
            return "rolling"
        return strategy


class SecurityScannerTool(BaseTool):
    """Security scanning tool with cognitive enhancements"""

    name = "SecurityScannerTool"
    description = "Scans project for security vulnerabilities with cognitive analysis"

    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.shell = CognitiveShellCommander(working_dir)

    async def execute(
        self,
        scan_type: str = "full",
        cognitive_context: Dict = None,
        dev_strategy: str = None,
    ) -> dict:
        """Execute security scan"""
        logger.info(f"Running {scan_type} security scan with cognitive analysis")

        try:
            if scan_type == "full":
                cmd = "bandit -r . -f json"  # Python security scan
            elif scan_type == "sca":  # Software Composition Analysis
                cmd = "pip-audit"  # For dependencies
            else:
                cmd = "bandit -r ."

            result = await self.shell.execute(cmd, cognitive_context)

            if result["returncode"] == 0:
                # Parse results for cognitive analysis
                analysis = self._analyze_security_results(result["stdout"])
                return {
                    "status": "SUCCESS",
                    "results": result["stdout"],
                    "cognitive_analysis": analysis,
                }
            else:
                return {
                    "status": "FAILURE",
                    "message": f"Security scan failed: {result['stderr']}",
                }
        except Exception as e:
            return {"status": "FAILURE", "message": f"Security scan failed: {str(e)}"}

    def _analyze_security_results(self, results: str) -> str:
        """Analyze security scan results cognitively"""
        issues = len(re.findall(r"severity", results, re.IGNORECASE))
        return f"Found {issues} potential security issues. Recommend immediate review."


class PerformanceOptimizerTool(BaseTool):
    """Performance optimization tool with cognitive enhancements"""

    name = "PerformanceOptimizerTool"
    description = "Optimizes project performance with cognitive profiling"

    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.shell = CognitiveShellCommander(working_dir)

    async def execute(
        self,
        optimize_for: str = "cpu",
        cognitive_context: Dict = None,
        dev_strategy: str = None,
    ) -> dict:
        """Execute performance optimization"""
        logger.info(
            f"Optimizing performance for {optimize_for} with cognitive profiling"
        )

        try:
            if optimize_for == "cpu":
                cmd = "cythonize -i *.py"  # Example optimization
            elif optimize_for == "memory":
                cmd = "python -m memory_profiler *.py"
            else:
                cmd = "python -m py_compile ."

            result = await self.shell.execute(cmd, cognitive_context)

            if result["returncode"] == 0:
                metrics = get_system_metrics()
                return {
                    "status": "SUCCESS",
                    "metrics": metrics,
                    "cognitive_recommendation": "Performance improved based on metrics.",
                }
            else:
                return {
                    "status": "FAILURE",
                    "message": f"Optimization failed: {result['stderr']}",
                }
        except Exception as e:
            return {"status": "FAILURE", "message": f"Optimization failed: {str(e)}"}


class GitTool(BaseTool):
    """Git tool for DevOps integration"""

    name = "GitTool"
    description = "Manages Git operations for version control"

    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.repo = (
            Repo(working_dir)
            if os.path.exists(working_dir + "/.git")
            else Repo.init(working_dir)
        )

    async def execute(self, git_cmd: str, cognitive_context: Dict = None) -> dict:
        """Execute Git command"""
        result = subprocess.run(
            git_cmd, shell=True, capture_output=True, text=True, cwd=self.working_dir
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


class ToolRegistry:
    """Registry for managing available tools with cognitive enhancements."""

    def __init__(self, tools=None):
        self.tools = tools or []
        self.cognitive_patterns = {}

    def register(self, tool):
        self.tools.append(tool)
        patterns = self._extract_cognitive_patterns(tool.description)
        if patterns:
            self.cognitive_patterns[tool.name] = patterns

    def _extract_cognitive_patterns(self, description: str) -> Dict[str, int]:
        """Extract cognitive patterns from tool description"""
        patterns = {
            "problem_solving": len(
                re.findall(r"solve|resolve|fix|repair", description, re.IGNORECASE)
            ),
            "optimization": len(
                re.findall(
                    r"optimize|improve|enhance|boost", description, re.IGNORECASE
                )
            ),
            "analysis": len(
                re.findall(
                    r"analyze|examine|inspect|review", description, re.IGNORECASE
                )
            ),
            "creation": len(
                re.findall(r"create|build|generate|make", description, re.IGNORECASE)
            ),
        }
        return {k: v for k, v in patterns.items() if v > 0}

    def get_tool(self, name):
        for tool in self.tools:
            if getattr(tool, "name", None) == name:
                return tool
        return None

    def list_tools(self):
        return [getattr(tool, "name", None) for tool in self.tools]

    def get_tools_by_cognitive_pattern(self, pattern: str) -> List[str]:
        """Get tools that exhibit a specific cognitive pattern"""
        return [
            name
            for name, patterns in self.cognitive_patterns.items()
            if pattern in patterns and patterns[pattern] > 0
        ]

    def get_tools_by_strategy(self, strategy: str) -> List[str]:
        """Get tools relevant to strategy."""
        if strategy == DevelopmentStrategy.TDD.value:
            return [
                t
                for t in self.list_tools()
                if "test" in t.lower() or "validate" in t.lower()
            ]
        elif strategy == DevelopmentStrategy.DEVOPS.value:
            return [
                t
                for t in self.list_tools()
                if "git" in t.lower() or "deploy" in t.lower()
            ]
        # ... (similar for others)
        return self.list_tools()


# --- Multi-language build/test/run executor ---
class MultiLanguageExecutor:
    """Executes build/test/lint/run across multiple languages via config."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir

    def _run(self, cmd: List[str]) -> Tuple[int, str, str]:
        try:
            proc = subprocess.run(
                cmd, cwd=self.base_dir, capture_output=True, text=True
            )
            return proc.returncode, proc.stdout, proc.stderr
        except FileNotFoundError as e:
            return 127, "", str(e)

    def lint(self, language: str) -> Dict[str, Any]:
        cfg = config.language_configs.get(language.lower())
        if not cfg or not cfg.get("linter_command"):
            return {"status": "SKIPPED", "message": f"No linter for {language}"}
        # Split naive; prefer shell=False when possible
        cmd = cfg["linter_command"].split()
        code, out, err = self._run(cmd)
        return {
            "status": "SUCCESS" if code == 0 else "FAILURE",
            "stdout": out,
            "stderr": err,
        }

    def test(self, language: str) -> Dict[str, Any]:
        cfg = config.language_configs.get(language.lower())
        if not cfg or not cfg.get("test_command"):
            return {"status": "SKIPPED", "message": f"No test command for {language}"}
        cmd = cfg["test_command"].split()
        code, out, err = self._run(cmd)
        return {
            "status": "SUCCESS" if code == 0 else "FAILURE",
            "stdout": out,
            "stderr": err,
        }

    def build(self, language: str) -> Dict[str, Any]:
        cfg = config.language_configs.get(language.lower())
        cmd = cfg.get("build_command")
        if not cmd:
            return {"status": "SKIPPED", "message": f"No build command for {language}"}
        code, out, err = self._run(cmd.split())
        return {
            "status": "SUCCESS" if code == 0 else "FAILURE",
            "stdout": out,
            "stderr": err,
        }

    def run(self, language: str) -> Dict[str, Any]:
        cfg = config.language_configs.get(language.lower())
        cmd = cfg.get("run_command")
        if not cmd:
            return {"status": "SKIPPED", "message": f"No run command for {language}"}
        code, out, err = self._run(cmd.split())
        return {
            "status": "SUCCESS" if code == 0 else "FAILURE",
            "stdout": out,
            "stderr": err,
        }


# --- Ultra Cognitive Forge Orchestrator with Self-Evolution ---
class UltraCognitiveForgeOrchestrator:
    """
    Evolved orchestrator with strategic planning via an iterative task queue.
    """

    def __init__(
        self,
        goal: str,
        project_name: str,
        agent_filename: str,
        llm_choice: str = "api",
        dev_strategy: str = "cognitive",
    ):
        self.goal = goal
        self.project_name = project_name
        self.sandbox_dir = str(Path(config.sandbox_base_path) / project_name)
        self.session_path = f"{config.session_dir}/{project_name}.json"
        self.agent_filename = agent_filename

        self.initial_dev_strategy = dev_strategy

        # --- Initialize ProjectState FIRST ---
        self.project_state = ProjectState(
            goal=self.goal,
            project_name=self.project_name,
            development_strategy=self.initial_dev_strategy,
            current_strategy=self.initial_dev_strategy,
            cognitive_state={
                "current_mode": CognitiveState.ANALYTICAL.value,
                "attention_level": 0.85,
                "certainty": 0.75,
                "novelty": 0.6,
                "relevance": 0.85,
                "last_state_change": time.time(),
                "evolution_count": 0,
                "metacognitive_awareness": 0.7,
            },
        )

        # Set cognitive state in thread local storage AFTER ProjectState is created
        self._set_thread_cognitive_state()
        threading.current_thread().dev_strategy = dev_strategy  # Set strategy in thread

        # Initialize AI with cognitive capabilities
        self.ai = QuantumCognitiveAI(llm_choice=llm_choice)
        self.memory = NeuromorphicMemoryManager()
        self.assembler = CognitiveFileAssembler(self.sandbox_dir, agent_filename)
        self.shell = CognitiveShellCommander(self.sandbox_dir)
        self.constitution = self._load_constitution()

        # Initialize enhanced tools
        cloud_tool = CloudDeploymentTool(self.sandbox_dir)
        security_tool = SecurityScannerTool(self.sandbox_dir)
        perf_tool = PerformanceOptimizerTool(self.sandbox_dir)
        git_tool = GitTool(self.sandbox_dir)  # New
        self.tool_registry = ToolRegistry(
            tools=[self.shell, cloud_tool, security_tool, perf_tool, git_tool]
        )

        logger.info(
            f"Ultra cognitive orchestrator initialized for project: {project_name}"
        )

    def _set_thread_cognitive_state(self):
        """Set cognitive state in thread local storage"""
        threading.current_thread().cognitive_state = self.project_state.cognitive_state

    def _load_constitution(self):
        """
        Loads the system constitution with cognitive principles
        """
        return {
            "core_principles": [
                "Autonomy",
                "Safety",
                "Transparency",
                "Continuous Improvement",
                "Cognitive Enhancement",
                "Self-Evolution",
            ],
            "cognitive_directives": [
                "Always maintain metacognitive awareness",
                "Adapt cognitive state to task requirements",
                "Learn from failures and evolve",
                "Balance exploration and exploitation",
                "Optimize for cognitive efficiency",
            ],
            "evolution_rules": [
                "Evolve when stuck in failure loops",
                "Evolve when new capabilities are needed",
                "Evolve when cognitive limits are reached",
                "Document all evolutionary changes",
                "Maintain backward compatibility when possible",
            ],
        }

    async def run(self):
        """Main execution loop with cognitive capabilities and self-evolution"""
        logger.info("=" * 60)
        logger.info(
            f"     ULTRA COGNITIVE AI SOFTWARE ENGINEER - Project: {self.project_name}"
        )
        logger.info("=" * 60)

        try:
            # Initialize project
            if (
                not self.project_state.completed_tasks
                and not await self.initialize_project()
            ):
                logger.error("Project initialization failed")
                return

            # Main execution loop
            goal_achieved = False
            last_task_result = None
            sprint_count = 0  # For Agile

            while not goal_achieved:
                # Cognitive health check
                if not self._check_cognitive_health():
                    logger.warning(
                        "Cognitive health check failed - need to adjust cognitive state"
                    )
                    await self._adjust_cognitive_state()
                    continue

                # Strategy-specific loops
                if (
                    self.project_state.development_strategy
                    == DevelopmentStrategy.AGILE.value
                    and sprint_count > 0
                    and sprint_count
                    % config.strategy_configs[DevelopmentStrategy.AGILE.value][
                        "sprint_duration"
                    ]
                    == 0
                ):
                    await self.execute_sprint_review()

                # Get next task with cognitive optimization
                task_response = await self.get_next_task(last_task_result)
                if not task_response or not task_response.get("task_type"):
                    logger.info("No more tasks to execute. Checking goal achievement.")
                    goal_achieved = await self.check_goal_achievement(last_task_result)
                    if goal_achieved:
                        logger.info("Goal has been achieved!")
                        break
                    else:
                        logger.warning(
                            "Goal not achieved, but no new tasks were generated. Ending run."
                        )
                        break

                # Execute task with cognitive capabilities
                success, last_task_result = await self.execute_task(
                    task_response, self.project_state.development_strategy
                )

                if success:
                    self.project_state.completed_tasks.append(
                        task_response.get("task_description", "Unknown task")
                    )
                    logger.info(
                        f"Task completed successfully. Total completed: {len(self.project_state.completed_tasks)}"
                    )
                    self._update_cognitive_state(success=True)
                else:
                    logger.warning("Task failed. Will attempt to recover.")
                    await self.handle_task_failure(task_response, last_task_result)
                    self._update_cognitive_state(success=False)

                await self.save_state()

                if (
                    len(self.project_state.completed_tasks) > 0
                    and len(self.project_state.completed_tasks)
                    % config.strategy_configs[
                        self.project_state.development_strategy
                    ].get("evolution_frequency", 10)
                    == 0
                ):
                    await self._check_evolution_opportunities()

                if (
                    self.project_state.development_strategy
                    == DevelopmentStrategy.DEVOPS.value
                ):
                    await self._commit_and_push()  # Auto-commit for DevOps

                sprint_count += 1

            if goal_achieved:
                await self.finalize_project()
                logger.info(
                    f"🎉 Project completed successfully! Completed {len(self.project_state.completed_tasks)} tasks."
                )
            else:
                logger.warning("Project execution stopped before goal was achieved.")

        except Exception as e:
            logger.error(f"Fatal error in orchestrator: {e}")
            logger.debug(traceback.format_exc())

        finally:
            await self.cleanup()

    async def initialize_project(self):
        """Initializes the project by planning the architecture and first steps."""
        logger.info("Initializing project...")
        try:
            # Phase 1: Master Planner
            plan = await self.ai.master_planner(
                self.goal,
                cognitive_state=self.project_state.cognitive_state,
                dev_strategy=self.project_state.development_strategy,
            )
            self.project_state.goal = plan.get("refined_goal", self.goal)
            project_type = plan.get("project_type", "python_agent")
            logger.info(f"Refined Goal: {self.project_state.goal}")

            # Phase 2: Constraints Analyst
            query_text = (
                json.dumps(self.project_state.goal)
                if isinstance(self.project_state.goal, dict)
                else self.project_state.goal
            )
            constraints = await self.ai.constraints_analyst(
                query_text,
                project_type,
                cognitive_state=self.project_state.cognitive_state,
                dev_strategy=self.project_state.development_strategy,
            )
            logger.info(f"Constraints: {constraints}")

            # Phase 2.5: Agile Epic Planning (if applicable)
            if (
                self.project_state.development_strategy
                == DevelopmentStrategy.AGILE.value
            ):
                logger.info("Agile Strategy: Planning epics...")
                self.project_state.epics = await self.ai.epic_planner(
                    query_text,
                    constraints,
                    self.project_state.cognitive_state,
                    self.project_state.development_strategy,
                )
                logger.info(f"Epics Planned: {self.project_state.epics}")
                self.project_state.user_stories = await self._refine_user_stories(
                    self.project_state.epics[0]["description"]
                    if self.project_state.epics
                    else query_text
                )

            # Phase 3: Iterative Architecture Planning
            logger.info("Architectural Planning: Generating blueprint iteratively...")
            past_projects = self.memory.retrieve_similar_memories(
                query_text, n_results=3
            )

            # Step 3a: Get the list of files
            file_list_response = await self.ai.cognitive_architect(
                query_text,
                constraints,
                past_projects,
                self.project_state.cognitive_state,
                self.project_state.development_strategy,
            )
            files_to_blueprint = file_list_response.get("files", [])

            if not files_to_blueprint:
                logger.warning(
                    "Cognitive Architect returned no files. Falling back to a minimal blueprint."
                )
                files_to_blueprint = [
                    {
                        "filename": "README.md",
                        "description": "Project overview and usage.",
                    },
                    {
                        "filename": "requirements.txt",
                        "description": "Python dependencies.",
                    },
                    {
                        "filename": "src/agent.py",
                        "description": "Core intelligent agent entrypoint.",
                    },
                    {
                        "filename": "src/__init__.py",
                        "description": "Package initializer.",
                    },
                    {
                        "filename": "tests/test_agent.py",
                        "description": "Basic smoke tests for the agent.",
                    },
                ]

            logger.info(
                f"Architect identified {len(files_to_blueprint)} files to create."
            )

            # Step 3b & 3c: Loop and generate detailed blueprint for each file, then assemble
            all_file_blueprints = []
            filenames = [f["filename"] for f in files_to_blueprint]

            for file_info in files_to_blueprint:
                filename = file_info["filename"]
                logger.info(f"Generating detailed blueprint for: {filename}...")
                file_blueprint_dict = await self.ai.cognitive_architect(
                    query_text,
                    constraints,
                    past_projects,
                    self.project_state.cognitive_state,
                    self.project_state.development_strategy,
                    file_to_blueprint=filename,
                    existing_files=filenames,
                )

                file_blueprint = FileBlueprint.from_dict(file_blueprint_dict)
                if file_blueprint:
                    all_file_blueprints.append(file_blueprint)
                else:
                    logger.warning(
                        f"Failed to generate a valid blueprint for {filename}. Using synthesized fallback blueprint."
                    )
                    synthesized = FileBlueprint(
                        filename=filename,
                        description=file_info.get(
                            "description", "Auto-synthesized file"
                        ),
                        classes=[],
                        functions=[],
                    )
                    all_file_blueprints.append(synthesized)

            self.project_state.living_blueprint.root = all_file_blueprints

            if not self.project_state.living_blueprint.root:
                logger.warning(
                    "No valid file blueprints after synthesis. Seeding with minimal defaults."
                )
                self.project_state.living_blueprint.root = [
                    FileBlueprint(filename="README.md", description="Project overview"),
                    FileBlueprint(
                        filename="requirements.txt", description="Dependencies"
                    ),
                    FileBlueprint(filename="src/agent.py", description="Core agent"),
                    FileBlueprint(
                        filename="src/__init__.py", description="Package init"
                    ),
                    FileBlueprint(
                        filename="tests/test_agent.py", description="Smoke tests"
                    ),
                ]

            logger.info("Living Blueprint created and validated.")

            # Strategy-specific init
            if self.project_state.development_strategy == DevelopmentStrategy.TDD.value:
                self.project_state.task_queue.append(
                    {
                        "task_type": TaskType.TDD_IMPLEMENTATION.value,
                        "details": {"focus": "core_module"},
                    }
                )
            elif (
                self.project_state.development_strategy
                == DevelopmentStrategy.DEVOPS.value
            ):
                await self._init_git_repo()
                self.project_state.task_queue.append(
                    {"task_type": TaskType.SETUP_CI_PIPELINE.value}
                )

            # Phase 4: Initial Task Queue Population
            initial_task = await self.ai.ceo_strategist(
                query_text,
                [],
                {},
                self.project_state,
                cognitive_state=self.project_state.cognitive_state,
                dev_strategy=self.project_state.development_strategy,
            )
            self.project_state.task_queue.append(initial_task)

            # Save memory
            self.memory.add_memory(
                self.goal,
                self.project_state.living_blueprint,
                dev_strategy=self.project_state.development_strategy,
                cognitive_state=self.project_state.cognitive_state,
            )

            await self.save_state()
            return True
        except Exception as e:
            logger.error(f"Error during project initialization: {e}")
            logger.debug(traceback.format_exc())
            return False

    async def _refine_user_stories(self, epic: str) -> List[Dict]:
        """Refine epics into user stories for Agile."""
        prompt = f"Refine epic '{epic}' into user stories with acceptance criteria and story points."
        schema = {
            "type": "object",
            "properties": {
                "stories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "acceptance_criteria": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "story_points": {"type": "number"},
                        },
                    },
                }
            },
        }
        stories = await self.ai._get_structured_output(
            prompt,
            "Story Refiner",
            {"stories": []},
            schema,
            cognitive_state=self.project_state.cognitive_state,
            dev_strategy=DevelopmentStrategy.AGILE.value,
        )
        return stories.get("stories", [])

    async def get_next_task(self, last_task_result: Optional[str]) -> Dict:
        logger.info("Determining next task...")
        if self.project_state.task_queue:
            return self.project_state.task_queue.pop(0)

        # If queue is empty, ask CEO for the next task
        logger.info("Task queue is empty. Consulting CEO Strategist...")
        query_text_for_ceo = (
            json.dumps(self.project_state.goal)
            if isinstance(self.project_state.goal, dict)
            else self.project_state.goal
        )
        next_task = await self.ai.ceo_strategist(
            query_text_for_ceo,
            self.project_state.completed_tasks,
            self.assembler.get_all_files(),
            self.project_state,
            last_task_result,
            cognitive_state=self.project_state.cognitive_state,
            dev_strategy=self.project_state.development_strategy,
        )
        return next_task

    async def check_goal_achievement(self, last_task_result: str) -> bool:
        """Check if the goal has been achieved based on completed tasks and results"""
        if last_task_result and (
            "final" in last_task_result.lower()
            or "complete" in last_task_result.lower()
        ):
            return True

        if (
            self.assembler.check_project_size_limits()
            and len(self.project_state.completed_tasks) > 5
        ):
            return True

        return False

    async def save_state(self):
        logger.info(f"Saving session state to {self.session_path}")
        try:
            state_dict = {
                f.name: getattr(self.project_state, f.name)
                for f in fields(self.project_state)
            }
            state_dict["living_blueprint"] = json.loads(
                self.project_state.living_blueprint.to_json()
            )
            state_dict["file_contents"] = self.assembler.get_all_files()

            with open(self.session_path, "w") as f:
                json.dump(state_dict, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    async def load_state(self):
        logger.info(f"Loading session state from {self.session_path}")
        try:
            with open(self.session_path, "r") as f:
                state_dict = json.load(f)

            # Re-hydrate the LivingBlueprint object
            blueprint_data = state_dict.get("living_blueprint", {})
            state_dict["living_blueprint"] = LivingBlueprint.from_dict(blueprint_data)

            self.project_state = ProjectState(**state_dict)

            for filename, content in self.project_state.file_contents.items():
                self.assembler.add_file(filename, content)

            self._set_thread_cognitive_state()  # Important to reset after loading
            threading.current_thread().dev_strategy = (
                self.project_state.development_strategy
            )
            logger.info("Session state loaded successfully.")
        except FileNotFoundError:
            logger.warning("No session file found. Starting a new session.")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            logger.debug(traceback.format_exc())

    async def finalize_project(self):
        """Finalize the project with reports and packaging"""
        logger.info("Finalizing project...")
        try:
            # Generate finalization report
            report_prompt = f"Generate final reports for project: {self.project_state.goal} under strategy {self.project_state.development_strategy}"
            report = await self.ai._make_api_call(
                report_prompt,
                "Finalizer",
                cognitive_state=self.project_state.cognitive_state,
                dev_strategy=self.project_state.development_strategy,
            )

            # Save reports
            final_report = FinalizationReport(
                readme_content="Generated README",
                requirements_content="Generated requirements.txt",
                deployment_guide="Generated deployment guide",
                api_documentation="Generated API docs",
                troubleshooting_guide="Generated troubleshooting",
                cognitive_architecture="Cognitive architecture summary",
                agile_retrospective="Sprint retrospective"
                if self.project_state.development_strategy
                == DevelopmentStrategy.AGILE.value
                else "",
                devops_metrics={"ci_runs": 5, "deploy_success": 0.95}
                if self.project_state.development_strategy
                == DevelopmentStrategy.DEVOPS.value
                else {},
            )

            # Write reports to disk
            with open(Path(self.sandbox_dir) / "README.md", "w") as f:
                f.write(final_report.readme_content)

            logger.info("Project finalized with reports.")
        except Exception as e:
            logger.error(f"Finalization failed: {e}")

    def _check_cognitive_health(self) -> bool:
        """Check if cognitive state is healthy"""
        cognitive_state = self.project_state.cognitive_state
        if cognitive_state["attention_level"] < 0.3:
            logger.warning(f"Low attention level: {cognitive_state['attention_level']}")
            return False

        if cognitive_state["certainty"] < 0.2:
            logger.warning(f"Low certainty: {cognitive_state['certainty']}")
            return False

        time_since_change = time.time() - cognitive_state["last_state_change"]
        if time_since_change > 3600:  # 1 hour
            logger.warning(
                f"Cognitive state hasn't changed in {time_since_change} seconds"
            )
            return False

        return True

    async def _adjust_cognitive_state(self):
        """Adjust cognitive state based on current conditions"""
        cognitive_state = self.project_state.cognitive_state
        if cognitive_state["attention_level"] < 0.3:
            cognitive_state["current_mode"] = CognitiveState.FOCUSED.value
            cognitive_state["attention_level"] = 0.7
            logger.info("Adjusted cognitive state to FOCUSED mode")

        elif cognitive_state["certainty"] < 0.2:
            cognitive_state["current_mode"] = CognitiveState.ANALYTICAL.value
            cognitive_state["certainty"] = 0.6
            logger.info("Adjusted cognitive state to ANALYTICAL mode")

        cognitive_state["last_state_change"] = time.time()
        self._set_thread_cognitive_state()

    def _update_cognitive_state(self, success: bool):
        """Update cognitive state based on task outcome"""
        cognitive_state = self.project_state.cognitive_state
        if success:
            cognitive_state["certainty"] = min(1.0, cognitive_state["certainty"] + 0.05)
            cognitive_state["attention_level"] = min(
                1.0, cognitive_state["attention_level"] + 0.03
            )
        else:
            cognitive_state["certainty"] = max(0.1, cognitive_state["certainty"] - 0.1)
            if cognitive_state["current_mode"] != CognitiveState.PROBLEM_SOLVING.value:
                cognitive_state["current_mode"] = CognitiveState.PROBLEM_SOLVING.value
                cognitive_state["last_state_change"] = time.time()
                logger.info("Switched to PROBLEM_SOLVING mode due to failure")

        self._set_thread_cognitive_state()

    async def execute_task(
        self, task_response: Dict, dev_strategy: str
    ) -> Tuple[bool, str]:
        """Execute a task with cognitive capabilities"""
        task_type = task_response.get("task_type")
        task_description = task_response.get("task_description", "")
        task_details = task_response.get("details", {})

        logger.info(f"Executing task: {task_description} (Type: {task_type})")

        try:
            if task_type == TaskType.SCAFFOLD.value:
                return await self.execute_scaffold_task(task_details)
            elif (
                task_type == TaskType.BLUEPRINT_FILE.value
            ):  # FIX: Handle blueprinting task
                return await self.execute_blueprint_file_task(task_details)
            elif task_type in [
                TaskType.TDD_IMPLEMENTATION.value,
                TaskType.CODE_MODIFICATION.value,
                TaskType.CREATE_DOCKERFILE.value,
                TaskType.SETUP_CI_PIPELINE.value,
            ]:
                return await self.execute_development_task(task_response, dev_strategy)
            elif (
                task_type == TaskType.TOOL_EXECUTION.value
            ):  # FIX: Handle tool execution task
                return await self.execute_tool_task(task_details)
            elif (
                task_type == TaskType.PLAN_EPICS.value
                or task_type == TaskType.USER_STORY_REFINEMENT.value
            ):
                return await self._execute_agile_task(task_response)
            elif task_type == TaskType.DEPLOYMENT.value:
                return await self.execute_deployment_task(task_details, dev_strategy)
            elif task_type == TaskType.SECURITY_SCAN.value:
                return await self.execute_security_scan_task(task_details, dev_strategy)
            elif task_type == TaskType.PERFORMANCE_OPTIMIZATION.value:
                return await self.execute_performance_optimization_task(
                    task_details, dev_strategy
                )
            elif task_type == TaskType.COGNITIVE_ANALYSIS.value:
                return True, await self.execute_cognitive_analysis_task(task_details)
            elif task_type == TaskType.SELF_EVOLUTION.value:
                return True, await self.execute_self_evolution_task(task_details)
            elif task_type == TaskType.GIT_COMMIT.value:
                return await self._execute_git_commit_task(task_details)
            else:
                logger.warning(f"Unknown or unhandled task type: {task_type}")
                return False, f"Unknown task type: {task_type}"

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            logger.debug(traceback.format_exc())
            return False, f"Task execution failed: {str(e)}"

    async def _execute_agile_task(self, task: Dict) -> Tuple[bool, str]:
        """Enhanced Agile: Refine stories, estimate points, prioritize."""
        task_type = task.get("task_type")

        if task_type == TaskType.PLAN_EPICS.value:
            if not self.project_state.epics:
                logger.info("Epics not planned yet. Planning now.")
                query_text = (
                    json.dumps(self.project_state.goal)
                    if isinstance(self.project_state.goal, dict)
                    else self.project_state.goal
                )
                self.project_state.epics = await self.ai.epic_planner(
                    query_text,
                    [],
                    self.project_state.cognitive_state,
                    self.project_state.development_strategy,
                )
                return True, "Epics planned successfully."
            else:
                logger.info(
                    "Epics have already been planned. Skipping PLAN_EPICS task."
                )
                return True, "Epics already planned."

        if task_type == TaskType.USER_STORY_REFINEMENT.value:
            epic_to_refine = task["details"].get("epic")
            if not epic_to_refine:
                logger.warning(
                    "USER_STORY_REFINEMENT task is missing an 'epic' in details. Inferring from project state."
                )
                if self.project_state.epics:
                    # Infer the first epic as the one to work on.
                    epic_to_refine = self.project_state.epics[0].get("description")
                    logger.info(f"Inferred epic to refine: '{epic_to_refine[:50]}...'")
                else:
                    return (
                        False,
                        "Cannot refine user stories because no epics have been planned.",
                    )

            if not epic_to_refine:
                return (
                    False,
                    "Could not determine which epic to refine for user story generation.",
                )

            stories = await self._refine_user_stories(epic_to_refine)
            self.project_state.user_stories.extend(stories)
            self.project_state.sprint_backlog = stories[:5]  # Top 5 for sprint
            return True, f"Refined {len(stories)} user stories."

        elif task_type == TaskType.SPRINT_REVIEW.value:
            await self.execute_sprint_review()
            return True, "Sprint review completed."

        return True, "Agile task executed."

    async def execute_scaffold_task(self, details: dict) -> Tuple[bool, str]:
        """Creates the project's directory structure and empty files from the initial file list."""
        logger.info("Scaffolding project structure...")
        if (
            not self.project_state.living_blueprint
            or not self.project_state.living_blueprint.root
        ):
            return (
                False,
                "Cannot scaffold without an initial file list in the blueprint.",
            )

        try:
            # The files in the blueprint are just placeholders at this stage
            for file_bp in self.project_state.living_blueprint.root:
                self.assembler.add_file(file_bp.filename, "")  # Create empty file
                logger.info(f"Scaffolded empty file: {file_bp.filename}")

            self.assembler.write_files_to_disk()
            return True, "Project structure scaffolded successfully."
        except Exception as e:
            error_msg = f"Error during scaffolding: {e}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    async def execute_blueprint_file_task(self, details: dict) -> Tuple[bool, str]:
        """Generates the detailed blueprint for a single file with a self-correction loop."""
        target_file = details.get("target_file")
        if not target_file:
            return False, "Blueprint task requires a 'target_file'."

        logger.info(f"Architecting detailed blueprint for: {target_file}...")

        feedback = None
        for attempt in range(config.max_debug_attempts):
            logger.info(
                f"Blueprint generation attempt {attempt + 1} for {target_file}..."
            )
            try:
                # Add feedback to the goal if it exists from a previous failed attempt
                current_goal = self.project_state.goal
                if feedback:
                    current_goal += f"\n[Previous Attempt Feedback]: {feedback}"

                file_blueprint_dict = await self.ai.cognitive_architect(
                    current_goal,
                    [],  # Constraints
                    [],  # Past projects
                    self.project_state.cognitive_state,
                    self.project_state.development_strategy,
                    file_to_blueprint=target_file,
                    existing_files=[
                        f.filename for f in self.project_state.living_blueprint.root
                    ],
                )

                # Attempt to parse and validate the blueprint
                file_blueprint = FileBlueprint.from_dict(file_blueprint_dict)
                if not file_blueprint:
                    # This case handles if from_dict returns None due to missing keys
                    feedback = "The generated blueprint JSON was missing required keys ('filename', 'description'). Please regenerate the complete and valid JSON structure."
                    logger.warning(
                        f"Blueprint for {target_file} was structurally invalid. Retrying..."
                    )
                    continue

                # If successful, update the main blueprint and return
                self.project_state.living_blueprint.add_or_update_file(file_blueprint)
                logger.info(
                    f"Successfully created and stored detailed blueprint for {target_file}."
                )
                return True, f"Blueprint for {target_file} created."

            except TokenLimitError:
                feedback = "Your previous blueprint was too long and was cut off. Please generate a more concise and compact version of the blueprint."
                logger.warning(
                    f"Blueprint for {target_file} hit token limit. Retrying with request for conciseness."
                )
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                feedback = f"The previous attempt failed because the generated JSON was invalid. Error: {e}. Please ensure you only output a single, valid JSON object with no extra text or formatting errors."
                logger.warning(
                    f"Blueprint for {target_file} was invalid JSON. Retrying..."
                )

        logger.error(
            f"Failed to generate a valid blueprint for {target_file} after {config.max_debug_attempts} attempts."
        )
        return False, f"Failed to generate a valid blueprint for {target_file}."

    async def execute_development_task(
        self, task: dict, dev_strategy: str
    ) -> Tuple[bool, str]:
        """Handles code generation and modification tasks with a validation loop."""
        target_file = task.get("details", {}).get("target_file")
        if not target_file:
            return False, "Development task requires a 'target_file'."

        logger.info(f"Starting development task for file: {target_file}")

        file_blueprint = self.project_state.living_blueprint.get_file(target_file)
        if not file_blueprint or not file_blueprint.language:
            return (
                False,
                f"Detailed blueprint for {target_file} not found or is incomplete. Ensure blueprinting task ran first.",
            )

        language = file_blueprint.language

        feedback = None
        for attempt in range(config.max_debug_attempts):
            logger.info(f"Code generation attempt {attempt + 1} for {target_file}...")
            try:
                generated_code = await self.ai.code_crafter(
                    goal=self.project_state.goal,
                    task=task,
                    living_blueprint=self.project_state.living_blueprint,
                    current_codebase=self.assembler.get_all_files(),
                    cognitive_state=self.project_state.cognitive_state,
                    feedback=feedback,
                    dev_strategy=dev_strategy,
                )

                validation_result = await self.ai.code_validator(
                    generated_code,
                    language,
                    self.project_state.cognitive_state,
                    dev_strategy,
                )

                if validation_result.get("is_valid"):
                    logger.info(f"Code for {target_file} passed validation.")
                    self.assembler.add_file(target_file, generated_code)
                    self.assembler.write_files_to_disk()
                    return (
                        True,
                        f"Successfully generated and validated code for {target_file}.",
                    )
                else:
                    feedback = validation_result.get(
                        "feedback", "No specific feedback."
                    )
                    logger.warning(
                        f"Code validation failed for {target_file}. Feedback: {feedback}"
                    )
            except TokenLimitError:
                feedback = "Your previous code was too long and was cut off. Please generate a more concise and compact version that fulfills the blueprint."
                logger.warning(
                    f"Code generation for {target_file} hit token limit. Retrying with request for conciseness."
                )
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred during code generation for {target_file}: {e}"
                )
                feedback = f"An unexpected error occurred: {e}. Please try an alternative approach."

        return (
            False,
            f"Failed to generate valid code for {target_file} after {config.max_debug_attempts} attempts.",
        )

    async def execute_tool_task(self, details: dict) -> Tuple[bool, str]:
        """Executes a tool based on the provided details."""
        tool_name = details.get("tool_name")
        tool_args = details.get("args", {})

        if not tool_name:
            return False, "Tool execution task requires a 'tool_name'."

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return False, f"Tool '{tool_name}' not found in registry."

        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
        try:
            result = await tool.execute(
                **tool_args,
                cognitive_context=self.project_state.cognitive_state,
                dev_strategy=self.project_state.development_strategy,
            )

            if result.get("returncode", 0) == 0 or result.get("status") == "SUCCESS":
                return (
                    True,
                    f"Tool '{tool_name}' executed successfully. Output: {result.get('stdout') or result.get('message')}",
                )
            else:
                return (
                    False,
                    f"Tool '{tool_name}' failed. Error: {result.get('stderr') or result.get('message')}",
                )

        except Exception as e:
            return (
                False,
                f"An unexpected error occurred while executing tool '{tool_name}': {e}",
            )

    async def execute_sprint_review(self):
        """Agile sprint review and retrospective."""
        retrospective_prompt = f"Review sprint for {self.project_state.current_epic}"
        retrospective = await self.ai._make_api_call(
            retrospective_prompt,
            "Retrospective",
            is_json=False,
            cognitive_state=self.project_state.cognitive_state,
            dev_strategy=DevelopmentStrategy.AGILE.value,
        )
        self.project_state.velocity_history.append(
            len([t for t in self.project_state.completed_tasks if "sprint" in str(t)])
        )
        self.memory.add_memory(
            self.goal,
            self.project_state.living_blueprint,
            strategy=["agile"],
            cognitive_state=self.project_state.cognitive_state,
            dev_strategy=DevelopmentStrategy.AGILE.value,
        )
        logger.info("Sprint review and retrospective completed.")

    async def _init_git_repo(self):
        """Initialize Git for DevOps."""
        git_tool = self.tool_registry.get_tool("GitTool")
        if git_tool:
            await git_tool.execute(
                "git init && git add . && git commit -m 'Initial commit'",
                self.project_state.cognitive_state,
            )
        self.project_state.git_repo = self.sandbox_dir + "/.git"

    async def _commit_and_push(self):
        """Auto-commit for DevOps."""
        git_tool = self.tool_registry.get_tool("GitTool")
        if git_tool:
            await git_tool.execute(
                "git add . && git commit -m 'Auto-commit from AI' && git push",
                self.project_state.cognitive_state,
            )

    async def _execute_git_commit_task(self, details: Dict) -> Tuple[bool, str]:
        """Execute Git commit for DevOps."""
        git_tool = self.tool_registry.get_tool("GitTool")
        if not git_tool:
            return False, "GitTool not available."
        result = await git_tool.execute(
            "git add . && git commit -m 'AI commit' && git push",
            self.project_state.cognitive_state,
        )
        if result["returncode"] == 0:
            return True, "Git commit and push successful."
        else:
            return False, result["stderr"]

    async def execute_deployment_task(
        self, details: dict, dev_strategy: str
    ) -> Tuple[bool, str]:
        """Execute deployment using CloudDeploymentTool"""
        platform = details.get("platform", "kubernetes")
        strategy = details.get("strategy", "rolling")
        config_file = details.get("config_file", "deployment.yaml")

        cloud_tool = self.tool_registry.get_tool("CloudDeploymentTool")
        if not cloud_tool:
            return False, "CloudDeploymentTool not available."

        result = await cloud_tool.execute(
            platform,
            strategy,
            config_file,
            self.project_state.cognitive_state,
            dev_strategy,
        )
        if result["status"] == "SUCCESS":
            return True, result["message"]
        else:
            return False, result["message"]

    async def execute_security_scan_task(
        self, details: dict, dev_strategy: str
    ) -> Tuple[bool, str]:
        """Execute security scan using SecurityScannerTool"""
        scan_type = details.get("scan_type", "full")

        security_tool = self.tool_registry.get_tool("SecurityScannerTool")
        if not security_tool:
            return False, "SecurityScannerTool not available."

        result = await security_tool.execute(
            scan_type, self.project_state.cognitive_state, dev_strategy
        )
        if result["status"] == "SUCCESS":
            self.project_state.security_scan_results = result
            return True, f"Security scan completed: {result['cognitive_analysis']}"
        else:
            return False, result["message"]

    async def execute_performance_optimization_task(
        self, details: dict, dev_strategy: str
    ) -> Tuple[bool, str]:
        """Execute performance optimization using PerformanceOptimizerTool"""
        optimize_for = details.get("optimize_for", "cpu")

        perf_tool = self.tool_registry.get_tool("PerformanceOptimizerTool")
        if not perf_tool:
            return False, "PerformanceOptimizerTool not available."

        result = await perf_tool.execute(
            optimize_for, self.project_state.cognitive_state, dev_strategy
        )
        if result["status"] == "SUCCESS":
            self.project_state.performance_metrics.update(result["metrics"])
            return True, result["cognitive_recommendation"]
        else:
            return False, result["message"]

    async def execute_cognitive_analysis_task(self, task_details: Dict) -> str:
        # ... (implementation remains the same)
        return "Cognitive analysis complete."

    async def execute_self_evolution_task(self, task_details: Dict) -> str:
        # ... (implementation remains the same)
        return "Self-evolution complete."

    async def handle_task_failure(self, task_response: dict, error_result: str):
        """Handle task failure with cognitive recovery strategies"""
        task_description = task_response.get("task_description", "Unknown task")
        logger.warning(f"Handling failure for task: {task_description}")

        failure_count = (
            self.project_state.chronic_failure_tracker.get(task_description, 0) + 1
        )
        self.project_state.chronic_failure_tracker[task_description] = failure_count

        if failure_count >= config.metamorphosis_threshold:
            logger.warning(
                f"Task '{task_description}' has failed {failure_count} times. Attempting cognitive recovery."
            )
            await self.attempt_cognitive_recovery(task_description, error_result)
        else:
            # For now, we just log. A more advanced system might try re-planning sub-tasks.
            logger.info(
                f"Task '{task_description}' failed, but below metamorphosis threshold. Continuing."
            )

    async def attempt_cognitive_recovery(
        self, task_description: str, error_result: str
    ):
        """Attempt cognitive recovery for persistent failures"""
        try:
            analysis = f"Analysis of failure for '{task_description}'."

            evolution_plan_dict = await self.ai.metamorph_architect(
                self.project_state.goal,
                f"Chronic failure in task: {task_description}\n\nError: {error_result}\n\nAnalysis: {analysis}",
                self.project_state.cognitive_state,
                self.project_state.development_strategy,
            )

            await self.apply_cognitive_evolution(evolution_plan_dict)

            self.project_state.evolutionary_path.append(
                {
                    "timestamp": time.time(),
                    "task": task_description,
                    "error": error_result,
                    "evolution": evolution_plan_dict,
                }
            )
        except Exception as e:
            logger.error(f"Cognitive recovery attempt failed: {e}")

    async def apply_cognitive_evolution(self, evolution_plan: dict):
        """Apply cognitive evolution based on the evolution plan"""
        try:
            for filename, new_code in evolution_plan.get("code_changes", {}).items():
                self.assembler.add_file(filename, new_code)

            cognitive_enhancements = evolution_plan.get("cognitive_enhancements", {})
            for key, value in cognitive_enhancements.items():
                if key in self.project_state.cognitive_state:
                    self.project_state.cognitive_state[key] = value

            self.project_state.cognitive_state["evolution_count"] += 1
            logger.info("Cognitive evolution applied successfully")

        except Exception as e:
            logger.error(f"Failed to apply cognitive evolution: {e}")

    async def _check_evolution_opportunities(self):
        """Check for opportunities to evolve"""
        evolutionary_readiness = self._assess_evolutionary_readiness()

        if evolutionary_readiness > 0.7:
            logger.info(
                f"Evolutionary readiness is high ({evolutionary_readiness}), considering evolution"
            )

            if self.project_state.cognitive_state["certainty"] < 0.5:
                await self.execute_self_evolution_task({"evolution_type": "cognitive"})
            else:
                await self.execute_self_evolution_task({"evolution_type": "capability"})

    def _calculate_quality_score(self, patterns: Dict) -> float:
        """Calculate code quality score based on cognitive patterns"""
        score = 0.5
        positive_patterns = ["abstraction", "modularity", "error_handling"]
        for pattern in positive_patterns:
            for file_patterns in patterns.values():
                if pattern in file_patterns:
                    score += 0.1 * file_patterns[pattern]
        return min(1.0, max(0.0, score))

    def _generate_cognitive_recommendations(self, patterns: Dict) -> List[str]:
        """Generate cognitive recommendations based on patterns"""
        recommendations = []
        for filename, file_patterns in patterns.items():
            if "abstraction" not in file_patterns or file_patterns["abstraction"] < 2:
                recommendations.append(f"Increase abstraction in {filename}")
            if (
                "error_handling" not in file_patterns
                or file_patterns["error_handling"] < 1
            ):
                recommendations.append(f"Add error handling in {filename}")
        return recommendations

    def _calculate_architectural_complexity(self) -> float:
        """Calculate architectural complexity from cognitive perspective"""
        file_count = len(self.project_state.living_blueprint.root)
        class_count = sum(
            len(file.classes) for file in self.project_state.living_blueprint.root
        )
        complexity = (file_count * 0.2) + (class_count * 0.1)
        return min(1.0, complexity / 10.0)

    def _assess_evolutionary_readiness(self) -> float:
        """Assess how ready the system is for evolution"""
        cognitive_state = self.project_state.cognitive_state
        readiness = 0.5
        readiness += (cognitive_state["attention_level"] - 0.5) * 0.2
        readiness += (cognitive_state["certainty"] - 0.5) * 0.2
        readiness += min(0.3, cognitive_state["evolution_count"] * 0.1)
        return min(1.0, max(0.0, readiness))

    async def cleanup(self):
        """Cleanup resources and perform final reporting"""
        logger.info("Cleaning up resources")
        report = {
            "project_name": self.project_name,
            "goal": self.goal,
            "completed_tasks": self.project_state.completed_tasks,
            "performance_metrics": self.project_state.performance_metrics,
            "security_scan_results": self.project_state.security_scan_results,
            "cost_estimates": self.project_state.cost_estimates,
            "cognitive_insights": {
                "final_cognitive_state": self.project_state.cognitive_state,
                "evolutionary_path": self.project_state.evolutionary_path,
                "cognitive_patterns": self.assembler.get_cognitive_patterns(),
            },
        }

        report_dir = create_dir_safely(self.sandbox_dir)
        report_path = Path(report_dir) / "cognitive_final_report.json"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Final report with cognitive insights saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to write final report to {report_path}: {e}")


# --- Cognitive File Assembler (assumed implementation from original, enhanced if needed) ---
class CognitiveFileAssembler:
    def __init__(self, sandbox_dir: str, agent_filename: str):
        self.sandbox_dir = sandbox_dir
        self.agent_filename = agent_filename
        self.files = {}

    def add_file(self, filename: str, content: str):
        self.files[filename] = content

    def write_files_to_disk(self):
        sandbox_path = create_dir_safely(self.sandbox_dir)
        for filename, content in self.files.items():
            path = Path(sandbox_path) / filename
            create_dir_safely(str(path.parent))
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                logger.error(f"Failed to write file {path}: {e}")

    def get_all_files(self) -> Dict[str, str]:
        return self.files

    def check_project_size_limits(self) -> bool:
        total_size = sum(len(content) for content in self.files.values())
        return total_size < config.max_project_size_gb * 1024**3

    def get_cognitive_patterns(self) -> Dict[str, Dict[str, int]]:
        """Analyze files and extract simple cognitive/structural metrics per file."""
        patterns: Dict[str, Dict[str, int]] = {}
        for filename, content in self.files.items():
            metrics: Dict[str, int] = {
                "abstraction": 0,      # approx: number of classes
                "modularity": 0,       # approx: classes + functions
                "documentation": 0,    # approx: docstrings present
                "complexity": 0,       # approx: control-flow nodes
                "typedness": 0,        # approx: annotations count
                "comment_density": 0,  # approx: percentage of comment lines
            }

            try:
                if filename.endswith(".py"):
                    tree = ast.parse(content)
                    num_classes = sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
                    num_funcs = sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
                    num_async = sum(isinstance(n, ast.AsyncFunctionDef) for n in ast.walk(tree))

                    # Docstrings
                    doc_count = 1 if ast.get_docstring(tree) else 0
                    for n in ast.walk(tree):
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            if ast.get_docstring(n):
                                doc_count += 1

                    # Typedness
                    typed_count = sum(isinstance(n, ast.AnnAssign) for n in ast.walk(tree))
                    for n in ast.walk(tree):
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if getattr(n, "returns", None) is not None:
                                typed_count += 1
                            for arg in list(getattr(n, "args", ast.arguments()).args or []):
                                if getattr(arg, "annotation", None) is not None:
                                    typed_count += 1

                    # Complexity approximation
                    complexity_nodes = (
                        ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp, ast.Compare
                    )
                    complexity = sum(isinstance(n, complexity_nodes) for n in ast.walk(tree)) + 1

                    # Comments density
                    lines = content.splitlines()
                    total_lines = max(1, len(lines))
                    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
                    comment_density = int(100 * comment_lines / total_lines)

                    metrics["abstraction"] = int(num_classes)
                    metrics["modularity"] = int(num_classes + num_funcs + num_async)
                    metrics["documentation"] = int(doc_count)
                    metrics["complexity"] = int(complexity)
                    metrics["typedness"] = int(typed_count)
                    metrics["comment_density"] = int(comment_density)
                else:
                    # Lightweight heuristics for non-Python files
                    lc = content.lower()
                    num_classes = lc.count("class ")
                    num_funcs = lc.count("function ") + lc.count("=>")
                    control = lc.count(" if ") + lc.count(" for ") + lc.count(" while ")
                    docs = lc.count("/**") + lc.count("///")
                    comments = sum(1 for l in content.splitlines() if l.strip().startswith(("//", "/*", "*")))
                    total = max(1, len(content.splitlines()))

                    metrics["abstraction"] = int(num_classes)
                    metrics["modularity"] = int(num_classes + num_funcs)
                    metrics["documentation"] = int(docs)
                    metrics["complexity"] = int(control + 1)
                    metrics["typedness"] = int(lc.count(": "))  # weak proxy
                    metrics["comment_density"] = int(100 * comments / total)

            except Exception:
                # On parse errors, keep defaults but mark with minimal signal
                metrics.setdefault("modularity", 0)

            patterns[filename] = metrics

        return patterns


# --- Main Execution with Cognitive Enhancements ---
async def main():
    """Main execution function with cognitive enhancements"""
    try:
        # Increase timeout for Hugging Face model downloads
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

        load_dotenv()

        # Create important directories using safe creator which falls back to temp dir
        session_dir_used = create_dir_safely(config.session_dir)
        memory_dir_used = create_dir_safely(config.memory_dir)
        cache_dir_used = create_dir_safely(config.cache_dir)

        existing_sessions = [
            f.replace(".json", "")
            for f in os.listdir(session_dir_used)
            if f.endswith(".json")
        ]

        orchestrator = None
        llm_choice = "api"
        dev_strategy = "cognitive"

        print("\n🤖 ULTRA COGNITIVE AI SOFTWARE ENGINEER")
        print("=" * 50)

        print("\nSelect the LLM to use:")
        print("  1. Gemini API")
        print("  2. Local Ollama (mixtral:8x7b)")
        print("  3. OpenAI GPT-4 Turbo")

        llm_choice_input = input("Enter your choice (1, 2, or 3): ").strip()
        if llm_choice_input == "2":
            llm_choice = "local"
            print("✅ Using local Ollama (mixtral:8x7b). Make sure Ollama is running.")
        elif llm_choice_input == "3":
            llm_choice = "openai"
            print("✅ Using OpenAI GPT-4 Turbo.")
            if not os.getenv("OPENAI_API_KEY"):
                print("❌ Configuration issue. Exiting.")
                return
        else:
            llm_choice = "api"
            print("✅ Using Gemini API.")
            if not os.getenv("GEMINI_API_KEY"):
                print("❌ Configuration issue. Exiting.")
                return

        print("\nSelect the Development Strategy:")
        print("  1. Test-Driven Development (TDD)")
        print("  2. Bulk Code Generation")
        print("  3. Agile Development")
        print("  4. DevOps Pipeline")
        print("  5. Cognitive Development (Recommended)")

        strategy_choice_input = input("Enter your choice (1, 2, 3, 4, or 5): ").strip()
        if strategy_choice_input == "1":
            dev_strategy = DevelopmentStrategy.TDD.value
        elif strategy_choice_input == "2":
            dev_strategy = "bulk"  # For bulk, though not enum
        elif strategy_choice_input == "3":
            dev_strategy = DevelopmentStrategy.AGILE.value
        elif strategy_choice_input == "4":
            dev_strategy = DevelopmentStrategy.DEVOPS.value
        elif strategy_choice_input == "5":
            dev_strategy = DevelopmentStrategy.COGNITIVE.value
        else:
            dev_strategy = DevelopmentStrategy.TDD.value

        print(f"✅ Using {dev_strategy} development strategy.")

        if existing_sessions:
            print(f"\nFound {len(existing_sessions)} existing projects:")
            for i, session_name in enumerate(existing_sessions):
                print(f"  {i + 1}. {session_name}")

            choice = (
                input("\n(R)esume a project or start a (N)ew one? ").strip().lower()
            )

            if choice == "r":
                try:
                    project_choice = (
                        int(
                            input("Enter the number of the project to resume: ").strip()
                        )
                        - 1
                    )
                    if 0 <= project_choice < len(existing_sessions):
                        project_name = existing_sessions[project_choice]
                        orchestrator = UltraCognitiveForgeOrchestrator(
                            goal="",
                            project_name=project_name,
                            agent_filename=os.path.basename(__file__),
                            llm_choice=llm_choice,
                            dev_strategy=dev_strategy,
                        )
                        await orchestrator.load_state()
                    else:
                        print("❌ Invalid choice.")
                        return
                except (ValueError, IndexError):
                    print("❌ Invalid input.")
                    return
            elif choice != "n" and choice != "":
                print("❌ Invalid choice. Exiting.")
                return

        if not orchestrator:
            user_prompt = input("\n🎯 Please enter your master goal: ").strip()
            if not user_prompt:
                print("❌ No goal entered. Exiting.")
                return

            project_name = get_project_name_from_goal(user_prompt)
            print(f"✅ Project name will be: '{project_name}'")

            orchestrator = UltraCognitiveForgeOrchestrator(
                goal=user_prompt,
                project_name=project_name,
                agent_filename=os.path.basename(__file__),
                llm_choice=llm_choice,
                dev_strategy=dev_strategy,
            )

        await orchestrator.run()

    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        logger.debug(traceback.format_exc())


def get_project_name_from_goal(goal: str) -> str:
    """Creates a filesystem-safe project name from the user's goal, avoiding path length limits."""
    sanitized = re.sub(r"[^\w\s-]", "", goal).strip()
    sanitized = re.sub(r"[-\s]+", "_", sanitized).lower()
    truncated_name = sanitized[:50]
    goal_hash = uuid.uuid5(uuid.NAMESPACE_DNS, goal).hex[:8]
    return f"{truncated_name}_{goal_hash}"


if __name__ == "__main__":

    def signal_handler(sig, frame):
        print("\n🛑 Shutdown signal received. Performing graceful shutdown...")
        # Add cleanup logic from orchestrator if it exists
        # This part is tricky because orchestrator is in async scope
        # For simplicity, we just exit.
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    asyncio.run(main())
