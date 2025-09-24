"""
Utility functions for Prometheus AI Orchestrator
Contains logging, path utilities, encryption, and other helper functions
"""

import logging
import logging.config
import os
import sys
import threading
import uuid
import re
from pathlib import Path
from typing import Dict, Any, Optional
import signal
import traceback

# Try to import psutil, fall back gracefully
try:
    import psutil
except ImportError:
    psutil = None

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


def normalize_path(path: str) -> str:
    """Normalize and validate file paths."""
    if not path:
        return ""
    return os.path.normpath(os.path.expanduser(path))


def sanitize_dir_path(path: str) -> str:
    """Sanitize directory path for safe creation."""
    if not path:
        return ""

    # Remove dangerous characters
    sanitized = re.sub(r'[<>:"|?*]', "", path)

    # Ensure it's not an absolute path (for security)
    if os.path.isabs(sanitized):
        sanitized = os.path.relpath(sanitized)

    return sanitized


def create_dir_safely(path: str) -> Path:
    """Create directory safely with proper error handling."""
    try:
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    except Exception as e:
        logging.error(f"Failed to create directory {path}: {e}")
        raise


def safe_execute(func):
    """Decorator for safe function execution with error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            logging.debug(traceback.format_exc())
            raise
    return wrapper


def validate_file_path(path: str, base_dir: str) -> bool:
    """Validate that a file path is within the allowed base directory."""
    try:
        resolved_path = os.path.realpath(os.path.join(base_dir, path))
        resolved_base = os.path.realpath(base_dir)
        return resolved_path.startswith(resolved_base)
    except Exception:
        return False


def encrypt_data(data: str, key: str = None) -> str:
    """Encrypt data using Fernet symmetric encryption."""
    if not data:
        return ""

    try:
        encryption_key = key or os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
        fernet = Fernet(encryption_key.encode())
        return fernet.encrypt(data.encode()).decode()
    except Exception as e:
        logging.error(f"Encryption failed: {e}")
        return data  # Return unencrypted if encryption fails


def decrypt_data(encrypted_data: str, key: str = None) -> str:
    """Decrypt data using Fernet symmetric encryption."""
    if not encrypted_data:
        return ""

    try:
        encryption_key = key or os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
        fernet = Fernet(encryption_key.encode())
        return fernet.decrypt(encrypted_data.encode()).decode()
    except Exception as e:
        logging.error(f"Decryption failed: {e}")
        return encrypted_data  # Return original if decryption fails


def get_system_metrics() -> Dict[str, float]:
    """Get current system performance metrics."""
    if psutil is None:
        logging.warning("psutil not available, returning empty metrics")
        return {}

    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
            "disk_percent": disk.percent,
            "disk_used_gb": disk.used / (1024**3),
            "disk_total_gb": disk.total / (1024**3),
        }
    except Exception as e:
        logging.error(f"Failed to get system metrics: {e}")
        return {}


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    print("\n🛑 Shutdown signal received. Performing graceful shutdown...")
    # Add cleanup logic from orchestrator if it exists
    # This part is tricky because orchestrator is in async scope
    # For simplicity, we just exit.
    sys.exit(0)


# Initialize logging
setup_logging()