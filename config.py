#!/usr/bin/env python3
"""
Configuration module for the P2P Chat Application.
"""

import os
import json
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration class for the P2P Chat Application."""

    # Network settings
    host: str = "0.0.0.0"
    port: int = 0  # Will be assigned automatically
    max_peers: int = 10

    # Discovery settings
    discovery_port: int = 8888
    broadcast_interval: float = 5.0  # seconds

    # Storage settings
    data_dir: str = "chat_data"
    max_messages_per_chat: int = 1000

    # User interface settings
    max_username_length: int = 20
    command_prompt: str = "chat> "

    # Security settings
    enable_encryption: bool = False
    max_message_length: int = 4096

    def __init__(self):
        """Initialize configuration with environment variables if available."""
        self.host = os.getenv("CHAT_HOST", "0.0.0.0")
        self.port = int(os.getenv("CHAT_PORT", "0"))
        self.discovery_port = int(os.getenv("CHAT_DISCOVERY_PORT", "8888"))
        self.data_dir = os.getenv("CHAT_DATA_DIR", "chat_data")
        self.max_peers = int(os.getenv("CHAT_MAX_PEERS", "10"))
        self.broadcast_interval = float(os.getenv("CHAT_BROADCAST_INTERVAL", "5.0"))
        self.max_messages_per_chat = int(os.getenv("CHAT_MAX_MESSAGES", "1000"))
        self.max_username_length = int(os.getenv("CHAT_MAX_USERNAME_LENGTH", "20"))
        self.enable_encryption = os.getenv("CHAT_ENABLE_ENCRYPTION", "false").lower() == "true"
        self.max_message_length = int(os.getenv("CHAT_MAX_MESSAGE_LENGTH", "4096"))

        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "max_peers": self.max_peers,
            "discovery_port": self.discovery_port,
            "broadcast_interval": self.broadcast_interval,
            "data_dir": self.data_dir,
            "max_messages_per_chat": self.max_messages_per_chat,
            "max_username_length": self.max_username_length,
            "command_prompt": self.command_prompt,
            "enable_encryption": self.enable_encryption,
            "max_message_length": self.max_message_length,
        }

    def save_to_file(self, filepath: str = "config.json") -> None:
        """Save configuration to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str = "config.json") -> "Config":
        """Load configuration from a JSON file."""
        if not os.path.exists(filepath):
            return cls()

        with open(filepath, 'r') as f:
            data = json.load(f)

        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"Config(host={self.host}, port={self.port}, data_dir={self.data_dir})"