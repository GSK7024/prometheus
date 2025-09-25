#!/usr/bin/env python3
"""
Unit tests for the P2P Chat Core functionality.
"""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import time

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from p2p_core import P2PChatCore, Message, Peer


class TestMessage(unittest.TestCase):
    """Test cases for the Message class."""

    def test_message_creation(self):
        """Test creating a message."""
        content = "Hello, world!"
        sender = "test_user"

        message = Message(sender, content)

        self.assertEqual(message.sender, sender)
        self.assertEqual(message.content, content)
        self.assertIsNotNone(message.timestamp)
        self.assertIsNotNone(message.id)

    def test_message_creation_with_timestamp(self):
        """Test creating a message with a specific timestamp."""
        content = "Hello, world!"
        sender = "test_user"
        timestamp = 1234567890.0

        message = Message(sender, content, timestamp)

        self.assertEqual(message.sender, sender)
        self.assertEqual(message.content, content)
        self.assertEqual(message.timestamp, timestamp)

    def test_message_to_dict(self):
        """Test converting a message to dictionary."""
        content = "Hello, world!"
        sender = "test_user"
        timestamp = 1234567890.0

        message = Message(sender, content, timestamp)
        message_dict = message.to_dict()

        expected = {
            "id": message.id,
            "sender": sender,
            "content": content,
            "timestamp": timestamp,
        }

        self.assertEqual(message_dict["sender"], expected["sender"])
        self.assertEqual(message_dict["content"], expected["content"])
        self.assertEqual(message_dict["timestamp"], expected["timestamp"])
        self.assertIn("id", message_dict)

    def test_message_from_dict(self):
        """Test creating a message from dictionary."""
        data = {
            "sender": "test_user",
            "content": "Hello, world!",
            "timestamp": 1234567890.0,
        }

        message = Message.from_dict(data)

        self.assertEqual(message.sender, data["sender"])
        self.assertEqual(message.content, data["content"])
        self.assertEqual(message.timestamp, data["timestamp"])


class TestPeer(unittest.TestCase):
    """Test cases for the Peer class."""

    def test_peer_creation(self):
        """Test creating a peer."""
        address = ("127.0.0.1", 8080)
        username = "test_peer"

        peer = Peer(address, username)

        self.assertEqual(peer.address, address)
        self.assertEqual(peer.username, username)
        self.assertIsNotNone(peer.connected_at)
        self.assertIsNotNone(peer.last_seen)
        self.assertIsNotNone(peer.id)

    def test_peer_to_dict(self):
        """Test converting a peer to dictionary."""
        address = ("127.0.0.1", 8080)
        username = "test_peer"

        peer = Peer(address, username)
        peer_dict = peer.to_dict()

        self.assertEqual(peer_dict["address"], address)
        self.assertEqual(peer_dict["username"], username)
        self.assertEqual(peer_dict["connected_at"], peer.connected_at)
        self.assertEqual(peer_dict["last_seen"], peer.last_seen)
        self.assertIn("id", peer_dict)


class TestP2PChatCore(unittest.TestCase):
    """Test cases for the P2PChatCore class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config()
        self.config.data_dir = self.temp_dir
        self.chat_core = P2PChatCore(self.config)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.chat_core.db_connection:
            self.chat_core.db_connection.close()

        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test P2P chat core initialization."""
        self.assertEqual(self.chat_core.config, self.config)
        self.assertEqual(self.chat_core.username, "")
        self.assertIsNone(self.chat_core.server)
        self.assertIsNone(self.chat_core.discovery_server)
        self.assertEqual(len(self.chat_core.peers), 0)
        self.assertEqual(len(self.chat_core.connected_peers), 0)
        self.assertEqual(len(self.chat_core.message_history), 0)

    def test_set_username(self):
        """Test setting username."""
        username = "test_user"
        self.chat_core.set_username(username)

        self.assertEqual(self.chat_core.username, username)

    def test_get_peers_empty(self):
        """Test getting peers when none exist."""
        peers = self.chat_core.get_peers()

        self.assertEqual(len(peers), 0)

    def test_get_messages_empty(self):
        """Test getting messages when none exist."""
        messages = self.chat_core.get_messages()

        self.assertEqual(len(messages), 0)

    def test_store_and_retrieve_message(self):
        """Test storing and retrieving a message."""
        # Create a test message
        message = Message("sender", "test content", 1234567890.0)
        peer_addr = ("127.0.0.1", 8080)

        # Store the message
        self.chat_core._store_message(message, peer_addr)

        # Retrieve messages from database
        messages = self.chat_core.get_messages_from_db()

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].sender, message.sender)
        self.assertEqual(messages[0].content, message.content)
        self.assertEqual(messages[0].timestamp, message.timestamp)

    def test_store_and_retrieve_peer(self):
        """Test storing and retrieving a peer."""
        # Create a test peer
        address = ("127.0.0.1", 8080)
        username = "test_peer"
        peer = Peer(address, username)

        # Store the peer
        self.chat_core._store_peer(peer)

        # Note: We can't easily test peer retrieval without more complex setup
        # This test mainly ensures the store method doesn't raise exceptions

    @patch('asyncio.open_connection')
    async def test_send_message_success(self, mock_open_connection):
        """Test successful message sending."""
        # Set up mock connection
        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_open_connection.return_value = (mock_reader, mock_writer)

        # Set username
        self.chat_core.set_username("sender")

        # Add a peer
        peer = Peer(("127.0.0.1", 8081), "recipient")
        self.chat_core.peers[peer.id] = peer

        # Send message
        success = await self.chat_core.send_message("recipient", "Hello!")

        self.assertTrue(success)
        mock_open_connection.assert_called_once_with("127.0.0.1", 8081)
        mock_writer.write.assert_called_once()
        mock_writer.drain.assert_called_once()
        mock_writer.close.assert_called_once()

    @patch('asyncio.open_connection')
    async def test_send_message_peer_not_found(self, mock_open_connection):
        """Test message sending when peer is not found."""
        # Set username
        self.chat_core.set_username("sender")

        # Try to send message to non-existent peer
        success = await self.chat_core.send_message("nonexistent", "Hello!")

        self.assertFalse(success)
        mock_open_connection.assert_not_called()

    @patch('asyncio.open_connection')
    async def test_send_message_connection_error(self, mock_open_connection):
        """Test message sending with connection error."""
        # Set up mock to raise exception
        mock_open_connection.side_effect = ConnectionError("Connection failed")

        # Set username
        self.chat_core.set_username("sender")

        # Add a peer
        peer = Peer(("127.0.0.1", 8081), "recipient")
        self.chat_core.peers[peer.id] = peer

        # Send message
        success = await self.chat_core.send_message("recipient", "Hello!")

        self.assertFalse(success)


if __name__ == "__main__":
    # Create test suite
    unittest.main(verbosity=2)