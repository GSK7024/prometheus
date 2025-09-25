#!/usr/bin/env python3
"""
Core P2P networking functionality for the chat application.
"""

import asyncio
import json
import logging
import socket
import sqlite3
import time
import uuid
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from config import Config

logger = logging.getLogger(__name__)


class Message:
    """Represents a chat message."""

    def __init__(self, sender: str, content: str, timestamp: Optional[float] = None):
        """Initialize a message."""
        self.sender = sender
        self.content = content
        self.timestamp = timestamp or time.time()
        self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "sender": self.sender,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        """Create message from dictionary."""
        return cls(
            sender=data["sender"],
            content=data["content"],
            timestamp=data.get("timestamp"),
        )


class Peer:
    """Represents a connected peer."""

    def __init__(self, address: Tuple[str, int], username: str):
        """Initialize a peer."""
        self.address = address
        self.username = username
        self.connected_at = time.time()
        self.last_seen = time.time()
        self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict:
        """Convert peer to dictionary."""
        return {
            "id": self.id,
            "address": self.address,
            "username": self.username,
            "connected_at": self.connected_at,
            "last_seen": self.last_seen,
        }


class P2PChatCore:
    """Core P2P chat functionality."""

    def __init__(self, config: Config):
        """Initialize the P2P chat core."""
        self.config = config
        self.username = ""
        self.server = None
        self.discovery_server = None
        self.peers: Dict[str, Peer] = {}
        self.connected_peers: Set[str] = set()
        self.message_history: List[Message] = []
        self.db_connection = None
        self.loop = None

    async def initialize(self) -> None:
        """Initialize the P2P chat core."""
        logger.info("Initializing P2P Chat Core...")

        # Initialize database
        self._init_database()

        # Set up networking
        self.loop = asyncio.get_event_loop()
        self.server = await self._create_server()
        self.discovery_server = await self._create_discovery_server()

        logger.info(f"Chat server listening on {self.server.sockets[0].getsockname()}")
        logger.info(f"Discovery server listening on port {self.config.discovery_port}")

        # Start discovery broadcasting
        asyncio.create_task(self._broadcast_presence())

    def _init_database(self) -> None:
        """Initialize the SQLite database for message storage."""
        db_path = f"{self.config.data_dir}/chat_history.db"

        self.db_connection = sqlite3.connect(db_path, check_same_thread=False)
        cursor = self.db_connection.cursor()

        # Create messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                sender TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                peer_address TEXT
            )
        ''')

        # Create peers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS peers (
                id TEXT PRIMARY KEY,
                address TEXT UNIQUE NOT NULL,
                username TEXT NOT NULL,
                connected_at REAL NOT NULL,
                last_seen REAL NOT NULL
            )
        ''')

        self.db_connection.commit()
        logger.info(f"Database initialized at {db_path}")

    async def _create_server(self) -> asyncio.AbstractServer:
        """Create the main chat server."""
        def get_free_port():
            """Get a free port for the server."""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                s.listen(1)
                port = s.getsockname()[1]
            return port

        if self.config.port == 0:
            self.config.port = get_free_port()

        return await asyncio.start_server(
            self._handle_client,
            self.config.host,
            self.config.port
        )

    async def _create_discovery_server(self) -> asyncio.AbstractServer:
        """Create the discovery server for peer discovery."""
        return await asyncio.start_server(
            self._handle_discovery,
            self.config.host,
            self.config.discovery_port
        )

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle incoming client connections."""
        peer_addr = writer.get_extra_info('peername')
        logger.info(f"New connection from {peer_addr}")

        try:
            # Receive peer information
            data = await reader.read(1024)
            if not data:
                return

            peer_info = json.loads(data.decode())
            peer = Peer(peer_addr, peer_info.get('username', 'Unknown'))
            self.peers[peer.id] = peer
            self.connected_peers.add(peer.id)

            # Send our information
            our_info = {"username": self.username, "port": self.config.port}
            writer.write(json.dumps(our_info).encode())
            await writer.drain()

            # Handle messages
            while True:
                data = await reader.read(4096)
                if not data:
                    break

                message_data = json.loads(data.decode())
                message = Message(
                    sender=peer.username,
                    content=message_data['content'],
                    timestamp=message_data.get('timestamp', time.time())
                )

                await self._handle_message(message, peer_addr)
                self.message_history.append(message)

                # Store in database
                self._store_message(message, peer_addr)

        except Exception as e:
            logger.error(f"Error handling client {peer_addr}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            # Remove peer from connected peers
            self.connected_peers.discard(peer.id)

    async def _handle_discovery(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle peer discovery requests."""
        peer_addr = writer.get_extra_info('peername')
        logger.info(f"Discovery request from {peer_addr}")

        # Send our peer information
        peer_info = {
            "username": self.username,
            "address": peer_addr,
            "port": self.config.port
        }

        writer.write(json.dumps(peer_info).encode())
        await writer.drain()

    async def _broadcast_presence(self) -> None:
        """Broadcast our presence to the network."""
        while True:
            try:
                # Broadcast to discovery port on all interfaces
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

                broadcast_info = {
                    "username": self.username,
                    "port": self.config.port,
                    "type": "presence"
                }

                sock.sendto(
                    json.dumps(broadcast_info).encode(),
                    ('<broadcast>', self.config.discovery_port)
                )
                sock.close()

                logger.debug(f"Broadcasted presence: {self.username}")

            except Exception as e:
                logger.error(f"Error broadcasting presence: {e}")

            await asyncio.sleep(self.config.broadcast_interval)

    async def _handle_message(self, message: Message, peer_addr: Tuple[str, int]) -> None:
        """Handle an incoming message."""
        logger.info(f"Message from {message.sender}: {message.content}")

        # Add timestamp if not present
        if not hasattr(message, 'timestamp') or message.timestamp is None:
            message.timestamp = time.time()

        # Store message in database
        self._store_message(message, peer_addr)

        # Add to local history
        self.message_history.append(message)

        # Keep only recent messages
        if len(self.message_history) > self.config.max_messages_per_chat:
            self.message_history = self.message_history[-self.config.max_messages_per_chat:]

    def _store_message(self, message: Message, peer_addr: Tuple[str, int]) -> None:
        """Store a message in the database."""
        if self.db_connection:
            cursor = self.db_connection.cursor()
            cursor.execute(
                "INSERT INTO messages (id, sender, content, timestamp, peer_address) VALUES (?, ?, ?, ?, ?)",
                (message.id, message.sender, message.content, message.timestamp, str(peer_addr))
            )
            self.db_connection.commit()

    def _store_peer(self, peer: Peer) -> None:
        """Store a peer in the database."""
        if self.db_connection:
            cursor = self.db_connection.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO peers (id, address, username, connected_at, last_seen) VALUES (?, ?, ?, ?, ?)",
                (peer.id, str(peer.address), peer.username, peer.connected_at, peer.last_seen)
            )
            self.db_connection.commit()

    async def send_message(self, recipient: str, content: str) -> bool:
        """Send a message to a specific peer."""
        # Find the peer
        target_peer = None
        for peer in self.peers.values():
            if peer.username == recipient:
                target_peer = peer
                break

        if not target_peer:
            logger.error(f"Peer {recipient} not found")
            return False

        # Create message
        message = Message(self.username, content)

        try:
            # Connect to peer and send message
            reader, writer = await asyncio.open_connection(
                target_peer.address[0], target_peer.address[1]
            )

            message_data = {
                "content": message.content,
                "timestamp": message.timestamp,
                "sender": message.sender
            }

            writer.write(json.dumps(message_data).encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()

            # Store locally
            self._store_message(message, target_peer.address)
            self.message_history.append(message)

            logger.info(f"Message sent to {recipient}: {content}")
            return True

        except Exception as e:
            logger.error(f"Error sending message to {recipient}: {e}")
            return False

    def get_peers(self) -> List[Peer]:
        """Get list of discovered peers."""
        return list(self.peers.values())

    def get_messages(self, limit: int = 100) -> List[Message]:
        """Get recent messages."""
        return self.message_history[-limit:]

    def get_messages_from_db(self, limit: int = 100) -> List[Message]:
        """Get messages from database."""
        if not self.db_connection:
            return []

        cursor = self.db_connection.cursor()
        cursor.execute(
            "SELECT id, sender, content, timestamp, peer_address FROM messages ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )

        messages = []
        for row in cursor.fetchall():
            message = Message(row[1], row[2], row[3])
            message.id = row[0]
            messages.append(message)

        return list(reversed(messages))  # Return in chronological order

    def set_username(self, username: str) -> None:
        """Set the username for this chat client."""
        self.username = username
        logger.info(f"Username set to: {username}")

    def shutdown(self) -> None:
        """Shutdown the P2P chat core."""
        logger.info("Shutting down P2P Chat Core...")

        if self.db_connection:
            self.db_connection.close()

        if self.server:
            self.server.close()

        if self.discovery_server:
            self.discovery_server.close()