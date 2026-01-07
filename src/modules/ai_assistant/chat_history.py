"""
Data Copilot Lab - Chat History
Manage conversation history with context and persistence
"""

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChatHistory:
    """
    Manage conversation history

    Features:
    - Store conversation messages
    - Maintain context
    - Export/import history
    - Search through history
    - Token-aware truncation
    """

    def __init__(
        self,
        max_length: int = 50,
        persist_to_file: Optional[Union[str, Path]] = None
    ):
        """
        Initialize Chat History

        Args:
            max_length: Maximum number of messages to keep
            persist_to_file: File path to persist history
        """
        self.logger = logger
        self.max_length = max_length
        self.persist_to_file = Path(persist_to_file) if persist_to_file else None

        # Use deque for efficient FIFO operations
        self.messages = deque(maxlen=max_length)

        # Session metadata
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now().isoformat()

        # Load from file if exists
        if self.persist_to_file and self.persist_to_file.exists():
            self._load_from_file()

        self.logger.info(f"Chat history initialized (max_length={max_length})")

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add message to history

        Args:
            role: 'user', 'assistant', or 'system'
            content: Message content
            metadata: Additional metadata
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.messages.append(message)

        # Persist if configured
        if self.persist_to_file:
            self._save_to_file()

        self.logger.debug(f"Added {role} message (length: {len(content)})")

    def get_recent_messages(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get n most recent messages

        Args:
            n: Number of messages to retrieve

        Returns:
            List of messages
        """
        return list(self.messages)[-n:]

    def get_all_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in history"""
        return list(self.messages)

    def get_messages_by_role(self, role: str) -> List[Dict[str, Any]]:
        """
        Get messages by role

        Args:
            role: 'user', 'assistant', or 'system'

        Returns:
            Filtered messages
        """
        return [msg for msg in self.messages if msg['role'] == role]

    def search_messages(
        self,
        query: str,
        case_sensitive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search through message content

        Args:
            query: Search query
            case_sensitive: Case sensitive search

        Returns:
            Matching messages
        """
        if not case_sensitive:
            query = query.lower()

        results = []
        for msg in self.messages:
            content = msg['content']
            if not case_sensitive:
                content = content.lower()

            if query in content:
                results.append(msg)

        return results

    def get_messages_since(self, timestamp: str) -> List[Dict[str, Any]]:
        """
        Get messages since timestamp

        Args:
            timestamp: ISO format timestamp

        Returns:
            Messages after timestamp
        """
        return [
            msg for msg in self.messages
            if msg['timestamp'] >= timestamp
        ]

    def count_messages(self) -> Dict[str, int]:
        """Count messages by role"""
        counts = {"user": 0, "assistant": 0, "system": 0, "total": len(self.messages)}

        for msg in self.messages:
            role = msg['role']
            if role in counts:
                counts[role] += 1

        return counts

    def get_conversation_pairs(self) -> List[Dict[str, str]]:
        """
        Get user-assistant conversation pairs

        Returns:
            List of {user: ..., assistant: ...} pairs
        """
        pairs = []
        user_msg = None

        for msg in self.messages:
            if msg['role'] == 'user':
                user_msg = msg['content']
            elif msg['role'] == 'assistant' and user_msg:
                pairs.append({
                    "user": user_msg,
                    "assistant": msg['content']
                })
                user_msg = None

        return pairs

    def get_context_window(
        self,
        max_tokens: int = 4000,
        tokens_per_char: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Get messages that fit within token limit

        Args:
            max_tokens: Maximum tokens
            tokens_per_char: Rough estimate of tokens per character

        Returns:
            Messages within token limit
        """
        messages = []
        total_chars = 0
        max_chars = int(max_tokens / tokens_per_char)

        # Add messages from most recent backwards
        for msg in reversed(self.messages):
            msg_chars = len(msg['content'])
            if total_chars + msg_chars <= max_chars:
                messages.insert(0, msg)
                total_chars += msg_chars
            else:
                break

        self.logger.debug(f"Context window: {len(messages)} messages (~{total_chars} chars)")

        return messages

    def clear(self):
        """Clear all messages"""
        self.messages.clear()

        if self.persist_to_file:
            self._save_to_file()

        self.logger.info("Chat history cleared")

    def export_to_json(self, filepath: Union[str, Path]) -> str:
        """
        Export history to JSON file

        Args:
            filepath: Export file path

        Returns:
            Export file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "session_id": self.session_id,
            "session_start": self.session_start,
            "export_timestamp": datetime.now().isoformat(),
            "message_count": len(self.messages),
            "messages": list(self.messages)
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"History exported to: {filepath}")

        return str(filepath)

    def import_from_json(self, filepath: Union[str, Path]):
        """
        Import history from JSON file

        Args:
            filepath: Import file path
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Clear current history
        self.messages.clear()

        # Import messages
        for msg in data.get('messages', []):
            self.messages.append(msg)

        # Update session info if available
        if 'session_id' in data:
            self.session_id = data['session_id']
        if 'session_start' in data:
            self.session_start = data['session_start']

        self.logger.info(f"History imported from: {filepath} ({len(self.messages)} messages)")

    def get_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        messages = list(self.messages)

        if not messages:
            return {"message_count": 0}

        # Calculate statistics
        total_chars = sum(len(msg['content']) for msg in messages)
        user_messages = [msg for msg in messages if msg['role'] == 'user']
        assistant_messages = [msg for msg in messages if msg['role'] == 'assistant']

        stats = {
            "session_id": self.session_id,
            "session_start": self.session_start,
            "message_count": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "total_characters": total_chars,
            "average_message_length": total_chars / len(messages) if messages else 0,
            "first_message_time": messages[0]['timestamp'] if messages else None,
            "last_message_time": messages[-1]['timestamp'] if messages else None
        }

        return stats

    def get_summary(self) -> str:
        """Get text summary of conversation"""
        stats = self.get_statistics()

        summary = f"""Conversation Summary:
Session ID: {stats['session_id']}
Started: {stats.get('session_start', 'N/A')}
Total Messages: {stats['message_count']}
  - User: {stats.get('user_messages', 0)}
  - Assistant: {stats.get('assistant_messages', 0)}
Average Message Length: {stats.get('average_message_length', 0):.0f} characters
"""

        return summary.strip()

    def _save_to_file(self):
        """Save history to persistent file"""
        if not self.persist_to_file:
            return

        self.persist_to_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "session_id": self.session_id,
            "session_start": self.session_start,
            "last_updated": datetime.now().isoformat(),
            "messages": list(self.messages)
        }

        with open(self.persist_to_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_from_file(self):
        """Load history from persistent file"""
        if not self.persist_to_file or not self.persist_to_file.exists():
            return

        try:
            with open(self.persist_to_file, 'r') as f:
                data = json.load(f)

            self.messages.clear()
            for msg in data.get('messages', []):
                self.messages.append(msg)

            if 'session_id' in data:
                self.session_id = data['session_id']
            if 'session_start' in data:
                self.session_start = data['session_start']

            self.logger.info(f"History loaded from file: {len(self.messages)} messages")

        except Exception as e:
            self.logger.error(f"Failed to load history: {e}")

    def __len__(self) -> int:
        """Get number of messages"""
        return len(self.messages)

    def __repr__(self) -> str:
        return f"ChatHistory(session_id={self.session_id}, messages={len(self.messages)})"
