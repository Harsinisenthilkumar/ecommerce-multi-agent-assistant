import re
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path

from agents.conversation_memory import ConversationMemoryStore


def fake_embedding(text: str) -> list[float]:
    words = set(re.findall(r"\w+", text.lower()))
    return [
        float("sku101" in words or "101" in words),
        float("sku102" in words or "102" in words),
        float("order" in words),
    ]


class ConversationMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = ConversationMemoryStore(
            db_path=Path(self.temp_dir.name) / "memory.sqlite3",
            embed_text=fake_embedding,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_retrieves_relevant_turns_only_from_requested_conversation(self):
        self.store.save_turn(
            "conversation-a",
            "Check stock for SKU-101.",
            "SKU-101 has 15 units.",
        )
        self.store.save_turn(
            "conversation-a",
            "Track my order.",
            "The order is in transit.",
        )
        self.store.save_turn(
            "conversation-b",
            "Check stock for SKU-101.",
            "SKU-101 has 8 units.",
        )

        memories = self.store.retrieve(
            "conversation-a",
            "How many are left for SKU-101?",
        )

        self.assertEqual(
            memories,
            ["User: Check stock for SKU-101.\nAssistant: SKU-101 has 15 units."],
        )
        self.assertEqual(self.store.retrieve("conversation-c", "SKU-101"), [])

    def test_conversations_in_the_same_room_are_isolated(self):
        self.store.save_turn("session-a", "SKU-101 stock", "15 units.")
        self.store.save_turn("session-b", "SKU-101 stock", "8 units.")

        self.assertEqual(
            self.store.retrieve("session-a", "SKU-101"),
            ["User: SKU-101 stock\nAssistant: 15 units."],
        )
        self.assertEqual(
            self.store.retrieve("session-b", "SKU-101"),
            ["User: SKU-101 stock\nAssistant: 8 units."],
        )

    def test_clear_conversation_deletes_only_that_conversation(self):
        self.store.save_turn("conversation-a", "SKU-101 stock", "15 units.")
        self.store.save_turn("conversation-b", "SKU-102 stock", "0 units.")

        self.assertEqual(self.store.clear_conversation("conversation-a"), 1)
        self.assertEqual(self.store.retrieve("conversation-a", "SKU-101"), [])
        self.assertEqual(
            self.store.retrieve("conversation-b", "SKU-102"),
            ["User: SKU-102 stock\nAssistant: 0 units."],
        )

    def test_migrates_legacy_room_column_without_losing_memories(self):
        legacy_path = Path(self.temp_dir.name) / "legacy.sqlite3"
        connection = sqlite3.connect(legacy_path)
        with connection:
            connection.execute(
                """
                CREATE TABLE conversation_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    room_name TEXT NOT NULL,
                    user_text TEXT NOT NULL,
                    assistant_text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX conversation_memories_by_room
                ON conversation_memories (room_name, id DESC)
                """
            )
            connection.execute(
                """
                INSERT INTO conversation_memories (
                    room_name, user_text, assistant_text, embedding
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    "legacy-room",
                    "SKU-101 stock",
                    "15 units.",
                    struct.pack("3f", *fake_embedding("SKU-101 stock 15 units")),
                ),
            )
        connection.close()

        legacy_store = ConversationMemoryStore(
            db_path=legacy_path,
            embed_text=fake_embedding,
        )
        self.assertEqual(
            legacy_store.retrieve("legacy-room", "SKU-101"),
            ["User: SKU-101 stock\nAssistant: 15 units."],
        )
        self.assertEqual(legacy_store.clear_room("legacy-room"), 1)


if __name__ == "__main__":
    unittest.main()
