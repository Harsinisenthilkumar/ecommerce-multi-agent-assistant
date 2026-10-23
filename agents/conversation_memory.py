import argparse
import os
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Generator

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
MAX_SEARCH_CANDIDATES = 2000


class ConversationMemoryError(RuntimeError):
    """Raised when persistent conversation retrieval or storage is unavailable."""


@lru_cache(maxsize=1)
def _get_embedding_model():
    try:
        from fastembed import TextEmbedding

        cache_path = Path(
            os.getenv(
                "CONVERSATION_EMBEDDING_CACHE",
                Path.home() / ".cache" / "ecommerce-multi-agent" / "fastembed",
            )
        )
        cache_path.mkdir(parents=True, exist_ok=True)
        return TextEmbedding(
            model_name=DEFAULT_MODEL,
            cache_dir=str(cache_path),
        )
    except Exception as exc:
        raise ConversationMemoryError(
            "Could not load the local conversation embedding model."
        ) from exc


def _embed_text(text: str) -> np.ndarray:
    try:
        vectors = list(_get_embedding_model().embed([text]))
    except ConversationMemoryError:
        raise
    except Exception as exc:
        raise ConversationMemoryError(
            "Could not create a local conversation embedding."
        ) from exc
    if len(vectors) != 1:
        raise ConversationMemoryError("The embedding model returned no vector.")
    vector = np.asarray(vectors[0], dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ConversationMemoryError("The embedding model returned an invalid vector.")
    return vector


class ConversationMemoryStore:
    def __init__(
        self,
        db_path: Path | None = None,
        embed_text: Callable[[str], Sequence[float]] = _embed_text,
    ) -> None:
        configured_path = os.getenv("CONVERSATION_MEMORY_DB")
        self.db_path = db_path or (
            Path(configured_path)
            if configured_path
            else Path(__file__).resolve().parents[1]
            / "data"
            / "conversation_memory.sqlite3"
        )
        self._embed_text = embed_text

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.db_path, timeout=10)
        except (OSError, sqlite3.Error) as exc:
            raise ConversationMemoryError(
                "Could not open the local conversation memory database."
            ) from exc

        try:
            connection.row_factory = sqlite3.Row
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    user_text TEXT NOT NULL,
                    assistant_text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(conversation_memories)"
                )
            }
            if "room_name" in columns and "conversation_id" not in columns:
                connection.execute(
                    "ALTER TABLE conversation_memories "
                    "RENAME COLUMN room_name TO conversation_id"
                )
                connection.execute(
                    "DROP INDEX IF EXISTS conversation_memories_by_room"
                )
            indexes = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA index_list(conversation_memories)"
                )
            }
            if "conversation_memories_by_conversation" not in indexes:
                connection.execute(
                    """
                    CREATE INDEX conversation_memories_by_conversation
                    ON conversation_memories (conversation_id, id DESC)
                    """
                )
            yield connection
            connection.commit()
        except sqlite3.Error as exc:
            connection.rollback()
            raise ConversationMemoryError(
                "Could not read or write the local conversation memory database."
            ) from exc
        finally:
            connection.close()

    @staticmethod
    def _validate_conversation(conversation_id: str) -> None:
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ValueError("conversation_id must be a non-empty string.")

    def retrieve(
        self,
        conversation_id: str,
        query: str,
        limit: int = 3,
    ) -> list[str]:
        self._validate_conversation(conversation_id)
        if not query.strip():
            return []
        if limit <= 0:
            raise ValueError("limit must be positive.")

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT user_text, assistant_text, embedding
                FROM conversation_memories
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, MAX_SEARCH_CANDIDATES),
            ).fetchall()
        if not rows:
            return []

        query_vector = np.asarray(self._embed_text(query), dtype=np.float32)
        if query_vector.ndim != 1 or query_vector.size == 0:
            raise ConversationMemoryError("The query embedding is invalid.")
        query_norm = float(np.linalg.norm(query_vector))
        if not np.isfinite(query_norm) or query_norm == 0:
            raise ConversationMemoryError("The query embedding has an invalid norm.")

        scored_rows: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            memory_vector = np.frombuffer(row["embedding"], dtype=np.float32)
            if memory_vector.shape != query_vector.shape:
                raise ConversationMemoryError(
                    "Stored memories use a different embedding model. "
                    "Clear this conversation's memories before changing models."
                )
            memory_norm = float(np.linalg.norm(memory_vector))
            if not np.isfinite(memory_norm) or memory_norm == 0:
                continue
            similarity = float(
                np.dot(query_vector, memory_vector) / (query_norm * memory_norm)
            )
            if similarity >= 0.2:
                scored_rows.append((similarity, row))

        scored_rows.sort(key=lambda item: item[0], reverse=True)
        return [
            f"User: {row['user_text']}\nAssistant: {row['assistant_text']}"
            for _, row in scored_rows[:limit]
        ]

    def save_turn(
        self,
        conversation_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        self._validate_conversation(conversation_id)
        if not user_text.strip() or not assistant_text.strip():
            raise ValueError("user_text and assistant_text must be non-empty.")

        passage = f"User: {user_text.strip()}\nAssistant: {assistant_text.strip()}"
        vector = np.asarray(self._embed_text(passage), dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ConversationMemoryError("The conversation embedding is invalid.")

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO conversation_memories (
                    conversation_id, user_text, assistant_text, embedding
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    user_text.strip(),
                    assistant_text.strip(),
                    vector.tobytes(),
                ),
            )

    def clear_room(self, room_name: str) -> int:
        self._validate_conversation(room_name)
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM conversation_memories WHERE conversation_id = ?",
                (room_name,),
            )
            return cursor.rowcount

    def clear_conversation(self, conversation_id: str) -> int:
        self._validate_conversation(conversation_id)
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM conversation_memories WHERE conversation_id = ?",
                (conversation_id,),
            )
            return cursor.rowcount


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage persistent conversation memories."
    )
    clear_options = parser.add_mutually_exclusive_group(required=True)
    clear_options.add_argument(
        "--clear-conversation",
        help="Delete all stored memories for this conversation ID.",
    )
    clear_options.add_argument(
        "--clear-room",
        help="Delete legacy room-scoped memories for this room name.",
    )
    args = parser.parse_args()
    store = ConversationMemoryStore()
    if args.clear_conversation:
        deleted = store.clear_conversation(args.clear_conversation)
        print(
            "Deleted "
            f"{deleted} conversation memory record(s) for "
            f"conversation {args.clear_conversation}."
        )
        return

    deleted = store.clear_room(args.clear_room)
    print(
        f"Deleted {deleted} legacy conversation memory record(s) "
        f"for room {args.clear_room}."
    )


if __name__ == "__main__":
    main()
