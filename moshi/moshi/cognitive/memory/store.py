# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Durable episodic + semantic memory storage (PHASES.md Phase 3.2/3.3).

SQLite (stdlib, no new dependency) rather than in-process state: memory is
supposed to survive process restarts and outlive any single conversation —
that's the entire point of distinguishing it from working memory
(``WorkingMemory`` / ``turn_manager.conversation_context``), which is
correctly ephemeral.

``user_id`` is a placeholder identity key: this repository has no
authentication layer (see the earlier execution audit), so there is no real
notion of "the same user across sessions" yet. Callers currently have nothing
better to pass than a per-connection id, which makes memory effectively
per-conversation until an identity system exists — that limitation lives in
the caller, not here; this store will do the right thing the moment a real
user id is available.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodicRecord:
    id: int
    user_id: str
    turn_index: int
    text: str
    salience: float
    created_at: float


@dataclass(frozen=True)
class SemanticFact:
    id: int
    user_id: str
    key: str
    value: str
    confidence: float
    created_at: float
    updated_at: float


class MemoryStore:
    """SQLite-backed store for episodic turns and semantic facts.

    Pass ``db_path=":memory:"`` (the default) for tests/ephemeral use, or a
    real file path for durability across restarts. Not safe for concurrent
    writers from multiple processes — one store per server process, matching
    how every other piece of per-process state in this codebase works.
    """

    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                salience REAL NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_episodic_user ON episodic(user_id, created_at)")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(user_id, key)
            )
            """
        )
        self._conn.commit()

    # -- Episodic -----------------------------------------------------------

    def add_episode(self, user_id: str, turn_index: int, text: str, salience: float, now: float) -> int:
        cur = self._conn.execute(
            "INSERT INTO episodic (user_id, turn_index, text, salience, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, turn_index, text, salience, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def recent_episodes(self, user_id: str, limit: int = 10) -> list[EpisodicRecord]:
        rows = self._conn.execute(
            "SELECT * FROM episodic WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [EpisodicRecord(**dict(row)) for row in rows]

    def search_episodes(self, user_id: str, keyword: str, limit: int = 10) -> list[EpisodicRecord]:
        """Substring search over stored episodes for this user.

        Plain ``LIKE``, not semantic search — a real implementation would embed
        episodes and the query and rank by similarity. This is enough to prove
        the storage/retrieval/merge machinery end to end.
        """
        if not keyword:
            return []
        rows = self._conn.execute(
            "SELECT * FROM episodic WHERE user_id = ? AND text LIKE ? ORDER BY created_at DESC LIMIT ?",
            (user_id, f"%{keyword}%", limit),
        ).fetchall()
        return [EpisodicRecord(**dict(row)) for row in rows]

    # -- Semantic -------------------------------------------------------------

    def upsert_fact(self, user_id: str, key: str, value: str, confidence: float, now: float) -> None:
        self._conn.execute(
            """
            INSERT INTO semantic (user_id, key, value, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET
                value = excluded.value,
                confidence = excluded.confidence,
                updated_at = excluded.updated_at
            """,
            (user_id, key, value, confidence, now, now),
        )
        self._conn.commit()

    def get_facts(self, user_id: str) -> list[SemanticFact]:
        rows = self._conn.execute(
            "SELECT * FROM semantic WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        return [SemanticFact(**dict(row)) for row in rows]

    def get_fact(self, user_id: str, key: str) -> SemanticFact | None:
        row = self._conn.execute(
            "SELECT * FROM semantic WHERE user_id = ? AND key = ?",
            (user_id, key),
        ).fetchone()
        return SemanticFact(**dict(row)) if row is not None else None

    def close(self) -> None:
        self._conn.close()
