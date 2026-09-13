"""
app.database.manager — SQLite Database Manager
================================================

Encapsulates all database I/O behind a single class that owns the
connection and exposes CRUD methods for memories and visual
references.

Usage
-----
    from app.database.manager import DatabaseManager

    db = DatabaseManager("path/to/db.sqlite")
    db.initialize()                         # create tables
    mem_id = db.insert_memory(memory)       # insert
    memory = db.get_memory(mem_id)          # read
    db.update_memory(memory)                # update
    db.deactivate_memory(mem_id)            # soft-delete
    db.close()                              # clean shutdown

Design notes
------------
- All SQL uses parameterized queries (``?`` placeholders) to prevent
  injection.
- Timestamps are stored as ISO-8601 strings and parsed back to
  ``datetime`` objects on read.
- ``VisualReference`` stores file *paths*, not binary blobs.
- The manager is **not** thread-safe.  If multiple threads need DB
  access, each should own its own ``DatabaseManager`` instance
  (SQLite supports this in WAL mode).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.database.models import EntityType, Memory, VisualReference, EmbeddingRecord


# ── SQL Statements ────────────────────────────────────────────

_CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT    NOT NULL CHECK(entity_type IN ('person', 'object')),
    name        TEXT    NOT NULL,
    giver_name  TEXT,
    occasion    TEXT,
    year        TEXT,
    narrative   TEXT,
    is_active   INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
"""

_CREATE_VISUAL_REFERENCES_TABLE = """
CREATE TABLE IF NOT EXISTS visual_references (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id       INTEGER NOT NULL,
    image_path      TEXT    NOT NULL,
    embedding_path  TEXT,
    created_at      TEXT    NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memories(id)
        ON DELETE CASCADE
);
"""

_CREATE_INDEX_MEMORY_TYPE = """
CREATE INDEX IF NOT EXISTS idx_memories_entity_type
    ON memories(entity_type);
"""

_CREATE_INDEX_MEMORY_ACTIVE = """
CREATE INDEX IF NOT EXISTS idx_memories_is_active
    ON memories(is_active);
"""

_CREATE_INDEX_VISREF_MEMORY = """
CREATE INDEX IF NOT EXISTS idx_visual_references_memory_id
    ON visual_references(memory_id);
"""

_CREATE_EMBEDDING_RECORDS_TABLE = """
CREATE TABLE IF NOT EXISTS embedding_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id       INTEGER NOT NULL,
    model_name      TEXT    NOT NULL,
    embedding_dim   INTEGER NOT NULL,
    embedding_path  TEXT    NOT NULL,
    source_type     TEXT    NOT NULL CHECK(source_type IN ('face', 'clip_object', 'clip_scene')),
    created_at      TEXT    NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memories(id)
        ON DELETE CASCADE
);
"""

_CREATE_INDEX_EMB_MEMORY = """
CREATE INDEX IF NOT EXISTS idx_embedding_records_memory_id
    ON embedding_records(memory_id);
"""

_CREATE_INDEX_EMB_SOURCE = """
CREATE INDEX IF NOT EXISTS idx_embedding_records_source_type
    ON embedding_records(source_type);
"""


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 string back to a datetime, or None."""
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _row_to_memory(row: sqlite3.Row) -> Memory:
    """Convert a sqlite3.Row from the memories table to a Memory."""
    return Memory(
        id=row["id"],
        entity_type=EntityType(row["entity_type"]),
        name=row["name"],
        giver_name=row["giver_name"],
        occasion=row["occasion"],
        year=row["year"],
        narrative=row["narrative"],
        is_active=bool(row["is_active"]),
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def _row_to_visual_reference(row: sqlite3.Row) -> VisualReference:
    """Convert a sqlite3.Row to a VisualReference."""
    return VisualReference(
        id=row["id"],
        memory_id=row["memory_id"],
        image_path=row["image_path"],
        embedding_path=row["embedding_path"],
        created_at=_parse_dt(row["created_at"]),
    )


def _row_to_embedding_record(row: sqlite3.Row) -> EmbeddingRecord:
    """Convert a sqlite3.Row to an EmbeddingRecord."""
    return EmbeddingRecord(
        id=row["id"],
        memory_id=row["memory_id"],
        model_name=row["model_name"],
        embedding_dim=row["embedding_dim"],
        embedding_path=row["embedding_path"],
        source_type=row["source_type"],
        created_at=_parse_dt(row["created_at"]),
    )


class DatabaseManager:
    """SQLite database manager for the ReminisceCV memory store.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.  Use ``":memory:"`` for
        an ephemeral in-memory database (useful for testing).
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    # ── Connection lifecycle ──────────────────────────────────

    def initialize(self) -> None:
        """Open the database connection and create tables if needed.

        Safe to call multiple times — uses ``CREATE TABLE IF NOT EXISTS``.
        """
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        # Enable foreign key enforcement (off by default in SQLite)
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.execute("PRAGMA journal_mode = WAL;")

        self._conn.execute(_CREATE_MEMORIES_TABLE)
        self._conn.execute(_CREATE_VISUAL_REFERENCES_TABLE)
        self._conn.execute(_CREATE_EMBEDDING_RECORDS_TABLE)
        self._conn.execute(_CREATE_INDEX_MEMORY_TYPE)
        self._conn.execute(_CREATE_INDEX_MEMORY_ACTIVE)
        self._conn.execute(_CREATE_INDEX_VISREF_MEMORY)
        self._conn.execute(_CREATE_INDEX_EMB_MEMORY)
        self._conn.execute(_CREATE_INDEX_EMB_SOURCE)
        self._conn.commit()

    def close(self) -> None:
        """Commit pending changes and close the connection."""
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the active connection, raising if not initialized."""
        if self._conn is None:
            raise RuntimeError(
                "Database not initialized. Call initialize() first."
            )
        return self._conn

    # ── Memory CRUD ───────────────────────────────────────────

    def insert_memory(self, memory: Memory) -> int:
        """Insert a new memory and return its auto-generated ID.

        Parameters
        ----------
        memory : Memory
            The memory to insert.  ``id``, ``created_at``, and
            ``updated_at`` are set automatically.

        Returns
        -------
        int
            The new row ID.
        """
        now = _now_iso()
        cursor = self.connection.execute(
            """
            INSERT INTO memories
                (entity_type, name, giver_name, occasion, year,
                 narrative, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.entity_type.value,
                memory.name,
                memory.giver_name,
                memory.occasion,
                memory.year,
                memory.narrative,
                int(memory.is_active),
                now,
                now,
            ),
        )
        self.connection.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_memory(self, memory_id: int) -> Optional[Memory]:
        """Retrieve a single memory by ID, or None if not found."""
        row = self.connection.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_memory(row)

    def get_all_memories(self, active_only: bool = True) -> list[Memory]:
        """Retrieve all memories, optionally filtering by active status.

        Parameters
        ----------
        active_only : bool
            If True (default), only return active memories.
        """
        if active_only:
            rows = self.connection.execute(
                "SELECT * FROM memories WHERE is_active = 1 ORDER BY id"
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM memories ORDER BY id"
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def get_memories_by_type(
        self, entity_type: EntityType, active_only: bool = True
    ) -> list[Memory]:
        """Retrieve memories filtered by entity type.

        Parameters
        ----------
        entity_type : EntityType
            Filter to ``"person"`` or ``"object"``.
        active_only : bool
            If True, exclude deactivated memories.
        """
        if active_only:
            rows = self.connection.execute(
                "SELECT * FROM memories WHERE entity_type = ? AND is_active = 1 ORDER BY id",
                (entity_type.value,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM memories WHERE entity_type = ? ORDER BY id",
                (entity_type.value,),
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def update_memory(self, memory: Memory) -> bool:
        """Update an existing memory.  Returns True if a row was updated.

        Parameters
        ----------
        memory : Memory
            Must have a valid ``id``.  ``updated_at`` is set
            automatically.
        """
        if memory.id is None:
            raise ValueError("Cannot update a memory without an ID.")
        now = _now_iso()
        cursor = self.connection.execute(
            """
            UPDATE memories SET
                entity_type = ?,
                name        = ?,
                giver_name  = ?,
                occasion    = ?,
                year        = ?,
                narrative   = ?,
                is_active   = ?,
                updated_at  = ?
            WHERE id = ?
            """,
            (
                memory.entity_type.value,
                memory.name,
                memory.giver_name,
                memory.occasion,
                memory.year,
                memory.narrative,
                int(memory.is_active),
                now,
                memory.id,
            ),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def deactivate_memory(self, memory_id: int) -> bool:
        """Soft-delete a memory by setting is_active = 0.

        Returns True if a row was affected.
        """
        now = _now_iso()
        cursor = self.connection.execute(
            "UPDATE memories SET is_active = 0, updated_at = ? WHERE id = ?",
            (now, memory_id),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def activate_memory(self, memory_id: int) -> bool:
        """Re-activate a previously deactivated memory.

        Returns True if a row was affected.
        """
        now = _now_iso()
        cursor = self.connection.execute(
            "UPDATE memories SET is_active = 1, updated_at = ? WHERE id = ?",
            (now, memory_id),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def delete_memory(self, memory_id: int) -> bool:
        """Permanently delete a memory and its visual references.

        Returns True if a row was deleted.  Use ``deactivate_memory``
        for a non-destructive alternative.
        """
        # Foreign key ON DELETE CASCADE removes visual_references
        cursor = self.connection.execute(
            "DELETE FROM memories WHERE id = ?", (memory_id,)
        )
        self.connection.commit()
        return cursor.rowcount > 0

    # ── Visual Reference CRUD ─────────────────────────────────

    def insert_visual_reference(self, ref: VisualReference) -> int:
        """Insert a visual reference and return its ID.

        Parameters
        ----------
        ref : VisualReference
            Must have a valid ``memory_id``.
        """
        now = _now_iso()
        cursor = self.connection.execute(
            """
            INSERT INTO visual_references
                (memory_id, image_path, embedding_path, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (ref.memory_id, ref.image_path, ref.embedding_path, now),
        )
        self.connection.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_visual_references(self, memory_id: int) -> list[VisualReference]:
        """Retrieve all visual references for a given memory."""
        rows = self.connection.execute(
            "SELECT * FROM visual_references WHERE memory_id = ? ORDER BY id",
            (memory_id,),
        ).fetchall()
        return [_row_to_visual_reference(r) for r in rows]

    def delete_visual_reference(self, ref_id: int) -> bool:
        """Delete a single visual reference by ID.

        Returns True if a row was deleted.
        """
        cursor = self.connection.execute(
            "DELETE FROM visual_references WHERE id = ?", (ref_id,)
        )
        self.connection.commit()
        return cursor.rowcount > 0

    # ── Embedding Record CRUD ─────────────────────────────────

    def insert_embedding_record(self, record: EmbeddingRecord) -> int:
        """Insert an embedding metadata record and return its ID."""
        now = _now_iso()
        cursor = self.connection.execute(
            """
            INSERT INTO embedding_records
                (memory_id, model_name, embedding_dim,
                 embedding_path, source_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.memory_id,
                record.model_name,
                record.embedding_dim,
                record.embedding_path,
                record.source_type,
                now,
            ),
        )
        self.connection.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_embedding_records(
        self, memory_id: int, source_type: Optional[str] = None
    ) -> list[EmbeddingRecord]:
        """Retrieve embedding records for a memory.

        Parameters
        ----------
        memory_id : int
            The memory to look up.
        source_type : str or None
            If given, filter by source type (e.g. ``"face"``).
        """
        if source_type is not None:
            rows = self.connection.execute(
                "SELECT * FROM embedding_records "
                "WHERE memory_id = ? AND source_type = ? ORDER BY id",
                (memory_id, source_type),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM embedding_records WHERE memory_id = ? ORDER BY id",
                (memory_id,),
            ).fetchall()
        return [_row_to_embedding_record(r) for r in rows]

    def get_all_embedding_records(
        self, source_type: Optional[str] = None
    ) -> list[EmbeddingRecord]:
        """Retrieve all embedding records, optionally by source type."""
        if source_type is not None:
            rows = self.connection.execute(
                "SELECT * FROM embedding_records WHERE source_type = ? ORDER BY id",
                (source_type,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM embedding_records ORDER BY id"
            ).fetchall()
        return [_row_to_embedding_record(r) for r in rows]

    def delete_embedding_record(self, record_id: int) -> bool:
        """Delete an embedding record by ID.  Returns True if deleted."""
        cursor = self.connection.execute(
            "DELETE FROM embedding_records WHERE id = ?", (record_id,)
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def delete_embedding_records_for_memory(self, memory_id: int) -> int:
        """Delete all embedding records for a memory.  Returns count deleted."""
        cursor = self.connection.execute(
            "DELETE FROM embedding_records WHERE memory_id = ?", (memory_id,)
        )
        self.connection.commit()
        return cursor.rowcount

    # ── Statistics ────────────────────────────────────────────

    def count_memories(self, active_only: bool = True) -> int:
        """Return the total number of memories."""
        if active_only:
            row = self.connection.execute(
                "SELECT COUNT(*) AS cnt FROM memories WHERE is_active = 1"
            ).fetchone()
        else:
            row = self.connection.execute(
                "SELECT COUNT(*) AS cnt FROM memories"
            ).fetchone()
        return row["cnt"]

    def count_visual_references(self) -> int:
        """Return the total number of visual references."""
        row = self.connection.execute(
            "SELECT COUNT(*) AS cnt FROM visual_references"
        ).fetchone()
        return row["cnt"]

    def count_embedding_records(self) -> int:
        """Return the total number of embedding records."""
        row = self.connection.execute(
            "SELECT COUNT(*) AS cnt FROM embedding_records"
        ).fetchone()
        return row["cnt"]
