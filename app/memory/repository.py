"""
app.memory.repository — Memory Repository
===========================================

Typed repository that wraps ``DatabaseManager`` and provides a
high-level, SQL-free API for the rest of the application.

The repository is the *only* gateway between business logic and the
database.  No other module should import ``DatabaseManager`` directly.

Usage
-----
    from app.memory.repository import MemoryRepository

    repo = MemoryRepository(db_path=":memory:")
    repo.open()

    mem_id = repo.create_object_memory(
        name="Watch",
        giver_name="Grandmother",
        occasion="Birthday",
        year="2015",
        narrative="A silver pocket watch.",
    )
    memory = repo.get_memory(mem_id)
    repo.close()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.database.manager import DatabaseManager
from app.database.models import (
    EmbeddingRecord,
    EntityType,
    Memory,
    VisualReference,
)


class MemoryRepository:
    """High-level repository for memory persistence.

    Wraps ``DatabaseManager`` and exposes typed convenience methods
    so the rest of the application never writes SQL.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite file, or ``":memory:"`` for testing.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db = DatabaseManager(db_path)

    # ── Lifecycle ─────────────────────────────────────────────

    def open(self) -> None:
        """Open the database and ensure all tables exist."""
        self._db.initialize()

    def close(self) -> None:
        """Flush and close the database connection."""
        self._db.close()

    # ── Create memories ───────────────────────────────────────

    def create_object_memory(
        self,
        name: str,
        giver_name: Optional[str] = None,
        occasion: Optional[str] = None,
        year: Optional[str] = None,
        narrative: Optional[str] = None,
    ) -> int:
        """Create a new *object* memory and return its ID.

        Parameters
        ----------
        name : str
            Display name of the object (e.g. "Grandmother's Watch").
        giver_name : str, optional
            Who gave the object.
        occasion : str, optional
            The occasion (e.g. "Birthday").
        year : str, optional
            Year or date string.
        narrative : str, optional
            A personal story or description.
        """
        memory = Memory(
            entity_type=EntityType.OBJECT,
            name=name,
            giver_name=giver_name,
            occasion=occasion,
            year=year,
            narrative=narrative,
        )
        return self._db.insert_memory(memory)

    def create_person_memory(
        self,
        name: str,
        giver_name: Optional[str] = None,
        occasion: Optional[str] = None,
        year: Optional[str] = None,
        narrative: Optional[str] = None,
    ) -> int:
        """Create a new *person* memory and return its ID.

        Parameters
        ----------
        name : str
            Display name of the person (e.g. "Dr. Smith").
        giver_name : str, optional
            Full name or alias.
        occasion : str, optional
            Relationship context (e.g. "Research Advisor").
        year : str, optional
            Year they were first met/known.
        narrative : str, optional
            Personal context about this person.
        """
        memory = Memory(
            entity_type=EntityType.PERSON,
            name=name,
            giver_name=giver_name,
            occasion=occasion,
            year=year,
            narrative=narrative,
        )
        return self._db.insert_memory(memory)

    # ── Read memories ─────────────────────────────────────────

    def get_memory(self, memory_id: int) -> Optional[Memory]:
        """Get a single memory by ID, or None if not found."""
        return self._db.get_memory(memory_id)

    def get_all_active_memories(self) -> list[Memory]:
        """Get all active (non-deactivated) memories."""
        return self._db.get_all_memories(active_only=True)

    def get_all_memories(self, include_inactive: bool = False) -> list[Memory]:
        """Get all memories, optionally including deactivated ones."""
        return self._db.get_all_memories(active_only=not include_inactive)

    def get_people(self, include_inactive: bool = False) -> list[Memory]:
        """Get all person-type memories."""
        return self._db.get_memories_by_type(
            EntityType.PERSON, active_only=not include_inactive
        )

    def get_objects(self, include_inactive: bool = False) -> list[Memory]:
        """Get all object-type memories."""
        return self._db.get_memories_by_type(
            EntityType.OBJECT, active_only=not include_inactive
        )

    # ── Update memories ───────────────────────────────────────

    def update_memory(self, memory: Memory) -> bool:
        """Update an existing memory.  Returns True if successful."""
        return self._db.update_memory(memory)

    def deactivate_memory(self, memory_id: int) -> bool:
        """Soft-delete a memory.  Returns True if successful."""
        return self._db.deactivate_memory(memory_id)

    def activate_memory(self, memory_id: int) -> bool:
        """Re-activate a deactivated memory.  Returns True if successful."""
        return self._db.activate_memory(memory_id)

    def delete_memory(self, memory_id: int) -> bool:
        """Permanently delete a memory and all associated data.

        Cascades to visual references and embedding records.
        """
        return self._db.delete_memory(memory_id)

    # ── Reference images ──────────────────────────────────────

    def add_reference_image(
        self,
        memory_id: int,
        image_path: str,
        embedding_path: Optional[str] = None,
    ) -> int:
        """Register a reference image for a memory.

        Parameters
        ----------
        memory_id : int
            The memory this image belongs to.
        image_path : str
            Path to the image file on disk.
        embedding_path : str, optional
            Path to a precomputed embedding file.

        Returns
        -------
        int
            The new visual reference ID.
        """
        ref = VisualReference(
            memory_id=memory_id,
            image_path=image_path,
            embedding_path=embedding_path,
        )
        return self._db.insert_visual_reference(ref)

    def get_reference_images(self, memory_id: int) -> list[VisualReference]:
        """Get all reference images for a memory."""
        return self._db.get_visual_references(memory_id)

    def delete_reference_image(self, ref_id: int) -> bool:
        """Delete a single reference image.  Returns True if deleted."""
        return self._db.delete_visual_reference(ref_id)

    # ── Embedding metadata ────────────────────────────────────

    def store_embedding(
        self,
        memory_id: int,
        model_name: str,
        embedding_dim: int,
        embedding_path: str,
        source_type: str = "clip_object",
    ) -> int:
        """Store embedding metadata for a memory.

        Parameters
        ----------
        memory_id : int
            The memory this embedding belongs to.
        model_name : str
            Name of the model that produced the embedding.
        embedding_dim : int
            Dimensionality of the embedding vector.
        embedding_path : str
            Path to the serialized ``.npy`` file.
        source_type : str
            ``"face"``, ``"clip_object"``, or ``"clip_scene"``.

        Returns
        -------
        int
            The new embedding record ID.
        """
        record = EmbeddingRecord(
            memory_id=memory_id,
            model_name=model_name,
            embedding_dim=embedding_dim,
            embedding_path=embedding_path,
            source_type=source_type,
        )
        return self._db.insert_embedding_record(record)

    def get_embeddings(
        self, memory_id: int, source_type: Optional[str] = None
    ) -> list[EmbeddingRecord]:
        """Retrieve embedding records for a memory.

        Parameters
        ----------
        memory_id : int
            The memory to look up.
        source_type : str, optional
            Filter by source type (e.g. ``"face"``).
        """
        return self._db.get_embedding_records(memory_id, source_type)

    def get_all_embeddings(
        self, source_type: Optional[str] = None
    ) -> list[EmbeddingRecord]:
        """Retrieve all embedding records, optionally filtered."""
        return self._db.get_all_embedding_records(source_type)

    def delete_embedding(self, record_id: int) -> bool:
        """Delete a single embedding record.  Returns True if deleted."""
        return self._db.delete_embedding_record(record_id)

    # ── Statistics ────────────────────────────────────────────

    def count_memories(self, include_inactive: bool = False) -> int:
        """Count memories in the database."""
        return self._db.count_memories(active_only=not include_inactive)

    def count_reference_images(self) -> int:
        """Count visual references in the database."""
        return self._db.count_visual_references()

    def count_embeddings(self) -> int:
        """Count embedding records in the database."""
        return self._db.count_embedding_records()
