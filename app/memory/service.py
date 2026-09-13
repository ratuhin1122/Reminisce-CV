"""
app.memory.service — Memory Service Layer
===========================================

Business logic on top of the ``MemoryRepository``.  Provides
higher-level operations that combine multiple repository calls
and enforce validation rules.

The service is the entry point for all memory-related operations
from the UI, CLI, or recognition engine.

Usage
-----
    from app.memory.service import MemoryService

    service = MemoryService(db_path=config.db_path)
    service.start()

    mem_id = service.register_object(
        name="Watch",
        giver_name="Grandmother",
        image_path="/photos/watch.jpg",
    )
    service.stop()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.database.models import EmbeddingRecord, EntityType, Memory, VisualReference
from app.memory.repository import MemoryRepository


@dataclass
class MemoryWithReferences:
    """A memory bundled with its visual references and embeddings.

    Convenience container returned by service methods when the caller
    needs the full picture for a single memory.
    """

    memory: Memory
    references: list[VisualReference]
    embeddings: list[EmbeddingRecord]


class MemoryService:
    """High-level service for memory management.

    Combines repository operations with validation, bundling, and
    business rules.  This is the class that the rest of the
    application should use for memory operations.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite file, or ``":memory:"`` for testing.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._repo = MemoryRepository(db_path)

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        """Open the database connection."""
        self._repo.open()

    def stop(self) -> None:
        """Close the database connection."""
        self._repo.close()

    @property
    def repository(self) -> MemoryRepository:
        """Direct access to the repository (for advanced use)."""
        return self._repo

    # ── Registration (create + reference in one call) ─────────

    def register_object(
        self,
        name: str,
        giver_name: Optional[str] = None,
        occasion: Optional[str] = None,
        year: Optional[str] = None,
        narrative: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> int:
        """Register a new object memory, optionally with a reference image.

        Parameters
        ----------
        name : str
            Display name for the object.
        giver_name, occasion, year, narrative : str, optional
            Additional memory metadata.
        image_path : str, optional
            If provided, registers this as a reference image.

        Returns
        -------
        int
            The new memory ID.

        Raises
        ------
        ValueError
            If ``name`` is empty.
        """
        if not name or not name.strip():
            raise ValueError("Memory name cannot be empty.")

        mem_id = self._repo.create_object_memory(
            name=name.strip(),
            giver_name=giver_name,
            occasion=occasion,
            year=year,
            narrative=narrative,
        )

        if image_path:
            self._repo.add_reference_image(mem_id, image_path)

        return mem_id

    def register_person(
        self,
        name: str,
        giver_name: Optional[str] = None,
        occasion: Optional[str] = None,
        year: Optional[str] = None,
        narrative: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> int:
        """Register a new person memory, optionally with a reference image.

        Parameters
        ----------
        name : str
            Display name for the person.
        giver_name, occasion, year, narrative : str, optional
            Additional memory metadata.
        image_path : str, optional
            If provided, registers this as a reference image.

        Returns
        -------
        int
            The new memory ID.

        Raises
        ------
        ValueError
            If ``name`` is empty.
        """
        if not name or not name.strip():
            raise ValueError("Memory name cannot be empty.")

        mem_id = self._repo.create_person_memory(
            name=name.strip(),
            giver_name=giver_name,
            occasion=occasion,
            year=year,
            narrative=narrative,
        )

        if image_path:
            self._repo.add_reference_image(mem_id, image_path)

        return mem_id

    # ── Retrieval ─────────────────────────────────────────────

    def get_memory(self, memory_id: int) -> Optional[Memory]:
        """Get a single memory by ID."""
        return self._repo.get_memory(memory_id)

    def get_full_memory(self, memory_id: int) -> Optional[MemoryWithReferences]:
        """Get a memory bundled with all its references and embeddings.

        Returns None if the memory does not exist.
        """
        memory = self._repo.get_memory(memory_id)
        if memory is None:
            return None
        refs = self._repo.get_reference_images(memory_id)
        embs = self._repo.get_embeddings(memory_id)
        return MemoryWithReferences(
            memory=memory, references=refs, embeddings=embs
        )

    def get_all_active(self) -> list[Memory]:
        """Get all active memories."""
        return self._repo.get_all_active_memories()

    def get_all_people(self) -> list[Memory]:
        """Get all active person memories."""
        return self._repo.get_people()

    def get_all_objects(self) -> list[Memory]:
        """Get all active object memories."""
        return self._repo.get_objects()

    # ── Update ────────────────────────────────────────────────

    def update_memory(self, memory: Memory) -> bool:
        """Update a memory's metadata.  Returns True if successful."""
        if memory.id is None:
            raise ValueError("Cannot update a memory without an ID.")
        if not memory.name or not memory.name.strip():
            raise ValueError("Memory name cannot be empty.")
        memory.name = memory.name.strip()
        return self._repo.update_memory(memory)

    def deactivate_memory(self, memory_id: int) -> bool:
        """Soft-delete a memory."""
        return self._repo.deactivate_memory(memory_id)

    def activate_memory(self, memory_id: int) -> bool:
        """Re-activate a deactivated memory."""
        return self._repo.activate_memory(memory_id)

    def delete_memory(self, memory_id: int) -> bool:
        """Permanently delete a memory and all associated data."""
        return self._repo.delete_memory(memory_id)

    # ── Reference images ──────────────────────────────────────

    def add_reference_image(
        self,
        memory_id: int,
        image_path: str,
        embedding_path: Optional[str] = None,
    ) -> int:
        """Add a reference image to an existing memory.

        Raises
        ------
        ValueError
            If the memory does not exist.
        """
        memory = self._repo.get_memory(memory_id)
        if memory is None:
            raise ValueError(f"Memory {memory_id} does not exist.")
        return self._repo.add_reference_image(
            memory_id, image_path, embedding_path
        )

    def get_reference_images(self, memory_id: int) -> list[VisualReference]:
        """Get all reference images for a memory."""
        return self._repo.get_reference_images(memory_id)

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

        Raises
        ------
        ValueError
            If the memory does not exist.
        """
        memory = self._repo.get_memory(memory_id)
        if memory is None:
            raise ValueError(f"Memory {memory_id} does not exist.")
        return self._repo.store_embedding(
            memory_id=memory_id,
            model_name=model_name,
            embedding_dim=embedding_dim,
            embedding_path=embedding_path,
            source_type=source_type,
        )

    def get_embeddings(
        self, memory_id: int, source_type: Optional[str] = None
    ) -> list[EmbeddingRecord]:
        """Retrieve embedding records for a memory."""
        return self._repo.get_embeddings(memory_id, source_type)

    def get_all_embeddings(
        self, source_type: Optional[str] = None
    ) -> list[EmbeddingRecord]:
        """Retrieve all embedding records, optionally filtered."""
        return self._repo.get_all_embeddings(source_type)

    # ── Statistics ────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        """Return a summary of database contents."""
        return {
            "active_memories": self._repo.count_memories(),
            "total_memories": self._repo.count_memories(include_inactive=True),
            "reference_images": self._repo.count_reference_images(),
            "embeddings": self._repo.count_embeddings(),
        }
