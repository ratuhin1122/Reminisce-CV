"""
tests.test_database — Unit Tests for the SQLite Memory Database
================================================================

Tests cover:
    - Database creation and schema initialization
    - Memory CRUD: insert, retrieve, update, deactivate, delete
    - Visual reference CRUD: insert, retrieve, delete
    - Edge cases: not-found, cascade delete, re-activate
    - Filtering by entity type and active status
    - Statistics queries

All tests use an in-memory SQLite database for speed and isolation.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from app.database import DatabaseManager, EntityType, Memory, VisualReference


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def db() -> DatabaseManager:
    """Provide a fresh in-memory database for each test."""
    manager = DatabaseManager(":memory:")
    manager.initialize()
    yield manager
    manager.close()


@pytest.fixture
def sample_object_memory() -> Memory:
    """A sample object memory for testing."""
    return Memory(
        entity_type=EntityType.OBJECT,
        name="Grandmother's Watch",
        giver_name="Grandmother",
        occasion="18th Birthday",
        year="2015",
        narrative="A silver pocket watch she gave me on my 18th birthday.",
    )


@pytest.fixture
def sample_person_memory() -> Memory:
    """A sample person memory for testing."""
    return Memory(
        entity_type=EntityType.PERSON,
        name="Dr. Smith",
        giver_name="Dr. John Smith",
        occasion="Research Advisor",
        year="2020",
        narrative="My PhD research advisor at the university.",
    )


# ── Schema / Initialization ──────────────────────────────────

class TestDatabaseInitialization:
    """Verify database creation and schema setup."""

    def test_initialize_creates_tables(self, db: DatabaseManager) -> None:
        """Both tables should exist after initialization."""
        tables = db.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t["name"] for t in tables]
        assert "memories" in table_names
        assert "visual_references" in table_names

    def test_initialize_is_idempotent(self, db: DatabaseManager) -> None:
        """Calling initialize() a second time should not raise."""
        db.initialize()  # second call
        count = db.count_memories(active_only=False)
        assert count == 0

    def test_foreign_keys_enabled(self, db: DatabaseManager) -> None:
        """Foreign key enforcement should be ON."""
        row = db.connection.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1

    def test_empty_database_counts(self, db: DatabaseManager) -> None:
        """A fresh database should have zero records."""
        assert db.count_memories() == 0
        assert db.count_visual_references() == 0


# ── Memory Insert ─────────────────────────────────────────────

class TestMemoryInsert:
    """Verify inserting memories."""

    def test_insert_returns_positive_id(
        self, db: DatabaseManager, sample_object_memory: Memory
    ) -> None:
        mem_id = db.insert_memory(sample_object_memory)
        assert isinstance(mem_id, int)
        assert mem_id > 0

    def test_insert_sets_timestamps(
        self, db: DatabaseManager, sample_object_memory: Memory
    ) -> None:
        mem_id = db.insert_memory(sample_object_memory)
        memory = db.get_memory(mem_id)
        assert memory is not None
        assert memory.created_at is not None
        assert memory.updated_at is not None
        assert isinstance(memory.created_at, datetime)

    def test_insert_preserves_fields(
        self, db: DatabaseManager, sample_object_memory: Memory
    ) -> None:
        mem_id = db.insert_memory(sample_object_memory)
        memory = db.get_memory(mem_id)
        assert memory is not None
        assert memory.entity_type == EntityType.OBJECT
        assert memory.name == "Grandmother's Watch"
        assert memory.giver_name == "Grandmother"
        assert memory.occasion == "18th Birthday"
        assert memory.year == "2015"
        assert "pocket watch" in memory.narrative
        assert memory.is_active is True

    def test_insert_person_memory(
        self, db: DatabaseManager, sample_person_memory: Memory
    ) -> None:
        mem_id = db.insert_memory(sample_person_memory)
        memory = db.get_memory(mem_id)
        assert memory is not None
        assert memory.entity_type == EntityType.PERSON
        assert memory.name == "Dr. Smith"

    def test_insert_with_nullable_fields(self, db: DatabaseManager) -> None:
        """Nullable fields should accept None."""
        memory = Memory(
            entity_type=EntityType.OBJECT,
            name="Unknown Item",
        )
        mem_id = db.insert_memory(memory)
        fetched = db.get_memory(mem_id)
        assert fetched is not None
        assert fetched.giver_name is None
        assert fetched.occasion is None
        assert fetched.year is None
        assert fetched.narrative is None

    def test_insert_increments_count(
        self, db: DatabaseManager, sample_object_memory: Memory
    ) -> None:
        assert db.count_memories() == 0
        db.insert_memory(sample_object_memory)
        assert db.count_memories() == 1
        db.insert_memory(sample_object_memory)
        assert db.count_memories() == 2


# ── Memory Retrieve ──────────────────────────────────────────

class TestMemoryRetrieve:
    """Verify retrieving memories."""

    def test_get_nonexistent_returns_none(self, db: DatabaseManager) -> None:
        assert db.get_memory(999) is None

    def test_get_all_empty(self, db: DatabaseManager) -> None:
        assert db.get_all_memories() == []

    def test_get_all_returns_multiple(
        self, db: DatabaseManager,
        sample_object_memory: Memory,
        sample_person_memory: Memory,
    ) -> None:
        db.insert_memory(sample_object_memory)
        db.insert_memory(sample_person_memory)
        all_mems = db.get_all_memories()
        assert len(all_mems) == 2

    def test_get_by_type_filters_correctly(
        self, db: DatabaseManager,
        sample_object_memory: Memory,
        sample_person_memory: Memory,
    ) -> None:
        db.insert_memory(sample_object_memory)
        db.insert_memory(sample_person_memory)

        objects = db.get_memories_by_type(EntityType.OBJECT)
        assert len(objects) == 1
        assert objects[0].entity_type == EntityType.OBJECT

        people = db.get_memories_by_type(EntityType.PERSON)
        assert len(people) == 1
        assert people[0].entity_type == EntityType.PERSON


# ── Memory Update ─────────────────────────────────────────────

class TestMemoryUpdate:
    """Verify updating memories."""

    def test_update_changes_fields(
        self, db: DatabaseManager, sample_object_memory: Memory
    ) -> None:
        mem_id = db.insert_memory(sample_object_memory)
        memory = db.get_memory(mem_id)
        assert memory is not None

        memory.name = "Grandfather's Watch"
        memory.narrative = "Updated narrative."
        result = db.update_memory(memory)
        assert result is True

        updated = db.get_memory(mem_id)
        assert updated is not None
        assert updated.name == "Grandfather's Watch"
        assert updated.narrative == "Updated narrative."

    def test_update_refreshes_updated_at(
        self, db: DatabaseManager, sample_object_memory: Memory
    ) -> None:
        mem_id = db.insert_memory(sample_object_memory)
        original = db.get_memory(mem_id)
        assert original is not None

        original.name = "Modified"
        db.update_memory(original)

        updated = db.get_memory(mem_id)
        assert updated is not None
        assert updated.updated_at >= original.updated_at  # type: ignore

    def test_update_nonexistent_returns_false(
        self, db: DatabaseManager
    ) -> None:
        memory = Memory(id=999, name="Ghost")
        result = db.update_memory(memory)
        assert result is False

    def test_update_without_id_raises(self, db: DatabaseManager) -> None:
        memory = Memory(name="No ID")
        with pytest.raises(ValueError, match="without an ID"):
            db.update_memory(memory)


# ── Memory Deactivate / Activate ──────────────────────────────

class TestMemoryDeactivation:
    """Verify soft-delete and re-activation."""

    def test_deactivate_sets_inactive(
        self, db: DatabaseManager, sample_object_memory: Memory
    ) -> None:
        mem_id = db.insert_memory(sample_object_memory)
        result = db.deactivate_memory(mem_id)
        assert result is True

        memory = db.get_memory(mem_id)
        assert memory is not None
        assert memory.is_active is False

    def test_deactivated_excluded_from_active_query(
        self, db: DatabaseManager, sample_object_memory: Memory
    ) -> None:
        mem_id = db.insert_memory(sample_object_memory)
        db.deactivate_memory(mem_id)

        active = db.get_all_memories(active_only=True)
        assert len(active) == 0

        all_mems = db.get_all_memories(active_only=False)
        assert len(all_mems) == 1

    def test_reactivate_memory(
        self, db: DatabaseManager, sample_object_memory: Memory
    ) -> None:
        mem_id = db.insert_memory(sample_object_memory)
        db.deactivate_memory(mem_id)
        db.activate_memory(mem_id)

        memory = db.get_memory(mem_id)
        assert memory is not None
        assert memory.is_active is True

    def test_deactivate_nonexistent_returns_false(
        self, db: DatabaseManager
    ) -> None:
        assert db.deactivate_memory(999) is False

    def test_deactivated_excluded_from_type_query(
        self, db: DatabaseManager,
        sample_object_memory: Memory,
        sample_person_memory: Memory,
    ) -> None:
        obj_id = db.insert_memory(sample_object_memory)
        db.insert_memory(sample_person_memory)
        db.deactivate_memory(obj_id)

        objects = db.get_memories_by_type(EntityType.OBJECT, active_only=True)
        assert len(objects) == 0

        objects_all = db.get_memories_by_type(EntityType.OBJECT, active_only=False)
        assert len(objects_all) == 1


# ── Memory Delete ─────────────────────────────────────────────

class TestMemoryDelete:
    """Verify permanent deletion."""

    def test_delete_removes_memory(
        self, db: DatabaseManager, sample_object_memory: Memory
    ) -> None:
        mem_id = db.insert_memory(sample_object_memory)
        result = db.delete_memory(mem_id)
        assert result is True
        assert db.get_memory(mem_id) is None

    def test_delete_nonexistent_returns_false(
        self, db: DatabaseManager
    ) -> None:
        assert db.delete_memory(999) is False

    def test_delete_cascades_to_visual_references(
        self, db: DatabaseManager, sample_object_memory: Memory
    ) -> None:
        mem_id = db.insert_memory(sample_object_memory)
        ref = VisualReference(
            memory_id=mem_id,
            image_path="/photos/watch.jpg",
            embedding_path="/embeddings/watch.npy",
        )
        db.insert_visual_reference(ref)
        assert db.count_visual_references() == 1

        db.delete_memory(mem_id)
        assert db.count_visual_references() == 0


# ── Visual Reference CRUD ────────────────────────────────────

class TestVisualReferences:
    """Verify visual reference operations."""

    def test_insert_reference(
        self, db: DatabaseManager, sample_object_memory: Memory
    ) -> None:
        mem_id = db.insert_memory(sample_object_memory)
        ref = VisualReference(
            memory_id=mem_id,
            image_path="/photos/watch_front.jpg",
            embedding_path="/embeddings/watch_front.npy",
        )
        ref_id = db.insert_visual_reference(ref)
        assert isinstance(ref_id, int)
        assert ref_id > 0

    def test_retrieve_references(
        self, db: DatabaseManager, sample_object_memory: Memory
    ) -> None:
        mem_id = db.insert_memory(sample_object_memory)
        for i in range(3):
            ref = VisualReference(
                memory_id=mem_id,
                image_path=f"/photos/watch_{i}.jpg",
            )
            db.insert_visual_reference(ref)

        refs = db.get_visual_references(mem_id)
        assert len(refs) == 3
        assert refs[0].image_path == "/photos/watch_0.jpg"

    def test_reference_timestamps(
        self, db: DatabaseManager, sample_object_memory: Memory
    ) -> None:
        mem_id = db.insert_memory(sample_object_memory)
        ref = VisualReference(
            memory_id=mem_id,
            image_path="/photos/watch.jpg",
        )
        ref_id = db.insert_visual_reference(ref)
        refs = db.get_visual_references(mem_id)
        assert len(refs) == 1
        assert refs[0].created_at is not None

    def test_delete_reference(
        self, db: DatabaseManager, sample_object_memory: Memory
    ) -> None:
        mem_id = db.insert_memory(sample_object_memory)
        ref = VisualReference(
            memory_id=mem_id,
            image_path="/photos/watch.jpg",
        )
        ref_id = db.insert_visual_reference(ref)
        result = db.delete_visual_reference(ref_id)
        assert result is True
        assert db.get_visual_references(mem_id) == []

    def test_nullable_embedding_path(
        self, db: DatabaseManager, sample_object_memory: Memory
    ) -> None:
        mem_id = db.insert_memory(sample_object_memory)
        ref = VisualReference(
            memory_id=mem_id,
            image_path="/photos/watch.jpg",
            embedding_path=None,
        )
        db.insert_visual_reference(ref)
        refs = db.get_visual_references(mem_id)
        assert refs[0].embedding_path is None

    def test_no_references_for_nonexistent_memory(
        self, db: DatabaseManager
    ) -> None:
        refs = db.get_visual_references(999)
        assert refs == []


# ── Connection Lifecycle ──────────────────────────────────────

class TestConnectionLifecycle:
    """Verify connection handling edge cases."""

    def test_access_before_initialize_raises(self) -> None:
        db = DatabaseManager(":memory:")
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = db.connection

    def test_close_and_reopen(self) -> None:
        db = DatabaseManager(":memory:")
        db.initialize()
        db.insert_memory(Memory(entity_type=EntityType.OBJECT, name="Test"))
        db.close()

        # After close, accessing connection should raise
        with pytest.raises(RuntimeError):
            _ = db.connection

    def test_double_close_is_safe(self) -> None:
        db = DatabaseManager(":memory:")
        db.initialize()
        db.close()
        db.close()  # should not raise
