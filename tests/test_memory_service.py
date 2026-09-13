"""
tests.test_memory_service — Tests for Repository + Service Layers
==================================================================

Tests cover the full stack through the ``MemoryService``, which
delegates to ``MemoryRepository``, which delegates to
``DatabaseManager``.  All tests use in-memory SQLite.

Sections:
    - Object memory creation
    - Person memory creation
    - Memory retrieval (single, all, by type)
    - Memory update
    - Deactivation / activation
    - Deletion
    - Reference images
    - Embedding metadata
    - Validation errors
    - MemoryWithReferences bundling
    - Statistics
"""

from __future__ import annotations

import pytest

from app.database.models import EntityType, Memory
from app.memory.service import MemoryService, MemoryWithReferences


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def svc() -> MemoryService:
    """Provide a fresh MemoryService backed by in-memory SQLite."""
    service = MemoryService(db_path=":memory:")
    service.start()
    yield service
    service.stop()


# ── Object Memory Creation ───────────────────────────────────

class TestCreateObjectMemory:

    def test_create_returns_id(self, svc: MemoryService) -> None:
        mem_id = svc.register_object(name="Watch")
        assert isinstance(mem_id, int)
        assert mem_id > 0

    def test_create_with_all_fields(self, svc: MemoryService) -> None:
        mem_id = svc.register_object(
            name="Grandmother's Watch",
            giver_name="Grandmother",
            occasion="18th Birthday",
            year="2015",
            narrative="A silver pocket watch.",
        )
        memory = svc.get_memory(mem_id)
        assert memory is not None
        assert memory.entity_type == EntityType.OBJECT
        assert memory.name == "Grandmother's Watch"
        assert memory.giver_name == "Grandmother"
        assert memory.occasion == "18th Birthday"
        assert memory.year == "2015"
        assert "pocket watch" in memory.narrative

    def test_create_with_image(self, svc: MemoryService) -> None:
        mem_id = svc.register_object(
            name="Watch",
            image_path="/photos/watch.jpg",
        )
        refs = svc.get_reference_images(mem_id)
        assert len(refs) == 1
        assert refs[0].image_path == "/photos/watch.jpg"

    def test_create_strips_whitespace(self, svc: MemoryService) -> None:
        mem_id = svc.register_object(name="  Watch  ")
        memory = svc.get_memory(mem_id)
        assert memory.name == "Watch"


# ── Person Memory Creation ───────────────────────────────────

class TestCreatePersonMemory:

    def test_create_returns_id(self, svc: MemoryService) -> None:
        mem_id = svc.register_person(name="Dr. Smith")
        assert isinstance(mem_id, int)
        assert mem_id > 0

    def test_create_with_all_fields(self, svc: MemoryService) -> None:
        mem_id = svc.register_person(
            name="Dr. Smith",
            giver_name="Dr. John Smith",
            occasion="Research Advisor",
            year="2020",
            narrative="My PhD advisor.",
        )
        memory = svc.get_memory(mem_id)
        assert memory is not None
        assert memory.entity_type == EntityType.PERSON
        assert memory.name == "Dr. Smith"

    def test_create_with_image(self, svc: MemoryService) -> None:
        mem_id = svc.register_person(
            name="Dr. Smith",
            image_path="/photos/smith.jpg",
        )
        refs = svc.get_reference_images(mem_id)
        assert len(refs) == 1


# ── Retrieval ─────────────────────────────────────────────────

class TestRetrieval:

    def test_get_nonexistent(self, svc: MemoryService) -> None:
        assert svc.get_memory(999) is None

    def test_get_all_active(self, svc: MemoryService) -> None:
        svc.register_object(name="Watch")
        svc.register_person(name="Alice")
        active = svc.get_all_active()
        assert len(active) == 2

    def test_get_all_people(self, svc: MemoryService) -> None:
        svc.register_object(name="Watch")
        svc.register_person(name="Alice")
        svc.register_person(name="Bob")
        people = svc.get_all_people()
        assert len(people) == 2
        assert all(p.entity_type == EntityType.PERSON for p in people)

    def test_get_all_objects(self, svc: MemoryService) -> None:
        svc.register_object(name="Watch")
        svc.register_object(name="Ring")
        svc.register_person(name="Alice")
        objects = svc.get_all_objects()
        assert len(objects) == 2
        assert all(o.entity_type == EntityType.OBJECT for o in objects)


# ── Full Memory Retrieval ────────────────────────────────────

class TestFullMemory:

    def test_get_full_memory(self, svc: MemoryService) -> None:
        mem_id = svc.register_object(
            name="Watch", image_path="/photos/watch.jpg"
        )
        svc.store_embedding(
            memory_id=mem_id,
            model_name="clip-vit-base-patch32",
            embedding_dim=512,
            embedding_path="/emb/watch.npy",
            source_type="clip_object",
        )

        full = svc.get_full_memory(mem_id)
        assert full is not None
        assert isinstance(full, MemoryWithReferences)
        assert full.memory.name == "Watch"
        assert len(full.references) == 1
        assert len(full.embeddings) == 1

    def test_get_full_memory_nonexistent(self, svc: MemoryService) -> None:
        assert svc.get_full_memory(999) is None

    def test_full_memory_empty_refs(self, svc: MemoryService) -> None:
        mem_id = svc.register_object(name="Watch")
        full = svc.get_full_memory(mem_id)
        assert full is not None
        assert full.references == []
        assert full.embeddings == []


# ── Update ────────────────────────────────────────────────────

class TestUpdate:

    def test_update_name(self, svc: MemoryService) -> None:
        mem_id = svc.register_object(name="Watch")
        memory = svc.get_memory(mem_id)
        memory.name = "Grandfather's Watch"
        result = svc.update_memory(memory)
        assert result is True

        updated = svc.get_memory(mem_id)
        assert updated.name == "Grandfather's Watch"

    def test_update_narrative(self, svc: MemoryService) -> None:
        mem_id = svc.register_object(name="Watch")
        memory = svc.get_memory(mem_id)
        memory.narrative = "Updated story."
        svc.update_memory(memory)

        updated = svc.get_memory(mem_id)
        assert updated.narrative == "Updated story."

    def test_update_strips_name(self, svc: MemoryService) -> None:
        mem_id = svc.register_object(name="Watch")
        memory = svc.get_memory(mem_id)
        memory.name = "  New Name  "
        svc.update_memory(memory)

        updated = svc.get_memory(mem_id)
        assert updated.name == "New Name"


# ── Deactivation / Activation ────────────────────────────────

class TestDeactivation:

    def test_deactivate(self, svc: MemoryService) -> None:
        mem_id = svc.register_object(name="Watch")
        result = svc.deactivate_memory(mem_id)
        assert result is True

        memory = svc.get_memory(mem_id)
        assert memory.is_active is False

    def test_deactivated_excluded_from_active(self, svc: MemoryService) -> None:
        mem_id = svc.register_object(name="Watch")
        svc.deactivate_memory(mem_id)
        assert len(svc.get_all_active()) == 0

    def test_reactivate(self, svc: MemoryService) -> None:
        mem_id = svc.register_object(name="Watch")
        svc.deactivate_memory(mem_id)
        svc.activate_memory(mem_id)

        memory = svc.get_memory(mem_id)
        assert memory.is_active is True
        assert len(svc.get_all_active()) == 1


# ── Deletion ─────────────────────────────────────────────────

class TestDeletion:

    def test_delete_memory(self, svc: MemoryService) -> None:
        mem_id = svc.register_object(name="Watch")
        result = svc.delete_memory(mem_id)
        assert result is True
        assert svc.get_memory(mem_id) is None

    def test_delete_cascades_refs_and_embeddings(
        self, svc: MemoryService
    ) -> None:
        mem_id = svc.register_object(
            name="Watch", image_path="/photos/watch.jpg"
        )
        svc.store_embedding(
            memory_id=mem_id,
            model_name="clip",
            embedding_dim=512,
            embedding_path="/emb/watch.npy",
        )
        assert svc.stats()["reference_images"] == 1
        assert svc.stats()["embeddings"] == 1

        svc.delete_memory(mem_id)
        assert svc.stats()["reference_images"] == 0
        assert svc.stats()["embeddings"] == 0


# ── Reference Images ─────────────────────────────────────────

class TestReferenceImages:

    def test_add_multiple_images(self, svc: MemoryService) -> None:
        mem_id = svc.register_object(name="Watch")
        svc.add_reference_image(mem_id, "/photos/front.jpg")
        svc.add_reference_image(mem_id, "/photos/back.jpg")
        svc.add_reference_image(mem_id, "/photos/side.jpg")

        refs = svc.get_reference_images(mem_id)
        assert len(refs) == 3

    def test_add_to_nonexistent_raises(self, svc: MemoryService) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            svc.add_reference_image(999, "/photos/ghost.jpg")


# ── Embedding Metadata ───────────────────────────────────────

class TestEmbeddingMetadata:

    def test_store_and_retrieve(self, svc: MemoryService) -> None:
        mem_id = svc.register_person(name="Alice")
        emb_id = svc.store_embedding(
            memory_id=mem_id,
            model_name="buffalo_l",
            embedding_dim=512,
            embedding_path="/emb/alice_face.npy",
            source_type="face",
        )
        assert isinstance(emb_id, int)

        embs = svc.get_embeddings(mem_id)
        assert len(embs) == 1
        assert embs[0].model_name == "buffalo_l"
        assert embs[0].embedding_dim == 512
        assert embs[0].source_type == "face"

    def test_filter_by_source_type(self, svc: MemoryService) -> None:
        mem_id = svc.register_person(name="Alice")
        svc.store_embedding(
            memory_id=mem_id,
            model_name="buffalo_l",
            embedding_dim=512,
            embedding_path="/emb/face.npy",
            source_type="face",
        )
        svc.store_embedding(
            memory_id=mem_id,
            model_name="clip-vit-base",
            embedding_dim=512,
            embedding_path="/emb/clip.npy",
            source_type="clip_object",
        )

        face_embs = svc.get_embeddings(mem_id, source_type="face")
        assert len(face_embs) == 1
        assert face_embs[0].source_type == "face"

        clip_embs = svc.get_embeddings(mem_id, source_type="clip_object")
        assert len(clip_embs) == 1

    def test_get_all_embeddings(self, svc: MemoryService) -> None:
        m1 = svc.register_person(name="Alice")
        m2 = svc.register_object(name="Watch")
        svc.store_embedding(m1, "buffalo_l", 512, "/e1.npy", "face")
        svc.store_embedding(m2, "clip", 512, "/e2.npy", "clip_object")

        all_embs = svc.get_all_embeddings()
        assert len(all_embs) == 2

        face_only = svc.get_all_embeddings(source_type="face")
        assert len(face_only) == 1

    def test_store_for_nonexistent_raises(self, svc: MemoryService) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            svc.store_embedding(999, "clip", 512, "/e.npy")

    def test_embedding_has_timestamp(self, svc: MemoryService) -> None:
        mem_id = svc.register_object(name="Watch")
        svc.store_embedding(mem_id, "clip", 512, "/e.npy")
        embs = svc.get_embeddings(mem_id)
        assert embs[0].created_at is not None


# ── Validation ────────────────────────────────────────────────

class TestValidation:

    def test_empty_name_object_raises(self, svc: MemoryService) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            svc.register_object(name="")

    def test_whitespace_name_object_raises(self, svc: MemoryService) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            svc.register_object(name="   ")

    def test_empty_name_person_raises(self, svc: MemoryService) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            svc.register_person(name="")

    def test_update_empty_name_raises(self, svc: MemoryService) -> None:
        mem_id = svc.register_object(name="Watch")
        memory = svc.get_memory(mem_id)
        memory.name = ""
        with pytest.raises(ValueError, match="cannot be empty"):
            svc.update_memory(memory)

    def test_update_without_id_raises(self, svc: MemoryService) -> None:
        memory = Memory(name="Ghost")
        with pytest.raises(ValueError, match="without an ID"):
            svc.update_memory(memory)


# ── Statistics ────────────────────────────────────────────────

class TestStatistics:

    def test_empty_stats(self, svc: MemoryService) -> None:
        stats = svc.stats()
        assert stats == {
            "active_memories": 0,
            "total_memories": 0,
            "reference_images": 0,
            "embeddings": 0,
        }

    def test_stats_after_registration(self, svc: MemoryService) -> None:
        m1 = svc.register_object(name="Watch", image_path="/p1.jpg")
        m2 = svc.register_person(name="Alice", image_path="/p2.jpg")
        svc.store_embedding(m1, "clip", 512, "/e1.npy")
        svc.store_embedding(m2, "buffalo_l", 512, "/e2.npy", "face")

        stats = svc.stats()
        assert stats["active_memories"] == 2
        assert stats["total_memories"] == 2
        assert stats["reference_images"] == 2
        assert stats["embeddings"] == 2

    def test_stats_with_deactivated(self, svc: MemoryService) -> None:
        mem_id = svc.register_object(name="Watch")
        svc.deactivate_memory(mem_id)
        svc.register_person(name="Alice")

        stats = svc.stats()
        assert stats["active_memories"] == 1
        assert stats["total_memories"] == 2
