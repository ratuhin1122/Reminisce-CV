"""
tests.test_memory_retrieval — Unit Tests for Personal Memory Retrieval
======================================================================

Tests cover:
    - Valid entity ID -> full stored memory successfully retrieved
    - Unknown entity ID -> no memory found
    - Inactive entity -> handled appropriately (excluded when active_only=True)
    - Missing memory fields -> clean None values, missing field reporting, zero hallucinations
    - Pipeline integration -> stable recognition automatically populates stable_memory
"""

import pytest

from app.database.models import EntityType, Memory
from app.memory.repository import MemoryRepository
from app.memory.retrieval import (
    MemoryRetrievalService,
    StructuredMemoryResponse,
)
from app.memory.service import MemoryService


@pytest.fixture
def memory_service() -> MemoryService:
    """In-memory database memory service fixture."""
    svc = MemoryService(db_path=":memory:")
    svc.start()
    yield svc
    svc.stop()


@pytest.fixture
def retrieval_service(memory_service: MemoryService) -> MemoryRetrievalService:
    """Retrieval service backed by in-memory SQLite database."""
    return MemoryRetrievalService(memory_service=memory_service)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestPersonalMemoryRetrieval:
    """Tests for MemoryRetrievalService."""

    def test_valid_entity_memory_found(
        self,
        memory_service: MemoryService,
        retrieval_service: MemoryRetrievalService,
    ) -> None:
        """Valid entity ID successfully retrieves user-stored personal memory."""
        mem_id = memory_service.register_object(
            name="Vintage Swiss Watch",
            giver_name="Grandfather Arthur",
            occasion="College Graduation",
            year="1978",
            narrative="Given to me after my engineering degree in Zurich. Still ticks reliably.",
        )

        response = retrieval_service.retrieve_memory(mem_id)

        assert isinstance(response, StructuredMemoryResponse)
        assert response.found is True
        assert response.entity_id == mem_id
        assert response.entity_type == "object"
        assert response.title == "Vintage Swiss Watch"
        assert response.name == "Vintage Swiss Watch"
        assert response.giver == "Grandfather Arthur"
        assert response.person == "Grandfather Arthur"
        assert response.occasion == "College Graduation"
        assert response.year == "1978"
        assert response.date == "1978"
        assert (
            response.narrative
            == "Given to me after my engineering degree in Zurich. Still ticks reliably."
        )
        assert response.is_active is True
        assert response.raw_memory is not None
        assert response.raw_memory.id == mem_id
        assert response.missing_fields == []

    def test_unknown_entity_no_memory(
        self,
        retrieval_service: MemoryRetrievalService,
    ) -> None:
        """Non-existent entity ID returns response with found=False."""
        # Query ID that doesn't exist
        response = retrieval_service.retrieve_memory(99999)
        assert response.found is False
        assert response.entity_id == 99999
        assert response.title == ""
        assert response.giver is None
        assert response.raw_memory is None

        # Query None / invalid ID
        res_none = retrieval_service.retrieve_memory(None)
        assert res_none.found is False
        assert res_none.entity_id is None

        res_neg = retrieval_service.retrieve_memory(-5)
        assert res_neg.found is False

    def test_inactive_entity_handling(
        self,
        memory_service: MemoryService,
        retrieval_service: MemoryRetrievalService,
    ) -> None:
        """Deactivated memories are excluded by default, but retrievable when requested."""
        mem_id = memory_service.register_object(
            name="Old Bicycle",
            giver_name="Uncle Bob",
            occasion="10th Birthday",
            year="1995",
            narrative="Red two-wheeler learned to ride on.",
        )

        # Deactivate the memory
        memory_service.deactivate_memory(mem_id)

        # 1. Default (active_only=True): marked as not found
        resp_active_only = retrieval_service.retrieve_memory(mem_id, active_only=True)
        assert resp_active_only.found is False
        assert resp_active_only.is_active is False

        # 2. Query with active_only=False: retrieves the inactive record
        resp_allow_inactive = retrieval_service.retrieve_memory(
            mem_id, active_only=False
        )
        assert resp_allow_inactive.found is True
        assert resp_allow_inactive.title == "Old Bicycle"
        assert resp_allow_inactive.is_active is False

    def test_missing_memory_fields_preserves_accuracy(
        self,
        memory_service: MemoryService,
        retrieval_service: MemoryRetrievalService,
    ) -> None:
        """Stored entity with only name returns None for missing fields without inventing data."""
        mem_id = memory_service.register_object(
            name="Silver Pocket Knife",
            giver_name=None,
            occasion=None,
            year=None,
            narrative=None,
        )

        response = retrieval_service.retrieve_memory(mem_id)

        assert response.found is True
        assert response.title == "Silver Pocket Knife"
        assert response.giver is None
        assert response.occasion is None
        assert response.year is None
        assert response.narrative is None
        assert response.has_giver is False
        assert response.has_occasion is False
        assert response.has_year is False
        assert response.has_narrative is False

        # Verify missing fields list
        assert set(response.missing_fields) == {
            "giver",
            "occasion",
            "year",
            "narrative",
        }

        # Formatted summary only reflects actual stored values
        summary = response.formatted_summary()
        assert summary == "Memory: Silver Pocket Knife"
        assert "Giver" not in summary
        assert "Occasion" not in summary

    def test_person_entity_memory_retrieval(
        self,
        memory_service: MemoryService,
        retrieval_service: MemoryRetrievalService,
    ) -> None:
        """Retrieval works equally for familiar person memories."""
        mem_id = memory_service.register_person(
            name="Alice Walker",
            giver_name="Daughter",
            occasion="Family",
            year="2005",
            narrative="Youngest daughter, works in renewable energy.",
        )

        response = retrieval_service.retrieve_memory(mem_id)

        assert response.found is True
        assert response.entity_type == "person"
        assert response.title == "Alice Walker"
        assert response.name == "Alice Walker"
        assert response.person == "Daughter"
        assert response.giver == "Daughter"
        assert response.narrative == "Youngest daughter, works in renewable energy."

    def test_to_dict_serialization(
        self,
        memory_service: MemoryService,
        retrieval_service: MemoryRetrievalService,
    ) -> None:
        """Response serializes into a clean dictionary structure."""
        mem_id = memory_service.register_object(
            name="Fountain Pen",
            giver_name="Mother",
            occasion="Graduation",
            year="1999",
            narrative="Black Parker pen.",
        )

        response = retrieval_service.retrieve_memory(mem_id)
        d = response.to_dict()

        assert d["found"] is True
        assert d["entity_id"] == mem_id
        assert d["title"] == "Fountain Pen"
        assert d["name"] == "Fountain Pen"
        assert d["giver"] == "Mother"
        assert d["person"] == "Mother"
        assert d["occasion"] == "Graduation"
        assert d["year"] == "1999"
        assert d["date"] == "1999"
        assert d["narrative"] == "Black Parker pen."
        assert d["is_active"] is True
        assert d["missing_fields"] == []
