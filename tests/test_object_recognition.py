"""
tests.test_object_recognition — Unit Tests for Object Recognition Pipeline
===========================================================================

Tests cover:
    - Empty gallery handling
    - Controlled/mock embeddings matching and cosine similarity thresholding
    - Structured RecognitionResult verification
    - Multiple reference images per object (best-view matching)
    - Threshold sensitivity and per-query overrides
    - Top-K candidate ranking via recognize_top_k
    - End-to-end flow with MemoryService and ObjectRegistrationService
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from app.memory import MemoryService, ObjectRegistrationService
from app.recognition import ObjectRecognitionService, RecognitionResult
from app.vision import MockVisionEmbeddingModel


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_vision() -> MockVisionEmbeddingModel:
    """Mock vision model with 512-dim normalized vectors."""
    return MockVisionEmbeddingModel(embedding_dim=512, auto_load=True)


@pytest.fixture
def mem_service() -> MemoryService:
    """In-memory MemoryService."""
    svc = MemoryService(db_path=":memory:")
    svc.start()
    yield svc
    svc.stop()


@pytest.fixture
def reg_service(
    mem_service: MemoryService,
    mock_vision: MockVisionEmbeddingModel,
    tmp_path: Path,
) -> ObjectRegistrationService:
    """ObjectRegistrationService backed by temporary test folders."""
    ref_dir = tmp_path / "references"
    emb_dir = tmp_path / "embeddings"
    return ObjectRegistrationService(
        memory_service=mem_service,
        vision_model=mock_vision,
        references_dir=ref_dir,
        embeddings_dir=emb_dir,
    )


@pytest.fixture
def recognition_service(
    mock_vision: MockVisionEmbeddingModel,
    reg_service: ObjectRegistrationService,
) -> ObjectRecognitionService:
    """ObjectRecognitionService wired to the mock vision and registration services."""
    return ObjectRecognitionService(
        vision_model=mock_vision,
        registration_service=reg_service,
        threshold=0.50,
    )


def make_unit_vector(dim: int, active_index: int) -> np.ndarray:
    """Generate a 1D unit basis vector with 1.0 at active_index."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[active_index] = 1.0
    return vec


# ── 1. Empty Gallery Tests ───────────────────────────────────────────────────


class TestEmptyGallery:
    """Verify behavior when no objects have been registered."""

    def test_recognize_empty_gallery_returns_no_match(
        self, recognition_service: ObjectRecognitionService
    ) -> None:
        img = Image.new("RGB", (64, 64), color="blue")
        result = recognition_service.recognize(img)

        assert isinstance(result, RecognitionResult)
        assert result.matched is False
        assert result.similarity == 0.0
        assert result.entity_id is None
        assert result.entity_type == "object"
        assert result.name is None
        assert result.candidate_id is None

    def test_top_k_empty_gallery_returns_empty_list(
        self, recognition_service: ObjectRecognitionService
    ) -> None:
        img = Image.new("RGB", (64, 64))
        results = recognition_service.recognize_top_k(img, k=3)
        assert results == []


# ── 2. Controlled / Mock Embeddings Tests ─────────────────────────────────────


class TestControlledEmbeddingsRecognition:
    """Test matching accuracy and scoring using deterministic unit vectors."""

    def test_exact_match_against_in_memory_gallery(
        self, mock_vision: MockVisionEmbeddingModel, mem_service: MemoryService
    ) -> None:
        dim = mock_vision.embedding_dim
        v1 = make_unit_vector(dim, 0)
        v2 = make_unit_vector(dim, 1)

        # Register two items in the database
        id1 = mem_service.register_object(name="Object One")
        id2 = mem_service.register_object(name="Object Two")

        rec_svc = ObjectRecognitionService(
            vision_model=mock_vision,
            memory_service=mem_service,
            threshold=0.60,
        )
        rec_svc.set_in_memory_gallery([v1, v2], [id1, id2], ["ref1.jpg", "ref2.jpg"])

        # Query matches Object 1
        mock_vision.set_next_embedding(v1)
        res1 = rec_svc.recognize("query_img.jpg")

        assert res1.matched is True
        assert res1.entity_id == id1
        assert res1.name == "Object One"
        assert np.isclose(res1.similarity, 1.0, atol=1e-5)
        assert res1.reference_image_path == "ref1.jpg"

        # Query matches Object 2
        mock_vision.set_next_embedding(v2)
        res2 = rec_svc.recognize("query_img.jpg")

        assert res2.matched is True
        assert res2.entity_id == id2
        assert res2.name == "Object Two"
        assert np.isclose(res2.similarity, 1.0, atol=1e-5)

    def test_threshold_rejection(
        self, mock_vision: MockVisionEmbeddingModel, mem_service: MemoryService
    ) -> None:
        dim = mock_vision.embedding_dim
        v1 = make_unit_vector(dim, 0)
        id1 = mem_service.register_object(name="Target Object")

        rec_svc = ObjectRecognitionService(
            vision_model=mock_vision,
            memory_service=mem_service,
            threshold=0.80,
        )
        rec_svc.set_in_memory_gallery([v1], [id1])

        # Query vector with similarity 0.707 (45 degrees off)
        query_vec = np.zeros(dim, dtype=np.float32)
        query_vec[0] = 0.7071
        query_vec[1] = 0.7071
        query_vec = query_vec / np.linalg.norm(query_vec)

        mock_vision.set_next_embedding(query_vec)
        result = rec_svc.recognize("query.jpg")

        # Below 0.80 threshold
        assert result.matched is False
        assert result.entity_id is None
        assert result.name is None
        assert result.candidate_id == id1
        assert np.isclose(result.similarity, 0.7071, atol=1e-3)
        assert result.threshold == 0.80

    def test_per_query_threshold_override(
        self, mock_vision: MockVisionEmbeddingModel, mem_service: MemoryService
    ) -> None:
        dim = mock_vision.embedding_dim
        v1 = make_unit_vector(dim, 0)
        id1 = mem_service.register_object(name="Target Object")

        rec_svc = ObjectRecognitionService(
            vision_model=mock_vision,
            memory_service=mem_service,
            threshold=0.80,  # Default high threshold
        )
        rec_svc.set_in_memory_gallery([v1], [id1])

        query_vec = np.zeros(dim, dtype=np.float32)
        query_vec[0] = 0.7071
        query_vec[1] = 0.7071

        mock_vision.set_next_embedding(query_vec)
        # Override threshold to 0.70 -> should now match
        result = rec_svc.recognize("query.jpg", threshold=0.70)
        assert result.matched is True
        assert result.entity_id == id1
        assert result.threshold == 0.70


# ── 3. Multi-Reference Selection Tests ────────────────────────────────────────


class TestMultiReferenceSelection:
    """Verify that when an object has multiple angles/references, the best one is selected."""

    def test_selects_best_reference_image(
        self, mock_vision: MockVisionEmbeddingModel, mem_service: MemoryService
    ) -> None:
        dim = mock_vision.embedding_dim
        # Object A has 2 reference angles
        angle_front = make_unit_vector(dim, 0)
        angle_side = make_unit_vector(dim, 1)

        id_a = mem_service.register_object(name="Multi Angle Object")

        rec_svc = ObjectRecognitionService(
            vision_model=mock_vision,
            memory_service=mem_service,
            threshold=0.50,
        )
        rec_svc.set_in_memory_gallery(
            [angle_front, angle_side],
            [id_a, id_a],
            ["front_view.jpg", "side_view.jpg"],
        )

        # Query resembles the side view
        query_side = make_unit_vector(dim, 1)
        mock_vision.set_next_embedding(query_side)

        result = rec_svc.recognize("query.jpg")
        assert result.matched is True
        assert result.entity_id == id_a
        assert result.reference_image_path == "side_view.jpg"
        assert np.isclose(result.similarity, 1.0)


# ── 4. Top-K Ranking Tests ────────────────────────────────────────────────────


class TestTopKRanking:
    """Verify recognize_top_k candidate ordering."""

    def test_top_k_ordering(
        self, mock_vision: MockVisionEmbeddingModel, mem_service: MemoryService
    ) -> None:
        dim = mock_vision.embedding_dim
        v1 = make_unit_vector(dim, 0)
        v2 = make_unit_vector(dim, 1)
        v3 = make_unit_vector(dim, 2)

        id1 = mem_service.register_object(name="First")
        id2 = mem_service.register_object(name="Second")
        id3 = mem_service.register_object(name="Third")

        rec_svc = ObjectRecognitionService(
            vision_model=mock_vision,
            memory_service=mem_service,
            threshold=0.10,
        )
        rec_svc.set_in_memory_gallery([v1, v2, v3], [id1, id2, id3])

        # Query: [0.8, 0.5, 0.2, 0, ...]
        query = np.zeros(dim, dtype=np.float32)
        query[0] = 0.8
        query[1] = 0.5
        query[2] = 0.2
        query = query / np.linalg.norm(query)
        mock_vision.set_next_embedding(query)

        results = rec_svc.recognize_top_k("query.jpg", k=2)
        assert len(results) == 2
        # Highest score first
        assert results[0].entity_id == id1
        assert results[1].entity_id == id2
        assert results[0].similarity > results[1].similarity


# ── 5. End-to-End Registration & Recognition Integration ───────────────────────


class TestEndToEndWorkflow:
    """Test full integration between ObjectRegistrationService and ObjectRecognitionService."""

    def test_register_and_recognize_full_flow(
        self,
        recognition_service: ObjectRecognitionService,
        reg_service: ObjectRegistrationService,
    ) -> None:
        # 1. Register two objects with images
        img_a = Image.new("RGB", (64, 64), color="blue")
        img_b = Image.new("RGB", (64, 64), color="red")

        obj_a = reg_service.register_object(
            name="Synthetic Blue Keepsake",
            giver="Relative A",
            occasion="Graduation 2021",
            year="2021",
            narrative="A blue keepsake item.",
            reference_images=[img_a],
        )

        obj_b = reg_service.register_object(
            name="Synthetic Red Keepsake",
            giver="Relative B",
            occasion="Birthday 2022",
            year="2022",
            narrative="A red keepsake item.",
            reference_images=[img_b],
        )

        # 2. Refresh gallery in recognition service
        recognition_service.refresh_gallery()
        assert recognition_service.gallery_size == 2

        # 3. Query with identical image A -> should match obj_a
        res_a = recognition_service.recognize(img_a)
        assert res_a.matched is True
        assert res_a.entity_id == obj_a.memory_id
        assert res_a.name == "Synthetic Blue Keepsake"
        assert res_a.memory is not None
        assert res_a.memory.giver_name == "Relative A"
        assert np.isclose(res_a.similarity, 1.0, atol=1e-4)

        # 4. Query with identical image B -> should match obj_b
        res_b = recognition_service.recognize(img_b)
        assert res_b.matched is True
        assert res_b.entity_id == obj_b.memory_id
        assert res_b.name == "Synthetic Red Keepsake"
        assert np.isclose(res_b.similarity, 1.0, atol=1e-4)

    def test_threshold_setter_validation(
        self, recognition_service: ObjectRecognitionService
    ) -> None:
        recognition_service.threshold = 0.75
        assert recognition_service.threshold == 0.75

        with pytest.raises(ValueError, match="between -1.0 and 1.0"):
            recognition_service.threshold = 1.5
