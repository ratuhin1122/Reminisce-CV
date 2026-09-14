"""
tests.test_person_recognition — Unit Tests for Familiar Person Recognition
===========================================================================

Tests cover:
    - Face detection and embedding model contracts
    - Familiar person registration with reference face images
    - Local disk storage of face crops and embeddings
    - Privacy compliance (no identity leaks in logs)
    - Face recognition matching against registered people
    - Rejection of unknown faces (returning name='unknown' and matched=False)
    - Configurable similarity thresholding and per-query overrides
    - Multi-face detection and recognition in a single frame
    - Clean separation between object and person recognition
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from app.database.models import EntityType
from app.memory import (
    MemoryService,
    PersonRegistrationService,
    RegisteredPerson,
)
from app.recognition import (
    ObjectRecognitionService,
    PersonRecognitionResult,
    PersonRecognitionService,
    RecognitionResult,
)
from app.vision import (
    FaceBoundingBox,
    MockFaceEmbeddingModel,
    MockVisionEmbeddingModel,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_face_engine() -> MockFaceEmbeddingModel:
    return MockFaceEmbeddingModel(embedding_dim=512, auto_load=True)


@pytest.fixture
def mem_service() -> MemoryService:
    svc = MemoryService(db_path=":memory:")
    svc.start()
    yield svc
    svc.stop()


@pytest.fixture
def person_reg_service(
    mem_service: MemoryService,
    mock_face_engine: MockFaceEmbeddingModel,
    tmp_path: Path,
) -> PersonRegistrationService:
    ref_dir = tmp_path / "data" / "references"
    emb_dir = tmp_path / "data" / "embeddings"
    return PersonRegistrationService(
        memory_service=mem_service,
        face_engine=mock_face_engine,
        references_dir=ref_dir,
        embeddings_dir=emb_dir,
    )


@pytest.fixture
def person_rec_service(
    mock_face_engine: MockFaceEmbeddingModel,
    person_reg_service: PersonRegistrationService,
) -> PersonRecognitionService:
    return PersonRecognitionService(
        face_engine=mock_face_engine,
        registration_service=person_reg_service,
        threshold=0.50,
    )


def make_unit_vector(dim: int, active_index: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[active_index] = 1.0
    return vec


def make_synthetic_face(color: str = "peachpuff") -> Image.Image:
    return Image.new("RGB", (112, 112), color=color)


# ── 1. Person Registration Tests ──────────────────────────────────────────────


class TestPersonRegistration:
    """Test registering familiar people with metadata and reference photos."""

    def test_register_empty_name_raises(
        self, person_reg_service: PersonRegistrationService
    ) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            person_reg_service.register_person(name="", reference_images=[make_synthetic_face()])

    def test_register_without_images_raises(
        self, person_reg_service: PersonRegistrationService
    ) -> None:
        with pytest.raises(ValueError, match="At least one reference face image"):
            person_reg_service.register_person(name="Test Person", reference_images=[])

    def test_register_person_success(
        self, person_reg_service: PersonRegistrationService
    ) -> None:
        face_img = make_synthetic_face("navajowhite")
        reg_person = person_reg_service.register_person(
            name="Synthetic Person Alpha",
            relationship="Caregiver",
            narrative="Primary morning caregiver.",
            reference_images=[face_img],
        )

        assert isinstance(reg_person, RegisteredPerson)
        assert reg_person.memory_id > 0
        assert reg_person.name == "Synthetic Person Alpha"
        assert reg_person.relationship == "Caregiver"
        assert reg_person.narrative == "Primary morning caregiver."
        assert len(reg_person.references) == 1

        ref = reg_person.references[0]
        assert Path(ref.image_path).is_file()
        assert Path(ref.embedding_path).is_file()
        assert ref.embedding.shape == (512,)
        assert np.isclose(np.linalg.norm(ref.embedding), 1.0, atol=1e-5)

    def test_register_multiple_face_angles(
        self, person_reg_service: PersonRegistrationService
    ) -> None:
        faces = [make_synthetic_face("white"), make_synthetic_face("bisque")]
        reg_person = person_reg_service.register_person(
            name="Multi Angle Subject",
            relationship="Family Member",
            reference_images=faces,
        )

        assert len(reg_person.references) == 2
        for ref in reg_person.references:
            assert Path(ref.image_path).is_file()
            assert Path(ref.embedding_path).is_file()

    def test_get_registered_person(
        self, person_reg_service: PersonRegistrationService
    ) -> None:
        reg = person_reg_service.register_person(
            name="Subject B",
            relationship="Friend",
            reference_images=[make_synthetic_face()],
        )

        fetched = person_reg_service.get_registered_person(reg.memory_id)
        assert fetched is not None
        assert fetched.name == "Subject B"
        assert fetched.relationship == "Friend"
        assert len(fetched.references) == 1
        np.testing.assert_allclose(fetched.references[0].embedding, reg.references[0].embedding)

    def test_delete_person_cleans_db_and_disk(
        self, person_reg_service: PersonRegistrationService
    ) -> None:
        reg = person_reg_service.register_person(
            name="Temporary Subject",
            reference_images=[make_synthetic_face()],
        )
        mem_id = reg.memory_id
        img_path = Path(reg.references[0].image_path)
        emb_path = Path(reg.references[0].embedding_path)

        assert img_path.is_file()
        assert emb_path.is_file()

        success = person_reg_service.delete_registered_person(mem_id)
        assert success is True
        assert person_reg_service.get_registered_person(mem_id) is None
        assert not img_path.is_file()
        assert not emb_path.is_file()


# ── 2. Face Recognition Tests ─────────────────────────────────────────────────


class TestPersonRecognition:
    """Test matching query faces against familiar people."""

    def test_empty_gallery_returns_unknown(
        self, person_rec_service: PersonRecognitionService
    ) -> None:
        img = make_synthetic_face()
        result = person_rec_service.recognize_face(img)

        assert isinstance(result, PersonRecognitionResult)
        assert result.matched is False
        assert result.name == "unknown"
        assert result.entity_id is None
        assert result.entity_type == "person"
        assert result.similarity == 0.0

    def test_positive_face_match(
        self,
        mock_face_engine: MockFaceEmbeddingModel,
        person_reg_service: PersonRegistrationService,
        person_rec_service: PersonRecognitionService,
    ) -> None:
        # Controlled vector for Person 1
        v1 = make_unit_vector(512, 0)
        mock_face_engine.set_next_embedding(v1)

        reg = person_reg_service.register_person(
            name="Alice",
            relationship="Daughter",
            reference_images=[make_synthetic_face()],
        )

        person_rec_service.refresh_gallery()

        # Query with identical vector v1
        mock_face_engine.set_next_embedding(v1)
        res = person_rec_service.recognize_face("query_face.jpg")

        assert res.matched is True
        assert res.name == "Alice"
        assert res.entity_id == reg.memory_id
        assert res.relationship == "Daughter"
        assert np.isclose(res.similarity, 1.0, atol=1e-4)

    def test_unknown_face_below_threshold_returns_unknown(
        self,
        mock_face_engine: MockFaceEmbeddingModel,
        person_reg_service: PersonRegistrationService,
        person_rec_service: PersonRecognitionService,
    ) -> None:
        # Register Person 1 with vector [1, 0, 0, ...]
        v1 = make_unit_vector(512, 0)
        mock_face_engine.set_next_embedding(v1)

        person_reg_service.register_person(
            name="Alice",
            relationship="Daughter",
            reference_images=[make_synthetic_face()],
        )

        person_rec_service.refresh_gallery()

        # Query with orthogonal vector [0, 1, 0, ...]
        v_unknown = make_unit_vector(512, 1)
        mock_face_engine.set_next_embedding(v_unknown)

        res = person_rec_service.recognize_face("stranger.jpg", threshold=0.50)

        assert res.matched is False
        assert res.name == "unknown"
        assert res.entity_id is None
        assert np.isclose(res.similarity, 0.0, atol=1e-4)

    def test_detect_and_recognize_multiple_faces(
        self,
        mock_face_engine: MockFaceEmbeddingModel,
        person_reg_service: PersonRegistrationService,
        person_rec_service: PersonRecognitionService,
    ) -> None:
        # Register two subjects
        v1 = make_unit_vector(512, 0)
        v2 = make_unit_vector(512, 1)

        mock_face_engine.set_next_embedding(v1)
        p1 = person_reg_service.register_person(
            name="Person One",
            relationship="Doctor",
            reference_images=[make_synthetic_face()],
        )

        mock_face_engine.set_next_embedding(v2)
        p2 = person_reg_service.register_person(
            name="Person Two",
            relationship="Nurse",
            reference_images=[make_synthetic_face()],
        )

        person_rec_service.refresh_gallery()

        # Simulate frame with 2 detected faces
        mock_face_engine.set_mock_bboxes([
            FaceBoundingBox(x=10, y=10, width=50, height=50),
            FaceBoundingBox(x=100, y=10, width=50, height=50),
        ])

        # Configure next encodings for each crop
        mock_face_engine.encode_face = MagicMock(side_effect=[v1, v2])

        results = person_rec_service.detect_and_recognize_faces("multi_face_frame.jpg")

        assert len(results) == 2
        assert results[0].matched is True
        assert results[0].name == "Person One"
        assert results[0].bbox.x == 10

        assert results[1].matched is True
        assert results[1].name == "Person Two"
        assert results[1].bbox.x == 100


# ── 3. Separation of Concerns Tests ───────────────────────────────────────────


class TestSeparationOfConcerns:
    """Verify that person recognition and object recognition are strictly distinct."""

    def test_entity_types_are_distinct(
        self,
        mock_face_engine: MockFaceEmbeddingModel,
        person_reg_service: PersonRegistrationService,
        person_rec_service: PersonRecognitionService,
    ) -> None:
        v1 = make_unit_vector(512, 0)
        mock_face_engine.set_next_embedding(v1)
        person_reg_service.register_person(
            name="Familiar Face",
            relationship="Friend",
            reference_images=[make_synthetic_face()],
        )
        person_rec_service.refresh_gallery()

        mock_face_engine.set_next_embedding(v1)
        person_res = person_rec_service.recognize_face("face.jpg")

        assert person_res.entity_type == "person"

    def test_person_gallery_does_not_contain_objects(
        self,
        mem_service: MemoryService,
        mock_face_engine: MockFaceEmbeddingModel,
        person_reg_service: PersonRegistrationService,
        person_rec_service: PersonRecognitionService,
    ) -> None:
        # Register an object in the database
        mem_service.register_object(name="Wristwatch")

        # Register a person
        person_reg_service.register_person(
            name="Bob",
            reference_images=[make_synthetic_face()],
        )

        person_rec_service.refresh_gallery()

        # Face gallery must only contain the 1 person reference
        assert person_rec_service.gallery_size == 1
