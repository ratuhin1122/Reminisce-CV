"""
tests.test_object_registration — Tests for Personal Object Registration Workflow
==================================================================================

Tests cover:
    - Input image validation (PIL, OpenCV BGR arrays, file paths, corruption checks)
    - Object registration with single and multiple reference images
    - Local file storage of images and .npy embeddings
    - Database relationship storage and embedding metadata tracking
    - Retrieval of registered objects and loading embeddings from disk
    - Batch embedding loading for recognition
    - Cascading deletion of records and disk files
    - Rollback behavior on unexpected errors
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from app.database.models import EntityType
from app.memory import (
    MemoryService,
    ObjectRegistrationService,
    RegisteredObject,
    RegisteredReference,
)
from app.vision import MockVisionEmbeddingModel


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_vision() -> MockVisionEmbeddingModel:
    """Mock vision model providing deterministic 512-dim normalized vectors."""
    model = MockVisionEmbeddingModel(embedding_dim=512, auto_load=True)
    return model


@pytest.fixture
def temp_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Provide isolated temporary directories for references and embeddings."""
    ref_dir = tmp_path / "data" / "references"
    emb_dir = tmp_path / "data" / "embeddings"
    ref_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    return ref_dir, emb_dir


@pytest.fixture
def reg_service(
    mock_vision: MockVisionEmbeddingModel,
    temp_dirs: tuple[Path, Path],
) -> ObjectRegistrationService:
    """Provide an ObjectRegistrationService with in-memory SQLite and isolated dirs."""
    ref_dir, emb_dir = temp_dirs
    mem_svc = MemoryService(db_path=":memory:")
    mem_svc.start()

    service = ObjectRegistrationService(
        memory_service=mem_svc,
        vision_model=mock_vision,
        references_dir=ref_dir,
        embeddings_dir=emb_dir,
    )
    yield service
    service.stop()


def create_sample_image(color: str = "blue", size: tuple[int, int] = (64, 64)) -> Image.Image:
    """Helper to generate a synthetic PIL Image."""
    return Image.new("RGB", size, color=color)


# ── 1. Image Validation Tests ─────────────────────────────────────────────────


class TestImageValidation:
    """Test validation and format normalization of reference images."""

    def test_validate_none_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be None"):
            ObjectRegistrationService.validate_image(None)

    def test_validate_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            ObjectRegistrationService.validate_image("   ")

    def test_validate_nonexistent_file_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            ObjectRegistrationService.validate_image("nonexistent_test_image.jpg")

    def test_validate_corrupted_file_raises(self, tmp_path: Path) -> None:
        corrupt_file = tmp_path / "bad.jpg"
        corrupt_file.write_bytes(b"not a valid image format header")
        with pytest.raises(ValueError, match="Corrupted or unreadable"):
            ObjectRegistrationService.validate_image(corrupt_file)

    def test_validate_pil_image(self) -> None:
        pil_rgba = Image.new("RGBA", (50, 50), color=(10, 20, 30, 200))
        res = ObjectRegistrationService.validate_image(pil_rgba)
        assert isinstance(res, Image.Image)
        assert res.mode == "RGB"
        assert res.size == (50, 50)

    def test_validate_opencv_bgr_array(self) -> None:
        # BGR blue array
        bgr = np.zeros((40, 40, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255
        res = ObjectRegistrationService.validate_image(bgr)
        assert isinstance(res, Image.Image)
        assert res.mode == "RGB"
        rgb_data = np.array(res)
        assert rgb_data[0, 0, 0] == 0    # R
        assert rgb_data[0, 0, 2] == 255  # B

    def test_validate_valid_file_path(self, tmp_path: Path) -> None:
        img_path = tmp_path / "valid.jpg"
        Image.new("RGB", (32, 32), color="green").save(img_path)
        res = ObjectRegistrationService.validate_image(img_path)
        assert res.size == (32, 32)

    def test_validate_empty_ndarray_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            ObjectRegistrationService.validate_image(np.array([]))


# ── 2. Registration Tests ─────────────────────────────────────────────────────


class TestObjectRegistration:
    """Test object registration with metadata, images, and embeddings."""

    def test_register_empty_name_raises(
        self, reg_service: ObjectRegistrationService
    ) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            reg_service.register_object(name="", reference_images=[create_sample_image()])

    def test_register_without_images_raises(
        self, reg_service: ObjectRegistrationService
    ) -> None:
        with pytest.raises(ValueError, match="At least one reference image is required"):
            reg_service.register_object(name="Test Item", reference_images=[])

    def test_register_single_image_success(
        self, reg_service: ObjectRegistrationService
    ) -> None:
        img = create_sample_image("red")
        reg_obj = reg_service.register_object(
            name="Test Object 01",
            giver="Test Giver",
            occasion="Event 2024",
            year="2024",
            narrative="A generic test item narrative.",
            reference_images=[img],
        )

        assert isinstance(reg_obj, RegisteredObject)
        assert reg_obj.memory_id > 0
        assert reg_obj.name == "Test Object 01"
        assert reg_obj.giver == "Test Giver"
        assert len(reg_obj.references) == 1

        ref = reg_obj.references[0]
        assert Path(ref.image_path).is_file()
        assert Path(ref.embedding_path).is_file()
        assert ref.embedding.shape == (512,)
        assert np.isclose(np.linalg.norm(ref.embedding), 1.0, atol=1e-5)

    def test_register_multiple_images(
        self, reg_service: ObjectRegistrationService
    ) -> None:
        imgs = [
            create_sample_image("red"),
            create_sample_image("green"),
            create_sample_image("blue"),
        ]
        reg_obj = reg_service.register_object(
            name="Multi-Angle Object",
            reference_images=imgs,
        )

        assert len(reg_obj.references) == 3
        for ref in reg_obj.references:
            assert Path(ref.image_path).is_file()
            assert Path(ref.embedding_path).is_file()
            assert ref.embedding.shape == (512,)
            assert np.isclose(np.linalg.norm(ref.embedding), 1.0, atol=1e-5)

    def test_register_with_opencv_array_and_path(
        self, reg_service: ObjectRegistrationService, tmp_path: Path
    ) -> None:
        # 1. File on disk
        disk_img = tmp_path / "source.png"
        Image.new("RGB", (48, 48), color="purple").save(disk_img)

        # 2. OpenCV numpy array
        cv_frame = np.full((48, 48, 3), 100, dtype=np.uint8)

        reg_obj = reg_service.register_object(
            name="Mixed Source Object",
            reference_images=[disk_img, cv_frame],
        )

        assert len(reg_obj.references) == 2
        for ref in reg_obj.references:
            assert Path(ref.image_path).is_file()
            assert Path(ref.embedding_path).is_file()

    def test_alias_fields_handled(
        self, reg_service: ObjectRegistrationService
    ) -> None:
        """Verify that object_name, title, giver_name, date, and narrative_memory work as aliases."""
        reg_obj = reg_service.register_object(
            object_name="Heirloom Item",
            title="Special Title",
            giver_name="Relative",
            date="2020",
            narrative_memory="Memory story.",
            reference_images=[create_sample_image()],
        )
        assert reg_obj.name == "Special Title"
        assert reg_obj.giver == "Relative"
        assert reg_obj.year == "2020"
        assert reg_obj.narrative == "Memory story."


# ── 3. Retrieval Tests ────────────────────────────────────────────────────────


class TestObjectRetrieval:
    """Test retrieving registered objects and loading embeddings from disk."""

    def test_get_nonexistent_returns_none(
        self, reg_service: ObjectRegistrationService
    ) -> None:
        assert reg_service.get_registered_object(9999) is None

    def test_get_registered_object_loads_embeddings(
        self, reg_service: ObjectRegistrationService
    ) -> None:
        orig = reg_service.register_object(
            name="Pocket Watch",
            giver="Grandparent",
            reference_images=[create_sample_image("yellow"), create_sample_image("cyan")],
        )

        retrieved = reg_service.get_registered_object(orig.memory_id)
        assert retrieved is not None
        assert retrieved.name == "Pocket Watch"
        assert retrieved.giver == "Grandparent"
        assert len(retrieved.references) == 2

        # Check that embeddings match the original saved vectors
        for i in range(2):
            orig_emb = orig.references[i].embedding
            ret_emb = retrieved.references[i].embedding
            np.testing.assert_allclose(orig_emb, ret_emb, rtol=1e-5)

    def test_get_all_registered_objects(
        self, reg_service: ObjectRegistrationService
    ) -> None:
        reg_service.register_object(
            name="Item Alpha",
            reference_images=[create_sample_image("red")],
        )
        reg_service.register_object(
            name="Item Beta",
            reference_images=[create_sample_image("blue")],
        )

        all_objs = reg_service.get_all_registered_objects()
        assert len(all_objs) == 2
        names = {o.name for o in all_objs}
        assert names == {"Item Alpha", "Item Beta"}

    def test_load_all_object_embeddings(
        self, reg_service: ObjectRegistrationService
    ) -> None:
        obj1 = reg_service.register_object(
            name="Item 1",
            reference_images=[create_sample_image("red"), create_sample_image("pink")],
        )
        obj2 = reg_service.register_object(
            name="Item 2",
            reference_images=[create_sample_image("green")],
        )

        ids, embs = reg_service.load_all_object_embeddings()
        assert len(ids) == 3
        assert len(embs) == 3
        assert ids == [obj1.memory_id, obj1.memory_id, obj2.memory_id]
        assert all(isinstance(e, np.ndarray) and e.shape == (512,) for e in embs)


# ── 4. Deletion Tests ─────────────────────────────────────────────────────────


class TestObjectDeletion:
    """Test object deletion and disk cleanup."""

    def test_delete_nonexistent_returns_false(
        self, reg_service: ObjectRegistrationService
    ) -> None:
        assert not reg_service.delete_registered_object(9999)

    def test_delete_removes_database_and_disk_files(
        self, reg_service: ObjectRegistrationService
    ) -> None:
        reg_obj = reg_service.register_object(
            name="Temporary Item",
            reference_images=[create_sample_image()],
        )
        mem_id = reg_obj.memory_id
        img_path = Path(reg_obj.references[0].image_path)
        emb_path = Path(reg_obj.references[0].embedding_path)

        assert img_path.is_file()
        assert emb_path.is_file()

        success = reg_service.delete_registered_object(mem_id)
        assert success is True

        # Database record gone
        assert reg_service.get_registered_object(mem_id) is None

        # Disk files cleaned up
        assert not img_path.is_file()
        assert not emb_path.is_file()


# ── 5. Error Recovery and Rollback ────────────────────────────────────────────


class TestRollbackOnFailure:
    """Test that failed registration cleans up partial database entries and files."""

    def test_rollback_on_encoding_error(
        self, reg_service: ObjectRegistrationService
    ) -> None:
        # Configure vision model to fail on second image
        mock_model = MagicMock()
        mock_model.is_loaded = True
        mock_model.model_name = "mock"
        mock_model.embedding_dim = 512
        mock_model.encode_image.side_effect = [
            np.ones(512, dtype=np.float32),  # 1st image succeeds
            RuntimeError("Simulated GPU out of memory"),  # 2nd image fails
        ]

        reg_service._vision_model = mock_model

        imgs = [create_sample_image("red"), create_sample_image("blue")]

        with pytest.raises(RuntimeError, match="Object registration failed"):
            reg_service.register_object(
                name="Failing Object",
                reference_images=imgs,
            )

        # Confirm no leftover memories in DB
        assert len(reg_service.get_all_registered_objects()) == 0
