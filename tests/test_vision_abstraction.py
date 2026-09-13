"""
tests.test_vision_abstraction — Unit Tests for Vision Embedding Model Abstraction
==================================================================================

Tests cover:
    - VisionEmbeddingModel ABC contract enforcement
    - MockVisionEmbeddingModel lifecycle (load, unload, is_loaded)
    - Single image encoding with varied input types (str, Path, ndarray, PIL)
    - Batch image encoding
    - Output dimensionality and L2 normalization
    - Determinism and preset/queued embeddings
    - Cosine similarity computation (single pair & batch gallery)
    - Error handling & edge cases (not loaded, dimension mismatch, zero vectors)
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.vision import (
    ImageInput,
    MockVisionEmbeddingModel,
    VisionEmbeddingModel,
    batch_cosine_similarity,
    cosine_similarity,
)


# ── 1. Interface & ABC Contract Tests ──────────────────────────────────────────


class TestVisionEmbeddingModelContract:
    """Verify that VisionEmbeddingModel behaves as a strict ABC."""

    def test_cannot_instantiate_abc_directly(self) -> None:
        """VisionEmbeddingModel is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            VisionEmbeddingModel()  # type: ignore[abstract]

    def test_subclass_missing_abstract_methods_fails(self) -> None:
        """A subclass without implementing abstract methods cannot be instantiated."""

        class IncompleteModel(VisionEmbeddingModel):
            @property
            def model_name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteModel()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self) -> None:
        """A subclass implementing all abstract members instantiates cleanly."""

        class ConcreteModel(VisionEmbeddingModel):
            @property
            def model_name(self) -> str:
                return "test-model"

            @property
            def embedding_dim(self) -> int:
                return 128

            @property
            def is_loaded(self) -> bool:
                return True

            def load_model(self) -> None:
                pass

            def encode_image(self, image: ImageInput) -> np.ndarray:
                vec = np.ones(self.embedding_dim, dtype=np.float32)
                return vec / np.linalg.norm(vec)

        model = ConcreteModel()
        assert model.model_name == "test-model"
        assert model.embedding_dim == 128
        assert model.is_loaded is True

        emb = model.encode_image("dummy.jpg")
        assert emb.shape == (128,)
        assert np.isclose(np.linalg.norm(emb), 1.0)


# ── 2. Mock Model Lifecycle Tests ─────────────────────────────────────────────


class TestMockModelLifecycle:
    """Test loading, unloading, and lifecycle properties of MockVisionEmbeddingModel."""

    def test_default_unloaded_state(self) -> None:
        """Model initializes unloaded by default."""
        model = MockVisionEmbeddingModel()
        assert model.is_loaded is False
        assert model.model_name == "mock-clip-vit-b32"
        assert model.embedding_dim == 512

    def test_auto_load_parameter(self) -> None:
        """Setting auto_load=True initializes the model in loaded state."""
        model = MockVisionEmbeddingModel(auto_load=True)
        assert model.is_loaded is True

    def test_custom_parameters(self) -> None:
        """Model respects custom model_name and embedding_dim."""
        model = MockVisionEmbeddingModel(model_name="custom-vision", embedding_dim=256)
        assert model.model_name == "custom-vision"
        assert model.embedding_dim == 256

    def test_load_and_unload_cycle(self) -> None:
        """Explicit load_model and unload_model calls update state."""
        model = MockVisionEmbeddingModel()
        assert not model.is_loaded

        model.load_model()
        assert model.is_loaded

        # Idempotent load
        model.load_model()
        assert model.is_loaded

        model.unload_model()
        assert not model.is_loaded


# ── 3. Image Encoding Tests ───────────────────────────────────────────────────


class TestMockModelEncoding:
    """Test single and batch image encoding functionality."""

    @pytest.fixture
    def loaded_model(self) -> MockVisionEmbeddingModel:
        model = MockVisionEmbeddingModel(embedding_dim=512, auto_load=True)
        return model

    def test_encode_when_not_loaded_raises(self) -> None:
        """Encoding with an unloaded model raises RuntimeError."""
        model = MockVisionEmbeddingModel(auto_load=False)
        with pytest.raises(RuntimeError, match="not loaded"):
            model.encode_image("sample.jpg")

    def test_encode_none_raises_value_error(
        self, loaded_model: MockVisionEmbeddingModel
    ) -> None:
        """Encoding None raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            loaded_model.encode_image(None)  # type: ignore[arg-type]

    def test_encode_empty_string_raises(
        self, loaded_model: MockVisionEmbeddingModel
    ) -> None:
        """Encoding an empty or whitespace string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            loaded_model.encode_image("   ")

    def test_encode_empty_ndarray_raises(
        self, loaded_model: MockVisionEmbeddingModel
    ) -> None:
        """Encoding an empty ndarray raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            loaded_model.encode_image(np.array([]))

    def test_encode_string_path(
        self, loaded_model: MockVisionEmbeddingModel
    ) -> None:
        """Encoding a string path returns a normalized 1D vector."""
        emb = loaded_model.encode_image("images/watch.jpg")
        assert isinstance(emb, np.ndarray)
        assert emb.dtype == np.float32
        assert emb.shape == (512,)
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)

    def test_encode_pathlib_path(
        self, loaded_model: MockVisionEmbeddingModel
    ) -> None:
        """Encoding a pathlib.Path returns a normalized vector."""
        emb = loaded_model.encode_image(Path("images/necklace.png"))
        assert emb.shape == (512,)
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)

    def test_encode_numpy_array(
        self, loaded_model: MockVisionEmbeddingModel
    ) -> None:
        """Encoding a simulated OpenCV frame (H, W, C) returns a valid embedding."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        emb = loaded_model.encode_image(frame)
        assert emb.shape == (512,)
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)

    def test_encode_pil_image(
        self, loaded_model: MockVisionEmbeddingModel
    ) -> None:
        """Encoding a PIL Image returns a valid embedding."""
        img = Image.new("RGB", (200, 200), color="blue")
        emb = loaded_model.encode_image(img)
        assert emb.shape == (512,)
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)

    def test_encoding_is_deterministic(
        self, loaded_model: MockVisionEmbeddingModel
    ) -> None:
        """Identical inputs produce identical embedding vectors."""
        emb1 = loaded_model.encode_image("data/wallet.jpg")
        emb2 = loaded_model.encode_image("data/wallet.jpg")
        np.testing.assert_allclose(emb1, emb2)
        assert np.isclose(loaded_model.similarity(emb1, emb2), 1.0)

    def test_different_inputs_produce_different_embeddings(
        self, loaded_model: MockVisionEmbeddingModel
    ) -> None:
        """Different inputs produce distinct embeddings with similarity < 1.0."""
        emb_a = loaded_model.encode_image("object_a.jpg")
        emb_b = loaded_model.encode_image("object_b.jpg")
        sim = loaded_model.similarity(emb_a, emb_b)
        assert sim < 0.99


# ── 4. Batch Encoding Tests ───────────────────────────────────────────────────


class TestBatchEncoding:
    """Test batch image encoding via encode_images."""

    @pytest.fixture
    def loaded_model(self) -> MockVisionEmbeddingModel:
        return MockVisionEmbeddingModel(embedding_dim=256, auto_load=True)

    def test_encode_images_empty_sequence(
        self, loaded_model: MockVisionEmbeddingModel
    ) -> None:
        """Empty sequence returns a 2D array of shape (0, dim)."""
        res = loaded_model.encode_images([])
        assert res.shape == (0, 256)
        assert res.dtype == np.float32

    def test_encode_images_multiple(
        self, loaded_model: MockVisionEmbeddingModel
    ) -> None:
        """Batch of N images returns (N, dim) matrix where each row is unit-norm."""
        inputs = ["img1.jpg", "img2.jpg", "img3.jpg"]
        matrix = loaded_model.encode_images(inputs)

        assert matrix.shape == (3, 256)
        assert matrix.dtype == np.float32

        for i, item in enumerate(inputs):
            single = loaded_model.encode_image(item)
            np.testing.assert_allclose(matrix[i], single)
            assert np.isclose(np.linalg.norm(matrix[i]), 1.0, atol=1e-5)


# ── 5. Preset and Queued Mock Embeddings ───────────────────────────────────────


class TestMockCustomEmbeddings:
    """Test injecting specific embeddings for deterministic test scenarios."""

    def test_preset_embedding(self) -> None:
        """Preset embedding is returned when matched key is encoded."""
        model = MockVisionEmbeddingModel(embedding_dim=4, auto_load=True)
        custom_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        model.set_preset_embedding("gold_watch", custom_vec)

        emb = model.encode_image("gold_watch")
        np.testing.assert_allclose(emb, custom_vec)

    def test_next_embedding_queue(self) -> None:
        """Queued next embedding is returned once, then clears."""
        model = MockVisionEmbeddingModel(embedding_dim=4, auto_load=True)
        special_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        model.set_next_embedding(special_vec)

        # First call returns queued vector
        emb1 = model.encode_image("any_image.jpg")
        np.testing.assert_allclose(emb1, special_vec)

        # Subsequent call returns default deterministic vector
        emb2 = model.encode_image("any_image.jpg")
        assert not np.allclose(emb2, special_vec)


# ── 6. Cosine Similarity Function Tests ────────────────────────────────────────


class TestCosineSimilarity:
    """Test mathematical properties and edge cases of cosine_similarity."""

    def test_identical_vectors_similarity_is_one(self) -> None:
        """Identical vectors produce a similarity of 1.0."""
        v = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        assert np.isclose(cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors_similarity_is_zero(self) -> None:
        """Orthogonal vectors produce a similarity of 0.0."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert np.isclose(cosine_similarity(v1, v2), 0.0)

    def test_opposite_vectors_similarity_is_minus_one(self) -> None:
        """Opposite vectors produce a similarity of -1.0."""
        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        assert np.isclose(cosine_similarity(v1, v2), -1.0)

    def test_zero_vector_returns_zero(self) -> None:
        """A zero-magnitude vector returns 0.0 similarity without dividing by zero."""
        v1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(v1, v2) == 0.0
        assert cosine_similarity(v1, v1) == 0.0

    def test_dimension_mismatch_raises_value_error(self) -> None:
        """Vectors of unequal lengths raise ValueError."""
        v1 = np.array([1.0, 2.0], dtype=np.float32)
        v2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="dimensions must match"):
            cosine_similarity(v1, v2)

    def test_clamps_to_range(self) -> None:
        """Slight floating point overflow does not exceed [-1.0, 1.0]."""
        v = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        sim = cosine_similarity(v, v)
        assert -1.0 <= sim <= 1.0


# ── 7. Batch Cosine Similarity Tests ──────────────────────────────────────────


class TestBatchCosineSimilarity:
    """Test batch similarity comparison between a query vector and a gallery matrix."""

    def test_batch_similarity_matches_pairwise(self) -> None:
        """Batch results match individual cosine_similarity calls."""
        query = np.array([1.0, 0.5, -0.2], dtype=np.float32)
        gallery = np.array(
            [
                [1.0, 0.5, -0.2],  # identical -> 1.0
                [-1.0, -0.5, 0.2], # opposite -> -1.0
                [0.0, 0.0, 1.0],   # third direction
            ],
            dtype=np.float32,
        )

        sims = batch_cosine_similarity(query, gallery)
        assert sims.shape == (3,)

        for i in range(3):
            expected = cosine_similarity(query, gallery[i])
            assert np.isclose(sims[i], expected, atol=1e-5)

    def test_batch_similarity_empty_gallery(self) -> None:
        """Empty gallery returns an empty 1D array."""
        query = np.array([1.0, 2.0], dtype=np.float32)
        gallery = np.empty((0, 2), dtype=np.float32)
        sims = batch_cosine_similarity(query, gallery)
        assert sims.shape == (0,)

    def test_batch_similarity_zero_query(self) -> None:
        """Zero query vector returns all zeros."""
        query = np.zeros(3, dtype=np.float32)
        gallery = np.eye(3, dtype=np.float32)
        sims = batch_cosine_similarity(query, gallery)
        np.testing.assert_allclose(sims, np.zeros(3, dtype=np.float32))

    def test_batch_dimension_mismatch_raises(self) -> None:
        """Dimension mismatch between query and gallery columns raises ValueError."""
        query = np.array([1.0, 2.0], dtype=np.float32)
        gallery = np.eye(3, dtype=np.float32)
        with pytest.raises(ValueError, match="does not match"):
            batch_cosine_similarity(query, gallery)

    def test_model_similarity_methods_delegate_correctly(self) -> None:
        """VisionEmbeddingModel instance methods similarity and batch_similarity match helpers."""
        model = MockVisionEmbeddingModel(embedding_dim=4, auto_load=True)
        a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        gallery = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

        assert model.similarity(a, b) == cosine_similarity(a, b)
        np.testing.assert_allclose(
            model.batch_similarity(a, gallery),
            batch_cosine_similarity(a, gallery),
        )
