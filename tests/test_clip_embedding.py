"""
tests.test_clip_embedding — Unit Tests for CLIP Vision Embedding Engine
========================================================================

Tests cover:
    - CLIPVisionModel initialization and properties
    - Device resolution (CPU vs CUDA auto-detection)
    - Image preprocessing for PIL, OpenCV BGR arrays, and file paths
    - Error handling for invalid shapes, missing files, and uninitialized models
    - Model caching and idempotent loading
    - Embedding extraction and L2-normalization
    - Batch inference via encode_images
    - Cosine similarity computation: (A · B) / (||A|| ||B||)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.vision import (
    CLIPEmbeddingEngine,
    CLIPVisionModel,
    VisionEmbeddingModel,
    cosine_similarity,
)


# ── 1. Initialization and Inheritance ─────────────────────────────────────────


class TestCLIPModelInitialization:
    """Test model configuration, default parameters, and interface compliance."""

    def test_inherits_from_vision_embedding_model(self) -> None:
        """CLIPVisionModel must satisfy the VisionEmbeddingModel interface."""
        assert issubclass(CLIPVisionModel, VisionEmbeddingModel)

    def test_alias_is_identical(self) -> None:
        """CLIPEmbeddingEngine is an alias for CLIPVisionModel."""
        assert CLIPEmbeddingEngine is CLIPVisionModel

    def test_default_attributes(self) -> None:
        """Verify default parameters from central configuration."""
        model = CLIPVisionModel()
        assert model.model_name == "openai/clip-vit-base-patch32"
        assert model.embedding_dim == 512
        assert model.is_loaded is False
        assert model.device in ("cpu", "cuda")

    def test_custom_attributes(self) -> None:
        """Model accepts custom configuration values."""
        model = CLIPVisionModel(
            model_name="custom/clip-model",
            device="cpu",
            embedding_dim=768,
        )
        assert model.model_name == "custom/clip-model"
        assert model.device == "cpu"
        assert model.embedding_dim == 768


# ── 2. Device Resolution ──────────────────────────────────────────────────────


class TestDeviceResolution:
    """Test CPU/CUDA selection and fallback logic."""

    def test_explicit_cpu(self) -> None:
        assert CLIPVisionModel._resolve_device("cpu") == "cpu"
        assert CLIPVisionModel._resolve_device("CPU") == "cpu"

    def test_explicit_cuda(self) -> None:
        assert CLIPVisionModel._resolve_device("cuda") == "cuda"
        assert CLIPVisionModel._resolve_device("CUDA ") == "cuda"

    def test_auto_detect_fallback(self) -> None:
        dev = CLIPVisionModel._resolve_device(None)
        assert dev in ("cpu", "cuda")


# ── 3. Image Preprocessing ────────────────────────────────────────────────────


class TestImagePreprocessing:
    """Test converting PIL, OpenCV numpy arrays, and file paths to RGB PIL Images."""

    @pytest.fixture
    def model(self) -> CLIPVisionModel:
        return CLIPVisionModel(device="cpu")

    def test_pil_image_converted_to_rgb(self, model: CLIPVisionModel) -> None:
        # RGBA PIL Image
        img_rgba = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
        res = model._preprocess_image(img_rgba)
        assert isinstance(res, Image.Image)
        assert res.mode == "RGB"

    def test_opencv_bgr_array_to_rgb(self, model: CLIPVisionModel) -> None:
        # OpenCV standard 3-channel BGR uint8 array (Blue=255, Green=0, Red=0)
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # Pure Blue in BGR
        res = model._preprocess_image(bgr)
        assert isinstance(res, Image.Image)
        assert res.mode == "RGB"
        # In RGB, the red channel should be 0 and the blue channel should be 255
        rgb_data = np.array(res)
        assert rgb_data[0, 0, 0] == 0    # R
        assert rgb_data[0, 0, 2] == 255  # B

    def test_opencv_grayscale_array(self, model: CLIPVisionModel) -> None:
        gray = np.full((60, 60), 128, dtype=np.uint8)
        res = model._preprocess_image(gray)
        assert res.mode == "RGB"
        assert res.size == (60, 60)

    def test_opencv_bgra_array(self, model: CLIPVisionModel) -> None:
        bgra = np.zeros((40, 40, 4), dtype=np.uint8)
        bgra[:, :, 2] = 255  # Red in BGR
        res = model._preprocess_image(bgra)
        assert res.mode == "RGB"
        rgb_data = np.array(res)
        assert rgb_data[0, 0, 0] == 255

    def test_file_path_loading(self, model: CLIPVisionModel, tmp_path: Path) -> None:
        test_file = tmp_path / "sample.jpg"
        Image.new("RGB", (30, 30), color="green").save(test_file)

        res = model._preprocess_image(test_file)
        assert isinstance(res, Image.Image)
        assert res.size == (30, 30)

        # String path also accepted
        res2 = model._preprocess_image(str(test_file))
        assert isinstance(res2, Image.Image)

    def test_nonexistent_file_raises(self, model: CLIPVisionModel) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            model._preprocess_image("nonexistent_file_12345.jpg")

    def test_none_image_raises(self, model: CLIPVisionModel) -> None:
        with pytest.raises(ValueError, match="cannot be None"):
            model._preprocess_image(None)  # type: ignore[arg-type]

    def test_empty_string_raises(self, model: CLIPVisionModel) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            model._preprocess_image("   ")

    def test_empty_ndarray_raises(self, model: CLIPVisionModel) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            model._preprocess_image(np.array([]))

    def test_invalid_channel_count_raises(self, model: CLIPVisionModel) -> None:
        invalid = np.zeros((10, 10, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported number of image channels"):
            model._preprocess_image(invalid)


# ── 4. Lifecycle & Caching ────────────────────────────────────────────────────


class TestModelLifecycleAndCaching:
    """Test model loading, idempotency, unloading, and error recovery."""

    def test_encode_before_load_raises(self) -> None:
        model = CLIPVisionModel()
        with pytest.raises(RuntimeError, match="not loaded"):
            model.encode_image(Image.new("RGB", (10, 10)))

    def test_idempotent_load_does_not_reload(self) -> None:
        model = CLIPVisionModel()
        mock_model = MagicMock()
        mock_proc = MagicMock()

        model._is_loaded = True
        model._model = mock_model
        model._processor = mock_proc

        # Calling load_model again should return immediately
        model.load_model()
        assert model.is_loaded
        assert model._model is mock_model

    def test_unload_model(self) -> None:
        model = CLIPVisionModel()
        model._is_loaded = True
        model._model = MagicMock()
        model._processor = MagicMock()

        model.unload_model()
        assert not model.is_loaded
        assert model._model is None
        assert model._processor is None

    def test_load_failure_handles_gracefully(self) -> None:
        model = CLIPVisionModel(model_name="nonexistent/fake-clip-model")
        with patch("transformers.CLIPModel.from_pretrained", side_effect=Exception("Download failed")):
            with patch("transformers.CLIPProcessor.from_pretrained"):
                with pytest.raises(RuntimeError, match="Failed to load CLIP model"):
                    model.load_model()
        assert not model.is_loaded
        assert model._model is None


# ── 5. Embedding Generation with Mocks ─────────────────────────────────────────


class TestEmbeddingGeneration:
    """Test single and batch embedding inference and L2 normalization."""

    @pytest.fixture
    def mock_loaded_model(self) -> CLIPVisionModel:
        import torch

        model = CLIPVisionModel(device="cpu", embedding_dim=512)
        model._is_loaded = True
        model._torch = torch

        # Mock processor
        mock_proc = MagicMock()
        mock_proc.return_value = {"pixel_values": torch.zeros((1, 3, 224, 224))}
        model._processor = mock_proc

        # Mock CLIPModel: return unnormalized vector [3.0, 4.0, 0, ..., 0]
        raw_feat = torch.zeros((1, 512), dtype=torch.float32)
        raw_feat[0, 0] = 3.0
        raw_feat[0, 1] = 4.0
        mock_clip = MagicMock()
        mock_clip.get_image_features.return_value = raw_feat
        model._model = mock_clip

        return model

    def test_encode_image_returns_normalized_vector(
        self, mock_loaded_model: CLIPVisionModel
    ) -> None:
        img = Image.new("RGB", (100, 100), color="red")
        emb = mock_loaded_model.encode_image(img)

        assert isinstance(emb, np.ndarray)
        assert emb.dtype == np.float32
        assert emb.shape == (512,)
        # raw was [3, 4], norm is 5, normalized is [0.6, 0.8]
        assert np.isclose(emb[0], 0.6, atol=1e-5)
        assert np.isclose(emb[1], 0.8, atol=1e-5)
        # Unit norm
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)

    def test_encode_images_batch(
        self, mock_loaded_model: CLIPVisionModel
    ) -> None:
        import torch

        # Update processor mock for batch of 2
        mock_loaded_model._processor.return_value = {
            "pixel_values": torch.zeros((2, 3, 224, 224))
        }
        raw_feat = torch.zeros((2, 512), dtype=torch.float32)
        raw_feat[0, 0] = 1.0
        raw_feat[1, 1] = 2.0
        mock_loaded_model._model.get_image_features.return_value = raw_feat

        images = [Image.new("RGB", (50, 50)), Image.new("RGB", (50, 50))]
        matrix = mock_loaded_model.encode_images(images)

        assert matrix.shape == (2, 512)
        assert matrix.dtype == np.float32
        assert np.isclose(np.linalg.norm(matrix[0]), 1.0, atol=1e-5)
        assert np.isclose(np.linalg.norm(matrix[1]), 1.0, atol=1e-5)

    def test_encode_images_empty(
        self, mock_loaded_model: CLIPVisionModel
    ) -> None:
        matrix = mock_loaded_model.encode_images([])
        assert matrix.shape == (0, 512)
        assert matrix.dtype == np.float32


# ── 6. Cosine Similarity Formula Verification ─────────────────────────────────


class TestCosineSimilarityFormula:
    """Verify exact formula: similarity(A, B) = (A · B) / (||A|| ||B||)."""

    def test_similarity_matches_formula(self) -> None:
        model = CLIPVisionModel()
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        # Manual calculation of (A · B) / (||A|| ||B||)
        expected = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        sim = model.similarity(a, b)
        assert np.isclose(sim, expected, atol=1e-6)
        assert np.isclose(sim, cosine_similarity(a, b), atol=1e-6)
