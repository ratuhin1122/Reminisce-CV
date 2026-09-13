"""
app.vision.mock — Mock Vision Embedding Model
=============================================

Lightweight test double and placeholder implementing ``VisionEmbeddingModel``.
Generates deterministic, L2-normalized pseudo-embeddings without requiring
PyTorch, GPU, or model weight downloads.

Used for:
- Unit and integration tests
- Offline development and CI/CD pipelines
- Validating the vision abstraction layer
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from app.vision.base import ImageInput, VisionEmbeddingModel, cosine_similarity


class MockVisionEmbeddingModel(VisionEmbeddingModel):
    """Mock implementation of ``VisionEmbeddingModel`` for testing and prototyping.

    Parameters
    ----------
    model_name : str, optional
        Identifier for the mock model. Default is ``"mock-clip-vit-b32"``.
    embedding_dim : int, optional
        Dimensionality of the mock embeddings. Default is 512.
    auto_load : bool, optional
        If True, initializes in loaded state. Default is False to test
        ``load_model()`` lifecycle checks.
    seed : int, optional
        Global seed for fallback random generation. Default is 42.
    """

    def __init__(
        self,
        model_name: str = "mock-clip-vit-b32",
        embedding_dim: int = 512,
        auto_load: bool = False,
        seed: int = 42,
    ) -> None:
        self._model_name = model_name
        self._embedding_dim = embedding_dim
        self._is_loaded = auto_load
        self._seed = seed
        self._preset_embeddings: Dict[str, np.ndarray] = {}
        self._next_embedding: Optional[np.ndarray] = None

    @property
    def model_name(self) -> str:
        """Identifier for the mock model."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the mock embeddings."""
        return self._embedding_dim

    @property
    def is_loaded(self) -> bool:
        """Whether the mock model is currently loaded."""
        return self._is_loaded

    def load_model(self) -> None:
        """Simulate loading model weights into memory."""
        self._is_loaded = True

    def unload_model(self) -> None:
        """Simulate unloading model weights from memory."""
        self._is_loaded = False

    def set_preset_embedding(self, key: str, embedding: np.ndarray) -> None:
        """Register a specific embedding to be returned when key is matched.

        Parameters
        ----------
        key : str
            String identifier matching the image input (e.g. image path or name).
        embedding : np.ndarray
            Embedding vector to return. Will be normalized to unit length.
        """
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        self._preset_embeddings[key] = emb

    def set_next_embedding(self, embedding: Optional[np.ndarray]) -> None:
        """Queue a specific embedding to be returned on the next ``encode_image`` call.

        Parameters
        ----------
        embedding : np.ndarray or None
            Vector to return on the next call, or None to clear.
        """
        if embedding is not None:
            emb = np.asarray(embedding, dtype=np.float32).ravel()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            self._next_embedding = emb
        else:
            self._next_embedding = None

    def _generate_deterministic_vector(self, seed_data: bytes) -> np.ndarray:
        """Generate a deterministic unit-length vector from binary input."""
        # Use SHA256 to produce an initial seed
        hash_digest = hashlib.sha256(seed_data).digest()
        seed = int.from_bytes(hash_digest[:4], byteorder="big")

        rng = np.random.RandomState(seed)
        raw = rng.randn(self._embedding_dim).astype(np.float32)
        norm = np.linalg.norm(raw)
        if norm == 0:
            raw[0] = 1.0
            norm = 1.0
        return (raw / norm).astype(np.float32)

    def _extract_seed_data(self, image: ImageInput) -> bytes:
        """Extract a stable byte sequence from various image input types."""
        if image is None:
            raise ValueError("Image input cannot be None.")

        if isinstance(image, (str, Path)):
            path_str = str(image).strip()
            if not path_str:
                raise ValueError("Image path cannot be empty.")
            # Check if file exists on disk, hash content if so, else hash path string
            p = Path(path_str)
            if p.is_file():
                try:
                    return p.read_bytes()
                except (OSError, PermissionError):
                    return path_str.encode("utf-8")
            return path_str.encode("utf-8")

        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Image array cannot be empty.")
            # Hash array shape and sample bytes for deterministic signature
            shape_bytes = f"{image.shape}_{image.dtype}".encode("utf-8")
            sample_bytes = image.tobytes()[:4096]
            return shape_bytes + sample_bytes

        # PIL Image or similar object
        if hasattr(image, "tobytes") and callable(getattr(image, "tobytes")):
            size = getattr(image, "size", (0, 0))
            mode = getattr(image, "mode", "RGB")
            header = f"{size}_{mode}".encode("utf-8")
            return header + image.tobytes()[:4096]

        return repr(image).encode("utf-8")

    def encode_image(self, image: ImageInput) -> np.ndarray:
        """Extract a mock normalized 1D embedding vector.

        Parameters
        ----------
        image : ImageInput
            Input image representation.

        Returns
        -------
        np.ndarray
            1D float32 numpy array of shape ``(embedding_dim,)`` with unit norm.

        Raises
        ------
        RuntimeError
            If model is not loaded.
        ValueError
            If image is invalid or empty.
        """
        if not self._is_loaded:
            raise RuntimeError(
                f"Model '{self._model_name}' is not loaded. Call load_model() first."
            )

        # Check queued one-time embedding
        if self._next_embedding is not None:
            emb = self._next_embedding
            self._next_embedding = None
            return emb.copy()

        # Check preset dictionary
        if isinstance(image, (str, Path)):
            key = str(image)
            if key in self._preset_embeddings:
                return self._preset_embeddings[key].copy()

        seed_data = self._extract_seed_data(image)
        return self._generate_deterministic_vector(seed_data)
