"""
app.vision.base — Vision Embedding Model Abstraction
====================================================

Abstract base class defining the contract for all vision embedding models.
Any concrete model (e.g. OpenCLIP, HuggingFace CLIP, mock/test models)
must implement this interface.

This ensures the rest of ReminisceCV remains decoupled from any specific
deep learning framework, model architecture, or model provider.

Public Interface
----------------
- ``VisionEmbeddingModel``: Abstract base class for vision encoders.
- ``cosine_similarity``: Cosine similarity between two 1D vectors.
- ``batch_cosine_similarity``: Cosine similarity between a 1D query and a 2D gallery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np

# Type alias for acceptable image inputs across implementations
ImageInput = Union[np.ndarray, "Image.Image", Path, str]


def cosine_similarity(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
) -> float:
    """Compute cosine similarity between two 1D embedding vectors.

    Parameters
    ----------
    embedding_a : np.ndarray
        First embedding vector (1D array of length D).
    embedding_b : np.ndarray
        Second embedding vector (1D array of length D).

    Returns
    -------
    float
        Cosine similarity in the range [-1.0, 1.0].
        Returns 0.0 if either vector has zero magnitude.

    Raises
    ------
    ValueError
        If vectors do not have matching shapes or are not 1-dimensional.
    """
    a = np.asarray(embedding_a, dtype=np.float32).ravel()
    b = np.asarray(embedding_b, dtype=np.float32).ravel()

    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"Embedding dimensions must match: got {a.shape[0]} and {b.shape[0]}"
        )

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot = float(np.dot(a, b))
    sim = dot / (norm_a * norm_b)
    # Clamp to [-1.0, 1.0] to guard against floating-point inaccuracies
    return float(np.clip(sim, -1.0, 1.0))


def batch_cosine_similarity(
    query: np.ndarray,
    gallery: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between a 1D query and a 2D gallery of embeddings.

    Parameters
    ----------
    query : np.ndarray
        Query embedding vector of shape (D,).
    gallery : np.ndarray
        Gallery embedding matrix of shape (N, D).

    Returns
    -------
    np.ndarray
        1D array of length N containing cosine similarities in [-1.0, 1.0].

    Raises
    ------
    ValueError
        If gallery is empty or dimensions do not match.
    """
    q = np.asarray(query, dtype=np.float32).ravel()
    g = np.asarray(gallery, dtype=np.float32)

    if g.ndim == 1:
        g = g.reshape(1, -1)

    if g.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)

    if q.shape[0] != g.shape[1]:
        raise ValueError(
            f"Query dimension ({q.shape[0]}) does not match gallery feature dimension ({g.shape[1]})"
        )

    norm_q = float(np.linalg.norm(q))
    if norm_q == 0.0:
        return np.zeros((g.shape[0],), dtype=np.float32)

    norm_g = np.linalg.norm(g, axis=1, keepdims=True)
    # Avoid divide-by-zero for zero-vector entries in gallery
    norm_g = np.where(norm_g == 0.0, 1.0, norm_g)

    # Normalized dot product
    q_normed = q / norm_q
    g_normed = g / norm_g
    sims = np.dot(g_normed, q_normed)
    return np.clip(sims, -1.0, 1.0).astype(np.float32)


class VisionEmbeddingModel(ABC):
    """Abstract base class for vision embedding models.

    Defines the contract for loading weights, extracting embeddings from
    single or batched images, and computing vector similarities.

    Subclasses wrap concrete backends (e.g. CLIP via OpenCLIP, HuggingFace
    Transformers, ONNX Runtime, TensorRT, or mock test doubles).
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the underlying model architecture."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of output feature vectors (e.g. 512, 768)."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model weights are currently loaded into memory."""
        ...

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and preprocessors into memory.

        Subclasses should set ``is_loaded = True`` upon successful completion.
        Should be idempotent if called multiple times.
        """
        ...

    @abstractmethod
    def encode_image(self, image: ImageInput) -> np.ndarray:
        """Extract a normalized 1D embedding vector from a single image.

        Parameters
        ----------
        image : ImageInput
            Input image. Can be an OpenCV BGR/RGB ndarray, PIL Image,
            file path string, or pathlib.Path.

        Returns
        -------
        np.ndarray
            1D float32 numpy array of shape ``(embedding_dim,)``,
            L2-normalized such that ``np.linalg.norm(result) ≈ 1.0``.

        Raises
        ------
        RuntimeError
            If the model has not been loaded via ``load_model()``.
        ValueError
            If the image input is invalid or cannot be processed.
        """
        ...

    def encode_images(self, images: Sequence[ImageInput]) -> np.ndarray:
        """Extract normalized embedding vectors from a sequence of images.

        The default implementation calls ``encode_image`` sequentially.
        Subclasses with native batched inference (e.g. PyTorch DataLoader,
        TensorRT) should override this method for higher throughput.

        Parameters
        ----------
        images : Sequence[ImageInput]
            Sequence of input images.

        Returns
        -------
        np.ndarray
            2D float32 numpy array of shape ``(N, embedding_dim)``,
            where N is the number of input images. Each row is L2-normalized.
        """
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings = [self.encode_image(img) for img in images]
        return np.vstack(embeddings).astype(np.float32)

    def similarity(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two embedding vectors.

        Parameters
        ----------
        embedding_a : np.ndarray
            First embedding vector.
        embedding_b : np.ndarray
            Second embedding vector.

        Returns
        -------
        float
            Cosine similarity in [-1.0, 1.0].
        """
        return cosine_similarity(embedding_a, embedding_b)

    def batch_similarity(
        self,
        query: np.ndarray,
        gallery: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between a query vector and gallery matrix.

        Parameters
        ----------
        query : np.ndarray
            1D query embedding of shape ``(embedding_dim,)``.
        gallery : np.ndarray
            2D gallery embeddings matrix of shape ``(N, embedding_dim)``.

        Returns
        -------
        np.ndarray
            1D float32 array of shape ``(N,)`` with similarities in [-1.0, 1.0].
        """
        return batch_cosine_similarity(query, gallery)
