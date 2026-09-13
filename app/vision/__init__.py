"""
app.vision — Vision Embedding Abstraction & Models
===================================================

This package encapsulates vision model loading, image feature extraction,
and embedding similarity comparisons for ReminisceCV.

All concrete vision encoders implement the ``VisionEmbeddingModel`` abstract
interface, ensuring the rest of the application is fully decoupled from
deep learning frameworks.

Public API
----------
.. code-block:: python

    from app.vision import (
        VisionEmbeddingModel,
        CLIPVisionModel,
        CLIPEmbeddingEngine,
        MockVisionEmbeddingModel,
        ImageInput,
        cosine_similarity,
        batch_cosine_similarity,
    )

    # Instantiate CLIP model
    model = CLIPVisionModel()
    model.load_model()

    # Extract normalized embeddings (NumPy ndarray)
    emb_a = model.encode_image("path/to/watch.jpg")
    emb_b = model.encode_image("path/to/gift.jpg")

    # Compute cosine similarity: (A · B) / (||A|| ||B||)
    score = model.similarity(emb_a, emb_b)
"""

from app.vision.base import (
    ImageInput,
    VisionEmbeddingModel,
    batch_cosine_similarity,
    cosine_similarity,
)
from app.vision.clip import CLIPEmbeddingEngine, CLIPVisionModel
from app.vision.mock import MockVisionEmbeddingModel

__all__ = [
    "CLIPEmbeddingEngine",
    "CLIPVisionModel",
    "ImageInput",
    "MockVisionEmbeddingModel",
    "VisionEmbeddingModel",
    "batch_cosine_similarity",
    "cosine_similarity",
]
