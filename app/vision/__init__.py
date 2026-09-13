"""
app.vision — Vision Embedding Abstraction & Models
===================================================

This package encapsulates vision model loading, image feature extraction,
and embedding similarity comparisons for ReminisceCV.

All concrete vision encoders (e.g. CLIP, OpenCLIP, future multimodal models)
implement the ``VisionEmbeddingModel`` abstract interface, ensuring the rest
of the application is fully decoupled from deep learning frameworks.

Public API
----------
.. code-block:: python

    from app.vision import (
        VisionEmbeddingModel,
        MockVisionEmbeddingModel,
        ImageInput,
        cosine_similarity,
        batch_cosine_similarity,
    )

    # Instantiate model
    model = MockVisionEmbeddingModel(embedding_dim=512)
    model.load_model()

    # Extract embeddings
    emb_a = model.encode_image("path/to/watch.jpg")
    emb_b = model.encode_image("path/to/gift.jpg")

    # Compute similarity
    score = model.similarity(emb_a, emb_b)
"""

from app.vision.base import (
    ImageInput,
    VisionEmbeddingModel,
    batch_cosine_similarity,
    cosine_similarity,
)
from app.vision.mock import MockVisionEmbeddingModel

__all__ = [
    "ImageInput",
    "MockVisionEmbeddingModel",
    "VisionEmbeddingModel",
    "batch_cosine_similarity",
    "cosine_similarity",
]
