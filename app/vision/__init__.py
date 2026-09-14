"""
app.vision — Vision Embedding Abstraction & Models
===================================================

This package encapsulates vision model loading, image feature extraction,
and embedding similarity comparisons for ReminisceCV.

Modules:
--------
- ``base``: VisionEmbeddingModel ABC and similarity functions.
- ``clip``: Pretrained CLIP vision embedding engine.
- ``face``: Local face detection and face embedding pipeline.
- ``mock``: Test doubles for offline verification.
"""

from app.vision.base import (
    ImageInput,
    VisionEmbeddingModel,
    batch_cosine_similarity,
    cosine_similarity,
)
from app.vision.camera import (
    CameraError,
    CameraFrame,
    CameraNotFoundError,
    CameraService,
    FrameCaptureError,
    WebcamService,
)
from app.vision.clip import CLIPEmbeddingEngine, CLIPVisionModel
from app.vision.face import (
    DetectedFace,
    FaceBoundingBox,
    FaceEmbeddingModel,
    LocalFaceEngine,
    MockFaceEmbeddingModel,
)
from app.vision.mock import MockVisionEmbeddingModel

__all__ = [
    "CLIPEmbeddingEngine",
    "CLIPVisionModel",
    "CameraError",
    "CameraFrame",
    "CameraNotFoundError",
    "CameraService",
    "DetectedFace",
    "FaceBoundingBox",
    "FaceEmbeddingModel",
    "FrameCaptureError",
    "ImageInput",
    "LocalFaceEngine",
    "MockFaceEmbeddingModel",
    "MockVisionEmbeddingModel",
    "VisionEmbeddingModel",
    "WebcamService",
    "batch_cosine_similarity",
    "cosine_similarity",
]

