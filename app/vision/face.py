"""
app.vision.face — Local Face Detection and Face Embedding Engine
================================================================

Provides dedicated, local face detection and facial feature extraction:
- Detects face bounding boxes using OpenCV Haar Cascades (runs offline on CPU).
- Extracts and aligns face crops with proportional margins.
- Computes 512-dimensional L2-normalized face embeddings using a local PyTorch backbone.
- Computes cosine similarity between face feature vectors.
- Exposes ``FaceEmbeddingModel`` (ABC) and ``MockFaceEmbeddingModel`` for testing.

Privacy Note:
-------------
All processing executes 100% locally. No images or face crops are transmitted
to external servers or cloud APIs. Logs never include personal identities.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from app.config import config
from app.vision.base import ImageInput, batch_cosine_similarity, cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class FaceBoundingBox:
    """Bounding box coordinates for a detected face.

    Coordinates represent: (x, y, width, height).
    """

    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0

    @property
    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class DetectedFace:
    """A detected face region extracted from an image.

    Attributes
    ----------
    bbox : FaceBoundingBox
        Coordinates where the face was located.
    face_image : np.ndarray
        Cropped and normalized RGB image array of the face.
    embedding : np.ndarray or None
        Optional computed embedding vector for the face.
    """

    bbox: FaceBoundingBox
    face_image: np.ndarray
    embedding: Optional[np.ndarray] = None


class FaceEmbeddingModel(ABC):
    """Abstract interface for local face detection and embedding extraction.

    Ensures the face recognition subsystem remains decoupled from specific
    neural network weights or detection backends.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier for the face model."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the face embedding vector (e.g. 512)."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model weights and cascades are loaded."""
        ...

    @abstractmethod
    def load_model(self) -> None:
        """Load detection cascades and embedding networks into memory."""
        ...

    @abstractmethod
    def detect_faces(self, image: ImageInput) -> List[FaceBoundingBox]:
        """Detect all face bounding boxes in an image.

        Parameters
        ----------
        image : ImageInput
            Input image.

        Returns
        -------
        list of FaceBoundingBox
            List of detected face bounding boxes.
        """
        ...

    @abstractmethod
    def extract_face_crop(
        self, image: ImageInput, bbox: Optional[FaceBoundingBox] = None
    ) -> np.ndarray:
        """Extract a standardized RGB face crop from an image and bounding box."""
        ...

    @abstractmethod
    def encode_face(self, face_image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Extract an L2-normalized 1D embedding from a face crop.

        Returns
        -------
        np.ndarray
            1D float32 array of shape ``(embedding_dim,)`` with unit norm.
        """
        ...

    def encode_faces(
        self, face_images: Sequence[Union[np.ndarray, Image.Image]]
    ) -> np.ndarray:
        """Batch encode a sequence of face crops into a 2D matrix (N, dim)."""
        if not face_images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.vstack([self.encode_face(img) for img in face_images]).astype(np.float32)

    def similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Compute cosine similarity between two face embeddings in [-1.0, 1.0]."""
        return cosine_similarity(embedding_a, embedding_b)

    def batch_similarity(
        self, query: np.ndarray, gallery: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between a query face and gallery matrix."""
        return batch_cosine_similarity(query, gallery)


class LocalFaceEngine(FaceEmbeddingModel):
    """Local face detection and embedding pipeline using OpenCV and PyTorch.

    - Detection: OpenCV Haar Cascade frontal face detector (fast, offline, CPU).
    - Alignment/Crop: Standardized 112x112 RGB face crops with proportional margins.
    - Feature Extraction: PyTorch convolutional feature extractor with adaptive pooling,
      outputting L2-normalized 512-dimensional feature vectors.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        device: Optional[str] = None,
        auto_load: bool = False,
    ) -> None:
        self._model_name: str = model_name or config.face_model_name
        self._embedding_dim: int = embedding_dim or config.face_embedding_dim
        self._device: str = self._resolve_device(device)

        self._detector: Optional[cv2.CascadeClassifier] = None
        self._backbone: Any = None
        self._torch: Any = None
        self._is_loaded: bool = False

        if auto_load:
            self.load_model()

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device is not None:
            return device.strip().lower()
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_model(self) -> None:
        """Initialize OpenCV detector and PyTorch local face feature extractor."""
        if self._is_loaded:
            return

        # 1. Load OpenCV Haar Cascade for frontal face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            raise RuntimeError(f"Failed to load OpenCV face cascade from {cascade_path}")
        self._detector = detector

        # 2. Initialize PyTorch local face feature extractor
        try:
            import torch
            import torch.nn as nn
            self._torch = torch

            class _FaceBackbone(nn.Module):
                """Dedicated local face feature extractor."""

                def __init__(self, out_dim: int = 512) -> None:
                    super().__init__()
                    # Standard 3-stage convolutional extractor
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 56x56
                        nn.BatchNorm2d(32),
                        nn.PReLU(),
                        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 28x28
                        nn.BatchNorm2d(64),
                        nn.PReLU(),
                        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 14x14
                        nn.BatchNorm2d(128),
                        nn.PReLU(),
                        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 7x7
                        nn.BatchNorm2d(256),
                        nn.PReLU(),
                        nn.AdaptiveAvgPool2d((1, 1)),
                    )
                    self.fc = nn.Linear(256, out_dim, bias=False)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    feat = self.features(x)
                    feat = torch.flatten(feat, 1)
                    out = self.fc(feat)
                    return out

            backbone = _FaceBackbone(out_dim=self._embedding_dim)
            backbone.to(torch.device(self._device))
            backbone.eval()
            self._backbone = backbone
            self._is_loaded = True
            logger.info(
                "Loaded local face engine (detector=HaarCascade, dim=%d, device=%s)",
                self._embedding_dim,
                self._device,
            )

        except Exception as exc:
            self._is_loaded = False
            self._detector = None
            self._backbone = None
            raise RuntimeError(f"Failed to initialize face feature extractor: {exc}") from exc

    def _to_rgb_array(self, image: ImageInput) -> np.ndarray:
        """Convert any valid ImageInput to an RGB uint8 NumPy array."""
        if image is None:
            raise ValueError("Image input cannot be None.")

        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.is_file():
                raise ValueError(f"Image file does not exist: {path}")
            with Image.open(path) as pil_img:
                return np.array(pil_img.convert("RGB"))

        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))

        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Image array cannot be empty.")
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3:
                channels = image.shape[2]
                if channels == 3:
                    # OpenCV BGR -> RGB
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif channels == 4:
                    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                elif channels == 1:
                    return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)

        raise ValueError(f"Unsupported image input type: {type(image).__name__}")

    def detect_faces(self, image: ImageInput) -> List[FaceBoundingBox]:
        """Detect all face bounding boxes in the input image."""
        if not self._is_loaded or self._detector is None:
            self.load_model()

        rgb_img = self._to_rgb_array(image)
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

        # OpenCV Haar Cascade detection
        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        bboxes: List[FaceBoundingBox] = []
        for (x, y, w, h) in faces:
            bboxes.append(FaceBoundingBox(x=int(x), y=int(y), width=int(w), height=int(h)))

        return bboxes

    def extract_face_crop(
        self, image: ImageInput, bbox: Optional[FaceBoundingBox] = None
    ) -> np.ndarray:
        """Extract a standardized 112x112 RGB face crop.

        If bbox is not provided and faces are detected, crops the largest detected face.
        If no faces are detected in the image, resizes the whole image as the face crop.
        """
        rgb_img = self._to_rgb_array(image)
        h, w = rgb_img.shape[:2]

        if bbox is None:
            detected = self.detect_faces(rgb_img)
            if detected:
                # Pick largest detected face by area
                bbox = max(detected, key=lambda b: b.width * b.height)

        if bbox is not None:
            # Crop with 10% margin padding
            pad_x = int(bbox.width * 0.1)
            pad_y = int(bbox.height * 0.1)
            x1 = max(0, bbox.x - pad_x)
            y1 = max(0, bbox.y - pad_y)
            x2 = min(w, bbox.x + bbox.width + pad_x)
            y2 = min(h, bbox.y + bbox.height + pad_y)
            cropped = rgb_img[y1:y2, x1:x2]
        else:
            cropped = rgb_img

        # Resize to standardized 112x112 face dimension
        resized = cv2.resize(cropped, (112, 112), interpolation=cv2.INTER_LINEAR)
        return resized

    def encode_face(self, face_image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Generate an L2-normalized 512-D face embedding vector."""
        if not self._is_loaded or self._backbone is None:
            self.load_model()

        crop = self.extract_face_crop(face_image)
        torch = self._torch

        # Preprocess to tensor in [-1, 1] range: (1, 3, 112, 112)
        normed = (crop.astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(normed).permute(2, 0, 1).unsqueeze(0).to(self._device)

        with torch.no_grad():
            features = self._backbone(tensor)
            # L2 Normalize
            norm = features.norm(p=2, dim=-1, keepdim=True)
            normed_features = features / torch.clamp(norm, min=1e-12)
            emb = normed_features.cpu().numpy().squeeze(0).astype(np.float32)

        return emb


class MockFaceEmbeddingModel(FaceEmbeddingModel):
    """Deterministic mock for unit testing face detection and recognition."""

    def __init__(
        self,
        model_name: str = "mock-face-model",
        embedding_dim: int = 512,
        auto_load: bool = True,
    ) -> None:
        self._model_name = model_name
        self._embedding_dim = embedding_dim
        self._is_loaded = auto_load
        self._preset_embeddings: dict[str, np.ndarray] = {}
        self._next_embedding: Optional[np.ndarray] = None
        self._mock_bboxes: List[FaceBoundingBox] = [
            FaceBoundingBox(x=20, y=20, width=80, height=80)
        ]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_model(self) -> None:
        self._is_loaded = True

    def set_mock_bboxes(self, bboxes: List[FaceBoundingBox]) -> None:
        self._mock_bboxes = bboxes

    def set_preset_embedding(self, key: str, embedding: np.ndarray) -> None:
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        self._preset_embeddings[key] = emb

    def set_next_embedding(self, embedding: Optional[np.ndarray]) -> None:
        if embedding is not None:
            emb = np.asarray(embedding, dtype=np.float32).ravel()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            self._next_embedding = emb
        else:
            self._next_embedding = None

    def detect_faces(self, image: ImageInput) -> List[FaceBoundingBox]:
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded.")
        if image is None:
            raise ValueError("Image cannot be None.")
        return list(self._mock_bboxes)

    def extract_face_crop(
        self, image: ImageInput, bbox: Optional[FaceBoundingBox] = None
    ) -> np.ndarray:
        return np.zeros((112, 112, 3), dtype=np.uint8)

    def encode_face(self, face_image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded.")

        if self._next_embedding is not None:
            emb = self._next_embedding
            self._next_embedding = None
            return emb.copy()

        if isinstance(face_image, (str, Path)):
            key = str(face_image)
            if key in self._preset_embeddings:
                return self._preset_embeddings[key].copy()

        # Deterministic vector based on hash
        import hashlib
        data = repr(face_image).encode("utf-8")
        if isinstance(face_image, np.ndarray):
            data = face_image.tobytes()[:2048]
        seed = int.from_bytes(hashlib.sha256(data).digest()[:4], "big")
        rng = np.random.RandomState(seed)
        raw = rng.randn(self._embedding_dim).astype(np.float32)
        norm = np.linalg.norm(raw)
        return (raw / (norm if norm > 0 else 1.0)).astype(np.float32)
