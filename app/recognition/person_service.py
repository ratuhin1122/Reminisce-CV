"""
app.recognition.person_service — Familiar Person Face Recognition Service
=========================================================================

Dedicated recognition pipeline for familiar people:
1. Detects faces in an image or camera frame.
2. Extracts and standardizes face regions.
3. Generates 512-D face embeddings.
4. Compares embeddings against registered people using cosine similarity.
5. Applies a configurable threshold (default: config.face_similarity_threshold).
6. Returns the best matching registered person or "unknown".

Privacy Note:
-------------
Private identity information is NEVER written to logs. Only match status,
similarity scores, and anonymous IDs are recorded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from app.config import config
from app.database.models import Memory
from app.memory.person_registration import (
    PersonRegistrationService,
    RegisteredPerson,
)
from app.memory.service import MemoryService
from app.vision.base import ImageInput, batch_cosine_similarity, cosine_similarity
from app.vision.face import (
    FaceBoundingBox,
    FaceEmbeddingModel,
    LocalFaceEngine,
)

logger = logging.getLogger(__name__)


@dataclass
class PersonRecognitionResult:
    """Structured result of a face recognition query.

    Attributes
    ----------
    matched : bool
        True if the top candidate score meets or exceeds the threshold.
    name : str
        The recognized person's name if matched, or "unknown" if unconfirmed.
    similarity : float
        Cosine similarity score of the top candidate in [-1.0, 1.0].
    entity_id : int or None
        Database ID of the matched person, or None if unknown.
    entity_type : str
        Always "person" for person recognition.
    relationship : str or None
        Relationship description (e.g. "Daughter", "Grandson") if matched.
    memory : Memory or None
        Complete memory record if matched.
    threshold : float
        Similarity threshold applied.
    bbox : FaceBoundingBox or None
        Bounding box of the detected face in the original frame.
    reference_image_path : str or None
        Path to the best matching reference face image.
    candidate_id : int or None
        ID of the top candidate even if below threshold.
    """

    matched: bool
    name: str
    similarity: float
    entity_id: Optional[int] = None
    entity_type: str = "person"
    relationship: Optional[str] = None
    memory: Optional[Memory] = None
    threshold: float = 0.40
    bbox: Optional[FaceBoundingBox] = None
    reference_image_path: Optional[str] = None
    candidate_id: Optional[int] = None


class PersonRecognitionService:
    """Service governing detection and matching of familiar faces.

    Parameters
    ----------
    face_engine : FaceEmbeddingModel, optional
        Face detection and embedding engine. Defaults to ``LocalFaceEngine``.
    registration_service : PersonRegistrationService, optional
        Service providing access to registered people and face embeddings.
    memory_service : MemoryService, optional
        Underlying database memory service.
    threshold : float, optional
        Cosine similarity threshold for confirming identity.
        Defaults to ``config.face_similarity_threshold`` (0.40).
    auto_refresh_gallery : bool, optional
        Whether to reload gallery embeddings on every query.
    """

    def __init__(
        self,
        face_engine: Optional[FaceEmbeddingModel] = None,
        registration_service: Optional[PersonRegistrationService] = None,
        memory_service: Optional[MemoryService] = None,
        threshold: Optional[float] = None,
        auto_refresh_gallery: bool = False,
    ) -> None:
        self._face_engine: FaceEmbeddingModel = face_engine or LocalFaceEngine()
        self._threshold: float = (
            threshold if threshold is not None else config.face_similarity_threshold
        )
        self._auto_refresh_gallery: bool = auto_refresh_gallery

        if registration_service is not None:
            self._reg_service = registration_service
            self._mem_service = registration_service.memory_service
        else:
            self._mem_service = memory_service or MemoryService(db_path=config.db_path)
            self._reg_service = PersonRegistrationService(
                memory_service=self._mem_service,
                face_engine=self._face_engine,
            )

        # In-memory face gallery cache
        self._gallery_embeddings: np.ndarray = np.empty((0, self._face_engine.embedding_dim), dtype=np.float32)
        self._gallery_entity_ids: List[int] = []
        self._gallery_ref_paths: List[str] = []
        self._gallery_loaded: bool = False

    @property
    def face_engine(self) -> FaceEmbeddingModel:
        return self._face_engine

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not (-1.0 <= value <= 1.0):
            raise ValueError(f"Threshold must be between -1.0 and 1.0, got {value}")
        self._threshold = value

    @property
    def gallery_size(self) -> int:
        return len(self._gallery_entity_ids)

    def refresh_gallery(self) -> None:
        """Load all registered face embeddings into an in-memory matrix."""
        try:
            people = self._reg_service.get_all_registered_people()
        except RuntimeError:
            self._mem_service.start()
            people = self._reg_service.get_all_registered_people()
        entity_ids: List[int] = []
        ref_paths: List[str] = []
        embs: List[np.ndarray] = []

        for p in people:
            for ref in p.references:
                entity_ids.append(p.memory_id)
                ref_paths.append(ref.image_path)
                embs.append(ref.embedding)

        if embs:
            self._gallery_embeddings = np.vstack(embs).astype(np.float32)
        else:
            self._gallery_embeddings = np.empty((0, self._face_engine.embedding_dim), dtype=np.float32)

        self._gallery_entity_ids = entity_ids
        self._gallery_ref_paths = ref_paths
        self._gallery_loaded = True

        logger.debug(
            "Refreshed face gallery: %d references for %d people.",
            len(self._gallery_entity_ids),
            len(people),
        )

    def _ensure_gallery(self) -> None:
        if not self._gallery_loaded or self._auto_refresh_gallery:
            self.refresh_gallery()

    def set_in_memory_gallery(
        self,
        embeddings: Sequence[np.ndarray],
        entity_ids: Sequence[int],
        reference_paths: Optional[Sequence[str]] = None,
    ) -> None:
        """Inject in-memory face embeddings for testing without disk/DB access."""
        if len(embeddings) != len(entity_ids):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(entity_ids)} entity IDs")

        if embeddings:
            matrix = np.vstack(embeddings).astype(np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._gallery_embeddings = (matrix / norms).astype(np.float32)
        else:
            self._gallery_embeddings = np.empty((0, self._face_engine.embedding_dim), dtype=np.float32)

        self._gallery_entity_ids = list(entity_ids)
        self._gallery_ref_paths = (
            list(reference_paths) if reference_paths is not None else ["" for _ in entity_ids]
        )
        self._gallery_loaded = True

    def recognize_face(
        self,
        face_image: ImageInput,
        bbox: Optional[FaceBoundingBox] = None,
        threshold: Optional[float] = None,
    ) -> PersonRecognitionResult:
        """Recognize a single face crop against registered familiar people.

        Parameters
        ----------
        face_image : ImageInput
            Face crop or image containing a face.
        bbox : FaceBoundingBox, optional
            Bounding box in the source frame if available.
        threshold : float, optional
            Similarity threshold override.

        Returns
        -------
        PersonRecognitionResult
            Structured result with identity or "unknown".
        """
        eff_threshold = threshold if threshold is not None else self._threshold

        if not self._face_engine.is_loaded:
            self._face_engine.load_model()

        # Extract face crop and 512-D embedding
        crop = self._face_engine.extract_face_crop(face_image, bbox=bbox)
        raw_emb = self._face_engine.encode_face(crop)
        query_emb = np.asarray(raw_emb, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(query_emb))
        if norm > 0:
            query_emb = query_emb / norm
        else:
            query_emb[0] = 1.0

        self._ensure_gallery()

        # Handle empty gallery
        if self._gallery_embeddings.shape[0] == 0:
            return PersonRecognitionResult(
                matched=False,
                name="unknown",
                similarity=0.0,
                entity_id=None,
                entity_type="person",
                relationship=None,
                memory=None,
                threshold=eff_threshold,
                bbox=bbox,
                reference_image_path=None,
                candidate_id=None,
            )

        # Vectorized cosine similarity
        similarities = batch_cosine_similarity(query_emb, self._gallery_embeddings)

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_candidate_id = self._gallery_entity_ids[best_idx]
        best_ref_path = self._gallery_ref_paths[best_idx]

        matched = best_score >= eff_threshold

        if not matched:
            # Privacy rule: never log identity details
            logger.debug("Face unrecognized: top similarity %.4f < threshold %.4f", best_score, eff_threshold)
            return PersonRecognitionResult(
                matched=False,
                name="unknown",
                similarity=best_score,
                entity_id=None,
                entity_type="person",
                relationship=None,
                memory=None,
                threshold=eff_threshold,
                bbox=bbox,
                reference_image_path=best_ref_path,
                candidate_id=best_candidate_id,
            )

        # Retrieve memory details for positive match
        memory = self._mem_service.get_memory(best_candidate_id)
        person_name = memory.name if memory else "unknown"
        relationship = memory.giver_name if memory else None

        # Privacy rule: log anonymous ID only
        logger.debug(
            "Familiar face recognized: memory ID %d (score=%.4f, threshold=%.4f)",
            best_candidate_id,
            best_score,
            eff_threshold,
        )

        return PersonRecognitionResult(
            matched=True,
            name=person_name,
            similarity=best_score,
            entity_id=best_candidate_id,
            entity_type="person",
            relationship=relationship,
            memory=memory,
            threshold=eff_threshold,
            bbox=bbox,
            reference_image_path=best_ref_path,
            candidate_id=best_candidate_id,
        )

    def detect_and_recognize_faces(
        self,
        frame: ImageInput,
        threshold: Optional[float] = None,
    ) -> List[PersonRecognitionResult]:
        """Detect all faces in an image/frame and recognize each one.

        Parameters
        ----------
        frame : ImageInput
            Full image or camera frame.
        threshold : float, optional
            Similarity threshold override.

        Returns
        -------
        list of PersonRecognitionResult
            Recognition results for each face found in the frame.
            Returns empty list if no faces are detected.
        """
        if not self._face_engine.is_loaded:
            self._face_engine.load_model()

        bboxes = self._face_engine.detect_faces(frame)
        if not bboxes:
            return []

        results: List[PersonRecognitionResult] = []
        for bbox in bboxes:
            res = self.recognize_face(frame, bbox=bbox, threshold=threshold)
            results.append(res)

        return results
