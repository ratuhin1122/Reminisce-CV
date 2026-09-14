"""
app.recognition.service — Personal Object Recognition Service
=============================================================

Executes embedding-based recognition of personal objects using vision embeddings:
1. Receives an image, crop, or camera frame.
2. Extracts its normalized vision embedding vector.
3. Compares the query embedding against all registered reference embeddings.
4. Computes cosine similarities in batch.
5. Identifies the highest-scoring candidate object.
6. Compares the score against a configurable similarity threshold:
   .. math::
       \\text{match} \\iff \\text{similarity}(A, B) \\ge \\theta
7. Returns a structured ``RecognitionResult``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from app.config import config
from app.database.models import Memory
from app.memory.registration import ObjectRegistrationService, RegisteredObject
from app.memory.service import MemoryService
from app.vision.base import (
    ImageInput,
    VisionEmbeddingModel,
    batch_cosine_similarity,
    cosine_similarity,
)
from app.vision.clip import CLIPVisionModel

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    """Structured outcome of an object recognition query.

    Attributes
    ----------
    matched : bool
        True if the best candidate's similarity >= threshold.
    similarity : float
        Cosine similarity score of the top candidate in [-1.0, 1.0].
        Returns 0.0 if no gallery items are registered.
    entity_id : int or None
        Database memory ID of the recognized object, or None if no match.
    entity_type : str
        Always "object" for object recognition.
    name : str or None
        Display name of the recognized object, or None if no match.
    memory : Memory or None
        Complete memory record from the database if matched.
    threshold : float
        The similarity threshold used for classification.
    reference_image_path : str or None
        Path to the reference image that yielded the highest score.
    candidate_id : int or None
        The best candidate ID even if similarity did not reach the threshold.
    """

    matched: bool
    similarity: float
    entity_id: Optional[int] = None
    entity_type: str = "object"
    name: Optional[str] = None
    memory: Optional[Memory] = None
    threshold: float = 0.25
    reference_image_path: Optional[str] = None
    candidate_id: Optional[int] = None


class ObjectRecognitionService:
    """Service that matches query image crops against registered personal objects.

    Parameters
    ----------
    vision_model : VisionEmbeddingModel, optional
        Model used to extract embeddings. Defaults to ``CLIPVisionModel``.
    registration_service : ObjectRegistrationService, optional
        Service providing access to registered objects and reference embeddings.
    memory_service : MemoryService, optional
        Database service. Used if registration_service is not provided.
    threshold : float, optional
        Cosine similarity threshold for a positive match.
        Defaults to ``config.object_similarity_threshold``.
    auto_refresh_gallery : bool, optional
        If True, reloads gallery embeddings on every call. Default False.
    """

    def __init__(
        self,
        vision_model: Optional[VisionEmbeddingModel] = None,
        registration_service: Optional[ObjectRegistrationService] = None,
        memory_service: Optional[MemoryService] = None,
        threshold: Optional[float] = None,
        auto_refresh_gallery: bool = False,
    ) -> None:
        self._vision_model: VisionEmbeddingModel = vision_model or CLIPVisionModel()
        self._threshold: float = (
            threshold if threshold is not None else config.object_similarity_threshold
        )
        self._auto_refresh_gallery: bool = auto_refresh_gallery

        if registration_service is not None:
            self._reg_service = registration_service
            self._mem_service = registration_service.memory_service
        else:
            self._mem_service = memory_service or MemoryService()
            self._reg_service = ObjectRegistrationService(
                memory_service=self._mem_service,
                vision_model=self._vision_model,
            )

        # In-memory gallery cache:
        # gallery_embeddings: 2D array of shape (N, D)
        # gallery_entity_ids: list of int of length N
        # gallery_ref_paths: list of str of length N
        self._gallery_embeddings: np.ndarray = np.empty((0, self._vision_model.embedding_dim), dtype=np.float32)
        self._gallery_entity_ids: List[int] = []
        self._gallery_ref_paths: List[str] = []
        self._gallery_loaded: bool = False

    @property
    def vision_model(self) -> VisionEmbeddingModel:
        """Active vision embedding model."""
        return self._vision_model

    @property
    def threshold(self) -> float:
        """Active similarity threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not (-1.0 <= value <= 1.0):
            raise ValueError(f"Threshold must be between -1.0 and 1.0, got {value}")
        self._threshold = value

    @property
    def gallery_size(self) -> int:
        """Number of reference embeddings in the active gallery."""
        return len(self._gallery_entity_ids)

    def refresh_gallery(self) -> None:
        """Load all registered object embeddings from disk into an in-memory matrix."""
        objects = self._reg_service.get_all_registered_objects()
        entity_ids: List[int] = []
        ref_paths: List[str] = []
        embs: List[np.ndarray] = []

        for obj in objects:
            for ref in obj.references:
                entity_ids.append(obj.memory_id)
                ref_paths.append(ref.image_path)
                embs.append(ref.embedding)

        if embs:
            self._gallery_embeddings = np.vstack(embs).astype(np.float32)
        else:
            self._gallery_embeddings = np.empty(
                (0, self._vision_model.embedding_dim), dtype=np.float32
            )

        self._gallery_entity_ids = entity_ids
        self._gallery_ref_paths = ref_paths
        self._gallery_loaded = True

        logger.debug(
            "Refreshed object recognition gallery: %d reference embeddings for %d objects.",
            len(self._gallery_entity_ids),
            len(objects),
        )

    def _ensure_gallery(self) -> None:
        """Ensure the gallery is populated if not yet loaded or auto_refresh is enabled."""
        if not self._gallery_loaded or self._auto_refresh_gallery:
            self.refresh_gallery()

    def set_in_memory_gallery(
        self,
        embeddings: Sequence[np.ndarray],
        entity_ids: Sequence[int],
        reference_paths: Optional[Sequence[str]] = None,
    ) -> None:
        """Directly inject in-memory embeddings for testing without database or disk access.

        Parameters
        ----------
        embeddings : Sequence[np.ndarray]
            List of 1D normalized embedding vectors.
        entity_ids : Sequence[int]
            Matching memory IDs for each embedding vector.
        reference_paths : Sequence[str], optional
            Optional mock file paths for each reference.
        """
        if len(embeddings) != len(entity_ids):
            raise ValueError(
                f"Length mismatch: {len(embeddings)} embeddings vs {len(entity_ids)} entity IDs"
            )

        if embeddings:
            # Stack and normalize
            matrix = np.vstack(embeddings).astype(np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._gallery_embeddings = (matrix / norms).astype(np.float32)
        else:
            self._gallery_embeddings = np.empty((0, self._vision_model.embedding_dim), dtype=np.float32)

        self._gallery_entity_ids = list(entity_ids)
        self._gallery_ref_paths = (
            list(reference_paths) if reference_paths is not None else ["" for _ in entity_ids]
        )
        self._gallery_loaded = True

    def recognize(
        self,
        image: ImageInput,
        threshold: Optional[float] = None,
    ) -> RecognitionResult:
        """Recognize a personal object from an image or crop.

        Parameters
        ----------
        image : ImageInput
            PIL Image, OpenCV frame (BGR array), or file path.
        threshold : float, optional
            Override the default similarity threshold for this query.

        Returns
        -------
        RecognitionResult
            Structured result containing matched status, similarity score,
            and object metadata if matched.
        """
        eff_threshold = threshold if threshold is not None else self._threshold

        # 1. Ensure vision model is loaded
        if not self._vision_model.is_loaded:
            self._vision_model.load_model()

        # 2. Extract and normalize query embedding
        raw_emb = self._vision_model.encode_image(image)
        query_emb = np.asarray(raw_emb, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(query_emb))
        if norm > 0:
            query_emb = query_emb / norm
        else:
            query_emb[0] = 1.0

        # 3. Ensure gallery is populated
        self._ensure_gallery()

        # 4. Check if gallery is empty
        if self._gallery_embeddings.shape[0] == 0:
            return RecognitionResult(
                matched=False,
                similarity=0.0,
                entity_id=None,
                entity_type="object",
                name=None,
                memory=None,
                threshold=eff_threshold,
                reference_image_path=None,
                candidate_id=None,
            )

        # 5. Compute cosine similarities against all gallery items
        similarities = batch_cosine_similarity(query_emb, self._gallery_embeddings)

        # 6. Select highest-scoring candidate
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_candidate_id = self._gallery_entity_ids[best_idx]
        best_ref_path = self._gallery_ref_paths[best_idx]

        # 7. Apply similarity threshold
        matched = best_score >= eff_threshold

        if not matched:
            return RecognitionResult(
                matched=False,
                similarity=best_score,
                entity_id=None,
                entity_type="object",
                name=None,
                memory=None,
                threshold=eff_threshold,
                reference_image_path=best_ref_path,
                candidate_id=best_candidate_id,
            )

        # 8. Fetch memory details from database
        memory = self._mem_service.get_memory(best_candidate_id)
        name = memory.name if memory else None

        return RecognitionResult(
            matched=True,
            similarity=best_score,
            entity_id=best_candidate_id,
            entity_type="object",
            name=name,
            memory=memory,
            threshold=eff_threshold,
            reference_image_path=best_ref_path,
            candidate_id=best_candidate_id,
        )

    def recognize_top_k(
        self,
        image: ImageInput,
        k: int = 3,
        threshold: Optional[float] = None,
    ) -> List[RecognitionResult]:
        """Return top-K candidate matches for a query image.

        Parameters
        ----------
        image : ImageInput
            Input image or crop.
        k : int, optional
            Number of top candidates to return. Default is 3.
        threshold : float, optional
            Threshold override.

        Returns
        -------
        list of RecognitionResult
            Ranked list of top candidate results from highest to lowest similarity.
        """
        eff_threshold = threshold if threshold is not None else self._threshold

        if not self._vision_model.is_loaded:
            self._vision_model.load_model()

        raw_emb = self._vision_model.encode_image(image)
        query_emb = np.asarray(raw_emb, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(query_emb))
        if norm > 0:
            query_emb = query_emb / norm

        self._ensure_gallery()
        if self._gallery_embeddings.shape[0] == 0:
            return []

        similarities = batch_cosine_similarity(query_emb, self._gallery_embeddings)

        # Sort descending
        top_indices = np.argsort(similarities)[::-1][:k]

        results: List[RecognitionResult] = []
        for idx in top_indices:
            score = float(similarities[idx])
            cand_id = self._gallery_entity_ids[idx]
            ref_path = self._gallery_ref_paths[idx]
            matched = score >= eff_threshold
            memory = self._mem_service.get_memory(cand_id) if matched else None
            name = memory.name if memory else None

            results.append(
                RecognitionResult(
                    matched=matched,
                    similarity=score,
                    entity_id=cand_id if matched else None,
                    entity_type="object",
                    name=name,
                    memory=memory,
                    threshold=eff_threshold,
                    reference_image_path=ref_path,
                    candidate_id=cand_id,
                )
            )

        return results
