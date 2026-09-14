"""
app.memory.person_registration — Familiar Person Registration Service
======================================================================

Orchestrates the registration and persistence of familiar people:
- Validates facial reference images.
- Extracts standardized face crops and generates 512-D face embeddings.
- Stores reference face photos locally under ``data/references/person_<id>/``.
- Serializes face embeddings to ``.npy`` files under ``data/embeddings/person_<id>/``.
- Persists person memories, relationships, and visual references in SQLite.

Privacy Rule:
-------------
Private identity details (e.g. person names, relationships) are NEVER written
to system logs or stdout. Only anonymous integer IDs are logged.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from app.config import config
from app.database.models import EntityType, Memory
from app.memory.service import MemoryService
from app.vision.base import ImageInput
from app.vision.face import FaceEmbeddingModel, LocalFaceEngine

logger = logging.getLogger(__name__)


@dataclass
class RegisteredFaceReference:
    """Details of a registered face reference image and its computed face embedding."""

    reference_id: int
    image_path: str
    embedding_path: str
    embedding: np.ndarray


@dataclass
class RegisteredPerson:
    """A registered familiar person memory bundled with all visual references.

    Attributes
    ----------
    memory_id : int
        Unique ID from ``memories`` table.
    name : str
        The person's name (kept in memory, never exposed in logs).
    relationship : str or None
        Relationship or description (e.g. "Daughter", "Grandson").
    narrative : str or None
        Personal story or assistive narrative.
    references : list[RegisteredFaceReference]
        Registered face references and their embeddings.
    created_at : str or None
        ISO timestamp when registered.
    """

    memory_id: int
    name: str
    relationship: Optional[str] = None
    narrative: Optional[str] = None
    references: List[RegisteredFaceReference] = field(default_factory=list)
    created_at: Optional[str] = None

    @property
    def embeddings(self) -> List[np.ndarray]:
        """List of all normalized face embedding vectors for this person."""
        return [ref.embedding for ref in self.references]

    @property
    def reference_image_paths(self) -> List[str]:
        """List of filesystem paths to stored face reference images."""
        return [ref.image_path for ref in self.references]


class PersonRegistrationService:
    """Service governing familiar person registration, face photo storage, and embedding persistence.

    Parameters
    ----------
    memory_service : MemoryService, optional
        Database memory service instance.
    face_engine : FaceEmbeddingModel, optional
        Face detection and embedding engine. Defaults to ``LocalFaceEngine``.
    references_dir : Path or str, optional
        Storage directory for face references. Defaults to ``config.references_dir``.
    embeddings_dir : Path or str, optional
        Storage directory for ``.npy`` face embeddings. Defaults to ``config.embeddings_dir``.
    db_path : Path or str, optional
        Custom SQLite database path if creating a default ``MemoryService``.
    """

    def __init__(
        self,
        memory_service: Optional[MemoryService] = None,
        face_engine: Optional[FaceEmbeddingModel] = None,
        references_dir: Optional[Union[str, Path]] = None,
        embeddings_dir: Optional[Union[str, Path]] = None,
        db_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._memory_service: MemoryService = memory_service or MemoryService(
            db_path=db_path or config.db_path
        )
        self._face_engine: FaceEmbeddingModel = face_engine or LocalFaceEngine()

        self._references_dir: Path = Path(references_dir) if references_dir else config.references_dir
        self._embeddings_dir: Path = Path(embeddings_dir) if embeddings_dir else config.embeddings_dir

        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Create target storage directories if they do not exist."""
        self._references_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Start the underlying database connection."""
        self._memory_service.start()

    def stop(self) -> None:
        """Close the underlying database connection."""
        self._memory_service.stop()

    @property
    def memory_service(self) -> MemoryService:
        return self._memory_service

    @property
    def face_engine(self) -> FaceEmbeddingModel:
        return self._face_engine

    def register_person(
        self,
        name: str,
        relationship: Optional[str] = None,
        description: Optional[str] = None,
        narrative: Optional[str] = None,
        memory_narrative: Optional[str] = None,
        reference_images: Optional[Sequence[ImageInput]] = None,
        require_images: bool = True,
    ) -> RegisteredPerson:
        """Register a familiar person with metadata and reference face images.

        Parameters
        ----------
        name : str
            Full name of the familiar person.
        relationship, description : str, optional
            Relationship to user (e.g. "Daughter", "Grandson", "Caregiver").
        narrative, memory_narrative : str, optional
            Personal narrative memory or assistive prompt.
        reference_images : Sequence[ImageInput], optional
            One or more face images (PIL Images, OpenCV arrays, or file paths).
        require_images : bool, optional
            If True, requires at least one reference face image.

        Returns
        -------
        RegisteredPerson
            Container with created memory details, face references, and embeddings.

        Raises
        ------
        ValueError
            If name is empty or validation of images fails.
        """
        person_name = (name or "").strip()
        if not person_name:
            raise ValueError("Person name cannot be empty.")

        resolved_rel = (relationship or description or "").strip() or None
        resolved_narrative = (narrative or memory_narrative or "").strip() or None

        images = list(reference_images) if reference_images is not None else []
        if require_images and not images:
            raise ValueError("At least one reference face image is required to register a person.")

        # Ensure face model is loaded
        if not self._face_engine.is_loaded:
            self._face_engine.load_model()

        # Validate images
        validated_crops: List[np.ndarray] = []
        for idx, img in enumerate(images):
            if img is None:
                raise ValueError(f"Reference face image at index {idx} cannot be None.")
            try:
                crop = self._face_engine.extract_face_crop(img)
                if crop is None or crop.size == 0:
                    raise ValueError("Empty face crop returned.")
                validated_crops.append(crop)
            except Exception as exc:
                raise ValueError(f"Failed to validate face image at index {idx}: {exc}") from exc

        # Create database person record
        mem_id = self._memory_service.register_person(
            name=person_name,
            giver_name=resolved_rel,
            narrative=resolved_narrative,
        )

        created_files: List[Path] = []
        registered_refs: List[RegisteredFaceReference] = []

        try:
            person_ref_dir = self._references_dir / f"person_{mem_id}"
            person_emb_dir = self._embeddings_dir / f"person_{mem_id}"
            person_ref_dir.mkdir(parents=True, exist_ok=True)
            person_emb_dir.mkdir(parents=True, exist_ok=True)

            for idx, crop in enumerate(validated_crops):
                raw_emb = self._face_engine.encode_face(crop)
                emb = np.asarray(raw_emb, dtype=np.float32).ravel()
                norm = float(np.linalg.norm(emb))
                if norm > 0:
                    emb = emb / norm
                else:
                    emb[0] = 1.0

                uid = uuid.uuid4().hex[:8]
                img_path = person_ref_dir / f"face_{mem_id}_{idx}_{uid}.jpg"
                emb_path = person_emb_dir / f"face_emb_{mem_id}_{idx}_{uid}.npy"

                # Save face crop locally
                Image.fromarray(crop).save(img_path, format="JPEG", quality=95)
                created_files.append(img_path)

                # Save embedding as .npy file
                np.save(emb_path, emb)
                created_files.append(emb_path)

                # Store database relationship
                ref_id = self._memory_service.add_reference_image(
                    memory_id=mem_id,
                    image_path=str(img_path),
                    embedding_path=str(emb_path),
                )

                # Store embedding metadata in database (source_type="face")
                self._memory_service.store_embedding(
                    memory_id=mem_id,
                    model_name=self._face_engine.model_name,
                    embedding_dim=self._face_engine.embedding_dim,
                    embedding_path=str(emb_path),
                    source_type="face",
                )

                registered_refs.append(
                    RegisteredFaceReference(
                        reference_id=ref_id,
                        image_path=str(img_path),
                        embedding_path=str(emb_path),
                        embedding=emb,
                    )
                )

        except Exception as exc:
            # Rollback: Clean up created files and delete DB record
            for f in created_files:
                try:
                    if f.is_file():
                        f.unlink()
                except Exception:
                    pass
            self._memory_service.delete_memory(mem_id)
            logger.error("Failed to register person memory ID %d during disk/DB write: %s", mem_id, exc)
            raise RuntimeError(f"Person registration failed: {exc}") from exc

        # Privacy compliance: never log the person's name or relationship
        logger.info("Successfully registered person memory ID %d with %d face references.", mem_id, len(registered_refs))

        memory_record = self._memory_service.get_memory(mem_id)
        created_at_str = (
            memory_record.created_at.isoformat()
            if memory_record and memory_record.created_at
            else None
        )

        return RegisteredPerson(
            memory_id=mem_id,
            name=person_name,
            relationship=resolved_rel,
            narrative=resolved_narrative,
            references=registered_refs,
            created_at=created_at_str,
        )

    def get_registered_person(self, memory_id: int) -> Optional[RegisteredPerson]:
        """Retrieve a registered person and load face embeddings from disk."""
        memory = self._memory_service.get_memory(memory_id)
        if memory is None or memory.entity_type != EntityType.PERSON:
            return None

        refs = self._memory_service.get_reference_images(memory_id)
        registered_refs: List[RegisteredFaceReference] = []

        for r in refs:
            emb = None
            if r.embedding_path and Path(r.embedding_path).is_file():
                try:
                    emb = np.load(r.embedding_path).astype(np.float32)
                except Exception as exc:
                    logger.warning("Could not load face embedding file '%s': %s", r.embedding_path, exc)

            if emb is None:
                emb = np.zeros(self._face_engine.embedding_dim, dtype=np.float32)

            registered_refs.append(
                RegisteredFaceReference(
                    reference_id=r.id or 0,
                    image_path=r.image_path,
                    embedding_path=r.embedding_path or "",
                    embedding=emb,
                )
            )

        created_at_str = memory.created_at.isoformat() if memory.created_at else None

        return RegisteredPerson(
            memory_id=memory.id or memory_id,
            name=memory.name,
            relationship=memory.giver_name,
            narrative=memory.narrative,
            references=registered_refs,
            created_at=created_at_str,
        )

    def get_all_registered_people(self) -> List[RegisteredPerson]:
        """Retrieve all active registered familiar people with their embeddings loaded."""
        people = self._memory_service.get_all_people()
        results: List[RegisteredPerson] = []
        for p in people:
            if p.id is not None:
                reg_p = self.get_registered_person(p.id)
                if reg_p is not None:
                    results.append(reg_p)
        return results

    def load_all_person_embeddings(self) -> Tuple[List[int], List[np.ndarray]]:
        """Load all registered face embeddings across all active people."""
        people = self.get_all_registered_people()
        ids: List[int] = []
        embs: List[np.ndarray] = []

        for p in people:
            for ref in p.references:
                ids.append(p.memory_id)
                embs.append(ref.embedding)

        return ids, embs

    def delete_registered_person(self, memory_id: int) -> bool:
        """Permanently delete a registered person and remove their face images and embeddings."""
        memory = self._memory_service.get_memory(memory_id)
        if memory is None or memory.entity_type != EntityType.PERSON:
            return False

        person_ref_dir = self._references_dir / f"person_{memory_id}"
        person_emb_dir = self._embeddings_dir / f"person_{memory_id}"

        if person_ref_dir.is_dir():
            shutil.rmtree(person_ref_dir, ignore_errors=True)
        if person_emb_dir.is_dir():
            shutil.rmtree(person_emb_dir, ignore_errors=True)

        return self._memory_service.delete_memory(memory_id)
