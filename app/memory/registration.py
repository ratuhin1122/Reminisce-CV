"""
app.memory.registration — Personal Object Registration Service
==============================================================

Orchestrates the registration and persistence of personal object memories:
1. Validates all reference images before committing changes.
2. Extracts and L2-normalizes CLIP vision embeddings.
3. Securely stores reference images in a private local data directory.
4. Serializes embeddings to local ``.npy`` files.
5. Persists the relationship between the object, images, and embedding metadata in SQLite.
6. Provides retrieval methods to fetch registered objects and load embeddings for recognition.

Privacy & Security:
-------------------
- All image files and embeddings are kept strictly local under ``data/``.
- Never commit private personal images or memory data to version control.
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
from app.database.models import Memory
from app.memory.service import MemoryService
from app.vision.base import ImageInput, VisionEmbeddingModel
from app.vision.clip import CLIPVisionModel

logger = logging.getLogger(__name__)


@dataclass
class RegisteredReference:
    """Details of a registered reference image and its computed embedding.

    Attributes
    ----------
    reference_id : int
        Database ID from ``visual_references`` table.
    image_path : str
        Local filesystem path to the stored reference image.
    embedding_path : str
        Local filesystem path to the saved ``.npy`` embedding file.
    embedding : np.ndarray
        In-memory L2-normalized float32 embedding vector.
    """

    reference_id: int
    image_path: str
    embedding_path: str
    embedding: np.ndarray


@dataclass
class RegisteredObject:
    """A registered personal object memory bundled with all visual references.

    Attributes
    ----------
    memory_id : int
        Unique ID from ``memories`` table.
    name : str
        Primary name or label of the object.
    title : str or None
        Display title or heirloom descriptor.
    giver : str or None
        Person who gave the object.
    occasion : str or None
        Occasion associated with the memory (e.g. "Graduation 2018").
    year : str or None
        Year or date string.
    narrative : str or None
        Personal story or assistive narrative.
    references : list[RegisteredReference]
        Registered visual references and their embeddings.
    created_at : str or None
        ISO timestamp when registered.
    """

    memory_id: int
    name: str
    title: Optional[str] = None
    giver: Optional[str] = None
    occasion: Optional[str] = None
    year: Optional[str] = None
    narrative: Optional[str] = None
    references: List[RegisteredReference] = field(default_factory=list)
    created_at: Optional[str] = None

    @property
    def embeddings(self) -> List[np.ndarray]:
        """List of all normalized embedding vectors for this object."""
        return [ref.embedding for ref in self.references]

    @property
    def reference_image_paths(self) -> List[str]:
        """List of filesystem paths to stored reference images."""
        return [ref.image_path for ref in self.references]


class ObjectRegistrationService:
    """Service governing personal object registration, image storage, and embedding persistence.

    Parameters
    ----------
    memory_service : MemoryService, optional
        Database memory service instance. If omitted, a new one is created.
    vision_model : VisionEmbeddingModel, optional
        Vision model used for embedding extraction. If omitted, uses ``CLIPVisionModel``.
    references_dir : Path or str, optional
        Directory where reference images will be stored. Defaults to ``config.references_dir``.
    embeddings_dir : Path or str, optional
        Directory where ``.npy`` embeddings will be saved. Defaults to ``config.embeddings_dir``.
    db_path : Path or str, optional
        Custom SQLite path if creating a default ``MemoryService``.
    """

    def __init__(
        self,
        memory_service: Optional[MemoryService] = None,
        vision_model: Optional[VisionEmbeddingModel] = None,
        references_dir: Optional[Union[str, Path]] = None,
        embeddings_dir: Optional[Union[str, Path]] = None,
        db_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._memory_service: MemoryService = memory_service or MemoryService(
            db_path=db_path or config.db_path
        )
        self._owns_memory_service: bool = memory_service is None
        self._vision_model: VisionEmbeddingModel = vision_model or CLIPVisionModel()

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
        """Underlying MemoryService."""
        return self._memory_service

    @property
    def vision_model(self) -> VisionEmbeddingModel:
        """Underlying vision embedding model."""
        return self._vision_model

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def validate_image(image: Any) -> Image.Image:
        """Validate an image input and convert it to an RGB PIL Image.

        Parameters
        ----------
        image : Any
            File path (str/Path), PIL Image, or OpenCV NumPy ndarray.

        Returns
        -------
        PIL.Image.Image
            Verified RGB PIL Image with non-zero dimensions.

        Raises
        ------
        ValueError
            If the image input is None, empty, corrupted, or unsupported.
        """
        if image is None:
            raise ValueError("Reference image cannot be None.")

        # 1. Path or filename string
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not str(path).strip():
                raise ValueError("Image path cannot be empty.")
            if not path.is_file():
                raise ValueError(f"Reference image file not found: {path}")

            try:
                # Open and verify file integrity
                with Image.open(path) as img:
                    img.verify()
                # Reopen to load image data
                with Image.open(path) as img:
                    pil_img = img.convert("RGB")
            except Exception as exc:
                raise ValueError(f"Corrupted or unreadable image file '{path}': {exc}") from exc

            if pil_img.width <= 0 or pil_img.height <= 0:
                raise ValueError(f"Invalid image dimensions for '{path}': {pil_img.size}")
            return pil_img

        # 2. PIL Image
        if isinstance(image, Image.Image):
            if image.width <= 0 or image.height <= 0:
                raise ValueError(f"Invalid PIL Image dimensions: {image.size}")
            return image.convert("RGB")

        # 3. NumPy ndarray (OpenCV format)
        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("NumPy image array cannot be empty.")

            if image.ndim == 2:
                pil_img = Image.fromarray(image).convert("RGB")
            elif image.ndim == 3:
                channels = image.shape[2]
                if channels == 3:
                    # OpenCV BGR -> RGB
                    rgb = image[..., ::-1]
                    pil_img = Image.fromarray(rgb, mode="RGB")
                elif channels == 4:
                    rgb = image[..., [2, 1, 0]]
                    pil_img = Image.fromarray(rgb, mode="RGB")
                elif channels == 1:
                    pil_img = Image.fromarray(image[:, :, 0]).convert("RGB")
                else:
                    raise ValueError(f"Unsupported number of image channels: {channels}")
            else:
                raise ValueError(f"Unsupported image array dimensions: {image.ndim}")

            if pil_img.width <= 0 or pil_img.height <= 0:
                raise ValueError(f"Invalid image dimensions: {pil_img.size}")
            return pil_img

        raise ValueError(
            f"Unsupported image type: {type(image).__name__}. Expected Path, str, PIL.Image, or np.ndarray."
        )

    # ── Object Registration ───────────────────────────────────────────────────

    def register_object(
        self,
        name: Optional[str] = None,
        title: Optional[str] = None,
        object_name: Optional[str] = None,
        giver: Optional[str] = None,
        giver_name: Optional[str] = None,
        occasion: Optional[str] = None,
        year: Optional[str] = None,
        date: Optional[str] = None,
        narrative: Optional[str] = None,
        narrative_memory: Optional[str] = None,
        reference_images: Optional[Sequence[ImageInput]] = None,
        require_images: bool = True,
    ) -> RegisteredObject:
        """Register a personal object with metadata and visual reference images.

        Parameters
        ----------
        name, title, object_name : str, optional
            Object name / title identifiers. At least one must be provided.
        giver, giver_name : str, optional
            Name of person who gave the object.
        occasion : str, optional
            Occasion associated with the memory.
        year, date : str, optional
            Year or date of the memory.
        narrative, narrative_memory : str, optional
            Narrative description or personal story.
        reference_images : Sequence[ImageInput], optional
            One or more visual reference images (paths, PIL images, or OpenCV arrays).
        require_images : bool, optional
            If True (default), raises ValueError if reference_images is empty.

        Returns
        -------
        RegisteredObject
            Container with created memory details, references, and normalized embeddings.

        Raises
        ------
        ValueError
            If validation of metadata or any reference image fails.
        """
        # Resolve names and titles
        primary_name = (title or object_name or name or "").strip()
        if not primary_name:
            raise ValueError("Object name or title cannot be empty.")

        resolved_title = title.strip() if title else None
        resolved_giver = (giver or giver_name or "").strip() or None
        resolved_occasion = occasion.strip() if occasion else None
        resolved_year = (year or date or "").strip() or None
        resolved_narrative = (narrative or narrative_memory or "").strip() or None

        # Resolve and validate images
        images = list(reference_images) if reference_images is not None else []
        if require_images and not images:
            raise ValueError("At least one reference image is required to register an object.")

        # 1. Validate all images upfront before writing to database or disk
        validated_pil_images: List[Image.Image] = []
        for idx, img in enumerate(images):
            try:
                validated_pil_images.append(self.validate_image(img))
            except Exception as exc:
                raise ValueError(f"Validation failed for reference image at index {idx}: {exc}") from exc

        # 2. Ensure vision model is loaded
        if not self._vision_model.is_loaded:
            self._vision_model.load_model()

        # 3. Create database memory record
        mem_id = self._memory_service.register_object(
            name=primary_name,
            giver_name=resolved_giver,
            occasion=resolved_occasion,
            year=resolved_year,
            narrative=resolved_narrative,
        )

        created_files: List[Path] = []
        registered_refs: List[RegisteredReference] = []

        try:
            # Create object-specific local folders
            obj_ref_dir = self._references_dir / f"obj_{mem_id}"
            obj_emb_dir = self._embeddings_dir / f"obj_{mem_id}"
            obj_ref_dir.mkdir(parents=True, exist_ok=True)
            obj_emb_dir.mkdir(parents=True, exist_ok=True)

            for idx, pil_img in enumerate(validated_pil_images):
                # Step 2 & 3: Generate and normalize embedding
                raw_emb = self._vision_model.encode_image(pil_img)
                emb = np.asarray(raw_emb, dtype=np.float32).ravel()
                norm = float(np.linalg.norm(emb))
                if norm > 0:
                    emb = emb / norm
                else:
                    emb[0] = 1.0

                uid = uuid.uuid4().hex[:8]
                img_path = obj_ref_dir / f"ref_{mem_id}_{idx}_{uid}.jpg"
                emb_path = obj_emb_dir / f"emb_{mem_id}_{idx}_{uid}.npy"

                # Step 4: Store reference image locally
                pil_img.save(img_path, format="JPEG", quality=95)
                created_files.append(img_path)

                # Step 6: Store embedding safely as .npy file
                np.save(emb_path, emb)
                created_files.append(emb_path)

                # Step 5: Store relationship in database
                ref_id = self._memory_service.add_reference_image(
                    memory_id=mem_id,
                    image_path=str(img_path),
                    embedding_path=str(emb_path),
                )

                # Store embedding metadata in database
                self._memory_service.store_embedding(
                    memory_id=mem_id,
                    model_name=self._vision_model.model_name,
                    embedding_dim=self._vision_model.embedding_dim,
                    embedding_path=str(emb_path),
                    source_type="clip_object",
                )

                registered_refs.append(
                    RegisteredReference(
                        reference_id=ref_id,
                        image_path=str(img_path),
                        embedding_path=str(emb_path),
                        embedding=emb,
                    )
                )

        except Exception as exc:
            # Transaction rollback: delete created files and database record
            logger.error(
                "Failed to complete object registration for memory ID %d: %s. Rolling back.",
                mem_id,
                exc,
            )
            for f in created_files:
                try:
                    if f.is_file():
                        f.unlink()
                except Exception:
                    pass
            self._memory_service.delete_memory(mem_id)
            raise RuntimeError(f"Object registration failed: {exc}") from exc

        memory_record = self._memory_service.get_memory(mem_id)
        created_at_str = (
            memory_record.created_at.isoformat()
            if memory_record and memory_record.created_at
            else None
        )

        return RegisteredObject(
            memory_id=mem_id,
            name=primary_name,
            title=resolved_title,
            giver=resolved_giver,
            occasion=resolved_occasion,
            year=resolved_year,
            narrative=resolved_narrative,
            references=registered_refs,
            created_at=created_at_str,
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_registered_object(self, memory_id: int) -> Optional[RegisteredObject]:
        """Retrieve a registered object memory and load its stored embeddings from disk.

        Parameters
        ----------
        memory_id : int
            The memory ID.

        Returns
        -------
        RegisteredObject or None
            Loaded object memory with embeddings, or None if not found.
        """
        memory = self._memory_service.get_memory(memory_id)
        if memory is None:
            return None

        refs = self._memory_service.get_reference_images(memory_id)
        registered_refs: List[RegisteredReference] = []

        for r in refs:
            emb = None
            if r.embedding_path and Path(r.embedding_path).is_file():
                try:
                    emb = np.load(r.embedding_path).astype(np.float32)
                except Exception as exc:
                    logger.warning("Could not load embedding file '%s': %s", r.embedding_path, exc)

            if emb is None:
                # Fallback to zero vector if embedding file was missing
                emb = np.zeros(self._vision_model.embedding_dim, dtype=np.float32)

            registered_refs.append(
                RegisteredReference(
                    reference_id=r.id or 0,
                    image_path=r.image_path,
                    embedding_path=r.embedding_path or "",
                    embedding=emb,
                )
            )

        created_at_str = memory.created_at.isoformat() if memory.created_at else None

        return RegisteredObject(
            memory_id=memory.id or memory_id,
            name=memory.name,
            giver=memory.giver_name,
            occasion=memory.occasion,
            year=memory.year,
            narrative=memory.narrative,
            references=registered_refs,
            created_at=created_at_str,
        )

    def get_all_registered_objects(self) -> List[RegisteredObject]:
        """Retrieve all active registered objects with their embeddings loaded."""
        objects = self._memory_service.get_all_objects()
        results: List[RegisteredObject] = []
        for obj in objects:
            if obj.id is not None:
                reg_obj = self.get_registered_object(obj.id)
                if reg_obj is not None:
                    results.append(reg_obj)
        return results

    def load_all_object_embeddings(self) -> Tuple[List[int], List[np.ndarray]]:
        """Load all reference embeddings for all active objects into memory.

        Returns
        -------
        tuple of (memory_ids, embeddings)
            - memory_ids : list of int corresponding to the object for each embedding.
            - embeddings : list of 1D normalized float32 numpy arrays.
        """
        objects = self.get_all_registered_objects()
        ids: List[int] = []
        embs: List[np.ndarray] = []

        for obj in objects:
            for ref in obj.references:
                ids.append(obj.memory_id)
                embs.append(ref.embedding)

        return ids, embs

    def delete_registered_object(self, memory_id: int) -> bool:
        """Permanently delete an object memory and delete its stored files on disk.

        Parameters
        ----------
        memory_id : int
            Memory ID to delete.

        Returns
        -------
        bool
            True if deleted, False if object did not exist.
        """
        memory = self._memory_service.get_memory(memory_id)
        if memory is None:
            return False

        # Clean up files on disk
        obj_ref_dir = self._references_dir / f"obj_{memory_id}"
        obj_emb_dir = self._embeddings_dir / f"obj_{memory_id}"

        if obj_ref_dir.is_dir():
            shutil.rmtree(obj_ref_dir, ignore_errors=True)
        if obj_emb_dir.is_dir():
            shutil.rmtree(obj_emb_dir, ignore_errors=True)

        return self._memory_service.delete_memory(memory_id)
