"""
app.vision.clip — Pretrained CLIP Vision Embedding Engine
=========================================================

Concrete implementation of ``VisionEmbeddingModel`` using a pretrained CLIP
vision model from Hugging Face Transformers.

Responsibilities:
-----------------
1. Load pretrained CLIP model (e.g. ``openai/clip-vit-base-patch32``) and processor.
2. Accept PIL Images, OpenCV BGR/RGB NumPy arrays, or image file paths.
3. Extract L2-normalized 1D feature vectors (NumPy float32 arrays).
4. Support native batched inference (``encode_images``).
5. Cache the model in memory to avoid repeated loading across video frames.
6. Automatic device selection: detect CUDA if available, fallback to CPU.
7. Graceful error handling for missing models, invalid inputs, or loading failures.
8. Cosine similarity computation:
   .. math::
       \\text{similarity}(A, B) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
from PIL import Image

from app.config import config
from app.vision.base import (
    ImageInput,
    VisionEmbeddingModel,
    batch_cosine_similarity,
    cosine_similarity,
)

logger = logging.getLogger(__name__)


class CLIPVisionModel(VisionEmbeddingModel):
    """Pretrained CLIP vision embedding model wrapping Hugging Face Transformers.

    Parameters
    ----------
    model_name : str, optional
        Hugging Face model identifier or local checkpoint path.
        Defaults to ``config.clip_model_name`` (e.g. ``"openai/clip-vit-base-patch32"``).
    device : str, optional
        Execution device, e.g. ``"cpu"`` or ``"cuda"``.
        If None, automatically selects ``"cuda"`` if a GPU is available, otherwise ``"cpu"``.
    embedding_dim : int, optional
        Expected embedding dimension. Defaults to ``config.clip_embedding_dim`` (512).
    cache_dir : Path or str, optional
        Custom directory to cache downloaded model weights.
    auto_load : bool, optional
        If True, automatically loads weights upon instantiation. Default is False.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        auto_load: bool = False,
    ) -> None:
        self._model_name: str = model_name or config.clip_model_name
        self._embedding_dim: int = embedding_dim or config.clip_embedding_dim
        self._cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else config.models_dir / "clip"

        # Determine target device
        self._device: str = self._resolve_device(device)

        # Model and processor handles (cached in memory)
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._is_loaded: bool = False

        if auto_load:
            self.load_model()

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        """Select CUDA if available and not overridden, otherwise fallback to CPU."""
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
        """Hugging Face model identifier or path."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of output feature vectors."""
        return self._embedding_dim

    @property
    def device(self) -> str:
        """Current execution device ('cpu' or 'cuda')."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Whether the model and processor are loaded into memory."""
        return self._is_loaded

    def load_model(self) -> None:
        """Load the CLIP model and processor into memory.

        Idempotent: if already loaded, returns immediately to avoid reloading.

        Raises
        ------
        RuntimeError
            If dependencies are missing or model download/initialization fails.
        """
        if self._is_loaded and self._model is not None and self._processor is not None:
            logger.debug("CLIP model '%s' is already loaded.", self._model_name)
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            msg = (
                "Deep learning dependencies ('torch', 'transformers') are not installed. "
                "Run: pip install torch torchvision transformers"
            )
            logger.error(msg)
            raise RuntimeError(msg) from exc

        self._torch = torch
        logger.info(
            "Loading CLIP model '%s' on device '%s' (cache_dir=%s)...",
            self._model_name,
            self._device,
            self._cache_dir,
        )

        try:
            cache_str = str(self._cache_dir) if self._cache_dir else None
            self._processor = CLIPProcessor.from_pretrained(
                self._model_name,
                cache_dir=cache_str,
            )
            self._model = CLIPModel.from_pretrained(
                self._model_name,
                cache_dir=cache_str,
            )
            device_obj = torch.device(self._device)
            self._model.to(device_obj)
            self._model.eval()

            # Dynamically confirm actual projection dimension if available
            if hasattr(self._model.config, "projection_dim"):
                self._embedding_dim = self._model.config.projection_dim

            self._is_loaded = True
            logger.info(
                "Successfully loaded CLIP model '%s' (dim=%d, device=%s).",
                self._model_name,
                self._embedding_dim,
                self._device,
            )
        except Exception as exc:
            self._is_loaded = False
            self._model = None
            self._processor = None
            error_msg = f"Failed to load CLIP model '{self._model_name}': {exc}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from exc

    def unload_model(self) -> None:
        """Unload model from memory and clear CUDA cache if applicable."""
        self._model = None
        self._processor = None
        self._is_loaded = False
        if self._torch is not None and self._device.startswith("cuda"):
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass
        logger.info("Unloaded CLIP model '%s'.", self._model_name)

    def _preprocess_image(self, image: ImageInput) -> Image.Image:
        """Convert arbitrary image inputs (PIL, OpenCV numpy array, or path) to RGB PIL Image.

        Parameters
        ----------
        image : ImageInput
            Input image.

        Returns
        -------
        PIL.Image.Image
            RGB PIL Image ready for processor ingestion.

        Raises
        ------
        ValueError
            If input is None, empty, or unreadable.
        """
        if image is None:
            raise ValueError("Image input cannot be None.")

        # 1. PIL Image
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        # 2. Path or str filename
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not str(path).strip():
                raise ValueError("Image file path cannot be empty.")
            if not path.is_file():
                raise ValueError(f"Image file does not exist: {path}")
            try:
                with Image.open(path) as img:
                    return img.convert("RGB")
            except Exception as exc:
                raise ValueError(f"Failed to open image from '{path}': {exc}") from exc

        # 3. NumPy ndarray (OpenCV frame or raw array)
        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("NumPy image array cannot be empty.")

            # OpenCV format is typically BGR with uint8 values
            if image.ndim == 2:
                # Grayscale (H, W) -> RGB
                return Image.fromarray(image).convert("RGB")
            elif image.ndim == 3:
                channels = image.shape[2]
                if channels == 3:
                    # Assume OpenCV standard BGR -> convert to RGB
                    rgb_array = image[..., ::-1]  # fast BGR to RGB channel reverse
                    return Image.fromarray(rgb_array, mode="RGB")
                elif channels == 4:
                    # BGRA -> RGB
                    rgb_array = image[..., [2, 1, 0]]
                    return Image.fromarray(rgb_array, mode="RGB")
                elif channels == 1:
                    return Image.fromarray(image[:, :, 0]).convert("RGB")
                else:
                    raise ValueError(f"Unsupported number of image channels: {channels}")
            else:
                raise ValueError(f"Unsupported image array shape: {image.shape}")

        raise ValueError(
            f"Unsupported image type '{type(image).__name__}'. Expected PIL.Image, np.ndarray, Path, or str."
        )

    def _extract_feature_tensor(self, features: Any) -> "torch.Tensor":
        """Extract the 2D feature tensor (N, D) from raw model output across Transformers versions."""
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            return features.pooler_output
        if hasattr(features, "image_embeds") and features.image_embeds is not None:
            return features.image_embeds
        if isinstance(features, self._torch.Tensor):
            return features
        if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
            return features.last_hidden_state[:, 0]
        return self._torch.as_tensor(features)

    def encode_image(self, image: ImageInput) -> np.ndarray:
        """Extract an L2-normalized 1D embedding vector from a single image.

        Parameters
        ----------
        image : ImageInput
            PIL Image, OpenCV frame (BGR ndarray), or file path.

        Returns
        -------
        np.ndarray
            1D float32 array of shape ``(embedding_dim,)``, L2-normalized such that:
            .. math::
                \\|v\\|_2 = 1.0

        Raises
        ------
        RuntimeError
            If ``load_model()`` has not been called.
        ValueError
            If image is invalid or cannot be processed.
        """
        if not self._is_loaded or self._model is None or self._processor is None:
            raise RuntimeError(
                f"CLIP model '{self._model_name}' is not loaded. Call load_model() first."
            )

        pil_img = self._preprocess_image(image)
        torch = self._torch

        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(torch.device(self._device)) for k, v in inputs.items()}

        with torch.no_grad():
            raw_output = self._model.get_image_features(**inputs)
            features = self._extract_feature_tensor(raw_output)
            # L2 Normalize
            norm = features.norm(p=2, dim=-1, keepdim=True)
            normalized_features = features / torch.clamp(norm, min=1e-12)
            embedding = normalized_features.cpu().numpy().squeeze(0).astype(np.float32)

        return embedding

    def encode_images(self, images: Sequence[ImageInput]) -> np.ndarray:
        """Extract L2-normalized embeddings from a batch of images.

        Uses native batched inference through PyTorch for high throughput.

        Parameters
        ----------
        images : Sequence[ImageInput]
            Sequence of input images.

        Returns
        -------
        np.ndarray
            2D float32 array of shape ``(N, embedding_dim)`` where each row is unit norm.
        """
        if not self._is_loaded or self._model is None or self._processor is None:
            raise RuntimeError(
                f"CLIP model '{self._model_name}' is not loaded. Call load_model() first."
            )

        if not images:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        pil_images = [self._preprocess_image(img) for img in images]
        torch = self._torch

        inputs = self._processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(torch.device(self._device)) for k, v in inputs.items()}

        with torch.no_grad():
            raw_output = self._model.get_image_features(**inputs)
            features = self._extract_feature_tensor(raw_output)
            norm = features.norm(p=2, dim=-1, keepdim=True)
            normalized_features = features / torch.clamp(norm, min=1e-12)
            embeddings = normalized_features.cpu().numpy().astype(np.float32)

        return embeddings

    def similarity(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two embedding vectors.

        Mathematical definition:
        .. math::
            \\text{similarity}(A, B) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}

        Parameters
        ----------
        embedding_a : np.ndarray
            First embedding vector.
        embedding_b : np.ndarray
            Second embedding vector.

        Returns
        -------
        float
            Cosine similarity in range [-1.0, 1.0].
        """
        return cosine_similarity(embedding_a, embedding_b)


# Alias for backward compatibility or alternate naming
CLIPEmbeddingEngine = CLIPVisionModel
