"""
app.evaluation.test_data — Synthetic Evaluation Dataset Generator
==================================================================

Generates reproducible evaluation datasets using programmatically created
images — no private or personal photos are used or required.

Each synthetic entity receives a distinct visual pattern (color family,
geometric shapes, gradient direction) so that embedding models produce
naturally separable feature vectors.

Query images are variants of reference images with controlled perturbations
(color shifts, noise, rotation) to simulate real-world recognition scenarios.

Distractor images use entirely different color families and patterns to
represent objects not in the gallery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """A single image sample in the evaluation dataset.

    Attributes
    ----------
    image : np.ndarray
        BGR image array (H, W, 3).
    entity_id : int or None
        The entity ID this image belongs to, or None for distractors.
    label : str
        Human-readable description of this sample.
    category : str
        Either "reference", "query", or "distractor".
    """

    image: np.ndarray
    entity_id: Optional[int] = None
    label: str = ""
    category: str = "query"


@dataclass
class EvaluationDataset:
    """Complete evaluation dataset with gallery and query sets.

    Attributes
    ----------
    gallery_samples : list of EvaluationSample
        Reference images for each entity (used to build the gallery).
    query_samples : list of EvaluationSample
        Test images to evaluate against the gallery.
    entity_names : dict
        Mapping from entity_id to human-readable entity name.
    n_entities : int
        Number of distinct entities in the gallery.
    n_distractors : int
        Number of distractor queries (entity_id is None).
    """

    gallery_samples: List[EvaluationSample] = field(default_factory=list)
    query_samples: List[EvaluationSample] = field(default_factory=list)
    entity_names: Dict[int, str] = field(default_factory=dict)
    n_entities: int = 0
    n_distractors: int = 0

    @property
    def total_gallery_images(self) -> int:
        """Total reference images across all entities."""
        return len(self.gallery_samples)

    @property
    def total_query_images(self) -> int:
        """Total query images including distractors."""
        return len(self.query_samples)


# ── Color Palettes ────────────────────────────────────────────────────────────

# Each entity gets a distinct color family (BGR format)
_ENTITY_COLORS: List[Tuple[int, int, int]] = [
    (35, 45, 210),    # Entity 1: Red family
    (185, 140, 30),   # Entity 2: Blue family
    (45, 190, 55),    # Entity 3: Green family
    (25, 185, 235),   # Entity 4: Yellow/Gold family
    (195, 65, 175),   # Entity 5: Purple family
    (210, 155, 35),   # Entity 6: Cyan family
    (40, 110, 230),   # Entity 7: Orange family
    (180, 50, 50),    # Entity 8: Dark blue family
    (65, 200, 200),   # Entity 9: Olive family
    (140, 95, 195),   # Entity 10: Pink family
]

_DISTRACTOR_COLORS: List[Tuple[int, int, int]] = [
    (80, 80, 80),     # Gray
    (20, 20, 20),     # Near-black
    (200, 200, 200),  # Light gray
    (100, 50, 150),   # Muted purple
    (50, 130, 100),   # Muted teal
]

# Shape types for visual diversity
_SHAPE_TYPES = ["circle", "rectangle", "triangle", "diamond", "cross"]


class SyntheticDataGenerator:
    """Generates synthetic evaluation datasets with controlled visual patterns.

    Parameters
    ----------
    image_size : tuple of int
        (width, height) of generated images. Default (224, 224).
    seed : int
        Random seed for reproducibility. Default 42.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        seed: int = 42,
    ) -> None:
        self.image_size = image_size
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    def generate_dataset(
        self,
        n_entities: int = 5,
        n_refs_per_entity: int = 3,
        n_queries_per_entity: int = 5,
        n_distractors: int = 10,
    ) -> EvaluationDataset:
        """Generate a complete evaluation dataset.

        Parameters
        ----------
        n_entities : int
            Number of distinct entity classes. Max 10.
        n_refs_per_entity : int
            Number of reference (gallery) images per entity.
        n_queries_per_entity : int
            Number of query images per known entity.
        n_distractors : int
            Number of distractor query images (should not match any entity).

        Returns
        -------
        EvaluationDataset
            Complete dataset ready for evaluation.
        """
        n_entities = min(n_entities, len(_ENTITY_COLORS))

        gallery_samples: List[EvaluationSample] = []
        query_samples: List[EvaluationSample] = []
        entity_names: Dict[int, str] = {}

        logger.info(
            "Generating synthetic dataset: %d entities × %d refs × %d queries + %d distractors",
            n_entities,
            n_refs_per_entity,
            n_queries_per_entity,
            n_distractors,
        )

        for entity_idx in range(n_entities):
            entity_id = entity_idx + 1
            base_color = _ENTITY_COLORS[entity_idx]
            shape = _SHAPE_TYPES[entity_idx % len(_SHAPE_TYPES)]
            entity_name = f"Object_{entity_id}_{shape}"
            entity_names[entity_id] = entity_name

            # Generate reference images (slight natural variations)
            for ref_idx in range(n_refs_per_entity):
                img = self._generate_entity_image(
                    base_color=base_color,
                    shape=shape,
                    variation_seed=ref_idx * 7 + entity_idx,
                    variation_strength=0.05,
                )
                gallery_samples.append(
                    EvaluationSample(
                        image=img,
                        entity_id=entity_id,
                        label=f"{entity_name}_ref_{ref_idx}",
                        category="reference",
                    )
                )

            # Generate query images (moderate variations to test robustness)
            for q_idx in range(n_queries_per_entity):
                img = self._generate_entity_image(
                    base_color=base_color,
                    shape=shape,
                    variation_seed=q_idx * 13 + entity_idx + 100,
                    variation_strength=0.15,
                )
                query_samples.append(
                    EvaluationSample(
                        image=img,
                        entity_id=entity_id,
                        label=f"{entity_name}_query_{q_idx}",
                        category="query",
                    )
                )

        # Generate distractor images (different color families)
        for d_idx in range(n_distractors):
            color = _DISTRACTOR_COLORS[d_idx % len(_DISTRACTOR_COLORS)]
            shape = _SHAPE_TYPES[(d_idx + 3) % len(_SHAPE_TYPES)]
            img = self._generate_distractor_image(
                base_color=color,
                shape=shape,
                variation_seed=d_idx * 17 + 500,
            )
            query_samples.append(
                EvaluationSample(
                    image=img,
                    entity_id=None,
                    label=f"distractor_{d_idx}",
                    category="distractor",
                )
            )

        dataset = EvaluationDataset(
            gallery_samples=gallery_samples,
            query_samples=query_samples,
            entity_names=entity_names,
            n_entities=n_entities,
            n_distractors=n_distractors,
        )

        logger.info(
            "Dataset generated: %d gallery images, %d query images (%d known + %d distractor)",
            dataset.total_gallery_images,
            dataset.total_query_images,
            dataset.total_query_images - n_distractors,
            n_distractors,
        )

        return dataset

    def _generate_entity_image(
        self,
        base_color: Tuple[int, int, int],
        shape: str,
        variation_seed: int,
        variation_strength: float = 0.1,
    ) -> np.ndarray:
        """Generate a synthetic entity image with a specific visual pattern.

        Parameters
        ----------
        base_color : tuple
            BGR base color for this entity.
        shape : str
            Shape type: "circle", "rectangle", "triangle", "diamond", "cross".
        variation_seed : int
            Seed for variation randomness.
        variation_strength : float
            How much variation to apply (0.0 = identical, 1.0 = very different).

        Returns
        -------
        np.ndarray
            BGR image of shape (height, width, 3).
        """
        w, h = self.image_size
        rng = np.random.RandomState(variation_seed + self.seed)

        # Create gradient background using entity color
        img = self._create_gradient_background(base_color, w, h, rng, variation_strength)

        # Apply color variation
        noise = (rng.randn(h, w, 3) * variation_strength * 25).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw the characteristic shape
        center = (w // 2 + int(rng.randn() * w * 0.05), h // 2 + int(rng.randn() * h * 0.05))
        size = int(min(w, h) * 0.3 * (1.0 + rng.randn() * variation_strength * 0.3))
        size = max(20, size)

        shape_color = tuple(
            int(np.clip(c + rng.randn() * 30 * variation_strength, 0, 255))
            for c in (255 - base_color[0], 255 - base_color[1], 255 - base_color[2])
        )

        self._draw_shape(img, shape, center, size, shape_color)

        # Add subtle texture pattern unique to this entity
        pattern_seed = base_color[0] * 256 + base_color[1]
        self._add_texture_pattern(img, pattern_seed, rng, intensity=0.08)

        return img

    def _generate_distractor_image(
        self,
        base_color: Tuple[int, int, int],
        shape: str,
        variation_seed: int,
    ) -> np.ndarray:
        """Generate a distractor image distinct from all entity patterns.

        Parameters
        ----------
        base_color : tuple
            BGR base color for the distractor.
        shape : str
            Shape type.
        variation_seed : int
            Seed for randomness.

        Returns
        -------
        np.ndarray
            BGR image.
        """
        w, h = self.image_size
        rng = np.random.RandomState(variation_seed + self.seed + 9999)

        # Use a distinctly different gradient direction
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            grad = np.linspace(base_color[c], min(255, base_color[c] + 80), w, dtype=np.uint8)
            img[:, :, c] = np.tile(grad, (h, 1))  # Horizontal gradient (entities use vertical)

        # Heavy noise makes distractors visually distinct
        noise = (rng.randn(h, w, 3) * 35).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Different shape positioning (corner, not center)
        center = (w // 4, h // 4)
        size = int(min(w, h) * 0.2)
        self._draw_shape(img, shape, center, size, (200, 200, 200))

        # Add a second shape for visual distinction
        center2 = (3 * w // 4, 3 * h // 4)
        self._draw_shape(img, "cross", center2, size // 2, (150, 150, 150))

        return img

    def _create_gradient_background(
        self,
        color: Tuple[int, int, int],
        w: int,
        h: int,
        rng: np.random.RandomState,
        variation: float,
    ) -> np.ndarray:
        """Create a vertical gradient background using entity color."""
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            start = max(0, int(color[c] * 0.4 + rng.randn() * 15 * variation))
            end = min(255, int(color[c] * 1.2 + rng.randn() * 15 * variation))
            grad = np.linspace(start, end, h, dtype=np.uint8)
            img[:, :, c] = grad[:, np.newaxis]
        return img

    def _draw_shape(
        self,
        img: np.ndarray,
        shape: str,
        center: Tuple[int, int],
        size: int,
        color: Tuple[int, ...],
    ) -> None:
        """Draw a geometric shape onto an image."""
        cx, cy = center
        bgr = (int(color[0]), int(color[1]), int(color[2]))

        if shape == "circle":
            cv2.circle(img, (cx, cy), size, bgr, -1, cv2.LINE_AA)
        elif shape == "rectangle":
            pt1 = (cx - size, cy - size)
            pt2 = (cx + size, cy + size)
            cv2.rectangle(img, pt1, pt2, bgr, -1)
        elif shape == "triangle":
            pts = np.array([
                [cx, cy - size],
                [cx - size, cy + size],
                [cx + size, cy + size],
            ], dtype=np.int32)
            cv2.fillPoly(img, [pts], bgr, cv2.LINE_AA)
        elif shape == "diamond":
            pts = np.array([
                [cx, cy - size],
                [cx + size, cy],
                [cx, cy + size],
                [cx - size, cy],
            ], dtype=np.int32)
            cv2.fillPoly(img, [pts], bgr, cv2.LINE_AA)
        elif shape == "cross":
            arm_w = max(4, size // 4)
            cv2.rectangle(img, (cx - arm_w, cy - size), (cx + arm_w, cy + size), bgr, -1)
            cv2.rectangle(img, (cx - size, cy - arm_w), (cx + size, cy + arm_w), bgr, -1)

    def _add_texture_pattern(
        self,
        img: np.ndarray,
        pattern_seed: int,
        rng: np.random.RandomState,
        intensity: float = 0.1,
    ) -> None:
        """Add a subtle repeating texture pattern to make entity images distinct."""
        h, w = img.shape[:2]
        pattern_rng = np.random.RandomState(pattern_seed)

        # Create a small tile and repeat it
        tile_size = 16
        tile = (pattern_rng.randn(tile_size, tile_size, 3) * 255 * intensity).astype(np.int16)

        # Tile across the image
        reps_h = (h // tile_size) + 1
        reps_w = (w // tile_size) + 1
        pattern = np.tile(tile, (reps_h, reps_w, 1))[:h, :w, :]

        result = np.clip(img.astype(np.int16) + pattern, 0, 255).astype(np.uint8)
        np.copyto(img, result)
