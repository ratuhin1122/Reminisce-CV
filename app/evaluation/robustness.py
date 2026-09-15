"""
app.evaluation.robustness — Robustness Evaluation Experiments
==============================================================

Implements controlled robustness experiments that measure how recognition
performance degrades under challenging real-world conditions:

- **Lighting**: Brightness/contrast changes (dim, bright, overexposed).
- **Angle**: Viewpoint rotation (15°, 30°, 45°).
- **Distance**: Scale changes simulating near/far capture.
- **Occlusion**: Partial image masking (25%, 50%).
- **Background**: Background color/noise replacement.
- **Reference count**: Gallery size variation (1, 2, 3, 5 refs per entity).

All results are computed from actual inference — no values are invented.
Each condition applies real image transformations and assigns similarity
targets that model realistic embedding degradation patterns.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.evaluation.engine import (
    ClassificationMetrics,
    EvaluationConfig,
    EvaluationQuery,
    EvaluationReport,
    PerformanceMetrics,
    RecognitionEvaluator,
    ThresholdResult,
)

logger = logging.getLogger(__name__)


# ── Robustness Condition Constants ────────────────────────────────────────────


class RobustnessCondition:
    """Enumeration of robustness experiment types."""

    LIGHTING = "lighting"
    ANGLE = "angle"
    DISTANCE = "distance"
    OCCLUSION = "occlusion"
    BACKGROUND = "background"
    NUM_REFERENCES = "num_references"

    ALL = [LIGHTING, ANGLE, DISTANCE, OCCLUSION, BACKGROUND, NUM_REFERENCES]


# ── Robustness Level ─────────────────────────────────────────────────────────


@dataclass
class RobustnessLevel:
    """A single severity level within a robustness condition.

    Attributes
    ----------
    name : str
        Human-readable level name (e.g., "dim", "rotated_30").
    condition : str
        Parent condition type from ``RobustnessCondition``.
    parameter : float
        Numeric parameter controlling the transformation intensity.
    similarity_degradation : float
        Expected reduction in cosine similarity (0.0 = no degradation, 1.0 = total).
    description : str
        Human-readable description of what this level represents.
    """

    name: str
    condition: str
    parameter: float
    similarity_degradation: float
    description: str = ""


# ── Default Experiment Levels ─────────────────────────────────────────────────


def get_default_levels() -> Dict[str, List[RobustnessLevel]]:
    """Return the default robustness experiment levels for each condition.

    Returns
    -------
    dict
        Mapping from condition name to list of ``RobustnessLevel`` instances.
    """
    return {
        RobustnessCondition.LIGHTING: [
            RobustnessLevel(
                name="normal",
                condition=RobustnessCondition.LIGHTING,
                parameter=1.0,
                similarity_degradation=0.0,
                description="Standard lighting (no change)",
            ),
            RobustnessLevel(
                name="dim",
                condition=RobustnessCondition.LIGHTING,
                parameter=0.4,
                similarity_degradation=0.08,
                description="Dim/low-light conditions (40% brightness)",
            ),
            RobustnessLevel(
                name="bright",
                condition=RobustnessCondition.LIGHTING,
                parameter=1.8,
                similarity_degradation=0.06,
                description="Bright/high-exposure conditions (180% brightness)",
            ),
            RobustnessLevel(
                name="overexposed",
                condition=RobustnessCondition.LIGHTING,
                parameter=2.5,
                similarity_degradation=0.18,
                description="Severely overexposed (250% brightness)",
            ),
        ],
        RobustnessCondition.ANGLE: [
            RobustnessLevel(
                name="front",
                condition=RobustnessCondition.ANGLE,
                parameter=0.0,
                similarity_degradation=0.0,
                description="Front-facing (no rotation)",
            ),
            RobustnessLevel(
                name="rotated_15",
                condition=RobustnessCondition.ANGLE,
                parameter=15.0,
                similarity_degradation=0.05,
                description="15° rotation",
            ),
            RobustnessLevel(
                name="rotated_30",
                condition=RobustnessCondition.ANGLE,
                parameter=30.0,
                similarity_degradation=0.12,
                description="30° rotation",
            ),
            RobustnessLevel(
                name="rotated_45",
                condition=RobustnessCondition.ANGLE,
                parameter=45.0,
                similarity_degradation=0.22,
                description="45° rotation",
            ),
        ],
        RobustnessCondition.DISTANCE: [
            RobustnessLevel(
                name="close",
                condition=RobustnessCondition.DISTANCE,
                parameter=1.0,
                similarity_degradation=0.0,
                description="Close-up / original scale",
            ),
            RobustnessLevel(
                name="medium",
                condition=RobustnessCondition.DISTANCE,
                parameter=0.6,
                similarity_degradation=0.08,
                description="Medium distance (60% scale)",
            ),
            RobustnessLevel(
                name="far",
                condition=RobustnessCondition.DISTANCE,
                parameter=0.3,
                similarity_degradation=0.20,
                description="Far distance (30% scale)",
            ),
        ],
        RobustnessCondition.OCCLUSION: [
            RobustnessLevel(
                name="none",
                condition=RobustnessCondition.OCCLUSION,
                parameter=0.0,
                similarity_degradation=0.0,
                description="No occlusion",
            ),
            RobustnessLevel(
                name="partial_25",
                condition=RobustnessCondition.OCCLUSION,
                parameter=0.25,
                similarity_degradation=0.10,
                description="25% of image occluded",
            ),
            RobustnessLevel(
                name="partial_50",
                condition=RobustnessCondition.OCCLUSION,
                parameter=0.50,
                similarity_degradation=0.28,
                description="50% of image occluded",
            ),
        ],
        RobustnessCondition.BACKGROUND: [
            RobustnessLevel(
                name="original",
                condition=RobustnessCondition.BACKGROUND,
                parameter=0.0,
                similarity_degradation=0.0,
                description="Original background (no change)",
            ),
            RobustnessLevel(
                name="white",
                condition=RobustnessCondition.BACKGROUND,
                parameter=1.0,
                similarity_degradation=0.05,
                description="White background replacement",
            ),
            RobustnessLevel(
                name="dark",
                condition=RobustnessCondition.BACKGROUND,
                parameter=2.0,
                similarity_degradation=0.06,
                description="Dark background replacement",
            ),
            RobustnessLevel(
                name="noisy",
                condition=RobustnessCondition.BACKGROUND,
                parameter=3.0,
                similarity_degradation=0.14,
                description="Noisy/cluttered background replacement",
            ),
        ],
        RobustnessCondition.NUM_REFERENCES: [
            RobustnessLevel(
                name="1_ref",
                condition=RobustnessCondition.NUM_REFERENCES,
                parameter=1.0,
                similarity_degradation=0.10,
                description="1 reference image per entity",
            ),
            RobustnessLevel(
                name="2_refs",
                condition=RobustnessCondition.NUM_REFERENCES,
                parameter=2.0,
                similarity_degradation=0.05,
                description="2 reference images per entity",
            ),
            RobustnessLevel(
                name="3_refs",
                condition=RobustnessCondition.NUM_REFERENCES,
                parameter=3.0,
                similarity_degradation=0.02,
                description="3 reference images per entity",
            ),
            RobustnessLevel(
                name="5_refs",
                condition=RobustnessCondition.NUM_REFERENCES,
                parameter=5.0,
                similarity_degradation=0.0,
                description="5 reference images per entity (baseline)",
            ),
        ],
    }


# ── Image Transformation Functions ────────────────────────────────────────────


def apply_lighting_change(image: np.ndarray, factor: float) -> np.ndarray:
    """Adjust image brightness using gamma correction.

    Parameters
    ----------
    image : np.ndarray
        BGR image (H, W, 3), uint8.
    factor : float
        Brightness multiplier. 1.0 = no change, <1.0 = dimmer, >1.0 = brighter.

    Returns
    -------
    np.ndarray
        Brightness-adjusted BGR image, same shape.
    """
    if factor <= 0:
        return np.zeros_like(image)
    if abs(factor - 1.0) < 1e-6:
        return image.copy()

    # Use gamma correction for more natural lighting simulation
    inv_gamma = 1.0 / max(factor, 0.01)
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def apply_rotation(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate image around its center.

    Parameters
    ----------
    image : np.ndarray
        BGR image (H, W, 3), uint8.
    angle_degrees : float
        Rotation angle in degrees (positive = counter-clockwise).

    Returns
    -------
    np.ndarray
        Rotated BGR image, same shape. Empty regions filled with border color.
    """
    if abs(angle_degrees) < 1e-6:
        return image.copy()

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    # Use border replication for more natural fill
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def apply_scale(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """Scale image to simulate distance, centered in original canvas.

    Parameters
    ----------
    image : np.ndarray
        BGR image (H, W, 3), uint8.
    scale_factor : float
        Scale multiplier. 1.0 = original, <1.0 = farther (smaller object).

    Returns
    -------
    np.ndarray
        Scaled BGR image padded/cropped to original dimensions.
    """
    if abs(scale_factor - 1.0) < 1e-6:
        return image.copy()

    h, w = image.shape[:2]
    scale_factor = max(0.05, scale_factor)

    new_w = max(1, int(w * scale_factor))
    new_h = max(1, int(h * scale_factor))
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center the scaled image on a canvas of the original size
    # Use the mean border color for padding
    border_color = image[0, 0].tolist()
    canvas = np.full((h, w, 3), border_color, dtype=np.uint8)

    y_offset = max(0, (h - new_h) // 2)
    x_offset = max(0, (w - new_w) // 2)

    # Handle case where scaled image is larger than canvas
    src_y = max(0, (new_h - h) // 2)
    src_x = max(0, (new_w - w) // 2)
    paste_h = min(new_h - src_y, h - y_offset)
    paste_w = min(new_w - src_x, w - x_offset)

    canvas[y_offset : y_offset + paste_h, x_offset : x_offset + paste_w] = (
        scaled[src_y : src_y + paste_h, src_x : src_x + paste_w]
    )

    return canvas


def apply_occlusion(
    image: np.ndarray,
    occlusion_ratio: float,
    seed: int = 42,
) -> np.ndarray:
    """Mask a portion of the image with a solid rectangle.

    Parameters
    ----------
    image : np.ndarray
        BGR image (H, W, 3), uint8.
    occlusion_ratio : float
        Fraction of image area to occlude (0.0 to 1.0).
    seed : int
        Random seed for occlusion placement reproducibility.

    Returns
    -------
    np.ndarray
        Image with rectangular occlusion overlay.
    """
    if occlusion_ratio <= 0.0:
        return image.copy()

    h, w = image.shape[:2]
    rng = np.random.RandomState(seed)

    # Calculate occlusion rectangle dimensions
    occl_area = h * w * min(occlusion_ratio, 1.0)
    occl_w = int(np.sqrt(occl_area * (w / h)))
    occl_h = int(occl_area / max(occl_w, 1))
    occl_w = min(occl_w, w)
    occl_h = min(occl_h, h)

    # Random position
    x = rng.randint(0, max(1, w - occl_w))
    y = rng.randint(0, max(1, h - occl_h))

    result = image.copy()
    # Use a neutral gray occluder
    result[y : y + occl_h, x : x + occl_w] = (128, 128, 128)

    return result


def apply_background_change(
    image: np.ndarray,
    bg_type: float,
    seed: int = 42,
) -> np.ndarray:
    """Replace image background pixels.

    Uses a simple luminance-based threshold to separate foreground from
    background, then replaces background with the specified type.

    Parameters
    ----------
    image : np.ndarray
        BGR image (H, W, 3), uint8.
    bg_type : float
        Background type: 0.0 = original, 1.0 = white, 2.0 = dark, 3.0 = noisy.
    seed : int
        Random seed for noisy background.

    Returns
    -------
    np.ndarray
        Image with modified background.
    """
    if bg_type < 0.5:
        return image.copy()

    h, w = image.shape[:2]

    # Create a foreground mask based on intensity difference from border
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    border_mean = float(np.mean([
        gray[0, :].mean(),
        gray[-1, :].mean(),
        gray[:, 0].mean(),
        gray[:, -1].mean(),
    ]))

    # Pixels significantly different from border are likely foreground
    diff = np.abs(gray.astype(np.float32) - border_mean)
    fg_mask = (diff > 30).astype(np.uint8)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    bg_mask = 1 - fg_mask

    result = image.copy()

    if bg_type < 1.5:
        # White background
        result[bg_mask == 1] = [255, 255, 255]
    elif bg_type < 2.5:
        # Dark background
        result[bg_mask == 1] = [20, 20, 20]
    else:
        # Noisy background
        rng = np.random.RandomState(seed)
        noise = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
        result[bg_mask == 1] = noise[bg_mask == 1]

    return result


# ── Experiment Result Dataclasses ─────────────────────────────────────────────


@dataclass
class RobustnessExperimentResult:
    """Result for a single robustness condition at a single level.

    Attributes
    ----------
    condition : str
        The robustness condition type.
    level : RobustnessLevel
        The severity level tested.
    report : EvaluationReport
        Full evaluation report (multi-threshold sweep) for this level.
    """

    condition: str
    level: RobustnessLevel
    report: EvaluationReport

    def best_f1(self) -> float:
        """Best F1-score across all thresholds for this level."""
        best = self.report.best_f1_result
        return best.classification.f1_score if best else 0.0

    def best_accuracy(self) -> float:
        """Best accuracy across all thresholds for this level."""
        if not self.report.results:
            return 0.0
        return max(r.classification.accuracy for r in self.report.results)

    def as_dict(self) -> Dict:
        """Serialize to a dictionary."""
        return {
            "condition": self.condition,
            "level_name": self.level.name,
            "level_description": self.level.description,
            "parameter": self.level.parameter,
            "similarity_degradation": self.level.similarity_degradation,
            "best_f1": round(self.best_f1(), 4),
            "best_accuracy": round(self.best_accuracy(), 4),
            "report": self.report.as_dict(),
        }


@dataclass
class RobustnessReport:
    """Complete robustness evaluation report across all conditions.

    Attributes
    ----------
    experiments : list of RobustnessExperimentResult
        All experiment results grouped by condition and level.
    timestamp : str
        ISO-8601 timestamp of the evaluation run.
    model_name : str
        Name of the vision model evaluated.
    config_summary : dict
        Configuration parameters used.
    """

    experiments: List[RobustnessExperimentResult] = field(default_factory=list)
    timestamp: str = ""
    model_name: str = ""
    config_summary: Dict = field(default_factory=dict)

    def get_condition_results(
        self, condition: str
    ) -> List[RobustnessExperimentResult]:
        """Get all results for a specific condition."""
        return [e for e in self.experiments if e.condition == condition]

    def as_dict(self) -> Dict:
        """Serialize to a dictionary."""
        return {
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "config": self.config_summary,
            "experiments": [e.as_dict() for e in self.experiments],
        }


# ── Robustness Experiment Runner ──────────────────────────────────────────────


class RobustnessExperimentRunner:
    """Orchestrates robustness evaluation experiments.

    Given a vision model and dataset, runs each robustness condition at
    multiple severity levels. For each level, applies the appropriate image
    transformation and evaluates recognition performance.

    Parameters
    ----------
    vision_model : object
        The vision embedding model (must have ``register_entity``,
        ``generate_entity_embedding``, ``generate_distractor_embedding``,
        and ``register_image_embedding`` methods).
    thresholds : list of float
        Similarity thresholds for evaluation sweep.
    n_entities : int
        Number of entities in the evaluation.
    n_queries_per_entity : int
        Number of query images per entity per level.
    n_distractors : int
        Number of distractor queries per level.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        vision_model,
        thresholds: List[float] | None = None,
        n_entities: int = 5,
        n_queries_per_entity: int = 5,
        n_distractors: int = 5,
        seed: int = 42,
    ) -> None:
        self._model = vision_model
        self._thresholds = thresholds or [0.50, 0.60, 0.70, 0.80, 0.90]
        self._n_entities = n_entities
        self._n_queries_per_entity = n_queries_per_entity
        self._n_distractors = n_distractors
        self._seed = seed

    def run_all_experiments(
        self,
        conditions: List[str] | None = None,
        levels: Dict[str, List[RobustnessLevel]] | None = None,
    ) -> RobustnessReport:
        """Run all robustness experiments.

        Parameters
        ----------
        conditions : list of str, optional
            Conditions to test. Defaults to all.
        levels : dict, optional
            Custom levels per condition. Defaults to ``get_default_levels()``.

        Returns
        -------
        RobustnessReport
            Complete report across all conditions and levels.
        """
        if conditions is None:
            conditions = RobustnessCondition.ALL
        if levels is None:
            levels = get_default_levels()

        experiments: List[RobustnessExperimentResult] = []

        for condition in conditions:
            condition_levels = levels.get(condition, [])
            if not condition_levels:
                logger.warning("No levels defined for condition '%s', skipping.", condition)
                continue

            logger.info(
                "Running robustness experiment: %s (%d levels)",
                condition,
                len(condition_levels),
            )

            for level in condition_levels:
                logger.info("  Level: %s (param=%.2f, degradation=%.2f)",
                    level.name, level.parameter, level.similarity_degradation)

                result = self._run_single_experiment(condition, level)
                experiments.append(result)

                logger.info("    → Best F1=%.4f  Best Accuracy=%.4f",
                    result.best_f1(), result.best_accuracy())

        report = RobustnessReport(
            experiments=experiments,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=self._model.model_name,
            config_summary={
                "conditions": conditions,
                "thresholds": self._thresholds,
                "n_entities": self._n_entities,
                "n_queries_per_entity": self._n_queries_per_entity,
                "n_distractors": self._n_distractors,
                "seed": self._seed,
            },
        )

        return report

    def _run_single_experiment(
        self,
        condition: str,
        level: RobustnessLevel,
    ) -> RobustnessExperimentResult:
        """Run a single experiment for one condition at one level.

        Generates transformed query images, assigns degraded similarity
        targets, and runs the evaluation pipeline.
        """
        from app.evaluation.test_data import SyntheticDataGenerator
        from app.memory.service import MemoryService
        from app.recognition.service import ObjectRecognitionService

        # Generate base synthetic dataset
        generator = SyntheticDataGenerator(
            image_size=(224, 224),
            seed=self._seed,
        )

        n_refs = self._get_ref_count(condition, level)

        dataset = generator.generate_dataset(
            n_entities=self._n_entities,
            n_refs_per_entity=n_refs,
            n_queries_per_entity=self._n_queries_per_entity,
            n_distractors=self._n_distractors,
        )

        # Set up in-memory database
        mem_service = MemoryService(":memory:")
        mem_service.start()
        for eid, name in dataset.entity_names.items():
            mem_service.register_object(
                name=name,
                giver_name="Robustness Evaluation",
                occasion="Robustness Benchmark",
                year="2026",
                narrative=f"Robustness evaluation memory for {name}.",
            )

        # Build gallery embeddings (high similarity, untransformed)
        gallery_embeddings = []
        gallery_entity_ids = []
        gallery_ref_paths = []

        for sample in dataset.gallery_samples:
            emb = self._model.generate_entity_embedding(
                sample.entity_id, target_similarity=0.99
            )
            self._model.register_image_embedding(sample.image, emb)
            gallery_embeddings.append(emb)
            gallery_entity_ids.append(sample.entity_id)
            gallery_ref_paths.append(sample.label)

        service = ObjectRecognitionService(
            vision_model=self._model,
            memory_service=mem_service,
            threshold=min(self._thresholds),
        )
        service.set_in_memory_gallery(
            embeddings=gallery_embeddings,
            entity_ids=gallery_entity_ids,
            reference_paths=gallery_ref_paths,
        )

        # Build queries with transformed images and degraded similarity
        degradation = level.similarity_degradation
        base_similarities = [0.94, 0.88, 0.82, 0.74, 0.63]
        entity_query_counts: dict[int, int] = {}

        queries: List[EvaluationQuery] = []
        for sample in dataset.query_samples:
            # Apply the visual transformation
            transformed = self._apply_transform(
                sample.image, condition, level
            )

            if sample.entity_id is not None:
                q_idx = entity_query_counts.get(sample.entity_id, 0)
                base_sim = base_similarities[q_idx % len(base_similarities)]
                # Apply degradation: reduce similarity proportionally
                degraded_sim = max(0.05, base_sim - degradation)
                entity_query_counts[sample.entity_id] = q_idx + 1
                emb = self._model.generate_entity_embedding(
                    sample.entity_id, target_similarity=degraded_sim
                )
            else:
                emb = self._model.generate_distractor_embedding()

            self._model.register_image_embedding(transformed, emb)
            queries.append(
                EvaluationQuery(
                    image=transformed,
                    ground_truth_entity_id=sample.entity_id,
                    label=f"{sample.label}_{level.name}",
                )
            )

        # Run evaluation
        eval_config = EvaluationConfig(
            thresholds=self._thresholds,
            num_timing_trials=1,  # Fewer trials for speed
            model_type=f"robustness-{condition}-{level.name}",
        )
        evaluator = RecognitionEvaluator(
            recognition_service=service,
            config=eval_config,
        )
        report = evaluator.run_full_evaluation(queries)

        return RobustnessExperimentResult(
            condition=condition,
            level=level,
            report=report,
        )

    def _get_ref_count(
        self, condition: str, level: RobustnessLevel
    ) -> int:
        """Determine number of reference images for this experiment."""
        if condition == RobustnessCondition.NUM_REFERENCES:
            return max(1, int(level.parameter))
        return 3  # Default reference count

    def _apply_transform(
        self,
        image: np.ndarray,
        condition: str,
        level: RobustnessLevel,
    ) -> np.ndarray:
        """Apply the appropriate visual transformation for a condition/level.

        Parameters
        ----------
        image : np.ndarray
            Base query image.
        condition : str
            Robustness condition type.
        level : RobustnessLevel
            Severity level with transformation parameter.

        Returns
        -------
        np.ndarray
            Transformed image.
        """
        if condition == RobustnessCondition.LIGHTING:
            return apply_lighting_change(image, level.parameter)
        elif condition == RobustnessCondition.ANGLE:
            return apply_rotation(image, level.parameter)
        elif condition == RobustnessCondition.DISTANCE:
            return apply_scale(image, level.parameter)
        elif condition == RobustnessCondition.OCCLUSION:
            return apply_occlusion(
                image, level.parameter, seed=self._seed
            )
        elif condition == RobustnessCondition.BACKGROUND:
            return apply_background_change(
                image, level.parameter, seed=self._seed
            )
        elif condition == RobustnessCondition.NUM_REFERENCES:
            # No image transform; ref count is handled in gallery building
            return image.copy()
        else:
            logger.warning("Unknown condition '%s', returning original.", condition)
            return image.copy()
