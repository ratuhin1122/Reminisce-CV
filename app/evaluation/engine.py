"""
app.evaluation.engine — Recognition Evaluation Engine
======================================================

Orchestrates the full evaluation lifecycle for ReminisceCV's personal object
recognition pipeline:

1. Accepts a gallery of reference embeddings (known entities).
2. Runs query images through the recognition service at each threshold.
3. Classifies each prediction as TP / FP / FN / TN against ground truth.
4. Computes classification metrics: accuracy, precision, recall, F1-score.
5. Measures performance: embedding inference time, recognition latency, FPS.
6. Sweeps across multiple similarity thresholds for comparative analysis.

All results are computed from actual inference — no values are invented.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.recognition.service import ObjectRecognitionService, RecognitionResult
from app.vision.base import VisionEmbeddingModel, batch_cosine_similarity

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class EvaluationConfig:
    """Configuration for an evaluation run.

    Attributes
    ----------
    thresholds : list of float
        Similarity thresholds to evaluate. Default: [0.50, 0.60, 0.70, 0.80, 0.90].
    num_timing_trials : int
        Number of timing repetitions for performance averaging. Default: 3.
    output_dir : str
        Directory path for saving results. Default: "evaluation/results".
    model_type : str
        Description of the model being evaluated. Default: "mock".
    """

    thresholds: List[float] = field(
        default_factory=lambda: [0.50, 0.60, 0.70, 0.80, 0.90]
    )
    num_timing_trials: int = 3
    output_dir: str = "evaluation/results"
    model_type: str = "mock"


# ── Metrics Dataclasses ───────────────────────────────────────────────────────


@dataclass
class ClassificationMetrics:
    """Classification quality metrics computed from confusion matrix counts.

    All metrics are computed from actual predictions, never invented.
    """

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    @property
    def total(self) -> int:
        """Total number of evaluated samples."""
        return (
            self.true_positives
            + self.false_positives
            + self.false_negatives
            + self.true_negatives
        )

    @property
    def accuracy(self) -> float:
        """Overall classification accuracy."""
        t = self.total
        if t == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / t

    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP)."""
        denom = self.true_positives + self.false_positives
        if denom == 0:
            return 0.0
        return self.true_positives / denom

    @property
    def recall(self) -> float:
        """Recall (sensitivity): TP / (TP + FN)."""
        denom = self.true_positives + self.false_negatives
        if denom == 0:
            return 0.0
        return self.true_positives / denom

    @property
    def f1_score(self) -> float:
        """F1-score: harmonic mean of precision and recall."""
        p = self.precision
        r = self.recall
        if p + r == 0.0:
            return 0.0
        return 2.0 * (p * r) / (p + r)

    def as_dict(self) -> Dict[str, float]:
        """Return all metrics as a flat dictionary."""
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
        }


@dataclass
class PerformanceMetrics:
    """Computational performance measurements.

    All timings are measured from actual execution, never estimated.
    """

    avg_embedding_time_ms: float = 0.0
    min_embedding_time_ms: float = 0.0
    max_embedding_time_ms: float = 0.0
    avg_recognition_latency_ms: float = 0.0
    min_recognition_latency_ms: float = 0.0
    max_recognition_latency_ms: float = 0.0
    throughput_fps: float = 0.0
    total_queries: int = 0

    def as_dict(self) -> Dict[str, float]:
        """Return all metrics as a flat dictionary."""
        return {
            "avg_embedding_time_ms": round(self.avg_embedding_time_ms, 3),
            "min_embedding_time_ms": round(self.min_embedding_time_ms, 3),
            "max_embedding_time_ms": round(self.max_embedding_time_ms, 3),
            "avg_recognition_latency_ms": round(self.avg_recognition_latency_ms, 3),
            "min_recognition_latency_ms": round(self.min_recognition_latency_ms, 3),
            "max_recognition_latency_ms": round(self.max_recognition_latency_ms, 3),
            "throughput_fps": round(self.throughput_fps, 2),
            "total_queries": self.total_queries,
        }


@dataclass
class ThresholdResult:
    """Combined evaluation result for a single similarity threshold.

    Attributes
    ----------
    threshold : float
        The similarity threshold used for this evaluation.
    classification : ClassificationMetrics
        Classification quality metrics.
    performance : PerformanceMetrics
        Computational performance metrics.
    similarity_scores : list of float
        All raw similarity scores observed during evaluation.
    """

    threshold: float
    classification: ClassificationMetrics
    performance: PerformanceMetrics
    similarity_scores: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict:
        """Return complete result as a nested dictionary."""
        return {
            "threshold": self.threshold,
            "classification": self.classification.as_dict(),
            "performance": self.performance.as_dict(),
            "similarity_score_stats": {
                "mean": round(float(np.mean(self.similarity_scores)), 4) if self.similarity_scores else 0.0,
                "std": round(float(np.std(self.similarity_scores)), 4) if self.similarity_scores else 0.0,
                "min": round(float(np.min(self.similarity_scores)), 4) if self.similarity_scores else 0.0,
                "max": round(float(np.max(self.similarity_scores)), 4) if self.similarity_scores else 0.0,
            },
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report across all thresholds.

    Attributes
    ----------
    results : list of ThresholdResult
        Per-threshold evaluation outcomes.
    model_name : str
        Name of the vision model evaluated.
    timestamp : str
        ISO-8601 timestamp of the evaluation run.
    config_summary : dict
        Summary of evaluation configuration parameters.
    dataset_summary : dict
        Summary of the evaluation dataset.
    """

    results: List[ThresholdResult]
    model_name: str = ""
    timestamp: str = ""
    config_summary: Dict = field(default_factory=dict)
    dataset_summary: Dict = field(default_factory=dict)

    def as_dict(self) -> Dict:
        """Return full report as a serializable dictionary."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "config": self.config_summary,
            "dataset": self.dataset_summary,
            "results": [r.as_dict() for r in self.results],
        }

    @property
    def best_f1_result(self) -> Optional[ThresholdResult]:
        """Return the threshold result with the highest F1 score."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.classification.f1_score)


# ── Evaluation Query Dataclass ────────────────────────────────────────────────


@dataclass
class EvaluationQuery:
    """A single evaluation query with ground truth.

    Attributes
    ----------
    image : np.ndarray
        Query image (BGR ndarray).
    ground_truth_entity_id : int or None
        Expected entity ID, or None if this is a distractor (should not match).
    label : str
        Human-readable label for this query.
    """

    image: np.ndarray
    ground_truth_entity_id: Optional[int] = None
    label: str = ""


# ── Core Evaluator ────────────────────────────────────────────────────────────


class RecognitionEvaluator:
    """Evaluator for the personal object recognition pipeline.

    Parameters
    ----------
    recognition_service : ObjectRecognitionService
        The recognition service to evaluate.
    config : EvaluationConfig, optional
        Evaluation configuration. Uses defaults if not provided.
    """

    def __init__(
        self,
        recognition_service: ObjectRecognitionService,
        config: Optional[EvaluationConfig] = None,
    ) -> None:
        self.service = recognition_service
        self.config = config or EvaluationConfig()

    def evaluate_threshold(
        self,
        queries: Sequence[EvaluationQuery],
        threshold: float,
    ) -> ThresholdResult:
        """Evaluate recognition accuracy at a specific similarity threshold.

        Parameters
        ----------
        queries : Sequence[EvaluationQuery]
            List of query images with ground truth labels.
        threshold : float
            Similarity threshold for match classification.

        Returns
        -------
        ThresholdResult
            Classification and performance metrics for this threshold.
        """
        tp = fp = fn = tn = 0
        embedding_times: List[float] = []
        recognition_times: List[float] = []
        similarity_scores: List[float] = []

        for query in queries:
            # Measure full recognition latency
            t_rec_start = time.perf_counter()
            result = self.service.recognize(query.image, threshold=threshold)
            t_rec_end = time.perf_counter()
            rec_ms = (t_rec_end - t_rec_start) * 1000.0
            recognition_times.append(rec_ms)

            # Measure embedding-only time (averaged over timing trials)
            emb_times_trial: List[float] = []
            for _ in range(self.config.num_timing_trials):
                t_emb_start = time.perf_counter()
                self.service.vision_model.encode_image(query.image)
                t_emb_end = time.perf_counter()
                emb_times_trial.append((t_emb_end - t_emb_start) * 1000.0)
            embedding_times.append(float(np.mean(emb_times_trial)))

            similarity_scores.append(result.similarity)

            # Classify prediction against ground truth
            predicted_id = result.entity_id if result.matched else None
            expected_id = query.ground_truth_entity_id

            category = self._classify_prediction(predicted_id, expected_id)
            if category == "TP":
                tp += 1
            elif category == "FP":
                fp += 1
            elif category == "FN":
                fn += 1
            elif category == "TN":
                tn += 1

        # Compute aggregate metrics
        classification = ClassificationMetrics(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
        )

        total_time_sec = sum(recognition_times) / 1000.0 if recognition_times else 0.0

        performance = PerformanceMetrics(
            avg_embedding_time_ms=float(np.mean(embedding_times)) if embedding_times else 0.0,
            min_embedding_time_ms=float(np.min(embedding_times)) if embedding_times else 0.0,
            max_embedding_time_ms=float(np.max(embedding_times)) if embedding_times else 0.0,
            avg_recognition_latency_ms=float(np.mean(recognition_times)) if recognition_times else 0.0,
            min_recognition_latency_ms=float(np.min(recognition_times)) if recognition_times else 0.0,
            max_recognition_latency_ms=float(np.max(recognition_times)) if recognition_times else 0.0,
            throughput_fps=len(queries) / total_time_sec if total_time_sec > 0 else 0.0,
            total_queries=len(queries),
        )

        return ThresholdResult(
            threshold=threshold,
            classification=classification,
            performance=performance,
            similarity_scores=similarity_scores,
        )

    def run_full_evaluation(
        self,
        queries: Sequence[EvaluationQuery],
    ) -> EvaluationReport:
        """Run evaluation across all configured similarity thresholds.

        Parameters
        ----------
        queries : Sequence[EvaluationQuery]
            List of query images with ground truth labels.

        Returns
        -------
        EvaluationReport
            Complete evaluation report with results per threshold.
        """
        logger.info(
            "Starting full evaluation sweep across %d thresholds with %d queries...",
            len(self.config.thresholds),
            len(queries),
        )

        results: List[ThresholdResult] = []
        for threshold in sorted(self.config.thresholds):
            logger.info("  Evaluating threshold=%.2f ...", threshold)
            result = self.evaluate_threshold(queries, threshold)
            results.append(result)
            logger.info(
                "    → Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f  "
                "TP=%d  FP=%d  FN=%d  TN=%d",
                result.classification.accuracy,
                result.classification.precision,
                result.classification.recall,
                result.classification.f1_score,
                result.classification.true_positives,
                result.classification.false_positives,
                result.classification.false_negatives,
                result.classification.true_negatives,
            )

        # Count unique expected entities and distractors
        known_ids = set()
        n_distractors = 0
        for q in queries:
            if q.ground_truth_entity_id is not None:
                known_ids.add(q.ground_truth_entity_id)
            else:
                n_distractors += 1

        report = EvaluationReport(
            results=results,
            model_name=self.service.vision_model.model_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config_summary={
                "thresholds": self.config.thresholds,
                "num_timing_trials": self.config.num_timing_trials,
                "model_type": self.config.model_type,
            },
            dataset_summary={
                "total_queries": len(queries),
                "known_entities": len(known_ids),
                "distractor_queries": n_distractors,
                "known_queries": len(queries) - n_distractors,
            },
        )

        logger.info("Evaluation sweep complete. Best F1=%.4f at threshold=%.2f",
            report.best_f1_result.classification.f1_score if report.best_f1_result else 0.0,
            report.best_f1_result.threshold if report.best_f1_result else 0.0,
        )

        return report

    @staticmethod
    def _classify_prediction(
        predicted_id: Optional[int],
        expected_id: Optional[int],
    ) -> str:
        """Classify a single prediction into TP, FP, FN, or TN.

        Classification logic:
        - **TP**: Expected a match AND predicted the correct entity.
        - **FP**: Expected no match (distractor) BUT predicted a match,
                  OR expected entity A but predicted entity B.
        - **FN**: Expected a match BUT predicted no match.
        - **TN**: Expected no match AND predicted no match.

        Parameters
        ----------
        predicted_id : int or None
            Entity ID returned by the recognition service (None = no match).
        expected_id : int or None
            Ground truth entity ID (None = distractor, should not match).

        Returns
        -------
        str
            One of "TP", "FP", "FN", "TN".
        """
        if expected_id is not None:
            # This is a known entity query
            if predicted_id is not None:
                if predicted_id == expected_id:
                    return "TP"  # Correct match
                else:
                    return "FP"  # Matched wrong entity (misidentification)
            else:
                return "FN"  # Should have matched but didn't
        else:
            # This is a distractor query (should NOT match)
            if predicted_id is not None:
                return "FP"  # Falsely matched a distractor
            else:
                return "TN"  # Correctly rejected distractor
