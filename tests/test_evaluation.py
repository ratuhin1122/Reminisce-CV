"""
tests/test_evaluation.py — Evaluation Framework Tests
======================================================

Verifies the AI recognition evaluation framework:

- Synthetic dataset generation correctness
- ClassificationMetrics computation from known counts
- RecognitionEvaluator produces valid results with mock model
- Multi-threshold sweep returns results for each threshold
- CSV and JSON export produce valid, parseable files
- Performance timing values are non-negative
"""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.evaluation.engine import (
    ClassificationMetrics,
    EvaluationConfig,
    EvaluationQuery,
    EvaluationReport,
    PerformanceMetrics,
    RecognitionEvaluator,
    ThresholdResult,
)
from app.evaluation.export import export_csv, export_json, export_summary_text
from app.evaluation.test_data import (
    EvaluationDataset,
    EvaluationSample,
    SyntheticDataGenerator,
)
from app.recognition.service import ObjectRecognitionService
from app.vision.mock import MockVisionEmbeddingModel


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_model():
    """Create a mock vision model in loaded state."""
    return MockVisionEmbeddingModel(embedding_dim=512, auto_load=True)


@pytest.fixture
def synthetic_dataset():
    """Generate a small synthetic evaluation dataset."""
    gen = SyntheticDataGenerator(image_size=(64, 64), seed=42)
    return gen.generate_dataset(
        n_entities=3,
        n_refs_per_entity=2,
        n_queries_per_entity=3,
        n_distractors=4,
    )


@pytest.fixture
def evaluation_service(mock_model, synthetic_dataset):
    """Build a recognition service with gallery from synthetic dataset."""
    service = ObjectRecognitionService(
        vision_model=mock_model,
        threshold=0.5,
    )

    embeddings = []
    entity_ids = []
    ref_paths = []

    for sample in synthetic_dataset.gallery_samples:
        emb = mock_model.encode_image(sample.image)
        embeddings.append(emb)
        entity_ids.append(sample.entity_id)
        ref_paths.append(sample.label)

    service.set_in_memory_gallery(
        embeddings=embeddings,
        entity_ids=entity_ids,
        reference_paths=ref_paths,
    )

    return service


@pytest.fixture
def evaluation_queries(synthetic_dataset):
    """Build evaluation queries from synthetic dataset."""
    return [
        EvaluationQuery(
            image=sample.image,
            ground_truth_entity_id=sample.entity_id,
            label=sample.label,
        )
        for sample in synthetic_dataset.query_samples
    ]


# ── Synthetic Data Generator Tests ────────────────────────────────────────────


class TestSyntheticDataGenerator:
    def test_generates_correct_entity_count(self, synthetic_dataset):
        assert synthetic_dataset.n_entities == 3

    def test_generates_correct_gallery_count(self, synthetic_dataset):
        # 3 entities × 2 refs = 6 gallery images
        assert synthetic_dataset.total_gallery_images == 6

    def test_generates_correct_query_count(self, synthetic_dataset):
        # 3 entities × 3 queries + 4 distractors = 13 query images
        assert synthetic_dataset.total_query_images == 13

    def test_generates_correct_distractor_count(self, synthetic_dataset):
        assert synthetic_dataset.n_distractors == 4

    def test_gallery_samples_have_entity_ids(self, synthetic_dataset):
        for sample in synthetic_dataset.gallery_samples:
            assert sample.entity_id is not None
            assert sample.entity_id >= 1
            assert sample.category == "reference"

    def test_query_samples_have_correct_categories(self, synthetic_dataset):
        known = [s for s in synthetic_dataset.query_samples if s.category == "query"]
        distractors = [s for s in synthetic_dataset.query_samples if s.category == "distractor"]
        assert len(known) == 9  # 3 × 3
        assert len(distractors) == 4

    def test_distractor_samples_have_no_entity_id(self, synthetic_dataset):
        distractors = [s for s in synthetic_dataset.query_samples if s.category == "distractor"]
        for sample in distractors:
            assert sample.entity_id is None

    def test_images_have_correct_shape(self, synthetic_dataset):
        for sample in synthetic_dataset.gallery_samples:
            assert sample.image.shape == (64, 64, 3)
            assert sample.image.dtype == np.uint8

    def test_entity_names_populated(self, synthetic_dataset):
        assert len(synthetic_dataset.entity_names) == 3
        for eid, name in synthetic_dataset.entity_names.items():
            assert isinstance(name, str)
            assert len(name) > 0

    def test_reproducible_with_same_seed(self):
        gen1 = SyntheticDataGenerator(image_size=(32, 32), seed=123)
        gen2 = SyntheticDataGenerator(image_size=(32, 32), seed=123)
        ds1 = gen1.generate_dataset(n_entities=2, n_refs_per_entity=1, n_queries_per_entity=1, n_distractors=1)
        ds2 = gen2.generate_dataset(n_entities=2, n_refs_per_entity=1, n_queries_per_entity=1, n_distractors=1)
        for s1, s2 in zip(ds1.gallery_samples, ds2.gallery_samples):
            np.testing.assert_array_equal(s1.image, s2.image)


# ── Classification Metrics Tests ──────────────────────────────────────────────


class TestClassificationMetrics:
    def test_perfect_classification(self):
        m = ClassificationMetrics(true_positives=10, false_positives=0, false_negatives=0, true_negatives=5)
        assert m.accuracy == 1.0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1_score == 1.0

    def test_zero_predictions(self):
        m = ClassificationMetrics(true_positives=0, false_positives=0, false_negatives=10, true_negatives=5)
        assert m.accuracy == pytest.approx(5 / 15)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1_score == 0.0

    def test_all_false_positives(self):
        m = ClassificationMetrics(true_positives=0, false_positives=10, false_negatives=0, true_negatives=0)
        assert m.accuracy == 0.0
        assert m.precision == 0.0
        assert m.recall == 0.0

    def test_mixed_results(self):
        m = ClassificationMetrics(true_positives=8, false_positives=2, false_negatives=3, true_negatives=7)
        assert m.total == 20
        assert m.accuracy == pytest.approx(15 / 20)
        assert m.precision == pytest.approx(8 / 10)
        assert m.recall == pytest.approx(8 / 11)
        expected_f1 = 2.0 * (0.8 * (8 / 11)) / (0.8 + (8 / 11))
        assert m.f1_score == pytest.approx(expected_f1, abs=1e-4)

    def test_empty_metrics(self):
        m = ClassificationMetrics()
        assert m.total == 0
        assert m.accuracy == 0.0
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1_score == 0.0

    def test_as_dict_keys(self):
        m = ClassificationMetrics(true_positives=5, false_positives=1, false_negatives=2, true_negatives=3)
        d = m.as_dict()
        assert "accuracy" in d
        assert "precision" in d
        assert "recall" in d
        assert "f1_score" in d
        assert "true_positives" in d
        assert d["true_positives"] == 5


# ── Performance Metrics Tests ─────────────────────────────────────────────────


class TestPerformanceMetrics:
    def test_default_values(self):
        p = PerformanceMetrics()
        assert p.avg_embedding_time_ms == 0.0
        assert p.throughput_fps == 0.0
        assert p.total_queries == 0

    def test_as_dict(self):
        p = PerformanceMetrics(
            avg_embedding_time_ms=1.5,
            throughput_fps=120.0,
            total_queries=100,
        )
        d = p.as_dict()
        assert d["avg_embedding_time_ms"] == 1.5
        assert d["throughput_fps"] == 120.0
        assert d["total_queries"] == 100


# ── Evaluator Prediction Classification Tests ────────────────────────────────


class TestPredictionClassification:
    def test_true_positive(self):
        assert RecognitionEvaluator._classify_prediction(1, 1) == "TP"

    def test_false_positive_wrong_entity(self):
        assert RecognitionEvaluator._classify_prediction(2, 1) == "FP"

    def test_false_positive_distractor_matched(self):
        assert RecognitionEvaluator._classify_prediction(1, None) == "FP"

    def test_false_negative(self):
        assert RecognitionEvaluator._classify_prediction(None, 1) == "FN"

    def test_true_negative(self):
        assert RecognitionEvaluator._classify_prediction(None, None) == "TN"


# ── RecognitionEvaluator Tests ────────────────────────────────────────────────


class TestRecognitionEvaluator:
    def test_evaluate_single_threshold(self, evaluation_service, evaluation_queries):
        evaluator = RecognitionEvaluator(
            recognition_service=evaluation_service,
            config=EvaluationConfig(thresholds=[0.50], num_timing_trials=1),
        )
        result = evaluator.evaluate_threshold(evaluation_queries, threshold=0.50)
        assert isinstance(result, ThresholdResult)
        assert result.threshold == 0.50
        assert result.classification.total == len(evaluation_queries)
        assert result.performance.total_queries == len(evaluation_queries)
        assert len(result.similarity_scores) == len(evaluation_queries)

    def test_evaluate_produces_nonnegative_timings(self, evaluation_service, evaluation_queries):
        evaluator = RecognitionEvaluator(
            recognition_service=evaluation_service,
            config=EvaluationConfig(thresholds=[0.60], num_timing_trials=1),
        )
        result = evaluator.evaluate_threshold(evaluation_queries, threshold=0.60)
        assert result.performance.avg_embedding_time_ms >= 0.0
        assert result.performance.min_embedding_time_ms >= 0.0
        assert result.performance.max_embedding_time_ms >= 0.0
        assert result.performance.avg_recognition_latency_ms >= 0.0
        assert result.performance.throughput_fps >= 0.0

    def test_full_evaluation_returns_all_thresholds(self, evaluation_service, evaluation_queries):
        thresholds = [0.50, 0.70, 0.90]
        evaluator = RecognitionEvaluator(
            recognition_service=evaluation_service,
            config=EvaluationConfig(thresholds=thresholds, num_timing_trials=1),
        )
        report = evaluator.run_full_evaluation(evaluation_queries)
        assert isinstance(report, EvaluationReport)
        assert len(report.results) == 3
        result_thresholds = [r.threshold for r in report.results]
        assert result_thresholds == sorted(thresholds)

    def test_report_metadata(self, evaluation_service, evaluation_queries):
        evaluator = RecognitionEvaluator(
            recognition_service=evaluation_service,
            config=EvaluationConfig(thresholds=[0.50], num_timing_trials=1),
        )
        report = evaluator.run_full_evaluation(evaluation_queries)
        assert report.model_name == "mock-clip-vit-b32"
        assert len(report.timestamp) > 0
        assert "total_queries" in report.dataset_summary

    def test_best_f1_result(self, evaluation_service, evaluation_queries):
        evaluator = RecognitionEvaluator(
            recognition_service=evaluation_service,
            config=EvaluationConfig(thresholds=[0.50, 0.70, 0.90], num_timing_trials=1),
        )
        report = evaluator.run_full_evaluation(evaluation_queries)
        best = report.best_f1_result
        assert best is not None
        assert isinstance(best.classification.f1_score, float)

    def test_confusion_matrix_sums_to_total(self, evaluation_service, evaluation_queries):
        evaluator = RecognitionEvaluator(
            recognition_service=evaluation_service,
            config=EvaluationConfig(thresholds=[0.60], num_timing_trials=1),
        )
        result = evaluator.evaluate_threshold(evaluation_queries, threshold=0.60)
        c = result.classification
        assert c.total == len(evaluation_queries)
        assert c.true_positives + c.false_positives + c.false_negatives + c.true_negatives == len(evaluation_queries)

    def test_higher_threshold_fewer_matches(self, evaluation_service, evaluation_queries):
        """Higher thresholds should generally produce fewer (or equal) positive matches."""
        evaluator = RecognitionEvaluator(
            recognition_service=evaluation_service,
            config=EvaluationConfig(thresholds=[0.30, 0.95], num_timing_trials=1),
        )
        low = evaluator.evaluate_threshold(evaluation_queries, threshold=0.30)
        high = evaluator.evaluate_threshold(evaluation_queries, threshold=0.95)
        low_positives = low.classification.true_positives + low.classification.false_positives
        high_positives = high.classification.true_positives + high.classification.false_positives
        assert high_positives <= low_positives


# ── Export Tests ──────────────────────────────────────────────────────────────


class TestExportCSV:
    def _make_report(self) -> EvaluationReport:
        return EvaluationReport(
            results=[
                ThresholdResult(
                    threshold=0.50,
                    classification=ClassificationMetrics(
                        true_positives=8, false_positives=2, false_negatives=1, true_negatives=4
                    ),
                    performance=PerformanceMetrics(
                        avg_embedding_time_ms=1.2, throughput_fps=100.0, total_queries=15
                    ),
                    similarity_scores=[0.55, 0.72, 0.88, 0.31],
                ),
                ThresholdResult(
                    threshold=0.70,
                    classification=ClassificationMetrics(
                        true_positives=6, false_positives=0, false_negatives=3, true_negatives=6
                    ),
                    performance=PerformanceMetrics(
                        avg_embedding_time_ms=1.1, throughput_fps=105.0, total_queries=15
                    ),
                    similarity_scores=[0.75, 0.82, 0.90, 0.45],
                ),
            ],
            model_name="test-model",
            timestamp="2026-01-01T00:00:00",
            config_summary={"thresholds": [0.50, 0.70]},
            dataset_summary={"total_queries": 15},
        )

    def test_csv_file_created(self):
        report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_csv(report, Path(tmpdir) / "test.csv")
            assert path.exists()
            assert path.suffix == ".csv"

    def test_csv_has_correct_rows(self):
        report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_csv(report, Path(tmpdir) / "test.csv")
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2
            assert float(rows[0]["threshold"]) == 0.50
            assert float(rows[1]["threshold"]) == 0.70

    def test_csv_has_required_columns(self):
        report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_csv(report, Path(tmpdir) / "test.csv")
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
            for col in ["threshold", "accuracy", "precision", "recall", "f1_score",
                        "true_positives", "false_positives", "avg_embedding_time_ms"]:
                assert col in fieldnames


class TestExportJSON:
    def _make_report(self) -> EvaluationReport:
        return EvaluationReport(
            results=[
                ThresholdResult(
                    threshold=0.60,
                    classification=ClassificationMetrics(true_positives=5, true_negatives=3),
                    performance=PerformanceMetrics(avg_embedding_time_ms=0.8),
                    similarity_scores=[0.65, 0.71],
                ),
            ],
            model_name="json-test-model",
            timestamp="2026-01-01T00:00:00",
            config_summary={},
            dataset_summary={"total_queries": 8},
        )

    def test_json_file_created(self):
        report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_json(report, Path(tmpdir) / "test.json")
            assert path.exists()

    def test_json_is_valid(self):
        report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_json(report, Path(tmpdir) / "test.json")
            with open(path, "r") as f:
                data = json.load(f)
            assert data["model_name"] == "json-test-model"
            assert len(data["results"]) == 1
            assert data["results"][0]["threshold"] == 0.60

    def test_json_has_nested_structure(self):
        report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_json(report, Path(tmpdir) / "test.json")
            with open(path, "r") as f:
                data = json.load(f)
            result = data["results"][0]
            assert "classification" in result
            assert "performance" in result
            assert "similarity_score_stats" in result


class TestExportSummaryText:
    def test_summary_contains_model_name(self):
        report = EvaluationReport(
            results=[
                ThresholdResult(
                    threshold=0.50,
                    classification=ClassificationMetrics(true_positives=5, true_negatives=3),
                    performance=PerformanceMetrics(),
                    similarity_scores=[0.6],
                ),
            ],
            model_name="summary-test-model",
            timestamp="2026-01-01",
        )
        text = export_summary_text(report)
        assert "summary-test-model" in text
        assert "0.50" in text

    def test_summary_contains_best_f1(self):
        report = EvaluationReport(
            results=[
                ThresholdResult(
                    threshold=0.50,
                    classification=ClassificationMetrics(true_positives=5, true_negatives=5),
                    performance=PerformanceMetrics(),
                    similarity_scores=[0.6],
                ),
            ],
            model_name="test",
            timestamp="2026-01-01",
        )
        text = export_summary_text(report)
        assert "Best F1" in text
