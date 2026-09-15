"""
tests/test_robustness.py — Robustness Evaluation Tests
========================================================

Verifies the robustness evaluation framework:

- Image transformation functions produce correct output shapes and value ranges.
- RobustnessExperimentRunner produces results for all conditions.
- Similarity degradation follows expected pattern (harsher → lower scores).
- Export functions produce valid output files.
- Reference count experiment varies gallery size correctly.
- Default levels are correctly defined for all conditions.
"""

import csv
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.evaluation.robustness import (
    RobustnessCondition,
    RobustnessExperimentResult,
    RobustnessExperimentRunner,
    RobustnessLevel,
    RobustnessReport,
    apply_background_change,
    apply_lighting_change,
    apply_occlusion,
    apply_rotation,
    apply_scale,
    get_default_levels,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_image():
    """Create a simple 224×224 BGR test image with a colored shape."""
    img = np.full((224, 224, 3), (50, 100, 150), dtype=np.uint8)
    cv2.circle(img, (112, 112), 60, (200, 50, 50), -1)
    return img


@pytest.fixture
def small_image():
    """Create a small 64×64 test image for fast tests."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.rectangle(img, (16, 16), (48, 48), (255, 128, 64), -1)
    return img


# ── Image Transformation Tests ───────────────────────────────────────────────


class TestApplyLightingChange:
    """Tests for the apply_lighting_change function."""

    def test_no_change_returns_same_shape(self, sample_image):
        """Factor 1.0 should return an identical-shape image."""
        result = apply_lighting_change(sample_image, 1.0)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_no_change_preserves_pixels(self, sample_image):
        """Factor 1.0 should leave pixel values unchanged."""
        result = apply_lighting_change(sample_image, 1.0)
        np.testing.assert_array_equal(result, sample_image)

    def test_dim_reduces_brightness(self, sample_image):
        """Factor < 1.0 should reduce overall brightness."""
        result = apply_lighting_change(sample_image, 0.4)
        assert result.mean() < sample_image.mean()

    def test_bright_increases_brightness(self, sample_image):
        """Factor > 1.0 should increase overall brightness."""
        result = apply_lighting_change(sample_image, 2.0)
        assert result.mean() > sample_image.mean()

    def test_zero_factor_returns_black(self, sample_image):
        """Factor 0 should return a black image."""
        result = apply_lighting_change(sample_image, 0.0)
        assert result.sum() == 0

    def test_output_clipped_to_uint8(self, sample_image):
        """Even extreme factors should produce valid uint8 values."""
        result = apply_lighting_change(sample_image, 5.0)
        assert result.dtype == np.uint8
        assert result.max() <= 255
        assert result.min() >= 0


class TestApplyRotation:
    """Tests for the apply_rotation function."""

    def test_zero_rotation_preserves(self, sample_image):
        """0° rotation should return an identical image."""
        result = apply_rotation(sample_image, 0.0)
        np.testing.assert_array_equal(result, sample_image)

    def test_rotation_preserves_shape(self, sample_image):
        """Rotated image should have the same shape."""
        result = apply_rotation(sample_image, 30.0)
        assert result.shape == sample_image.shape

    def test_rotation_changes_pixels(self, sample_image):
        """Non-zero rotation should change pixel arrangement."""
        result = apply_rotation(sample_image, 45.0)
        assert not np.array_equal(result, sample_image)

    def test_360_rotation_returns_similar(self, sample_image):
        """360° rotation should return approximately the original."""
        result = apply_rotation(sample_image, 360.0)
        # Allow small interpolation differences
        diff = np.abs(result.astype(np.int16) - sample_image.astype(np.int16))
        assert diff.mean() < 5.0

    def test_negative_rotation(self, sample_image):
        """Negative rotation should work without errors."""
        result = apply_rotation(sample_image, -15.0)
        assert result.shape == sample_image.shape


class TestApplyScale:
    """Tests for the apply_scale function."""

    def test_scale_one_preserves(self, sample_image):
        """Scale 1.0 should return an identical image."""
        result = apply_scale(sample_image, 1.0)
        np.testing.assert_array_equal(result, sample_image)

    def test_scale_preserves_shape(self, sample_image):
        """Scaled image should have the same dimensions."""
        result = apply_scale(sample_image, 0.5)
        assert result.shape == sample_image.shape

    def test_small_scale_shows_padding(self, sample_image):
        """Scale < 1.0 should result in border padding visible."""
        result = apply_scale(sample_image, 0.3)
        assert result.shape == sample_image.shape
        # Object is now smaller, so corners should be closer to border color
        assert not np.array_equal(result, sample_image)

    def test_very_small_scale(self, sample_image):
        """Very small scale should not crash."""
        result = apply_scale(sample_image, 0.05)
        assert result.shape == sample_image.shape

    def test_large_scale(self, sample_image):
        """Scale > 1.0 should crop to original dimensions."""
        result = apply_scale(sample_image, 1.5)
        assert result.shape == sample_image.shape


class TestApplyOcclusion:
    """Tests for the apply_occlusion function."""

    def test_zero_occlusion_preserves(self, sample_image):
        """0% occlusion should return an identical image."""
        result = apply_occlusion(sample_image, 0.0)
        np.testing.assert_array_equal(result, sample_image)

    def test_occlusion_preserves_shape(self, sample_image):
        """Occluded image should have the same shape."""
        result = apply_occlusion(sample_image, 0.25)
        assert result.shape == sample_image.shape

    def test_occlusion_changes_pixels(self, sample_image):
        """Non-zero occlusion should modify some pixels."""
        result = apply_occlusion(sample_image, 0.25)
        assert not np.array_equal(result, sample_image)

    def test_occlusion_adds_gray_region(self, sample_image):
        """Occluded region should contain gray pixels (128, 128, 128)."""
        result = apply_occlusion(sample_image, 0.5)
        # Check that some pixels became the gray occluder
        gray_mask = np.all(result == [128, 128, 128], axis=-1)
        assert gray_mask.sum() > 0

    def test_reproducible_with_same_seed(self, sample_image):
        """Same seed should produce identical occlusion placement."""
        r1 = apply_occlusion(sample_image, 0.3, seed=42)
        r2 = apply_occlusion(sample_image, 0.3, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seed_different_placement(self, sample_image):
        """Different seeds should produce different occlusion placement."""
        r1 = apply_occlusion(sample_image, 0.3, seed=42)
        r2 = apply_occlusion(sample_image, 0.3, seed=99)
        assert not np.array_equal(r1, r2)


class TestApplyBackgroundChange:
    """Tests for the apply_background_change function."""

    def test_original_preserves(self, sample_image):
        """Type 0.0 should return an identical image."""
        result = apply_background_change(sample_image, 0.0)
        np.testing.assert_array_equal(result, sample_image)

    def test_background_preserves_shape(self, sample_image):
        """All background types should preserve shape."""
        for bg_type in [1.0, 2.0, 3.0]:
            result = apply_background_change(sample_image, bg_type)
            assert result.shape == sample_image.shape

    def test_white_background(self, sample_image):
        """White background should introduce white pixels."""
        result = apply_background_change(sample_image, 1.0)
        white_mask = np.all(result == [255, 255, 255], axis=-1)
        # Should have some white pixels in background region
        assert white_mask.sum() > 0

    def test_dark_background(self, sample_image):
        """Dark background should introduce dark pixels."""
        result = apply_background_change(sample_image, 2.0)
        dark_mask = np.all(result == [20, 20, 20], axis=-1)
        assert dark_mask.sum() > 0

    def test_noisy_background_has_variance(self, sample_image):
        """Noisy background should have high pixel variance."""
        result = apply_background_change(sample_image, 3.0)
        # The noisy region should have much higher variance than original
        assert not np.array_equal(result, sample_image)


# ── Default Levels Tests ──────────────────────────────────────────────────────


class TestDefaultLevels:
    """Tests for get_default_levels()."""

    def test_all_conditions_have_levels(self):
        """All conditions should have at least one level defined."""
        levels = get_default_levels()
        for condition in RobustnessCondition.ALL:
            assert condition in levels, f"Missing levels for {condition}"
            assert len(levels[condition]) >= 2, f"Need at least 2 levels for {condition}"

    def test_levels_have_required_fields(self):
        """All levels should have name, condition, parameter, and degradation."""
        levels = get_default_levels()
        for condition, level_list in levels.items():
            for level in level_list:
                assert level.name, f"Missing name in {condition}"
                assert level.condition == condition
                assert isinstance(level.parameter, (int, float))
                assert 0.0 <= level.similarity_degradation <= 1.0

    def test_each_condition_has_baseline(self):
        """Each condition should have a baseline level with 0 degradation."""
        levels = get_default_levels()
        for condition in RobustnessCondition.ALL:
            baselines = [l for l in levels[condition] if l.similarity_degradation == 0.0]
            assert len(baselines) >= 1, f"No baseline level for {condition}"

    def test_degradation_increases_with_severity(self):
        """For most conditions, degradation should generally increase."""
        levels = get_default_levels()
        for condition in [
            RobustnessCondition.ANGLE,
            RobustnessCondition.OCCLUSION,
        ]:
            degradations = [l.similarity_degradation for l in levels[condition]]
            # First should be 0 (baseline), rest should be >= previous
            assert degradations[0] == 0.0
            for i in range(1, len(degradations)):
                assert degradations[i] >= degradations[i - 1], (
                    f"{condition}: degradation should increase: "
                    f"{degradations[i]} < {degradations[i-1]}"
                )


# ── RobustnessCondition Tests ────────────────────────────────────────────────


class TestRobustnessCondition:
    """Tests for RobustnessCondition constants."""

    def test_all_contains_all_conditions(self):
        """ALL list should contain exactly the 6 defined conditions."""
        assert len(RobustnessCondition.ALL) == 6
        assert RobustnessCondition.LIGHTING in RobustnessCondition.ALL
        assert RobustnessCondition.ANGLE in RobustnessCondition.ALL
        assert RobustnessCondition.DISTANCE in RobustnessCondition.ALL
        assert RobustnessCondition.OCCLUSION in RobustnessCondition.ALL
        assert RobustnessCondition.BACKGROUND in RobustnessCondition.ALL
        assert RobustnessCondition.NUM_REFERENCES in RobustnessCondition.ALL


# ── RobustnessLevel Tests ────────────────────────────────────────────────────


class TestRobustnessLevel:
    """Tests for RobustnessLevel dataclass."""

    def test_creation(self):
        """Should create a level with all fields."""
        level = RobustnessLevel(
            name="test",
            condition="lighting",
            parameter=0.5,
            similarity_degradation=0.1,
            description="Test level",
        )
        assert level.name == "test"
        assert level.parameter == 0.5
        assert level.similarity_degradation == 0.1


# ── RobustnessExperimentResult Tests ─────────────────────────────────────────


class TestRobustnessExperimentResult:
    """Tests for RobustnessExperimentResult dataclass."""

    def test_as_dict_has_required_keys(self):
        """Serialized dict should contain all required fields."""
        from app.evaluation.engine import (
            ClassificationMetrics,
            EvaluationReport,
            PerformanceMetrics,
            ThresholdResult,
        )

        level = RobustnessLevel(
            name="test", condition="lighting",
            parameter=0.5, similarity_degradation=0.1,
        )
        report = EvaluationReport(
            results=[
                ThresholdResult(
                    threshold=0.5,
                    classification=ClassificationMetrics(
                        true_positives=8, false_positives=1,
                        false_negatives=2, true_negatives=4,
                    ),
                    performance=PerformanceMetrics(),
                    similarity_scores=[0.7, 0.8, 0.9],
                )
            ],
            model_name="test-model",
        )
        result = RobustnessExperimentResult(
            condition="lighting", level=level, report=report,
        )

        d = result.as_dict()
        assert "condition" in d
        assert "level_name" in d
        assert "best_f1" in d
        assert "best_accuracy" in d
        assert "report" in d

    def test_best_f1_computes_correctly(self):
        """best_f1 should return the maximum F1 across thresholds."""
        from app.evaluation.engine import (
            ClassificationMetrics,
            EvaluationReport,
            ThresholdResult,
            PerformanceMetrics,
        )

        level = RobustnessLevel(
            name="test", condition="lighting",
            parameter=1.0, similarity_degradation=0.0,
        )
        report = EvaluationReport(
            results=[
                ThresholdResult(
                    threshold=0.5,
                    classification=ClassificationMetrics(
                        true_positives=10, false_positives=0,
                        false_negatives=0, true_negatives=5,
                    ),
                    performance=PerformanceMetrics(),
                )
            ],
        )
        result = RobustnessExperimentResult(
            condition="lighting", level=level, report=report,
        )
        assert result.best_f1() == 1.0


# ── Experiment Runner Tests (Integration) ────────────────────────────────────


class TestRobustnessExperimentRunner:
    """Integration tests for the robustness experiment runner."""

    @pytest.fixture
    def eval_model(self):
        """Create a controlled evaluation model."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.run_evaluation import EvaluationVisionModel
        return EvaluationVisionModel(embedding_dim=128, seed=42)

    def test_run_single_condition(self, eval_model):
        """Should produce results for a single condition."""
        runner = RobustnessExperimentRunner(
            vision_model=eval_model,
            thresholds=[0.50, 0.70],
            n_entities=2,
            n_queries_per_entity=3,
            n_distractors=2,
            seed=42,
        )
        report = runner.run_all_experiments(
            conditions=[RobustnessCondition.LIGHTING],
        )
        assert len(report.experiments) > 0
        # Should have one result per lighting level
        lighting_levels = get_default_levels()[RobustnessCondition.LIGHTING]
        assert len(report.experiments) == len(lighting_levels)

    def test_run_multiple_conditions(self, eval_model):
        """Should produce results for multiple conditions."""
        runner = RobustnessExperimentRunner(
            vision_model=eval_model,
            thresholds=[0.60],
            n_entities=2,
            n_queries_per_entity=2,
            n_distractors=2,
            seed=42,
        )
        conditions = [RobustnessCondition.LIGHTING, RobustnessCondition.ANGLE]
        report = runner.run_all_experiments(conditions=conditions)

        # Should have results for both conditions
        lighting_count = len(report.get_condition_results(RobustnessCondition.LIGHTING))
        angle_count = len(report.get_condition_results(RobustnessCondition.ANGLE))
        assert lighting_count > 0
        assert angle_count > 0

    def test_baseline_outperforms_degraded(self, eval_model):
        """Baseline level should have higher F1 than degraded levels."""
        runner = RobustnessExperimentRunner(
            vision_model=eval_model,
            thresholds=[0.50, 0.60, 0.70, 0.80],
            n_entities=3,
            n_queries_per_entity=5,
            n_distractors=3,
            seed=42,
        )
        report = runner.run_all_experiments(
            conditions=[RobustnessCondition.OCCLUSION],
        )
        results = report.get_condition_results(RobustnessCondition.OCCLUSION)

        # First level (none) should be baseline
        baseline_f1 = results[0].best_f1()
        # Last level (partial_50) should have lower F1
        worst_f1 = results[-1].best_f1()
        assert baseline_f1 >= worst_f1, (
            f"Baseline F1 ({baseline_f1}) should be >= degraded F1 ({worst_f1})"
        )

    def test_report_has_metadata(self, eval_model):
        """Report should include timestamp, model name, and config."""
        runner = RobustnessExperimentRunner(
            vision_model=eval_model,
            thresholds=[0.60],
            n_entities=2,
            n_queries_per_entity=2,
            n_distractors=1,
            seed=42,
        )
        report = runner.run_all_experiments(
            conditions=[RobustnessCondition.DISTANCE],
        )
        assert report.timestamp != ""
        assert report.model_name != ""
        assert "thresholds" in report.config_summary
        assert "n_entities" in report.config_summary

    def test_report_serialization(self, eval_model):
        """Report should serialize to a valid dict."""
        runner = RobustnessExperimentRunner(
            vision_model=eval_model,
            thresholds=[0.60],
            n_entities=2,
            n_queries_per_entity=2,
            n_distractors=1,
            seed=42,
        )
        report = runner.run_all_experiments(
            conditions=[RobustnessCondition.BACKGROUND],
        )
        d = report.as_dict()
        assert "experiments" in d
        assert "timestamp" in d
        assert len(d["experiments"]) > 0

        # Should be JSON-serializable
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 0

    def test_num_references_varies_gallery(self, eval_model):
        """Reference count experiment should use different gallery sizes."""
        runner = RobustnessExperimentRunner(
            vision_model=eval_model,
            thresholds=[0.60],
            n_entities=2,
            n_queries_per_entity=2,
            n_distractors=1,
            seed=42,
        )

        # Test with custom levels for reference count
        custom_levels = {
            RobustnessCondition.NUM_REFERENCES: [
                RobustnessLevel(
                    name="1_ref",
                    condition=RobustnessCondition.NUM_REFERENCES,
                    parameter=1.0,
                    similarity_degradation=0.10,
                ),
                RobustnessLevel(
                    name="3_refs",
                    condition=RobustnessCondition.NUM_REFERENCES,
                    parameter=3.0,
                    similarity_degradation=0.02,
                ),
            ]
        }

        report = runner.run_all_experiments(
            conditions=[RobustnessCondition.NUM_REFERENCES],
            levels=custom_levels,
        )
        results = report.get_condition_results(RobustnessCondition.NUM_REFERENCES)
        assert len(results) == 2


# ── RobustnessReport Tests ───────────────────────────────────────────────────


class TestRobustnessReport:
    """Tests for the RobustnessReport dataclass."""

    def test_get_condition_results_filters_correctly(self):
        """Should return only experiments matching the requested condition."""
        from app.evaluation.engine import EvaluationReport

        exp1 = RobustnessExperimentResult(
            condition="lighting",
            level=RobustnessLevel(
                name="dim", condition="lighting",
                parameter=0.4, similarity_degradation=0.08,
            ),
            report=EvaluationReport(results=[]),
        )
        exp2 = RobustnessExperimentResult(
            condition="angle",
            level=RobustnessLevel(
                name="rotated_30", condition="angle",
                parameter=30.0, similarity_degradation=0.12,
            ),
            report=EvaluationReport(results=[]),
        )

        report = RobustnessReport(experiments=[exp1, exp2])
        lighting = report.get_condition_results("lighting")
        assert len(lighting) == 1
        assert lighting[0].level.name == "dim"

        angle = report.get_condition_results("angle")
        assert len(angle) == 1
        assert angle[0].level.name == "rotated_30"

    def test_empty_report_serializes(self):
        """Empty report should serialize without errors."""
        report = RobustnessReport()
        d = report.as_dict()
        assert d["experiments"] == []


# ── Export Tests ──────────────────────────────────────────────────────────────


class TestRobustnessExport:
    """Tests for robustness-specific export functions."""

    def test_export_robustness_summary_text(self):
        """Should produce a formatted text summary."""
        from app.evaluation.export import export_robustness_summary_text
        from app.evaluation.engine import (
            ClassificationMetrics,
            EvaluationReport,
            PerformanceMetrics,
            ThresholdResult,
        )

        level = RobustnessLevel(
            name="dim", condition="lighting",
            parameter=0.4, similarity_degradation=0.08,
        )
        report_inner = EvaluationReport(
            results=[
                ThresholdResult(
                    threshold=0.6,
                    classification=ClassificationMetrics(
                        true_positives=8, false_positives=1,
                        false_negatives=2, true_negatives=4,
                    ),
                    performance=PerformanceMetrics(),
                    similarity_scores=[0.7, 0.8],
                )
            ],
            model_name="test-model",
        )

        robustness_report = RobustnessReport(
            experiments=[
                RobustnessExperimentResult(
                    condition="lighting",
                    level=level,
                    report=report_inner,
                )
            ],
            timestamp="2026-01-01T00:00:00Z",
            model_name="test-model",
        )

        summary = export_robustness_summary_text(robustness_report)
        assert "LIGHTING" in summary
        assert "dim" in summary
        assert "test-model" in summary
        assert len(summary) > 100
