"""
app.evaluation — AI Recognition Evaluation Framework
=====================================================

Provides a modular evaluation system for measuring ReminisceCV's object
recognition pipeline quality and computational performance:

- ``RecognitionEvaluator``: Orchestrates multi-threshold evaluation sweeps.
- ``SyntheticDataGenerator``: Generates reproducible test datasets without private content.
- ``export_csv`` / ``export_json``: Machine-readable result serialization.

Designed for AI Lab reports with metrics including accuracy, precision,
recall, F1-score, and inference latency benchmarks.
"""

from app.evaluation.engine import (
    ClassificationMetrics,
    EvaluationConfig,
    EvaluationReport,
    PerformanceMetrics,
    RecognitionEvaluator,
    ThresholdResult,
)
from app.evaluation.export import (
    export_csv,
    export_json,
    export_summary_text,
)
from app.evaluation.test_data import (
    EvaluationDataset,
    EvaluationSample,
    SyntheticDataGenerator,
)

__all__ = [
    "ClassificationMetrics",
    "EvaluationConfig",
    "EvaluationDataset",
    "EvaluationReport",
    "EvaluationSample",
    "PerformanceMetrics",
    "RecognitionEvaluator",
    "SyntheticDataGenerator",
    "ThresholdResult",
    "export_csv",
    "export_json",
    "export_summary_text",
]
