"""
app.evaluation.export — Evaluation Results Exporter
=====================================================

Serializes evaluation reports to machine-readable formats for AI Lab reporting:

- **CSV**: One row per threshold, flat columns for all metrics. Easy to import
  into spreadsheet tools, pandas, or plotting libraries.
- **JSON**: Complete nested structure preserving the full report hierarchy.
  Suitable for programmatic analysis and archival.
- **Text Summary**: Formatted plaintext table for terminal display and
  inclusion in documentation.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Union

from app.evaluation.engine import EvaluationReport

logger = logging.getLogger(__name__)


def export_csv(
    report: EvaluationReport,
    filepath: Union[str, Path],
) -> Path:
    """Export evaluation report as a CSV file.

    Each row represents one threshold experiment. Columns include the threshold
    value, all classification metrics, and all performance metrics.

    Parameters
    ----------
    report : EvaluationReport
        Completed evaluation report.
    filepath : str or Path
        Output CSV file path.

    Returns
    -------
    Path
        Absolute path to the written CSV file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "avg_embedding_time_ms",
        "min_embedding_time_ms",
        "max_embedding_time_ms",
        "avg_recognition_latency_ms",
        "min_recognition_latency_ms",
        "max_recognition_latency_ms",
        "throughput_fps",
        "total_queries",
        "similarity_mean",
        "similarity_std",
        "similarity_min",
        "similarity_max",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in report.results:
            row = {"threshold": result.threshold}
            row.update(result.classification.as_dict())
            row.update(result.performance.as_dict())

            # Add similarity score statistics
            stats = result.as_dict()["similarity_score_stats"]
            row["similarity_mean"] = stats["mean"]
            row["similarity_std"] = stats["std"]
            row["similarity_min"] = stats["min"]
            row["similarity_max"] = stats["max"]

            writer.writerow(row)

    logger.info("Evaluation results exported to CSV: %s", filepath)
    return filepath.resolve()


def export_json(
    report: EvaluationReport,
    filepath: Union[str, Path],
) -> Path:
    """Export evaluation report as a JSON file.

    Preserves the full nested structure of the evaluation report including
    configuration, dataset summary, and per-threshold results.

    Parameters
    ----------
    report : EvaluationReport
        Completed evaluation report.
    filepath : str or Path
        Output JSON file path.

    Returns
    -------
    Path
        Absolute path to the written JSON file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = report.as_dict()

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info("Evaluation results exported to JSON: %s", filepath)
    return filepath.resolve()


def export_summary_text(report: EvaluationReport) -> str:
    """Generate a formatted plaintext summary table of the evaluation report.

    Uses ASCII-safe characters for cross-platform console compatibility.

    Parameters
    ----------
    report : EvaluationReport
        Completed evaluation report.

    Returns
    -------
    str
        Multi-line formatted summary string.
    """
    lines = []

    # Header
    lines.append("=" * 94)
    lines.append(f"  ReminisceCV Recognition Evaluation Report")
    lines.append(f"  Model: {report.model_name}")
    lines.append(f"  Timestamp: {report.timestamp}")
    lines.append("=" * 94)

    # Dataset summary
    ds = report.dataset_summary
    lines.append("")
    lines.append("  Dataset Summary:")
    lines.append(f"    Total queries:     {ds.get('total_queries', 0)}")
    lines.append(f"    Known entities:    {ds.get('known_entities', 0)}")
    lines.append(f"    Known queries:     {ds.get('known_queries', 0)}")
    lines.append(f"    Distractor queries: {ds.get('distractor_queries', 0)}")

    # Classification metrics table (ASCII borders)
    lines.append("")
    lines.append("  Classification Metrics by Threshold:")
    sep = "  +----------+----------+----------+----------+----------+------+------+------+------+"
    lines.append(sep)
    lines.append(
        "  | Thresh   | Accuracy | Precision| Recall   | F1 Score |  TP  |  FP  |  FN  |  TN  |"
    )
    lines.append(sep)

    for result in report.results:
        c = result.classification
        lines.append(
            f"  | {result.threshold:>6.2f}   | {c.accuracy:>8.4f} | {c.precision:>8.4f} |"
            f" {c.recall:>8.4f} | {c.f1_score:>8.4f} | {c.true_positives:>4d} |"
            f" {c.false_positives:>4d} | {c.false_negatives:>4d} | {c.true_negatives:>4d} |"
        )

    lines.append(sep)

    # Performance metrics table (ASCII borders)
    lines.append("")
    lines.append("  Performance Metrics by Threshold:")
    perf_sep = "  +----------+---------------+---------------+---------------+----------+"
    lines.append(perf_sep)
    lines.append(
        "  | Thresh   | Emb Avg (ms)  | Emb Min (ms)  | Emb Max (ms)  |  FPS     |"
    )
    lines.append(perf_sep)

    for result in report.results:
        p = result.performance
        lines.append(
            f"  | {result.threshold:>6.2f}   | {p.avg_embedding_time_ms:>11.3f}   |"
            f" {p.min_embedding_time_ms:>11.3f}   | {p.max_embedding_time_ms:>11.3f}   |"
            f" {p.throughput_fps:>6.1f}   |"
        )

    lines.append(perf_sep)

    # Best threshold recommendation
    best = report.best_f1_result
    if best:
        lines.append("")
        lines.append(
            f"  * Best F1 Score: {best.classification.f1_score:.4f} "
            f"at threshold={best.threshold:.2f}"
        )

    lines.append("")
    lines.append("=" * 94)

    return "\n".join(lines)


def export_robustness_summary_text(report) -> str:
    """Generate a formatted plaintext summary of robustness evaluation results.

    Produces a comparative table showing how F1-score and accuracy degrade
    across different robustness conditions and severity levels.

    Parameters
    ----------
    report : RobustnessReport
        Completed robustness evaluation report (from ``app.evaluation.robustness``).

    Returns
    -------
    str
        Multi-line formatted summary string.
    """
    lines = []

    lines.append("=" * 94)
    lines.append("  ReminisceCV Robustness Evaluation Summary")
    lines.append(f"  Model: {report.model_name}")
    lines.append(f"  Timestamp: {report.timestamp}")
    lines.append("=" * 94)

    # Collect unique conditions in order
    conditions_seen: list[str] = []
    for exp in report.experiments:
        if exp.condition not in conditions_seen:
            conditions_seen.append(exp.condition)

    for condition in conditions_seen:
        results = report.get_condition_results(condition)
        if not results:
            continue

        lines.append("")
        lines.append(f"  Condition: {condition.upper()}")
        sep = "  +------------------+----------+----------+----------+----------+"
        lines.append(sep)
        lines.append(
            "  | Level            | Best F1  | Best Acc | Param    | Degrad.  |"
        )
        lines.append(sep)

        for exp in results:
            lines.append(
                f"  | {exp.level.name:<16s} | {exp.best_f1():>8.4f} |"
                f" {exp.best_accuracy():>8.4f} | {exp.level.parameter:>8.2f} |"
                f" {exp.level.similarity_degradation:>8.2f} |"
            )

        lines.append(sep)

    lines.append("")
    lines.append("=" * 94)

    return "\n".join(lines)
