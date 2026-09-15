"""
scripts/run_robustness.py — ReminisceCV Robustness Evaluation Runner
=====================================================================

Runs controlled robustness experiments that measure how recognition
performance degrades under challenging conditions:

- Lighting: dim, bright, overexposed
- Angle: 15°, 30°, 45° rotation
- Distance: medium, far
- Occlusion: 25%, 50% masked
- Background: white, dark, noisy
- Reference count: 1, 2, 3, 5 reference images

Usage
-----
Run all experiments:
    python scripts/run_robustness.py

Run specific conditions:
    python scripts/run_robustness.py --conditions lighting,angle,distance

Custom thresholds:
    python scripts/run_robustness.py --thresholds 0.50,0.60,0.70,0.80,0.90

Full options:
    python scripts/run_robustness.py --help
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.export import export_csv, export_json
from app.evaluation.robustness import (
    RobustnessCondition,
    RobustnessExperimentRunner,
    RobustnessReport,
    get_default_levels,
)

# Reuse the controlled model from run_evaluation
from scripts.run_evaluation import EvaluationVisionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reminisce_robustness")


def format_robustness_summary(report: RobustnessReport) -> str:
    """Format a comparative summary table of all robustness experiments.

    Parameters
    ----------
    report : RobustnessReport
        Complete robustness evaluation report.

    Returns
    -------
    str
        Multi-line formatted summary string.
    """
    lines = []
    lines.append("=" * 94)
    lines.append("  ReminisceCV Robustness Evaluation Report")
    lines.append(f"  Model: {report.model_name}")
    lines.append(f"  Timestamp: {report.timestamp}")
    lines.append("=" * 94)

    # Group experiments by condition
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
        lines.append("  " + "-" * 88)

        sep = "  +------------------+----------+----------+----------+----------+------+------+------+"
        lines.append(sep)
        lines.append(
            "  | Level            | Best F1  | Best Acc | Prec@BF1 | Rec@BF1  |  TP  |  FP  |  FN  |"
        )
        lines.append(sep)

        for exp in results:
            best = exp.report.best_f1_result
            if best:
                c = best.classification
                lines.append(
                    f"  | {exp.level.name:<16s} | {exp.best_f1():>8.4f} |"
                    f" {exp.best_accuracy():>8.4f} | {c.precision:>8.4f} |"
                    f" {c.recall:>8.4f} | {c.true_positives:>4d} |"
                    f" {c.false_positives:>4d} | {c.false_negatives:>4d} |"
                )
            else:
                lines.append(
                    f"  | {exp.level.name:<16s} |   0.0000 |   0.0000 |   0.0000 |"
                    f"   0.0000 |    0 |    0 |    0 |"
                )

        lines.append(sep)

    # Summary
    lines.append("")
    lines.append("  Degradation Summary (Best F1 per condition):")
    lines.append("  " + "-" * 60)

    for condition in conditions_seen:
        results = report.get_condition_results(condition)
        if not results:
            continue
        baseline_f1 = results[0].best_f1() if results else 0.0
        worst_f1 = min(r.best_f1() for r in results) if results else 0.0
        drop = baseline_f1 - worst_f1

        lines.append(
            f"    {condition:<16s}  baseline={baseline_f1:.4f}  "
            f"worst={worst_f1:.4f}  drop={drop:+.4f}"
        )

    lines.append("")
    lines.append("=" * 94)
    return "\n".join(lines)


def export_robustness_report(
    report: RobustnessReport,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Export robustness report to CSV and JSON.

    Parameters
    ----------
    report : RobustnessReport
        Completed robustness report.
    output_dir : str or Path
        Output directory.

    Returns
    -------
    tuple of Path
        (csv_path, json_path) of the exported files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export individual condition reports as CSV
    for exp in report.experiments:
        condition_dir = output_dir / exp.condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        csv_path = condition_dir / f"{exp.level.name}_{timestamp_tag}.csv"
        export_csv(exp.report, csv_path)

    # Export full report as JSON
    json_path = output_dir / f"robustness_{timestamp_tag}.json"
    json_data = report.as_dict()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)
    logger.info("Full robustness report exported to JSON: %s", json_path)

    # Also write a "latest" copy
    json_latest = output_dir / "robustness_latest.json"
    with open(json_latest, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)

    # Export summary CSV (one row per condition × level)
    summary_csv_path = output_dir / f"robustness_summary_{timestamp_tag}.csv"
    _export_summary_csv(report, summary_csv_path)

    summary_csv_latest = output_dir / "robustness_summary_latest.csv"
    _export_summary_csv(report, summary_csv_latest)

    return json_path, summary_csv_path


def _export_summary_csv(report: RobustnessReport, filepath: Path) -> None:
    """Write a flat summary CSV with one row per condition × level."""
    import csv

    filepath.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "condition",
        "level",
        "parameter",
        "similarity_degradation",
        "best_f1",
        "best_accuracy",
        "best_threshold",
        "precision_at_best",
        "recall_at_best",
        "tp_at_best",
        "fp_at_best",
        "fn_at_best",
        "tn_at_best",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for exp in report.experiments:
            best = exp.report.best_f1_result
            row = {
                "condition": exp.condition,
                "level": exp.level.name,
                "parameter": exp.level.parameter,
                "similarity_degradation": exp.level.similarity_degradation,
                "best_f1": round(exp.best_f1(), 4),
                "best_accuracy": round(exp.best_accuracy(), 4),
                "best_threshold": best.threshold if best else 0.0,
                "precision_at_best": round(best.classification.precision, 4) if best else 0.0,
                "recall_at_best": round(best.classification.recall, 4) if best else 0.0,
                "tp_at_best": best.classification.true_positives if best else 0,
                "fp_at_best": best.classification.false_positives if best else 0,
                "fn_at_best": best.classification.false_negatives if best else 0,
                "tn_at_best": best.classification.true_negatives if best else 0,
            }
            writer.writerow(row)

    logger.info("Robustness summary CSV exported: %s", filepath)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ReminisceCV Robustness Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=",".join(RobustnessCondition.ALL),
        help="Comma-separated list of conditions to evaluate",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.50,0.60,0.70,0.80,0.90",
        help="Comma-separated similarity thresholds",
    )
    parser.add_argument(
        "--entities",
        type=int,
        default=5,
        help="Number of distinct entity classes",
    )
    parser.add_argument(
        "--queries-per-entity",
        type=int,
        default=5,
        help="Number of query images per entity per level",
    )
    parser.add_argument(
        "--distractors",
        type=int,
        default=5,
        help="Number of distractor queries per level",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/results/robustness",
        help="Output directory for result files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",")]
    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]

    print()
    print("=" * 62)
    print("  ReminisceCV — Robustness Evaluation")
    print("=" * 62)
    print()
    print(f"  Conditions: {', '.join(conditions)}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Entities:   {args.entities}")
    print()

    t_start = time.perf_counter()

    # Create the controlled evaluation model
    eval_model = EvaluationVisionModel(embedding_dim=512, seed=args.seed)

    runner = RobustnessExperimentRunner(
        vision_model=eval_model,
        thresholds=thresholds,
        n_entities=args.entities,
        n_queries_per_entity=args.queries_per_entity,
        n_distractors=args.distractors,
        seed=args.seed,
    )

    report = runner.run_all_experiments(conditions=conditions)

    t_elapsed = time.perf_counter() - t_start

    # Print summary
    summary = format_robustness_summary(report)
    print(summary)

    # Export results
    json_path, csv_path = export_robustness_report(report, args.output_dir)

    print(f"  Results saved to:")
    print(f"    JSON: {json_path}")
    print(f"    CSV:  {csv_path}")
    print(f"    (latest: robustness_latest.json, robustness_summary_latest.csv)")
    print()
    print(f"  Total evaluation time: {t_elapsed:.2f}s")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
