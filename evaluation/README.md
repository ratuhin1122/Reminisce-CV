# ReminisceCV — AI Recognition Evaluation

## Purpose

This evaluation framework measures the quality and performance of ReminisceCV's
personal object recognition pipeline. It produces quantitative metrics suitable
for AI Lab reports, research documentation, and system tuning.

**All results are computed from actual inference — no values are invented.**

## What It Measures

### Classification Metrics (per threshold)

| Metric            | Description                                      |
|-------------------|--------------------------------------------------|
| **Accuracy**      | (TP + TN) / Total                                |
| **Precision**     | TP / (TP + FP) — how many predicted matches are correct |
| **Recall**        | TP / (TP + FN) — how many real objects are found |
| **F1-Score**      | Harmonic mean of precision and recall            |
| **True Positives**  | Correct match to the right entity              |
| **False Positives** | Incorrect match (wrong entity or distractor matched) |
| **False Negatives** | Known entity not recognized                    |
| **True Negatives**  | Distractor correctly rejected                  |

### Performance Metrics

| Metric                     | Description                               |
|----------------------------|-------------------------------------------|
| **Avg Embedding Time (ms)**  | Mean time to extract a single image embedding |
| **Avg Recognition Latency (ms)** | Mean time for full recognition (embed + compare) |
| **Throughput (FPS)**        | Queries processed per second               |

### Threshold Sweep

The evaluation runs at multiple similarity thresholds (default: 0.50, 0.60, 0.70, 0.80, 0.90)
to show how threshold selection affects the precision–recall tradeoff.

## How to Run

### Default Evaluation (Mock Model)

```bash
python scripts/run_evaluation.py
```

This uses a lightweight mock embedding model and synthetic test images, requiring
no GPU, no model downloads, and no private photos.

### Custom Configuration

```bash
# More entities and queries
python scripts/run_evaluation.py --entities 8 --queries-per-entity 10 --distractors 20

# Custom thresholds
python scripts/run_evaluation.py --thresholds 0.40,0.50,0.60,0.70,0.80,0.90

# More timing repetitions for accurate performance measurement
python scripts/run_evaluation.py --timing-trials 10
```

### Live CLIP Model Evaluation

To evaluate with the real CLIP vision model (requires model download):

```bash
python scripts/run_evaluation.py --live
```

### Full Options

```bash
python scripts/run_evaluation.py --help
```

## Output Files

Results are saved to `evaluation/results/` in two formats:

### CSV (`evaluation_YYYYMMDD_HHMMSS.csv`)

One row per threshold. Columns:

```
threshold, accuracy, precision, recall, f1_score,
true_positives, false_positives, false_negatives, true_negatives,
avg_embedding_time_ms, min_embedding_time_ms, max_embedding_time_ms,
avg_recognition_latency_ms, throughput_fps,
similarity_mean, similarity_std, similarity_min, similarity_max
```

### JSON (`evaluation_YYYYMMDD_HHMMSS.json`)

Complete nested structure:

```json
{
  "model_name": "mock-clip-vit-b32",
  "timestamp": "2026-09-14T06:30:00+00:00",
  "config": { "thresholds": [0.5, 0.6, 0.7, 0.8, 0.9], ... },
  "dataset": { "total_queries": 35, "known_entities": 5, ... },
  "results": [
    {
      "threshold": 0.5,
      "classification": { "accuracy": 0.92, "precision": 0.88, ... },
      "performance": { "avg_embedding_time_ms": 0.15, ... },
      "similarity_score_stats": { "mean": 0.72, ... }
    },
    ...
  ]
}
```

A `_latest.csv` and `_latest.json` file is always updated with the most recent run.

## Interpreting Results

- **High F1 at a low threshold** → the model is confident and separates entities well.
- **Precision drops as threshold decreases** → more false positives at lenient thresholds.
- **Recall drops as threshold increases** → known objects go unrecognized at strict thresholds.
- The **best threshold** is typically the one that maximizes F1-score.

## Test Data

By default, the framework generates synthetic test images:

- Each entity gets a unique **color family** and **geometric shape** (circle, rectangle, triangle, diamond, cross).
- **Reference images** have minimal variation (used to build the gallery).
- **Query images** have moderate variation (color shifts, noise) to test recognition robustness.
- **Distractor images** use entirely different visual patterns and should not match any entity.

No private or personal photos are used. All images are generated programmatically.

## Adding Custom Test Data

To evaluate with your own images, you can build `EvaluationQuery` objects
directly in Python:

```python
from app.evaluation.engine import EvaluationQuery, RecognitionEvaluator
import cv2

queries = [
    EvaluationQuery(
        image=cv2.imread("path/to/test_image.jpg"),
        ground_truth_entity_id=1,  # or None for distractors
        label="my_object_query",
    ),
    ...
]
```

Do **not** commit personal photos to the repository.

## Robustness Evaluation Experiments

The robustness evaluation framework measures how recognition performance degrades
under challenging real-world conditions. It complements the standard evaluation
by isolating specific visual challenges.

### Conditions Tested

| Condition           | Levels                              | What It Measures                        |
|---------------------|-------------------------------------|-----------------------------------------|
| **Lighting**        | normal, dim, bright, overexposed    | Sensitivity to brightness/contrast changes |
| **Angle**           | front, 15°, 30°, 45° rotation      | Viewpoint invariance                    |
| **Distance**        | close, medium, far                  | Scale invariance (object size in frame) |
| **Occlusion**       | none, 25%, 50% masked               | Tolerance to partial visibility         |
| **Background**      | original, white, dark, noisy        | Background independence                 |
| **Reference Count** | 1, 2, 3, 5 reference images         | Gallery size impact on accuracy         |

### How to Run

Run all robustness experiments:

```bash
python scripts/run_robustness.py
```

Run specific conditions:

```bash
python scripts/run_robustness.py --conditions lighting,angle,distance
```

Custom thresholds and dataset size:

```bash
python scripts/run_robustness.py --thresholds 0.50,0.60,0.70,0.80 --entities 8
```

Full options:

```bash
python scripts/run_robustness.py --help
```

### Output Files

Robustness results are saved to `evaluation/results/robustness/`:

- **`robustness_YYYYMMDD_HHMMSS.json`**: Full nested report with all conditions and levels.
- **`robustness_summary_YYYYMMDD_HHMMSS.csv`**: One row per condition × level with best F1/accuracy.
- **Per-condition CSVs**: `evaluation/results/robustness/<condition>/<level>_*.csv`.
- **`robustness_latest.json`** and **`robustness_summary_latest.csv`**: Most recent run.

### Interpreting Robustness Results

- **Small F1 drop**: The model is robust to that condition (e.g., lighting changes barely affect recognition).
- **Large F1 drop**: The model is sensitive to that condition (e.g., 50% occlusion significantly hurts accuracy).
- Compare the **baseline level** (no perturbation) against each **degraded level** to quantify the impact.
- The **degradation summary** at the bottom of the output shows the worst-case F1 drop per condition.

### Dataset Organization Convention

If you have your own test images for robustness evaluation, place them in:

```
data/evaluation/
├── lighting/
│   ├── normal/
│   ├── dim/
│   ├── bright/
│   └── overexposed/
├── angle/
│   ├── front/
│   ├── rotated_15/
│   ├── rotated_30/
│   └── rotated_45/
├── distance/
│   ├── close/
│   ├── medium/
│   └── far/
├── occlusion/
│   ├── none/
│   ├── partial_25/
│   └── partial_50/
└── background/
    ├── original/
    ├── white/
    ├── dark/
    └── noisy/
```

All image files in `data/evaluation/` are gitignored. See `data/evaluation/README.md` for details.
