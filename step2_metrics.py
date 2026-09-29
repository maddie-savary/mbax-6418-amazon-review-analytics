"""
Step 2 metrics — recompute complete evaluation metrics from the SAVED predictions.

Reads the authoritative rows in step2_results.json (no model/API calls ever made
here) and stores all requested metrics as NUMERIC values into the JSON's 'metrics'
block. Also prints a readable report.

All values are computed programmatically from the saved rows; nothing is hardcoded.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent / "step2_results.json"
LABELS = ["POSITIVE", "NEGATIVE"]


def f1_from_p_r(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def main() -> int:
    data = json.loads(OUT.read_text(encoding="utf-8"))
    rows = data["rows"]

    # --- selection / validity counts ---
    selected = len(rows)
    valid_rows = [r for r in rows if r["valid"]]
    invalid_rows = [r for r in rows if not r["valid"]]

    # --- distributions (actual/reference, predicted) ---
    ref_dist = Counter(r["reference"] for r in rows)
    pred_dist = Counter(r["predicted"] for r in rows)

    # --- confusion matrix over VALID predictions: actual(reference) rows x predicted cols ---
    cm = {a: {p: 0 for p in LABELS} for a in LABELS}
    for r in valid_rows:
        cm[r["reference"]][r["predicted"]] += 1

    correct = sum(cm[a][a] for a in LABELS)
    mismatches = [r for r in rows if not r["agree_with_rating"]]
    invalid_list = [r["row_index"] for r in invalid_rows]

    # --- per-class metrics (numeric) ---
    per_class = {}
    for a in LABELS:
        tp = cm[a][a]
        fp = sum(cm[x][a] for x in LABELS if x != a)   # other actuals predicted as a
        fn = sum(cm[a][x] for x in LABELS if x != a)   # this class predicted as others
        support = sum(cm[a].values())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support if support else 0.0
        per_class[a] = {
            "support": support,
            "correct": tp,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1_from_p_r(precision, recall), 6),
        }

    # --- majority-class baseline (always-POSITIVE) ---
    pos_support = ref_dist["POSITIVE"]
    majority_baseline_pct = round(100 * pos_support / selected, 2)
    model_acc_pct = round(100 * correct / selected, 2)

    metrics = {
        "selected_rows": selected,
        "valid_predictions": len(valid_rows),
        "invalid_predictions": len(invalid_rows),
        "actual_distribution": {"POSITIVE": ref_dist["POSITIVE"],
                                 "NEGATIVE": ref_dist["NEGATIVE"]},
        "predicted_distribution": {"POSITIVE": pred_dist["POSITIVE"],
                                    "NEGATIVE": pred_dist["NEGATIVE"]},
        "correct_predictions": correct,
        "accuracy_all_rows_pct": model_acc_pct,
        "accuracy_valid_pct": round(100 * correct / len(valid_rows), 2)
        if valid_rows else 0.0,
        "accuracy_as_fraction": {"correct": correct, "total": selected},
        "confusion_matrix": {a: cm[a] for a in LABELS},
        "per_class": per_class,
        "mismatch_count": len(mismatches),
        "mismatch_rows": [{"row_index": r["row_index"], "reference": r["reference"],
                            "predicted": r["predicted"], "rating": r["rating"],
                            "title": r["title"]} for r in mismatches],
        "invalid_rows": invalid_list,
        "majority_class_baseline": {
            "always_positive_accuracy_pct": majority_baseline_pct,
            "model_accuracy_pct": model_acc_pct,
            "improvement_over_baseline_pp": round(model_acc_pct - majority_baseline_pct, 2),
            "note": f"The negative class has only {ref_dist['NEGATIVE']} examples; "
                    "97% overall accuracy alone does not establish strong performance.",
        },
    }

    data["metrics"] = metrics
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # ---- readable report ----
    print(f"Selected rows          : {selected}")
    print(f"Valid predictions      : {len(valid_rows)}")
    print(f"Invalid predictions    : {len(invalid_rows)}")
    print(f"Actual   dist          : POSITIVE={ref_dist['POSITIVE']}, NEGATIVE={ref_dist['NEGATIVE']}")
    print(f"Predicted dist         : POSITIVE={pred_dist['POSITIVE']}, NEGATIVE={pred_dist['NEGATIVE']}")
    print(f"Correct predictions    : {correct}")
    print(f"Accuracy (all rows)    : {metrics['accuracy_all_rows_pct']}%  "
          f"({correct}/{selected})")
    print(f"Accuracy (valid only)  : {metrics['accuracy_valid_pct']}%")
    print("Confusion matrix (actual->predicted):")
    header = "Actual\\Predicted"
    print(f"  {header:20} {'POSITIVE':>8} {'NEGATIVE':>8}")
    for a in LABELS:
        print(f"  {a:20} {cm[a]['POSITIVE']:>8} {cm[a]['NEGATIVE']:>8}")
    for a in LABELS:
        p = per_class[a]
        print(f"{a:8}: support={p['support']} correct={p['correct']} "
              f"precision={p['precision']:.4f} recall={p['recall']:.4f} f1={p['f1']:.4f}")
    print(f"\nMismatches ({len(mismatches)}):")
    for r in mismatches:
        print(f"  row {r['row_index']}: ref={r['reference']} pred={r['predicted']} "
              f"rating={r['rating']} title={r['title']!r}")
    print(f"Invalid rows: {invalid_list}")
    print(f"\nMajority-class baseline (always-POSITIVE): {majority_baseline_pct}%  |  "
          f"model: {model_acc_pct}%  |  improvement: "
          f"{metrics['majority_class_baseline']['improvement_over_baseline_pp']} pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
