"""
Step 2 independent verification — re-derives every metric from the SAVED rows and
asserts the persisted metrics match, as well as the raw integrity invariants.

This is a SECOND, independent code path (does not import step2_metrics or step2;
it reads only step2_results.json). It fails loudly (via assertion errors) if anything
is wrong. No model/API calls are made.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "step2_results.json"
LABELS = ["POSITIVE", "NEGATIVE"]
N = 100


def main() -> int:
    data = json.loads(OUT.read_text(encoding="utf-8"))
    rows = data["rows"]
    metrics = data["metrics"]

    # --- invariant assertions ---
    indices = sorted(r["row_index"] for r in rows)
    assert len(indices) == 100, f"expect 100 rows, got {len(indices)}"
    assert len(set(indices)) == 100, "row indices not unique"
    assert indices == list(range(100)), f"indices not exactly 0..99: {indices[:5]}...{indices[-3:]}"

    valid = [r for r in rows if r["valid"]]
    invalid = [r for r in rows if not r["valid"]]
    assert len(valid) == 100, f"expect 100 valid, got {len(valid)}"
    assert len(invalid) == 0, f"expect 0 invalid, got {len(invalid)}"

    # --- recompute distributions / confusion / per-class from scratch ---
    from collections import Counter
    ref = Counter(r["reference"] for r in rows)
    pred = Counter(r["predicted"] for r in rows)
    assert ref["POSITIVE"] + ref["NEGATIVE"] == 100, "actual supports must sum to 100"
    assert pred["POSITIVE"] + pred["NEGATIVE"] == 100, "predicted counts must sum to 100"

    cm = {a: {p: 0 for p in LABELS} for a in LABELS}
    for r in valid:
        cm[r["reference"]][r["predicted"]] += 1
    assert sum(cm[a][p] for a in LABELS for p in LABELS) == 100, "confusion cells != 100"
    correct = sum(cm[a][a] for a in LABELS)
    assert correct == 97, f"diagonal should sum to 97, got {correct}"

    mismatches = [r for r in rows if not r["agree_with_rating"]]
    assert len(mismatches) == 3, f"expect 3 mismatches, got {len(mismatches)}"
    assert len(invalid) == 0, f"invalid-response list should be empty, got {len(invalid)}"

    # --- assert persisted metrics equal recomputed metrics ---
    def f1(p, r):
        return 2 * p * r / (p + r) if (p + r) else 0.0

    assert metrics["selected_rows"] == 100
    assert metrics["valid_predictions"] == 100
    assert metrics["invalid_predictions"] == 0
    assert metrics["actual_distribution"] == {"POSITIVE": ref["POSITIVE"],
                                               "NEGATIVE": ref["NEGATIVE"]}, "actual dist"
    assert metrics["predicted_distribution"] == {"POSITIVE": pred["POSITIVE"],
                                                  "NEGATIVE": pred["NEGATIVE"]}, "pred dist"
    assert metrics["correct_predictions"] == 97
    assert abs(metrics["accuracy_all_rows_pct"] - 97.0) < 1e-9
    assert abs(metrics["accuracy_valid_pct"] - 97.0) < 1e-9
    assert metrics["confusion_matrix"] == cm, "confusion matrix mismatch"
    for a in LABELS:
        p = metrics["per_class"][a]
        tp = cm[a][a]
        fp = sum(cm[x][a] for x in LABELS if x != a)
        fn = sum(cm[a][x] for x in LABELS if x != a)
        support = sum(cm[a].values())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / support if support else 0.0
        assert p["support"] == support, f"{a} support"
        assert p["correct"] == tp, f"{a} correct"
        assert abs(p["precision"] - prec) < 1e-6, f"{a} precision"
        assert abs(p["recall"] - rec) < 1e-6, f"{a} recall"
        assert abs(p["f1"] - f1(prec, rec)) < 1e-6, f"{a} f1"
    assert metrics["mismatch_count"] == len(mismatches) == 3, "mismatch count"
    assert metrics["invalid_rows"] == [], "invalid list must be empty"
    assert abs(metrics["majority_class_baseline"]["always_positive_accuracy_pct"] - 93.0) < 1e-9
    assert abs(metrics["majority_class_baseline"]["model_accuracy_pct"] - 97.0) < 1e-9
    assert abs(metrics["majority_class_baseline"]["improvement_over_baseline_pp"] - 4.0) < 1e-9

    # --- print evidence ---
    print("INDEPENDENT VERIFICATION PASSED (no API calls).")
    print(f"  rows={len(indices)} indices={indices[0]}..{indices[-1]} unique="
          f"{len(set(indices))} valid={len(valid)} invalid={len(invalid)}")
    print(f"  actual  = {dict(ref)}  predicted = {dict(pred)}")
    print("  confusion (actual row x predicted col): newline-inline table")
    print("         POS   NEG")
    for a in LABELS:
        print(f"    {a:8}{cm[a]['POSITIVE']:>5}{cm[a]['NEGATIVE']:>5}")
    print(f"  correct={correct} mismatches={len(mismatches)} invalid={len(invalid)}")
    print("  metrics.accuracy_all_rows_pct =", metrics["accuracy_all_rows_pct"])
    print("  per-class O:",
          metrics["per_class"]["POSITIVE"])
    print("  per-class N:",
          metrics["per_class"]["NEGATIVE"])
    print("  baseline: always-POSITIVE", metrics["majority_class_baseline"]["always_positive_accuracy_pct"],
          "% | model", metrics["majority_class_baseline"]["model_accuracy_pct"],
          "% | improvement", metrics["majority_class_baseline"]["improvement_over_baseline_pp"], "pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
