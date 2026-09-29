"""
Step 6B — classify the locked 150-review balanced three-class sample.

Uses ONLY classifier_3class_emotion.classify_3class(row["title"], row["text"]).
Never sends rating, reference_label, sample_index, source_row_index, previous
predictions, correctness, or any other metadata. Locked config: approved endpoint/
model, key from env, temperature 0, thinking disabled, strict schema output,
one review per request.

CHECKPOINTING:
  - Writes balanced_3class_results.json after every completed response.
  - Records sample-file SHA-256, prompt-file SHA-256, model id, temperature,
    thinking, structured-output, start/update timestamps, completed + failed rows.
  - On restart: verifies saved hashes/settings, skips completed rows, processes
    only missing rows, never silently overwrites a saved prediction, final output
    ordered by sample_index.
  - Transient API errors: recorded as failed/incomplete, no invented label.
  - Schema-invalid responses: preserved raw with valid=false.

Metrics are computed only over completed rows, with explicit denominators.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from classifier_3class_emotion import (
    classify_3class, SENTIMENTS, EMOTIONS, MODEL, THINKING_OFF, TEMPERATURE,
)

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "balanced_sample_3class.json"
PROMPT = ROOT / "prompt_classifier_3class_emotion.txt"
OUT = ROOT / "balanced_3class_results.json"
N = 150
CLASS_ORDER = ["POSITIVE", "NEUTRAL", "NEGATIVE"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_checkpoint() -> dict:
    """Loading rule:
      - file absent        -> empty checkpoint (a genuinely new run) is acceptable.
      - file present, unreadable or malformed JSON -> STOP with a clear error.
        An existing checkpoint is never silently treated as a new run, because that
        could overwrite saved results.
    """
    if not OUT.exists():
        return {}
    if not OUT.is_file():
        raise SystemExit(f"CHECKPOINT ERROR: {OUT} exists but is not a file — refusing to start.")
    try:
        raw = OUT.read_text(encoding="utf-8")
        cp = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"CHECKPOINT ERROR: {OUT} exists but is not valid JSON "
            f"(line {exc.lineno}, col {exc.colno}) — refusing to overwrite saved results. "
            "Fix or remove the checkpoint deliberately before rerunning."
        ) from exc
    except OSError as exc:
        raise SystemExit(
            f"CHECKPOINT ERROR: cannot read {OUT}: {exc} — refusing to overwrite "
            "saved results."
        ) from exc
    if not isinstance(cp, dict):
        raise SystemExit(f"CHECKPOINT ERROR: {OUT} has unexpected top-level type "
                         f"{type(cp).__name__} — refusing to start.")
    return cp


def save_checkpoint(cp: dict) -> None:
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(cp, f, indent=2)


def compute_metrics(rows: list[dict]) -> dict:
    """Metrics over completed rows. Explicit denominators stated in every rate."""
    selected = N
    completed = rows
    valid = [r for r in completed if r["valid"]]
    invalid = [r for r in completed if not r["valid"]]
    failed_missing = [i for i in range(N) if not any(r["sample_index"] == i for r in completed)]

    ref_dist = Counter(r["reference_label"] for r in valid)
    pred_dist = Counter(r["predicted_sentiment"] for r in valid)
    cm = {a: {p: 0 for p in CLASS_ORDER} for a in CLASS_ORDER}
    for r in valid:
        cm[r["reference_label"]][r["predicted_sentiment"]] += 1

    correct = sum(cm[a][a] for a in CLASS_ORDER)
    mismatches = [r for r in valid if not r["correct"]]

    # per-class TP/FP/FN, precision, recall, F1 (over valid predictions)
    per_class = {}
    for a in CLASS_ORDER:
        tp = cm[a][a]
        fp = sum(cm[x][a] for x in CLASS_ORDER if x != a)
        fn = sum(cm[a][x] for x in CLASS_ORDER if x != a)
        support = sum(cm[a].values())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[a] = {"support": support, "tp": tp, "fp": fp, "fn": fn,
                        "precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4)}
    macro_prec = sum(per_class[a]["precision"] for a in CLASS_ORDER) / len(CLASS_ORDER)
    macro_rec = sum(per_class[a]["recall"] for a in CLASS_ORDER) / len(CLASS_ORDER)
    macro_f1 = sum(per_class[a]["f1"] for a in CLASS_ORDER) / len(CLASS_ORDER)

    # six off-diagonal error directions
    error_directions = {}
    for a in CLASS_ORDER:
        for p in CLASS_ORDER:
            if a != p:
                error_directions[f"{a} → {p}"] = cm[a][p]

    # predictions specifically for the 50 reference-NEUTRAL reviews
    neutral_preds = Counter(r["predicted_sentiment"] for r in valid
                            if r["reference_label"] == "NEUTRAL")

    emo_dist = Counter(r["predicted_primary_emotion"] for r in valid)
    emotion_dist = {e: emo_dist.get(e, 0) for e in EMOTIONS}

    return {
        "selected_rows": selected,
        "completed_responses": len(completed),
        "valid_predictions": len(valid),
        "invalid_predictions": len(invalid),
        "failed_or_missing_rows": failed_missing,
        "actual_distribution": {c: ref_dist.get(c, 0) for c in CLASS_ORDER},
        "predicted_distribution": {c: pred_dist.get(c, 0) for c in CLASS_ORDER},
        "correct_predictions": correct,
        "accuracy_over_all_150_pct": round(100 * correct / selected, 2),
        "accuracy_over_valid_pct": round(100 * correct / len(valid), 2) if valid else 0.0,
        "majority_class_baseline_pct": round(100 * 50 / 150, 2),  # 33.33
        "confusion_matrix": cm,
        "per_class": per_class,
        "macro_precision": round(macro_prec, 4),
        "macro_recall_balanced_accuracy_pct": round(100 * macro_rec, 2),
        "macro_f1": round(macro_f1, 4),
        "mismatch_count": len(mismatches),
        "mismatch_rows": [r["sample_index"] for r in sorted(mismatches, key=lambda x: x["sample_index"])],
        "error_directions": error_directions,
        "reference_neutral_prediction_distribution": dict(neutral_preds),
        "llm_primary_emotion_distribution": emotion_dist,
        "denominator_notes": {
            "accuracy_over_all_150": f"{correct} correct over all {selected} selected rows",
            "accuracy_over_valid": f"{correct} correct over {len(valid)} valid responses",
            "per_class": "computed over valid predictions only",
        },
    }


def main() -> int:
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    rows = sample["rows"]
    assert len(rows) == 150
    assert sorted(r["sample_index"] for r in rows) == list(range(150))
    by_idx = {r["sample_index"]: r for r in rows}

    sample_hash = sha256(SAMPLE)
    prompt_hash = sha256(PROMPT)

    cp = load_checkpoint()
    cp_meta = cp.get("run_metadata", {})
    if cp_meta:
        if cp_meta.get("sample_sha256") != sample_hash:
            raise SystemExit("CHECKPOINT SAMPLE HASH MISMATCH — refusing to mix runs")
        if cp_meta.get("prompt_sha256") != prompt_hash:
            raise SystemExit("CHECKPOINT PROMPT HASH MISMATCH — refusing to mix runs")
        if cp_meta.get("model") != MODEL:
            raise SystemExit("CHECKPOINT MODEL MISMATCH")
        # locked settings
        if cp_meta.get("thinking_disabled") != (THINKING_OFF.get("enable_thinking") is False):
            raise SystemExit("CHECKPOINT THINKING SETTING MISMATCH")
        if cp_meta.get("temperature") != TEMPERATURE:
            raise SystemExit("CHECKPOINT TEMPERATURE MISMATCH")
        print("checkpoint verified: hashes/settings match locked configuration")
    else:
        cp = {
            "run_metadata": {
                "sample_file": "balanced_sample_3class.json",
                "sample_sha256": sample_hash,
                "prompt_file": "prompt_classifier_3class_emotion.txt",
                "prompt_sha256": prompt_hash,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "thinking_disabled": THINKING_OFF.get("enable_thinking") is False,
                "structured_output": True,
                "start_timestamp": now_iso(),
                "update_timestamp": now_iso(),
            },
            "rows": [],
            "failed_rows": [],
        }

    done = {r["sample_index"] for r in cp.get("rows", [])}
    print(f"completed so far: {len(done)}/{N}; failed: {cp.get('failed_rows', [])}")

    results = list(cp.get("rows", []))
    result_map = {r["sample_index"]: r for r in results}
    failed = list(cp.get("failed_rows", []))

    api_calls = 0
    for sample_index in range(N):
        if sample_index in result_map:
            # already completed: ensure this index is not listed as failed anymore
            failed = [i for i in failed if i != sample_index]
            continue  # never silently overwrite a saved prediction
        row = by_idx[sample_index]
        try:
            r = classify_3class(row["title"], row["text"])
        except Exception as exc:  # transient API/network error -> record, stop at row
            if sample_index not in failed:
                failed.append(sample_index)
            cp["rows"] = sorted(results, key=lambda x: x["sample_index"])
            cp["failed_rows"] = sorted(set(failed))
            cp["run_metadata"]["update_timestamp"] = now_iso()
            save_checkpoint(cp)
            print(f"API FAILURE at sample_index {sample_index}: {type(exc).__name__}: {exc}")
            raise SystemExit(f"stopped for intentional resume (index {sample_index} incomplete)")
        api_calls += 1
        pred = r["sentiment"]
        emo = r["primary_emotion"]
        valid_flag = r["valid"]
        correct = (valid_flag and pred == row["reference_label"])
        results.append({
            "sample_index": sample_index,
            "source_row_index": row["source_row_index"],
            "title": row["title"],
            "text": row["text"],
            "rating": row["rating"],
            "reference_label": row["reference_label"],
            "predicted_sentiment": pred,
            "predicted_primary_emotion": emo,
            "valid": valid_flag,
            "correct": correct,
            "raw_response": r["raw"],
            "model": r["model"],
        })
        # a formerly failed index that now succeeded is no longer unresolved
        failed = [i for i in failed if i != sample_index]
        result_map = {r["sample_index"]: r for r in results}
        cp["rows"] = sorted(results, key=lambda x: x["sample_index"])
        cp["failed_rows"] = sorted(set(failed))
        cp["run_metadata"]["update_timestamp"] = now_iso()
        save_checkpoint(cp)
        print(f"  row {sample_index:3d} -> {pred}/{emo} valid={valid_flag} correct={correct}")

    # final write: failed_rows contains only indices still unresolved (no completed row)
    failed = [i for i in failed if i not in result_map]
    results = sorted(results, key=lambda x: x["sample_index"])
    metrics = compute_metrics(results)
    cp["rows"] = results
    cp["metrics"] = metrics
    cp["failed_rows"] = sorted(set(failed))
    cp["run_metadata"]["update_timestamp"] = now_iso()
    cp["data_handling"] = {
        "model_request_fields": ["title", "text"],
        "rating": "joined locally after model response for evaluation only; "
                  "never sent to the model",
        "reference_label": "joined locally after model response for evaluation only; "
                           "never sent to the model",
    }
    save_checkpoint(cp)
    print(f"\ncompleted: {len(results)}/{N}; API calls this run: {api_calls}")
    print("metrics:", json.dumps({k: v for k, v in metrics.items()
                                  if k not in ("confusion_matrix", "per_class",
                                               "mismatch_rows", "failed_or_missing_rows")},
                                 indent=1)[:1600])
    return 0


if __name__ == "__main__":
    sys.exit(main())
