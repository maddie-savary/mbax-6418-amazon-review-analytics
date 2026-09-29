"""
Step 6B — independent verification of the balanced three-class run (NO model/network calls).

Reads the locked sample, the run metadata, and the results file; independently
recomputes every metric / invariant and fails loudly on any difference.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "balanced_sample_3class.json"
PROMPT = ROOT / "prompt_classifier_3class_emotion.txt"
RESULTS = ROOT / "balanced_3class_results.json"

CLASS_ORDER = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
EMOTIONS = ["ANGER", "ANTICIPATION", "DISGUST", "FEAR",
            "JOY", "SADNESS", "SURPRISE", "TRUST"]


def fail(msg):
    raise SystemExit(f"VERIFY FAIL: {msg}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    out = json.loads(RESULTS.read_text(encoding="utf-8"))
    meta = out["run_metadata"]
    rows = out["rows"]
    m = out["metrics"]

    # --- hashes match run metadata ---
    if meta["sample_sha256"] != sha256(SAMPLE):
        fail("sample hash mismatch")
    if meta["prompt_sha256"] != sha256(PROMPT):
        fail("prompt hash mismatch")
    if meta["thinking_disabled"] is not True or meta["structured_output"] is not True:
        fail("settings mismatch")
    if meta["temperature"] != 0.0:
        fail("temperature mismatch")

    # --- sample alignment ---
    sample_rows = {r["sample_index"]: r for r in sample["rows"]}
    if len(sample_rows) != 150:
        fail("sample rows != 150")
    if len(rows) != 150:
        fail(f"result rows != 150: {len(rows)}")
    for r in rows:
        s = sample_rows[r["sample_index"]]
        if r["source_row_index"] != s["source_row_index"]:
            fail(f"row {r['sample_index']} source mismatch")
        if r["title"] != s["title"] or r["text"] != s["text"]:
            fail(f"row {r['sample_index']} title/text mismatch")
        if r["rating"] != s["rating"] or r["reference_label"] != s["reference_label"]:
            fail(f"row {r['sample_index']} rating/reference mismatch")

    # --- validity & enum ---
    valid = [r for r in rows if r["valid"]]
    invalid = [r for r in rows if not r["valid"]]
    if len(valid) != 150:
        fail(f"valid != 150: {len(valid)}")
    for r in valid:
        if r["predicted_sentiment"] not in CLASS_ORDER:
            fail(f"row {r['sample_index']} sentiment out of enum")
        if r["predicted_primary_emotion"] not in EMOTIONS:
            fail(f"row {r['sample_index']} emotion out of enum")
        # raw response re-parses to saved prediction
        try:
            obj = json.loads(r["raw_response"])
        except (json.JSONDecodeError, TypeError):
            fail(f"row {r['sample_index']} raw_response not JSON")
        if obj.get("sentiment") != r["predicted_sentiment"]:
            fail(f"row {r['sample_index']} raw sentiment mismatch")
        if obj.get("primary_emotion") != r["predicted_primary_emotion"]:
            fail(f"row {r['sample_index']} raw emotion mismatch")
        if not r.get("model"):
            fail(f"row {r['sample_index']} missing model id")

    # --- recompute everything ---
    ref = Counter(r["reference_label"] for r in valid)
    pred = Counter(r["predicted_sentiment"] for r in valid)
    if ref["POSITIVE"] != 50 or ref["NEUTRAL"] != 50 or ref["NEGATIVE"] != 50:
        fail("actual support not 50/50/50")
    cm = {a: {p: 0 for p in CLASS_ORDER} for a in CLASS_ORDER}
    for r in valid:
        cm[r["reference_label"]][r["predicted_sentiment"]] += 1
    if m["confusion_matrix"] != cm:
        fail("confusion matrix mismatch")
    if m["predicted_distribution"] != {c: pred.get(c, 0) for c in CLASS_ORDER}:
        fail("predicted distribution mismatch")
    if sum(cm[a][p] for a in CLASS_ORDER for p in CLASS_ORDER) != 150:
        fail("confusion cells do not sum to valid predictions")
    correct = sum(cm[a][a] for a in CLASS_ORDER)
    if m["correct_predictions"] != correct:
        fail("correct count mismatch")
    if m["confusion_matrix"]["POSITIVE"]["POSITIVE"] + m["confusion_matrix"]["NEUTRAL"]["NEUTRAL"] + m["confusion_matrix"]["NEGATIVE"]["NEGATIVE"] != correct:
        fail("diagonal != correct")

    # per-class
    for a in CLASS_ORDER:
        tp = cm[a][a]
        fp = sum(cm[x][a] for x in CLASS_ORDER if x != a)
        fn = sum(cm[a][x] for x in CLASS_ORDER if x != a)
        support = sum(cm[a].values())
        prec = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
        rec = round(tp / support, 4) if support else 0.0
        f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0
        p = m["per_class"][a]
        if (p["support"], p["tp"], p["fp"], p["fn"]) != (support, tp, fp, fn):
            fail(f"{a} support/tp/fp/fn mismatch")
        if abs(p["precision"] - prec) > 1e-9 or abs(p["recall"] - rec) > 1e-9 or abs(p["f1"] - f1) > 1e-9:
            fail(f"{a} precision/recall/f1 mismatch")

    macro_prec = sum(m["per_class"][a]["precision"] for a in CLASS_ORDER) / 3
    macro_rec = sum(m["per_class"][a]["recall"] for a in CLASS_ORDER) / 3
    macro_f1 = sum(m["per_class"][a]["f1"] for a in CLASS_ORDER) / 3
    if abs(m["macro_precision"] - round(macro_prec, 4)) > 1e-9:
        fail("macro precision mismatch")
    if abs(m["macro_recall_balanced_accuracy_pct"] - round(100 * macro_rec, 2)) > 1e-9:
        fail("macro recall mismatch")
    if abs(m["macro_f1"] - round(macro_f1, 4)) > 1e-9:
        fail("macro f1 mismatch")

    # error directions (six off-diagonal)
    for a in CLASS_ORDER:
        for p in CLASS_ORDER:
            if a != p:
                key = f"{a} → {p}"
                if m["error_directions"][key] != cm[a][p]:
                    fail(f"error direction {key} mismatch")

    # neutral prediction distribution (50 reference-NEUTRAL rows, all valid)
    neutral_preds = Counter(r["predicted_sentiment"] for r in valid if r["reference_label"] == "NEUTRAL")
    if m["reference_neutral_prediction_distribution"] != dict(neutral_preds):
        fail("neutral prediction distribution mismatch")
    if sum(neutral_preds.values()) != 50:
        fail("neutral prediction counts != 50")

    # emotion distribution
    emo = Counter(r["predicted_primary_emotion"] for r in valid)
    emo_dist = {e: emo.get(e, 0) for e in EMOTIONS}
    if m["llm_primary_emotion_distribution"] != emo_dist:
        fail("emotion distribution mismatch")
    if sum(emo_dist.values()) != 150:
        fail("emotion dist sum != 150")

    # accuracy / baseline / mismatch lists
    if abs(m["accuracy_over_all_150_pct"] - round(100 * correct / 150, 2)) > 1e-9:
        fail("accuracy all mismatch")
    if abs(m["accuracy_over_valid_pct"] - round(100 * correct / len(valid), 2)) > 1e-9:
        fail("accuracy valid mismatch")
    if abs(m["majority_class_baseline_pct"] - 33.33) > 1e-9:
        fail("baseline mismatch")
    mismatch_rows = sorted(r["sample_index"] for r in valid if not r["correct"])
    if m["mismatch_count"] != len(mismatch_rows) or m["mismatch_rows"] != mismatch_rows:
        fail("mismatch list mismatch")
    if m["invalid_predictions"] != 0 or m["failed_or_missing_rows"] != []:
        fail("invalid/failed should be empty")
    if m["completed_responses"] != 150 or m["valid_predictions"] != 150:
        fail("completion counts mismatch")

    # --- no API key in output ---
    blob = RESULTS.read_text(encoding="utf-8")
    if "SENTIMENT_API_KEY" in blob or ("Bearer" in blob):
        fail("credentials found in output")

    print("STEP 6B VERIFICATION PASSED (no network/model calls).")
    print(f"  rows=150 valid=150 invalid=0 failed=[]  correct={correct}/150 "
          f"({round(100*correct/150,2)}%) baseline=33.33%")
    print("  confusion:", {a: cm[a] for a in CLASS_ORDER})
    print("  neutral preds:", dict(neutral_preds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
