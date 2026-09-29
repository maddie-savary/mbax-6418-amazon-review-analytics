"""
Step 5A1 — independent verification of the saved predictions (NO API calls).

Reads ONLY step5a1_results.json and step2_results.json and independently recomputes
every metric / invariant, asserting loudly on any failure.

Important note on field names: this file uses the normalized Step 5A1 output fields
(llm_sentiment, llm_primary_emotion, raw_response). The restriction "no rating or
rating-derived fields appear as structured fields" refers to structured metadata and
model-request construction; the words 'rating' or 'star' may legitimately appear
inside authentic review text (title/text are review content, not structured fields).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "step5a1_results.json"
SRC2 = ROOT / "step2_results.json"
N = 100

SENTINEL = ["POSITIVE", "NEGATIVE"]
EMOTIONS = ["ANGER", "ANTICIPATION", "DISGUST", "FEAR",
            "JOY", "SADNESS", "SURPRISE", "TRUST"]


def fail(msg: str) -> None:
    print(f"VERIFY FAIL: {msg}")
    raise SystemExit(1)


def main() -> int:
    data = json.loads(OUT.read_text(encoding="utf-8"))
    s2 = json.loads(SRC2.read_text(encoding="utf-8"))["rows"]
    rows = data["results"]
    metrics = data["metrics"]

    # --- row identity ---
    if len(rows) != N:
        fail(f"expected {N} rows, got {len(rows)}")
    idxs = [r["row_index"] for r in rows]
    if len(set(idxs)) != N or sorted(idxs) != list(range(N)):
        fail("row indices not exactly 0..99 and unique")

    valid = [r for r in rows if r["valid"]]
    invalid = [r for r in rows if not r["valid"]]
    if len(valid) != N or len(invalid) != 0:
        fail(f"expected 100 valid / 0 invalid, got {len(valid)}/{len(invalid)}")

    # --- enum validity ---
    for r in valid:
        if r["llm_sentiment"] not in SENTINEL:
            fail(f"sentiment not in enum: {r['llm_sentiment']}")
        if r["llm_primary_emotion"] not in EMOTIONS:
            fail(f"emotion not in enum: {r['llm_primary_emotion']}")

    # --- distributions sum + saved match recomputed ---
    sent_cnt = Counter(r["llm_sentiment"] for r in valid)
    emo_cnt = Counter(r["llm_primary_emotion"] for r in valid)
    if sum(sent_cnt.values()) != N:
        fail(f"sentiment counts sum {sum(sent_cnt.values())} != 100")
    if sum(emo_cnt.values()) != N:
        fail(f"emotion counts sum {sum(emo_cnt.values())} != 100")
    saved_sent = metrics["llm_sentiment_distribution"]
    saved_emo = metrics["llm_primary_emotion_distribution"]
    if not all(e in saved_emo for e in EMOTIONS):
        fail("saved emotion distribution missing an enum category (incl. zero-count)")
    recomputed_sent = {s: sent_cnt.get(s, 0) for s in SENTINEL}
    recomputed_emo = {e: emo_cnt.get(e, 0) for e in EMOTIONS}
    if saved_sent != recomputed_sent:
        fail(f"sentiment dist mismatch: {saved_sent} vs {recomputed_sent}")
    if saved_emo != recomputed_emo:
        fail(f"emotion dist mismatch: {saved_emo} vs {recomputed_emo}")

    # --- raw responses re-parse to same sentiment + emotion ---
    for r in valid:
        try:
            obj = json.loads(r["raw_response"])
        except (json.JSONDecodeError, TypeError):
            fail(f"raw_response not valid JSON for row {r['row_index']}")
        if obj.get("sentiment") != r["llm_sentiment"]:
            fail(f"raw sentiment mismatch row {r['row_index']}")
        if obj.get("primary_emotion") != r["llm_primary_emotion"]:
            fail(f"raw emotion mismatch row {r['row_index']}")

    # --- model ids present ---
    if not all(r.get("model") for r in rows):
        fail("some rows missing 'model'")

    # --- Step 2 agreement (recompute independently) ---
    s2map = {r["row_index"]: r["predicted"] for r in s2}
    disagree = sorted(r["row_index"] for r in valid if s2map[r["row_index"]] != r["llm_sentiment"])
    agreements = len(valid) - len(disagree)
    denom = len(valid)
    pct = round(100 * agreements / denom, 2) if denom else 0.0
    if agreements != N:
        fail(f"expected 100/100 Step2 agreement, got {agreements}/{denom}")
    agree_m = metrics["step2_sentiment_agreement"]
    if agree_m["agreements"] != agreements:
        fail("saved agreement count != recomputed")
    if agree_m["eligible_denominator"] != denom:
        fail("saved denominator != recomputed")
    if agree_m["percentage_pct"] != pct:
        fail("saved percentage != recomputed")
    saved_dis = sorted(d["row_index"] for d in agree_m["disagreement_rows"])
    if saved_dis != disagree:
        fail(f"saved disagreement list mismatch: {saved_dis} vs {disagree}")

    # --- no rating/reference as STRUCTURED fields in step5a1 rows ---
    ALLOWED = {"row_index", "title", "text", "llm_sentiment",
               "llm_primary_emotion", "valid", "raw_response", "model"}
    for r in rows:
        if set(r.keys()) != ALLOWED:
            fail(f"row {r['row_index']} keys != the eight required: {set(r.keys())}")
        if "rating" in r or "reference" in r or "predict" in r.keys():
            fail(f"row {r['row_index']} carries evaluation metadata as a field")
    # (title/text may contain the words 'rating'/'star' — that is review content, allowed)

    # --- title and text exactly match the locked Step 2 rows ---
    s2map_full = {r["row_index"]: r for r in s2}
    for r in rows:
        s = s2map_full[r["row_index"]]
        if r["title"] != s["title"]:
            fail(f"row {r['row_index']} title mismatch vs Step 2")
        if r["text"] != s["text"]:
            fail(f"row {r['row_index']} text mismatch vs Step 2")
        if not r.get("text"):
            fail(f"row {r['row_index']} text missing/empty")

    print("STEP 5A1 VERIFICATION PASSED (no API calls).")
    print(f"  rows={len(rows)} indices 0..99 valid={len(valid)} invalid={len(invalid)}")
    print(f"  sentiment dist: {saved_sent}")
    print(f"  emotion dist:   {saved_emo}")
    print(f"  raw re-parse:   all {N} match saved sentiment+emotion")
    print(f"  Step2 agreement: {agreements}/{denom} ({pct}%), disagreements={disagree}")
    print("  model ids present: True; no structured rating/reference fields in rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
