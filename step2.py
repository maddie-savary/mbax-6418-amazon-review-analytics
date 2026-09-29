"""
Step 2 — Score the binary classifier against ratings on the FIRST 100 reviews.

Reuses the approved Step 1 `classify(title, text)` from classifier.py. The fixed
sample is exactly the first 100 records in file order of data/Gift_Cards.jsonl.gz,
each with a zero-based row_index (no shuffle / balance / substitution / removal).

RESUMABILITY:
    Completed rows are checkpointed to step2_results.json after each classification.
    On (re)start the script reads that checkpoint and SKIPS row indices that already
    have a completed prediction, so an interrupted run resumes without repeating API
    calls. Running against fully-completed output (rows 0-99) makes ZERO API calls.

    The current 100 saved rows were produced by the ORIGINAL run, before
    raw_response/model were retained, so those rows carry NO raw_response/model key.
    They are left absent (not fabricated). New rows produced by a future run DO
    preserve classify()'s 'raw' and 'model' values, stored as 'raw_response'/'model'.

Data handling:
    - Model request  : has access ONLY to review['title'] and review['text'].
                       classifier.py sends exactly those two fields. Nothing else.
    - Reference label: derived FROM the rating AFTER classification, and stored
                       locally as evaluation data only. Never sent to the model.
    - Reference rule : rating >= 4 -> POSITIVE ; rating < 4 -> NEGATIVE.

Metrics / verification live in separate scripts (step2_metrics.py recomputes and
stores numeric metrics into the JSON; step2_verify.py independently asserts).
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from classifier import classify

DATA = Path(__file__).resolve().parent / "data" / "Gift_Cards.jsonl.gz"
OUT = Path(__file__).resolve().parent / "step2_results.json"
N = 100  # fixed sample size (do not change)


def load_first_n(path: Path, n: int) -> list[dict]:
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= n:
                break
    return records


def reference_label(rating: float) -> str:
    return "POSITIVE" if rating >= 4 else "NEGATIVE"


def load_checkpoint() -> dict[int, dict]:
    """Read previously completed rows keyed by row_index (if any)."""
    if not OUT.exists():
        return {}
    try:
        data = json.loads(OUT.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    rows = data.get("rows", []) if isinstance(data, dict) else []
    return {r["row_index"]: r for r in rows
            if isinstance(r, dict) and "row_index" in r}


def save_checkpoint(rows: list[dict]) -> None:
    """Persist all rows plus a summary, preserving other top-level keys."""
    existing = {}
    if OUT.exists():
        try:
            existing = json.loads(OUT.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing["rows"] = rows
    existing["summary"] = {
        "records": len(rows),
        "reference_pos": sum(1 for r in rows if r["reference"] == "POSITIVE"),
        "reference_neg": sum(1 for r in rows if r["reference"] == "NEGATIVE"),
        "valid": sum(1 for r in rows if r["valid"]),
        "invalid": sum(1 for r in rows if not r["valid"]),
        "agreement": sum(1 for r in rows if r["agree_with_rating"]),
        "accuracy_pct": round(100 * sum(1 for r in rows if r["agree_with_rating"])
                              / len(rows), 2) if rows else 0.0,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def main() -> int:
    records = load_first_n(DATA, N)
    assert len(records) == N, f"expected {N} records, got {len(records)}"

    checkpoint = load_checkpoint()
    completed = {i: r for i, r in checkpoint.items()
                 if r.get("valid") is True}
    print(f"checkpoint: {len(completed)}/{N} rows already completed")

    rows = []
    calls_made = 0
    for idx, rec in enumerate(records):  # idx == zero-based row_index, file order
        if idx in completed:
            rows.append(completed[idx])  # reuse prior result, no API call
            continue

        title, text = rec["title"], rec["text"]
        rating = rec["rating"]
        result = classify(title, text)  # ONLY title+text go to the model
        calls_made += 1
        pred = result["sentiment"]
        valid = result["valid"]
        ref = reference_label(rating)
        row = {
            "row_index": idx,
            "title": title,
            "text": text,
            "rating": rating,                # evaluation-only reference
            "reference": ref,                # derived after classification
            "predicted": pred,
            "valid": valid,
            "agree_with_rating": valid and (pred == ref),
            "raw_response": result.get("raw"),   # preserved for new rows only
            "model": result.get("model"),
        }
        rows.append(row)
        save_checkpoint(sorted(rows, key=lambda x: x["row_index"]))  # incremental
        print(f"  classified row {idx} -> {pred} (valid={valid})")

    rows = sorted(rows, key=lambda x: x["row_index"])
    assert len(rows) == N
    save_checkpoint(rows)
    print(f"\nrecords checkpointed: {len(rows)}/{N}; API calls made this run: {calls_made}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
