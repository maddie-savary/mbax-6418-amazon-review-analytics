"""
Step 5A1 — produce and verify LLM sentiment-and-emotion predictions for rows 0-99.

Uses ONLY the following from step2_results.json: row_index, title, text.
No rating, reference, Step 2 prediction, correctness, or other metadata is read from
the file or sent to the model. For each review a fresh LLM request returns:
    - binary sentiment: POSITIVE | NEGATIVE
    - one primary emotion: ANGER|ANTICIPATION|DISGUST|FEAR|JOY|SADNESS|SURPRISE|TRUST
via the separate Step 5 prompt and classifier (prompt_classifier_sentiment_emotion.txt
and classifier_sentiment_emotion.py). The approved Step 1 prompt/classifier are not used
or modified.

RESUMABILITY:
    Completed VALID rows are checkpointed to step5a1_results.json after each new
    response. On (re)start the script loads prior completed valid rows, skips their
    row indices, preserves held INVALID responses explicitly, and never repeats
    completed valid API calls. Running against fully-completed output (rows 0-99 all
    valid) makes ZERO API calls and does not require the API key.

Field names (after normalization): row_index, title, text, llm_sentiment,
    llm_primary_emotion, valid, raw_response, model.
    The Step 2 comparison runs ONLY as local post-processing after classification;
    a Step 2 prediction never enters a model request.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from classifier_sentiment_emotion import (
    classify_sentiment_emotion, SENTIMENTS, EMOTIONS,
)

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "step2_results.json"
OUT = ROOT / "step5a1_results.json"
N = 100

EMOTION_ORDER = ["ANGER", "ANTICIPATION", "DISGUST", "FEAR",
                 "JOY", "SADNESS", "SURPRISE", "TRUST"]
SENTIMENT_ORDER = ["POSITIVE", "NEGATIVE"]


def read_source_rows() -> list[dict]:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rows = data["rows"]
    assert len(rows) == N and {r["row_index"] for r in rows} == set(range(N)), \
        "step2_results.json rows are not exactly 0..99"
    return rows


def load_checkpoint() -> dict[int, dict]:
    """Read previously completed rows keyed by row_index (any prior field names)."""
    if not OUT.exists():
        return {}
    try:
        data = json.loads(OUT.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    rows = data.get("results", []) if isinstance(data, dict) else []
    cp = {}
    for r in rows:
        if not isinstance(r, dict) or "row_index" not in r:
            continue
        idx = r["row_index"]
        cp[idx] = {
            "row_index": idx,
            "title": r.get("title", ""),
            "text": r.get("text", ""),
            # accept new names or fall back to the legacy names (sentiment/...)
            "llm_sentiment": r.get("llm_sentiment", r.get("sentiment")),
            "llm_primary_emotion": r.get("llm_primary_emotion", r.get("primary_emotion")),
            "valid": r.get("valid"),
            "raw_response": r.get("raw_response", r.get("raw")),
            "model": r.get("model"),
        }
    return cp


def compute_metrics(rows: list[dict], src_rows: list[dict]) -> dict:
    valid = [r for r in rows if r["valid"]]
    invalid = [r for r in rows if not r["valid"]]

    sent_count = Counter(r["llm_sentiment"] for r in valid)
    emo_count = Counter(r["llm_primary_emotion"] for r in valid)

    # Step 2 agreement (local post-processing vs locked Step 2 predictions)
    s2 = {r["row_index"]: r["predicted"] for r in src_rows}
    disagreements = []
    for r in sorted(valid, key=lambda x: x["row_index"]):
        if s2[r["row_index"]] != r["llm_sentiment"]:
            disagreements.append({
                "row_index": r["row_index"],
                "step2_predicted": s2[r["row_index"]],
                "llm_sentiment": r["llm_sentiment"],
            })
    agreements = len(valid) - len(disagreements)
    denom = len(valid)
    pct = round(100 * agreements / denom, 2) if denom else 0.0

    return {
        "selected_rows": len(rows),
        "valid_predictions": len(valid),
        "invalid_predictions": len(invalid),
        "llm_sentiment_distribution": {s: sent_count.get(s, 0) for s in SENTIMENT_ORDER},
        "llm_primary_emotion_distribution": {e: emo_count.get(e, 0) for e in EMOTION_ORDER},
        "step2_sentiment_agreement": {
            "agreements": agreements,
            "eligible_denominator": denom,
            "percentage_pct": pct,
            "agreement_fraction": {"agreements": agreements, "eligible": denom},
            "disagreement_rows": disagreements,
        },
        "model_configuration": {
            "prompt_filename": "prompt_classifier_sentiment_emotion.txt",
            "model_identifier": "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
            "temperature": 0.0,
            "thinking_disabled": True,
            "structured_output": True,
        },
    }


def save_results(rows: list[dict], src_rows: list[dict]) -> None:
    rows.sort(key=lambda r: r["row_index"])
    metrics = compute_metrics(rows, src_rows)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "note": "Step 5A1 LLM sentiment+emotion predictions for rows 0-99. "
                    "Only title and text were sent to the model. No rating/reference/"
                    "Step2 prediction/metadata sent or stored. No API key.",
            "sample": "rows 0..99 from step2_results.json (row_index, title, text only)",
            "sentiment_enum": SENTIMENTS,
            "primary_emotion_enum": EMOTIONS,
            "metrics": metrics,
            "results": rows,
        }, f, indent=2)


def main() -> int:
    src_rows = read_source_rows()
    src_titles = {r["row_index"]: r["title"] for r in src_rows}
    src_texts = {r["row_index"]: r["text"] for r in src_rows}

    # state keyed by row_index, seeded from the checkpoint. EVERY existing row —
    # valid or invalid — is preserved and auto-skipped; only indices entirely
    # absent from the checkpoint trigger an API call.
    state = load_checkpoint()

    # guard against a legacy checkpoint missing text: fill from source rows
    for idx in range(N):
        if idx in state and not state[idx].get("text"):
            assert state[idx]["title"] == src_titles[idx], f"title mismatch row {idx}"
            state[idx]["text"] = src_texts[idx]

    api_calls = 0
    to_process = [i for i in range(N) if i not in state]
    print(f"resume state: {len(state)}/{N} rows already present "
          f"(valid={sum(1 for r in state.values() if r['valid'] is True)}, "
          f"invalid={sum(1 for r in state.values() if r['valid'] is not True)}); "
          f"missing={len(to_process)}")

    for idx in to_process:  # only absent indices -> new API call
        rec = src_rows[idx]
        title, text = rec["title"], rec["text"]
        r = classify_sentiment_emotion(title, text)  # ONLY title+text go to model
        api_calls += 1
        state[idx] = {
            "row_index": idx,
            "title": title,
            "text": text,
            "llm_sentiment": r["sentiment"],
            "llm_primary_emotion": r["primary_emotion"],
            "valid": r["valid"],
            "raw_response": r["raw"],
            "model": r["model"],
        }
        # save the FULL currently known state without dropping later existing rows
        save_results([state[i] for i in range(N) if i in state], src_rows)
        print(f"  row {idx} -> {r['sentiment']} / {r['primary_emotion']} (valid={r['valid']})")

    # pad any still-absent indices (shouldn't happen) so the write is always full 0..99
    for idx in range(N):
        if idx not in state:
            state[idx] = {
                "row_index": idx,
                "title": src_titles[idx],
                "text": src_texts[idx],
                "llm_sentiment": None,
                "llm_primary_emotion": None,
                "valid": False,
                "raw_response": None,
                "model": None,
            }

    final_rows = [state[i] for i in range(N)]
    save_results(final_rows, src_rows)
    print(f"\nrecords: {len(final_rows)}/{N}; API calls this run: {api_calls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
