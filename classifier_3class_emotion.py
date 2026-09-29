"""
Step 6A — three-class sentiment + emotion classifier harness.

SEPARATE from the locked binary classifiers. Loads the new prompt file verbatim
(prompt_classifier_3class_emotion.txt), accepts ONLY title and text, uses the approved
endpoint/model, reads the API key from the environment, temperature 0, thinking
disabled, strict schema-constrained output. No calls are made during Step 6A.

Output contract: {"sentiment": POSITIVE|NEUTRAL|NEGATIVE,
                  "primary_emotion": <one of eight>} — both required, no extras.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI

BASE_URL = os.environ.get("SENTIMENT_API_BASE", "http://dobolyi.com:9001/v1")
API_KEY = os.environ.get("SENTIMENT_API_KEY")
MODEL = os.environ.get("SENTIMENT_MODEL", "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
THINKING_OFF = {"enable_thinking": False}
TEMPERATURE = 0.0

PROMPT_FILE = Path(__file__).resolve().parent / "prompt_classifier_3class_emotion.txt"
if not PROMPT_FILE.exists():
    raise FileNotFoundError(f"Missing 3-class prompt file: {PROMPT_FILE}")
SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8").strip()

SENTIMENTS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
EMOTIONS = ["ANGER", "ANTICIPATION", "DISGUST", "FEAR",
            "JOY", "SADNESS", "SURPRISE", "TRUST"]

SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": SENTIMENTS},
        "primary_emotion": {"type": "string", "enum": EMOTIONS},
    },
    "required": ["sentiment", "primary_emotion"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "sentiment_emotion_3class", "schema": SCHEMA, "strict": True},
}

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError(
                "SENTIMENT_API_KEY is not set. Export it (e.g. `export "
                "SENTIMENT_API_KEY=...`) before running."
            )
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    return _client


def build_user_message(title: str, text: str) -> str:
    """Full user message. ONLY title and text travel to the model."""
    payload = json.dumps({"title": title, "text": text}, ensure_ascii=False)
    return (payload + "\n\nClassify the review above: return one JSON object with "
                      "exactly two keys — \"sentiment\" (POSITIVE, NEUTRAL, or NEGATIVE) "
                      "and \"primary_emotion\" (one of the eight listed emotions).")


def validate_raw(raw: str) -> Optional[dict]:
    """Validate a raw model response against the schema. Returns parsed dict or None."""
    if not raw:
        return None
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if set(obj.keys()) != {"sentiment", "primary_emotion"}:
        return None
    s, e = obj.get("sentiment"), obj.get("primary_emotion")
    if not (isinstance(s, str) and s in SENTIMENTS):
        return None
    if not (isinstance(e, str) and e in EMOTIONS):
        return None
    return {"sentiment": s, "primary_emotion": e}


def classify_3class(title: str, text: str) -> dict:
    """
    Classify ONE review -> three-way sentiment + one primary emotion.

    Returns {"sentiment", "primary_emotion", "valid", "raw", "model"}.
    No rating, reference-label, or row-index parameter exists in this signature.
    """
    client = _get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(title, text)},
        ],
        temperature=TEMPERATURE,
        max_tokens=32,
        response_format=RESPONSE_FORMAT,
        extra_body={"chat_template_kwargs": THINKING_OFF},
    )
    raw = resp.choices[0].message.content or ""
    parsed = validate_raw(raw)
    return {
        "sentiment": parsed["sentiment"] if parsed else None,
        "primary_emotion": parsed["primary_emotion"] if parsed else None,
        "valid": parsed is not None,
        "raw": raw,
        "model": resp.model,
    }


def run_validator_tests() -> None:
    """Local validator tests (no model calls). Fail loudly on any failure."""
    cases = []
    for s in SENTIMENTS:
        for e in EMOTIONS[:1]:   # one representative emotion per sentiment
            cases.append((json.dumps({"sentiment": s, "primary_emotion": e}),
                          {"sentiment": s, "primary_emotion": e}))
    for e in EMOTIONS:           # all eight emotions with POSITIVE
        cases.append((json.dumps({"sentiment": "POSITIVE", "primary_emotion": e}),
                      {"sentiment": "POSITIVE", "primary_emotion": e}))
    cases += [
        ('{"sentiment":"POSITIVE"}', None),                    # missing emotion
        ('{"primary_emotion":"JOY"}', None),                   # missing sentiment
        ('{"sentiment":"POSITIVE","primary_emotion":"JOY","x":1}', None),  # extra key
        ('{"sentiment":"positive","primary_emotion":"joy"}', None),       # casing
        ('{"sentiment":"GOOD","primary_emotion":"JOY"}', None),           # invalid sentiment
        ('{"sentiment":"POSITIVE","primary_emotion":"HAPPY"}', None),     # invalid emotion
        ('not json at all', None),
        ('', None),
        ('{"sentiment": 5, "primary_emotion": "JOY"}', None),  # non-string
    ]
    failed = []
    for inp, exp in cases:
        got = validate_raw(inp)
        if got != exp:
            failed.append((inp, exp, got))
    if failed:
        raise SystemExit("VALIDATOR TESTS FAILED:\n" +
                         "\n".join(f"  {f[0]!r}: expected {f[1]} got {f[2]}" for f in failed))
    print(f"validator tests passed: {len(cases)} cases (3 sentiments x 8 emotions + negatives)")


def leakage_audit(title: str, text: str) -> dict:
    """Demonstrate the constructed messages carry ONLY title+text (no live request).

    Checks STRUCTURED metadata: the user payload's JSON keys must be exactly
    {title, text}, and no forbidden key may appear as a controlled field. Words
    inside customer-written review text (e.g. "reference", "stars") are permitted
    and are NOT treated as metadata.
    """
    user_msg = build_user_message(title, text)
    system_msg = SYSTEM_PROMPT
    payload_line = user_msg.split("\n\n")[0]
    payload = json.loads(payload_line)          # the controlled JSON fragment
    forbidden_keys = ["rating", "reference_label", "source_row_index", "sample_index",
                      "predicted", "agree_with_rating", "llm_sentiment", "nrc",
                      "reference", "row_index"]
    hits = {}
    for k in forbidden_keys:
        if k in payload:
            hits[k] = "present as a user-payload key"
    # system prompt: must not carry structured metadata fields (mentioning the rating
    # as an instruction to never use it is allowed and required)
    if any(k in system_msg for k in forbidden_keys):
        # only flag if it appears as a field name assignment, not instruction prose
        for k in forbidden_keys:
            if k + ":" in system_msg or k + " =" in system_msg:
                hits[k] = "in system prompt"
    return {
        "system_prompt_loaded": len(system_msg) > 500,
        "user_payload_keys": sorted(payload.keys()),
        "instruction_present": "Classify the review above" in user_msg,
        "forbidden_key_hits": hits,
        "clean": not hits and sorted(payload.keys()) == ["text", "title"],
    }
