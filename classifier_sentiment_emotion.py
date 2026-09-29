"""
Step 5 classifier — binary sentiment + one primary emotion for a single review.

SEPARATE from the Step 1 classifier (`classifier.py`); does not modify it.

Input restriction: sends ONLY review title + text. Never sends row index, rating,
reference label, Step 2 prediction, correctness, NRC data, or any metadata.

Single source of truth: the system prompt is loaded VERBATIM from
`prompt_classifier_sentiment_emotion.txt` (separate from the approved Step 1 prompt).

Schema-constrained output (strict, enforced by the vLLM server):
    {"sentiment": "POSITIVE"|"NEGATIVE",
     "primary_emotion": "ANGER"|"ANTICIPATION"|"DISGUST"|"FEAR"|"JOY"|
                        "SADNESS"|"SURPRISE"|"TRUST"}
    both keys required, additionalProperties false.
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

PROMPT_FILE = Path(__file__).resolve().parent / "prompt_classifier_sentiment_emotion.txt"
if not PROMPT_FILE.exists():
    raise FileNotFoundError(f"Missing Step 5 prompt file: {PROMPT_FILE}")
SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8").strip()

SENTIMENTS = ["POSITIVE", "NEGATIVE"]
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
    "json_schema": {"name": "sentiment_emotion", "schema": SCHEMA, "strict": True},
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
                      "exactly two keys — \"sentiment\" (POSITIVE or NEGATIVE) and "
                      "\"primary_emotion\" (one of the eight listed emotions).")


def validate_raw(raw: str) -> Optional[dict]:
    """Validate raw response against the schema. Returns parsed dict or None."""
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
    s = obj.get("sentiment")
    e = obj.get("primary_emotion")
    if not (isinstance(s, str) and s in SENTIMENTS):
        return None
    if not (isinstance(e, str) and e in EMOTIONS):
        return None
    return {"sentiment": s, "primary_emotion": e}


def classify_sentiment_emotion(title: str, text: str) -> dict:
    """
    Classify ONE review → binary sentiment + one primary emotion.

    Returns {"sentiment", "primary_emotion", "valid": bool, "raw": str, "model": str}.
    Raises immediately if the structured-output request is rejected (no silent fallback).
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
