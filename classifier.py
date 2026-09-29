"""
Step 1 harness — classify a single Amazon review as POSITIVE or NEGATIVE.

Uses only the review's `title` and `text`. Never sends rating or any other metadata.

Single source of truth for the prompt:
    The system prompt lives in `prompt_classifier.txt` and is loaded VERBATIM from
    that file at runtime (see SYSTEM_PROMPT). That file IS the submitted prompt; the
    code and documentation cannot diverge. The user message carries only the review's
    title and text plus a one-line structural instruction.

Endpoint / model / key come from configuration:
    base_url = $SENTIMENT_API_BASE (default: the class endpoint)
    api_key  = $SENTIMENT_API_KEY  (required — read from env, never hard-coded)
    model    = $SENTIMENT_MODEL    (default: the class model)

Structured (schema-constrained) output:
    response_format={"type":"json_schema", ...} with schema
        {"sentiment": <"POSITIVE"|"NEGATIVE">}   required, no additional properties.
    vLLM enforces this strictly (confirmed empirically). If the server rejects the
    structured request, classify() raises immediately — there is NO silent one-token
    fallback, because the endpoint supports structured JSON output.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI

# ---- configuration (key from environment; URL/model overridable) ----
BASE_URL = os.environ.get("SENTIMENT_API_BASE", "http://dobolyi.com:9001/v1")
API_KEY = os.environ.get("SENTIMENT_API_KEY")
MODEL = os.environ.get("SENTIMENT_MODEL", "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
THINKING_OFF = {"enable_thinking": False}  # tested extra-body setting for this vLLM/Qwen model
TEMPERATURE = 0.0  # deterministic (decision #6)

# ---- single source of truth: the parameterized system prompt is loaded from disk ----
PROMPT_FILE = Path(__file__).resolve().parent / "prompt_classifier.txt"
if not PROMPT_FILE.exists():
    raise FileNotFoundError(
        f"Missing prompt file: {PROMPT_FILE}. The classifier executes the prompt "
        "from this file verbatim (single source of truth)."
    )
SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8").strip()

# ---- schema (decision #1: require 'sentiment', restrict to 2 values, no extras) ----
SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["POSITIVE", "NEGATIVE"]}
    },
    "required": ["sentiment"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "sentiment",
        "schema": SENTIMENT_SCHEMA,
        "strict": True,
    },
}

VALID = {"POSITIVE", "NEGATIVE"}

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError(
                "SENTIMENT_API_KEY is not set. Export it (e.g. `export "
                "SENTIMENT_API_KEY=...`) before running. The key is read from the "
                "environment only and is never committed or written to output files."
            )
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    return _client


def build_user_message(title: str, text: str) -> str:
    """The full user message. ONLY title and text travel to the model."""
    payload = json.dumps({"title": title, "text": text}, ensure_ascii=False)
    return (payload + "\n\nClassify the review above as POSITIVE or NEGATIVE. "
                      "Reply with exactly one JSON object: {\"sentiment\": \"POSITIVE\"} "
                      "or {\"sentiment\": \"NEGATIVE\"}.")


def validate_raw(raw: str) -> Optional[dict]:
    """Validate a raw model response against the schema. Returns parsed dict or None."""
    if not raw:
        return None
    text = raw.strip()
    # trim surrounding fences / prose if a non-conforming server returns any
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    # must be a dict, have exactly our schema's shape
    if not isinstance(obj, dict) or set(obj.keys()) != {"sentiment"}:
        return None
    val = obj.get("sentiment")
    if isinstance(val, str) and val in VALID:
        return {"sentiment": val}
    return None


def classify(title: str, text: str) -> dict:
    """
    Classify ONE review using strict schema-constrained JSON output.

    Returns: {"sentiment": "POSITIVE"|"NEGATIVE", "valid": bool, "raw": str}
    - 'valid' True iff the response parsed AND obeyed the schema.
    - If the server rejects the structured request, this raises immediately
      (no silent fallback) so a misconfiguration is loud, not hidden.
    """
    client = _get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(title, text)},
        ],
        temperature=TEMPERATURE,
        max_tokens=16,
        response_format=RESPONSE_FORMAT,
        extra_body={"chat_template_kwargs": THINKING_OFF},
    )
    raw = resp.choices[0].message.content or ""
    model_id = resp.model

    parsed = validate_raw(raw)
    return {
        "sentiment": parsed["sentiment"] if parsed else None,
        "valid": parsed is not None,
        "model": model_id,
        "raw": raw,
    }
