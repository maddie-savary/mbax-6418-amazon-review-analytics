"""
Step 5A2b — independent verification of NRC emotion scoring (NO network/model calls).

Reloads the official validated lexicon from data/nrc/, reads title/text from the
LOCKED Step 2 results, and independently recomputes every token count, matched count,
and emotion score. Asserts each saved value is correct and every aggregate matches.
Fails loudly on any difference. Does NOT read step5a1_results.json or any LLM output.
No rating, reference, correctness, or LLM emotion field may appear in NRC result rows.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEX_FILE = (ROOT / "data" / "nrc" / "NRC-Emotion-Lexicon"
            / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
SRC2 = ROOT / "step2_results.json"
OUT = ROOT / "step5a2b_results.json"

EMOTIONS = ["ANGER", "ANTICIPATION", "DISGUST", "FEAR",
            "JOY", "SADNESS", "SURPRISE", "TRUST"]
LOWER_TO_EMOTION = {e.lower(): e for e in EMOTIONS}
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

REQUIRED_FIELDS = {"row_index", "title", "text", "token_count",
                   "matched_token_count", "emotion_scores",
                   "nrc_top_emotions", "nrc_primary_emotion", "nrc_tie"}


class _Extract(HTMLParser):
    # Independent implementation (not imported from the scorer): convert_charrefs=True
    # decodes entities, and separator elements contribute a space so adjacent words
    # never concatenate (e.g. "gift<br />Very" -> "gift Very").
    SEPS = {"br", "p", "div", "li"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.chunks = []

    def handle_data(self, d):
        if d:
            self.chunks.append(d)

    def handle_starttag(self, tag, attrs):
        if tag in self.SEPS:
            self.chunks.append(" ")

    def handle_startendtag(self, tag, attrs):
        if tag in self.SEPS:
            self.chunks.append(" ")

    def text(self):
        return "".join(self.chunks)


def clean_and_tokenize(title: str, text: str) -> list[str]:
    import html
    p = _Extract()
    p.feed(title + "\n" + text)
    p.close()
    s = html.unescape(p.text()).lower()
    return TOKEN_RE.findall(s)


def load_lexicon() -> dict[str, set[str]]:
    mapping = {}
    cats = set()
    with open(LEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 3 or not parts[0]:
                continue
            term, cat, flag = parts
            cats.add(cat.lower())
            if flag != "1" or cat.lower() not in LOWER_TO_EMOTION:
                continue
            mapping.setdefault(term.lower(), set()).add(LOWER_TO_EMOTION[cat.lower()])
    present = {LOWER_TO_EMOTION[c] for c in cats if c in LOWER_TO_EMOTION}
    missing = set(EMOTIONS) - present
    if missing:
        raise RuntimeError(f"lexicon missing emotion categories: {sorted(missing)}")
    return mapping


def recompute_row(rec: dict, lex: dict) -> dict:
    tokens = clean_and_tokenize(rec["title"], rec["text"])
    counts = Counter(tokens)
    scores = {e: 0 for e in EMOTIONS}
    matched = 0
    for token, cnt in counts.items():
        emo = lex.get(token)
        if not emo:
            continue
        matched += cnt
        for e in emo:
            scores[e] += cnt
    mx = max(scores.values()) if any(scores.values()) else 0
    if mx == 0:
        top, primary, tie = [], "NO_MATCH", False
    else:
        top = [e for e in EMOTIONS if scores[e] == mx]
        primary, tie = top[0], len(top) > 1
    return {"token_count": len(tokens), "matched": matched, "scores": scores,
            "top": top, "primary": primary, "tie": tie}


def fail(msg):
    raise SystemExit(f"VERIFY FAIL: {msg}")


def main() -> int:
    data = json.loads(OUT.read_text(encoding="utf-8"))
    s2 = json.loads(SRC2.read_text(encoding="utf-8"))["rows"]
    rows = data["results"]
    lex = load_lexicon()

    # --- identity ---
    if len(rows) != 100:
        fail(f"rows != 100: {len(rows)}")
    idxs = [r["row_index"] for r in rows]
    if len(set(idxs)) != 100 or sorted(idxs) != list(range(100)):
        fail("indices not unique 0..99")
    s2map = {r["row_index"]: r for r in s2}

    # --- fields & invariants ---
    for r in rows:
        if set(r.keys()) != REQUIRED_FIELDS:
            fail(f"row {r['row_index']} keys != required: {set(r.keys())}")
        if r["title"] != s2map[r["row_index"]]["title"] or r["text"] != s2map[r["row_index"]]["text"]:
            fail(f"row {r['row_index']} title/text differs from Step 2")
        if not r["token_count"] >= 0 or not r["matched_token_count"] >= 0:
            fail(f"row {r['row_index']} negative token count")
        if r["matched_token_count"] > r["token_count"]:
            fail(f"row {r['row_index']} matched > token count")
        for e in EMOTIONS:
            if e not in r["emotion_scores"]:
                fail(f"row {r['row_index']} missing score key {e}")
            if not isinstance(r["emotion_scores"][e], int) or r["emotion_scores"][e] < 0:
                fail(f"row {r['row_index']} score {e} not nonneg int")
        if set(r["emotion_scores"].keys()) != set(EMOTIONS):
            fail(f"row {r['row_index']} score keys wrong")

    # --- independent recomputation ---
    for r in rows:
        rc = recompute_row(r, lex)
        if rc["token_count"] != r["token_count"]:
            fail(f"row {r['row_index']} token_count {r['token_count']} != recomputed {rc['token_count']}")
        if rc["matched"] != r["matched_token_count"]:
            fail(f"row {r['row_index']} matched {r['matched_token_count']} != recomputed {rc['matched']}")
        if rc["scores"] != r["emotion_scores"]:
            fail(f"row {r['row_index']} emotion_scores mismatch")
        if rc["top"] != r["nrc_top_emotions"]:
            fail(f"row {r['row_index']} top emotions mismatch")
        if rc["primary"] != r["nrc_primary_emotion"]:
            fail(f"row {r['row_index']} primary mismatch")
        if rc["tie"] != r["nrc_tie"]:
            fail(f"row {r['row_index']} tie mismatch")
        # tie semantics: ties preserve all tied maxima in canonical order, first = primary
        if r["nrc_tie"]:
            mx = max(r["emotion_scores"].values())
            tied = [e for e in EMOTIONS if r["emotion_scores"][e] == mx]
            if r["nrc_top_emotions"] != tied:
                fail(f"row {r['row_index']} tie top list not all tied in canonical order")
            if r["nrc_primary_emotion"] != tied[0]:
                fail(f"row {r['row_index']} tie primary != first tied")
        else:
            if r["nrc_primary_emotion"] != "NO_MATCH":
                mx = max(r["emotion_scores"].values())
                tied = [e for e in EMOTIONS if r["emotion_scores"][e] == mx]
                if len(tied) != 1 or r["nrc_primary_emotion"] != tied[0]:
                    fail(f"row {r['row_index']} non-tie unique max violated")
        # NO_MATCH policy
        if r["nrc_primary_emotion"] == "NO_MATCH":
            if sum(r["emotion_scores"].values()) != 0 or r["nrc_top_emotions"] != [] \
               or r["matched_token_count"] != 0 or r["nrc_tie"] is not False:
                fail(f"row {r['row_index']} NO_MATCH invariant violated")
        else:
            if sum(r["emotion_scores"].values()) == 0:
                fail(f"row {r['row_index']} non-NO_MATCH with all-zero scores")

    # --- aggregates recompute ---
    matched_rows = [r for r in rows if r["matched_token_count"] > 0]
    no_match_rows = [r["row_index"] for r in rows if r["nrc_primary_emotion"] == "NO_MATCH"]
    tie_rows = [r["row_index"] for r in rows if r["nrc_tie"]]
    primary_dist = Counter(r["nrc_primary_emotion"] for r in rows)
    total_tokens = sum(r["token_count"] for r in rows)
    total_matched = sum(r["matched_token_count"] for r in rows)

    m = data["metrics"]
    if m["selected_rows"] != 100 or m["rows_with_matched_emotion_token"] != len(matched_rows):
        fail("selected/matched-row metrics mismatch")
    if m["no_match_rows_count"] != len(no_match_rows):
        fail("no-match count mismatch")
    if abs(m["coverage_percentage_pct"] - round(100*len(matched_rows)/100, 2)) > 1e-9:
        fail("coverage mismatch")
    if m["tie_count"] != len(tie_rows):
        fail("tie count mismatch")
    exp_dist = {e: primary_dist.get(e, 0) for e in EMOTIONS + ["NO_MATCH"]}
    if m["primary_emotion_distribution"] != exp_dist:
        fail("primary distribution mismatch")
    if sum(exp_dist.values()) != 100:
        fail("distribution counts do not sum to 100")
    if m["total_token_count"] != total_tokens or m["total_matched_token_occurrences"] != total_matched:
        fail("token totals mismatch")
    if abs(m["overall_matched_token_rate_pct"] - round(100*total_matched/total_tokens, 2)) > 1e-9:
        fail("matched-token rate mismatch")
    if abs(m["average_matched_tokens_per_review"] - round(total_matched/100, 3)) > 1e-6:
        fail("avg matched mismatch")
    if m["tie_rows"] != tie_rows or sorted(m["no_match_rows"]) != sorted(no_match_rows):
        fail("tie/no-match lists mismatch")

    # --- no LLM / rating / reference / correctness fields ---
    for r in rows:
        for bad in ("llm", "rating", "reference", "correct", "agree", "predict"):
            hit = [k for k in r.keys() if bad in k.lower()]
            if hit:
                fail(f"row {r['row_index']} illegal field(s): {hit}")

    print("STEP 5A2b VERIFICATION PASSED (no network/model calls).")
    print(f"  rows={len(rows)} coverage={m['coverage_percentage_pct']}% "
          f"no_match={len(no_match_rows)} ties={len(tie_rows)}")
    print(f"  tokens={total_tokens} matched={total_matched} "
          f"({m['overall_matched_token_rate_pct']}%) primary_dist={exp_dist}")
    return 0


if __name__ == "__main__":
    main()
