"""
Step 5A2b — score the same 100 reviews with the official NRC Emotion Lexicon.

Independent of the LLM method: reads ONLY row_index, title, text from
step2_results.json. Does NOT read step5a1_results.json. No rating, reference,
prediction, correctness, or LLM emotion is used. No model/API calls.

Method (documented choices / limitations):
  - Per review, combine title + "\\n" + text.
  - Remove HTML tags with Python's standard library (html.parser) and decode HTML
    entities via html.unescape.
  - Lowercase, then tokenize with the exact rule: r"[a-z]+(?:'[a-z]+)?"
  - Count EVERY token occurrence (repeats counted).
  - NO stemming, lemmatization, stop-word removal, negation handling, or synonym
    expansion (by spec). This is a pure word-count method: it does NOT understand
    context, sarcasm, phrasal meaning, or negation.

Lexicon: in-memory map term -> set(emotion categories with flag==1), ignoring
   flag==0 records and the positive/negative sentiment categories; words associated
   with multiple emotions keep all of them. The eight emotion categories are
   validated before scoring.

Scoring: each token occurrence adds one to every associated emotion (multiple
   emotions => add one to each). matched_token_count counts each token occurrence
   that matched >=1 emotion once, regardless of how many emotions it fired.

Tie / no-match policy (deterministic, canonical order):
   ANGER, ANTICIPATION, DISGUST, FEAR, JOY, SADNESS, SURPRISE, TRUST
   - unique max      -> top=[that], primary=that, tie=false
   - shared max      -> top=all tied in canonical order, primary=first, tie=true
   - all zeros       -> top=[], primary="NO_MATCH", tie=false, matched=0
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
EMOTION_SET = set(EMOTIONS)
LOWER_TO_EMOTION = {e.lower(): e for e in EMOTIONS}
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


class _TextExtractor(HTMLParser):
    """Extracts plain text from HTML.

    With convert_charrefs=True, HTMLParser itself decodes character/entity references
    (e.g. &#34; -> ", &amp; -> &, &#39; -> '). Separator elements (br, p, div, li)
    contribute a space so adjacent words do not concatenate (e.g. "gift<br />Very"
    -> "gift Very", not "giftVery"). Both <br> and <br /> forms are handled via
    handle_starttag / handle_startendtag.
    """

    SEPARATORS = {"br", "p", "div", "li"}

    def __init__(self):
        super().__init__(convert_charrefs=True)  # decode entities/charrefs
        self.chunks = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SEPARATORS:
            self.chunks.append(" ")

    def handle_startendtag(self, tag, attrs):
        if tag in self.SEPARATORS:
            self.chunks.append(" ")

    def handle_data(self, data):
        if data:
            self.chunks.append(data)

    def text(self) -> str:
        return "".join(self.chunks)


def strip_html_and_unescape(s: str) -> str:
    p = _TextExtractor()
    p.feed(s)
    p.close()
    # defensive second pass: unescape any entities the parser left as tokens
    return html_unescape(p.text())


def html_unescape(s: str) -> str:
    import html
    return html.unescape(s)


def clean(text: str) -> str:
    """Title+text -> lowercase plain text (tags stripped, entities decoded, spaced)."""
    joined = text  # caller joins title + "\n" + text
    stripped = strip_html_and_unescape(joined)
    return stripped.lower()


def tokenize(lowered_text: str) -> list[str]:
    """Deterministic token rule. Returns every occurrence (repeats counted)."""
    return TOKEN_RE.findall(lowered_text)


def clean_and_tokenize(title: str, text: str) -> list[str]:
    """title + "\\n" + text -> lowercased spaced plain text -> token occurrences."""
    return tokenize(clean(title + "\n" + text))


def load_lexicon() -> dict[str, set[str]]:
    """term -> set of emotion names (flag==1, emotions only). Validation included."""
    mapping: dict[str, set[str]] = {}
    cats_seen = set()
    with open(LEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.endswith("\r"):
                line = line[:-1]
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 3 or not parts[0]:
                continue
            term, cat, flag = parts
            cats_seen.add(cat.lower())
            if flag != "1":
                continue
            if cat.lower() not in LOWER_TO_EMOTION:
                continue  # ignore positive/negative and anything else
            emotion = LOWER_TO_EMOTION[cat.lower()]
            mapping.setdefault(term.lower(), set()).add(emotion)
    present_emotions = {LOWER_TO_EMOTION[c] for c in cats_seen
                        if c in LOWER_TO_EMOTION}
    missing = EMOTION_SET - present_emotions
    if missing:
        raise RuntimeError(f"lexicon missing emotion categories: {sorted(missing)}")
    return mapping


def pick_primary(scores: dict[str, int]) -> tuple[list[str], str, bool]:
    """Returns (top_emotions_in_canonical_order, primary, tie)."""
    mx = max(scores.values()) if any(scores.values()) else 0
    if mx == 0:
        return [], "NO_MATCH", False
    tied = [e for e in EMOTIONS if scores[e] == mx]
    if len(tied) == 1:
        return tied, tied[0], False
    return tied, tied[0], True


def clean_self_test() -> None:
    """Explicit cleaning/tokenization regression tests (must fail on concatenation)."""
    cases = [
        ("Gift<br />Very easy", ["gift", "very", "easy"]),          # <br /> separates
        ("Gift<br>Very easy", ["gift", "very", "easy"]),            # <br> form
        ("One &amp; two", ["one", "two"]),                           # entity decoded
        ("It&#39;s great", ["it's", "great"]),                       # charref decoded
        ("<p>Good</p><p>gift</p>", ["good", "gift"]),                # p separators
        ("<div>nice</div><li>card</li>", ["nice", "card"]),          # div/li separators
        ("Plain text, no html here", ["plain", "text", "no", "html", "here"]),
    ]
    failed = []
    for raw, expected in cases:
        toks = clean_and_tokenize("", raw)
        if toks != expected:
            failed.append((raw, expected, toks))
    if failed:
        raise SystemExit(
            "CLEANING/SELF-TEST FAILURE:\n" +
            "\n".join(f"  {f[0]!r}: expected {f[1]} got {f[2]}" for f in failed))
    print("cleaning self-test: ", len(cases), "cases passed")


def main() -> int:
    clean_self_test()  # fail loudly before rescoring if cleaning regresses
    data = json.loads(SRC2.read_text(encoding="utf-8"))
    s2rows = data["rows"]
    assert len(s2rows) == 100 and [r["row_index"] for r in s2rows] == list(range(100))

    lexicon = load_lexicon()

    results = []
    for rec in s2rows:
        idx = rec["row_index"]
        tokens = clean_and_tokenize(rec["title"], rec["text"])
        counts = Counter(tokens)

        scores = {e: 0 for e in EMOTIONS}
        matched_occurrences = 0
        for token, cnt in counts.items():
            emo = lexicon.get(token)
            if not emo:
                continue
            matched_occurrences += cnt           # one matched token occurrence per token
            for e in emo:
                scores[e] += cnt                  # add one per associated emotion occurrence

        top, primary, tie = pick_primary(scores)
        results.append({
            "row_index": idx,
            "title": rec["title"],
            "text": rec["text"],
            "token_count": len(tokens),
            "matched_token_count": matched_occurrences,
            "emotion_scores": scores,
            "nrc_top_emotions": top,
            "nrc_primary_emotion": primary,
            "nrc_tie": tie,
        })

    results.sort(key=lambda r: r["row_index"])

    # ---- aggregates ----
    matched_rows = [r for r in results if r["matched_token_count"] > 0]
    no_match_rows = [r for r in results if r["nrc_primary_emotion"] == "NO_MATCH"]
    tie_rows = [r for r in results if r["nrc_tie"]]
    total_tokens = sum(r["token_count"] for r in results)
    total_matched = sum(r["matched_token_count"] for r in results)
    primary_dist = Counter(r["nrc_primary_emotion"] for r in results)

    metrics = {
        "selected_rows": len(results),
        "rows_with_matched_emotion_token": len(matched_rows),
        "no_match_rows_count": len(no_match_rows),
        "coverage_percentage_pct": round(100 * len(matched_rows) / len(results), 2),
        "tie_count": len(tie_rows),
        "primary_emotion_distribution": {
            e: primary_dist.get(e, 0) for e in EMOTIONS + ["NO_MATCH"]
        },
        "total_token_count": total_tokens,
        "total_matched_token_occurrences": total_matched,
        "overall_matched_token_rate_pct":
            round(100 * total_matched / total_tokens, 2) if total_tokens else 0.0,
        "average_matched_tokens_per_review":
            round(total_matched / len(results), 3) if results else 0.0,
        "tie_rows": [r["row_index"] for r in tie_rows],
        "no_match_rows": [r["row_index"] for r in no_match_rows],
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "note": "Step 5A2b NRC Emotion Lexicon scoring for rows 0-99. "
                    "Independent of the LLM method; no rating/reference/prediction/LLM "
                    "emotion used. No API key. Word-count method; does not understand "
                    "context, sarcasm, or negation. NRC positive/negative sentiment "
                    "categories ignored.",
            "method": {
                "input_fields": ["row_index", "title", "text"],
                "text_combination": "title + '\\n' + text",
                "html_cleanup": "Python html.parser strips tags; html.unescape decodes "
                                "entities",
                "token_rule": r"[a-z]+(?:'[a-z]+)?",
                "lowercase": True,
                "count_occurrences": True,
                "no_stem_no_lemmatize_no_stopword_no_negation_no_synonyms": True,
                "multi_emotion_tokens": "add one to every associated emotion",
                "tie_rule": "canonical order: ANGER, ANTICIPATION, DISGUST, FEAR, JOY, "
                            "SADNESS, SURPRISE, TRUST; first tied = primary",
                "no_match_policy": "all-zero scores -> primary=NO_MATCH, top=[], "
                                   "matched=0, tie=false",
            },
            "metrics": metrics,
            "results": results,
        }, f, indent=2)

    print(f"wrote {OUT.name}")
    print(f"  selected={len(results)} matched_rows={len(matched_rows)} "
          f"no_match={len(no_match_rows)} ties={len(tie_rows)} coverage={metrics['coverage_percentage_pct']}%")
    print(f"  total_tokens={total_tokens} total_matched={total_matched} "
          f"match_rate={metrics['overall_matched_token_rate_pct']}% avg={metrics['average_matched_tokens_per_review']}")
    return 0


if __name__ == "__main__":
    main()
