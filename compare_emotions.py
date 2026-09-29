"""
Step 5A3 — combine and compare the locked LLM and NRC emotion results.

Reads ONLY:
  - step5a1_results.json  (LLM: llm_primary_emotion, valid, ...)
  - step5a2b_results.json (NRC: nrc_primary_emotion, nrc_top_emotions, nrc_tie,
                           emotion_scores, matched_token_count)
and joins by row_index. No model/API calls, no downloads, no rescoring. The rating is
never read and is never used as emotion ground truth.

Eligibility: a row is eligible for LLM-NRC emotion agreement iff the LLM response is
valid AND nrc_primary_emotion != "NO_MATCH". For ineligible NO_MATCH rows the agreement
fields are null and they do not enter any agreement denominator.

Agreement definitions (eligible rows only):
  exact_emotion_agreement      = (llm_primary_emotion == nrc_primary_emotion)
  tie_aware_emotion_agreement  = (llm_primary_emotion in nrc_top_emotions)

For a unique NRC maximum, exact and tie-aware agree; for a tied NRC maximum tie-aware
may be true while exact is false (canonical tie-break selected a different primary).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LLM = ROOT / "step5a1_results.json"
NRC = ROOT / "step5a2b_results.json"
OUT = ROOT / "step5a3_comparison.json"

EMOTIONS = ["ANGER", "ANTICIPATION", "DISGUST", "FEAR",
            "JOY", "SADNESS", "SURPRISE", "TRUST"]
EMO_SET = set(EMOTIONS)


def main() -> int:
    llm = json.loads(LLM.read_text(encoding="utf-8"))
    nrc = json.loads(NRC.read_text(encoding="utf-8"))
    lrows = llm["results"]
    nrows = nrc["results"]

    # ---- pre-conditions (assert before comparing) ----
    assert len(lrows) == 100 and len({r["row_index"] for r in lrows}) == 100
    assert len(nrows) == 100 and len({r["row_index"] for r in nrows}) == 100
    assert sorted(r["row_index"] for r in lrows) == list(range(100))
    assert sorted(r["row_index"] for r in nrows) == list(range(100))
    lmap = {r["row_index"]: r for r in lrows}
    nmap = {r["row_index"]: r for r in nrows}
    for i in range(100):
        assert i in lmap and i in nmap, f"missing row {i}"
    for i in range(100):
        assert lmap[i]["title"] == nmap[i]["title"], f"title mismatch row {i}"
        assert lmap[i]["text"] == nmap[i]["text"], f"text mismatch row {i}"
    # all 100 LLM rows valid
    assert all(lrows[i]["valid"] for i in range(100)), "not all LLM rows valid"
    for r in lrows:
        assert r["llm_primary_emotion"] in EMO_SET, r["row_index"]
    for r in nrows:
        assert set(r["emotion_scores"].keys()) == EMO_SET, r["row_index"]
    # locked aggregate distributions still match their files (full enum incl. zeros)
    def full_llm_dist():
        c = Counter(r["llm_primary_emotion"] for r in lrows)
        return {e: c.get(e, 0) for e in EMOTIONS}

    def full_nrc_dist():
        c = Counter(r["nrc_primary_emotion"] for r in nrows)
        d = {e: c.get(e, 0) for e in EMOTIONS}
        d["NO_MATCH"] = c.get("NO_MATCH", 0)
        return d

    assert llm["metrics"]["llm_primary_emotion_distribution"] == full_llm_dist(), "LLM dist"
    assert nrc["metrics"]["primary_emotion_distribution"] == full_nrc_dist(), "NRC dist"

    # ---- join & compute agreement ----
    combined = []
    for i in range(100):
        L, N = lmap[i], nmap[i]
        llm_emo = L["llm_primary_emotion"]
        nrc_emo = N["nrc_primary_emotion"]
        nrc_top = N["nrc_top_emotions"]
        eligible = (L["valid"] is True) and (nrc_emo != "NO_MATCH")
        if not eligible:
            exact, tie_aware = None, None
        else:
            exact = llm_emo == nrc_emo
            tie_aware = llm_emo in nrc_top
        combined.append({
            "row_index": i,
            "title": L["title"],
            "text": L["text"],
            "llm_primary_emotion": llm_emo,
            "nrc_primary_emotion": nrc_emo,
            "nrc_top_emotions": nrc_top,
            "nrc_tie": N["nrc_tie"],
            "nrc_emotion_scores": N["emotion_scores"],
            "nrc_matched_token_count": N["matched_token_count"],
            "comparison_eligible": eligible,
            "exact_emotion_agreement": exact,
            "tie_aware_emotion_agreement": tie_aware,
        })

    eligible = [r for r in combined if r["comparison_eligible"]]
    nomatch = [r for r in combined if r["nrc_primary_emotion"] == "NO_MATCH"]
    total = len(combined)

    # ---- agreements (eligible only) ----
    n_exact = sum(1 for r in eligible if r["exact_emotion_agreement"])
    n_tieaware = sum(1 for r in eligible if r["tie_aware_emotion_agreement"])
    n_elt = len(eligible)
    recovered = n_tieaware - n_exact  # additional recovered by tie-aware

    # ---- disagreement lists ----
    exact_disagree = [r["row_index"] for r in eligible if not r["exact_emotion_agreement"]]
    tieaware_disagree = [r["row_index"] for r in eligible if not r["tie_aware_emotion_agreement"]]
    tiebreak_sensitive = [r["row_index"] for r in eligible
                          if (not r["exact_emotion_agreement"]) and r["tie_aware_emotion_agreement"]]
    nomatch_rows = [r["row_index"] for r in nomatch]

    # ---- distributions (include zero-count) ----
    llm_dist = {e: sum(1 for r in combined if r["llm_primary_emotion"] == e)
                for e in EMOTIONS}
    nrc_dist = {e: sum(1 for r in combined if r["nrc_primary_emotion"] == e)
                for e in EMOTIONS}
    nrc_dist["NO_MATCH"] = len(nomatch)

    # ---- 8x8 crosstab (eligible only): rows=LLM, cols=NRC primary ----
    ctab = {llm_e: {nrc_e: 0 for nrc_e in EMOTIONS} for llm_e in EMOTIONS}
    for r in eligible:
        ctab[r["llm_primary_emotion"]][r["nrc_primary_emotion"]] += 1

    # ---- split by NRC structure ----
    unique_max = [r for r in eligible if not r["nrc_tie"]]
    tied_max = [r for r in eligible if r["nrc_tie"]]
    def grp_agreements(rows, field):
        k = len(rows)
        c = sum(1 for r in rows if r[field])
        return {"rows": k, "agreements": c, "denominator": k,
                "percentage_pct": round(100 * c / k, 2) if k else 0.0}
    grouped = {
        "unique_maximum": {
            "rows": len(unique_max),
            "exact": grp_agreements(unique_max, "exact_emotion_agreement"),
            "tie_aware": grp_agreements(unique_max, "tie_aware_emotion_agreement"),
        },
        "tied_maximum": {
            "rows": len(tied_max),
            "exact": grp_agreements(tied_max, "exact_emotion_agreement"),
            "tie_aware": grp_agreements(tied_max, "tie_aware_emotion_agreement"),
        },
    }

    def pct(c, d):
        return round(100 * c / d, 2) if d else 0.0

    metrics = {
        "total_rows": total,
        "valid_llm_rows": len(combined),  # all 100 LLM rows valid (asserted above)
        "nrc_covered_rows": sum(1 for r in combined if r["nrc_primary_emotion"] != "NO_MATCH"),
        "nrc_no_match_rows": len(nomatch),
        "eligible_comparison_rows": len(eligible),
        "exact_agreement": {"count": n_exact, "eligible_denominator": n_elt,
                            "percentage_pct": pct(n_exact, n_elt)},
        "tie_aware_agreement": {"count": n_tieaware, "eligible_denominator": n_elt,
                                "percentage_pct": pct(n_tieaware, n_elt)},
        "additional_agreements_recovered_by_tie_aware": recovered,
        "llm_emotion_distribution": llm_dist,
        "nrc_primary_emotion_distribution": nrc_dist,
        "cross_tabulation_8x8_llm_by_nrc": ctab,
        "exact_disagreement_rows": sorted(exact_disagree),
        "tie_aware_disagreement_rows": sorted(tieaware_disagree),
        "tie_break_sensitive_rows": sorted(tiebreak_sensitive),
        "no_match_rows": sorted(nomatch_rows),
        "grouped_by_nrc_structure": grouped,
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "note": "Step 5A3 LLM vs NRC emotion comparison. Neither method is treated "
                    "as ground truth. No model/API calls, no rescoring, rating never "
                    "used as emotion ground truth. No API key.",
            "method": {
                "eligibility": "LLM valid AND nrc_primary_emotion != NO_MATCH",
                "exact_definition": "llm_primary_emotion == nrc_primary_emotion",
                "tie_aware_definition": "llm_primary_emotion in nrc_top_emotions",
                "no_match_handling": "agreement null, excluded from denominator",
                "rating_used": False,
            },
            "metrics": metrics,
            "rows": combined,
        }, f, indent=2)

    # ---- report ----
    print(f"total={total} valid_llm=100 nrc_covered={metrics['nrc_covered_rows']} "
          f"no_match={len(nomatch)} eligible={len(eligible)}")
    print(f"exact      = {n_exact}/{n_elt} ({pct(n_exact, n_elt)}%)")
    print(f"tie-aware  = {n_tieaware}/{n_elt} ({pct(n_tieaware, n_elt)}%)  "
          f"recovered={recovered}")
    print("LLM dist:", llm_dist)
    print("NRC dist:", nrc_dist)
    print("\n8x8 crosstab (LLM row x NRC col):")
    print("          " + "".join(f"{e[:4]:>6}" for e in EMOTIONS))
    for llm_e in EMOTIONS:
        print(f"{llm_e[:4]:>6}   " + "".join(f"{ctab[llm_e][nrc_e]:>6}" for nrc_e in EMOTIONS))
    print("\nunique_max:", grouped["unique_maximum"])
    print("tied_max:  ", grouped["tied_maximum"])
    print("exact_disagree:", sorted(exact_disagree))
    print("tieaware_disagree:", sorted(tieaware_disagree))
    print("tiebreak_sensitive:", sorted(tiebreak_sensitive))
    print("no_match:", sorted(nomatch_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
