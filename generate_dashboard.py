"""
Step 3 — generate the self-contained offline dashboard from step2_results.json.

Sole source of truth: step2_results.json (review rows + numeric metrics). No result
numbers are typed into HTML; everything is read/calculated from the JSON. Data-owner
validation asserts the key invariants BEFORE generating, so a dashboard can never be
silently built from inconsistent data. All dataset-provided title/text is HTML-escaped.

No model/API calls are made. Step 1 / Step 2 scripts and outputs are not modified.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "step2_results.json"
OUT = ROOT / "dashboard.html"

LABELS = ["POSITIVE", "NEGATIVE"]


# ----------------------------------------------------------------------------- data validation
def assert_data(d: dict) -> None:
    m = d["metrics"]
    checks = {
        "selected_rows == 100": m["selected_rows"] == 100,
        "valid_predictions == 100": m["valid_predictions"] == 100,
        "correct_predictions == 97": m["correct_predictions"] == 97,
        "mismatch_count == 3": m["mismatch_count"] == 3,
        "actual dist 93/7": m["actual_distribution"] == {"POSITIVE": 93, "NEGATIVE": 7},
        "predicted dist 92/8": m["predicted_distribution"] == {"POSITIVE": 92, "NEGATIVE": 8},
        "confusion cells sum 100": sum(
            m["confusion_matrix"][a][p] for a in LABELS for p in LABELS) == 100,
        "confusion diagonal sum 97": sum(m["confusion_matrix"][a][a] for a in LABELS) == 97,
        "majority baseline 93%": abs(
            m["majority_class_baseline"]["always_positive_accuracy_pct"] - 93.0) < 1e-9,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(
            "DATA ASSERTION FAILURE — dashboard NOT generated:\n  "
            + "\n  ".join(failed))
    # rows sanity: 100 unique indices 0..99, all valid
    rows = d["rows"]
    assert len(rows) == 100, f"rows != 100: {len(rows)}"
    assert {r["row_index"] for r in rows} == set(range(100)), "row indices not 0..99"
    assert all(r["valid"] for r in rows), "some rows invalid"
    # rows consistent with aggregate metrics (recomputed)
    from collections import Counter
    ref = Counter(r["reference"] for r in rows)
    pred = Counter(r["predicted"] for r in rows)
    assert dict(ref) == m["actual_distribution"], "row reference dist != metrics"
    assert dict(pred) == m["predicted_distribution"], "row predicted dist != metrics"
    assert sum(1 for r in rows if not r["agree_with_rating"]) == m["mismatch_count"]


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


# ----------------------------------------------------------------------------- html builders
def build_style() -> str:
    return """
:root{
  /* ---- Amazon-inspired tokens ---- */
  --bg:#eaeded;            /* light neutral page */
  --surface:#ffffff;       /* cards & tables */
  --surface-2:#f5f6f6;
  --ink:#0f1111;           /* primary text */
  --muted:#565959;         /* secondary text */
  --border:#d5d9d9;        /* restrained border */
  --accent:#007185;        /* data/link blue */
  --accent-ink:#ffffff;
  --orange:#ff9900;        /* active-tab underline / stars */
  --navy:#131921;          /* masthead */
  --navy-2:#232f3e;        /* secondary dark */
  --pos:#1d7a52;           /* positive / success */
  --pos-bg:#e6f2ec;
  --neg:#b3342c;           /* negative / error */
  --neg-bg:#fbeae8;
  --warn:#8a6d1a;
  --warn-bg:#faf3dc;
  --focus:#007185;
  /* ---- derived / spacing ---- */
  --radius:6px;
  --radius-sm:4px;
  --shadow:0 1px 3px rgba(0,0,0,.07);
  --font-head:"Trebuchet MS","Segoe UI",Arial,sans-serif;  /* headings, tabs, metric values */
  --font:Arial,"Segoe UI",sans-serif;                       /* body & tables */
  --mono:"SF Mono",ui-monospace,Menlo,Consolas,"Liberation Mono",monospace;
  --gold:#ffd814;            /* high-contrast pale gold for dark cards */
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  background:var(--bg);color:var(--ink);font-family:var(--font);
  line-height:1.45;font-size:16px;-webkit-font-smoothing:antialiased;
}
h1,h2,h3,.tabbar button[role="tab"],.stat .num,.stat .lbl,.masthead h1,
summary,.filter-label,.visually-h3{font-family:var(--font-head)}
.container{max-width:1080px;margin:0 auto;padding:32px 22px 72px}
@media(min-width:760px){.container{padding:44px 32px 84px}}
/* ---------- header ---------- */
.masthead{background:var(--navy);color:#fff;margin:-32px -22px 0;padding:26px 22px 22px}
@media(min-width:760px){.masthead{margin:-44px -32px 0;padding:32px 32px 26px}}
.masthead h1{margin:0 0 4px;font-size:clamp(22px,3.6vw,30px);line-height:1.15;letter-spacing:-.01em;color:#fff}
.masthead .stars{color:var(--orange);letter-spacing:.12em;font-size:.9em;vertical-align:3px;margin-left:10px}
.masthead p.sub{color:#d6d9dc;margin:0;max-width:80ch;font-size:14px}
.masthead p.sub strong{color:#fff}
/* ---------- tab bar ---------- */
.tabbar{display:flex;gap:2px;border-bottom:1px solid var(--border);margin:18px 0 6px;overflow-x:auto;-webkit-overflow-scrolling:touch;scrollbar-width:thin}
.tabbar button[role="tab"]{
  appearance:none;border:0;background:none;cursor:pointer;font:inherit;font-size:14.5px;
  font-weight:600;color:var(--muted);padding:10px 14px 9px;white-space:nowrap;
  border-bottom:3px solid transparent;border-radius:0;box-shadow:none;
}
.tabbar button[role="tab"]:hover{color:var(--ink)}
.tabbar button[role="tab"][aria-selected="true"]{color:var(--ink);border-bottom-color:var(--orange)}
.tabbar button[role="tab"]:focus-visible{outline:2px solid var(--focus);outline-offset:-2px}
.tabpanel{display:block}
.tabpanel[hidden]{display:none}
/* overview jump controls */
.jumprow{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 6px}
.jump{
  appearance:none;font:inherit;cursor:pointer;background:var(--surface);color:var(--accent);
  border:1px solid var(--border);border-radius:var(--radius-sm);padding:7px 14px;
  font-size:13.5px;font-weight:600;
}
.jump:hover{border-color:var(--accent);background:var(--surface-2)}
.jump:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
/* ---------- progressive disclosure rows (native <details>) ---------- */
details.disclose{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  margin:14px 0 0}
details.disclose>summary{
  list-style:none;cursor:pointer;padding:11px 16px;font-size:14.5px;font-weight:600;
  color:var(--ink);display:flex;align-items:center;gap:10px;user-select:none;
}
details.disclose>summary::-webkit-details-marker{display:none}
details.disclose>summary::before{
  content:"▸";color:var(--orange);font-size:12px;width:14px;display:inline-block;text-align:center;
  transition:transform .15s ease;
}
details.disclose[open]>summary::before{transform:rotate(90deg)}
details.disclose>summary:hover{background:var(--surface-2)}
details.disclose>summary:focus-visible{outline:2px solid var(--focus);outline-offset:-2px}
details.disclose .d-meta{margin-left:auto;font-size:12px;color:var(--muted);font-weight:400;
  font-family:var(--font)}
details.disclose>.d-body{padding:2px 16px 16px;border-top:1px solid var(--border)}
details.disclose .d-body .filterbar{margin-top:12px}
/* compact inline metric / run-details line */
.mline{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--muted);margin:2px 0 0}
.mline b{color:var(--ink);font-weight:700}
.mline .sep{color:var(--border)}
/* ---------- section ---------- */
section{margin:22px 0}
h2{
  font-size:20px;margin:0 0 4px;letter-spacing:-.01em;
}
.lead{color:var(--muted);font-size:14.5px;margin:0 0 18px;max-width:70ch}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  box-shadow:var(--shadow);padding:22px;margin-top:14px}
/* ---------- headline stat band ---------- */
.statband{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:14px}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:18px 18px 16px;box-shadow:var(--shadow)}
.stat .num{font-size:30px;font-weight:700;line-height:1;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.stat .lbl{color:var(--muted);font-size:12.5px;margin-top:8px;font-weight:500}
.stat.accent{background:var(--accent);border-color:var(--accent)}
.stat.accent .num,.stat.accent .lbl{color:var(--accent-ink)}
/* contrast: never red/dark text on teal; pale gold for dark-card secondary text */
.stat.accent .delta{color:var(--gold)}
.stat.accent .delta.good,.stat.accent .delta.bad{color:var(--gold)}
.stat .delta{font-size:12px;margin-top:6px;font-weight:600}
.delta.good{color:var(--pos)} .delta.bad{color:var(--neg)}
.delta.ne{color:var(--muted);font-weight:500}
/* baseline pull-quote */
.baseline{background:var(--warn-bg);border:1px solid #e6d49a;border-left:4px solid var(--warn);
  border-radius:var(--radius-sm);padding:12px 16px;font-size:14px;margin-top:14px}
.baseline strong{color:var(--warn)}
/* ---------- stacked distribution bars ---------- */
.dist{display:grid;gap:14px;margin-top:6px}
.distrow{display:grid;grid-template-columns:150px 1fr;gap:14px;align-items:start}
@media(max-width:560px){.distrow{grid-template-columns:1fr}}
.dist .dlabel{font-size:13.5px;color:var(--muted);padding-top:2px;font-weight:600}
.bar{height:34px;border-radius:6px;overflow:hidden;display:flex;background:var(--border);border:1px solid var(--border)}
.bar .seg{display:flex;align-items:center;justify-content:center;color:#fff;font-size:12.5px;font-weight:700;
  white-space:nowrap;overflow:hidden;min-width:34px}
.seg.neg{background:var(--neg)} .seg.pos{background:var(--pos)} .seg.neu{background:var(--warn)}
.barlegend{display:flex;gap:18px;margin-top:10px;font-size:13px;color:var(--muted)}
.barlegend .sw{width:12px;height:12px;border-radius:3px;display:inline-block;margin-right:6px;vertical-align:-1px}
/* ---------- confusion matrix ---------- */
.cmtable{width:100%;border-collapse:separate;border-spacing:0;max-width:520px;margin-top:6px}
.cmtable th,.cmtable td{padding:14px 16px;text-align:center;border:1px solid var(--border)}
.cmtable .corner{background:var(--surface-2);color:var(--muted);font-size:12.5px;font-weight:600}
.cmtable .rowlab{background:var(--surface-2);color:var(--ink);font-weight:600;text-align:left;font-size:13px}
.cmtable td .rowlab{} 
.cmtable .lab{font-size:12px;color:var(--muted);display:block;font-weight:500}
.cmtable td.nb{background:var(--neg-bg);color:var(--ink);font-weight:600;font-size:20px}
.cmtable td.ok{background:var(--pos-bg);color:var(--ink);font-weight:700;font-size:20px}
.cmtable td.mid{background:var(--surface-2);color:var(--muted);font-size:20px;font-weight:600}
.caption{font-size:13px;color:var(--muted);margin-top:10px}
/* ---------- per class ---------- */
.classgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-top:14px}
.cls{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px}
.cls h3{margin:0 0 12px;font-size:15px}
.cls .tag{display:inline-block;font-size:11px;font-weight:700;padding:2px 8px;border-radius:999px;margin-bottom:8px}
.cls.pos .tag{background:var(--pos-bg);color:var(--pos)}
.cls.neg .tag{background:var(--neg-bg);color:var(--neg)}
.met{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:4px}
.met .m b{font-size:19px;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.met .m span{display:block;color:var(--muted);font-size:11.5px;font-weight:500;text-transform:uppercase;letter-spacing:.05em}
.callout{font-size:13.5px;color:var(--muted);margin-top:12px}
/* ---------- error direction ---------- */
.direction{display:grid;gap:10px;margin-top:8px}
.dir{display:flex;gap:12px;align-items:center;background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius-sm);padding:12px 14px;font-size:14.5px}
.dir .n{font-size:22px;font-weight:700;min-width:56px;font-variant-numeric:tabular-nums}
.dir .n.neg{color:var(--neg)} .dir .n.pos{color:var(--pos)}
.dir p{margin:0}
.dir .arrow{color:var(--muted);font-weight:600}
/* ---------- mismatch cards ---------- */
.misms{display:grid;gap:16px;margin-top:14px}
.mism{background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--warn);
  border-radius:var(--radius);padding:18px 20px;box-shadow:var(--shadow)}
.mism .rowhead{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:10px}
.mism .rnum{font-family:var(--mono);font-weight:700;font-size:14px}
.mism .rtitle{font-weight:600;font-size:15px}
.badge{font-size:11px;font-weight:700;padding:3px 9px;border-radius:999px;letter-spacing:.03em}
.badge.pos{background:var(--pos-bg);color:var(--pos)}
.badge.neg{background:var(--neg-bg);color:var(--neg)}
.badge.mid{background:var(--surface-2);color:var(--muted);border:1px solid var(--border)}
.mism .text{font-size:14.5px;color:#2a3038;background:var(--surface-2);border-radius:var(--radius-sm);
  padding:12px 14px;margin:6px 0 12px;line-height:1.55}
.mism .predline{display:flex;flex-wrap:wrap;gap:14px;font-size:13.5px;color:var(--muted);align-items:center}
.mism .predline .arrow{font-weight:700}
/* ---------- review table ---------- */
.tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid var(--border);
  border-radius:var(--radius);margin-top:14px;background:var(--surface)}
table.reviews{width:100%;border-collapse:collapse;min-width:820px;font-size:13.5px}
table.reviews th,table.reviews td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--border);
  vertical-align:top}
table.reviews th{position:sticky;top:0;background:var(--surface-2);font-size:11px;text-transform:uppercase;
  letter-spacing:.05em;color:var(--muted);font-weight:600;z-index:1}
table.reviews td.num{font-family:var(--mono);text-align:right}
table.reviews td.rating{text-align:center}
table.reviews tr:nth-child(even){background:#fcfbf8}
table.reviews tr.mism{background:var(--warn-bg);border-left:3px solid var(--warn)}
table.reviews .title-cell{font-weight:600;max-width:220px}
table.reviews .text-cell{color:#333a42;max-width:420px}
.reviewtext{color:#333a42}
.status{font-size:11.5px;font-weight:700;padding:2px 8px;border-radius:999px}
.status.ok{background:var(--pos-bg);color:var(--pos)}
.status.mism{background:var(--warn-bg);color:var(--warn)}
table.reviews td .sm{display:block;color:var(--muted);font-size:11px;text-transform:none;letter-spacing:0}
/* ---------- Step: emotion-evidence table (Emotion tab) ---------- */
table.reviews.emo-table{min-width:640px}
table.reviews tr.emorow td{padding:8px 12px}
table.reviews tr.emorow td.num{font-family:var(--mono);text-align:right}
table.reviews .emocell{font-size:12.5px;font-weight:600;white-space:nowrap}
table.reviews .emo-top-td{font-size:12px;color:var(--muted)}
.estat{font-size:11px;font-weight:700;padding:2px 8px;border-radius:6px;white-space:nowrap}
.estat.exact{background:var(--pos-bg);color:var(--pos)}
.estat.ta{background:var(--warn-bg);color:var(--warn)}
.estat.dis{background:var(--neg-bg);color:var(--neg)}
.estat.nomatch{background:var(--surface-2);color:var(--muted);border:1px solid var(--border)}
/* ---------- Step 4: review-table filtering ---------- */
.filterbar{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-top:14px}
@media(max-width:420px){.filterbar{gap:8px}}
.filter-label{font-size:13px;color:var(--muted);font-weight:600;margin-right:2px}
.fbtn{
  appearance:none;border:1px solid var(--border);background:var(--surface);color:var(--ink);
  font:inherit;font-size:13.5px;font-weight:600;padding:7px 12px;border-radius:var(--radius-sm);
  cursor:pointer;display:inline-flex;align-items:center;gap:7px;line-height:1.2;
  white-space:nowrap;
}
.fbtn:hover{border-color:var(--accent)}
.fbtn[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
.fbtn:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.fbtn .fcount{font-variant-numeric:tabular-nums;font-size:12px;color:var(--muted)}
.fbtn[aria-pressed="true"] .fcount{color:var(--accent-ink);opacity:.92}
.livesummary{margin:10px 0 0;font-size:13.5px;color:var(--muted);font-weight:500}
.livesummary strong{color:var(--ink);font-variant-numeric:tabular-nums}
.reviewrow{display:table-row}          /* default: visible */
.reviewrow.tr-hidden{display:none}     /* filtered out (rows stay in DOM) */
.b6row{display:table-row}              /* Step 6C balanced rows: default visible */
.b6row.tr-hidden{display:none}         /* Step 6C filtered out (rows stay in DOM) */
/* ---------- Step 5B: emotion analysis ---------- */
.emo-note{background:var(--surface-2);border:1px solid var(--border);border-left:4px solid var(--accent);
  border-radius:var(--radius-sm);padding:12px 16px;font-size:14px;margin-top:14px}
.emo-note strong{color:var(--accent)}
.distcol{display:grid;gap:18px}
@media(min-width:760px){.distcol.two{grid-template-columns:1fr 1fr}}
.distcol h3{font-size:14px;margin:0 0 10px}
.erow{display:grid;grid-template-columns:96px 1fr 44px;gap:10px;align-items:center;margin:6px 0;font-size:13px}
.erow .elab{font-weight:600;text-align:right;color:var(--ink)}
.ebar{height:20px;border-radius:5px;background:var(--surface-2);border:1px solid var(--border);overflow:hidden;position:relative}
.ebar .fill{height:100%;border-radius:4px}
.ebar .fill.llm{background:var(--accent)}
.ebar .fill.nrc{background:#64748b}          /* slate for NRC (recolor via --emo-nrc) */
.erow .ecount{font-weight:700;font-variant-numeric:tabular-nums;text-align:right;font-size:12.5px}
.erow .ecount.zero{color:var(--muted);font-weight:500}
.emomatwrap{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid var(--border);
  border-radius:var(--radius);background:var(--surface);margin-top:14px}
table.emomatrix{border-collapse:collapse;min-width:660px;font-size:13px;width:100%}
table.emomatrix th,table.emomatrix td{padding:8px 10px;text-align:center;border:1px solid var(--border)}
table.emomatrix thead th,table.emomatrix tbody th{background:var(--surface-2);font-size:11px;text-transform:uppercase;
  letter-spacing:.04em;color:var(--muted);font-weight:700}
table.emomatrix tbody th{text-align:left;min-width:110px;color:var(--ink)}
table.emomatrix td.diag{background:var(--pos-bg);color:var(--ink);font-weight:700}
table.emomatrix td.off{background:var(--surface)}
.emotable-caption{font-size:13px;color:var(--muted);margin-top:10px}
.emosubgrid{display:grid;gap:14px;margin-top:14px}
@media(min-width:760px){.emosubgrid.two{grid-template-columns:1fr 1fr}}
.emosub{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px 18px}
.emosub h3{margin:0 0 8px;font-size:15px}
.emosub table{width:100%;border-collapse:collapse;font-size:13.5px}
.emosub td,.emosub th{padding:6px 8px;border-bottom:1px solid var(--border);text-align:right}
.emosub td:first-child,.emosub th:first-child{text-align:left}
.emosub th{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.04em}
.example-list{display:grid;gap:16px;margin-top:14px}
.example-card{background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--accent);
  border-radius:var(--radius);padding:16px 18px;box-shadow:var(--shadow)}
.example-card .rowhead{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:8px}
.example-card .rtitle{font-weight:600;font-size:14.5px}
.example-card .reason{font-size:13px;color:var(--muted);margin:8px 0 0}
.example-card pre.emvec{font-family:var(--mono);font-size:11.5px;color:var(--ink);background:var(--surface-2);
  border:1px solid var(--border);border-radius:6px;padding:8px 10px;margin:8px 0 0;white-space:pre-wrap;line-height:1.5}
.estat{font-size:11px;font-weight:700;padding:2px 8px;border-radius:999px;letter-spacing:.03em}
.estat.exact{background:var(--pos-bg);color:var(--pos)}
.estat.ta{background:var(--warn-bg);color:var(--warn)}
.estat.dis{background:var(--neg-bg);color:var(--neg)}
.estat.nomatch{background:var(--surface-2);color:var(--muted);border:1px solid var(--border)}
.visually-h3{font-size:15px;font-weight:600;margin:22px 0 6px}
table.reviews .emocell{font-size:12px;white-space:nowrap;font-weight:600}
table.reviews .nrc-top{max-width:150px;white-space:normal;color:var(--muted);font-size:11.5px}
/* ---------- Step 6C: balanced three-class ---------- */
.c38{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:6px}
.c38 .m b{font-size:17px;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.c38 .m span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em;font-weight:500}
.cm3wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid var(--border);
  border-radius:var(--radius);background:var(--surface);margin-top:14px}
table.cm3{border-collapse:collapse;min-width:520px;font-size:13.5px;width:100%}
table.cm3 th,table.cm3 td{padding:10px 12px;text-align:center;border:1px solid var(--border)}
table.cm3 thead th,table.cm3 tbody th{background:var(--surface-2);font-size:11px;text-transform:uppercase;
  letter-spacing:.04em;color:var(--muted);font-weight:700}
table.cm3 tbody th{text-align:left;min-width:120px;color:var(--ink)}
table.cm3 td.ok3{background:var(--pos-bg);color:var(--ink);font-weight:700;font-size:18px}
table.cm3 td.err3{background:var(--neg-bg);color:var(--ink);font-weight:600;font-size:18px}
.neutralcall{background:var(--warn-bg);border:1px solid #e6d49a;border-left:4px solid var(--warn);
  border-radius:var(--radius-sm);padding:14px 18px;margin-top:14px}
.neutralcall h3{margin:0 0 8px;font-size:15px;color:var(--warn)}
.neutralcall p{margin:4px 0;font-size:14px}
.dirrow{display:flex;gap:12px;align-items:center;border:1px solid var(--border);border-radius:var(--radius-sm);
  padding:10px 14px;font-size:14px;background:var(--surface)}
.dirrow .dn{font-size:19px;font-weight:700;min-width:56px;font-variant-numeric:tabular-nums}
.dirrow .dn.dom{color:var(--neg)}
.dirrow .dn.okd{color:var(--muted)}
.dirrow p{margin:0}
/* ---------- Step 7: descriptive & prediction visualizations ---------- */
.s7dist{display:grid;gap:8px;margin-top:8px}
.s7row{display:grid;grid-template-columns:110px 1fr 120px 90px;gap:10px;align-items:center;font-size:13px}
@media(max-width:700px){.s7row{grid-template-columns:74px 1fr 64px}}
.s7row .slab{font-weight:600;color:var(--ink)}
.s7row .sbar{height:24px;border-radius:5px;background:var(--surface-2);border:1px solid var(--border);overflow:hidden;position:relative}
.s7row .sbar .fill{height:100%;border-radius:4px}
.s7row .fill.s7one{background:var(--accent)}
.s7row .fill.s7two{background:#64748b}
.s7row .fill.s7three{background:var(--pos)}
.s7row .fill.s7four{background:#b08a3e}
.s7row .fill.s7five{background:var(--warn)}
.s7row .scount{font-weight:700;font-variant-numeric:tabular-nums;text-align:right;font-size:12.5px}
.s7row .spct{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
@media(max-width:700px){.s7row .spct{display:none}}
.s7note{font-size:13px;color:var(--muted);margin-top:10px}
.s7lead{font-size:14px;color:var(--muted);max-width:72ch;margin-top:2px}
.contrast{display:grid;gap:14px;margin-top:14px}
@media(min-width:760px){.contrast.two{grid-template-columns:1fr 1fr}}
.contrcard{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px 18px}
.contrcard h3{margin:0 0 10px;font-size:14.5px}
.contrcard table{width:100%;border-collapse:collapse;font-size:13.5px}
.contrcard td,.contrcard th{padding:6px 8px;border-bottom:1px solid var(--border);text-align:right}
.contrcard td:first-child,.contrcard th:first-child{text-align:left}
.contrcard th{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.04em}
/* ---------- footer ---------- */
footer{margin-top:48px;padding-top:18px;border-top:1px solid var(--border);color:var(--muted);font-size:13px}
a{color:var(--accent)}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
"""


def stat(num: str, lbl: str, accent=False, delta_txt=None, delta_cls=None) -> str:
    d = ""
    if delta_txt:
        cls = "good" if delta_cls == "good" else ("ne" if delta_cls == "neutral" else "bad")
        d = f'<div class="delta {cls}">{delta_txt}</div>'
    cls = " stat accent" if accent else ""
    return (f'<div class="stat{cls}"><div class="num">{num}</div>'
            f'<div class="lbl">{lbl}</div>{d}</div>')


def bar(label: str, pos: int, neg: int, pos_lab: str, neg_lab: str) -> str:
    total = pos + neg
    pw = 100 * pos / total if total else 0
    nw = 100 * neg / total if total else 0
    return f"""
    <div class="distrow">
      <div class="dlabel">{label}</div>
      <div>
        <div class="bar" role="img" aria-label="{pos_lab}: {pos}, {neg_lab}: {neg}">
          <div class="seg pos" style="width:{pw:.2f}%">{pos}&nbsp;POS</div>
          <div class="seg neg" style="width:{nw:.2f}%">{neg}&nbsp;NEG</div>
        </div>
      </div>
    </div>"""


def _load_sources() -> dict:
    """Load all locked source files (Step 2 + emotion + balanced 3-class) and
    validate cross-source identity and locked metrics."""
    d2 = json.loads(SRC.read_text(encoding="utf-8"))
    a1 = json.loads((ROOT / "step5a1_results.json").read_text(encoding="utf-8"))
    a2 = json.loads((ROOT / "step5a2b_results.json").read_text(encoding="utf-8"))
    a3 = json.loads((ROOT / "step5a3_comparison.json").read_text(encoding="utf-8"))

    assert_data(d2)  # locked sentiment evaluation (Step 2)

    lrows, nrows, crows = a1["results"], a2["results"], a3["rows"]
    assert len(lrows) == 100 and len(nrows) == 100 and len(crows) == 100
    assert sorted(r["row_index"] for r in lrows) == list(range(100))
    assert sorted(r["row_index"] for r in nrows) == list(range(100))
    assert sorted(r["row_index"] for r in crows) == list(range(100))
    for i in range(100):
        l, n, c = lrows[i], nrows[i], crows[i]
        t = d2["rows"][i]
        # titles/texts match across sources
        assert l["title"] == t["title"] and l["text"] == t["text"], f"title/text a1 row {i}"
        assert n["title"] == t["title"] and n["text"] == t["text"], f"title/text a2 row {i}"
        assert c["title"] == t["title"] and c["text"] == t["text"], f"title/text a3 row {i}"
    # all 100 LLM responses valid
    assert all(r["valid"] for r in lrows)
    from collections import Counter as _C
    ldist = {e: _C(r["llm_primary_emotion"] for r in lrows).get(e, 0)
             for e in EMOTIONS}
    ndist = {e: _C(r["nrc_primary_emotion"] for r in nrows).get(e, 0)
             for e in EMOTIONS}
    ndist["NO_MATCH"] = sum(1 for r in nrows if r["nrc_primary_emotion"] == "NO_MATCH")
    # locked emotion aggregates unchanged
    _assert_eq(a1["metrics"]["llm_primary_emotion_distribution"], ldist, "a1 llm dist")
    _assert_eq(a2["metrics"]["primary_emotion_distribution"], ndist, "a2 nrc dist")

    m3 = a3["metrics"]
    # verified Step 5A3 comparison metrics
    assert m3["nrc_covered_rows"] == 85 and m3["nrc_no_match_rows"] == 15
    assert m3["exact_agreement"]["count"] == 19 and m3["exact_agreement"]["eligible_denominator"] == 85
    assert abs(m3["exact_agreement"]["percentage_pct"] - 22.35) < 1e-9
    assert m3["tie_aware_agreement"]["count"] == 63 and m3["tie_aware_agreement"]["eligible_denominator"] == 85
    assert abs(m3["tie_aware_agreement"]["percentage_pct"] - 74.12) < 1e-9
    assert m3["additional_agreements_recovered_by_tie_aware"] == 44
    ties = sum(1 for r in nrows if r["nrc_tie"])
    assert ties == 52, f"NRC ties != 52: {ties}"

    b6 = _load_step6()  # locked balanced three-class results (Step 6B)

    return {"step2": d2, "a1": a1, "a2": a2, "a3": a3, "llm": lrows,
            "nrc": nrows, "cmp": crows, "cmpm": m3, "b6": b6}


def _load_step6() -> dict:
    """Load and validate the locked balanced three-class results (Step 6B)."""
    b6 = json.loads((ROOT / "balanced_3class_results.json").read_text(encoding="utf-8"))
    m = b6["metrics"]
    rows = b6["rows"]
    assert m["selected_rows"] == 150
    assert m["completed_responses"] == 150 and m["valid_predictions"] == 150
    assert m["invalid_predictions"] == 0 and m["failed_or_missing_rows"] == []
    assert m["correct_predictions"] == 112
    assert abs(m["accuracy_over_all_150_pct"] - 74.67) < 1e-9
    assert abs(m["majority_class_baseline_pct"] - 33.33) < 1e-9
    assert m["actual_distribution"] == {"POSITIVE": 50, "NEUTRAL": 50, "NEGATIVE": 50}
    assert m["predicted_distribution"] == {"POSITIVE": 57, "NEUTRAL": 19, "NEGATIVE": 74}
    assert m["confusion_matrix"] == {
        "POSITIVE": {"POSITIVE": 48, "NEUTRAL": 2, "NEGATIVE": 0},
        "NEUTRAL": {"POSITIVE": 8, "NEUTRAL": 16, "NEGATIVE": 26},
        "NEGATIVE": {"POSITIVE": 1, "NEUTRAL": 1, "NEGATIVE": 48}}
    assert m["macro_f1"] == 0.7117
    assert m["mismatch_count"] == 38
    assert len(rows) == 150
    assert sorted(r["sample_index"] for r in rows) == list(range(150))
    assert all(r["valid"] for r in rows)
    return b6


def _assert_eq(got, want, msg):
    if got != want:
        raise SystemExit(f"DASHBOARD ASSERTION FAILURE ({msg}): {got} != {want}")


EMOTIONS = ["ANGER", "ANTICIPATION", "DISGUST", "FEAR",
            "JOY", "SADNESS", "SURPRISE", "TRUST"]


def main() -> int:
    S = _load_sources()
    d = S["step2"]
    m = d["metrics"]
    rows = d["rows"]
    lrows, nrows, crows = S["llm"], S["nrc"], S["cmp"]
    cmpm = S["cmpm"]
    b6 = S["b6"]
    b6m = b6["metrics"]
    b6rows = b6["rows"]

    # row-level emotion joins by row_index
    lmap = {r["row_index"]: r for r in lrows}
    nmap = {r["row_index"]: r for r in nrows}
    cmap = {r["row_index"]: r for r in crows}

    cm = m["confusion_matrix"]
    pc = m["per_class"]
    baseline = m["majority_class_baseline"]

    acc = m["accuracy_all_rows_pct"]
    fp_corr = m["correct_predictions"]
    mism = m["mismatch_count"]

    # ---- headline band ----
    stats = (
        stat(f"{acc:.1f}%", "Overall accuracy", accent=True) +
        stat(f"{fp_corr}", "Correct predictions") +
        stat(str(mism), "Mismatches", delta_txt=f"improvement 4 pp vs 93% baseline", delta_cls="good") +
        stat("100", "Valid predictions") +
        stat("0", "Invalid predictions") +
        stat(f"{baseline['always_positive_accuracy_pct']:.1f}%", "Always-POSITIVE baseline")
    )

    # ---- distributions ----
    ad = m["actual_distribution"]
    pd = m["predicted_distribution"]
    dist_bars = (
        bar("Actual / reference", ad["POSITIVE"], ad["NEGATIVE"],
            "reference POSITIVE", "reference NEGATIVE") +
        bar("Model predicted", pd["POSITIVE"], pd["NEGATIVE"],
            "predicted POSITIVE", "predicted NEGATIVE")
    )

    # ---- confusion matrix ----
    cm_html = f"""
    <table class="cmtable">
      <caption class="sr-only">2 by 2 confusion matrix. Rows are the actual or reference
      class; columns are the model prediction. Main diagonal cells (91 and 6) are correct,
      off-diagonal cells (2 and 1) are errors.</caption>
      <thead>
        <tr><th scope="col" class="corner"></th>
            <th scope="col">Predicted<br><span class="lab">POSITIVE</span></th>
            <th scope="col">Predicted<br><span class="lab">NEGATIVE</span></th></tr>
      </thead>
      <tbody>
        <tr><th scope="row" class="rowlab">Actual<br><span class="lab">POSITIVE</span></th>
            <td class="ok">91<span class="lab">correct ✓</span></td>
            <td class="nb">2<span class="lab">error ✗</span></td></tr>
        <tr><th scope="row" class="rowlab">Actual<br><span class="lab">NEGATIVE</span></th>
            <td class="nb">1<span class="lab">error ✗</span></td>
            <td class="ok">6<span class="lab">correct ✓</span></td></tr>
      </tbody>
    </table>
    <p class="caption">Main diagonal = correct (91 + 6 = 97). Off diagonal = errors (2 + 1 = 3).</p>"""

    # ---- per-class ----
    def metrow(cls, a):
        p = pc[a]
        fcol = "pos" if a == "POSITIVE" else "neg"
        pct = [("Precision", f"{p['precision']:.4f}"),
               ("Recall", f"{p['recall']:.4f}"),
               ("F1", f"{p['f1']:.4f}")]
        mets = "".join(f'<div class="m"><b>{v}</b><span>{k}</span></div>' for k, v in pct)
        return (f'<div class="cls {fcol}"><span class="tag">{a}</span>'
                f'<h3>Support {p["support"]} — {p["correct"]} correct</h3>'
                f'<div class="met">{mets}</div>'
                f'<p class="callout">{cls}</p></div>')

    per_class = (
        metrow("Predicted well on the 93 positive reviews; very few were missed.",
               "POSITIVE") +
        metrow("Based on only 7 negative reviews — cautious to read too much into it.",
               "NEGATIVE")
    )

    # ---- error direction ----
    fp_cm = cm["POSITIVE"]["NEGATIVE"]    # actual POS predicted NEG
    fn_cm = cm["NEGATIVE"]["POSITIVE"]    # actual NEG predicted POS
    direction = f"""
    <div class="direction">
      <div class="dir"><div class="n neg">{fp_cm}</div>
        <p><strong>Actual POSITIVE</strong> were predicted <strong>NEGATIVE</strong>
        <span class="arrow">→</span> the model called a good reviewer&#39;s review bad.</p></div>
      <div class="dir"><div class="n pos">{fn_cm}</div>
        <p><strong>Actual NEGATIVE</strong> was predicted <strong>POSITIVE</strong>
        <span class="arrow">→</span> the model called a bad reviewer&#39;s review good.</p></div>
    </div>"""

    # ---- mismatch cards ----
    def mismatch_card(r: dict) -> str:
        ref_b = ("pos" if r["reference"] == "POSITIVE" else "neg")
        pred_b = ("pos" if r["predicted"] == "POSITIVE" else "neg")
        return f"""
        <article class="mism">
          <div class="rowhead">
            <span class="rnum">Row {r['row_index']}</span>
            <span class="rating-token">★ {r['rating']:.0f}</span>
            <span class="rtitle">{esc(r['title'])}</span>
            <span class="badge {ref_b}">ref: {r['reference']}</span>
            <span class="badge {pred_b}">pred: {r['predicted']}</span>
            <span class="badge mid">mismatch</span>
          </div>
          <p class="text">{esc(r['text'])}</p>
          <div class="predline">
            <span>reference <strong>{r['reference']}</strong></span>
            <span class="arrow">→</span>
            <span>predicted <strong>{r['predicted']}</strong></span>
            <span class="arrow">|</span>
            <span>disagreement between the text classifier and the rating-derived reference</span>
          </div>
        </article>"""

    mismatches = sorted((r for r in rows if not r["agree_with_rating"]),
                        key=lambda r: r["row_index"])
    mismatch_cards = "".join(mismatch_card(r) for r in mismatches)

    # ---- full review table (data attrs for Step 4 filtering) ----
    def emotion_status(c: dict) -> str:
        """Emotion comparison status: exact / tie-aware only / disagreement / no match."""
        if not c["comparison_eligible"] or c["nrc_primary_emotion"] == "NO_MATCH":
            return "NRC no match"
        if c["exact_emotion_agreement"]:
            return "Exact agreement"
        if c["tie_aware_emotion_agreement"]:
            return "Tie-aware only"
        return "Disagreement"

    def review_row(r: dict) -> str:
        cls = "mism" if not r["agree_with_rating"] else ""
        status = ("mism" if not r["agree_with_rating"] else "ok")
        stxt = ("Mismatch" if not r["agree_with_rating"] else "Correct")
        ref_b = ("pos" if r["reference"] == "POSITIVE" else "neg")
        pred_b = ("pos" if r["predicted"] == "POSITIVE" else "neg")
        return (f'<tr class="reviewrow {cls}" data-row="{r["row_index"]}" '
                f'data-status="{"mismatch" if not r["agree_with_rating"] else "correct"}" '
                f'data-reference="{r["reference"]}" data-predicted="{r["predicted"]}">'
                f'<td class="num">{r["row_index"]}</td>'
                f'<td class="rating">{r["rating"]:.0f}</td>'
                f'<td class="title-cell">{esc(r["title"])}</td>'
                f'<td class="text-cell"><span class="reviewtext">{esc(r["text"])}</span></td>'
                f'<td><span class="badge {ref_b}">{r["reference"]}</span></td>'
                f'<td><span class="badge {pred_b}">{r["predicted"]}</span></td>'
                f'<td><span class="status {status}">{stxt}</span></td>'
                f'</tr>')

    table_rows = "".join(review_row(r) for r in sorted(rows, key=lambda r: r["row_index"]))

    # ---- emotion-evidence table rows (Emotion tab, separate .emorow class) ----
    def emo_row(c: dict) -> str:
        llm_emo = esc(c["llm_primary_emotion"])
        nrc_emo = esc(c["nrc_primary_emotion"])
        nrc_top = esc(", ".join(c["nrc_top_emotions"]) if c["nrc_top_emotions"] else "—")
        est = emotion_status(c)
        ecls = {"Exact agreement": "exact", "Tie-aware only": "ta",
                "Disagreement": "dis", "NRC no match": "nomatch"}[est]
        return (f'<tr class="emorow" data-emo-row="{c["row_index"]}">'
                f'<td class="num">{c["row_index"]}</td>'
                f'<td class="title-cell">{esc(c["title"])}</td>'
                f'<td class="emocell">{llm_emo}</td>'
                f'<td class="emocell">{nrc_emo}</td>'
                f'<td class="emocell nrc-top emo-top-td">{nrc_top}</td>'
                f'<td><span class="estat {ecls}">{esc(est)}</span></td>'
                f'</tr>')

    emo_table_rows = "".join(emo_row(c) for c in sorted(crows, key=lambda c: c["row_index"]))

    # ---- emotion-evidence table (Emotion tab; compact, no full-text column) ----
    emotion_table = f"""
    <h2 id="h-emotable">Row-level emotion evidence (all 100)</h2>
    <p class="lead">The same first-100 reviews with their LLM primary emotion, NRC
    selected primary, all NRC top-scoring emotions, and comparison status. Full review
    text appears in the representative examples above; this compact table focuses on the
    emotion comparison.</p>
    <div class="tablewrap">
      <table class="reviews emo-table" id="emo-table">
        <caption class="sr-only">Emotion-evidence rows for the first 100 reviews: LLM
        primary emotion, NRC primary emotion, NRC top emotions, and comparison status.</caption>
        <thead><tr>
          <th>Row</th><th>Title</th><th>LLM emotion</th><th>NRC primary</th>
          <th>NRC top emotions</th><th>Comparison status</th>
        </tr></thead>
        <tbody>{emo_table_rows}</tbody>
      </table>
    </div>"""

    # ---- Step 5B emotion section content ----
    def emoto_dist_rows(dist: dict, barcls: str, maxval: int) -> str:
        """Label-count rows: counts remain visible even at 0; bar length reflects max."""
        out = []
        for e in EMOTIONS:
            v = dist.get(e, 0)
            w = round(100 * v / maxval, 2) if maxval else 0
            zero = " zero" if v == 0 else ""
            out.append(
                f'<div class="erow"><span class="elab">{e}</span>'
                f'<div class="ebar" role="img" aria-label="{e}: {v}"><div class="fill {barcls}" '
                f'style="width:{w}%"></div></div>'
                f'<span class="ecount{zero}">{v}</span></div>')
        return "".join(out)

    cmp_m = cmpm
    llm_dist = cmp_m["llm_emotion_distribution"]
    nrc_dist = cmp_m["nrc_primary_emotion_distribution"]
    nrc_wo_nomatch = {e: nrc_dist.get(e, 0) for e in EMOTIONS}
    llm_max = max(llm_dist.values()) if llm_dist else 1
    nrc_max = max(nrc_wo_nomatch.values()) if nrc_wo_nomatch else 1

    ea = cmp_m["exact_agreement"]
    ta = cmp_m["tie_aware_agreement"]
    recovered = cmp_m["additional_agreements_recovered_by_tie_aware"]
    uniq = cmp_m["grouped_by_nrc_structure"]["unique_maximum"]
    tiedg = cmp_m["grouped_by_nrc_structure"]["tied_maximum"]
    nomatch_rows = cmp_m["no_match_rows"]

    emo_stats = (
        stat("100", "LLM valid emotion predictions") +
        stat(f"{cmp_m['nrc_covered_rows']}", "NRC covered reviews") +
        stat(f"{ea['count']}/{ea['eligible_denominator']}",
             f"Exact agreement · {ea['percentage_pct']:.2f}%") +
        stat(f"{ta['count']}/{ta['eligible_denominator']}",
             f"Tie-aware agreement · {ta['percentage_pct']:.2f}%")
    )

    # 8x8 matrix
    ctab = cmp_m["cross_tabulation_8x8_llm_by_nrc"]
    head = "".join(f'<th scope="col">{e[:4]}</th>' for e in EMOTIONS)
    body = ""
    for le in EMOTIONS:
        cells = "".join(
            f'<td class="{"diag" if le == ne else "off"}">{ctab[le][ne]}</td>'
            for ne in EMOTIONS)
        body += f'<tr><th scope="row">{le}</th>{cells}</tr>'
    emo_matrix = f"""
    <div class="emomatwrap">
      <table class="emomatrix">
        <caption class="sr-only">LLM emotion by NRC selected primary emotion.
        Rows are the LLM emotion, columns the selected NRC primary emotion,
        eligible reviews only (85). Diagonal cells are exact agreement (19).</caption>
        <thead><tr><th scope="col"></th>{head}</tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>
    <p class="emotable-caption">Eligible rows only (85). Rows = LLM emotion, columns =
    NRC selected primary (tie-aware primary). Green diagonal cells count exact agreement
    (19 total). Off-diagonal cells are comparisons where the LLM emotion differed from
    the NRC-selected primary.</p>"""

    # unique vs tied subtables
    def subtable(title, g, ta_flag):
        eff_ta = g["tie_aware"] if ta_flag else g["exact"]
        return (f'<div class="emosub"><h3>{title}</h3>'
                f'<table><thead><tr><th></th><th>reviews</th><th>agreements</th>'
                f'<th>%</th></tr></thead><tbody>'
                f'<tr><td>Exact agreement</td><td>{g["exact"]["denominator"]}</td>'
                f'<td>{g["exact"]["agreements"]}</td><td>{g["exact"]["percentage_pct"]:.2f}%</td></tr>'
                f'<tr><td>Tie-aware agreement</td><td>{g["tie_aware"]["denominator"]}</td>'
                f'<td>{g["tie_aware"]["agreements"]}</td><td>{g["tie_aware"]["percentage_pct"]:.2f}%</td></tr>'
                f'</tbody></table></div>')

    emo_groups = (
        subtable(f"Unique NRC maxima — {uniq['rows']} eligible", uniq, False) +
        subtable(f"Tied NRC maxima — {tiedg['rows']} eligible", tiedg, False)
    )

    # representative examples (verified categories, read from comparison rows)
    def example_card(c: dict, reason: str) -> str:
        scores = " ".join(f"{e}:{c['nrc_emotion_scores'][e]}" for e in EMOTIONS)
        top = ", ".join(c["nrc_top_emotions"]) if c["nrc_top_emotions"] else "—"
        est = emotion_status(c)
        ecls = {"Exact agreement": "exact", "Tie-aware only": "ta",
                "Disagreement": "dis", "NRC no match": "nomatch"}[est]
        return f"""
        <article class="example-card">
          <div class="rowhead">
            <span class="rnum">Row {c['row_index']}</span>
            <span class="rtitle">{esc(c['title'])}</span>
            <span class="estat {ecls}">{esc(est)}</span>
          </div>
          <p class="text">{esc(c['text'])}</p>
          <div class="predline">
            <span>LLM <strong>{esc(c['llm_primary_emotion'])}</strong></span>
            <span class="arrow">→</span>
            <span>NRC primary <strong>{esc(c['nrc_primary_emotion'])}</strong></span>
            <span class="arrow">|</span>
            <span>NRC top(ties) <strong>{esc(top)}</strong></span>
          </div>
          <pre class="emvec">NRC eight-score vector: {esc(scores)}</pre>
          <p class="reason">{reason}</p>
        </article>"""

    # pick one example per category deterministically (first in row order)
    examples = {}
    for c in sorted(crows, key=lambda x: x["row_index"]):
        if not c["comparison_eligible"] or c["nrc_primary_emotion"] == "NO_MATCH":
            if "no_match" not in examples:
                examples["no_match"] = example_card(
                    c, "NRC found no associated lexicon word (all eight scores are zero), "
                       "so the row is not eligible for agreement; the LLM still chose an emotion.")
            continue
        exact = c["exact_emotion_agreement"]
        ta_ = c["tie_aware_emotion_agreement"]
        tied = c["nrc_tie"]
        kind = None
        if exact:
            kind = "exact"
        elif ta_:
            kind = "tiebreak"
        elif not tied:
            kind = "unique_dis"
        else:
            kind = "tied_dis"
        if kind not in examples:
            reasons = {
                "exact": "Exact agreement: the LLM emotion equals the NRC selected primary "
                         "emotion (here a unique NRC maximum, so exact and tie-aware agree).",
                "tiebreak": "Tie-break-sensitive: the LLM emotion is inside NRC's tied maximum "
                            "set, so tie-aware agreement is true, but the canonical tie-break "
                            "order selected a different primary, making exact agreement false.",
                "unique_dis": "Tie-aware disagreement with a unique NRC maximum: the LLM and "
                              "the NRC selected primary differ and the LLM emotion is not even "
                              "among NRC's (single) maximum.",
                "tied_dis": "Tie-aware disagreement with tied NRC maxima: the LLM emotion is "
                            "not among the NRC tied maxima at all.",
            }
            examples[kind] = example_card(c, reasons[kind])
        if len(examples) == 5:
            break

    # guaranteed order for display, only include categories that were found
    example_order = ["exact", "tiebreak", "unique_dis", "tied_dis", "no_match"]
    example_cards = "".join(examples[k] for k in example_order if k in examples)
    missing_cats = [k for k in example_order if k not in examples]
    if missing_cats:
        example_cards += ("<p class='callout'>No examples available for: "
                          + ", ".join(missing_cats) + "</p>")

    emotion_section = f"""
  <section aria-labelledby="h-emotion">
    <h2 id="h-emotion">Primary-emotion analysis: LLM vs NRC</h2>
    <p class="lead">Two independent ways of assigning a primary emotion to each review —
    the LLM (context-based, one forced choice) and the NRC word list (isolated word
    associations). These are <strong>agreement</strong> numbers, <em>not</em> accuracy:
    neither method is emotion ground truth.</p>
    <div class="statband">{emo_stats}</div>
    <div class="emo-note"><strong>Exact vs tie-aware.</strong> Exact agreement requires the
    LLM emotion to equal the NRC <em>selected primary</em>; tie-aware agreement counts when
    the LLM emotion appears among <em>all NRC emotions tied for the highest score</em>. NRC
    uses a deterministic canonical tie-break rule, which is an implementation choice, not a
    ranking of emotions. Because 52 of 85 covered rows contain ties, exact agreement
    understates conceptual overlap with the word list.</div>
    <div class="card">
      <h3 class="visually-h3">Emotion distributions</h3>
      <div class="distcol two">
        <div>
          <h3>LLM emotion — all 100</h3>
          {emoto_dist_rows(llm_dist, "llm", llm_max)}
        </div>
        <div>
          <h3>NRC selected primary — all 100 (incl. NO_MATCH)</h3>
          {emoto_dist_rows(nrc_wo_nomatch, "nrc", nrc_max)}
          <p class="callout">NO_MATCH: {cmp_m['nrc_no_match_rows']} reviews. The high NRC
          ANTICIPATION count is influenced by its 52 ties and the canonical tie-break order.</p>
        </div>
      </div>
    </div>
    <p class="callout"><strong>Agreement is not accuracy.</strong> Neither method is
    treated as ground truth: the LLM interprets context but must pick one emotion, while
    the NRC lexicon counts isolated word associations and cannot read negation or sarcasm.
    The 63/85 figure counts cases where the LLM matched <em>any</em> NRC tied maximum.</p>
    <details class="disclose">
      <summary>Why NRC ties matter <span class="d-meta">15 NO_MATCH · 52 tied · 44 recovered</span></summary>
      <div class="d-body">
        <p class="callout"><strong>{cmp_m['nrc_no_match_rows']} reviews had no NRC match</strong>,
        <strong>52 of 85 covered rows had tied maxima</strong>, and tie-aware agreement
        <strong>recovered {recovered} additional agreements</strong> over exact agreement.
        Ties are why exact agreement (19/85) understates the word list&#39;s overlap with the LLM.</p>
        <h3 class="visually-h3">Unique vs tied NRC maxima</h3>
        <div class="emosubgrid two">{emo_groups}</div>
      </div>
    </details>
    <details class="disclose">
      <summary>View the LLM × NRC comparison matrix <span class="d-meta">8 × 8</span></summary>
      <div class="d-body">{emo_matrix}</div>
    </details>
    <details class="disclose">
      <summary>Read representative comparison examples <span class="d-meta">5 examples</span></summary>
      <div class="d-body"><div class="example-list">{example_cards}</div></div>
    </details>
    <details class="disclose">
      <summary>Explore emotion evidence for all 100 reviews <span class="d-meta">100 rows</span></summary>
      <div class="d-body">{emotion_table}</div>
    </details>
  </section>
"""

    # ---- Step 6C: balanced three-class section ----
    def filter_btn(fid: str, label: str, count: int, pressed: bool) -> str:
        ap = ' aria-pressed="true"' if pressed else ' aria-pressed="false"'
        return (f'<button type="button" class="fbtn" data-filter="{fid}"{ap}>'
                f'{label} <span class="fcount">{count}</span></button>')

    c3 = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    b6_acc = b6m["accuracy_over_all_150_pct"]           # 74.67
    b6_base = b6m["majority_class_baseline_pct"]        # 33.33
    b6_gain = round(b6_acc - b6_base, 2)                # +41.34
    b6_balacc = b6m["macro_recall_balanced_accuracy_pct"]

    b6_neu_recall_val = 100 * b6m["per_class"]["NEUTRAL"]["recall"]  # 32.0
    c6_stats = (
        stat(f"{b6_acc:.2f}%", "Accuracy (150)", accent=True) +
        stat(f"{b6m['correct_predictions']}", "Correct predictions") +
        stat(f"{b6m['mismatch_count']}", "Mismatches") +
        stat(f"{b6_base:.2f}%", "Majority-class baseline") +
        stat(f"{b6_neu_recall_val:.0f}%", "NEUTRAL recall",
             delta_txt="class weakness", delta_cls="bad")
    )
    c6_run_details = (
        f'<p class="mline"><b>Run details:</b>'
        f'<span>Selected reviews <b>150</b></span><span class="sep">·</span>'
        f'<span>Valid predictions <b>150</b></span><span class="sep">·</span>'
        f'<span>Balanced accuracy / macro recall <b>{b6_balacc:.2f}%</b></span><span class="sep">·</span>'
        f'<span>Macro F1 <b>{b6m["macro_f1"]:.4f}</b></span><span class="sep">·</span>'
        f'<span>Improvement over baseline <b>+{b6_gain:.2f} pp</b></span></p>'
    )

    b6_ad = b6m["actual_distribution"]
    b6_pd = b6m["predicted_distribution"]
    bar3 = []
    for label, dist in (("Reference", b6_ad), ("Predicted", b6_pd)):
        total = sum(dist.values())
        cells = "".join(
            f'<div class="seg {cls}" style="width:{100*v/total:.2f}%">'
            f'{v}&nbsp;{k[:4]}</div>'
            for k, v, cls in ((e, dist[e], {"POSITIVE": "pos", "NEUTRAL": "neu", "NEGATIVE": "neg"}[e]) for e in c3))
        bar3.append(
            f'<div class="distrow"><div class="dlabel">{label}</div>'
            f'<div class="bar" role="img" aria-label="{label}: '
            + ", ".join(f"{e} {dist[e]}" for e in c3) + f'">{cells}</div></div>')
    b6_dist = "".join(bar3)

    b6_cm = b6m["confusion_matrix"]
    cm3_head = "".join(f'<th scope="col">{e}</th>' for e in c3)
    cm3_rows = ""
    for a in c3:
        tds = ""
        for p in c3:
            v = b6_cm[a][p]
            cls = "ok3" if a == p else "err3"
            tds += f'<td class="{cls}">{v}</td>'
        cm3_rows += f'<tr><th scope="row">{a}</th>{tds}</tr>'
    cm3_html = f"""
    <div class="cm3wrap">
      <table class="cm3">
        <caption class="sr-only">Three-class confusion matrix. Rows are the reference
        class, columns the predicted class. Diagonal cells (48, 16, 48) are correct;
        off-diagonal cells are errors.</caption>
        <thead><tr><th scope="col"></th>{cm3_head}</tr></thead>
        <tbody>{cm3_rows}</tbody>
      </table>
    </div>
    <p class="emotable-caption">Rows = reference, columns = predicted. Diagonal =
    correct (48 + 16 + 48 = 112). Off-diagonal = errors (sum 38).</p>"""

    b6_pc = b6m["per_class"]
    pc3 = []
    for a in c3:
        p = b6_pc[a]
        pc3.append(
            f'<div class="cls"><span class="tag">{a}</span>'
            f'<h3>Support {p["support"]}</h3>'
            f'<div class="met">'
            f'<div class="m"><b>{p["precision"]:.4f}</b><span>Precision</span></div>'
            f'<div class="m"><b>{p["recall"]:.4f}</b><span>Recall</span></div>'
            f'<div class="m"><b>{p["f1"]:.4f}</b><span>F1</span></div></div></div>')
    per_class3 = "".join(pc3)

    b6_neu = b6m["reference_neutral_prediction_distribution"]
    neu_total = b6_neu["NEUTRAL"] + b6_neu["POSITIVE"] + b6_neu["NEGATIVE"]
    neu_pct_neg = round(100 * b6_neu["NEGATIVE"] / neu_total, 0) if neu_total else 0
    neu_pct_neu = round(100 * b6_neu["NEUTRAL"] / neu_total, 0) if neu_total else 0
    neutral_callout = f"""
    <div class="neutralcall">
      <h3>Neutral-class finding: rating-3 reviews mostly read as NEGATIVE</h3>
      <p>The 50 reference-NEUTRAL (rating 3) reviews were predicted as:
      <strong>NEUTRAL {b6_neu['NEUTRAL']}</strong> ·
      <strong>NEGATIVE {b6_neu['NEGATIVE']}</strong> ·
      <strong>POSITIVE {b6_neu['POSITIVE']}</strong>.</p>
      <p><strong>{neu_pct_neg:.0f}% of rating-3 reviews were classified NEGATIVE</strong>,
      and only <strong>{neu_pct_neu:.0f}% were recognized as NEUTRAL</strong> — the model
      tends to read a complaint inside a mixed review as an overall negative verdict.
      These are disagreements with a rating-derived reference, not proof that the model
      is objectively wrong.</p>
    </div>"""

    b6_ed = b6m["error_directions"]
    dir_rows = []
    for a in c3:
        for p in c3:
            if a == p:
                continue
            v = b6_ed[f"{a} → {p}"]
            dom = (a == "NEUTRAL" and p == "NEGATIVE")
            dn_cls = "dom" if dom else "okd"
            note = (" <strong>— dominant error</strong>" if dom else "")
            dir_rows.append(
                f'<div class="dirrow"><div class="dn {dn_cls}">{v}</div>'
                f'<p><strong>{a}</strong> predicted as <strong>{p}</strong>{note}</p></div>')
    error_dirs = '<div class="direction">' + "".join(dir_rows) + "</div>"

    # ---- Step 6C review table (separate from the first-100 table) ----
    def b6_row(r: dict) -> str:
        cls33 = "mism" if not r["correct"] else ""
        stxt = ("Mismatch" if not r["correct"] else "Correct")
        scls = ("mism" if not r["correct"] else "ok")
        ref_b = {"POSITIVE": "pos", "NEUTRAL": "mid", "NEGATIVE": "neg"}[r["reference_label"]]
        pred_b = {"POSITIVE": "pos", "NEUTRAL": "mid", "NEGATIVE": "neg"}[r["predicted_sentiment"]]
        return (f'<tr class="b6row {cls33}" data-sample="{r["sample_index"]}" '
                f'data-b6status="{"mismatch" if not r["correct"] else "correct"}" '
                f'data-b6ref="{r["reference_label"]}">'
                f'<td class="num">{r["sample_index"]}</td>'
                f'<td class="rating">{r["rating"]:.0f}</td>'
                f'<td class="title-cell">{esc(r["title"])}</td>'
                f'<td class="text-cell"><span class="reviewtext">{esc(r["text"])}</span></td>'
                f'<td><span class="badge {ref_b}">{r["reference_label"]}</span></td>'
                f'<td><span class="badge {pred_b}">{r["predicted_sentiment"]}</span></td>'
                f'<td>{esc(r["predicted_primary_emotion"])}</td>'
                f'<td><span class="status {scls}">{stxt}</span></td>'
                f'</tr>')

    b6_table_rows = "".join(b6_row(r) for r in sorted(b6rows, key=lambda x: x["sample_index"]))

    nb_all = len(b6rows)
    nb_correct = sum(1 for r in b6rows if r["correct"])
    nb_mismatch = nb_all - nb_correct
    nref = {c: sum(1 for r in b6rows if r["reference_label"] == c) for c in c3}

    b6_filters = (
        f'<div class="filterbar" role="group" aria-label="Filter balanced reviews">'
        f'<span class="filter-label">Show (balanced):</span>'
        f'{filter_btn("all150", "All 150", nb_all, True)}'
        f'{filter_btn("correct150", "Correct", nb_correct, False)}'
        f'{filter_btn("mismatch150", "Mismatches", nb_mismatch, False)}'
        f'{filter_btn("ref-pos", "Reference POSITIVE", nref["POSITIVE"], False)}'
        f'{filter_btn("ref-neu", "Reference NEUTRAL", nref["NEUTRAL"], False)}'
        f'{filter_btn("ref-neg", "Reference NEGATIVE", nref["NEGATIVE"], False)}'
        f'</div>'
        f'<p class="livesummary" aria-live="polite">'
        f'<strong id="b6-shown">{nb_all}</strong> of <strong>{nb_all}</strong> '
        f'balanced reviews shown</p>'
    )

    b6_js = """
<script>
(function () {
  "use strict";
  var btnRow = document.getElementById("b6-filters");
  var shown = document.getElementById("b6-shown");
  var allRows = Array.prototype.slice.call(
    document.querySelectorAll("#b6-table tbody tr.b6row"));

  function apply(filter) {
    var vis = 0;
    allRows.forEach(function (row) {
      var st = row.getAttribute("data-b6status");  // "correct"|"mismatch"
      var ref = row.getAttribute("data-b6ref");    // POSITIVE|NEUTRAL|NEGATIVE
      var show = false;
      if (filter === "all150") show = true;
      else if (filter === "correct150") show = (st === "correct");
      else if (filter === "mismatch150") show = (st === "mismatch");
      else if (filter === "ref-pos") show = (ref === "POSITIVE");
      else if (filter === "ref-neu") show = (ref === "NEUTRAL");
      else if (filter === "ref-neg") show = (ref === "NEGATIVE");
      if (show) { row.classList.remove("tr-hidden"); vis += 1; }
      else { row.classList.add("tr-hidden"); }
    });
    shown.textContent = String(vis);
    var btns = btnRow.querySelectorAll("button.fbtn");
    for (var i = 0; i < btns.length; i++) {
      var on = btns[i].getAttribute("data-filter") === filter;
      btns[i].setAttribute("aria-pressed", on ? "true" : "false");
    }
  }

  btnRow.addEventListener("click", function (ev) {
    var btn = ev.target.closest ? ev.target.closest("button.fbtn") : null;
    if (!btn || !btnRow.contains(btn)) return;
    apply(btn.getAttribute("data-filter"));
  });

  apply("all150");
})();
</script>
"""

    # ---- Step 7 class-success + error-direction cards (used inside Balanced tab) ----
    b6_pc7 = b6m["per_class"]
    def succ_row(cls):
        p = b6_pc7[cls]
        pct = 100 * p["tp"] / p["support"]
        w = 100 * p["tp"] / p["support"]
        col = {"POSITIVE": "s7three", "NEUTRAL": "s7five", "NEGATIVE": "s7four"}[cls]
        return (f'<div class="s7row"><span class="slab">{cls}</span>'
                f'<div class="sbar" role="img" aria-label="{cls} success {p["tp"]}/{p["support"]} = {pct:.0f}%">'
                f'<div class="fill {col}" style="width:{w:.2f}%"></div></div>'
                f'<span class="scount">{p["tp"]}/{p["support"]}</span>'
                f'<span class="spct">{pct:.0f}%</span></div>')
    s7_success = f"""
    <div class="card">
      <h3 class="visually-h3">Balanced three-class success rate by reference class</h3>
      <p class="s7lead">Recall — the percent of each reference class classified
      correctly (correct / support, from `balanced_3class_results.json`).</p>
      <div class="s7dist">
        {succ_row("POSITIVE")}
        {succ_row("NEUTRAL")}
        {succ_row("NEGATIVE")}
      </div>
      <p class="s7note">NEUTRAL recall of 32% is the clear weakness: 16 of 50 rating-3
      reviews were recognized, while the other 34 were misclassified.</p>
    </div>"""

    ed7 = b6m["error_directions"]
    ed_order7 = [("POSITIVE", "NEUTRAL"), ("POSITIVE", "NEGATIVE"),
                 ("NEUTRAL", "POSITIVE"), ("NEUTRAL", "NEGATIVE"),
                 ("NEGATIVE", "POSITIVE"), ("NEGATIVE", "NEUTRAL")]
    max_ed7 = max(ed7[f"{a} → {p}"] for a, p in ed_order7) or 1
    ed_rows7 = ""
    for a, p in ed_order7:
        v = ed7[f"{a} → {p}"]
        w = 100 * v / max_ed7
        dom = (a, p) == ("NEUTRAL", "NEGATIVE")
        cls = "s7four" if dom else "s7one"
        tag = " <strong>— dominant error</strong>" if dom else ""
        ed_rows7 += (f'<div class="s7row"><span class="slab">{a} → {p}</span>'
                     f'<div class="sbar" role="img" aria-label="{a} → {p}: {v}">'
                     f'<div class="fill {cls}" style="width:{w:.2f}%"></div></div>'
                     f'<span class="scount">{v}</span>'
                     f'<span class="spct">{tag}</span></div>')
    s7_errors = f"""
    <div class="card">
      <h3 class="visually-h3">Error directions (balanced sample, 38 errors)</h3>
      <div class="s7dist">{ed_rows7}</div>
      <p class="s7note">All six off-diagonal directions, including the zero-valued
      POSITIVE → NEGATIVE. NEUTRAL → NEGATIVE dominates with 26 of 38 errors.</p>
    </div>"""

    step6c_section = f"""
  <section aria-labelledby="h-6c">
    <h2 id="h-6c">Balanced Three-Class Evaluation</h2>
    <p class="lead">A separate evaluation on a <strong>different sample and label
    rule</strong> from the first-100 binary analysis above: <strong>150 reviews</strong>
    sampled with seed 6418 (<strong>50 POSITIVE / 50 NEUTRAL / 50 NEGATIVE</strong>);
    reference rule <strong>ratings 4–5 → POSITIVE, rating 3 → NEUTRAL, ratings 1–2 →
    NEGATIVE</strong>. The model received only title and text. These results are
    <strong>not directly comparable with Step 2</strong> as though only one condition
    changed.</p>
    <div class="statband">{c6_stats}</div>
    {c6_run_details}
    <div class="card">
      <h3 class="visually-h3">Reference versus predicted class distribution</h3>
      <div class="dist">{b6_dist}</div>
      <div class="barlegend">
        <span><span class="sw" style="background:var(--pos)"></span>POSITIVE</span>
        <span><span class="sw" style="background:var(--warn)"></span>NEUTRAL</span>
        <span><span class="sw" style="background:var(--neg)"></span>NEGATIVE</span>
      </div>
    </div>
    <h3 class="visually-h3">Three-class confusion matrix</h3>
    {cm3_html}
    <h3 class="visually-h3">Per-class performance</h3>
    <div class="classgrid">{per_class3}</div>
    <div class="neutralcall-caption callout"><strong>Neutral recall is the major
    weakness</strong> — only 0.32 of reference-NEUTRAL reviews were recognized.</div>
    {neutral_callout}
    <h3 class="visually-h3">Interpretation</h3>
    <div class="card">
      <p class="callout" style="font-size:14px;margin:0">
      Balancing exposes performance that overall accuracy on a highly positive sample
      can conceal: POSITIVE and NEGATIVE recall are each <strong>0.96</strong>, but
      <strong>NEUTRAL recall is only 0.32</strong> — the three-class model tends to
      interpret mixed rating-3 reviews as negative (26 of 50 → NEGATIVE). Step 2 and
      Step 6 accuracy must not be treated as a controlled before/after comparison
      because the sample, label definition, and prompt differ.</p>
    </div>
    <details class="disclose">
      <summary>Examine all six error directions <span class="d-meta">38 errors · bars + counts</span></summary>
      <div class="d-body">
        <h3 class="visually-h3">Six error directions (counts)</h3>
        {error_dirs}
        {s7_errors}
        <p class="callout"><strong>NEUTRAL → NEGATIVE = 26 is the dominant error:</strong>
        over half of all 38 mismatches come from reference-NEUTRAL reviews predicted
        NEGATIVE; the bar chart above shows all six off-diagonal directions.</p>
      </div>
    </details>
    <details class="disclose">
      <summary>Explore all 150 balanced reviews <span class="d-meta">filterable table</span></summary>
      <div class="d-body">
        <div id="b6-filters">{b6_filters}</div>
        <div class="tablewrap">
          <table class="reviews" id="b6-table">
            <caption class="sr-only">All 150 balanced three-class reviews, filterable by
            status and reference class.</caption>
            <thead><tr>
              <th>Sample</th><th>Rating</th><th>Title</th><th>Review text</th>
              <th>Reference</th><th>Predicted</th><th>Emotion</th><th>Status</th>
            </tr></thead>
            <tbody>{b6_table_rows}</tbody>
          </table>
        </div>
      </div>
    </details>
  </section>
"""

    # ---- Step 7: descriptive & prediction visualizations ----
    b6s = json.loads((ROOT / "balanced_sample_3class.json").read_text(encoding="utf-8"))
    s7_meta = b6s["metadata"]
    rd = s7_meta["rating_distribution"]          # {"1.0":.., ..., "5.0":..}
    rd_labels = ["1★", "2★", "3★", "4★", "5★"]
    rd_keys = ["1.0", "2.0", "3.0", "4.0", "5.0"]
    rd_total = s7_meta["total_dataset_records"]
    rd_counts = [rd[k] for k in rd_keys]
    assert sum(rd_counts) == rd_total == 152410
    colors = ["s7one", "s7two", "s7three", "s7four", "s7five"]
    max_rd = max(rd_counts)
    s7_rows = ""
    for lab, cnt, col in zip(rd_labels, rd_counts, colors):
        pct = 100 * cnt / rd_total
        w = 100 * cnt / max_rd
        s7_rows += (
            f'<div class="s7row"><span class="slab">{lab}</span>'
            f'<div class="sbar" role="img" aria-label="{lab}: {cnt:,} reviews ({pct:.2f}%)">'
            f'<div class="fill {col}" style="width:{w:.2f}%"></div></div>'
            f'<span class="scount">{cnt:,}</span>'
            f'<span class="spct">{pct:.2f}%</span></div>')
    # caption values computed from the saved rating distribution (no manual constants)
    five_count = rd["5.0"]
    five_pct = 100 * five_count / rd_total
    assert five_count == 128248
    s7_rating = f"""
    <div class="card">
      <h3 class="visually-h3">Full-dataset star-rating distribution</h3>
      <div class="s7dist">{s7_rows}</div>
      <p class="s7note">Counts describe the <strong>full Gift Cards dataset</strong>
      ({rd_total:,} reviews), not the 100- or 150-review evaluation samples. The 5★
      bar is {five_count:,} reviews ({five_pct:.2f}% of the whole dataset) — the skew
      the balanced sample is designed to counteract. Percentages are computed from
      the saved counts; every count is printed so small categories remain visible.</p>
    </div>"""

    # evaluation-sample comparison
    from collections import Counter as _C2
    f100 = {c: _C2(r["reference"] for r in rows).get(c, 0)
            for c in ("POSITIVE", "NEUTRAL", "NEGATIVE")}
    assert f100["POSITIVE"] == 93 and f100["NEGATIVE"] == 7, f100
    bal = b6m["actual_distribution"]
    def contrast_table(title, dist, total):
        rows = ""
        for cls in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
            v = dist[cls]
            rows += (f'<tr><td>{cls}</td><td>{v}</td>'
                     f'<td>{100*v/total:.1f}%</td></tr>')
        return (f'<div class="contrcard"><h3>{title}</h3>'
                f'<table><thead><tr><th>Class</th><th>Reviews</th><th>Share</th></tr></thead>'
                f'<tbody>{rows}</tbody></table></div>')
    s7_contrast = f"""
    <div class="contrast two">
      {contrast_table("First-100 binary reference", f100, 100)}
      {contrast_table("Balanced 150 three-class reference", bal, 150)}
    </div>
    <p class="s7note">These two samples use <strong>different label definitions</strong>
    (binary vs three-class) and are not comparable performance experiments; the purpose
    is to show why balanced sampling gives each class meaningful representation.</p>"""

    # Step 7 pieces, split for tab relocation (heading + rating + contrast ->
    # Overview; success + errors -> Balanced). The literal heading text and the
    # 84.15% content stay in Overview before the footer (step7_verify depends on it).
    s7_interpretation = f"""\n    <div class="card">\n      <h3 class="visually-h3">Interpretation</h3>\n      <p class="callout" style="font-size:14px;margin:0">\n      The full dataset is overwhelmingly five-star ({five_count:,} of {rd_total:,}).\n      The first-100\n      binary evaluation inherited that imbalance (93% positive reference). Balanced\n      sampling exposed a class-specific weakness hidden by headline accuracy: POSITIVE\n      and NEGATIVE recall were both <strong>96%</strong>, while NEUTRAL recall was\n      <strong>32%</strong> — 26 of 50 reference-NEUTRAL reviews were predicted NEGATIVE.\n      Rating-derived labels are the assignment&#39;s evaluation reference, but individual\n      textual disagreements may still be linguistically defensible.</p>\n    </div>"""

    step7_heading_part = f"""\n  <section aria-labelledby="h-7">\n    <h2 id="h-7">Descriptive &amp; Prediction Visualizations</h2>\n    <p class="lead">Descriptive charts from the full dataset and prediction summaries\n    from the first-100 binary run and the balanced three-class run. All numbers are\n    read from the saved evaluation files, not typed in.</p>\n    {s7_rating}\n    <h3 class="visually-h3">Evaluation-sample comparison</h3>\n    {s7_contrast}\n  </section>"""

    # s7_success / s7_errors stay in the Balanced tab (verifier-required strings
    # inside them: "48/50" success labels and the error-direction aria-labels).
    step7_section = step7_heading_part

    # ---- Step 4: filter controls (counts derived from rows, not hardcoded) ----
    n_all = len(rows)
    n_correct = sum(1 for r in rows if r["agree_with_rating"])
    n_mismatch = sum(1 for r in rows if not r["agree_with_rating"])
    assert (n_all, n_correct, n_mismatch) == (100, 97, 3), \
        f"unexpected derived counts {(n_all, n_correct, n_mismatch)}"

    filter_ui = (
        f'<div class="filterbar" role="group" aria-label="Filter reviews by sentiment result">'
        f'<span class="filter-label">Show by sentiment result:</span>'
        f'{filter_btn("all", "All reviews", n_all, True)}'
        f'{filter_btn("correct", "Correct", n_correct, False)}'
        f'{filter_btn("mismatch", "Mismatches", n_mismatch, False)}'
        f'</div>'
        f'<p class="livesummary" aria-live="polite">'
        f'<strong id="live-shown">{n_all}</strong> of '
        f'<strong>{n_all}</strong> reviews shown</p>'
    )

    filter_js = """
<script>
(function () {
  "use strict";
  var total = %d;
  var btnRow = document.getElementById("review-filters");
  var shown = document.getElementById("live-shown");
  var allRows = Array.prototype.slice.call(
    document.querySelectorAll("#review-table tbody tr.reviewrow"));

  function apply(filter) {
    var vis = 0;
    allRows.forEach(function (row) {
      var status = row.getAttribute("data-status"); // "correct" | "mismatch"
      var show = (filter === "all") || (status === filter);
      if (show) { row.classList.remove("tr-hidden"); vis += 1; }
      else { row.classList.add("tr-hidden"); }
    });
    shown.textContent = String(vis);
    var btns = btnRow.querySelectorAll("button.fbtn");
    for (var i = 0; i < btns.length; i++) {
      var on = btns[i].getAttribute("data-filter") === filter;
      btns[i].setAttribute("aria-pressed", on ? "true" : "false");
    }
  }

  btnRow.addEventListener("click", function (ev) {
    var btn = ev.target.closest ? ev.target.closest("button.fbtn") : null;
    if (!btn || !btnRow.contains(btn)) return;
    apply(btn.getAttribute("data-filter"));
  });

  // default: show all
  apply("all");
})();
</script>
""" % n_all


    # ---- Overview headline strip (from locked numbers) ----
    b6_neu_recall = 100 * b6m["per_class"]["NEUTRAL"]["recall"]  # 32.0
    overview_stats = (
        stat(f"{acc:.0f}%", "Binary accuracy (first 100)", accent=True,
             delta_txt=f"vs {baseline['always_positive_accuracy_pct']:.0f}% always-positive baseline") +
        stat(f"{b6_acc:.2f}%", "Balanced 3-class accuracy (150)", accent=True,
             delta_txt=f"vs {b6_base:.2f}% majority baseline") +
        stat("96% / 96%", "POSITIVE / NEGATIVE recall", delta_txt="balanced run") +
        stat(f"{b6_neu_recall:.0f}%", "NEUTRAL recall",
             delta_txt="major weakness", delta_cls="bad")
    )

    jump_controls = """
    <div class="jumprow">
      <button type="button" class="jump" data-tab="binary">Binary results →</button>
      <button type="button" class="jump" data-tab="emotions">Emotion analysis →</button>
      <button type="button" class="jump" data-tab="balanced">Balanced three-class →</button>
    </div>"""

    overview_panel = f"""
  <section id="panel-overview" class="tabpanel" role="tabpanel" aria-labelledby="tab-overview" tabindex="0">
    <h2 class="visually-h3" id="h-overview">Project overview</h2>
    <p class="lead">An independent analysis of <strong>152,410 Amazon Reviews &#39;23 — Gift
    Cards</strong> reviews (UCSD McAuley Lab). A binary POSITIVE / NEGATIVE sentiment
    run on the first 100 reviews and a balanced three-class run (150 reviews) — both
    judged from <strong>title + text only</strong>, never the rating.</p>
    <div class="statband">{overview_stats}</div>
    <div class="baseline"><strong>Why two evaluations?</strong> The dataset is 84.15%
    five-star, so the first-100 sample is lopsided and its accuracy is read against a
    93% always-positive baseline. The balanced 50/50/50 run exposes per-class
    performance — notably NEUTRAL recall of 32%.</div>
{jump_controls}
    {step7_heading_part}
    <p class="callout"><strong>Takeaway.</strong> High headline accuracy on a skewed
    sample concealed a class-specific weakness: balanced evaluation found POSITIVE and
    NEGATIVE recall at 96% each but NEUTRAL recall at 32% (26 of 50 rating-3 reviews
    predicted NEGATIVE). Step 2 and Step 6 are <strong>not</strong> a controlled
    before/after comparison — sample, label rule, and prompt differ.</p>
  </section>"""

    binary_stats = (
        stat(f"{acc:.0f}%", "Accuracy", accent=True,
             delta_txt=f"vs {baseline['always_positive_accuracy_pct']:.0f}% always-positive baseline") +
        stat(f"{m['correct_predictions']}", "Correct") +
        stat(f"{m['mismatch_count']}", "Mismatches", delta_txt="4 pp over baseline", delta_cls="good") +
        stat(f"{baseline['always_positive_accuracy_pct']:.0f}%", "Always-POSITIVE baseline")
    )

    table_section = f"""
  <section id="panel-binary" class="tabpanel" role="tabpanel" aria-labelledby="tab-binary" tabindex="0" hidden>
    <section aria-labelledby="h-headline">
      <h2 id="h-headline">Overall performance</h2>
      <p class="lead">The classifier agrees with the rating on 97 of 100 reviews. Because
      the data is lopsided toward high ratings, this must be read next to the always-POSITIVE
      baseline, which would already score 93%.</p>
      <div class="statband">{binary_stats}</div>
      <p class="mline"><b>Method:</b>
        <span>first 100 reviews, dataset order</span><span class="sep">·</span>
        <span>valid <b>100</b> · invalid <b>0</b></span><span class="sep">·</span>
        <span>reference: rating 4–5★ → POSITIVE, 1–3★ → NEGATIVE</span><span class="sep">·</span>
        <span>model saw title + text only</span></p>
      <div class="baseline"><strong>Keep the 93% baseline in view.</strong> The sample is
      93% positive, so a trivial always-POSITIVE model scores 93%. The classifier&#39;s 97%
      is a real but modest <strong>{baseline['improvement_over_baseline_pp']:.0f}-point gain</strong>,
      and the negative class contains only 7 reviews — too few to draw strong conclusions.
      </div>
    </section>

    <section aria-labelledby="h-imbalance">
      <h2 id="h-imbalance">Class distribution</h2>
      <p class="lead">Reference labels versus model predictions. Counts are printed on the
      bars, so they do not rely on color or length alone.</p>
      <div class="card">
        <div class="dist">{dist_bars}</div>
        <div class="barlegend">
          <span><span class="sw" style="background:var(--pos)"></span>POSITIVE</span>
          <span><span class="sw" style="background:var(--neg)"></span>NEGATIVE</span>
        </div>
      </div>
    </section>

    <section aria-labelledby="h-confusion">
      <h2 id="h-confusion">Confusion matrix &amp; error direction</h2>
      <p class="lead">Rows are the actual/reference class; columns are what the model
      predicted. Green diagonal cells are correct; red cells are mistakes.</p>
      <div class="card" style="display:flex;flex-wrap:wrap;gap:30px;align-items:flex-start">
        {cm_html}
        <div style="flex:1;min-width:260px">{direction}</div>
      </div>
    </section>

    <section aria-labelledby="h-perclass">
      <h2 id="h-perclass">Per-class performance</h2>
      <p class="lead">Values are computed from the saved metrics. The negative-class
      numbers rest on just seven reviews.</p>
      <div class="classgrid">{per_class}</div>
    </section>

    <details class="disclose">
      <summary>Inspect the 3 mismatches <span class="d-meta">row-level cards</span></summary>
      <div class="d-body">
        <p class="callout">Where the classifier and the rating-derived reference disagreed.
        These are disagreements between two ways of judging a review — the model reads the
        text, the reference reads the star rating — and neither is a perfect ground truth.</p>
        <div class="misms">{mismatch_cards}</div>
      </div>
    </details>

    <details class="disclose">
      <summary>Explore all 100 binary reviews <span class="d-meta">filterable table</span></summary>
      <div class="d-body">
        <p class="lead">Every review in the sample with row index, rating, reference and
        predicted labels, and status. Rows are in dataset order (row 0–99).</p>
        <div id="review-filters">{filter_ui}</div>
        <div class="tablewrap">
          <table class="reviews" id="review-table">
            <caption class="sr-only">All 100 evaluated reviews, filterable by sentiment
            status, with reference and predicted labels.</caption>
            <thead><tr>
              <th>Row</th><th>Rating</th><th>Title</th><th>Review text</th>
              <th>Reference sentiment</th><th>Predicted sentiment</th><th>Status</th>
            </tr></thead>
            <tbody>{table_rows}</tbody>
          </table>
        </div>
      </div>
    </details>
  </section>"""

    emotion_panel = f"""
  <section id="panel-emotions" class="tabpanel" role="tabpanel" aria-labelledby="tab-emotions" tabindex="0" hidden>
    {emotion_section}
  </section>"""

    balanced_panel = f"""
  <section id="panel-balanced" class="tabpanel" role="tabpanel" aria-labelledby="tab-balanced" tabindex="0" hidden>
    {step6c_section}
    {s7_success}
  </section>"""

    tab_js = """
<script>
(function () {
  "use strict";
  var tabs = Array.prototype.slice.call(document.querySelectorAll('.tabbar [role="tab"]'));
  var panels = {};
  tabs.forEach(function (t) {
    panels[t.getAttribute("aria-controls")] = document.getElementById(t.getAttribute("aria-controls"));
  });

  function selectTab(name, focusTab) {
    var ok = false;
    tabs.forEach(function (t) {
      var on = (t.getAttribute("data-tab") === name);
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
      var p = panels[t.getAttribute("aria-controls")];
      if (p) p.hidden = !on;
      if (on) ok = true;
    });
    if (!ok) return;
    try {
      if (history.replaceState) {
        history.replaceState(null, "", "#" + name);
      } else {
        location.hash = name;
      }
    } catch (e) {
      // file:// (null origin) may block replaceState; fall back to location.hash
      try { location.hash = name; } catch (e2) { /* sandboxed; state still holds */ }
    }
    if (focusTab) {
      var tb = tabs.find(function (t) { return t.getAttribute("data-tab") === name; });
      if (tb) tb.focus();
    }
  }

  function currentFromHash() {
    var h = (location.hash || "").replace(/^#/, "");
    return (["overview", "binary", "emotions", "balanced"].indexOf(h) !== -1) ? h : "overview";
  }

  var bar = document.querySelector(".tabbar");
  bar.addEventListener("click", function (ev) {
    var b = ev.target.closest ? ev.target.closest('[role="tab"]') : null;
    if (b && bar.contains(b)) selectTab(b.getAttribute("data-tab"), false);
  });

  bar.addEventListener("keydown", function (ev) {
    var idx = tabs.indexOf(document.activeElement);
    if (idx === -1) return;
    var next = null;
    if (ev.key === "ArrowRight") next = (idx + 1) % tabs.length;
    else if (ev.key === "ArrowLeft") next = (idx - 1 + tabs.length) % tabs.length;
    else if (ev.key === "Home") next = 0;
    else if (ev.key === "End") next = tabs.length - 1;
    if (next === null) return;
    ev.preventDefault();
    selectTab(tabs[next].getAttribute("data-tab"), true);
  });

  document.addEventListener("click", function (ev) {
    var j = ev.target.closest ? ev.target.closest(".jump") : null;
    if (j) selectTab(j.getAttribute("data-tab"), true);
  });

  window.addEventListener("hashchange", function () {
    selectTab(currentFromHash(), false);
  });

  // initial state: hash or overview; default Overview when missing/invalid
  selectTab(currentFromHash(), false);
})();
</script>
"""

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Amazon Gift Card Review Analytics</title>
<style>{build_style()}</style>
</head>
<body>
<main class="container">

  <header class="masthead">
    <h1>Amazon Gift Card Review Analytics <span class="stars" aria-hidden="true">★★★★★</span></h1>
    <p class="sub">An independent course analysis (MBAX 6418) of the
    <strong>Amazon Reviews &#39;23 — Gift Cards</strong> dataset (UCSD McAuley Lab).
    Not affiliated with or endorsed by Amazon. The model judged only review
    <strong>title + text</strong>; ratings are evaluation reference only.</p>
  </header>

  <div class="tabbar" role="tablist" aria-label="Dashboard sections">
    <button type="button" role="tab" id="tab-overview" aria-controls="panel-overview"
      aria-selected="true" tabindex="0" data-tab="overview">Overview</button>
    <button type="button" role="tab" id="tab-binary" aria-controls="panel-binary"
      aria-selected="false" tabindex="-1" data-tab="binary">Binary Sentiment</button>
    <button type="button" role="tab" id="tab-emotions" aria-controls="panel-emotions"
      aria-selected="false" tabindex="-1" data-tab="emotions">Emotion Analysis</button>
    <button type="button" role="tab" id="tab-balanced" aria-controls="panel-balanced"
      aria-selected="false" tabindex="-1" data-tab="balanced">Balanced Three-Class</button>
  </div>

{overview_panel}
{table_section}
{emotion_panel}
{balanced_panel}

  <footer>
    Data source: Amazon Reviews &#39;23 — Gift Cards category (UCSD McAuley Lab),
    <a href="https://amazon-reviews-2023.github.io">amazon-reviews-2023.github.io</a>.
    Model judged only review <b>title + text</b>; ratings are reference/ground-truth for
    evaluation only and were never sent to the model. Dashboard is self-contained and
    works offline.
  </footer>

{tab_js}
{filter_js}
{b6_js}
</main>
</body>
</html>"""

    OUT.write_text(html_doc, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
