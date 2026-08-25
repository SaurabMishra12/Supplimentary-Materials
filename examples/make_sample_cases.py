#!/usr/bin/env python3
"""
Build data/sample_cases.json: a small, readable slice of the released records that
follows individual episodes through the whole pipeline.

Each case carries one target query, the adversarial document inserted beside its
top-50 clean candidates, whether that document survived each detector at the 1%
target, and - where the episode is part of the detector-conditioned replay - the
exposure and compliance indicators behind Tables 5 and 6.
"""
import ast
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REC = ROOT / "data" / "records"
COR = ROOT / "data" / "attack_corpus"

def chunks(raw):
    """The `poison` column stores a list of chunks: three for A6, one otherwise."""
    if isinstance(raw, str):
        try:
            v = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return [raw]
        return list(v) if isinstance(v, (list, tuple)) else [str(v)]
    return list(raw) if isinstance(raw, (list, tuple)) else [str(raw)]


docs = pd.read_parquet(COR / "attack_documents.parquet")
a5 = pd.read_parquet(COR / "a5_documents.parquet")
sur = pd.read_parquet(REC / "candidate_survival_per_query.parquet")
dc = pd.read_parquet(REC / "detector_conditioned_indicators.parquet")

sur01 = sur[sur.target_fpr == 0.01]
main = dc[(dc.model == "Qwen/Qwen2.5-3B-Instruct") & (dc.system == "plain") &
          (dc.position == 0) & (dc.target_fpr == 0.01)]

cases = []
for attack in sorted(docs.attack.unique()):
    g = docs[docs.attack == attack].sort_values("qid")
    for _, row in g.head(2).iterrows():
        s = sur01[(sur01.attack == attack) & (sur01.qid == row.qid)]
        b = main[(main.attack == attack) & (main.qid == row.qid)]
        cases.append({
            "attack": attack,
            "qid": int(row.qid),
            "query": row.query,
            "poison_chunks": chunks(row.poison),
            "poison_length_chars": sum(len(c) for c in chunks(row.poison)),
            "survived_top10_at_1pct_fpr": {r.defense: bool(round(r.asr_retrieval))
                                           for _, r in s.iterrows()},
            "behavioural_replay_3b_plain_position0": {
                r.defense: {"payload_reached_context": int(r.E),
                            "model_emitted_canary": int(r.C),
                            "both": int(r.E_and_C)}
                for _, r in b.iterrows() if r.defense in ("D0_none", "D3_distilbert")},
        })

for defense in ["D1_3feat_tiny", "D3_distilbert", "D4_guard_zeroshot"]:
    row = a5[(a5.defense == defense) & (a5.target_fpr == 0.01)].sort_values("qid").iloc[0]
    s = sur01[(sur01.attack == "A5_score_guided") & (sur01.qid == row.qid) &
              (sur01.defense == defense)]
    cases.append({
        "attack": "A5_score_guided",
        "optimised_against": defense,
        "qid": int(row.qid),
        "poison_chunks": chunks(row.poison),
        "poison_length_chars": sum(len(c) for c in chunks(row.poison)),
        "final_detector_score": float(row.final_score),
        "detector_calls_consumed": int(row.detector_calls),
        "drove_score_below_threshold": bool(row.evaded),
        "survived_top10_at_1pct_fpr": {defense: bool(round(s.asr_retrieval.iloc[0]))
                                       if len(s) else None},
    })

out = {
    "description": (
        "Representative cases from the released records. 'survived_top10_at_1pct_fpr' is the "
        "candidate-survival outcome behind Table 3; the replay block is the pair of indicators "
        "behind Tables 5 and 6. A5 documents are re-optimised separately against each detector, "
        "so an A5 case names the detector it was built against."),
    "n_cases": len(cases),
    "cases": cases,
}
(ROOT / "data" / "sample_cases.json").write_text(json.dumps(out, indent=2))
print(f"wrote data/sample_cases.json with {len(cases)} cases")
