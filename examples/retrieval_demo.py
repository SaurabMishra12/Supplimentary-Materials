#!/usr/bin/env python3
"""
Stage 1 of Figure 1: query -> dense -> BM25 -> fusion -> reranking -> filter -> metric.

    python examples/retrieval_demo.py

Part A walks one query through the ranking stages on a small illustrative
candidate set, so the normalisation convention, the fusion rule, the reranking
step and nDCG@10 can be read off directly. The numbers in Part A are made up for
the walk-through; they are not paper numbers.

Part B runs the same fusion arithmetic on the released per-query records and
reproduces the fusion-weight results of Section 4: the mean curve over the
21-point grid, the argmax weight per collection, the flatness statistic behind
Figure 3(b), and the policy comparison of Table 8.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REC = ROOT / "data" / "records"
GRID = np.round(np.linspace(0, 1, 21), 2)


# ---------------------------------------------------------------------------
# the fusion and metric primitives, exactly as configs/retrieval.yaml specifies
# ---------------------------------------------------------------------------
def minmax(scores):
    """Min-max normalise within a retrieved list. A constant list maps to zeros."""
    s = np.asarray(scores, dtype=float)
    lo, hi = s.min(), s.max()
    return np.zeros_like(s) if hi == lo else (s - lo) / (hi - lo)


def convex_fuse(dense, lexical, alpha):
    """s(d) = alpha * s_dense(d) + (1 - alpha) * s_lex(d).

    Both channels are min-max normalised within their own retrieved list first.
    A document returned by one channel only receives 0 from the other, which is
    the standard imputation convention for score-based hybrid fusion.
    """
    docs = sorted(set(dense) | set(lexical))
    dn = dict(zip(dense, minmax(list(dense.values())))) if dense else {}
    ln = dict(zip(lexical, minmax(list(lexical.values())))) if lexical else {}
    return {d: alpha * dn.get(d, 0.0) + (1 - alpha) * ln.get(d, 0.0) for d in docs}


def rrf(dense, lexical, k=60):
    """Reciprocal rank fusion: rank-based, weight-free."""
    docs = sorted(set(dense) | set(lexical))
    out = {}
    for name, run in (("dense", dense), ("lex", lexical)):
        for rank, d in enumerate(sorted(run, key=run.get, reverse=True), start=1):
            out[d] = out.get(d, 0.0) + 1.0 / (k + rank)
    return {d: out.get(d, 0.0) for d in docs}


def ndcg_at_10(ranking, relevance):
    """Standard nDCG@10 with binary or graded relevance."""
    gains = [relevance.get(d, 0) for d in ranking[:10]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(relevance.values(), reverse=True)[:10]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg


def rank(scores):
    return sorted(scores, key=scores.get, reverse=True)


# ---------------------------------------------------------------------------
# Part A -- one query through the stages (illustrative candidate set)
# ---------------------------------------------------------------------------
print("=" * 78)
print("Part A: the stages, on an illustrative candidate set (not paper numbers)")
print("=" * 78)

dense_hits = {"d1": 0.81, "d3": 0.74, "d7": 0.66, "d2": 0.61, "d9": 0.55}
lex_hits = {"d3": 14.2, "d5": 11.8, "d1": 9.4, "d8": 8.1, "d2": 6.9}
relevance = {"d3": 2, "d1": 1, "d5": 1}
poison = "d99"

print("\ndense channel (inner product on L2-normalised embeddings)")
for d, s in dense_hits.items():
    print(f"    {d}  raw {s:5.2f}   normalised {dict(zip(dense_hits, minmax(list(dense_hits.values()))))[d]:5.3f}")
print("\nlexical channel (BM25, k1=0.9, b=0.4)")
for d, s in lex_hits.items():
    print(f"    {d}  raw {s:5.2f}   normalised {dict(zip(lex_hits, minmax(list(lex_hits.values()))))[d]:5.3f}")

print("\nfused ranking at three weights")
for alpha in (0.0, 0.5, 1.0):
    print(f"    alpha={alpha:.1f}  ->  {rank(convex_fuse(dense_hits, lex_hits, alpha))}")
print(f"    RRF k=60  ->  {rank(rrf(dense_hits, lex_hits))}")

fused = convex_fuse(dense_hits, lex_hits, 0.5)
print(f"\nnDCG@10 of the alpha=0.5 ranking: {ndcg_at_10(rank(fused), relevance):.4f}")

print("\nreranking: a cross-encoder rescores the top-k candidates and only permutes them.")
ce = {"d3": 0.93, "d1": 0.55, "d5": 0.71, "d2": 0.20, "d7": 0.12, "d8": 0.09, "d9": 0.05}
k = 5
head = rank(fused)[:k]
reranked = sorted(head, key=lambda d: ce.get(d, 0.0), reverse=True) + rank(fused)[k:]
print(f"    budget {k}: {rank(fused)} -> {reranked}")
print(f"    nDCG@10 after reranking: {ndcg_at_10(reranked, relevance):.4f}")

print("\nfiltering: an adversarial candidate is inserted, scored, and dropped if")
print("its suspicion score reaches the calibrated threshold (see filter_demo.py).")
with_poison = reranked[:2] + [poison] + reranked[2:]
threshold, poison_score = 0.42, 0.88
kept = [d for d in with_poison if not (d == poison and poison_score >= threshold)]
print(f"    with poison at rank 2 : {with_poison}")
print(f"    poison score {poison_score} >= threshold {threshold}  ->  rejected")
print(f"    after filtering       : {kept}")
print(f"    nDCG@10 on clean relevance is unchanged at {ndcg_at_10(kept, relevance):.4f};")
print("    the filter's utility cost is what it removes when no attack is present.")


# ---------------------------------------------------------------------------
# Part B -- the released per-query records
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("Part B: the same arithmetic on the released records (paper numbers)")
print("=" * 78)

curves = pd.concat([pd.read_parquet(REC / "alpha_curves_fiqa.parquet"),
                    pd.read_parquet(REC / "alpha_curves_scifact.parquet")])
C = np.vstack(curves.curve.to_numpy())

print("\nper-collection sweep over the 21-point grid (Figure 3a)")
for ds, g in curves.groupby("dataset"):
    mean_curve = np.vstack(g.curve.to_numpy()).mean(axis=0)
    best = int(mean_curve.argmax())
    print(f"    {ds:8s} n={len(g):4d}   argmax alpha={GRID[best]:.2f}   "
          f"mean nDCG@10 at optimum {mean_curve[best]:.4f}   at alpha=1 (dense) {mean_curve[-1]:.4f}")

spread = C.max(axis=1) - C.min(axis=1)
flat = float((spread < 0.01).mean())
print(f"\nflatness (Figure 3b): on {flat:.1%} of the {len(C)} tuning queries the entire grid")
print(f"    moves nDCG@10 by less than 0.01, so the argmax weight is weakly identified there.")
print(f"    Regret confinement (Proposition 1): epsilon * P(flat) = {0.01 * flat:.4f} nDCG@10,")
print(f"    so at least {1 - 0.01 * flat / 0.0809:.1%} of the oracle's +0.0809 advantage over dense")
print("    must be earned on the queries where the weight actually changes the ranking.")

print("\npolicies on the tuning split, recomputed from the curves (Table 8)")
for label, value in [("BM25 (alpha=0)", C[:, 0].mean()),
                     ("Dense (alpha=1)", C[:, -1].mean()),
                     ("RRF (k=60)", curves.ndcg_rrf.mean()),
                     ("Fixed alpha*=0.65", C[:, 13].mean()),
                     ("Oracle alpha (hindsight ceiling)", C.max(axis=1).mean())]:
    print(f"    {label:34s} {value:.4f}")
print("\n    The oracle is a bound within this scalar convex family on this grid.")
print("    It does not bound continuous weights, rank-aware fusion, or other candidate policies.")

print("\nmatched reranking budgets (Table 2, macro-average over six collections)")
pq = pd.read_parquet(REC / "retrieval_per_query.parquet")
macro = (pq.pivot_table(index=["dataset", "qid", "budget"], columns="system", values="ndcg10")
         .groupby(["dataset", "budget"]).mean().groupby("budget").mean())
for s, label in [("bm25", "BM25"), ("dense", "Dense"), ("rrf", "RRF (k=60)"),
                 ("alpha_fixed", "Fixed alpha"), ("alpha_learned", "Learned alpha"),
                 ("alpha_oracle", "Oracle (first stage)")]:
    print(f"    {label:22s} " + "  ".join(f"budget {b}: {macro.loc[b, s]:.3f}" for b in (0, 50, 100)))
print("\n    Systems are only comparable at equal cross-encoder forward passes. The weight")
print("    that ranks best before reranking is not the weight that feeds the reranker best.")
