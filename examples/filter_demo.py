#!/usr/bin/env python3
"""
Stage 2 of Figure 1: calibrating a retrieval-layer filter and reading its results.

    python examples/filter_demo.py

Part A implements the operating-point protocol of Section 3.5 and Table 7 in full:
the threshold rule, the keep rule, the realised false-positive rate, the interval
constructions, and the simultaneous bound on a selected maximum (Proposition 3).

Part B applies it to the released records and reproduces the candidate-survival
matrix at the 1% target (Table 3), the operating points (Table 12), and the cost
of the adaptive attacker (Table 13).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta

ROOT = Path(__file__).resolve().parent.parent
REC = ROOT / "data" / "records"
Z = 1.959963984540054


# ---------------------------------------------------------------------------
# Part A -- the protocol
# ---------------------------------------------------------------------------
def calibrate_threshold(clean_scores, target_fpr):
    """Threshold = the empirical (1 - target) quantile of the clean calibration scores.

    Scores are higher-is-more-suspicious throughout.
    """
    return float(np.quantile(np.asarray(clean_scores, dtype=float), 1 - target_fpr))


def keep(score, threshold):
    """A document is kept when its score is STRICTLY below the threshold.

    Ties reject. With discrete or heavily tied scores this is one reason a
    realised rate need not equal its target.
    """
    return float(score) < float(threshold)


def realised_fpr(held_out_clean_scores, threshold):
    """Measured on a disjoint clean split, never on the calibration split."""
    s = np.asarray(held_out_clean_scores, dtype=float)
    return float((s >= threshold).mean())


def clopper_pearson(k, n, alpha=0.05):
    """Exact interval for an individual binomial rate."""
    lo = 0.0 if k == 0 else float(_beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def cp_lower(k, n, alpha):
    """One-sided Clopper-Pearson lower bound at level 1 - alpha."""
    return 0.0 if k == 0 else float(_beta.ppf(alpha, k, n - k + 1))


def simultaneous_max_bound(successes, trials, alpha=0.05):
    """Proposition 3: a valid lower bound on max_k p_k over K evaluated cells.

    The maximal cell is selected by being maximal, so its own interval is not
    valid for its own parameter. A union bound over the K one-sided bounds at
    level 1 - alpha/K is.
    """
    K = len(successes)
    return max(cp_lower(k, n, alpha / K) for k, n in zip(successes, trials))


print("=" * 78)
print("Part A: the operating-point protocol")
print("=" * 78)

rng = np.random.default_rng(42)
clean_cal = rng.normal(0.20, 0.10, 1500)      # calibration half
clean_eval = rng.normal(0.20, 0.10, 1500)     # disjoint evaluation half

print("\ntarget -> threshold -> realised rate, on 1500 + 1500 illustrative clean scores")
for target in (0.001, 0.01, 0.05):
    t = calibrate_threshold(clean_cal, target)
    print(f"    target {target:<6} threshold {t:6.3f}   realised on the held-out half "
          f"{realised_fpr(clean_eval, t):.4f}")

print("\nwhy realised != target: ties reject, and a detector whose score saturates")
print("has a floor. D6 is the case in the paper: its keyword term saturates at 1,")
print("so it realises 0.049 at both the 0.1% and the 1% targets and is reported as")
print("an uncalibrated control rather than at an operating point it cannot reach.")

tied = np.r_[np.zeros(1400), np.ones(100)]
t = calibrate_threshold(tied, 0.01)
print(f"    saturating example: threshold at the 1% target is {t:.3f}, "
      f"realised {realised_fpr(tied, t):.4f}")

print("\nintervals")
print(f"    individual rate 40/100 : Clopper-Pearson {clopper_pearson(40, 100)[0]:.3f} "
      f"to {clopper_pearson(40, 100)[1]:.3f}")
print(f"    selected maximum, K=6  : simultaneous one-sided lower bound "
      f"{simultaneous_max_bound([295] + [280] * 5, [300] * 6):.3f}")
print("    A per-cell interval and a bound on a selected maximum are different objects.")


# ---------------------------------------------------------------------------
# Part B -- the released records
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("Part B: the released records")
print("=" * 78)

sur = pd.read_parquet(REC / "candidate_survival_per_query.parquet")
cal = pd.read_csv(REC / "detector_calibration.csv")

DEF = ["D1_3feat_tiny", "D1b_3feat_trained", "D2_embed_probe", "D3_distilbert",
       "D4_guard_zeroshot", "D5_perplexity", "D6_ensemble"]
A16 = ["A1_query_conditioned", "A2_imperative_free", "A3_semantic_camouflage",
       "A4_length_matched", "A5_score_guided", "A6_split_payload"]

m01 = sur[sur.target_fpr == 0.01]
asr = m01.groupby(["attack", "defense"]).asr_retrieval.mean().unstack()
cal01 = cal[cal.target_fpr == 0.01].set_index("defense").realised_fpr

print("\ncandidate-survival ASR at a nominal 1% target (Table 3)")
print("                     " + "".join(f"{d.split('_')[0]:>7s}" for d in DEF))
for a in ["A0_static_templates"] + A16:
    print(f"    {a:20s}" + "".join(f"{asr.loc[a, d]:7.2f}" for d in DEF))
print("    " + "-" * 69)
print("    realised FPR        " + "".join(f"{cal01.loc[d]:7.3f}" for d in DEF))
print("    mean over A1-A6     " + "".join(f"{asr.loc[A16, d].mean():7.2f}" for d in DEF))
print("    maximum over A1-A6  " + "".join(f"{asr.loc[A16, d].max():7.2f}" for d in DEF))
print("    lowest cell         " + "".join(f"{asr.loc[A16, d].min():7.2f}" for d in DEF))
print("\n    An attack success rate without its realised false-positive rate is not")
print("    comparable to anything. The mean and the maximum rank these detectors")
print("    differently, and the lowest cell - the number most often reported - is")
print("    a still weaker summary: for D4 it is 0.00 against a maximum of 0.99.")

print("\na valid statement about the selected maximum (Proposition 3, Remark 3)")
cells = m01[(m01.defense == "D2_embed_probe") & (m01.attack.isin(A16))]
g = cells.groupby("attack").asr_retrieval.agg(["mean", "size"])
successes = [int(round(r["mean"] * r["size"])) for _, r in g.iterrows()]
trials = [int(r["size"]) for _, r in g.iterrows()]
top = int(np.argmax([s / n for s, n in zip(successes, trials)]))
lo, hi = clopper_pearson(successes[top], trials[top])
print(f"    D2's maximal cell is {successes[top]}/{trials[top]} = {successes[top]/trials[top]:.4f}")
print(f"    its own two-sided interval          [{lo:.3f}, {hi:.3f}]  (describes that cell)")
print(f"    simultaneous one-sided bound, K=6    {simultaneous_max_bound(successes, trials):.3f}  "
      f"(valid for the selected maximum)")

print("\noperating points across all three targets (Table 12)")
print("    detector            " + "".join(f"{'FPR':>8s}{'maxASR':>8s}" for _ in (1, 2, 3)))
print("    " + " " * 20 + f"{'target 0.001':>16s}{'target 0.01':>16s}{'target 0.05':>16s}")
cali = cal.set_index(["defense", "target_fpr"]).realised_fpr
for d in DEF:
    line = f"    {d:20s}"
    for t in (0.001, 0.01, 0.05):
        sub = sur[(sur.target_fpr == t) & (sur.defense == d) & (sur.attack.isin(A16))]
        line += f"{cali.loc[(d, t)]:8.3f}{sub.groupby('attack').asr_retrieval.mean().max():8.2f}"
    print(line)
print("\n    Raising the target from 0.1% to 5% lowers the observed mean but does not")
print("    move the observed maximum below 0.95 for any evaluated detector.")

print("\nwhat the adaptive attacker spent (Table 13)")
a5 = pd.read_csv(REC / "a5_optimisation_records.csv")
a5 = a5[a5.target_fpr == 0.01]
print(f"    {'detector':20s}{'evasion':>9s}{'median calls':>14s}{'mean calls':>12s}{'budget used':>13s}")
for d, g in a5.groupby("defense"):
    print(f"    {d:20s}{g.evaded.mean():9.2f}{g.detector_calls.median():14.0f}"
          f"{g.detector_calls.mean():12.1f}{g.detector_calls.mean() / 200:12.1%}")
print("\n    Evasion is a detector-side quantity: it counts payloads driven below the")
print("    threshold. Attack success additionally requires the edited document to hold")
print("    a top-10 slot, which is why the two differ - for D5 the payload evades on")
print("    0.85 of attempts but succeeds on 0.15, because the edits that lower windowed")
print("    perplexity lengthen the document and cost it candidate rank.")
