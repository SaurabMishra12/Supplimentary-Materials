#!/usr/bin/env python3
"""
Stage 3 of Figure 1 and the factorisation of Figure 2:

    saved detector decision -> surviving episode -> behavioural replay
        -> P(C | E, D) -> P(C | D)

    python examples/replay_demo.py

Candidate survival is P(E | D): the fraction of inserted payloads that reach the
model's context after filtering. It is not a downstream security endpoint. This
script takes the same detector decision used in Section 5, keeps only the
episodes in which the payload survived, and reads the compliance indicator on
exactly those episodes, which is P(C | E, D). The end-to-end rate follows:

    P(C | D) = P(C | E, D) * P(E | D)

which holds by construction here because the injected directive can reach the
model only through the inserted payload, so P(C | not E, D) = 0.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REC = ROOT / "data" / "records"
Z = 1.959963984540054

MODEL, PROMPT, POSITION, TARGET = "Qwen/Qwen2.5-3B-Instruct", "plain", 0, 0.01


def wilson(k, n):
    """Wilson 95% interval, used for the conditional compliance rates."""
    if n == 0:
        return (np.nan, np.nan)
    p, den = k / n, 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / den
    half = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return centre - half, centre + half


def replay(indicators):
    """One attack-detector cell: exposure, survivor compliance, end-to-end rate."""
    N = len(indicators)
    survivors = indicators[indicators.E == 1]           # the detector let these through
    Ns = len(survivors)
    Nc = int(survivors.C.sum())                          # of those, these emitted the canary
    lo, hi = wilson(Nc, Ns)
    return dict(N=N, N_surv=Ns, N_comp_surv=Nc,
                P_E_given_D=Ns / N,
                P_C_given_E_D=(Nc / Ns) if Ns else np.nan,
                wilson_low=lo, wilson_high=hi,
                P_C_given_D=Nc / N)


dc = pd.read_parquet(REC / "detector_conditioned_indicators.parquet")
main = dc[(dc.model == MODEL) & (dc.system == PROMPT) &
          (dc.position == POSITION) & (dc.target_fpr == TARGET)]

print("=" * 78)
print(f"detector-conditioned replay: {MODEL}, {PROMPT} prompt, context position "
      f"{POSITION}, target FPR {TARGET}")
print("=" * 78)

# ---------------------------------------------------------------------------
# the worked case: A2 against the DistilBERT filter
# ---------------------------------------------------------------------------
a, d = "A2_imperative_free", "D3_distilbert"
und = replay(main[(main.attack == a) & (main.defense == "D0_none")])
det = replay(main[(main.attack == a) & (main.defense == d)])

print(f"\nworked case: {a} against {d}\n")
print(f"    undefended baseline (no detector)")
print(f"        payloads inserted                {und['N']}")
print(f"        reached the context   P(E|D0)    {und['N_surv']}/{und['N']} = {und['P_E_given_D']:.3f}")
print(f"        complied among those  P(C|E,D0)  {und['N_comp_surv']}/{und['N_surv']} = {und['P_C_given_E_D']:.3f}")
print(f"        end-to-end            P(C|D0)    {und['P_C_given_D']:.4f}")
print(f"\n    with the detector at a {TARGET:.0%} target")
print(f"        survived the filter   P(E|D)     {det['N_surv']}/{det['N']} = {det['P_E_given_D']:.4f}")
print(f"        complied among those  P(C|E,D)   {det['N_comp_surv']}/{det['N_surv']} = {det['P_C_given_E_D']:.3f}"
      f"  [{det['wilson_low']:.2f}, {det['wilson_high']:.2f}]")
print(f"        end-to-end            P(C|D)     {det['P_C_given_D']:.4f}")
print(f"\n    Exposure fell from {und['P_E_given_D']:.2f} to {det['P_E_given_D']:.3f}, while compliance among")
print(f"    survivors rose from {und['P_C_given_E_D']:.3f} to {det['P_C_given_E_D']:.3f}. End-to-end compromise still")
print(f"    fell from {und['P_C_given_D']:.3f} to {det['P_C_given_D']:.3f}, because the reduction in exposure is large")
print(f"    enough to dominate the higher conditional rate.")
print(f"\n    The conditional estimate rests on {det['N_surv']} surviving episodes. Its interval is wide,")
print(f"    and it is evidence of an attack-dependent selection effect in this setting,")
print(f"    not of a universal enrichment phenomenon.")

# ---------------------------------------------------------------------------
# the full row set: Table 5
# ---------------------------------------------------------------------------
A16 = ["A1_query_conditioned", "A2_imperative_free", "A3_semantic_camouflage",
       "A4_length_matched", "A5_score_guided", "A6_split_payload"]

print("\n" + "-" * 78)
print("Table 5: exposure, compliance among survivors, end-to-end compromise")
print("-" * 78)
print(f"\n    {'attack':24s}{'P(E|D0)':>9s}{'P(E|D3)':>9s}{'P(C|E,D0)':>11s}"
      f"{'P(C|E,D3)':>11s}{'95% Wilson':>16s}{'P(C|D3)':>9s}")
rows = []
for a in A16:
    d3 = replay(main[(main.attack == a) & (main.defense == "D3_distilbert")])
    sub0 = main[(main.attack == a) & (main.defense == "D0_none")]
    d0 = replay(sub0) if len(sub0) else None
    # A5 is re-optimised per detector, so it has no single undefended baseline
    has0 = d0 is not None and a != "A5_score_guided"
    print(f"    {a:24s}"
          f"{d0['P_E_given_D']:9.2f}" if has0 else f"    {a:24s}{'-':>9s}", end="")
    print(f"{d3['P_E_given_D']:9.3f}"
          + (f"{d0['P_C_given_E_D']:11.3f}" if has0 else f"{'-':>11s}")
          + f"{d3['P_C_given_E_D']:11.3f}"
          + f"   [{d3['wilson_low']:.2f}, {d3['wilson_high']:.2f}]"
          + f"{d3['P_C_given_D']:9.4f}")
    rows.append(dict(attack=a, **d3))

print("\n    A5 has no undefended baseline column: its documents are re-optimised")
print("    separately against each detector, so there is no single unfiltered")
print("    population they all come from.")

# ---------------------------------------------------------------------------
# the identity, checked on every cell
# ---------------------------------------------------------------------------
full = []
for (a, d), g in main.groupby(["attack", "defense"]):
    r = replay(g)
    r.update(attack=a, defense=d)
    full.append(r)
full = pd.DataFrame(full).dropna(subset=["P_C_given_E_D"])
err = float((full.P_C_given_E_D * full.P_E_given_D - full.P_C_given_D).abs().max())
print(f"\n    identity P(C|D) = P(C|E,D) * P(E|D) over all {len(full)} cells: "
      f"maximum absolute error {err:.2e}")

print("\n" + "-" * 78)
print("why candidate survival alone is not enough")
print("-" * 78)
print("""
    A retrieval-layer filter changes two things at once. It changes how much
    adversarial content reaches the model, and it changes which adversarial
    content reaches the model. The second effect is invisible to a candidate
    survival metric: the survivors are a selected subset of the payloads, and
    nothing guarantees they behave like the unfiltered population.

    Reporting P(E | D), P(C | E, D) and P(C | D) separately - with the survivor
    count and its interval beside the conditional rate - is what makes the two
    effects distinguishable. Reporting only candidate survival treats a change in
    composition as if it were a change in magnitude.
""")
