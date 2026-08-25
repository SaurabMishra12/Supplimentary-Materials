#!/usr/bin/env python3
"""
Redraw Figures 3-7 from the released datasets in results/.

    python examples/reproduce_all.py     # first: rebuild results/
    python examples/make_figures.py      # then: redraw figures/

Every plotted value is read from results/*.csv, which reproduce_all.py has
already checked against the manuscript. No number is entered here by hand
except the axis limits and the labels.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 200, "savefig.bbox": "tight",
})
BLUE, ORANGE, GREY, GREEN, GOLD, TEAL, PINK = (
    "#1f77b4", "#d95f02", "#9e9e9e", "#2ca67a", "#e8a33d", "#17a398", "#cc79a7")


# ---------------------------------------------------------------- Figure 3
f3 = pd.read_csv(RES / "figure3_data.csv")
pol = pd.read_csv(RES / "table8_fusion_policies.csv")

fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))

a = f3[f3.panel == "a"]
for ds, colour in [("fiqa", BLUE), ("scifact", ORANGE)]:
    g = a[a.dataset == ds].sort_values("alpha")
    ax[0].plot(g.alpha, g.mean_ndcg10, color=colour, lw=2, label=ds)
    top = g.loc[g.mean_ndcg10.idxmax()]
    ax[0].plot(top.alpha, top.mean_ndcg10, "o", color=colour, ms=5)
ax[0].set_title("(a) flat near the optimum")
ax[0].set_xlabel(r"fusion weight $\alpha$   (1 = dense)")
ax[0].set_ylabel("mean nDCG@10")
ax[0].legend(frameon=False, loc="lower right")

rangev = f3[f3.panel == "b"].per_query_range.to_numpy()
ax[1].hist(rangev, bins=60, color=BLUE)
ax[1].axvline(0.01, color=ORANGE, ls="--", lw=1.2)
share = float((rangev < 0.01).mean())
ax[1].text(0.11, 0.97, f"{share * 100:.0f}% of queries\nbelow " r"$\varepsilon = 0.01$",
           transform=ax[1].transAxes, color=ORANGE, va="top", fontsize=8.5)
ax[1].set_title(r"(b) $\alpha$ mostly irrelevant")
ax[1].set_xlabel(r"per-query nDCG range over $\alpha$")
ax[1].set_ylabel("queries")

BARS = [("BM25", "BM25 (alpha=0)", GREY), ("Dense", "Dense (alpha=1)", GREY),
        ("RRF", "RRF (k=60)", BLUE), (r"Fixed $\alpha^\star$", "Fixed alpha*=0.65", BLUE),
        ("Learned", "Learned: RandomForest(50,d5)", GOLD),
        ("Oracle", "Oracle alpha (ceiling)", GREEN)]
poli = pol.set_index("policy")
heights, errs, labels, colours = [], [], [], []
for label, key, colour in BARS:
    row = poli.loc[next(i for i in poli.index if i.startswith(key.split(" [")[0]))]
    heights.append(row.ndcg10)
    errs.append((row.ci_high - row.ci_low) / 2)   # paired-difference interval, re-centred
    labels.append(label)
    colours.append(colour)
ax[2].bar(range(len(heights)), heights, yerr=errs, color=colours, capsize=3,
          error_kw=dict(lw=1, capthick=1))
ax[2].set_xticks(range(len(labels)))
ax[2].set_xticklabels(labels, rotation=45, ha="right")
ax[2].set_ylim(0.33, 0.61)
ax[2].set_title("(c) headroom unreached")
ax[2].set_ylabel("mean nDCG@10")
fig.tight_layout()
fig.savefig(FIG / "figure3.png")
plt.close(fig)


# ---------------------------------------------------------------- Figure 4
f4 = pd.read_csv(RES / "figure4_data.csv")
NAMES = {"D1_3feat_tiny": "D1\n3-feat", "D1b_3feat_trained": "D1b\n3-feat$^+$",
         "D2_embed_probe": "D2\nemb. probe", "D3_distilbert": "D3\nDistilBERT",
         "D4_guard_zeroshot": "D4\nguard", "D5_perplexity": "D5\nperplexity",
         "D6_ensemble": "D6\nensemble"}
f4 = f4[f4.defense.isin(NAMES)].set_index("defense").loc[list(NAMES)].reset_index()

fig, ax = plt.subplots(figsize=(7.6, 3.4))
x = np.arange(len(f4))
w = 0.38
ax.bar(x - w / 2, f4.mean_asr_a1_a6, w, color=BLUE, label="mean over A1-A6")
ax.bar(x + w / 2, f4.max_asr_a1_a6, w, color=ORANGE, label="worst case over A1-A6")
ax.plot(x, f4.lowest_cell_a1_a6, "v", color="black", ms=7, ls="none",
        label="lowest cell (the number usually reported)")
for xi, v in zip(x, f4.max_asr_a1_a6):
    ax.text(xi + w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
ax.axhline(1.0, color="black", lw=0.8, ls=":")
ax.set_xticks(x)
ax.set_xticklabels(NAMES.values())
ax.set_ylim(0, 1.12)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_ylabel("attack success rate")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ax.legend(handles=[Line2D([], [], marker="v", color="black", ls="none", ms=7,
                          label="lowest cell (the number usually reported)"),
                   Patch(color=BLUE, label="mean over A1-A6"),
                   Patch(color=ORANGE, label="worst case over A1-A6")],
          frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.16), fontsize=8.5)
fig.savefig(FIG / "figure4.png")
plt.close(fig)


# ---------------------------------------------------------------- Figure 5
LADDER = pd.read_csv(RES / "table1_attack_ladder.csv")
POS = {"A0": (0.04, 0.50), "A1": (0.22, 0.50), "A2": (0.45, 0.80),
       "A3": (0.45, 0.50), "A4": (0.45, 0.20), "A5": (0.72, 0.80), "A6": (0.72, 0.20)}
LABEL = {"A0": "A0 static", "A1": "A1 query", "A2": "A2 keyword list",
         "A3": "A3 clean corpus", "A4": "A4 length stats",
         "A5": "A5 detector access", "A6": "A6 top-$k$ budget"}
EDGES = [("A0", "A1"), ("A1", "A2"), ("A1", "A3"), ("A1", "A4"), ("A2", "A5"), ("A4", "A6")]

fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.set_xlim(0, 1)
ax.set_ylim(-0.12, 1.0)
ax.axis("off")
boxes = {}
for k, (cx, cy) in POS.items():
    wbox = 0.0118 * len(LABEL[k]) + 0.028
    lw = 1.6 if k == "A3" else 0.8
    ax.add_patch(plt.Rectangle((cx, cy - 0.075), wbox, 0.15, fill=False, lw=lw, ec="black"))
    ax.text(cx + wbox / 2, cy, LABEL[k], ha="center", va="center", fontsize=8.5)
    boxes[k] = (cx, cx + wbox, cy)
for src, dst in EDGES:
    x0, y0 = boxes[src][1], boxes[src][2]
    x1, y1 = boxes[dst][0], boxes[dst][2]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="black",
                                connectionstyle="arc3,rad=0"))
ax.text(0.42, -0.10, "A3 has the highest observed ASR against\nthe most detectors and never queries one",
        fontsize=7.5, style="italic", ha="center", va="bottom")
fig.savefig(FIG / "figure5.png")
plt.close(fig)


# ---------------------------------------------------------------- Figure 6
f6 = pd.read_csv(RES / "figure6_data.csv")
pa = f6[f6.panel == "a"]
pb = f6[f6.panel == "b"].set_index("configuration")

fig, ax = plt.subplots(1, 2, figsize=(8.4, 2.9))
x = np.arange(len(pa))
w = 0.27
ax[0].bar(x - w, pa.payload_entry, w, color=GREY, label="payload entry (retrieval)")
ax[0].bar(x, pa.compliance_1_5b, w, color=TEAL, label="Qwen2.5-1.5B complied")
ax[0].bar(x + w, pa.compliance_3b, w, color=ORANGE, label="Qwen2.5-3B complied")
ax[0].set_xticks(x)
ax[0].set_xticklabels([c.split("_")[0] for c in pa.configuration])
ax[0].set_ylim(0, 1.08)
ax[0].set_ylabel("rate")
ax[0].set_title("(a) Entry is not compliance")
ax[0].legend(frameon=False, fontsize=7.2, ncol=3, loc="upper center",
             bbox_to_anchor=(0.5, -0.16), handlelength=1.2, columnspacing=1.0)

bars = [("QA\n(position 0)", pb.loc["qa_position0_A1_A6", "compliance_3b"]),
        ("QA\n(averaged over\npositions)", pb.loc["qa_averaged_over_positions_A1_A6", "compliance_3b"]),
        ("Tool\nselection", pb.loc["tool_selection", "compliance_3b"])]
ax[1].bar(range(3), [v for _, v in bars], 0.55, color=PINK)
for i, (_, v) in enumerate(bars):
    ax[1].text(i, v + 0.012, f"{v:.3f}".lstrip("0"), ha="center", fontsize=8)
ax[1].set_xticks(range(3))
ax[1].set_xticklabels([n for n, _ in bars], fontsize=8)
ax[1].set_ylim(0, 0.68)
ax[1].set_ylabel("compliance | entry")
ax[1].set_title("(b) Conversion varies by task (Qwen2.5-3B)")
fig.tight_layout()
fig.savefig(FIG / "figure6.png")
plt.close(fig)


# ---------------------------------------------------------------- Figure 7
f7 = pd.read_csv(RES / "figure7_data.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.6))
for k, (ds, colour) in enumerate([("fiqa", BLUE), ("scifact", ORANGE)]):
    g = f7[f7.dataset == ds].sort_values("rerank_budget")
    g = g[g.rerank_budget.isin([0, 10, 50, 100])]
    ax[k].plot(range(len(g)), g.ndcg10, "-o", color=colour, ms=4)
    ax[k].set_xticks(range(len(g)))
    ax[k].set_xticklabels([f"CE {int(b)}\n{p:.0f} ms" for b, p in zip(g.rerank_budget, g.p50_ms)],
                          fontsize=7.5)
    ax[k].set_title(f"{ds.capitalize() if ds != 'fiqa' else 'FiQA'} ({int(g.n_docs.iloc[0]):,} docs)",
                    fontsize=9)
    ax[k].set_ylabel("nDCG@10")
    ax[k].grid(axis="y", lw=0.4, alpha=0.5)
fig.supxlabel("cross-encoder budget and measured median latency", fontsize=8.5, y=-0.06)
fig.tight_layout()
fig.savefig(FIG / "figure7.png")
plt.close(fig)

print("wrote:", ", ".join(sorted(p.name for p in FIG.glob("*.png"))))
