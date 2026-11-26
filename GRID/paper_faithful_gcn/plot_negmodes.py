#!/usr/bin/env python3
"""Plot Attack-1 AUC (no-defense vs GRID) across evaluation negative-sampling
modes, against the paper's reported Table-3 values."""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results", "d4", "negmode_results.csv")
OUT = os.path.join(HERE, "results", "d4", "negmode_attack1.png")

# paper Table 3, Attack-1, GCN
PAPER = {"cora": (0.942, 0.728), "citeseer": (0.965, 0.732)}
MODES = ["random", "d1", "near3"]
LABELS = ["random\n(He et al.)", "d1\n(paper D4)", "near3\n(n-hop floor)", "paper\n(reported)"]

vals = {}
with open(CSV) as f:
    for r in csv.DictReader(f):
        if r["attack"] == "Attack-1":
            vals[(r["neg_mode"], r["dataset"])] = (float(r["auc_no_defense"]), float(r["auc_grid"]))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, ds in zip(axes, ["cora", "citeseer"]):
    base = [vals[(m, ds)][0] for m in MODES] + [PAPER[ds][0]]
    grid = [vals[(m, ds)][1] for m in MODES] + [PAPER[ds][1]]
    x = np.arange(len(LABELS)); w = 0.38
    b1 = ax.bar(x - w/2, base, w, label="no defense", color="#4c72b0")
    b2 = ax.bar(x + w/2, grid, w, label="GRID (theta=0.4)", color="#dd8452")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.axvspan(2.5, 3.5, color="0.9", zorder=0)  # shade the paper column
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=8.5)
    ax.set_title(f"{ds}  -  Attack-1 (supervised, shadow)")
    ax.set_ylim(0.45, 1.02); ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("link-stealing AUC")
axes[0].legend(loc="lower left", fontsize=8.5)
fig.suptitle("GRID on soft GCN: the drop matches the paper only under n-hop-floor negatives,\n"
             "not under the paper's own D4 (random-unlinked) description", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=135)
print("wrote", OUT)
