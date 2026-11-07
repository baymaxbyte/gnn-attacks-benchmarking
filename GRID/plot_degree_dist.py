import os, sys
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ATTACK = os.path.join(HERE, "attack")
sys.path.insert(0, ATTACK)
from lsa.data import load_data
DATAPATH = os.path.join(ATTACK, "data", "dataset", "original") + os.sep
FGC_COARSE = ("/Users/chitraansh.pandey/Downloads/GNN DP/Benchmarking/temp/lsa/"
              "FGC Benchmarking/artifacts/coarsened")


def degrees_from_adj(adj):
    A = sp.coo_matrix(adj)
    A = ((A + A.T) > 0).astype(float).tolil()   # symmetrise + binarise
    A.setdiag(0)                                 # drop self-loops
    A = sp.csr_matrix(A); A.eliminate_zeros()
    return np.asarray(A.sum(1)).ravel()


def degrees_from_edge_index(ei, n):
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    return degrees_from_adj(A)


# --- uncoarsened cora ---
adj = load_data(DATAPATH, "cora")[0]
series = [("Uncoarsened Cora", degrees_from_adj(adj))]

# --- FGC coarsened, all ratios ---
for r in ["0.3", "0.1", "0.05"]:
    z = np.load(os.path.join(FGC_COARSE, f"fgc_cora_r{r}.npz"), allow_pickle=True)
    n = z["X_c"].shape[0]
    series.append((f"FGC r={r}", degrees_from_edge_index(z["edge_index"], n)))

# --- plot: P(k) vs k (log-log) + CCDF ---
colors = ["#222222", "#1f77b4", "#ff7f0e", "#d62728"]
markers = ["o", "s", "^", "D"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

for (label, deg), c, mk in zip(series, colors, markers):
    n = len(deg); mean_k = deg.mean(); mx = int(deg.max())
    leg = f"{label}  (n={n}, \u27e8k\u27e9={mean_k:.1f}, max={mx})"
    # P(k)
    k, cnt = np.unique(deg.astype(int), return_counts=True)
    ax1.plot(np.maximum(k, 0.5), cnt / n, marker=mk, ms=4, lw=1.2, color=c,
             alpha=0.85, label=leg)
    # CCDF  P(K>=k)
    ks = np.sort(deg)
    ccdf = 1.0 - np.arange(len(ks)) / len(ks)
    ax2.plot(np.maximum(ks, 0.5), ccdf, drawstyle="steps-post", lw=1.6,
             color=c, alpha=0.85, label=leg)

for ax, ttl, yl in [(ax1, "Degree distribution  P(k)", "fraction of nodes  P(k)"),
                    (ax2, "Complementary CDF  P(K \u2265 k)", "P(K \u2265 k)")]:
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("degree  k"); ax.set_ylabel(yl); ax.set_title(ttl)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, frameon=True)

fig.suptitle("Degree distribution: uncoarsened Cora vs FGC-coarsened (all ratios)",
             fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
OUT = os.path.expanduser("~/Downloads/cora_fgc_degree_distribution.png")
fig.savefig(OUT, dpi=150)
print("saved", OUT)
for label, deg in series:
    print(f"  {label:20s} n={len(deg):5d}  mean_deg={deg.mean():.2f}  "
          f"max_deg={int(deg.max())}  isolated={(deg==0).sum()}")
