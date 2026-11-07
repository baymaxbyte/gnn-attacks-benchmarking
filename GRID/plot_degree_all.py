import os, sys, glob
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
LSA = os.path.join(HERE, "..", "lsa")
OUTDIR = os.path.expanduser("~/Downloads")

# method -> (artifact subdir, ratios, filename template, loader kind)
METHODS = {
    "FGC":   dict(sub="FGC Benchmarking/artifacts/coarsened",  ratios=["0.3", "0.1", "0.05"],
                  tmpl="fgc_{ds}_r{r}.npz", kind="npz_ei"),
    "UGC":   dict(sub="UGC Benchmarking/artifacts/coarsened",  ratios=["30", "50", "70"],
                  tmpl="ugc_{ds}_ratio{r}.npz", kind="npz_ei"),
    "KRON":  dict(sub="KRON Benchmarking/artifacts/reduced",   ratios=["0.3", "0.5", "0.7"],
                  tmpl="kron_{ds}_r{r}.npz", kind="npz_ei"),
    "GOREN": dict(sub="GOREN Benchmarking/artifacts/coarsened", ratios=["0.3", "0.5", "0.7"],
                  tmpl="goren_{ds}_ratio{r}.npz", kind="npz_ei"),
    "GCond": dict(sub="GCOND Benchmarking/artifacts/condensed", ratios=["0.25", "0.5", "1.0"],
                  tmpl="adj_{ds}_{r}_1.pt", kind="pt_dense"),
}
GCOND_THRESH = 0.5


def deg_from_adj(adj):
    A = sp.coo_matrix(adj)
    A = ((A + A.T) > 0).astype(float).tolil()
    A.setdiag(0)
    A = sp.csr_matrix(A); A.eliminate_zeros()
    return np.asarray(A.sum(1)).ravel()


def coarsened_degrees(method, ds, r):
    cfg = METHODS[method]
    path = os.path.join(LSA, cfg["sub"], cfg["tmpl"].format(ds=ds, r=r))
    if not os.path.exists(path):
        return None
    if cfg["kind"] == "npz_ei":
        z = np.load(path, allow_pickle=True)
        n = int(z["X_c"].shape[0]) if "X_c" in z else int(z["Y_c"].shape[0])
        ei = z["edge_index"]
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
        return deg_from_adj(A)
    else:  # pt_dense (GCond): small dense learned adjacency, thresholded
        import torch
        a = torch.load(path, map_location="cpu", weights_only=False)
        a = np.asarray(a.detach().cpu().numpy() if hasattr(a, "detach") else a, dtype=float)
        A = (a > GCOND_THRESH).astype(float)
        return deg_from_adj(sp.coo_matrix(A))


colors = ["#222222", "#1f77b4", "#ff7f0e", "#d62728"]
markers = ["o", "s", "^", "D"]

for method, cfg in METHODS.items():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, ds in enumerate(["cora", "citeseer"]):
        series = [("Uncoarsened", deg_from_adj(load_data(DATAPATH, ds)[0]))]
        for r in cfg["ratios"]:
            d = coarsened_degrees(method, ds, r)
            if d is not None:
                series.append((f"{method} r={r}", d))
        axP, axC = axes[row]
        for (label, deg), c, mk in zip(series, colors, markers):
            n = len(deg); mk_lab = f"{label} (n={n}, \u27e8k\u27e9={deg.mean():.1f})"
            k, cnt = np.unique(deg.astype(int), return_counts=True)
            axP.plot(np.maximum(k, 0.5), cnt / n, marker=mk, ms=4, lw=1.2,
                     color=c, alpha=0.85, label=mk_lab)
            ks = np.sort(deg); ccdf = 1.0 - np.arange(len(ks)) / len(ks)
            axC.plot(np.maximum(ks, 0.5), ccdf, drawstyle="steps-post", lw=1.6,
                     color=c, alpha=0.85, label=mk_lab)
        for ax, ttl, yl in [(axP, f"{ds} — P(k)", "fraction P(k)"),
                            (axC, f"{ds} — CCDF", "P(K \u2265 k)")]:
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("degree k"); ax.set_ylabel(yl); ax.set_title(ttl)
            ax.grid(True, which="both", ls=":", alpha=0.4)
            ax.legend(fontsize=8)
    note = "  (GCond: dense learned adj thresholded at 0.5)" if method == "GCond" else ""
    fig.suptitle(f"Degree distribution: uncoarsened vs {method} (all ratios){note}",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTDIR, f"{method.lower()}_degree_distribution.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("saved", out)
    for ds in ["cora", "citeseer"]:
        for r in cfg["ratios"]:
            d = coarsened_degrees(method, ds, r)
            if d is not None:
                print(f"    {method:5s} {ds:9s} r={r:5s} n={len(d):5d} mean_deg={d.mean():5.2f} max={int(d.max())}")
