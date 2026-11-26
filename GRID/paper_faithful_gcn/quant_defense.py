#!/usr/bin/env python3
"""todo #9 (rate-distortion posterior quantization) + #12 (OT/GW class barycenter).

Both collapse each node's served posterior to a per-class REPRESENTATIVE:
  * #12 barycenter  = one representative per class  (M=1, full collapse).
  * #9 quantization = M codewords per class (k-means); larger M = higher rate /
    more fidelity / more leak. M=1 coincides with #12.

Quantise the SAME soft-GCN whole-graph posteriors GRID/Fisher-Rao used, run the
identical LSA harness (Attack-0/1/6, random + near3). argmax is preserved by
construction (per predicted-class clustering) -> node accuracy unchanged.
Reuses step3_negmodes helpers.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_negmodes as S
from sklearn.cluster import KMeans

RES = os.path.join(S.HERE, "results", "quant"); os.makedirs(RES, exist_ok=True)
MS = [1, 4]            # codebook sizes per class; M=1 == #12 barycenter
ATT = ["0", "1", "6"]
EPS = 1e-12


def quantize(V, M):
    V = np.asarray(V, float); C = V.shape[1]; pred = V.argmax(1); R = V.copy()
    for c in range(C):
        idx = np.where(pred == c)[0]
        if len(idx) == 0:
            continue
        k = min(M, len(idx))
        km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(V[idx])
        R[idx] = km.cluster_centers_[km.labels_]
    R = np.clip(R, EPS, None); R = R / R.sum(1, keepdims=True)
    return R


def run_attack(target, shadow, tag):
    out = os.path.join(RES, f"{target}__{shadow}__{tag}.json")
    S.sh([S.PY, "run_attack.py", "--attack", "0,1,6", "--dataset", target, "--shadow", shadow,
          "--attack_dataset_path", "data/attack_dataset/", "--result_path", RES, "--runs", "5",
          "--tag", tag, "--link_prediction", "--quiet"])
    return out


def main():
    D = {}
    for ds in S.DIRS:
        adj, nbrs, feat, g, fb, fg = S.load_ds(ds)
        V = np.asarray(fb, float)
        quants = {"base": V}
        for M in MS:
            R = quantize(V, M)
            lc = int((V.argmax(1) != R.argmax(1)).sum())
            quants[f"q{M}"] = R
            print(f"[{ds}] M={M}: label_changes={lc}/{len(V)} (argmax preserved => node acc unchanged)", flush=True)
        D[ds] = dict(adj=adj, nbrs=nbrs, feat=feat, g=g, quants=quants, n=len(V))

    res = {}
    for mode in ("random", "near3"):
        for ds in S.DIRS:
            d = D[ds]
            links, negs = S.sample_pairs(d["adj"], d["nbrs"], d["n"], mode, 1234)
            for name, P in d["quants"].items():
                S.write_ds(f"{ds}_{mode}_{name}", links, negs, P.tolist(), d["g"], d["feat"])
        for ds, shadow in S.DIRS.items():
            for name in D[ds]["quants"]:
                res[(mode, ds, name)] = S.parse(run_attack(f"{ds}_{mode}_{name}", f"{shadow}_{mode}_base", f"{ds}_{mode}_{name}"))

    labels = {"base": "no-defense", "q1": "#12 barycenter (M=1)", "q4": "#9 quant (M=4)"}
    names = {"0": "Attack-0", "1": "Attack-1", "6": "Attack-6"}
    csv = os.path.join(RES, "quant_auc.csv")
    with open(csv, "w") as f:
        f.write("eval,dataset,config,attack,auc,drop_vs_base\n")
        for mode in ("random", "near3"):
            for ds in S.DIRS:
                base = res[(mode, ds, "base")]
                print(f"\n=== {ds} / {mode} eval ===", flush=True)
                for name in D[ds]["quants"]:
                    r = res[(mode, ds, name)]
                    line = "  ".join(f"{names[a]}={r[a]:.3f}({r[a]-base[a]:+.3f})" for a in ATT if a in r)
                    print(f"  {labels[name]:22s} {line}", flush=True)
                    for a in ATT:
                        if a in r:
                            f.write(f"{mode},{ds},{labels[name]},{names[a]},{r[a]:.4f},{r[a]-base[a]:+.4f}\n")
    for fn in os.listdir(S.ADS):
        if any(f"_{m}_{c}_attack_" in fn for m in ("random", "near3") for c in ("base", "q1", "q4")):
            os.remove(os.path.join(S.ADS, fn))
    print(f"\nwrote {csv}", flush=True)


if __name__ == "__main__":
    main()
