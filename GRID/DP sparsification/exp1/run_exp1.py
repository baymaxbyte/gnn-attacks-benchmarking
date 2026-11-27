#!/usr/bin/env python3
"""EXP1 attack (multi-method, NO GRID): LSA baseline leakage of edge-DP-sparsified
-> coarsened boxes, for FGC / UGC / KRON / MCC at each method's native ratios and
eps in {4,8}. Target = {ds}_exp1{method}_{r}_e{eps}; reference g per dataset
(gmcc for cora/citeseer, gkron for squirrel/chameleon); shadow = same-family dataset.
Attack ground-truth = ORIGINAL edges. Writes results/attacks/{tag}_attack_auc.csv.

Generalised over datasets: `--datasets cora citeseer` (default) reproduces the
original run; `--datasets squirrel chameleon --tag exp1coarsenhet` runs the
heterophilous pair (shadow/reference/positive-cap resolved per dataset)."""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _lsa_common as L

RES = os.path.join(HERE, "results", "attacks")
EPS = ["8.0", "4.0"]
RATIOS = {"mcc": ["0.3", "0.1"], "fgc": ["0.3", "0.1", "0.05"],
          "kron": ["0.3", "0.5", "0.7"], "ugc": ["0.3", "0.5", "0.7"]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    ap.add_argument("--methods", nargs="+", default=["mcc", "fgc", "kron", "ugc"])
    ap.add_argument("--tag", default="exp1coarsen")
    a = ap.parse_args()
    SETTINGS = [f"{m}_{r}_e{e}" for m in a.methods for r in RATIOS[m] for e in EPS]
    dirs, ref_stem, max_pos = L.attack_config(a.datasets)
    L.run_leakage(method=a.tag, settings=SETTINGS,
                  target_stem=lambda ds, s: f"{ds}_exp1{s}",
                  ref_stem=ref_stem,
                  res_dir=RES, runs=3, setting_col="method_ratio_eps", grid=False,
                  dirs=dirs, max_pos=max_pos)
