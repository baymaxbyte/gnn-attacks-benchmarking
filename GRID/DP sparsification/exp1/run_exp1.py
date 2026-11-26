#!/usr/bin/env python3
"""EXP1 attack (multi-method, NO GRID): LSA baseline leakage of edge-DP-sparsified
-> coarsened boxes, for FGC / UGC / KRON / MCC at each method's native ratios and
eps in {4,8}. Target = {ds}_exp1{method}_{r}_e{eps}; reference g = {ds}_gmcc;
shadow = other dataset same setting. Attack ground-truth = ORIGINAL edges.
Writes results/attacks/exp1coarsen_attack_auc.csv (baseline AUC + blackbox_acc)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _lsa_common as L

RES = os.path.join(HERE, "results", "attacks")
EPS = ["8.0", "4.0"]
RATIOS = {"mcc": ["0.3", "0.1"], "fgc": ["0.3", "0.1", "0.05"],
          "kron": ["0.3", "0.5", "0.7"], "ugc": ["0.3", "0.5", "0.7"]}
SETTINGS = [f"{m}_{r}_e{e}" for m in ("mcc", "fgc", "kron", "ugc") for r in RATIOS[m] for e in EPS]

if __name__ == "__main__":
    L.run_leakage(method="exp1coarsen", settings=SETTINGS,
                  target_stem=lambda ds, s: f"{ds}_exp1{s}",
                  ref_stem=lambda ds: f"{ds}_gmcc",
                  res_dir=RES, runs=3, setting_col="method_ratio_eps", grid=False)
