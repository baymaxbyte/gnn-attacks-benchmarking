#!/usr/bin/env python3
"""EXP3 (extended) attack, NO GRID: baseline LSA leakage of the feature-KNN(k=10)
graph coarsened/condensed by MCC/FGC/KRON/UGC (and optionally GOREN/GCond) at their
native ratios. Target = {ds}_exp3{method}_{r}; reference g per dataset (gmcc for
cora/citeseer, gkron for squirrel/chameleon); shadow = same-family dataset. Attack
ground-truth = ORIGINAL edges. -> results/attacks/{tag}_attack_auc.csv.

Generalised over datasets: `--datasets cora citeseer` (default) reproduces the
original run; `--datasets squirrel chameleon --methods mcc fgc kron ugc
--tag exp3coarsenhet` runs the heterophilous pair."""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _lsa_common as L

RES = os.path.join(HERE, "results", "attacks")
RATIOS = {"mcc": ["0.3", "0.1"], "fgc": ["0.3", "0.1", "0.05"],
          "kron": ["0.3", "0.5", "0.7"], "ugc": ["0.3", "0.5", "0.7"],
          "goren": ["0.3", "0.5", "0.7"], "gcond": ["0.25", "0.5", "1.0"]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    ap.add_argument("--methods", nargs="+", default=["mcc", "fgc", "kron", "ugc", "goren", "gcond"])
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or ("exp3coarsen" if len(a.methods) > 1 else f"exp3{a.methods[0]}atk")
    SETTINGS = [f"{m}_{r}" for m in a.methods for r in RATIOS[m]]
    dirs, ref_stem, max_pos = L.attack_config(a.datasets)
    L.run_leakage(method=tag, settings=SETTINGS,
                  target_stem=lambda ds, s: f"{ds}_exp3{s}",
                  ref_stem=ref_stem,
                  res_dir=RES, runs=3, setting_col="method_ratio", grid=False,
                  dirs=dirs, max_pos=max_pos)
