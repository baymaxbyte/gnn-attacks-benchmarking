#!/usr/bin/env python3
"""EXP3 attack: LSA leakage of the feature-KNN(k=10) GCN box, baseline vs GRID.
Target = {ds}_exp3knn; reference g per dataset (gmcc for cora/citeseer, gkron for
squirrel/chameleon); shadow = same-family dataset's KNN box. Attack ground-truth
pairs are the ORIGINAL edges; GRID uses the ORIGINAL graph.

Generalised over datasets: `--datasets cora citeseer` (default) reproduces the
original run; `--datasets squirrel chameleon --tag exp3knnhet` runs the
heterophilous pair."""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _lsa_common as L

RES = os.path.join(HERE, "results", "attacks")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    ap.add_argument("--tag", default="exp3knn")
    ap.add_argument("--no_grid", action="store_true")
    a = ap.parse_args()
    dirs, ref_stem, max_pos = L.attack_config(a.datasets)
    L.run_leakage(method=a.tag, settings=["box"],
                  target_stem=lambda ds, s: f"{ds}_exp3knn",
                  ref_stem=ref_stem,
                  res_dir=RES, runs=3, setting_col="box", grid=not a.no_grid,
                  dirs=dirs, max_pos=max_pos)
