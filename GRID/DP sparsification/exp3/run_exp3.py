#!/usr/bin/env python3
"""EXP3 attack: LSA leakage of the feature-KNN(k=10) GCN box, baseline vs GRID.
Target = {ds}_exp3knn; reference g = {ds}_gmcc; shadow = other dataset's KNN box.
Attack ground-truth pairs are the ORIGINAL edges; GRID uses the ORIGINAL graph."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _lsa_common as L

RES = os.path.join(HERE, "results", "attacks")

if __name__ == "__main__":
    L.run_leakage(method="exp3knn", settings=["box"],
                  target_stem=lambda ds, s: f"{ds}_exp3knn",
                  ref_stem=lambda ds: f"{ds}_gmcc",
                  res_dir=RES, runs=3, setting_col="box")
