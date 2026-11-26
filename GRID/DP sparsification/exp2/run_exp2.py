#!/usr/bin/env python3
"""EXP2 attack: LSA leakage of the MLP (features-only, no-graph) box, baseline vs GRID.
Target = {ds}_exp2mlp; reference g = {ds}_gmcc; shadow = other dataset's MLP box."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # DP sparsification/
import _lsa_common as L

RES = os.path.join(HERE, "results", "attacks")

if __name__ == "__main__":
    L.run_leakage(method="exp2mlp", settings=["box"],
                  target_stem=lambda ds, s: f"{ds}_exp2mlp",
                  ref_stem=lambda ds: f"{ds}_gmcc",
                  res_dir=RES, runs=3, setting_col="box")
