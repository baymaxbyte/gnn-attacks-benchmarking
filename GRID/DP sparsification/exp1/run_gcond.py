#!/usr/bin/env python3
"""EXP1 GCond attack (no GRID): baseline LSA leakage of the DP-sparsified ->
GCond-condensed boxes, reduction_rate {0.25,0.5,1.0} x eps {4,8}."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _lsa_common as L

RES = os.path.join(HERE, "results", "attacks")
SETTINGS = [f"{r}_e{e}" for r in ("0.25", "0.5", "1.0") for e in ("8.0", "4.0")]

if __name__ == "__main__":
    L.run_leakage(method="exp1gcond", settings=SETTINGS,
                  target_stem=lambda ds, s: f"{ds}_exp1gcond_{s}",
                  ref_stem=lambda ds: f"{ds}_gmcc",
                  res_dir=RES, runs=3, setting_col="rrate_eps", grid=False)
