#!/usr/bin/env python3
"""EXP2 on the heterophilous datasets: LSA leakage of the MLP (features-only, no
graph) box on squirrel/chameleon, baseline vs GRID. Target = {ds}_exp2mlp;
reference g = {ds}_gkron (graph-free ref available for these datasets); shadow =
the other heterophilous dataset. Positives capped at 10k (squirrel has 198k edges).
Writes results/attacks/exp2mlphet_attack_auc.csv."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _lsa_common as L

RES = os.path.join(HERE, "results", "attacks")

if __name__ == "__main__":
    L.run_leakage(method="exp2mlphet", settings=["box"],
                  target_stem=lambda ds, s: f"{ds}_exp2mlp",
                  ref_stem=lambda ds: f"{ds}_gkron",
                  res_dir=RES, runs=3, setting_col="box", grid=True,
                  dirs={"squirrel": "chameleon", "chameleon": "squirrel"},
                  max_pos=10000)
