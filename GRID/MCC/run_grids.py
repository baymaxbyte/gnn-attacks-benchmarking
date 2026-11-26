#!/usr/bin/env python3
"""Driver: run the MCC alpha/beta/gamma grid search for the settings we need
tuned hyperparameters for. CiteSeer first (its values were only borrowed from
cora), then cora 0.1/0.3 (validate the author values on OUR box) and cora 0.5
(a new ratio). Each writes results/gridsearch/{ds}_{setting}_grid.csv and prints
its top settings. Selection is by held-out validation accuracy."""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
JOBS = [("citeseer", "0.1"), ("citeseer", "0.3"),
        ("cora", "0.1"), ("cora", "0.3"), ("cora", "0.5")]

t0 = time.time()
for ds, s in JOBS:
    print(f"\n########## GRID {ds} / {s}  (t+{time.time()-t0:.0f}s) ##########", flush=True)
    r = subprocess.run([PY, os.path.join(HERE, "mcc_gridsearch.py"),
                        "--dataset", ds, "--setting", s,
                        "--epochs", "200", "--val_every", "5"])
    if r.returncode:
        print(f"!! {ds}/{s} returned {r.returncode}", flush=True)
print(f"\nALL GRIDS DONE  (total {time.time()-t0:.0f}s)", flush=True)
