#!/usr/bin/env python3
"""Run GCond edge-DP condensation for all (dataset, eps, reduction_rate) settings,
each as a fresh subprocess (clean state). ~6.5 min/run on CPU."""
import os, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "bin", "python3")
JOBS = [(ds, e, r) for ds in ("cora", "citeseer") for e in ("8.0", "4.0") for r in ("0.25", "0.5", "1.0")]
t0 = time.time()
for ds, e, r in JOBS:
    print(f"\n######## GCOND {ds} eps={e} r={r}  (t+{time.time()-t0:.0f}s) ########", flush=True)
    subprocess.run([PY, os.path.join(HERE, "gcond_dp.py"), "--dataset", ds, "--eps", e, "--r", r])
print(f"\nALL GCOND DONE ({time.time()-t0:.0f}s)", flush=True)
