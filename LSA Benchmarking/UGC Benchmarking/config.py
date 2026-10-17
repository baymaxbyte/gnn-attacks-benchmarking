"""Paths, dataset constants and environment self-check for the UGC benchmark.

No external repository is needed: the UGC coarsening is reimplemented in
ugc_solver.py and the attack implementation is vendored under attack/.

    python config.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

ATTACK_HOME = os.environ.get("UGC_ATTACK_HOME", os.path.join(HERE, "attack"))
DATA = os.path.join(ATTACK_HOME, "data", "dataset", "original")
ATTACK_PRED = os.path.join(ATTACK_HOME, "data", "pred")
ATTACK_DATASET = os.path.join(ATTACK_HOME, "data", "attack_dataset")

ARTIFACTS = os.path.join(HERE, "artifacts")
COARSENED = os.path.join(ARTIFACTS, "coarsened")
POSTERIORS = os.path.join(ARTIFACTS, "posteriors")
RESULTS = os.path.join(HERE, "results")
LOGS = os.path.join(HERE, "logs")
MANIFEST = os.path.join(HERE, "manifest.json")

# Absolute, because the attack scripts run with cwd=ATTACK_HOME.
PYTHON = os.path.abspath(os.environ.get("UGC_PYTHON", sys.executable))

DATASETS = ("cora", "citeseer")

# Heterophily factor alpha, paper Table 6 and run.sh. One value per dataset, held
# fixed across all coarsening ratios.
ALPHA = {"cora": 0.19, "citeseer": 0.26}

# --ratio is the percentage of nodes REMOVED: Find_Binwidth computes
# ratio = 1 - k/n, so k = n*(1 - ratio/100).
RATIOS = (30, 50, 70)

# Bin width shipped in UGC_bin_widths.BIN_WIDTH_DICTONARY. Only ratio 50 was
# released, and only for datasets other than citeseer, so the rest are calibrated
# here and this value is the yardstick for that calibration.
SHIPPED_BIN_WIDTH = {("cora", 50): 0.004}

_PICKLES = ("x", "y", "tx", "ty", "allx", "ally", "graph", "test.index")


def n_supernodes(n, ratio):
    """Supernodes implied by a removal percentage."""
    return int(round(n * (1.0 - ratio / 100.0)))


def check(verbose=True):
    ok = True

    def say(m):
        if verbose:
            print(m)

    say(f"  HERE          {HERE}")
    for ds in DATASETS:
        missing = [s for s in _PICKLES
                   if not os.path.exists(os.path.join(DATA, f"ind.{ds}.{s}"))]
        if missing:
            ok = False
            say(f"  {ds:9s}     MISSING {missing}")
        else:
            say(f"  {ds:9s}     8/8 Planetoid files, alpha={ALPHA[ds]}")

    for script in ("partial_graph_generation.py", "run_attack.py"):
        if os.path.exists(os.path.join(ATTACK_HOME, script)):
            say(f"  attack        {script} found")
        else:
            ok = False
            say(f"  attack        MISSING {script}")

    for mod in ("torch", "torch_geometric", "numpy", "scipy"):
        try:
            m = __import__(mod)
            say(f"  {mod:13s} {getattr(m, '__version__', '?')}")
        except ImportError:
            ok = False
            say(f"  {mod:13s} MISSING")

    say("  all dependencies resolved" if ok else "  UNRESOLVED (see above)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
