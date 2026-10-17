"""Paths, constants and environment self-check for the KRON leakage benchmark.

    python config.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

ATTACK_HOME = os.environ.get("KRON_ATTACK_HOME", os.path.join(HERE, "attack"))
DATA = os.path.join(ATTACK_HOME, "data", "dataset", "original")
ATTACK_PRED = os.path.join(ATTACK_HOME, "data", "pred")
ATTACK_DATASET = os.path.join(ATTACK_HOME, "data", "attack_dataset")

ARTIFACTS = os.path.join(HERE, "artifacts")
REDUCED = os.path.join(ARTIFACTS, "reduced")
POSTERIORS = os.path.join(ARTIFACTS, "posteriors")
RESULTS = os.path.join(HERE, "results")
LOGS = os.path.join(HERE, "logs")
MANIFEST = os.path.join(HERE, "manifest.json")

PYTHON = os.path.abspath(os.environ.get("KRON_PYTHON", sys.executable))

DATASETS = ("cora", "citeseer")

# r is the fraction of nodes REMOVED, matching the GOREN and UGC conventions: the
# reduced graph keeps about n*(1-r) nodes. keep = 0.7 / 0.5 / 0.3.
RATIOS = (0.3, 0.5, 0.7)

# Which nodes become INTERIOR (eliminated). "degree" keeps the high-degree backbone
# and eliminates the low-degree periphery, one boundary node per component minimum.
SELECTION = "degree"


def n_boundary(n, r):
    return int(round(n * (1.0 - r)))


def check(verbose=True):
    ok = True

    def say(m):
        if verbose:
            print(m)

    say(f"  HERE          {HERE}")
    _pickles = ("x", "y", "tx", "ty", "allx", "ally", "graph", "test.index")
    for ds in DATASETS:
        missing = [s for s in _pickles
                   if not os.path.exists(os.path.join(DATA, f"ind.{ds}.{s}"))]
        if missing:
            ok = False
            say(f"  {ds:9s}     MISSING {missing}")
        else:
            say(f"  {ds:9s}     8/8 Planetoid files")

    for script in ("partial_graph_generation.py", "run_attack.py"):
        if os.path.exists(os.path.join(ATTACK_HOME, script)):
            say(f"  attack        {script} found")
        else:
            ok = False
            say(f"  attack        MISSING {script}")

    for mod in ("torch", "torch_geometric", "numpy", "scipy", "sklearn"):
        try:
            m = __import__(mod)
            say(f"  {mod:13s} {getattr(m, '__version__', 'ok')}")
        except Exception as e:
            ok = False
            say(f"  {mod:13s} MISSING ({type(e).__name__})")

    say("  all dependencies resolved" if ok else "  UNRESOLVED (see above)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
