"""Paths and environment self-check for the FGC leakage benchmark.

Unlike the GCond benchmark this pipeline has no external code dependency: the
FGC solver is re-implemented in fgc_solver.py and the datasets are read from the
Planetoid pickles vendored under data/dataset/original/.

    python config.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

DATA = os.path.join(HERE, "attack", "data", "dataset", "original")

# Attack implementation, vendored under attack/.
ATTACK_HOME = os.environ.get("FGC_ATTACK_HOME", os.path.join(HERE, "attack"))
ATTACK_PRED = os.path.join(ATTACK_HOME, "data", "pred")
ATTACK_DATASET = os.path.join(ATTACK_HOME, "data", "attack_dataset")
# Absolute, because the attack scripts are invoked with cwd=ATTACK_HOME and a
# relative interpreter path would not resolve from there.
PYTHON = os.path.abspath(os.environ.get("FGC_PYTHON", sys.executable))
ARTIFACTS = os.path.join(HERE, "artifacts")
COARSENED = os.path.join(ARTIFACTS, "coarsened")
RESULTS = os.path.join(HERE, "results")
LOGS = os.path.join(HERE, "logs")
MANIFEST = os.path.join(HERE, "manifest.json")

DATASETS = ("cora", "citeseer")
_PICKLE_SUFFIXES = ("x", "y", "tx", "ty", "allx", "ally", "graph", "test.index")


def check(verbose=True):
    """Return True if everything the pipeline needs is present."""
    ok = True

    def say(msg):
        if verbose:
            print(msg)

    say(f"  HERE          {HERE}")
    for ds in DATASETS:
        missing = [s for s in _PICKLE_SUFFIXES
                   if not os.path.exists(os.path.join(DATA, f"ind.{ds}.{s}"))]
        if missing:
            ok = False
            say(f"  {ds:9s}     MISSING: {', '.join('ind.%s.%s' % (ds, m) for m in missing)}")
        else:
            say(f"  {ds:9s}     8/8 Planetoid files present")

    for script in ("partial_graph_generation.py", "run_attack.py"):
        if os.path.exists(os.path.join(ATTACK_HOME, script)):
            say(f"  attack        {script} found")
        else:
            ok = False
            say(f"  attack        MISSING {os.path.join(ATTACK_HOME, script)}")

    for mod in ("torch", "torch_geometric", "numpy", "scipy"):
        try:
            m = __import__(mod)
            say(f"  {mod:13s} {getattr(m, '__version__', '?')}")
        except ImportError:
            ok = False
            say(f"  {mod:13s} MISSING")

    say("  all dependencies resolved" if ok else "  UNRESOLVED dependencies (see above)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
