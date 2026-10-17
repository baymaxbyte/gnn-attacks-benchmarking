"""Paths, constants and environment self-check for the GOREN leakage benchmark.

    python config.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

ATTACK_HOME = os.environ.get("GOREN_ATTACK_HOME", os.path.join(HERE, "attack"))
DATA = os.path.join(ATTACK_HOME, "data", "dataset", "original")
ATTACK_PRED = os.path.join(ATTACK_HOME, "data", "pred")
ATTACK_DATASET = os.path.join(ATTACK_HOME, "data", "attack_dataset")

ARTIFACTS = os.path.join(HERE, "artifacts")
COARSENED = os.path.join(ARTIFACTS, "coarsened")
POSTERIORS = os.path.join(ARTIFACTS, "posteriors")
RESULTS = os.path.join(HERE, "results")
LOGS = os.path.join(HERE, "logs")
MANIFEST = os.path.join(HERE, "manifest.json")

PYTHON = os.path.abspath(os.environ.get("GOREN_PYTHON", sys.executable))

# Loukas's graph_coarsening ships inside the GOREN repository. Override with GOREN_HOME.
GOREN_HOME = os.environ.get(
    "GOREN_HOME",
    os.path.abspath(os.path.join(HERE, "..", "..", "GraphCoarsening",
                                 "GraphCoarsening-main")))
COARSENING_PKG = os.path.join(GOREN_HOME, "sparsenet", "evaluation",
                              "graph-coarsening")

# Stage 1 needs pygsp and the GOREN repo's `sparsenet`, so it runs in that
# checkout's interpreter. Override with GOREN_COARSEN_PYTHON.
_repo_py = os.path.join(GOREN_HOME, ".venv", "bin", "python3")
COARSEN_PYTHON = os.path.abspath(os.environ.get(
    "GOREN_COARSEN_PYTHON", _repo_py if os.path.exists(_repo_py) else sys.executable))

DATASETS = ("cora", "citeseer")

# r is the fraction of nodes REMOVED; the coarse graph keeps about n*(1-r).
RATIOS = (0.3, 0.5, 0.7)

# Loukas scheme GOREN builds on.
METHOD = "variation_neighborhoods"
K_EIGS = 10               # Loukas's default K, the number of eigenvectors preserved

# Edge-weight variants served as separate black boxes.
VARIANTS = ("bl", "goren")

_PICKLES = ("x", "y", "tx", "ty", "allx", "ally", "graph", "test.index")


def n_supernodes(n, r):
    return int(round(n * (1.0 - r)))


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
            say(f"  {ds:9s}     8/8 Planetoid files")

    for script in ("partial_graph_generation.py", "run_attack.py"):
        if os.path.exists(os.path.join(ATTACK_HOME, script)):
            say(f"  attack        {script} found")
        else:
            ok = False
            say(f"  attack        MISSING {script}")

    if os.path.isdir(COARSENING_PKG):
        say(f"  coarsening    graph-coarsening found under GOREN_HOME")
    else:
        ok = False
        say(f"  coarsening    MISSING {COARSENING_PKG}")

    for mod in ("torch", "torch_geometric", "numpy", "scipy"):
        try:
            m = __import__(mod)
            say(f"  {mod:13s} {getattr(m, '__version__', 'ok')}")
        except Exception as e:
            ok = False
            say(f"  {mod:13s} MISSING ({type(e).__name__})")

    # pygsp and graph_coarsening are needed only by stage 1, which runs under
    # COARSEN_PYTHON. Their absence here does not block stages 2a onward.
    if COARSENING_PKG not in sys.path:
        sys.path.insert(0, COARSENING_PKG)
    stage1 = True
    for mod in ("pygsp", "graph_coarsening"):
        try:
            m = __import__(mod)
            say(f"  {mod:13s} {getattr(m, '__version__', 'ok')}  (stage 1)")
        except Exception:
            stage1 = False
            say(f"  {mod:13s} not in this interpreter  (stage 1 only)")

    say("  core dependencies resolved" if ok else "  UNRESOLVED core deps (see above)")
    if not stage1:
        say("  stage 1 needs GOREN_COARSEN_PYTHON; stages 2a onward run from the "
            "shipped artifacts")
    return ok


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
