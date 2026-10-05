"""Config for the node membership-inference benchmark.

V_in (80%) = members, V_out (20%) = non-members (nodes and edges removed). Every
reduction method is applied to V_in only, so membership is defined identically
across methods (member = in V_in). Shadow is the other dataset at the same cell.

    python config.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

SPLITS = os.path.join(HERE, "artifacts", "splits")
REDUCED = os.path.join(HERE, "artifacts", "reduced")
POSTERIORS = os.path.join(HERE, "artifacts", "posteriors")
RESULTS = os.path.join(HERE, "results")
LOGS = os.path.join(HERE, "logs")
MANIFEST = os.path.join(HERE, "manifest.json")

PYTHON = os.path.abspath(os.environ.get("MIA_PYTHON", sys.executable))

DATASETS = ("cora", "citeseer")
SHADOW_OF = {"cora": "citeseer", "citeseer": "cora"}   # cross-dataset shadow

# Inductive member / non-member split.
HOLDOUT_FRACTION = 0.20        # fraction of nodes that become non-members (V_out)
SPLIT_SEED = 42                # one split, shared by every method

# Three target models per cell, for error bars on the attack AUC.
TARGET_SEEDS = (0, 1, 2)

# Reduction methods. "control" = a plain GNN trained on V_in with no reduction, the
# reference leakage level. Per-method ratio conventions live with each adapter in
# reductions/, since they differ (GCond condensation rate, UGC keep%, KRON fraction
# removed, GOREN's bl/goren variants); METHOD_CELLS is filled in there.
METHODS = ("control", "gcond", "fgc", "ugc", "goren", "kron")


def check(verbose=True):
    ok = True

    def say(m):
        if verbose:
            print(m)

    say(f"  HERE          {HERE}")
    for ds in DATASETS:
        p = os.path.join(HERE, f"{ds}.npz")
        if os.path.exists(p):
            say(f"  {ds:9s}     whole graph present ({ds}.npz)")
        else:
            ok = False
            say(f"  {ds:9s}     MISSING {ds}.npz")

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
