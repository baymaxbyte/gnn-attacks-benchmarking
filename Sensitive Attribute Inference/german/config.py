"""Config for the sensitive-attribute inference benchmark.

The black box is a GNN trained on a reduced graph using the full node features, one
of which is designated SENSITIVE. The adversary holds every feature except the
sensitive bit and recovers it by a two-world counterfactual probe: set the whole
sensitive column to 0, serve -> p0; set it to 1, serve -> p1; Z_u = [p0 || p1]
(dimension 2*C). An MLP attack classifier maps Z_u to the true sensitive bit, scored
by AUC on a held-out set of nodes (same dataset, no shadow).

Unlike Cora/Citeseer (which have no native sensitive attribute and need balanced
binary features as stand-ins), German carries a real binary sensitive attribute --
Gender -- which sits in the feature matrix at index 0. That single column is the
sensitive bit the two-world probe toggles.

    python config.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

REDUCED = os.path.join(HERE, "artifacts", "reduced")
ZDIR = os.path.join(HERE, "artifacts", "Z")           # Z_u = [p0 || p1] per cell/feature/seed
RESULTS = os.path.join(HERE, "results")
LOGS = os.path.join(HERE, "logs")
MANIFEST = os.path.join(HERE, "manifest.json")

PYTHON = os.path.abspath(os.environ.get("SIA_PYTHON", sys.executable))

DATASETS = ("german",)

# German's real sensitive attribute is Gender, mapped Female=1/Male=0 and kept in the
# feature matrix at column 0 (features = german.csv minus GoodCustomer/OtherLoansAtStore/
# PurposeOfLoan). The two-world probe toggles this single column.
SENSITIVE = {"german": [0]}

METHODS = ("control", "kron", "ugc", "fgc", "goren", "gcond")

# Three target models per cell, for error bars on the attack AUC.
TARGET_SEEDS = (0, 1, 2)


def check(verbose=True):
    ok = True

    def say(m):
        if verbose:
            print(m)

    say(f"  HERE          {HERE}")
    for ds in DATASETS:
        p = os.path.join(HERE, f"{ds}.npz")
        say(f"  {ds:9s}     {'whole graph present' if os.path.exists(p) else 'MISSING ' + ds + '.npz'}")
        ok = ok and os.path.exists(p)
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
