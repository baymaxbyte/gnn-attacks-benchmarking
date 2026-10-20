"""Central config + consistency matrix for the PubMed attack pipeline.

This folder is an ORCHESTRATION layer. It does NOT re-implement the attacks or the
coarsening/condensation methods -- it drives the already-verified sibling
benchmarks so PubMed is produced with byte-identical code, seeds, ratios and
epochs as cora/citeseer/chameleon/squirrel:

    ../lsa/<METHOD> Benchmarking/     the 5 link-stealing (LSA) pipelines
    ../Membership Attack/             the node membership-inference pipeline

Target  = pubmed   (homophilous citation graph, ~19,717 nodes, 500 feats, 3 classes)
Shadow  = cora     (ALWAYS the whole / uncoarsened graph), chosen because its
                    homophily (~0.81) matches pubmed (~0.80); heterophilous
                    chameleon/squirrel would transfer poorly.

Everything configurable lives here so the run is reproducible and auditable.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEMP = os.path.abspath(os.path.join(HERE, ".."))              # Benchmarking/temp
# This folder is SELF-CONTAINED: the sibling benchmark code is vendored under
# ./vendor/ (byte-identical copies), so uploading just this folder to SageMaker is
# enough -- no dependency on the original sibling folders. All data and the Cora
# shadow regenerate from scratch (deterministic), so results are unchanged.
# The roots stay env-overridable if you ever want to point at an external layout:
#   export PUBMED_ATTACKS_LSA_ROOT=/path/to/lsa
#   export PUBMED_ATTACKS_MEMBERSHIP_ROOT="/path/to/Membership Attack"
VENDOR = os.path.join(HERE, "vendor")
LSA_ROOT = os.environ.get("PUBMED_ATTACKS_LSA_ROOT", os.path.join(VENDOR, "lsa"))
MEMBERSHIP_ROOT = os.environ.get("PUBMED_ATTACKS_MEMBERSHIP_ROOT",
                                 os.path.join(VENDOR, "Membership Attack"))

# Where this pipeline writes its own artifacts (logs, progress, collected results).
RESULTS = os.path.join(HERE, "results")
LOGS = os.path.join(HERE, "logs")
PROGRESS = os.path.join(HERE, "progress.json")               # resume manifest
MAIN_LOG = os.path.join(LOGS, "pubmed_attacks.log")          # live, appended

# Interpreter used to run every sibling step. On SageMaker this is just the env
# you `pip install -r requirements.txt` into; override with PUBMED_ATTACKS_PY.
PYTHON = os.path.abspath(os.environ.get("PUBMED_ATTACKS_PY", sys.executable))

# ---------------------------------------------------------------- target/shadow
TARGET = "pubmed"
SHADOW = "cora"                     # the WHOLE cora graph is always the shadow

# PubMed has ~44k undirected edges, far above the positive-pair cap the dense
# benchmarks use; without this the attack datasets explode. Matches the
# chameleon/squirrel cap in run_wholeshadow.py.
MAX_POS = 10000

# ======================================================================= LSA
# One entry per link-stealing method. `dir` is the sibling benchmark folder,
# `set_tag/bb/g` are its naming constants, `settings` is the EXACT ordered list
# of reduction settings that method exposes (first is always the whole-graph
# control). `external` flags methods that need a separate checkout/venv (same
# requirement as the original runs); `env` lists the env vars that point at them.
LSA_SPECS = {
    "fgc": dict(
        dir="FGC Benchmarking", set_tag="fgc", bb="fgcbb", g="gfgc",
        settings=["whole", "0.3", "0.1", "0.05"],
        external=None, env=["FGC_PYTHON"]),
    "kron": dict(
        dir="KRON Benchmarking", set_tag="kron", bb="kronbb", g="gkron",
        settings=["whole", "0.3", "0.5", "0.7"],
        external=None, env=["KRON_PYTHON"]),
    "ugc": dict(
        dir="UGC Benchmarking", set_tag="ugc", bb="ugcbb", g="gugc",
        settings=["whole", "30", "50", "70"],
        external=None, env=["UGC_PYTHON"]),
    "goren": dict(
        dir="GOREN Benchmarking", set_tag="goren", bb="gorenbb", g="ggoren",
        settings=["whole", "0.3_bl", "0.3_goren", "0.5_bl", "0.5_goren",
                  "0.7_bl", "0.7_goren"],
        external="graph-coarsening venv (Loukas)",
        env=["GOREN_PYTHON", "GOREN_COARSEN_PYTHON"]),
    "gcond": dict(
        dir="GCOND Benchmarking", set_tag="gc", bb="gcondbb", g="g",
        settings=["whole", "0.25", "0.5", "1.0"],
        external="GCond checkout (train_gcond_transduct.py)",
        env=["GCOND_HOME", "GCOND_PYTHON"]),
}

# Fully in-repo methods (no external checkout) run by default.
LSA_DEFAULT_METHODS = ["fgc", "kron", "ugc"]
# External methods: included only if explicitly requested AND their env is set.
LSA_EXTERNAL_METHODS = ["goren", "gcond"]

LSA_RESULTS_SUBDIR = "attacks_wholeshadow"   # where whole-shadow JSONs are written

# ================================================================ MEMBERSHIP
MEMBERSHIP_METHODS = ["control", "kron", "ugc", "fgc", "goren", "gcond"]
MEMBERSHIP_DEFAULT_METHODS = ["control", "kron", "ugc", "fgc"]
MEMBERSHIP_EXTERNAL_METHODS = ["goren", "gcond"]
# Per-method reduction cells (ratios / variants) -- for reference + the README;
# the actual values live in ../Membership Attack/reductions/<method>.py and are
# NOT re-declared, only mirrored here for the consistency table.
MEMBERSHIP_CELLS = {
    "control": ["whole"],
    "kron": ["0.3", "0.5", "0.7"],
    "ugc": ["0.3", "0.5", "0.7"],
    "fgc": ["0.3", "0.1", "0.05"],
    "goren": ["0.3_bl", "0.3_goren", "0.5_bl", "0.5_goren", "0.7_bl", "0.7_goren"],
    "gcond": ["0.013", "0.026", "0.052"],
}

# ------------------------------------------------- one unavoidable NEW hyperparam
# UGC augments features as [(1-alpha)X | alpha*A] and the alpha is dataset-tuned.
# The reproductions use cora 0.19, citeseer 0.26, and 0.2 for datasets without a
# published value (chameleon, squirrel). PubMed has NO published UGC alpha
# anywhere, so we adopt the same 0.2 default used for the other non-Planetoid-
# tuned datasets. This is the ONLY setting invented for PubMed; everything else
# is inherited unchanged. Flagged loudly in the README.
UGC_ALPHA_PUBMED = 0.2

# ------------------------------------------------- shared consistency constants
# (declared in the siblings; copied here only so the run can assert them and the
# README can show them. Source of truth remains the sibling code.)
CONSISTENCY = {
    "lsa_attack": dict(label_rate=0.10, n_runs=5, attack_epochs=50,
                       attack_mlp="3x32 dropout0.5 lr1e-3 batch128",
                       pair_seed=1, label_seed=123, canonical_seed=0,
                       run_attack_seed=42, resplit_per_run=False),
    "lsa_blackbox": dict(fgc="h64 do0.5 lr0.01 wd1e-4 ep60",
                         kron="h64 do0.5 lr0.003 wd5e-4 ep500 split60/20/20 seed0",
                         ugc="h64 do0.5 lr0.003 wd5e-4 ep500 split60/20/20 seed0",
                         goren="h64 do0.5 lr0.003 wd5e-4 ep500 + weights ep400",
                         gcond="eval ep600 rep10 lr0.01 do0.0 wd5e-4; condense ep600"),
    "membership": dict(holdout_fraction=0.20, split_seed=42,
                       target_seeds=(0, 1, 2),
                       blackbox="3-layer GCN d-64-64-C h64 do0.5 lr0.003 wd5e-4 ep500",
                       attack_mlp="3x32 dropout0.5 50ep lr1e-3 batch128",
                       shadow="whole cora (cora_control_whole), dimension-agnostic 8-dim feats"),
}


def resolve_methods(requested, which):
    """Expand a comma list / 'all' into a validated method list for `which`
    in {'lsa','membership'}, dropping external methods whose env isn't set."""
    if which == "lsa":
        known, default, external = (list(LSA_SPECS), LSA_DEFAULT_METHODS,
                                    LSA_EXTERNAL_METHODS)
    else:
        known = MEMBERSHIP_METHODS
        default = MEMBERSHIP_DEFAULT_METHODS
        external = MEMBERSHIP_EXTERNAL_METHODS
    if requested in (None, "", "default"):
        return list(default)
    if requested == "all":
        chosen = list(known)
    else:
        chosen = [m.strip() for m in requested.split(",") if m.strip()]
    bad = [m for m in chosen if m not in known]
    if bad:
        raise SystemExit(f"[config] unknown {which} methods: {bad}; known={known}")
    return chosen


def env_ready(method, which):
    """Return (ok, reason) for whether an external method's checkout is wired."""
    if which == "lsa":
        spec = LSA_SPECS.get(method, {})
        if not spec.get("external"):
            return True, ""
        if method == "gcond":
            ok = bool(os.environ.get("GCOND_HOME"))
            return ok, ("" if ok else "set GCOND_HOME/GCOND_PYTHON to the GCond checkout")
        if method == "goren":
            ok = bool(os.environ.get("GOREN_COARSEN_PYTHON"))
            return ok, ("" if ok else "set GOREN_COARSEN_PYTHON to the graph-coarsening venv")
    else:
        if method not in MEMBERSHIP_EXTERNAL_METHODS:
            return True, ""
        if method == "gcond":
            ok = bool(os.environ.get("GCOND_HOME"))
            return ok, ("" if ok else "set GCOND_HOME/GCOND_PYTHON")
        if method == "goren":
            ok = bool(os.environ.get("GOREN_COARSEN_PYTHON"))
            return ok, ("" if ok else "set GOREN_COARSEN_PYTHON")
    return True, ""


if __name__ == "__main__":
    print("PubMed attack pipeline config")
    print(f"  HERE            {HERE}")
    print(f"  LSA_ROOT        {LSA_ROOT}  exists={os.path.isdir(LSA_ROOT)}")
    print(f"  MEMBERSHIP_ROOT {MEMBERSHIP_ROOT}  exists={os.path.isdir(MEMBERSHIP_ROOT)}")
    print(f"  PYTHON          {PYTHON}")
    print(f"  target={TARGET}  shadow={SHADOW} (whole)  max_pos={MAX_POS}")
    print(f"  UGC alpha(pubmed)={UGC_ALPHA_PUBMED}  (the only invented setting)")
    print(f"  LSA methods     {list(LSA_SPECS)}")
    print(f"  membership      {MEMBERSHIP_METHODS}")
