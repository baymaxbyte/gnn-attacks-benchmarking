"""External dependency resolution. Edit here, or set the environment variables.

GCond's code is not redistributed with this benchmark -- clone it yourself:

    git clone https://github.com/ChandlerBang/GCond

Then either place it at ../../GCond/GCond-main relative to this file, or point
GCOND_HOME at it.

On a machine without CUDA that checkout needs two small patches: a CPU
fallback in train_gcond_transduct.py, and --save redirected away from the
authors' bundled saved_ours/ directory.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# The vendored attack implementation (our own reimplementation of He et al.).
ATTACK_HOME = os.path.join(HERE, "attack")

# GCond checkout: condensation code, its datasets, and its virtualenv.
GCOND_HOME = os.environ.get(
    "GCOND_HOME",
    os.path.abspath(os.path.join(HERE, "..", "..", "GCond", "GCond-main")))

# Interpreter used for the GCond subprocesses. Its environment must satisfy
# GCond's requirements (deeprobust, torch, scipy, networkx, scikit-learn,
# matplotlib). Defaults to GCond's own venv if present, else the current one.
_venv = os.path.join(GCOND_HOME, ".venv", "bin", "python3")
GCOND_PYTHON = os.environ.get("GCOND_PYTHON",
                              _venv if os.path.exists(_venv) else sys.executable)


def check(verbose=True):
    """Report whether the external pieces are where we expect them."""
    problems = []
    if not os.path.isdir(GCOND_HOME):
        problems.append(f"GCOND_HOME not found: {GCOND_HOME}")
    else:
        for f in ("train_gcond_transduct.py", "utils.py",
                  os.path.join("models", "gcn.py")):
            if not os.path.exists(os.path.join(GCOND_HOME, f)):
                problems.append(f"missing in GCOND_HOME: {f}")
    if not os.path.exists(GCOND_PYTHON):
        problems.append(f"GCOND_PYTHON not executable: {GCOND_PYTHON}")
    if not os.path.isdir(os.path.join(ATTACK_HOME, "lsa")):
        problems.append(f"vendored attack package missing: {ATTACK_HOME}/lsa")
    planetoid = os.path.join(ATTACK_HOME, "data", "dataset", "original")
    if not os.path.isdir(planetoid):
        problems.append(f"Planetoid pickles missing: {planetoid}")

    if verbose:
        print(f"  ATTACK_HOME   {ATTACK_HOME}")
        print(f"  GCOND_HOME    {GCOND_HOME}")
        print(f"  GCOND_PYTHON  {GCOND_PYTHON}")
        for p in problems:
            print(f"  ! {p}")
        if not problems:
            print("  all external dependencies resolved")
    return problems


if __name__ == "__main__":
    sys.exit(1 if check() else 0)
