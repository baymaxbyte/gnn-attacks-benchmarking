"""Master script: one GOREN leakage benchmark run, end to end, all four datasets.

    python run_benchmark.py                        # cora, citeseer, chameleon, squirrel
    python run_benchmark.py --directions A,B        # cora/citeseer only

Stages:

    1   Loukas coarsening (needs the GraphCoarsening checkout's interpreter) (step01)
    2a  GOREN edge-weight learning                                          (step02a)
    2   build the black boxes (bl + goren variants)                         (step02)
    3   reference model g + attack datasets on ORIGINAL edges               (step03)
    4   all 8 attacks + link-prediction baseline                           (step04)
    5   CSVs, figures and the writeup                                       (step05)

Datasets are driven by config.DATASETS (cora, citeseer, chameleon, squirrel):
stages 1-3 prepare all of them, and --directions selects which get attacked
(A: cora->citeseer, B: citeseer->cora, C: chameleon->squirrel, D: squirrel->
chameleon).

Stage 1 shells out to the GraphCoarsening checkout under its own venv. This script
points GOREN_HOME / GOREN_COARSEN_PYTHON at ../../../../GraphCoarsening/
GraphCoarsening-main by default (config.py's built-in default is one layer too
shallow); override by exporting those variables before running.

Resumable: step01 and step02a are skipped when their artifacts already exist for
every dataset (unless --force); step03/step04 skip what already exists; step05 is
idempotent.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ("cora", "citeseer", "chameleon", "squirrel")

# config.py resolves GOREN_HOME relative to HERE but one layer too shallow, so set
# the correct default here (project-root GraphCoarsening). setdefault respects an
# explicit export.
_GOREN_HOME = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..",
                                           "GraphCoarsening", "GraphCoarsening-main"))
os.environ.setdefault("GOREN_HOME", _GOREN_HOME)
_coarsen_py = os.path.join(os.environ["GOREN_HOME"], ".venv", "bin", "python3")
if os.path.exists(_coarsen_py):
    os.environ.setdefault("GOREN_COARSEN_PYTHON", _coarsen_py)


def stage(label, script, args, python):
    print("\n" + "=" * 78)
    print(f"  {label}")
    print("=" * 78, flush=True)
    t0 = time.time()
    p = subprocess.run([python, script] + args, cwd=HERE)
    dt = (time.time() - t0) / 60
    if p.returncode != 0:
        raise SystemExit(f"\n{script} failed (exit {p.returncode}) after {dt:.1f} min")
    print(f"\n  {label} done in {dt:.1f} min", flush=True)


def _all_exist(pattern):
    return all(glob.glob(os.path.join(HERE, "artifacts", "coarsened",
                                      pattern.format(ds=ds))) for ds in DATASETS)


def main():
    ap = argparse.ArgumentParser(description="GOREN leakage benchmark, end to end")
    ap.add_argument("--ratios", default="0.3,0.5,0.7")
    ap.add_argument("--directions", default="A,B,C,D",
                    help="A cora->citeseer, B citeseer->cora, "
                         "C chameleon->squirrel, D squirrel->chameleon")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--skip_coarsen", action="store_true")
    ap.add_argument("--skip_weights", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="recoarsen / relearn weights even if artifacts exist")
    ap.add_argument("--skip_attacks", action="store_true")
    ap.add_argument("--no_figures", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    if not os.path.isdir(os.environ.get("GOREN_HOME", "")):
        print(f"  ! GOREN_HOME not found: {os.environ.get('GOREN_HOME')}\n"
              f"    export GOREN_HOME / GOREN_COARSEN_PYTHON, or pass --skip_coarsen "
              f"to reuse shipped coarsenings.")

    variants = ("bl", "goren")
    settings = ",".join(["whole"] + [f"{r}_{v}" for r in args.ratios.split(",")
                                    for v in variants])
    t0 = time.time()

    if args.skip_coarsen or (not args.force and _all_exist("C_{ds}_r*.npz")):
        print("STAGE 1  Loukas coarsening -- skipped (every dataset already "
              "coarsened; --force to redo)")
    else:
        print(f"  GOREN_HOME={os.environ.get('GOREN_HOME')}")
        print(f"  GOREN_COARSEN_PYTHON={os.environ.get('GOREN_COARSEN_PYTHON')}")
        stage("STAGE 1  Loukas coarsening", "step01_coarsen.py",
              ["--ratios", args.ratios], args.python)

    if args.skip_weights or (not args.force and _all_exist("goren_w_{ds}_r*.npy")):
        print("STAGE 2a  GOREN weight learning -- skipped (weights already exist "
              "for every dataset; --force to redo)")
    else:
        stage("STAGE 2a  GOREN weight learning", "step02a_goren_weights.py",
              ["--ratios", args.ratios], args.python)

    stage("STAGE 2  black boxes", "step02_blackbox.py",
          ["--repeats", str(args.repeats), "--workers", str(args.workers)],
          args.python)

    if args.skip_attacks:
        print(f"\nstopped after the black boxes; total {(time.time()-t0)/60:.1f} min")
        return

    stage("STAGE 3  attacker: reference model g + attack datasets",
          "step03_attacker.py", ["--settings", settings], args.python)
    stage("STAGE 4  the 8 attacks", "step04_attacks.py",
          ["--settings", settings, "--directions", args.directions,
           "--runs", str(args.runs), "--workers", str(args.workers)], args.python)
    stage("STAGE 5  CSVs, figures, writeup", "step05_aggregate.py",
          ["--no_figures"] if args.no_figures else [], args.python)

    print(f"\n{'=' * 78}")
    print(f"  BENCHMARK COMPLETE in {(time.time() - t0) / 60:.1f} min")
    print(f"  datasets={list(DATASETS)}  directions={args.directions}")
    print("  see results/GOREN_RESULTS.md")
    print("=" * 78)


if __name__ == "__main__":
    main()
