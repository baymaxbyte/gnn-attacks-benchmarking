"""Master script: one FGC leakage benchmark run, end to end.

    python run_benchmark.py
    python run_benchmark.py --settings whole,0.1 --runs 3

Runs every stage in order:

    1  coarsen both datasets, selecting beta per setting        (step01)
    2  build the black boxes and validate them                  (step02)
    3  train the adversary's reference model g                  (step03)
    4  build attack datasets on ORIGINAL edges                  (step03)
    5  run all 8 attacks + link-prediction baseline             (step04)
    6  compile CSVs, figures and the writeup                    (step05)

r is the fraction of nodes KEPT, so the coarse graph has k = int(n*r) supernodes.
Stage 3 writes several GB of attack datasets under attack/data/attack_dataset/.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


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
    return dt


def main():
    ap = argparse.ArgumentParser(description="FGC leakage benchmark, end to end")
    ap.add_argument("--settings", default="whole,0.3,0.1,0.05")
    ap.add_argument("--beta", type=float, default=None,
                    help="skip the beta search and use this value everywhere")
    ap.add_argument("--repeats", type=int, default=10,
                    help="GNN re-inits per black box")
    ap.add_argument("--runs", type=int, default=5, help="attack-model repeats")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip_coarsen", action="store_true",
                    help="reuse artifacts/coarsened/")
    ap.add_argument("--skip_attacks", action="store_true",
                    help="stop after the black boxes are validated")
    ap.add_argument("--no_figures", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    coarse = [s for s in args.settings.split(",") if s.strip() != "whole"]
    t0 = time.time()

    if not args.skip_coarsen and coarse:
        a = ["--workers", str(args.workers)]
        if args.beta is not None:
            a += ["--beta", str(args.beta)]
        stage("STAGE 1  coarsen", "step01_coarsen.py", a, args.python)

    stage("STAGE 2  black boxes + validation", "step02_blackbox.py",
          ["--repeats", str(args.repeats), "--workers", str(args.workers)],
          args.python)

    if args.skip_attacks:
        print(f"\nstopped after validation; total {(time.time()-t0)/60:.1f} min")
        return

    stage("STAGE 3  attacker: reference model g + attack datasets",
          "step03_attacker.py", ["--settings", args.settings], args.python)

    stage("STAGE 4  the 8 attacks", "step04_attacks.py",
          ["--settings", args.settings, "--runs", str(args.runs),
           "--workers", str(args.workers)], args.python)

    a = ["--no_figures"] if args.no_figures else []
    stage("STAGE 5  CSVs, figures, writeup", "step05_aggregate.py", a, args.python)

    print(f"\n{'=' * 78}")
    print(f"  BENCHMARK COMPLETE in {(time.time() - t0) / 60:.1f} min")
    print(f"  see results/FGC_RESULTS.md")
    print("=" * 78)


if __name__ == "__main__":
    main()
