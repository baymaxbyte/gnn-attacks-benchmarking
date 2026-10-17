"""Master script: one GOREN leakage benchmark run, end to end.

    python run_benchmark.py

Stage 1 needs the GOREN checkout's interpreter for Loukas's coarsening; set
GOREN_HOME and GOREN_COARSEN_PYTHON, or pass --skip_coarsen to reuse the shipped
coarsening artifacts.
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


def main():
    ap = argparse.ArgumentParser(description="GOREN leakage benchmark, end to end")
    ap.add_argument("--ratios", default="0.3,0.5,0.7")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--skip_coarsen", action="store_true")
    ap.add_argument("--skip_weights", action="store_true")
    ap.add_argument("--skip_attacks", action="store_true")
    ap.add_argument("--no_figures", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    variants = ("bl", "goren")
    settings = ",".join(["whole"] + [f"{r}_{v}" for r in args.ratios.split(",")
                                    for v in variants])
    t0 = time.time()

    if not args.skip_coarsen:
        stage("STAGE 1  Loukas coarsening", "step01_coarsen.py",
              ["--ratios", args.ratios], args.python)
    if not args.skip_weights:
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
          ["--settings", settings, "--runs", str(args.runs),
           "--workers", str(args.workers)], args.python)
    stage("STAGE 5  CSVs, figures, writeup", "step05_aggregate.py",
          ["--no_figures"] if args.no_figures else [], args.python)

    print(f"\n{'=' * 78}")
    print(f"  BENCHMARK COMPLETE in {(time.time() - t0) / 60:.1f} min")
    print("  see results/GOREN_RESULTS.md")
    print("=" * 78)


if __name__ == "__main__":
    main()
