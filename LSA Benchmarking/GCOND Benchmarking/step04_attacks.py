"""Step 04 -- run all 8 attacks against every black box, both directions.

    direction A:  target Cora,     shadow Citeseer   (same setting)
    direction B:  target Citeseer, shadow Cora       (same setting)

    python step04_attacks.py
    python step04_attacks.py --settings whole,0.5 --directions A
"""
import argparse
import os
import subprocess
import time

import common as C

GCOND_PY = C.GCOND_PY
SETTINGS = ["whole", "0.25", "0.5", "1.0"]
DIRECTIONS = {"A": ("cora", "citeseer"), "B": ("citeseer", "cora")}
RESULTS = os.path.join(C.HERE, "results", "attacks")


def run_one(direction, target, shadow, setting, runs, quiet=True):
    tag = f"gcond_{direction}_{setting}"
    out = os.path.join(RESULTS, f"{target}_gc_{setting}__{shadow}_gc_"
                                f"{setting}__{tag}.json")
    if os.path.exists(out):
        print(f"  [cached] {os.path.basename(out)}")
        return out, None

    cmd = [GCOND_PY, "run_attack.py",
           "--attack", "all",
           "--dataset", f"{target}_gc_{setting}",
           "--shadow", f"{shadow}_gc_{setting}",
           "--attack_dataset_path", "data/attack_dataset/",
           "--result_path", RESULTS,
           "--runs", str(runs),
           "--tag", tag,
           "--link_prediction",
           "--buckets"]
    if quiet:
        cmd.append("--quiet")

    print(f"\n>>> direction {direction}: target={target} shadow={shadow} "
          f"setting={setting}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=C.LSA, capture_output=True, text=True)
    logf = os.path.join(C.DIRS["logs"], f"step04_{tag}_{target}.log")
    with open(logf, "w") as f:
        f.write(p.stdout + "\n===== STDERR =====\n" + p.stderr)
    if p.returncode != 0:
        print(p.stdout[-2500:]); print(p.stderr[-2500:])
        raise SystemExit(f"attacks FAILED for {tag}; see {logf}")

    keep = False
    for line in p.stdout.splitlines():
        if line.startswith("SUMMARY"):
            keep = True
        if keep and line.strip():
            print("   ", line.rstrip())
    print(f"    {(time.time() - t0) / 60:.1f} min")
    return out, p.stdout


def main():
    ap = argparse.ArgumentParser(description="Run the 8 attacks per black box")
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--directions", default="A,B")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    C.ensure_dirs()
    os.makedirs(RESULTS, exist_ok=True)
    settings = [s.strip() for s in args.settings.split(",")]
    dirs = [d.strip() for d in args.directions.split(",")]

    outputs = []
    t0 = time.time()
    for d in dirs:
        target, shadow = DIRECTIONS[d]
        for s in settings:
            out, _ = run_one(d, target, shadow, s, args.runs)
            outputs.append(C.rel(out))

    C.record(
        step="04_attacks_gcond",
        summary=(f"Ran all 8 attacks + link-prediction baseline against "
                 f"{len(outputs)} black-box configurations "
                 f"({len(dirs)} directions x {len(settings)} settings), "
                 f"{args.runs} runs each"),
        inputs=["Benchmarking/lsa_paper/data/attack_dataset/*_gc_*.json"],
        outputs=outputs,
        numbers={"configurations": len(outputs), "attack_runs": args.runs},
        notes="Shadow target uses the same condensation setting as the target. Attacks 0 and 2 are unsupervised and run once.")
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
