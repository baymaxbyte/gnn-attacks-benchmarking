"""Step 04 -- run all 8 link-stealing attacks against every GOREN black box.

    direction A:  target Cora,     shadow Citeseer
    direction B:  target Citeseer, shadow Cora

    python step04_attacks.py
    python step04_attacks.py --settings whole,0.7_bl,0.7_goren --directions A
"""
import argparse
import os
import subprocess
import time

import common as K

os.environ.update(K.single_thread_env())

import config as C

SETTINGS = ["whole"] + [f"{r}_{v}" for r in C.RATIOS for v in C.VARIANTS]
DIRECTIONS = {"A": ("cora", "citeseer"), "B": ("citeseer", "cora")}
RESULTS = os.path.join(C.RESULTS, "attacks")
SET_TAG = "goren"


def _job(args):
    direction, target, shadow, setting, runs, force = args
    K.pin_single_thread()
    tag = f"{SET_TAG}_{direction}_{setting}"
    out = os.path.join(RESULTS, f"{target}_{SET_TAG}_{setting}__"
                                f"{shadow}_{SET_TAG}_{setting}__{tag}.json")
    if os.path.exists(out) and not force:
        return {"tag": tag, "path": out, "cached": True, "minutes": 0.0}
    cmd = [C.PYTHON, "run_attack.py",
           "--attack", "all",
           "--dataset", f"{target}_{SET_TAG}_{setting}",
           "--shadow", f"{shadow}_{SET_TAG}_{setting}",
           "--attack_dataset_path", "data/attack_dataset/",
           "--result_path", RESULTS,
           "--runs", str(runs),
           "--tag", tag,
           "--link_prediction", "--buckets", "--quiet"]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=C.ATTACK_HOME, capture_output=True, text=True)
    logf = os.path.join(C.LOGS, f"step04_{tag}_{target}.log")
    with open(logf, "w") as f:
        f.write(p.stdout + "\n===== STDERR =====\n" + p.stderr)
    if p.returncode != 0:
        return {"tag": tag, "path": out, "failed": True, "log": K.rel(logf),
                "stderr": p.stderr[-1500:], "minutes": (time.time() - t0) / 60}
    return {"tag": tag, "path": out, "cached": False,
            "minutes": round((time.time() - t0) / 60, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--directions", default="A,B")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    K.ensure_dirs()
    os.makedirs(RESULTS, exist_ok=True)
    settings = [s.strip() for s in args.settings.split(",")]
    dirs = [d.strip() for d in args.directions.split(",")]

    jobs = [(d, *DIRECTIONS[d], s, args.runs, args.force)
            for d in dirs for s in settings]
    print(f"{len(jobs)} configurations x 8 attacks + link-prediction baseline, "
          f"{args.runs} runs each, {args.workers} workers\n")

    t0 = time.time()
    res = K.parallel_map(_job, jobs, workers=args.workers, serial=args.serial)

    failed = [r for r in res if r.get("failed")]
    for r in res:
        state = "FAILED" if r.get("failed") else ("cached" if r["cached"] else "ok")
        print(f"  {r['tag']:26s} {state:7s} {r['minutes']:5.2f} min")
    if failed:
        for r in failed:
            print(f"\n{r['tag']} stderr tail:\n{r['stderr']}")
        raise SystemExit(f"{len(failed)} configuration(s) failed")

    K.record(
        "step04_attacks",
        summary=(f"Ran all 8 attacks + link-prediction baseline against {len(res)} "
                 f"GOREN black-box configurations ({len(dirs)} directions x "
                 f"{len(settings)} settings), {args.runs} runs each"),
        inputs=[f"attack/data/attack_dataset/*_{SET_TAG}_*.json"],
        outputs=[K.rel(r["path"]) for r in res],
        numbers={"configurations": len(res), "attack_runs": args.runs},
        notes=("Attack model per He et al. USENIX Security 2021. bl and goren boxes "
               "at the same ratio differ only in the coarse edge weights their target "
               "was trained on. Attacks 0 and 2 are unsupervised and run once."))
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
