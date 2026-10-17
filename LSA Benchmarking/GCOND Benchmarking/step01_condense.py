"""Step 01 -- re-run GCond's condensation for Cora and Citeseer at all three
reduction rates. Produces the condensed graphs used from here on.

    python step01_condense.py
    python step01_condense.py --only cora
"""
import argparse
import os
import re
import shutil
import subprocess
import time

import common as C

GCOND = C.GCOND
GCOND_PY = C.GCOND_PY
CONDENSED = os.path.join(C.HERE, "artifacts", "condensed")

GRID = [
    ("cora", "0.25", "1.3%", 35),
    ("cora", "0.5", "2.6%", 70),
    ("cora", "1.0", "5.2%", 140),
    ("citeseer", "0.25", "0.9%", 30),
    ("citeseer", "0.5", "1.8%", 60),
    ("citeseer", "1.0", "3.6%", 120),
]

AUTHOR_ARGS = ["--nlayers=2", "--sgc=1", "--lr_feat=1e-4", "--lr_adj=1e-4",
               "--seed=1", "--epochs=600"]

SEED = 1


def condense(dataset, r, expect_nodes, force=False):
    tag = f"{dataset}_{r}_{SEED}"
    dest = {k: os.path.join(CONDENSED, f"{k}_{tag}.pt")
            for k in ("adj", "feat", "label")}
    if all(os.path.exists(p) for p in dest.values()) and not force:
        print(f"  [cached] {tag}")
        return dest, None

    cmd = [GCOND_PY, "-W", "ignore", "-u", "train_gcond_transduct.py",
           f"--dataset={dataset}", f"--r={r}", "--save=1", *AUTHOR_ARGS]
    print(f"\n>>> condensing {dataset} r={r}\n    {' '.join(cmd)}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=GCOND, capture_output=True, text=True)
    dur = time.time() - t0
    logf = os.path.join(C.DIRS["logs"], f"step01_condense_{tag}.log")
    with open(logf, "w") as f:
        f.write(p.stdout + "\n===== STDERR =====\n" + p.stderr)
    if p.returncode != 0:
        print(p.stdout[-2500:])
        print(p.stderr[-2500:])
        raise SystemExit(f"condensation FAILED for {tag}; see {logf}")

    acc = None
    for m in re.finditer(r"Train/Test Mean Accuracy: \[array\(\[([\d.,\s]+)\]\), "
                         r"array\(\[([\d.,\s]+)\]\)\]", p.stdout):
        means = [float(x) for x in m.group(1).split(",")]
        stds = [float(x) for x in m.group(2).split(",")]
        acc = {"gcond_internal_test_mean": round(means[-1], 4),
               "gcond_internal_test_std": round(stds[-1], 4)}

    os.makedirs(CONDENSED, exist_ok=True)
    for k, dst in dest.items():
        src = os.path.join(GCOND, "saved_repro", f"{k}_{tag}.pt")
        if not os.path.exists(src):
            raise SystemExit(f"expected artifact missing: {src}")
        shutil.copy2(src, dst)

    import torch
    A = torch.load(dest["adj"], map_location="cpu", weights_only=False)
    n = A.shape[0]
    flag = "" if n == expect_nodes else f"  !! expected {expect_nodes}"
    print(f"    done in {dur/60:.1f} min   synthetic nodes={n}{flag}")
    if acc:
        print(f"    (GCond's own internal eval: test="
              f"{acc['gcond_internal_test_mean']:.4f} "
              f"+-{acc['gcond_internal_test_std']:.4f})")
    return dest, dict(nodes=n, seconds=round(dur, 1), **(acc or {}))


def main():
    ap = argparse.ArgumentParser(description="Re-run GCond condensation")
    ap.add_argument("--only", default=None, help="restrict to one dataset")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    os.makedirs(CONDENSED, exist_ok=True)
    grid = [g for g in GRID if not args.only or g[0] == args.only]

    numbers, outputs = {}, []
    t0 = time.time()
    for dataset, r, paper_r, expect in grid:
        dest, info = condense(dataset, r, expect, force=args.force)
        outputs += [C.rel(p) for p in dest.values()]
        if info:
            numbers[f"{dataset}_r{r}"] = dict(paper_r=paper_r, **info)

    C.record(
        step="01_condense_gcond",
        summary=(f"Re-ran GCond condensation for {len(grid)} (dataset, r) pairs "
                 f"using the authors' run_main.sh hyperparameters"),
        inputs=["GCond/GCond-main/train_gcond_transduct.py",
                "GCond/GCond-main/scripts/run_main.sh (hyperparameters)"],
        outputs=outputs,
        numbers=numbers,
        notes=("Condensation config verbatim from the authors' scripts/run_main.sh: "
               "nlayers=2, sgc=1, lr_feat=1e-4, lr_adj=1e-4, seed=1, epochs=600. "
               "The paper gives only search ranges, never the chosen values. "
               "gcond_internal_* is GCond's own evaluation, recorded for "
               "reference only; the black box re-evaluates in step 02 with the "
               "paper's evaluation spec (2-layer GCN, 256 hidden, dropout 0, "
               "weight decay 5e-4, 600 epochs, 10 repeats). "
               "Authors' saved_ours/ bundle deliberately NOT used."))
    print(f"\ntotal {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
