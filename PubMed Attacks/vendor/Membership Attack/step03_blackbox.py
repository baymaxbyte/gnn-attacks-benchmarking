"""Step 03 -- train the black boxes and serve posteriors on the original graph.

For every reduced cell from step02, train TARGET_SEEDS boxes and serve each on the
full original graph, saving posteriors of shape (n, n_classes). Those posteriors are
what the membership attack queries; members are the V_in rows, non-members the V_out
rows. Reports the member-vs-non-member accuracy gap as a leakage sanity signal.

    python step03_blackbox.py
    python step03_blackbox.py --only cora --methods gcond
"""
import argparse
import glob
import json
import os
import time

import common as K

os.environ.update(K.single_thread_env())

import numpy as np

import blackbox as B
import config as C
import data as D


def _tag_dataset(tag):
    return tag.split("_", 1)[0]


def _one(job):
    tag, seed = job
    K.pin_single_thread()
    ds = _tag_dataset(tag)
    sp_ = D.load_split(ds)
    member_ids = sp_["member_ids"]
    nonmember_ids = sp_["nonmember_ids"]
    m = D.load_whole(ds)
    labels, n_classes = m["labels"], m["n_classes"]

    reduced_path = os.path.join(C.REDUCED, f"{tag}.npz")
    t0 = time.time()
    post, info = B.train_and_serve(reduced_path, ds, member_ids, n_classes,
                                   seed=seed)
    out = os.path.join(C.POSTERIORS, f"{tag}_seed{seed}.npy")
    np.save(out, post.astype(np.float32))

    mem_acc = float((post[member_ids].argmax(1) == labels[member_ids]).mean())
    non_acc = float((post[nonmember_ids].argmax(1) == labels[nonmember_ids]).mean())
    return {"tag": tag, "seed": seed, "member_acc": round(100 * mem_acc, 2),
            "nonmember_acc": round(100 * non_acc, 2),
            "acc_gap": round(100 * (mem_acc - non_acc), 2),
            "k_train": info["k_train"], "serve": info["serve_mode"],
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--methods", default=None, help="comma list to restrict")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--serial", action="store_true")
    args = ap.parse_args()

    K.ensure_dirs()
    tags = sorted(os.path.basename(p)[:-4]
                  for p in glob.glob(os.path.join(C.REDUCED, "*.npz")))
    if args.only:
        tags = [t for t in tags if _tag_dataset(t) == args.only]
    if args.methods:
        want = args.methods.split(",")
        tags = [t for t in tags if t.split("_")[1] in want]

    jobs = []
    for tag in tags:
        for seed in C.TARGET_SEEDS:
            out = os.path.join(C.POSTERIORS, f"{tag}_seed{seed}.npy")
            if os.path.exists(out) and not args.force:
                continue
            jobs.append((tag, seed))

    print(f"{len(tags)} cells, {len(jobs)} box trainings to run "
          f"({len(C.TARGET_SEEDS)} seeds each), {args.workers} workers\n")
    if not jobs:
        print("nothing to do (all cached)")
        return

    t0 = time.time()
    rows = K.parallel_map(_one, jobs, workers=args.workers, serial=args.serial)

    # aggregate per cell
    by_cell = {}
    for r in rows:
        by_cell.setdefault(r["tag"], []).append(r)
    print(f"{'cell':28s} {'k_tr':>5s} {'serve':>10s} {'mem acc':>8s} "
          f"{'non acc':>8s} {'gap':>7s}")
    print("-" * 76)
    for tag in sorted(by_cell):
        rs = by_cell[tag]
        ma = np.mean([x["member_acc"] for x in rs])
        na = np.mean([x["nonmember_acc"] for x in rs])
        print(f"{tag:28s} {rs[0]['k_train']:>5d} {rs[0]['serve']:>10s} "
              f"{ma:>7.2f}% {na:>7.2f}% {ma - na:>+6.2f}")

    path = os.path.join(C.RESULTS, "step03_boxes.json")
    prev = []
    if os.path.exists(path):
        prev = json.load(open(path)).get("boxes", [])
    seen = {(r["tag"], r["seed"]) for r in rows}
    merged = [b for b in prev if (b["tag"], b["seed"]) not in seen] + rows
    with open(path, "w") as f:
        json.dump({"target_seeds": list(C.TARGET_SEEDS), "boxes": merged}, f,
                  indent=2)

    K.record(
        "step03_blackbox",
        summary=(f"Trained {len(rows)} boxes ({len(by_cell)} cells x "
                 f"{len(C.TARGET_SEEDS)} seeds) and served posteriors on the "
                 f"original graph"),
        inputs=["artifacts/reduced/*.npz", "artifacts/splits/*_split.npz"],
        outputs=[K.rel(os.path.join(C.POSTERIORS, f"{r['tag']}_seed{r['seed']}.npy"))
                 for r in rows],
        numbers={"boxes": len(rows), "cells": len(by_cell)},
        notes=("Uniform 3-layer GCN across all methods, trained on the reduced graph "
               "and served on the original graph. member_acc/nonmember_acc gap is a "
               "leakage sanity signal (members typically higher)."))
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
