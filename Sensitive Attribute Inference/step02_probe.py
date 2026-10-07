"""Step 02 -- train the boxes and run the two-world sensitive-attribute probe.

For every reduced cell and seed: train one box, then for each sensitive feature of
that dataset serve it with the feature column forced to 0 and to 1, and save
Z_u = [p0 || p1] (shape (n, 2*C)). Those Z are what the attack classifier consumes.
Reports node-classification accuracy as a box-quality sanity check.

    python step02_probe.py
    python step02_probe.py --only cora --methods gcond
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
    n_classes = D.load_whole(ds)["n_classes"]
    reduced_path = K.reduced_path(tag)

    t0 = time.time()
    model, rd = B.train_box(reduced_path, n_classes, seed=seed)
    post, labels = B.serve_true(model, ds, rd)
    box_acc = float((post.argmax(1) == labels).mean())

    saved = []
    for s_idx in C.SENSITIVE[ds]:
        p0, p1 = B.two_world_posteriors(model, ds, rd, s_idx)
        Z = np.concatenate([p0, p1], axis=1).astype(np.float32)   # (n, 2C)
        np.save(K.z_path(tag, s_idx, seed), Z)
        saved.append(s_idx)
    return {"tag": tag, "seed": seed, "box_acc": round(100 * box_acc, 2),
            "serve": rd["serve_mode"], "features": saved,
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--methods", default=None)
    ap.add_argument("--workers", type=int, default=6)
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
        ds = _tag_dataset(tag)
        for seed in C.TARGET_SEEDS:
            done = all(os.path.exists(K.z_path(tag, s, seed))
                       for s in C.SENSITIVE[ds])
            if done and not args.force:
                continue
            jobs.append((tag, seed))

    print(f"{len(tags)} cells, {len(jobs)} (cell,seed) probes to run "
          f"({len(C.TARGET_SEEDS)} seeds each, 3 features), {args.workers} workers\n")
    if not jobs:
        print("nothing to do (all cached)")
        return

    t0 = time.time()
    rows = K.parallel_map(_one, jobs, workers=args.workers, serial=args.serial)

    by_cell = {}
    for r in rows:
        by_cell.setdefault(r["tag"], []).append(r)
    print(f"{'cell':28s} {'serve':>10s} {'box acc':>8s}")
    print("-" * 50)
    for tag in sorted(by_cell):
        acc = np.mean([x["box_acc"] for x in by_cell[tag]])
        print(f"{tag:28s} {by_cell[tag][0]['serve']:>10s} {acc:>7.2f}%")

    path = os.path.join(C.RESULTS, "step02_boxes.json")
    prev = json.load(open(path)).get("boxes", []) if os.path.exists(path) else []
    seen = {(r["tag"], r["seed"]) for r in rows}
    merged = [b for b in prev if (b["tag"], b["seed"]) not in seen] + rows
    with open(path, "w") as f:
        json.dump({"target_seeds": list(C.TARGET_SEEDS), "boxes": merged}, f, indent=2)

    K.record("step02_probe",
             summary=(f"Trained {len(rows)} boxes and saved two-world Z = [p0||p1] "
                      f"for {len(by_cell)} cells x {len(C.TARGET_SEEDS)} seeds x 3 "
                      f"sensitive features"),
             inputs=["artifacts/reduced/*.npz", "cora.npz", "citeseer.npz"],
             outputs=["artifacts/Z/*.npy"],
             numbers={"boxes": len(rows), "cells": len(by_cell)},
             notes=("Two-world probe: sensitive column forced to 0 then 1, served on "
                    "the original graph; Z_u = [p0||p1] (n, 2C). box_acc is a "
                    "node-classification sanity check, not the attack metric."))
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
