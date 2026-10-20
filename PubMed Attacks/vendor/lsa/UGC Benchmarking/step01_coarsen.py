"""Step 01 -- produce one UGC coarsening per (dataset, ratio).

UGC is LSH-based, so the supernode count is not set directly: it emerges from a bin
width. The authors ship BIN_WIDTH_DICTONARY with only ratio 50, and only for
datasets other than Citeseer, so five of our six settings need calibrating with the
search from BinWidthFinder.py.

Cora at ratio 50 is the yardstick: the authors shipped 0.004 for it, so we calibrate
it too and report both, which tells us whether the calibration is trustworthy for
the settings they did not ship.

ratio is the percentage of nodes REMOVED (ratio = 1 - k/n).

    python step01_coarsen.py
    python step01_coarsen.py --only cora --ratios 50
"""
import argparse
import json
import os
import time

import common as K

os.environ.update(K.single_thread_env())

import numpy as np
import torch

import blackbox_ugc as B
import config as C
import data as D
import ugc_solver as S


def _probe(job):
    """Calibrate a bin width and build the coarsening for one (dataset, ratio)."""
    dataset, ratio = job
    K.pin_single_thread()
    t0 = time.perf_counter()

    bb = B.UGCBlackBox(dataset, ratio=ratio)
    cal = S.find_bin_width(bb.X_aug, ratio / 100.0, seed=bb.seed)
    cg = bb.coarsen(bin_width=cal["bin_width"])

    shipped = C.SHIPPED_BIN_WIDTH.get((dataset, ratio))
    shipped_stats = None
    if shipped is not None:
        bb2 = B.UGCBlackBox(dataset, ratio=ratio)
        cg2 = bb2.coarsen(bin_width=shipped)
        shipped_stats = {"bin_width": shipped, "k": cg2["k"],
                         "achieved_ratio": cg2["achieved_ratio"]}

    return {"dataset": dataset, "ratio": ratio, "alpha": bb.alpha,
            "target_k": bb.k, "bin_width": cal["bin_width"],
            "achieved_ratio": cg["achieved_ratio"], "calibration_error": cal["error"],
            "calibration_converged": cal["converged"],
            "calibration_iters": cal["iters"],
            "shipped": shipped_stats,
            "k": cg["k"], "trainable_supernodes": cg["trainable_supernodes"],
            "max_supernode_size": cg["max_supernode_size"],
            "mean_supernode_size": cg["mean_supernode_size"],
            "singleton_supernodes": cg["singleton_supernodes"],
            "coarse_edges": cg["coarse_edges"],
            "augmented_dim": int(bb.X_aug.shape[1]),
            "seconds": round(time.perf_counter() - t0, 1)}


def _save(dataset, ratio, bin_width):
    bb = B.UGCBlackBox(dataset, ratio=ratio)
    cg = bb.coarsen(bin_width=bin_width)
    path = K.coarse_path(dataset, ratio)
    np.savez_compressed(
        path,
        assignment=cg["assignment"].numpy().astype(np.int32),
        X_c=cg["X_c"].numpy().astype(np.float32),
        Y_c=cg["Y_c"].numpy().astype(np.int64),
        edge_index=cg["A_c_edge_index"].numpy().astype(np.int64),
        edge_attr=cg["A_c_edge_attr"].numpy().astype(np.float32),
        keep=cg["keep"].numpy(),
        sizes=cg["sizes"].numpy().astype(np.int32),
        bin_width=np.array([bin_width], dtype=np.float64),
        alpha=np.array([bb.alpha], dtype=np.float64),
        train_mask=bb.split["train_mask"], val_mask=bb.split["val_mask"],
        test_mask=bb.split["test_mask"])
    return path, cg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--ratios", default=",".join(str(r) for r in C.RATIOS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--serial", action="store_true")
    args = ap.parse_args()

    K.ensure_dirs()
    os.makedirs(C.POSTERIORS, exist_ok=True)
    ratios = [int(r) for r in args.ratios.split(",")]
    grid = [(ds, r) for ds in C.DATASETS for r in ratios
            if args.only in (None, ds)]

    print(f"calibrating + coarsening {len(grid)} settings, alpha fixed per dataset "
          f"(cora {C.ALPHA['cora']}, citeseer {C.ALPHA['citeseer']})\n")
    t0 = time.perf_counter()
    probes = K.parallel_map(_probe, grid, workers=args.workers, serial=args.serial)

    print(f"{'dataset':9s} {'ratio':>5s} {'alpha':>6s} {'bw':>10s} {'target k':>8s} "
          f"{'k':>6s} {'ratio*':>7s} {'trainable':>9s} {'maxsz':>6s} {'c_edges':>8s} {'sec':>6s}")
    print("-" * 104)
    for p in probes:
        print(f"{p['dataset']:9s} {p['ratio']:>5d} {p['alpha']:>6.2f} "
              f"{p['bin_width']:>10.6f} {p['target_k']:>8d} {p['k']:>6d} "
              f"{p['achieved_ratio']:>7.4f} {p['trainable_supernodes']:>9d} "
              f"{p['max_supernode_size']:>6d} {p['coarse_edges']:>8d} {p['seconds']:>6.1f}")

    ship = [p for p in probes if p["shipped"]]
    if ship:
        print("\ncalibration check against the authors' shipped bin width:")
        for p in ship:
            s = p["shipped"]
            print(f"  {p['dataset']} ratio {p['ratio']}: "
                  f"ours bw={p['bin_width']:.6f} -> k={p['k']}, ratio={p['achieved_ratio']:.4f}   |   "
                  f"shipped bw={s['bin_width']:.6f} -> k={s['k']}, ratio={s['achieved_ratio']:.4f}")

    print("\nsaving coarsenings:")
    outputs = []
    for p in probes:
        path, cg = _save(p["dataset"], p["ratio"], p["bin_width"])
        outputs.append(K.rel(path))
        print(f"  {K.rel(path)}  k={cg['k']} trainable={cg['trainable_supernodes']}")

    with open(os.path.join(C.RESULTS, "step01_calibration.json"), "w") as f:
        json.dump({"alpha": C.ALPHA, "ratio_means": "percent of nodes REMOVED",
                   "probes": probes}, f, indent=2)

    K.record(
        "step01_coarsen",
        summary=(f"UGC LSH coarsenings for {len(grid)} (dataset, ratio) settings; "
                 f"bin widths calibrated, alpha fixed per dataset"),
        inputs=["attack/data/dataset/original/ind.{cora,citeseer}.*",
                "UGC paper Table 6 (alpha), Table 9 (accuracy targets)"],
        outputs=outputs + [K.rel(os.path.join(C.RESULTS, "step01_calibration.json"))],
        numbers={"settings": len(grid),
                 **{f"{p['dataset']}_{p['ratio']}_k": p["k"] for p in probes},
                 **{f"{p['dataset']}_{p['ratio']}_bin_width": round(p["bin_width"], 6)
                    for p in probes}},
        notes=("ratio is the percent of nodes REMOVED. The authors shipped bin "
               "widths only for ratio 50 and not for citeseer, so five of six "
               "settings are calibrated here with the BinWidthFinder search. That "
               "search cannot satisfy its own precision test and would loop "
               "forever, so we cap the iterations and keep the closest bin width."))
    print(f"\ntotal {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
