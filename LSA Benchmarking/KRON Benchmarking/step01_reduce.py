"""Step 01 -- Kron-reduce Cora and Citeseer at each ratio.

For every (dataset, r) eliminates the lowest-degree interior nodes by Schur complement
and writes the reduced graph (kept ids, local edge_index/weight, the kept nodes' own
features and labels, and the original 60/20/20 split for evaluation). Asserts
max|L_red @ 1| ~ 0 as a correctness check that the reduction is a valid Laplacian.

    python step01_reduce.py
    python step01_reduce.py --only cora --ratios 0.5
"""
import argparse
import json
import os
import time

import common as K

os.environ.update(K.single_thread_env())

import numpy as np

import config as C
import data as D
import kron_reduce as KR

SPLIT_SCHEME = "paper"      # 60/20/20, same split as the UGC and GOREN benchmarks
SPLIT_SEED = 0
LAP_TOL = 1e-6             # max|L_red @ 1| tolerated before we call it not a Laplacian


def build(dataset, r):
    m = D.load(dataset)
    sp_ = D.make_split(m["labels"], m["n_classes"], seed=SPLIT_SEED,
                       scheme=SPLIT_SCHEME)
    rd = KR.kron_reduce(m["adj"], r, selection=C.SELECTION)

    if rd["laplacian_residual"] > LAP_TOL:
        raise SystemExit(f"{dataset} r={r}: reduced matrix is not a Laplacian "
                         f"(max|L_red @ 1|={rd['laplacian_residual']:.2e})")

    kept = rd["kept_idx"]
    X = np.asarray(m["features"].todense(), dtype=np.float32)
    labels = m["labels"]

    path = K.reduced_path(dataset, r)
    np.savez_compressed(
        path,
        kept_idx=kept.astype(np.int64),
        edge_index=rd["edge_index"].astype(np.int64),
        edge_weight=rd["edge_weight"].astype(np.float32),
        X_c=X[kept],
        Y_c=labels[kept].astype(np.int64),
        keep_train=sp_["train_mask"][kept],
        train_mask=sp_["train_mask"], val_mask=sp_["val_mask"],
        test_mask=sp_["test_mask"],
        r=np.array([r], dtype=np.float64))
    return path, rd, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--ratios", default=",".join(str(r) for r in C.RATIOS))
    args = ap.parse_args()

    K.ensure_dirs()
    ratios = [float(x) for x in args.ratios.split(",")]
    grid = [(ds, r) for ds in C.DATASETS for r in ratios
            if args.only in (None, ds)]

    print(f"Kron reduction, selection='{C.SELECTION}', {len(grid)} settings\n")
    print(f"{'dataset':9s} {'r':>5s} {'n':>6s} {'k':>6s} {'keep':>6s} {'r*':>7s} "
          f"{'comps':>6s} {'edges->':>8s} {'reduced':>8s} {'fill':>6s} "
          f"{'w_min':>8s} {'w_max':>7s} {'|L1|':>8s}")
    print("-" * 104)
    rows, outputs = [], []
    t0 = time.perf_counter()
    for ds, r in grid:
        path, rd, m = build(ds, r)
        outputs.append(K.rel(path))
        rows.append({"dataset": ds, "r": r, "target_k": C.n_boundary(m["n"], r),
                     **{k: v for k, v in rd.items()
                        if not isinstance(v, np.ndarray)}})
        print(f"{ds:9s} {r:>5} {rd['n']:>6d} {rd['k']:>6d} "
              f"{rd['keep_fraction']:>6.4f} {rd['achieved_ratio']:>7.4f} "
              f"{rd['n_components']:>6d} {rd['original_edges']:>8d} "
              f"{rd['reduced_edges']:>8d} {rd['fill_in_ratio']:>6.2f} "
              f"{rd['edge_weight_min']:>8.4g} {rd['edge_weight_max']:>7.3g} "
              f"{rd['laplacian_residual']:>8.1e}")

    with open(os.path.join(C.RESULTS, "step01_reduction.json"), "w") as f:
        json.dump({"selection": C.SELECTION,
                   "ratio_means": "fraction of nodes REMOVED",
                   "split": {"scheme": SPLIT_SCHEME, "seed": SPLIT_SEED},
                   "settings": rows}, f, indent=2, default=str)

    K.record(
        "step01_reduce",
        summary=(f"Kron-reduced {len(grid)} (dataset, r) settings by Schur "
                 f"complement, eliminating the lowest-degree interior nodes"),
        inputs=["attack/data/dataset/original/ind.{cora,citeseer}.*"],
        outputs=outputs + [K.rel(os.path.join(C.RESULTS, "step01_reduction.json"))],
        numbers={"settings": len(grid),
                 **{f"{x['dataset']}_{x['r']}_k": x["k"] for x in rows}},
        notes=("r is the fraction of nodes REMOVED; the boundary set keeps the "
               "highest-degree n*(1-r) nodes, one per connected component minimum. "
               "Kept nodes are ORIGINAL nodes with their own features and labels, "
               "so nothing is aggregated. Reduced edge weights are Schur-complement "
               "conductances; on these tree-like citation graphs the reduction is "
               "sparser than the original (leaf pruning dominates fill-in). "
               "max|L_red @ 1| ~ 1e-14 confirms the reduced matrix is a valid "
               "Laplacian, as the paper proves."))
    print(f"\ntotal {(time.perf_counter() - t0):.1f}s")


if __name__ == "__main__":
    main()
