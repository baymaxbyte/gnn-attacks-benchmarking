"""Step 01 -- baseline coarsenings, the coarse graph GOREN relearns weights on.

r is the fraction of nodes REMOVED.

    python step01_coarsen.py
    python step01_coarsen.py --only cora --ratios 0.5
"""
import argparse
import json
import os
import subprocess
import time

import common as K

os.environ.update(K.single_thread_env())

import numpy as np
import scipy.sparse as sp

import config as C
import data as D
import goren_coarsen as G

SPLIT_SCHEME = "paper"      # 60/20/20, same split as the UGC benchmark
SPLIT_SEED = 0


def run_worker(dataset, r, force=False):
    """Produce C via the GOREN checkout's interpreter. Returns (C, worker stats)."""
    out = os.path.join(C.COARSENED, f"C_{dataset}_r{r}.npz")
    if os.path.exists(out) and not force:
        return sp.load_npz(out), {"cached": True}
    cmd = [C.COARSEN_PYTHON, os.path.join(C.HERE, "_coarsen_worker.py"),
           "--dataset", dataset, "--r", str(r), "--out", out,
           "--datapath", C.DATA, "--goren_home", C.GOREN_HOME,
           "--method", C.METHOD, "--K", str(C.K_EIGS)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-1500:]); print(p.stderr[-1500:])
        raise SystemExit(f"coarsening worker failed for {dataset} r={r}")
    stats = {"cached": False}
    for line in p.stdout.splitlines():
        if line.startswith("COARSEN_OK"):
            for tok in line.split()[1:]:
                k, v = tok.split("=", 1)
                stats[k] = v
    return sp.load_npz(out), stats


def build(dataset, r, force=False):
    Cmat, wstats = run_worker(dataset, r, force=force)
    m = D.load(dataset)
    sp_ = D.make_split(m["labels"], m["n_classes"], seed=SPLIT_SEED,
                       scheme=SPLIT_SCHEME)
    cg = G.coarse_graph(Cmat, m["adj"], m["features"], m["labels"],
                        m["n_classes"], sp_["train_mask"])

    path = K.coarse_path(dataset, r)
    np.savez_compressed(
        path,
        assignment=cg["assignment"].astype(np.int32),
        X_c=cg["X_c"].astype(np.float32),
        Y_c=cg["Y_c"].astype(np.int64),
        edge_index=cg["edge_index"].astype(np.int64),
        edge_weight=cg["edge_weight"].astype(np.float32),
        keep=cg["keep"], sizes=cg["sizes"].astype(np.int32),
        r=np.array([r], dtype=np.float64),
        train_mask=sp_["train_mask"], val_mask=sp_["val_mask"],
        test_mask=sp_["test_mask"])
    return path, cg, wstats, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--ratios", default=",".join(str(r) for r in C.RATIOS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    K.ensure_dirs()
    os.makedirs(C.POSTERIORS, exist_ok=True)
    ratios = [float(x) for x in args.ratios.split(",")]
    grid = [(ds, r) for ds in C.DATASETS for r in ratios
            if args.only in (None, ds)]

    print(f"Loukas '{C.METHOD}' coarsening, K={C.K_EIGS}, "
          f"{len(grid)} settings\ncoarsening interpreter: {C.COARSEN_PYTHON}\n")
    t0 = time.perf_counter()
    rows, outputs = [], []
    print(f"{'dataset':9s} {'r':>5s} {'n':>6s} {'k':>6s} {'ratio*':>7s} {'used':>6s} "
          f"{'train.':>7s} {'maxsz':>6s} {'meansz':>7s} {'c_edges':>8s} {'orthon':>7s} {'sec':>6s}")
    print("-" * 104)
    for ds, r in grid:
        path, cg, w, m = build(ds, r, force=args.force)
        outputs.append(K.rel(path))
        rows.append({"dataset": ds, "r": r, "n": m["n"],
                     "target_k": C.n_supernodes(m["n"], r),
                     **{kk: vv for kk, vv in cg.items()
                        if not isinstance(vv, (np.ndarray, sp.spmatrix))},
                     "worker": w})
        print(f"{ds:9s} {r:>5} {m['n']:>6d} {cg['k']:>6d} "
              f"{cg['achieved_ratio']:>7.4f} "
              f"{cg['k'] - cg['empty_supernodes']:>6d} "
              f"{cg['trainable_supernodes']:>7d} {cg['max_supernode_size']:>6d} "
              f"{cg['mean_supernode_size']:>7.2f} {cg['coarse_edges']:>8d} "
              f"{str(w.get('orthonormal', '-')):>7s} {w.get('seconds', '-'):>6s}")

    with open(os.path.join(C.RESULTS, "step01_coarsening.json"), "w") as f:
        json.dump({"method": C.METHOD, "K": C.K_EIGS,
                   "ratio_means": "fraction of nodes REMOVED",
                   "split": {"scheme": SPLIT_SCHEME, "seed": SPLIT_SEED},
                   "settings": rows}, f, indent=2, default=str)

    K.record(
        "step01_coarsen",
        summary=(f"Loukas {C.METHOD} coarsenings for {len(grid)} (dataset, r) "
                 f"settings, supplying the coarse topology GOREN relearns weights on"),
        inputs=["attack/data/dataset/original/ind.{cora,citeseer}.*",
                f"{C.METHOD} via graph_coarsening (GOREN repo)"],
        outputs=outputs + [K.rel(os.path.join(C.RESULTS, "step01_coarsening.json"))],
        numbers={"settings": len(grid),
                 **{f"{x['dataset']}_{x['r']}_k": x["k"] for x in rows}},
        notes=("r is the fraction of nodes REMOVED. Baseline edge weights are "
               "A_c = C A C^T; GOREN replaces these in step 02. Cora and Citeseer "
               "are disconnected, so the variation cost divides by (nc-1) and emits "
               "divide-by-zero warnings on singleton subgraphs; the coarsening is "
               "unaffected and hits its target ratio."))
    print(f"\ntotal {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
