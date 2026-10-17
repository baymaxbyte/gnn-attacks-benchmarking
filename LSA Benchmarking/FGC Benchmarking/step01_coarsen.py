"""Step 01 -- produce one FGC coarsening per (dataset, reduction ratio).

beta has no published value, so it is selected per setting from the grid the
authors' own notebook sweeps, by closeness to the paper's Table 4 accuracy. The
notebook hardcodes k = int(p*0.1), so the authors necessarily re-ran per ratio.


    python step01_coarsen.py
    python step01_coarsen.py --only cora --beta 0.001      # skip the search
"""
import argparse
import json
import os
import time

import common as K

os.environ.update(K.single_thread_env())

import numpy as np

import blackbox_fgc as B
import config as C
import fgc_solver as F

BETA_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)      # notebook driver 2's beta grid
GRID = [(ds, r) for ds in C.DATASETS for r in B.PAPER_R]
SELECT_REPEATS = 3                              # GNN re-inits per beta candidate


def _probe(job):
    """Coarsen at one (dataset, r, beta) and score Mode A against the paper."""
    ds, r, beta = job
    K.pin_single_thread()
    t0 = time.perf_counter()
    bb = B.FGCBlackBox(ds, r=r, beta=beta)
    cg = bb.coarsen()
    accs = []
    for s in range(SELECT_REPEATS):
        bb.build(run_seed=s)
        accs.append(bb.accuracy("orig") * 100)
    paper = B.PAPER_ACC[(ds, r)][0]
    mean = float(np.mean(accs))
    return {
        "dataset": ds, "r": r, "beta": beta, "k": bb.k,
        "mode_a_mean": mean, "mode_a_std": float(np.std(accs)),
        "paper_mean": paper, "abs_delta": abs(mean - paper),
        "nonempty_supernodes": cg["nonempty_supernodes"],
        "mean_nonempty_size": cg["mean_nonempty_size"],
        "max_supernode_size": cg["max_supernode_size"],
        "coarse_edges": cg["coarse_edges"],
        "seconds": round(time.perf_counter() - t0, 1),
    }


def _save(ds, r, beta):
    """Re-run the winning coarsening and persist it."""
    bb = B.FGCBlackBox(ds, r=r, beta=beta)
    cg = bb.coarsen()
    path = K.coarse_path(ds, r)
    np.savez_compressed(
        path,
        C=bb.C.numpy().astype(np.float32),
        assignment=cg["assignment"].numpy().astype(np.int32),
        X_c=cg["X_c"].numpy().astype(np.float32),
        Y_c=cg["Y_c"].numpy().astype(np.int64),
        edge_index=cg["edge_index"].numpy().astype(np.int64),
        beta=np.array([beta], dtype=np.float64),
        hp=np.array([bb.hp["lam"], bb.hp["alpha"], bb.hp["gamma"], beta], dtype=np.float64),
    )
    stats = {kk: vv for kk, vv in cg.items() if not hasattr(vv, "numpy")}
    return path, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="restrict to one dataset")
    ap.add_argument("--beta", type=float, default=None,
                    help="skip the search and use this beta everywhere")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--serial", action="store_true", help="one process, for debugging")
    args = ap.parse_args()

    K.ensure_dirs()
    grid = [(ds, r) for ds, r in GRID if args.only in (None, ds)]

    t0 = time.perf_counter()
    if args.beta is None:
        jobs = [(ds, r, b) for ds, r in grid for b in BETA_GRID]
        print(f"beta search: {len(jobs)} coarsenings x {SELECT_REPEATS} GNN runs, "
              f"{args.workers} workers\n")
        probes = K.parallel_map(_probe, jobs, workers=args.workers,
                               serial=args.serial)

        print(f"{'dataset':9s} {'r':>5s} {'beta':>7s} {'ModeA':>14s} {'paper':>7s} "
              f"{'|delta|':>8s} {'nonemp':>7s} {'meansz':>7s} {'c_edges':>8s} {'sec':>5s}")
        print("-" * 92)
        for p in probes:
            print(f"{p['dataset']:9s} {p['r']:>5} {p['beta']:>7g} "
                  f"{p['mode_a_mean']:6.2f}+/-{p['mode_a_std']:5.2f} {p['paper_mean']:7.2f} "
                  f"{p['abs_delta']:8.2f} {p['nonempty_supernodes']:7d} "
                  f"{p['mean_nonempty_size']:7.1f} {p['coarse_edges']:8d} {p['seconds']:5.1f}")

        best = {}
        for ds, r in grid:
            cands = [p for p in probes if p["dataset"] == ds and p["r"] == r]
            best[(ds, r)] = min(cands, key=lambda p: p["abs_delta"])
        with open(os.path.join(C.RESULTS, "step01_beta_search.json"), "w") as f:
            json.dump({"grid": list(BETA_GRID), "repeats": SELECT_REPEATS,
                       "probes": probes,
                       "selected": {f"{ds}_{r}": best[(ds, r)] for ds, r in grid}},
                      f, indent=2)
    else:
        best = {(ds, r): {"beta": args.beta} for ds, r in grid}
        probes = []

    print("\nselected beta and saved coarsenings:")
    outputs, stats_all = [], {}
    for ds, r in grid:
        beta = best[(ds, r)]["beta"]
        path, stats = _save(ds, r, beta)
        outputs.append(K.rel(path))
        stats_all[f"{ds}_{r}"] = dict(stats, beta=beta)
        d = best[(ds, r)].get("abs_delta")
        extra = f"  |delta vs paper| = {d:.2f}" if d is not None else ""
        print(f"  {ds:9s} r={r:<5} beta={beta:<7g} k={stats['k']:4d} "
              f"nonempty={stats['nonempty_supernodes']:4d} "
              f"mean_size={stats['mean_nonempty_size']:5.1f} "
              f"coarse_edges={stats['coarse_edges']:5d}{extra}")

    K.record(
        "step01_coarsen",
        summary=(f"FGC-with-sparsity coarsenings for {len(grid)} (dataset, r) "
                 f"settings; beta selected per setting from the notebook grid"),
        inputs=["data/dataset/original/ind.{cora,citeseer}.*",
                "kumar23a.pdf (Table 4 targets)",
                "Node_classification_FGC_with_sparsity.ipynb (solver, beta grid)"],
        outputs=outputs,
        numbers={"settings": len(grid), "beta_candidates": len(BETA_GRID) if probes else 0,
                 **{f"{kk}_beta": vv["beta"] for kk, vv in stats_all.items()}},
        notes=("lam=0.001 alpha=1 gamma=0.001 from the notebook driver that calls "
               "get_accuracy; the paper's Appendix B values are scale-incompatible "
               "with this solver's step. beta unpublished, selected per setting."))
    print(f"\ntotal {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
