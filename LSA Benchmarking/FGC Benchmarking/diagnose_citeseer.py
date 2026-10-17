"""Why does Citeseer r=0.3 collapse while r=0.1 and r=0.05 stay healthy?

Three candidate causes, separated by construction:

  A  adjacency mismatch -- does our theta equal the notebook's?
     The notebook uses to_dense_adj(Planetoid.edge_index). PyG's reader calls
     remove_self_loops then coalesce, so it should be binary, symmetric and
     self-loop free. Verified against our loader on the same raw files.

  B  step size -- the C update steps by gradient * k, so larger k means a larger
     step. If the collapse tracks k rather than the dataset, it is the step.
     Swept over matched k for both datasets.

  C  isolated nodes -- Citeseer has 48 nodes with no edges, giving all-zero rows
     in theta = D - A, so their C rows see no graph signal at all. Cora has none.
     Tested by dropping them before coarsening.

    python diagnose_citeseer.py
"""
import json
import os

import common as K

os.environ.update(K.single_thread_env())

import numpy as np
import scipy.sparse as sp
import torch

import blackbox_fgc as B
import config as C
import data as D
import fgc_solver as F

K_GRID = (166, 332, 500, 700, 812, 998, 1200)


# ------------------------------------------------------------------ test A
def adjacency_matches_pyg(dataset):
    """Rebuild the adjacency the way PyG's Planetoid reader does, and compare."""
    import pickle as pkl

    from torch_geometric.utils import coalesce, remove_self_loops, to_dense_adj
    with open(os.path.join(C.DATA, f"ind.{dataset}.graph"), "rb") as f:
        graph = pkl.load(f, encoding="latin1")

    rows, cols = [], []
    for key, value in graph.items():
        rows += [key] * len(value)
        cols += list(value)
    ei = torch.stack([torch.tensor(rows), torch.tensor(cols)], dim=0)
    n_raw = int(ei.max()) + 1
    ei, _ = remove_self_loops(ei)
    ei = coalesce(ei, num_nodes=n_raw, sort_by_row=False)

    m = D.load(dataset)
    n = m["n"]
    A_pyg = to_dense_adj(ei, max_num_nodes=n)[0].numpy()
    A_ours = np.asarray(m["adj"].todense(), dtype=np.float32)
    diff = int(np.abs(A_pyg - A_ours).sum())
    return {"dataset": dataset, "n": n, "differing_entries": diff,
            "pyg_edges": int(A_pyg.sum() // 2), "our_edges": m["n_edges"],
            "identical": diff == 0}


# ------------------------------------------------------------------ tests B, C
def _health(job):
    dataset, k, drop_isolated, beta = job
    K.pin_single_thread()
    m = D.load(dataset)
    adj, labels = m["adj"], m["labels"]
    keep = None
    if drop_isolated:
        deg = np.asarray(adj.sum(1)).ravel()
        keep = np.where(deg > 0)[0]
        adj = adj[keep][:, keep]
        labels = labels[keep]
    X = np.asarray(m["features"].todense(), dtype=np.float32)
    if keep is not None:
        X = X[keep]
    n = adj.shape[0]
    if k >= n:
        return None
    A = np.asarray(adj.todense(), dtype=np.float32)
    theta = np.diag(A.sum(1)) - A

    Cm, _, _ = F.coarsen(theta, X, k, beta=beta, lam=B.LAM, alpha=B.ALPHA,
                         gamma=B.GAMMA, step=B.STEP, variant=B.VARIANT)
    cg = F.coarse_graph(F.harden(Cm), theta, X, labels, m["n_classes"])
    sizes = np.bincount(cg["assignment"].numpy(), minlength=k)
    return {"dataset": dataset, "n": n, "k": k, "drop_isolated": drop_isolated,
            "beta": beta, "nonempty": cg["nonempty_supernodes"],
            "frac_used": cg["nonempty_supernodes"] / k,
            "max_size": int(sizes.max()),
            "max_frac_of_n": float(sizes.max() / n),
            "mean_size": cg["mean_nonempty_size"],
            "coarse_edges": cg["coarse_edges"]}


def main():
    torch.set_num_threads(4)
    K.ensure_dirs()
    report = {}

    print("=" * 96)
    print("A. does our adjacency equal PyG's Planetoid adjacency (what the notebook uses)?")
    print("=" * 96)
    rows = [adjacency_matches_pyg(ds) for ds in C.DATASETS]
    for x in rows:
        print(f"  {x['dataset']:9s} n={x['n']:5d}  our_edges={x['our_edges']:5d}  "
              f"pyg_edges={x['pyg_edges']:5d}  differing_entries={x['differing_entries']}  "
              f"-> {'IDENTICAL' if x['identical'] else 'MISMATCH'}")
    report["adjacency"] = rows

    print("\n" + "=" * 96)
    print("B. partition health vs k, matched across datasets (beta=0.001 throughout)")
    print("=" * 96)
    jobs = [(ds, k, False, 0.001) for ds in C.DATASETS for k in K_GRID]
    res = [r for r in K.parallel_map(_health, jobs, workers=8) if r]
    print(f"{'dataset':9s} {'n':>5s} {'k':>5s} {'nonempty':>9s} {'used':>6s} "
          f"{'maxsz':>6s} {'max/n':>7s} {'meansz':>7s} {'c_edges':>8s}")
    print("-" * 96)
    for x in sorted(res, key=lambda z: (z["dataset"], z["k"])):
        flag = "   <-- COLLAPSED" if x["max_frac_of_n"] > 0.1 else ""
        print(f"{x['dataset']:9s} {x['n']:>5d} {x['k']:>5d} {x['nonempty']:>9d} "
              f"{100*x['frac_used']:>5.0f}% {x['max_size']:>6d} "
              f"{100*x['max_frac_of_n']:>6.1f}% {x['mean_size']:>7.1f} "
              f"{x['coarse_edges']:>8d}{flag}")
    report["k_sweep"] = res

    print("\n" + "=" * 96)
    print("C. Citeseer with its 48 isolated nodes dropped before coarsening")
    print("=" * 96)
    jobs = [("citeseer", k, True, 0.001) for k in (332, 812, 983, 998)]
    res2 = [r for r in K.parallel_map(_health, jobs, workers=8) if r]
    base = {x["k"]: x for x in res if x["dataset"] == "citeseer"}
    print(f"{'k':>5s} {'n':>5s} | {'nonempty':>9s} {'used':>6s} {'maxsz':>6s} {'max/n':>7s} | "
          f"{'baseline maxsz':>15s} {'baseline max/n':>15s}")
    print("-" * 96)
    for x in sorted(res2, key=lambda z: z["k"]):
        b = base.get(x["k"])
        bm = f"{b['max_size']:>15d}" if b else f"{'-':>15s}"
        bf = f"{100*b['max_frac_of_n']:>14.1f}%" if b else f"{'-':>15s}"
        print(f"{x['k']:>5d} {x['n']:>5d} | {x['nonempty']:>9d} {100*x['frac_used']:>5.0f}% "
              f"{x['max_size']:>6d} {100*x['max_frac_of_n']:>6.1f}% | {bm} {bf}")
    report["drop_isolated"] = res2

    path = os.path.join(C.RESULTS, "diagnose_citeseer.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {K.rel(path)}")


if __name__ == "__main__":
    main()
