"""Diagnostics for the coarsening itself, independent of the GNN.

Written to explain why Cora r=0.05 misses kumar23a Table 4 by ~14 points while
r=0.3 and r=0.1 reproduce. Answers four questions:

  1. purity      -- how much label information survives majority-vote coarsening
  2. coverage    -- do any classes disappear from the coarse label vector Y_c
  3. seed        -- is the shortfall an artifact of the notebook's random_state=1
  4. iterations  -- does the notebook's 50-iteration budget converge at small k

    python diagnose_coarsening.py
    python diagnose_coarsening.py --only cora --r 0.05
"""
import argparse
import json
import os
import time

import common as K

os.environ.update(K.single_thread_env())

import numpy as np
import torch

import blackbox_fgc as B
import config as C
import data as D
import fgc_solver as F

SEEDS = (1, 2, 3, 7, 42)
ITER_GRID = (25, 50, 100, 200)
REPEATS = 5


def purity(assignment, labels, k):
    """Accuracy of 'every node takes its supernode's majority label'.

    This is the ceiling for MODE B only: a lifted answer is constant within a
    supernode, so it cannot beat the best constant label per supernode. Mode A is
    not bounded by it, because it runs the GCN over the original graph and so
    generalises per node beyond anything the partition encodes.
    """
    assignment = np.asarray(assignment)
    labels = np.asarray(labels)
    n_classes = int(labels.max()) + 1
    pred = np.zeros_like(labels)
    for s in range(k):
        members = np.where(assignment == s)[0]
        if len(members) == 0:
            continue
        pred[members] = np.bincount(labels[members], minlength=n_classes).argmax()
    return float((pred == labels).mean()), pred


def coarse_label_hist(Y_c, labels, n_classes):
    """Class histogram of Y_c against the original class distribution."""
    yc = np.bincount(np.asarray(Y_c), minlength=n_classes)
    y = np.bincount(np.asarray(labels), minlength=n_classes)
    return y, yc


def random_partition_purity(labels, k, n, seed=0, trials=5):
    """Purity of a uniformly random partition into k groups -- the floor."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(trials):
        a = rng.randint(0, k, size=n)
        out.append(purity(a, labels, k)[0])
    return float(np.mean(out))


def _mode_a(bb, repeats=REPEATS):
    accs = [bb.build(run_seed=s).accuracy("orig") * 100 for s in range(repeats)]
    return float(np.mean(accs)), float(np.std(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--r", type=float, default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    K.ensure_dirs()
    report = {}

    settings = [(ds, r) for ds in C.DATASETS for r in B.PAPER_R
                if args.only in (None, ds) and args.r in (None, r)]

    # ---------------------------------------------------------------- 1 + 2
    print("=" * 100)
    print("PARTITION PURITY  --  ceiling for Mode B; Mode A is NOT bounded by it")
    print("=" * 100)
    print(f"{'box':16s} {'k':>5s} {'meansz':>7s} {'purity':>8s} {'random':>8s} "
          f"{'vs rand':>8s} | {'ModeB':>7s} {'of ceil':>8s} | {'ModeA':>7s}")
    print("-" * 100)
    purity_rows = []
    v = json.load(open(os.path.join(C.RESULTS, "step02_validation.json")))
    for ds, r in settings:
        m = D.load(ds)
        z = np.load(K.coarse_path(ds, r))
        k = int(m["n"] * r)
        pur, _ = purity(z["assignment"], m["labels"], k)
        rnd = random_partition_purity(m["labels"], k, m["n"])
        box = next(b for b in v["boxes"] if b["dataset"] == ds and b["r"] == r)
        mode_a, mode_b = box["orig"]["mean"], box["lift"]["mean"]
        sizes = np.bincount(z["assignment"], minlength=k)
        print(f"{ds}_{r:<10} {k:>5d} {sizes[sizes>0].mean():>7.1f} {100*pur:>7.2f}% "
              f"{100*rnd:>7.2f}% {100*(pur-rnd):>+7.2f} | {mode_b:>7.2f} "
              f"{100*mode_b/(100*pur):>7.1f}% | {mode_a:>7.2f}")
        purity_rows.append({"dataset": ds, "r": r, "k": k, "purity": 100*pur,
                            "random_purity": 100*rnd, "mode_a": mode_a,
                            "mode_b": mode_b,
                            "mode_b_frac_of_ceiling": mode_b / (100*pur)})
    report["purity"] = purity_rows

    print("\n" + "=" * 100)
    print("CLASS COVERAGE  --  original class sizes vs how many supernodes carry each class")
    print("=" * 100)
    cov_rows = []
    for ds, r in settings:
        m = D.load(ds)
        z = np.load(K.coarse_path(ds, r))
        y, yc = coarse_label_hist(z["Y_c"], m["labels"], m["n_classes"])
        used = np.bincount(z["assignment"], minlength=int(m["n"]*r))
        yc_used = np.bincount(np.asarray(z["Y_c"])[used > 0], minlength=m["n_classes"])
        missing = [int(c) for c in range(m["n_classes"]) if yc_used[c] == 0]
        print(f"  {ds}_{r}  k={int(m['n']*r)}")
        print(f"    original : {list(y)}")
        print(f"    Y_c (used supernodes only): {list(yc_used)}"
              + (f"   MISSING CLASSES {missing}" if missing else "   all classes present"))
        lost = int(sum(y[c] for c in missing))
        if missing:
            print(f"    -> {lost} of {m['n']} nodes ({100*lost/m['n']:.1f}%) belong to a "
                  f"class no supernode carries; unreachable ceiling loss")
        cov_rows.append({"dataset": ds, "r": r, "original": [int(q) for q in y],
                         "Y_c": [int(q) for q in yc_used], "missing_classes": missing,
                         "unreachable_nodes": lost})
    report["coverage"] = cov_rows

    # ---------------------------------------------------------------- 3
    if args.only or args.r:
        targets = settings
    else:
        targets = [("cora", 0.05), ("cora", 0.1)]
    print("\n" + "=" * 100)
    print("INITIALISATION SEED SENSITIVITY  (notebook hardcodes random_state=1)")
    print("=" * 100)
    print(f"{'box':16s} {'seed':>5s} {'purity':>8s} {'nonempty':>9s} {'maxsz':>6s} "
          f"{'c_edges':>8s} {'ModeA':>14s}")
    print("-" * 100)
    seed_rows = []
    for ds, r in targets:
        m = D.load(ds); theta = D.laplacian(m["adj"])
        X = np.asarray(m["features"].todense(), dtype=np.float32)
        k = int(m["n"] * r)
        beta = float(np.load(K.coarse_path(ds, r))["beta"][0])
        for sd in SEEDS:
            Cm, _, _ = F.coarsen(theta, X, k, seed=sd, beta=beta,
                                 lam=B.LAM, alpha=B.ALPHA, gamma=B.GAMMA, step=B.STEP)
            cg = F.coarse_graph(F.harden(Cm), theta, X, m["labels"], m["n_classes"])
            pur, _ = purity(cg["assignment"].numpy(), m["labels"], k)
            bb = B.FGCBlackBox(ds, r=r, beta=beta); bb.coarsen(C=Cm.numpy())
            mu, sg = _mode_a(bb)
            print(f"{ds}_{r:<10} {sd:>5d} {100*pur:>7.2f}% {cg['nonempty_supernodes']:>9d} "
                  f"{cg['max_supernode_size']:>6d} {cg['coarse_edges']:>8d} "
                  f"{mu:>6.2f}+/-{sg:>5.2f}")
            seed_rows.append({"dataset": ds, "r": r, "seed": sd, "purity": 100*pur,
                              "nonempty": cg["nonempty_supernodes"],
                              "max_size": cg["max_supernode_size"],
                              "coarse_edges": cg["coarse_edges"],
                              "mode_a_mean": mu, "mode_a_std": sg})
    report["seeds"] = seed_rows

    # ---------------------------------------------------------------- 4
    print("\n" + "=" * 100)
    print("ITERATION BUDGET  (notebook uses 50)")
    print("=" * 100)
    print(f"{'box':16s} {'iters':>6s} {'purity':>8s} {'nonempty':>9s} {'maxsz':>6s} "
          f"{'c_edges':>8s} {'ModeA':>14s} {'sec':>6s}")
    print("-" * 100)
    iter_rows = []
    for ds, r in targets:
        m = D.load(ds); theta = D.laplacian(m["adj"])
        X = np.asarray(m["features"].todense(), dtype=np.float32)
        k = int(m["n"] * r)
        beta = float(np.load(K.coarse_path(ds, r))["beta"][0])
        for it in ITER_GRID:
            t0 = time.perf_counter()
            Cm, _, _ = F.coarsen(theta, X, k, iters=it, beta=beta,
                                 lam=B.LAM, alpha=B.ALPHA, gamma=B.GAMMA, step=B.STEP)
            cg = F.coarse_graph(F.harden(Cm), theta, X, m["labels"], m["n_classes"])
            pur, _ = purity(cg["assignment"].numpy(), m["labels"], k)
            bb = B.FGCBlackBox(ds, r=r, beta=beta); bb.coarsen(C=Cm.numpy())
            mu, sg = _mode_a(bb)
            dt = time.perf_counter() - t0
            print(f"{ds}_{r:<10} {it:>6d} {100*pur:>7.2f}% {cg['nonempty_supernodes']:>9d} "
                  f"{cg['max_supernode_size']:>6d} {cg['coarse_edges']:>8d} "
                  f"{mu:>6.2f}+/-{sg:>5.2f} {dt:>6.1f}")
            iter_rows.append({"dataset": ds, "r": r, "iters": it, "purity": 100*pur,
                              "nonempty": cg["nonempty_supernodes"],
                              "max_size": cg["max_supernode_size"],
                              "coarse_edges": cg["coarse_edges"],
                              "mode_a_mean": mu, "mode_a_std": sg})
    report["iterations"] = iter_rows

    path = os.path.join(C.RESULTS, "diagnose_coarsening.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {K.rel(path)}")


if __name__ == "__main__":
    main()
