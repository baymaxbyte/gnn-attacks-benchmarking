"""Compare the notebook-exact block update against the BSUM/Gauss-Seidel reading.

The notebook's `update` recomputes X_tilde from the OLD C and thresholds/normalises
C only afterwards. A textbook alternating minimisation would use the NEW C. The two
differ; this measures by how much, per setting.

    python compare_variants.py
"""
import json
import os

import common as K

os.environ.update(K.single_thread_env())

import numpy as np
import torch

import blackbox_fgc as B
import config as C
import data as D
import fgc_solver as F
from diagnose_coarsening import purity

REPEATS = 5


def run(ds, r, beta, variant):
    m = D.load(ds)
    theta = D.laplacian(m["adj"])
    X = np.asarray(m["features"].todense(), dtype=np.float32)
    k = int(m["n"] * r)
    Cm, _, _ = F.coarsen(theta, X, k, beta=beta, lam=B.LAM, alpha=B.ALPHA,
                         gamma=B.GAMMA, step=B.STEP, variant=variant)
    cg = F.coarse_graph(F.harden(Cm), theta, X, m["labels"], m["n_classes"])
    pur, _ = purity(cg["assignment"].numpy(), m["labels"], k)
    bb = B.FGCBlackBox(ds, r=r, beta=beta)
    bb.coarsen(C=Cm.numpy())
    accs = [bb.build(run_seed=s).accuracy("orig") * 100 for s in range(REPEATS)]
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
            "purity": 100 * pur, "nonempty": cg["nonempty_supernodes"],
            "max_size": cg["max_supernode_size"], "coarse_edges": cg["coarse_edges"]}


def main():
    torch.set_num_threads(4)
    K.ensure_dirs()
    v = json.load(open(os.path.join(C.RESULTS, "step02_validation.json")))
    betas = {(b["dataset"], b["r"]): b["beta"] for b in v["boxes"] if b["r"] is not None}

    print("notebook = X_tilde from OLD C (literal code)   "
          "paper = X_tilde from NEW normalised C (BSUM)\n")
    print(f"{'box':16s} {'paper':>7s} | {'notebook':>14s} {'paper-var':>14s} "
          f"{'diff':>7s} | {'nb pur':>7s} {'pp pur':>7s} | {'nb nonemp':>9s} {'pp nonemp':>9s}")
    print("-" * 104)
    rows = []
    for ds in C.DATASETS:
        for r in B.PAPER_R:
            beta = betas[(ds, r)]
            nb = run(ds, r, beta, "notebook")
            pp = run(ds, r, beta, "paper")
            target = B.PAPER_ACC[(ds, r)][0]
            print(f"{ds}_{r:<10} {target:>7.2f} | {nb['mean']:6.2f}+/-{nb['std']:5.2f} "
                  f"{pp['mean']:6.2f}+/-{pp['std']:5.2f} {nb['mean']-pp['mean']:+7.2f} | "
                  f"{nb['purity']:6.2f}% {pp['purity']:6.2f}% | "
                  f"{nb['nonempty']:>9d} {pp['nonempty']:>9d}")
            rows.append({"dataset": ds, "r": r, "beta": beta, "paper_target": target,
                         "notebook": nb, "paper_variant": pp})

    print("\ndelta vs kumar23a Table 4")
    print(f"{'box':16s} {'notebook':>10s} {'paper-var':>10s}")
    print("-" * 40)
    for x in rows:
        print(f"{x['dataset']}_{x['r']:<10} "
              f"{x['notebook']['mean']-x['paper_target']:>+10.2f} "
              f"{x['paper_variant']['mean']-x['paper_target']:>+10.2f}")

    path = os.path.join(C.RESULTS, "compare_variants.json")
    with open(path, "w") as f:
        json.dump({"repeats": REPEATS, "rows": rows}, f, indent=2)
    print(f"\nwrote {K.rel(path)}")


if __name__ == "__main__":
    main()
