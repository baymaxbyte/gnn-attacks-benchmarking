"""Step 02a -- train GOREN's GIN and produce the learned coarse edge weights.

Reports the paper's own success metric, the relative Eigenerror reduction, so the
weight learner can be validated without involving any classifier.

    python step02a_goren_weights.py
    python step02a_goren_weights.py --only cora --epochs 400
"""
import argparse
import json
import os

import common as K

os.environ.update(K.single_thread_env())

import time

import numpy as np
import torch

import config as C
import data as D
import goren_weights as W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--ratios", default=",".join(str(r) for r in C.RATIOS))
    ap.add_argument("--epochs", type=int, default=W.EPOCHS)
    ap.add_argument("--n_bottomk", type=int, default=W.N_BOTTOMK)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    K.ensure_dirs()
    ratios = [float(x) for x in args.ratios.split(",")]
    grid = [(ds, r) for ds in C.DATASETS for r in ratios
            if args.only in (None, ds)]

    print(f"GOREN weight learning, {len(grid)} settings, "
          f"n_bottomk={args.n_bottomk}, {args.epochs} epochs\n")
    print(f"{'setting':16s} {'k':>6s} {'und.edges':>9s} | {'quad base':>9s} "
          f"{'quad goren':>10s} {'d%':>6s} | {'EE base':>8s} {'EE goren':>9s} "
          f"{'EE reduction':>13s} | {'sec':>4s}")
    print("-" * 104)

    out, outputs = {}, []
    t0 = time.perf_counter()
    for ds, r in grid:
        m = D.load(ds)
        z = np.load(K.coarse_path(ds, r))
        k = int(z["assignment"].max()) + 1
        t1 = time.perf_counter()
        res = W.train_weights(m["adj"], z["edge_index"], z["edge_weight"],
                              z["assignment"], k, epochs=args.epochs,
                              n_bottomk=args.n_bottomk)
        path = os.path.join(C.COARSENED, f"goren_w_{ds}_r{r}.npy")
        np.save(path, res["weight"])
        outputs.append(K.rel(path))
        out[f"{ds}_{r}"] = {kk: vv for kk, vv in res.items()
                            if kk not in ("weight", "history")}
        print(f"{ds}_r{r:<10} {k:>6d} {res['n_undirected_edges']:>9d} | "
              f"{res['quad_loss_baseline']:9.5f} {res['quad_loss_goren']:10.5f} "
              f"{100 * res['quad_reduction']:+5.0f}% | "
              f"{res['eigenerror_baseline']:8.4f} {res['eigenerror_goren']:9.4f} "
              f"{100 * res['eigenerror_reduction']:+12.1f}% | "
              f"{time.perf_counter() - t1:4.0f}")

    path = os.path.join(C.RESULTS, "step02a_goren_weights.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    imp = [v["eigenerror_reduction"] for v in out.values()]
    n_better = sum(1 for x in imp if x > 0)
    print(f"\nEigenerror reduction: {n_better}/{len(imp)} settings improved, "
          f"mean {100 * float(np.mean(imp)):+.1f}%")

    K.record(
        "step02a_goren_weights",
        summary=(f"Trained GOREN's GIN for {len(grid)} settings; "
                 f"{n_better}/{len(imp)} improved Eigenerror, mean "
                 f"{100 * float(np.mean(imp)):+.1f}%"),
        inputs=["artifacts/coarsened/goren_*.npz"],
        outputs=outputs + [K.rel(path)],
        numbers={f"{kk}_eigenerror_reduction": round(float(vv["eigenerror_reduction"]), 4)
                 for kk, vv in out.items()},
        notes=("Loss follows the authors' loss_util.quaratic_loss: diagonal only, "
               "bottom-k eigenvectors, rayleigh-normalised, 1/|s| projection."))
    print(f"total {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
