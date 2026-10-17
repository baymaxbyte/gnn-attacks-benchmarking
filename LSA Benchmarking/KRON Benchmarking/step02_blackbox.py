"""Step 02 -- build the KRON black boxes.

Eight boxes: two datasets x three reduction ratios, plus two no-reduction controls.
Each reduced box trains on the Kron-reduced graph and is evaluated on the original
graph, so the only thing that differs from its own control is the training graph.

There is no published accuracy to gate against -- Kron reduction never runs node
classification -- so this stage reports rather than gates.

    python step02_blackbox.py
    python step02_blackbox.py --repeats 5 --only cora
"""
import argparse
import json
import os
import time

import common as K

os.environ.update(K.single_thread_env())

import numpy as np

import blackbox_kron as B
import config as C

CANONICAL_SEED = 0


def _one(job):
    dataset, r, repeats = job
    K.pin_single_thread()
    t0 = time.perf_counter()
    bb = B.KRONBlackBox(dataset, r=r)
    out = bb.validate(repeats=repeats)

    bb.build(run_seed=CANONICAL_SEED)
    post = bb.serve()
    tag = f"{dataset}_{'whole' if r is None else r}"
    path = os.path.join(C.POSTERIORS, f"{tag}.npy")
    np.save(path, post.astype(np.float32))
    out["tag"] = tag
    out["posteriors"] = {
        "path": K.rel(path), "shape": list(post.shape),
        "distinct_rows": int(len(np.unique(np.round(post, 10), axis=0))),
        "mean_max_posterior": float(post.max(1).mean()),
        "canonical_test_acc": float(bb.accuracy("test") * 100),
    }
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--only", default=None)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--serial", action="store_true")
    args = ap.parse_args()

    K.ensure_dirs()
    os.makedirs(C.POSTERIORS, exist_ok=True)

    jobs = []
    for ds in C.DATASETS:
        if args.only not in (None, ds):
            continue
        jobs.append((ds, None, args.repeats))               # control
        for r in C.RATIOS:
            jobs.append((ds, r, args.repeats))

    print(f"building {len(jobs)} black boxes x {args.repeats} runs, "
          f"{args.workers} workers, CPU\n")
    t0 = time.perf_counter()
    rows = K.parallel_map(_one, jobs, workers=args.workers, serial=args.serial)

    print("held-out test accuracy (60/20/20, best-validation weights, "
          "evaluated on the ORIGINAL graph)")
    print(f"{'box':16s} {'k':>6s} {'edges':>7s} {'w_mean':>8s} | "
          f"{'test acc':>14s} {'val':>7s} {'sec':>6s}")
    print("-" * 74)
    for x in rows:
        c = x.get("reduced")
        k_str = str(x["k"]) if x["k"] else str(x["split"]["train"]
                                                + x["split"]["val"]
                                                + x["split"]["test"])
        edges = str(c["reduced_edges"]) if c else "-"
        w_mean = f"{c['edge_weight_mean']:8.3f}" if c else f"{'-':>8s}"
        print(f"{x['tag']:16s} {k_str:>6s} {edges:>7s} {w_mean} | "
              f"{x['test_acc_mean']:6.2f}+/-{x['test_acc_std']:5.2f} "
              f"{x['best_val_mean']:7.2f} {x['seconds']:>6.1f}")

    print("\nreduction effect (each ratio vs its own un-reduced control)")
    print(f"{'setting':16s} {'control':>9s} {'reduced':>9s} {'reduced - control':>18s}")
    print("-" * 56)
    for ds in C.DATASETS:
        ctrl = next((x for x in rows if x["dataset"] == ds and x["r"] is None),
                    None)
        if not ctrl:
            continue
        for r in C.RATIOS:
            g = next((x for x in rows if x["dataset"] == ds and x["r"] == r), None)
            if not g:
                continue
            print(f"{ds}_r{r:<9} {ctrl['test_acc_mean']:9.2f} "
                  f"{g['test_acc_mean']:9.2f} "
                  f"{g['test_acc_mean'] - ctrl['test_acc_mean']:+18.2f}")

    print("\nserved posteriors")
    print(f"{'endpoint':16s} {'shape':>12s} {'distinct':>9s} {'of n':>6s} "
          f"{'mean max p':>11s} {'acc':>7s}")
    print("-" * 68)
    for x in rows:
        p = x["posteriors"]; n = p["shape"][0]
        print(f"{x['tag']:16s} {str(tuple(p['shape'])):>12s} {p['distinct_rows']:>9d} "
              f"{100 * p['distinct_rows'] / n:>5.1f}% {p['mean_max_posterior']:>11.4f} "
              f"{p['canonical_test_acc']:>7.2f}")

    path = os.path.join(C.RESULTS, "step02_validation.json")
    with open(path, "w") as f:
        json.dump({"repeats": args.repeats, "canonical_seed": CANONICAL_SEED,
                   "boxes": rows}, f, indent=2, default=str)

    K.record(
        "step02_blackbox",
        summary=(f"Built {len(rows)} KRON black boxes across {len(C.RATIOS)} "
                 f"reduction ratios plus no-reduction controls"),
        inputs=["artifacts/reduced/kron_*.npz"],
        outputs=[K.rel(path)] + [x["posteriors"]["path"] for x in rows],
        numbers={"boxes": len(rows), "repeats": args.repeats,
                 **{x["tag"]: round(x["test_acc_mean"], 2) for x in rows}},
        notes=("No published accuracy exists for Kron reduction on any dataset -- "
               "the paper is spectral/resistive on electrical networks -- so this "
               "stage reports rather than gates. Each box trains on the reduced "
               "graph and is evaluated on the original graph, so the only "
               "difference from its control is the training graph. Classifier is "
               "the UGC benchmark's, which already consumes edge weights."))
    print(f"\ntotal {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
