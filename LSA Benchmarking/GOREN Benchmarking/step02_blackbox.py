"""Step 02 -- build the GOREN black boxes.

Fourteen boxes: two datasets x three ratios x two weight variants, plus two
no-coarsening controls. Reports accuracy; there is no published number to gate on.

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

import blackbox_goren as B
import config as C

CANONICAL_SEED = 0


def _one(job):
    dataset, r, variant, repeats = job
    K.pin_single_thread()
    t0 = time.perf_counter()
    bb = B.GORENBlackBox(dataset, r=r, variant=variant)
    out = bb.validate(repeats=repeats)

    bb.build(run_seed=CANONICAL_SEED)
    post = bb.serve()
    tag = f"{dataset}_{'whole' if r is None else f'{r}_{variant}'}"
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
        jobs.append((ds, None, "bl", args.repeats))          # control
        for r in C.RATIOS:
            for v in C.VARIANTS:
                jobs.append((ds, r, v, args.repeats))

    print(f"building {len(jobs)} black boxes x {args.repeats} runs, "
          f"{args.workers} workers, CPU\n")
    t0 = time.perf_counter()
    rows = K.parallel_map(_one, jobs, workers=args.workers, serial=args.serial)

    spec_path = os.path.join(C.RESULTS, "step02a_goren_weights.json")
    spec = json.load(open(spec_path)) if os.path.exists(spec_path) else {}

    print("held-out test accuracy (60/20/20, best-validation weights)")
    print(f"{'box':22s} {'k':>6s} {'w_mean':>8s} {'w_zeros':>8s} | "
          f"{'test acc':>14s} {'val':>7s} | {'EE reduction':>13s}")
    print("-" * 92)
    for x in rows:
        c = x.get("coarse")
        key = f"{x['dataset']}_{x['r']}"
        ee = spec.get(key, {}).get("eigenerror_reduction")
        ee_s = (f"{100 * float(ee):+12.1f}%" if ee is not None
                and x["variant"] == "goren" else f"{'-':>13s}")
        sp = x["split"]
        k_str = str(x["k"]) if x["k"] else str(sp["train"] + sp["val"] + sp["test"])
        w_mean = f"{c['edge_weight_mean']:8.3f}" if c else f"{'-':>8s}"
        w_zero = str(c["edge_weight_zeros"]) if c else "-"
        print(f"{x['tag']:22s} {k_str:>6s} {w_mean} {w_zero:>8s} | "
              f"{x['test_acc_mean']:6.2f}+/-{x['test_acc_std']:5.2f} "
              f"{x['best_val_mean']:7.2f} | {ee_s}")

    print("\nbl vs goren, the paired comparison")
    print(f"{'setting':16s} {'bl acc':>14s} {'goren acc':>14s} {'goren - bl':>11s} | "
          f"{'EE reduction':>13s}")
    print("-" * 78)
    for ds in C.DATASETS:
        for r in C.RATIOS:
            b = next((x for x in rows if x["dataset"] == ds and x["r"] == r
                      and x["variant"] == "bl"), None)
            g = next((x for x in rows if x["dataset"] == ds and x["r"] == r
                      and x["variant"] == "goren"), None)
            if not (b and g):
                continue
            ee = spec.get(f"{ds}_{r}", {}).get("eigenerror_reduction")
            ee_s = f"{100 * float(ee):+12.1f}%" if ee is not None else f"{'-':>13s}"
            print(f"{ds}_r{r:<10} {b['test_acc_mean']:6.2f}+/-{b['test_acc_std']:5.2f} "
                  f"{g['test_acc_mean']:6.2f}+/-{g['test_acc_std']:5.2f} "
                  f"{g['test_acc_mean'] - b['test_acc_mean']:+11.2f} | {ee_s}")

    print("\nserved posteriors")
    print(f"{'endpoint':22s} {'shape':>12s} {'distinct':>9s} {'of n':>6s} "
          f"{'mean max p':>11s} {'acc':>7s}")
    print("-" * 74)
    for x in rows:
        p = x["posteriors"]; n = p["shape"][0]
        print(f"{x['tag']:22s} {str(tuple(p['shape'])):>12s} {p['distinct_rows']:>9d} "
              f"{100 * p['distinct_rows'] / n:>5.1f}% {p['mean_max_posterior']:>11.4f} "
              f"{p['canonical_test_acc']:>7.2f}")

    path = os.path.join(C.RESULTS, "step02_validation.json")
    with open(path, "w") as f:
        json.dump({"repeats": args.repeats, "canonical_seed": CANONICAL_SEED,
                   "boxes": rows}, f, indent=2, default=str)

    K.record(
        "step02_blackbox",
        summary=(f"Built {len(rows)} GOREN black boxes: bl and goren edge weights "
                 f"across {len(C.RATIOS)} ratios plus no-coarsening controls"),
        inputs=["artifacts/coarsened/*.npz", "artifacts/coarsened/goren_w_*.npy"],
        outputs=[K.rel(path)] + [x["posteriors"]["path"] for x in rows],
        numbers={"boxes": len(rows), "repeats": args.repeats,
                 **{x["tag"]: round(x["test_acc_mean"], 2) for x in rows}},
        notes=("No published accuracy exists for GOREN on any dataset -- the paper "
               "evaluates only spectrally -- so this stage reports rather than "
               "gates. bl and goren differ in exactly one vector, the coarse edge "
               "weights. Classifier is the UGC benchmark's, which already consumes "
               "edge weights."))
    print(f"\ntotal {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
