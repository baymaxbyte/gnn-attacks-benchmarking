"""Step 02 -- build the UGC black boxes and validate them against the paper.

Eight boxes: two datasets x (three coarsening ratios + a no-coarsening control).
Each trains on its coarse graph, selects weights by validation accuracy measured on
the ORIGINAL graph, and is then served over the original graph.

Accuracy here is a genuine held-out estimate: the split is a proper 60/20/20 and the
coarse labels aggregate only TRAINING labels, so nothing is in-sample.

Gate: Cora has published targets at all three ratios (paper Table 9). Citeseer has
no published UGC accuracy anywhere in the paper, so its boxes are reported without a
target.

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

import blackbox_ugc as B
import config as C

SETTINGS = [None] + list(C.RATIOS)
CANONICAL_SEED = 0


def _one(job):
    dataset, ratio, repeats = job
    K.pin_single_thread()
    t0 = time.perf_counter()

    bb = B.UGCBlackBox(dataset, ratio=ratio)
    if ratio is not None:
        z = np.load(K.coarse_path(dataset, ratio))
        bb.bin_width = float(z["bin_width"][0])
        bb.coarsen(assignment=z["assignment"])

    out = bb.validate(repeats=repeats)

    bb.build(run_seed=CANONICAL_SEED)
    post = bb.serve()
    tag = f"{dataset}_{'whole' if ratio is None else ratio}"
    path = os.path.join(C.POSTERIORS, f"{tag}.npy")
    np.save(path, post.astype(np.float32))
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
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gate_tolerance", type=float, default=2.0)
    ap.add_argument("--serial", action="store_true")
    args = ap.parse_args()

    K.ensure_dirs()
    os.makedirs(C.POSTERIORS, exist_ok=True)
    jobs = [(ds, r, args.repeats) for ds in C.DATASETS for r in SETTINGS
            if args.only in (None, ds)]

    print(f"building {len(jobs)} black boxes x {args.repeats} runs, "
          f"{args.workers} workers, CPU\n")
    t0 = time.perf_counter()
    rows = K.parallel_map(_one, jobs, workers=args.workers, serial=args.serial)

    print("held-out test accuracy (60/20/20 split, best-validation weights)")
    print(f"{'box':16s} {'alpha':>6s} {'k':>6s} {'bw':>9s} {'aug dim':>8s} | "
          f"{'test acc':>14s} {'paper':>7s} {'delta':>7s} gate")
    print("-" * 104)
    failures = []
    for x in rows:
        lbl = f"{x['dataset']}_{'whole' if x['ratio'] is None else x['ratio']}"
        pm, dv = x["paper_mean"], x["delta_vs_paper"]
        if pm is None:
            gate = "n/a"
        else:
            ok = abs(dv) <= args.gate_tolerance
            gate = "pass" if ok else "FAIL"
            if not ok:
                failures.append(lbl)
        sp = x["split"]
        n_nodes = sp["train"] + sp["val"] + sp["test"]
        k_str = str(x["k"]) if x["k"] else str(n_nodes)
        bw_str = f"{x['bin_width']:.6f}" if x["bin_width"] else "-"
        pm_str = f"{pm:7.2f}" if pm else f"{'-':>7s}"
        dv_str = f"{dv:+7.2f}" if dv is not None else f"{'-':>7s}"
        print(f"{lbl:16s} {x['alpha']:>6.2f} {k_str:>6s} {bw_str:>9s} "
              f"{x['feature_dim_augmented']:>8d} | "
              f"{x['test_acc_mean']:6.2f}+/-{x['test_acc_std']:5.2f} "
              f"{pm_str} {dv_str} {gate}")

    print("\ncoarsening geometry and served posteriors")
    print(f"{'box':16s} {'trainable':>9s} {'maxsz':>6s} {'meansz':>7s} {'c_edges':>8s} | "
          f"{'distinct':>9s} {'of n':>6s} {'mean max p':>11s}")
    print("-" * 96)
    for x in rows:
        lbl = f"{x['dataset']}_{'whole' if x['ratio'] is None else x['ratio']}"
        p = x["posteriors"]; n = p["shape"][0]
        c = x.get("coarse")
        geo = (f"{c['trainable_supernodes']:>9d} {c['max_supernode_size']:>6d} "
               f"{c['mean_supernode_size']:>7.1f} {c['coarse_edges']:>8d}"
               if c else f"{'-':>9s} {'-':>6s} {'-':>7s} {'-':>8s}")
        print(f"{lbl:16s} {geo} | {p['distinct_rows']:>9d} "
              f"{100*p['distinct_rows']/n:>5.1f}% {p['mean_max_posterior']:>11.4f}")

    path = os.path.join(C.RESULTS, "step02_validation.json")
    with open(path, "w") as f:
        json.dump({"repeats": args.repeats, "gate_tolerance": args.gate_tolerance,
                   "canonical_seed": CANONICAL_SEED, "alpha": C.ALPHA,
                   "boxes": rows}, f, indent=2)

    gated = [x for x in rows if x["paper_mean"]]
    dev = max((abs(x["delta_vs_paper"]) for x in gated), default=0.0)
    K.record(
        "step02_blackbox",
        summary=(f"Built and validated {len(rows)} UGC black boxes; "
                 f"{len(gated) - len(failures)}/{len(gated)} gated boxes inside "
                 f"tolerance, largest deviation {dev:.2f} points"),
        inputs=["artifacts/coarsened/*.npz", "UGC paper Table 8 (model), Table 9 (targets)"],
        outputs=[K.rel(path)] + [x["posteriors"]["path"] for x in rows],
        numbers={"boxes": len(rows), "repeats": args.repeats,
                 "gate_failures": len(failures),
                 "max_deviation": round(dev, 2),
                 **{f"{x['dataset']}_{'whole' if x['ratio'] is None else x['ratio']}":
                    round(x["test_acc_mean"], 2) for x in rows}},
        notes=("Held-out 60/20/20 accuracy with best-validation weight selection. "
               "Features are the augmented [(1-alpha)X | alpha*A] at both training "
               "and inference, so the graph reaches the model twice. Citeseer has "
               "no published UGC accuracy, so those four boxes are ungated."))
    print(f"\ntotal {(time.perf_counter() - t0) / 60:.1f} min")
    if failures:
        print(f"outside tolerance: {', '.join(failures)}")


if __name__ == "__main__":
    main()
