"""Step 02 -- build the FGC black boxes and validate them against the paper.

Eight trained models: two datasets x (whole + r in {0.3, 0.1, 0.05}). Each is
served two ways, so fourteen endpoints -- Mode B is undefined for the whole-graph
control because there is no partition to lift through.

Accuracy is measured over ALL n original nodes, which is the paper's protocol: the
coarse labels derive from every original label and there is no held-out split, so
the number is in-sample. It is a reproduction check, not a generalisation estimate.

    python step02_blackbox.py
    python step02_blackbox.py --repeats 10 --strict
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

SETTINGS = [None] + list(B.PAPER_R)
POSTERIOR_DIR = os.path.join(C.ARTIFACTS, "posteriors")
CANONICAL_SEED = 0


def _load_coarsening(dataset, r):
    """Return (C, beta) from the artifact step01 wrote."""
    z = np.load(K.coarse_path(dataset, r))
    return z["C"], float(z["beta"][0])


def _one(job):
    """Validate one box and save its canonical posteriors for both modes."""
    dataset, r, repeats = job
    K.pin_single_thread()
    t0 = time.perf_counter()

    if r is None:
        bb = B.FGCBlackBox(dataset, r=None)
    else:
        Cmat, beta = _load_coarsening(dataset, r)
        bb = B.FGCBlackBox(dataset, r=r, beta=beta)
        bb.coarsen(C=Cmat)

    out = bb.validate(repeats=repeats)
    out["beta"] = bb.hp["beta"]
    out["hp"] = {k: float(v) for k, v in bb.hp.items()}

    # canonical build -> the posteriors downstream stages will query
    bb.build(run_seed=CANONICAL_SEED)
    tag = f"{dataset}_{'whole' if r is None else r}"
    saved = {}
    for mode in (("orig",) if r is None else ("orig", "lift")):
        post = bb.serve(mode=mode)
        path = os.path.join(POSTERIOR_DIR, f"{tag}_{mode}.npy")
        np.save(path, post.astype(np.float32))
        saved[mode] = {
            "path": K.rel(path),
            "shape": list(post.shape),
            "distinct_rows": int(len(np.unique(np.round(post, 10), axis=0))),
            "mean_max_posterior": float(post.max(1).mean()),
            "canonical_accuracy": float(
                (post.argmax(1) == bb.meta["labels"]).mean() * 100),
        }
    out["posteriors"] = saved
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10,
                    help="GNN re-inits per box, with the coarsening held fixed")
    ap.add_argument("--only", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gate_tolerance", type=float, default=1.0,
                    help="accuracy points allowed on top of the paper's own std")
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero if any box fails its gate")
    ap.add_argument("--serial", action="store_true", help="one process, for debugging")
    args = ap.parse_args()

    K.ensure_dirs()
    os.makedirs(POSTERIOR_DIR, exist_ok=True)
    jobs = [(ds, r, args.repeats) for ds in C.DATASETS for r in SETTINGS
            if args.only in (None, ds)]

    print(f"building {len(jobs)} black boxes x {args.repeats} GNN re-inits, "
          f"{args.workers} workers, CPU\n")
    t0 = time.perf_counter()
    rows = K.parallel_map(_one, jobs, workers=args.workers, serial=args.serial)

    # ---------------------------------------------------------------- Mode A gate
    print("MODE A  -- model(X_original, A_original)[u]   gate vs kumar23a T4/T6")
    print(f"{'box':18s} {'k':>5s} {'beta':>6s} | {'ours':>14s} {'paper':>14s} "
          f"{'delta':>7s} {'band':>5s}  gate")
    print("-" * 88)
    failures = []
    for x in rows:
        a = x["orig"]
        band = max(x["paper_std"], args.gate_tolerance)
        ok = abs(a["delta_vs_paper"]) <= band
        if not ok:
            failures.append(x)
        label = f"{x['dataset']}_{'whole' if x['r'] is None else x['r']}"
        print(f"{label:18s} {str(x['k'] or '-'):>5s} {x['beta']:>6g} | "
              f"{a['mean']:6.2f}+/-{a['std']:5.2f} {x['paper_mean']:7.2f}"
              f"+/-{x['paper_std']:5.2f} {a['delta_vs_paper']:+7.2f} {band:5.2f}  "
              f"{'pass' if ok else 'FAIL'}")

    # ---------------------------------------------------------------- Mode B
    print("\nMODE B  -- model(X_coarse, A_coarse)[s(u)]    no published target")
    print(f"{'box':18s} {'k':>5s} | {'ours':>14s} {'vs ModeA':>9s} | "
          f"{'distinct':>8s} {'meansz':>7s} {'maxsz':>6s} {'ident.pairs':>12s}")
    print("-" * 88)
    for x in rows:
        if x["r"] is None:
            continue
        b, col = x["lift"], x["collisions"]
        label = f"{x['dataset']}_{x['r']}"
        print(f"{label:18s} {x['k']:>5d} | {b['mean']:6.2f}+/-{b['std']:5.2f} "
              f"{b['mean'] - x['orig']['mean']:+9.2f} | "
              f"{col['distinct_posteriors_lift']:8d} {col['mean_group_size']:7.1f} "
              f"{col['max_group_size']:6d} {col['identical_pairs']:12d}")

    # ---------------------------------------------------------------- posteriors
    print("\nSERVED POSTERIORS (canonical build, model.eval())")
    print(f"{'endpoint':26s} {'shape':>12s} {'distinct':>9s} {'mean max p':>11s} {'acc':>7s}")
    print("-" * 72)
    for x in rows:
        for mode, p in x["posteriors"].items():
            label = f"{x['dataset']}_{'whole' if x['r'] is None else x['r']}_{mode}"
            print(f"{label:26s} {str(tuple(p['shape'])):>12s} {p['distinct_rows']:9d} "
                  f"{p['mean_max_posterior']:11.4f} {p['canonical_accuracy']:7.2f}")

    path = os.path.join(C.RESULTS, "step02_validation.json")
    with open(path, "w") as f:
        json.dump({"repeats": args.repeats, "gate_tolerance": args.gate_tolerance,
                   "canonical_seed": CANONICAL_SEED, "boxes": rows}, f, indent=2)

    dev = max(abs(x["orig"]["delta_vs_paper"]) for x in rows)
    K.record(
        "step02_blackbox",
        summary=(f"Built and validated {len(rows)} FGC black boxes; "
                 f"{len(rows) - len(failures)}/{len(rows)} inside gate, "
                 f"largest Mode A deviation {dev:.2f} points"),
        inputs=["artifacts/coarsened/*.npz",
                "data/dataset/original/ind.{cora,citeseer}.*",
                "kumar23a.pdf (Table 4, Table 6)"],
        outputs=[K.rel(path)] + [p["path"] for x in rows for p in x["posteriors"].values()],
        numbers={"boxes": len(rows), "endpoints": sum(len(x["posteriors"]) for x in rows),
                 "repeats": args.repeats, "gate_failures": len(failures),
                 "max_mode_a_deviation": round(dev, 2),
                 **{f"{x['dataset']}_{'whole' if x['r'] is None else x['r']}_modeA":
                    round(x["orig"]["mean"], 2) for x in rows},
                 **{f"{x['dataset']}_{x['r']}_modeB": round(x["lift"]["mean"], 2)
                    for x in rows if x["r"] is not None}},
        notes=("Accuracy is in-sample over all n nodes, reproducing the paper's "
               "protocol. Served with model.eval(); the notebook never calls it and "
               "therefore predicts with dropout active."))

    print(f"\ntotal {(time.perf_counter() - t0) / 60:.1f} min")
    if failures:
        print(f"\n{len(failures)} box(es) outside gate: " +
              ", ".join(f"{x['dataset']}_{'whole' if x['r'] is None else x['r']}"
                        for x in failures))
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
