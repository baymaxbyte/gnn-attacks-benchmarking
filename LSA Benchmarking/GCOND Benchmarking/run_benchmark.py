"""Master script: one GCond leakage benchmark run, end to end.

    python run_benchmark.py --dataset cora --shadow citeseer --r 0.5

Runs every stage in order:

    1  condense the target and shadow graphs at r          (step01)
    2  build the black boxes, validate against the paper    (step02)
    3  train the adversary's reference model g              (step03)
    4  build attack datasets on ORIGINAL edges              (step03)
    5  run all 8 attacks + link-prediction baseline         (step04)
    6  compile CSVs and plots                              (step05)

r is GCond's CLI reduction rate -- a fraction of the dataset's LABELING rate, so
0.25 / 0.5 / 1.0 are the paper's three settings.

The no-condensation control ("whole") is built alongside by default. Without it
the numbers are uninterpretable, because any AUC difference could be the
architecture or the split rather than condensation. --no_whole skips it.
"""
import argparse
import os
import sys
import time

import config


def parse_args():
    p = argparse.ArgumentParser(
        description="GCond link-leakage benchmark, end to end",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="cora",
                   help="target dataset: the graph the black box is built on")
    p.add_argument("--shadow", default="citeseer",
                   help="shadow dataset the adversary owns (attacks 1,4,5,7)")
    p.add_argument("--r", default="0.5",
                   help="GCond CLI reduction rate: 0.25, 0.5 or 1.0")
    p.add_argument("--no_whole", action="store_true",
                   help="skip the no-condensation control box")
    p.add_argument("--repeats", type=int, default=10,
                   help="black-box evaluation repeats (paper: 10)")
    p.add_argument("--runs", type=int, default=5,
                   help="attack-model repeats (paper: 5)")
    p.add_argument("--skip_condense", action="store_true",
                   help="reuse condensed artifacts already in artifacts/condensed")
    p.add_argument("--skip_attacks", action="store_true",
                   help="stop after the validation gate")
    p.add_argument("--no_clustering", action="store_true",
                   help="skip the K-means figures (they re-read the attack JSONL)")
    p.add_argument("--force", action="store_true",
                   help="recondense even if artifacts exist")
    p.add_argument("--device", default="cpu")
    p.add_argument("--gate_tolerance", type=float, default=3.0,
                   help="fail if a box deviates from the paper by more than this "
                        "many accuracy points")
    return p.parse_args()


def banner(n, title):
    print(f"\n{'#' * 78}\n# STAGE {n}: {title}\n{'#' * 78}", flush=True)


def main():
    args = parse_args()
    t_start = time.time()

    problems = config.check(verbose=True)
    if problems:
        raise SystemExit("\nresolve the dependency problems above first "
                         "(see config.py)")

    settings = ([] if args.no_whole else ["whole"]) + [args.r]
    datasets = [args.dataset] + ([args.shadow] if args.shadow != args.dataset
                                 else [])
    print(f"\ntarget={args.dataset}  shadow={args.shadow}  r={args.r}")
    print(f"settings={settings}  datasets={datasets}")
    print(f"black-box repeats={args.repeats}  attack runs={args.runs}")

    import common as C
    import step01_condense as s1
    import step02_blackbox as s2          # noqa: F401  (kept for parity)
    import step03_attacker as s3
    import step04_attacks as s4
    import step05_aggregate as s5
    from blackbox_gcond import GCondBlackBox
    C.ensure_dirs()

    # ---- 1. condensation --------------------------------------------------
    banner(1, f"condense {datasets} at r={args.r}")
    if args.skip_condense:
        print("  skipped (--skip_condense)")
    else:
        expect = {(ds, r): n for ds, r, _, n in s1.GRID}
        for ds in datasets:
            key = (ds, args.r)
            if key not in expect:
                raise SystemExit(
                    f"r={args.r} is not one of the paper's settings for {ds}. "
                    f"Known: {sorted({r for d, r in expect if d == ds})}")
            s1.condense(ds, args.r, expect[key], force=args.force)

    # ---- 2. black boxes + validation gate --------------------------------
    banner(2, "build black boxes and validate against GCond Table 2")
    gate, worst = [], 0.0
    for ds in datasets:
        for s in settings:
            print(f"\n  {ds}  setting={s}", flush=True)
            bb = GCondBlackBox(ds, s, device=args.device)
            res = bb.validate(repeats=args.repeats)
            post = res.pop("served_posteriors")
            C.save_posteriors(f"{ds}_gcondbb_{s}", post)
            gate.append(res)
            if res["delta"] is not None:
                worst = max(worst, abs(res["delta"]))
            ref = (f"paper {res['paper_mean']:.1f}"
                   if res["paper_mean"] is not None else "paper n/a")
            print(f"    -> {res['acc_mean']:.2f} +-{res['acc_std']:.2f}   {ref}"
                  f"   delta {res['delta']:+.2f}"
                  if res["delta"] is not None else
                  f"    -> {res['acc_mean']:.2f} +-{res['acc_std']:.2f}   {ref}")

    print(f"\n  gate: largest deviation from the paper = {worst:.2f} points "
          f"(tolerance {args.gate_tolerance})")
    if worst > args.gate_tolerance:
        raise SystemExit(
            f"\nVALIDATION GATE FAILED. A black box is {worst:.2f} points from "
            f"the paper, beyond the {args.gate_tolerance} tolerance. Attacks not "
            f"run -- fix the box first, or raise --gate_tolerance deliberately.")
    _write_gate(gate, args)

    if args.skip_attacks:
        print("\nstopping after the gate (--skip_attacks)")
        return

    # ---- 3. reference model g --------------------------------------------
    banner(3, "adversary's reference model g (graph-free)")
    for ds in datasets:
        path = os.path.join(C.LSA_PRED, f"{ds}_g_pred.pkl")
        if os.path.exists(path) and not args.force:
            print(f"  [cached] {ds}_g_pred.pkl")
            continue
        print(f"\n  {ds}:")
        post, _ = s3.train_reference_model(ds, device=args.device)
        C.save_posteriors(f"{ds}_g", post)

    # ---- 4. attack datasets ----------------------------------------------
    banner(4, "attack datasets on ORIGINAL edges")
    for ds in datasets:
        for s in settings:
            s3.build_attack_dataset(ds, s)

    # ---- 5. attacks -------------------------------------------------------
    banner(5, f"all 8 attacks, target={args.dataset} shadow={args.shadow}")
    for s in settings:
        s4.run_one("A", args.dataset, args.shadow, s, args.runs)

    # ---- 6. compile -------------------------------------------------------
    banner(6, "compile CSVs and plots")
    argv = sys.argv
    sys.argv = ["step05_aggregate.py"] + (["--no_clustering"]
                                          if args.no_clustering else [])
    try:
        s5.main()
    finally:
        sys.argv = argv

    print(f"\n{'#' * 78}")
    print(f"# DONE in {(time.time() - t_start) / 60:.1f} min")
    print(f"# results/  {args.dataset}_attacks_detailed.csv, "
          f"{args.dataset}_attacks_summary.csv, figures/, GCOND_RESULTS.md")
    print(f"{'#' * 78}")


def _write_gate(gate, args):
    import json

    import common as C
    p = os.path.join(C.DIRS["results"], "step02_validation.json")
    rows = []
    if os.path.exists(p):
        rows = json.load(open(p)).get("rows", [])
    keep = {(r["dataset"], r["setting"]) for r in gate}
    rows = [r for r in rows if (r["dataset"], r["setting"]) not in keep] + gate
    with open(p, "w") as f:
        json.dump({"eval_spec": {
            "model": "2-layer GCN", "hidden": 256, "dropout": 0.0,
            "weight_decay": 5e-4, "epochs": 600, "lr": 0.01,
            "repeats": args.repeats,
            "source": "paper Appendix A.2 + tester_other_arcs.py"},
            "rows": rows}, f, indent=2)
    print(f"  [saved] {C.rel(p)}")


if __name__ == "__main__":
    main()
