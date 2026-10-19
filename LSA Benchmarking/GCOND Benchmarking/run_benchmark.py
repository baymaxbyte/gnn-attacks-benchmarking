"""Master script: one GCond leakage benchmark run, end to end, all four datasets.

    python run_benchmark.py                         # cora, citeseer, chameleon, squirrel
    python run_benchmark.py --directions A,B         # cora/citeseer only
    python run_benchmark.py --datasets chameleon,squirrel --directions C,D

Runs every stage in order, for every dataset:

    1  condense each dataset at each r                     (step01)
    2  build the black boxes, validate against the paper    (step02)
    3  train the adversary's reference model g              (step03)
    4  build attack datasets on ORIGINAL edges              (step03)
    5  run all 8 attacks + link-prediction baseline         (step04)
    6  compile CSVs and plots                              (step05)

r is GCond's CLI reduction rate (0.25 / 0.5 / 1.0, a fraction of the labeling
rate). --directions picks the target/shadow pairs to attack: A cora->citeseer,
B citeseer->cora, C chameleon->squirrel, D squirrel->chameleon. The no-condensation
control ("whole") is built by default; --no_whole skips it.

Condensation runs under the GCond checkout. This script points GCOND_HOME /
GCOND_PYTHON at ../../../../GCond/GCond-main by default (config.py's built-in
default is one layer too shallow); override by exporting those variables.

Resumable: condensation is cached per cell by step01, black boxes are skipped when
their posteriors already exist, the reference model and attack datasets are cached,
and step04 skips attack results that already exist. --force redoes everything.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Point GCOND_HOME/GCOND_PYTHON at the project-root GCond checkout *before* importing
# config (config.py resolves its default one layer too shallow). setdefault respects
# an explicit export.
_GCOND_HOME = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..",
                                           "GCond", "GCond-main"))
os.environ.setdefault("GCOND_HOME", _GCOND_HOME)
_gcond_py = os.path.join(os.environ["GCOND_HOME"], ".venv", "bin", "python3")
if os.path.exists(_gcond_py):
    os.environ.setdefault("GCOND_PYTHON", _gcond_py)

import config

DATASETS = ("cora", "citeseer", "chameleon", "squirrel")


def parse_args():
    p = argparse.ArgumentParser(
        description="GCond link-leakage benchmark, end to end, all four datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets", default=",".join(DATASETS),
                   help="datasets to condense/box/attack")
    p.add_argument("--ratios", default="0.25,0.5,1.0",
                   help="GCond CLI reduction rates")
    p.add_argument("--directions", default="A,B,C,D",
                   help="A cora->citeseer, B citeseer->cora, "
                        "C chameleon->squirrel, D squirrel->chameleon")
    p.add_argument("--no_whole", action="store_true",
                   help="skip the no-condensation control box")
    p.add_argument("--repeats", type=int, default=10,
                   help="black-box evaluation repeats (paper: 10; squirrel is slow, "
                        "try 3)")
    p.add_argument("--runs", type=int, default=5, help="attack-model repeats")
    p.add_argument("--skip_condense", action="store_true")
    p.add_argument("--skip_attacks", action="store_true",
                   help="stop after the validation gate")
    p.add_argument("--no_clustering", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="recondense / rebuild even if artifacts exist")
    p.add_argument("--device", default="cpu")
    p.add_argument("--gate_tolerance", type=float, default=3.0,
                   help="fail if a box with a paper target deviates by more than "
                        "this many accuracy points")
    return p.parse_args()


def banner(n, title):
    print(f"\n{'#' * 78}\n# STAGE {n}: {title}\n{'#' * 78}", flush=True)


def main():
    args = parse_args()
    t_start = time.time()

    problems = config.check(verbose=True)
    if problems:
        raise SystemExit("\nresolve the dependency problems above first "
                         "(see config.py, or export GCOND_HOME / GCOND_PYTHON)")

    datasets = [d.strip() for d in args.datasets.split(",")]
    ratios = [r.strip() for r in args.ratios.split(",")]
    settings = ([] if args.no_whole else ["whole"]) + ratios
    directions = [d.strip() for d in args.directions.split(",")]
    print(f"\ndatasets={datasets}  ratios={ratios}  settings={settings}")
    print(f"directions={directions}  box repeats={args.repeats}  attack runs={args.runs}")

    import common as C
    import step01_condense as s1
    import step03_attacker as s3
    import step04_attacks as s4
    import step05_aggregate as s5
    from blackbox_gcond import GCondBlackBox
    C.ensure_dirs()

    # ---- 1. condensation (step01 caches per cell) -------------------------
    banner(1, f"condense {datasets} at r={ratios}")
    if args.skip_condense:
        print("  skipped (--skip_condense)")
    else:
        expect = {(ds, r): n for ds, r, _, n in s1.GRID}
        for ds in datasets:
            for r in ratios:
                if (ds, r) not in expect:
                    known = sorted({rr for d, rr in expect if d == ds})
                    raise SystemExit(f"r={r} is not a known setting for {ds}. "
                                     f"Known: {known}")
                s1.condense(ds, r, expect[(ds, r)], force=args.force)

    # ---- 2. black boxes + validation gate --------------------------------
    banner(2, "build black boxes and validate against GCond Table 2")
    gate, worst = [], 0.0
    for ds in datasets:
        for s in settings:
            pkl = os.path.join(C.LSA_PRED, f"{ds}_gcondbb_{s}_pred.pkl")
            if os.path.exists(pkl) and not args.force:
                print(f"  [cached] {ds}_gcondbb_{s}")
                continue
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
            if res["delta"] is not None:
                print(f"    -> {res['acc_mean']:.2f} +-{res['acc_std']:.2f}   "
                      f"{ref}   delta {res['delta']:+.2f}")
            else:
                print(f"    -> {res['acc_mean']:.2f} +-{res['acc_std']:.2f}   {ref}")

    print(f"\n  gate: largest deviation from the paper = {worst:.2f} points "
          f"(tolerance {args.gate_tolerance}; paperless datasets are not gated)")
    if worst > args.gate_tolerance:
        raise SystemExit(
            f"\nVALIDATION GATE FAILED. A box with a paper target is {worst:.2f} "
            f"points from the paper, beyond the {args.gate_tolerance} tolerance. "
            f"Attacks not run -- fix the box, or raise --gate_tolerance.")
    if gate:
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

    # ---- 4. attack datasets (step03 caches) ------------------------------
    banner(4, "attack datasets on ORIGINAL edges")
    for ds in datasets:
        for s in settings:
            s3.build_attack_dataset(ds, s)

    # ---- 5. attacks (step04 caches) --------------------------------------
    banner(5, f"all 8 attacks, directions={directions}")
    for d in directions:
        if d not in s4.DIRECTIONS:
            print(f"  ! unknown direction {d}; skipping")
            continue
        target, shadow = s4.DIRECTIONS[d]
        if target not in datasets or shadow not in datasets:
            print(f"  ! direction {d} needs {target}+{shadow}, not both selected; "
                  f"skipping")
            continue
        for s in settings:
            s4.run_one(d, target, shadow, s, args.runs)

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
    print(f"# datasets={datasets}  directions={directions}")
    print(f"# results/  <ds>_attacks_{{detailed,summary}}.csv, figures/, GCOND_RESULTS.md")
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
