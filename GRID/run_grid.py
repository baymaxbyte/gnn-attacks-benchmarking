#!/usr/bin/env python3
"""Run the He et al. link-stealing attacks on cora/citeseer with vs without the
GRID defence, and tabulate per-attack AUC.

Pipeline (self-contained in this GRID/ folder):
  1. grid_defense.apply_grid: perturb the served (target) posteriors of the
     whole/uncoarsened black box -> {ds}_kronbb_whole_grid_pred.pkl
  2. partial_graph_generation.py: build attack datasets for the clean baseline
     target, the GRID target, and the clean shadow (the other dataset).
  3. run_attack.py: run all 8 attacks + link-pred, 5 runs, both directions.
     The shadow (attacker's own model) is always the CLEAN other-dataset box;
     only the target's posteriors are defended, per GRID's threat model.
  4. Parse results -> results/grid_vs_baseline_theta{θ}.csv and a printed table.
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ATTACK = os.path.join(HERE, "attack")
RESULTS = os.path.join(HERE, "results", "attacks")
PY = sys.executable
sys.path.insert(0, HERE)
import grid_defense as G  # noqa: E402

DIRECTIONS = {"A": ("cora", "citeseer"), "B": ("citeseer", "cora")}
ATTACKS = [str(i) for i in range(8)]


def sh(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-1500:]); print(p.stderr[-1500:])
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return p.stdout


def build_dataset(ds, target_pred, name, force):
    probe = os.path.join(ATTACK, "data", "attack_dataset", f"{name}_attack_test.json")
    if os.path.exists(probe) and not force:
        return
    sh([PY, "partial_graph_generation.py", "--dataset", ds,
        "--datapath", "data/dataset/original/", "--prediction_path", "data/pred/",
        "--saving_path", "data/attack_dataset/", "--target_pred", target_pred,
        "--reference_pred", f"{ds}_gkron_pred.pkl", "--name", name, "--seed", "1"], ATTACK)


def run_attack(target, shadow, tag, runs):
    out = os.path.join(RESULTS, f"{target}__{shadow}__{tag}.json")
    sh([PY, "run_attack.py", "--attack", "all", "--dataset", target,
        "--shadow", shadow, "--attack_dataset_path", "data/attack_dataset/",
        "--result_path", RESULTS, "--runs", str(runs), "--tag", tag,
        "--link_prediction", "--quiet"], ATTACK)
    return out


def auc_of(entry):
    return entry.get("auc_mean", entry.get("auc"))


def parse(path):
    d = json.load(open(path))
    out = {a: auc_of(d["results"][a]) for a in ATTACKS if a in d["results"]}
    lp = d.get("link_prediction")
    if isinstance(lp, dict):
        out["link_pred"] = auc_of(lp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=0.4)
    ap.add_argument("--nhop", type=int, default=3)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    # 1) GRID posteriors + 2) attack datasets
    for ds in ("cora", "citeseer"):
        print(f"\n### GRID defence on {ds} (theta={args.theta}, n={args.nhop})")
        _, st = G.apply_grid(ds, "whole", theta=args.theta, n_hop=args.nhop)
        G.save_pred(ds, "whole", _, bb_tag="kronbb", suffix="grid")
        build_dataset(ds, f"{ds}_kronbb_whole_pred.pkl", f"{ds}_kron_whole", args.force)
        build_dataset(ds, f"{ds}_kronbb_whole_grid_pred.pkl", f"{ds}_kron_whole_grid", True)

    # 3) attacks, both directions, baseline + GRID
    rows = {}
    for d, (tgt, shadow) in DIRECTIONS.items():
        print(f"\n### attacks direction {d}: target={tgt} shadow={shadow}")
        base = run_attack(f"{tgt}_kron_whole", f"{shadow}_kron_whole",
                          f"base_{d}", args.runs)
        grid = run_attack(f"{tgt}_kron_whole_grid", f"{shadow}_kron_whole",
                          f"grid_t{args.theta}_{d}", args.runs)
        rows[tgt] = (parse(base), parse(grid))

    # 4) table
    names = {"0": "Attack-0", "1": "Attack-1", "2": "Attack-2", "3": "Attack-3",
             "4": "Attack-4", "5": "Attack-5", "6": "Attack-6", "7": "Attack-7",
             "link_pred": "link_pred"}
    order = ATTACKS + ["link_pred"]
    csv = os.path.join(HERE, "results", f"grid_vs_baseline_theta{args.theta}.csv")
    with open(csv, "w") as f:
        f.write("dataset,attack,auc_no_defense,auc_grid,delta\n")
        for ds in ("cora", "citeseer"):
            base, grid = rows[ds]
            print(f"\n=== {ds}  (no-defense -> GRID theta={args.theta}) ===")
            print(f"{'attack':10s} {'no_def':>8s} {'GRID':>8s} {'delta':>8s}")
            for a in order:
                if a not in base:
                    continue
                b, g = base[a], grid.get(a)
                dl = (g - b) if (b is not None and g is not None) else None
                print(f"{names[a]:10s} {b:8.4f} {g:8.4f} {dl:+8.4f}")
                f.write(f"{ds},{names[a]},{b:.4f},{g:.4f},{dl:+.4f}\n")
    print(f"\nwrote {os.path.relpath(csv, HERE)}")


if __name__ == "__main__":
    main()
