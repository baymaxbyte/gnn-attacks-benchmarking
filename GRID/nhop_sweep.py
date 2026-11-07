"""Sweep GRID's n_hop (n=3,5,6) on the WHOLE cora/citeseer boxes, theta=0.4,
whole shadow. Reports attack AUC baseline vs GRID at each n, plus the delta
(n-hop floor) and achieved adjacent-similarity reduction per n.
"""
import os, sys, json, pickle, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ATTACK = os.path.join(HERE, "attack")
PRED = os.path.join(ATTACK, "data", "pred")
RES = os.path.join(HERE, "results", "nhop_sweep")
PY = sys.executable
sys.path.insert(0, HERE)
import grid_defense as G

DIRS = {"cora": "citeseer", "citeseer": "cora"}
ATTACKS = [str(i) for i in range(8)]
NHOPS = [3, 5, 6]


def sh(cmd):
    p = subprocess.run(cmd, cwd=ATTACK, capture_output=True, text=True)
    if p.returncode:
        sys.stderr.write(p.stdout[-1200:] + p.stderr[-1200:]); raise SystemExit("fail " + " ".join(cmd))


def build(ds, tgt, name):
    probe = os.path.join(ATTACK, "data", "attack_dataset", f"{name}_attack_test.json")
    if os.path.exists(probe) and os.path.getsize(probe) > 0:
        return
    sh([PY, "partial_graph_generation.py", "--dataset", ds, "--datapath", "data/dataset/original/",
        "--prediction_path", "data/pred/", "--saving_path", "data/attack_dataset/",
        "--target_pred", tgt, "--reference_pred", f"{ds}_gkron_pred.pkl", "--name", name, "--seed", "1"])


def attack(target, shadow, tag):
    out = os.path.join(RES, f"{target}__{shadow}__{tag}.json")
    if os.path.exists(out):
        return out
    sh([PY, "run_attack.py", "--attack", "all", "--dataset", target, "--shadow", shadow,
        "--attack_dataset_path", "data/attack_dataset/", "--result_path", RES, "--runs", "5",
        "--tag", tag, "--link_prediction", "--quiet"])
    return out


def parse(p):
    d = json.load(open(p)); out = {a: d["results"][a].get("auc_mean", d["results"][a].get("auc"))
                                   for a in ATTACKS if a in d.get("results", {})}
    lp = d.get("link_prediction")
    if isinstance(lp, dict): out["link_pred"] = lp.get("auc_mean", lp.get("auc"))
    return out


def main():
    os.makedirs(RES, exist_ok=True)
    simstats = {}
    # build clean whole datasets (targets + shadows) and run baseline
    base = {}
    for ds, shadow in DIRS.items():
        build(ds, f"{ds}_kronbb_whole_pred.pkl", f"{ds}_kron_whole")
    for ds, shadow in DIRS.items():
        base[ds] = parse(attack(f"{ds}_kron_whole", f"{shadow}_kron_whole", f"{ds}_base"))
    # GRID at each n_hop
    grid = {ds: {} for ds in DIRS}
    for n in NHOPS:
        for ds, shadow in DIRS.items():
            R, st = G.apply_grid(ds, "whole", theta=0.4, n_hop=n, verbose=False)
            simstats[(ds, n)] = (st["adj_before"], st["adj_after"], st["nhop_after"], st["core"])
            with open(os.path.join(PRED, f"{ds}_kronbb_whole_gN{n}_pred.pkl"), "wb") as f:
                pickle.dump(R, f)
            build(ds, f"{ds}_kronbb_whole_gN{n}_pred.pkl", f"{ds}_kron_whole_gN{n}")
            grid[ds][n] = parse(attack(f"{ds}_kron_whole_gN{n}", f"{shadow}_kron_whole", f"{ds}_gN{n}"))

    print("\n=== n-hop floor (delta) + adjacent-sim reduction (whole boxes) ===")
    for ds in DIRS:
        for n in NHOPS:
            b, a, fl, c = simstats[(ds, n)]
            print(f"  {ds:9s} n={n}: adj_sim {b:.3f}->{a:.3f}  floor(delta)={fl:.3f}  core={c}")
    for key in ["0", "1", "6"]:
        print(f"\n=== Attack-{key}: baseline -> GRID at n=3/5/6 ===")
        for ds in DIRS:
            b = base[ds][key]
            gs = "  ".join(f"n{n}:{grid[ds][n][key]:.3f}({grid[ds][n][key]-b:+.3f})" for n in NHOPS)
            print(f"  {ds:9s} base {b:.3f}  ->  {gs}")
    csv = os.path.join(HERE, "results", "nhop_sweep.csv")
    with open(csv, "w") as f:
        f.write("dataset,attack,auc_base," + ",".join(f"auc_grid_n{n}" for n in NHOPS) + "\n")
        for ds in DIRS:
            for a in ATTACKS + ["link_pred"]:
                if a in base[ds]:
                    f.write(f"{ds},Attack-{a} if a.isdigit() else a," .replace("Attack-link_pred", "link_pred"))
                    f.write(f"{base[ds][a]:.4f}," + ",".join(f"{grid[ds][n].get(a, float('nan')):.4f}" for n in NHOPS) + "\n")
    print("\nwrote", csv)


if __name__ == "__main__":
    main()
