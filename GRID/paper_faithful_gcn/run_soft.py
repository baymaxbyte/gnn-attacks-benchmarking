"""Run the link-stealing attacks on the paper-faithful SOFT GCN (whole cora /
citeseer), baseline vs GRID (theta=0.4, n=3), whole-dataset shadow. Tests
whether de-saturating the target reproduces the paper's large AUC drops.
"""
import os, sys, json, pickle, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDROOT = os.path.dirname(HERE)
ATTACK = os.path.join(HERE, "attack")
PRED = os.path.join(ATTACK, "data", "pred")
RES = os.path.join(HERE, "results", "attacks")
PY = sys.executable
sys.path.insert(0, GRIDROOT)
import grid_defense as G
import train_soft_gcn as T

DIRS = {"cora": "citeseer", "citeseer": "cora"}
ATTACKS = [str(i) for i in range(8)]


def sh(cmd):
    p = subprocess.run(cmd, cwd=ATTACK, capture_output=True, text=True)
    if p.returncode:
        sys.stderr.write(p.stdout[-1500:] + "\n" + p.stderr[-1500:] + "\n")
        raise SystemExit("cmd failed: " + " ".join(cmd))
    return p.stdout


def build(ds, target_pred, name):
    sh([PY, "partial_graph_generation.py", "--dataset", ds,
        "--datapath", "data/dataset/original/", "--prediction_path", "data/pred/",
        "--saving_path", "data/attack_dataset/", "--target_pred", target_pred,
        "--reference_pred", f"{ds}_gkron_pred.pkl", "--name", name, "--seed", "1"])


def attack(target, shadow, tag):
    out = os.path.join(RES, f"{target}__{shadow}__{tag}.json")
    sh([PY, "run_attack.py", "--attack", "all", "--dataset", target, "--shadow", shadow,
        "--attack_dataset_path", "data/attack_dataset/", "--result_path", RES,
        "--runs", "5", "--tag", tag, "--link_prediction", "--quiet"])
    return out


def auc(e):
    return e.get("auc_mean", e.get("auc"))


def parse(p):
    d = json.load(open(p))
    out = {a: auc(d["results"][a]) for a in ATTACKS if a in d.get("results", {})}
    lp = d.get("link_prediction")
    if isinstance(lp, dict):
        out["link_pred"] = auc(lp)
    return out


def main():
    os.makedirs(RES, exist_ok=True)
    softstats = {}
    # 1) soft posteriors (train if missing) + GRID-perturbed version
    for ds in DIRS:
        p = os.path.join(PRED, f"{ds}_softgcn_whole_pred.pkl")
        if not os.path.exists(p):
            _, softstats[ds] = T.train(ds, verbose=True)
        V = G.load_pred(p)
        _, nbrs = G.build_adjacency(ds)
        R, st = G.grid_posteriors(V, nbrs, dsname=f"soft:{ds}", theta=0.4, n_hop=3, verbose=True)
        with open(os.path.join(PRED, f"{ds}_softgcn_whole_grid_pred.pkl"), "wb") as f:
            pickle.dump(R, f)
        build(ds, f"{ds}_softgcn_whole_pred.pkl", f"{ds}_soft_whole")
        build(ds, f"{ds}_softgcn_whole_grid_pred.pkl", f"{ds}_soft_whole_grid")

    # 2) attacks: baseline vs GRID, whole-shadow (the other dataset's soft box)
    res = {}
    for ds, shadow in DIRS.items():
        b = attack(f"{ds}_soft_whole", f"{shadow}_soft_whole", f"{ds}_base")
        g = attack(f"{ds}_soft_whole_grid", f"{shadow}_soft_whole", f"{ds}_grid")
        res[ds] = (parse(b), parse(g))

    # 3) table
    names = {**{str(i): f"Attack-{i}" for i in range(8)}, "link_pred": "link_pred"}
    order = ATTACKS + ["link_pred"]
    csv = os.path.join(HERE, "results", "soft_gcn_grid.csv")
    with open(csv, "w") as f:
        f.write("dataset,attack,auc_no_defense,auc_grid,delta\n")
        for ds in DIRS:
            base, grid = res[ds]
            print(f"\n=== {ds} (soft GCN, no-defense -> GRID theta=0.4) ===")
            print(f"{'attack':10s} {'no_def':>7s} {'GRID':>7s} {'delta':>7s}")
            for a in order:
                if a not in base or a not in grid:
                    continue
                bb, gg = base[a], grid[a]
                print(f"{names[a]:10s} {bb:7.3f} {gg:7.3f} {gg-bb:+7.3f}")
                f.write(f"{ds},{names[a]},{bb:.4f},{gg:.4f},{gg-bb:+.4f}\n")
    print("\nwrote", os.path.relpath(csv, HERE))


if __name__ == "__main__":
    main()
