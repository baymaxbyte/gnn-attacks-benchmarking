#!/usr/bin/env python3
"""MCC leakage: run the He et al. link-stealing attacks on the MCC black boxes,
baseline vs GRID defence, exactly as the other coarsening benchmarks + run_grid_all.

For each dataset x setting {whole,0.1,0.3}:
  * baseline: partial_graph_generation (target=mccbb, reference=gmcc, pair seed 1)
    then run_attack (attacks 0-7 + link-pred), directions A (cora target / citeseer
    shadow) and B (reverse), same setting -- step03/step04 convention.
  * GRID: perturb the served posteriors (grid_defense, theta=0.4 n=3, original-graph
    adjacency, argmax-preserving), attack again; shadow stays undefended.
Outputs results/attacks/mcc_attack_auc.csv.
"""
import os, sys, json, pickle, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDROOT = os.path.dirname(HERE)
ATTACK = os.path.join(GRIDROOT, "attack")
PRED = os.path.join(ATTACK, "data", "pred")
RES = os.path.join(HERE, "results", "attacks")
PY = sys.executable
sys.path.insert(0, GRIDROOT)
import grid_defense as G
os.makedirs(RES, exist_ok=True)

DIRS = {"cora": "citeseer", "citeseer": "cora"}
SETTINGS = ["whole", "0.1", "0.3"]
ATT = [str(i) for i in range(8)]
RUNS = 3


def sh(cmd):
    p = subprocess.run(cmd, cwd=ATTACK, capture_output=True, text=True)
    if p.returncode:
        sys.stderr.write(p.stdout[-1500:] + p.stderr[-1500:]); raise SystemExit("fail " + " ".join(cmd[:6]))
    return p.stdout


def grid_perturb(ds, setting):
    out = os.path.join(PRED, f"{ds}_mccbb_{setting}_grid_pred.pkl")
    V = np.asarray(pickle.load(open(os.path.join(PRED, f"{ds}_mccbb_{setting}_pred.pkl"), "rb")), float)
    _, nbrs = G.build_adjacency(ds)
    R, st = G.grid_posteriors(V, nbrs, dsname=f"mcc:{ds}/{setting}", theta=0.4, n_hop=3, verbose=False)
    pickle.dump(R, open(out, "wb"))
    print(f"  [grid] {ds}/{setting} label_changes={st['label_changes']} adj {st['adj_before']:.3f}->{st['adj_after']:.3f}", flush=True)


def pgg(ds, target_pred, name):
    sh([PY, "partial_graph_generation.py", "--dataset", ds, "--datapath", "data/dataset/original/",
        "--prediction_path", "data/pred/", "--saving_path", "data/attack_dataset/",
        "--target_pred", target_pred, "--reference_pred", f"{ds}_gmcc_pred.pkl", "--name", name, "--seed", "1"])


def run_attack(target, shadow, tag):
    out = os.path.join(RES, f"{target}__{shadow}__{tag}.json")
    sh([PY, "run_attack.py", "--attack", "all", "--dataset", target, "--shadow", shadow,
        "--attack_dataset_path", "data/attack_dataset/", "--result_path", RES, "--runs", str(RUNS),
        "--tag", tag, "--link_prediction", "--quiet"])
    return out


def parse(p):
    d = json.load(open(p))
    o = {a: d["results"][a].get("auc_mean", d["results"][a].get("auc")) for a in ATT if a in d.get("results", {})}
    lp = d.get("link_prediction")
    if isinstance(lp, dict):
        o["link_pred"] = lp.get("auc_mean", lp.get("auc"))
    return o


_ACC = {}
def box_acc(ds, setting):
    """Held-out test accuracy of the MCC box from its saved posteriors (same
    split_6020 seed-0 split). GRID preserves argmax, so base and GRID share it."""
    if (ds, setting) not in _ACC:
        import mcc_boxes as M
        _, _, labels, Cn = M.load(ds)
        _, _, mte = M.split_6020(labels, Cn)
        post = np.asarray(pickle.load(open(os.path.join(PRED, f"{ds}_mccbb_{setting}_pred.pkl"), "rb")), float)
        _ACC[(ds, setting)] = float((post.argmax(1)[mte] == labels[mte]).mean())
    return _ACC[(ds, setting)]


def main():
    for ds in DIRS:
        for s in SETTINGS:
            grid_perturb(ds, s)
    for ds in DIRS:
        for s in SETTINGS:
            pgg(ds, f"{ds}_mccbb_{s}_pred.pkl", f"{ds}_mcc_{s}")
            pgg(ds, f"{ds}_mccbb_{s}_grid_pred.pkl", f"{ds}_mcc_{s}_grid")

    res = {}
    for ds, shadow in DIRS.items():
        for s in SETTINGS:
            res[(ds, s, "base")] = parse(run_attack(f"{ds}_mcc_{s}", f"{shadow}_mcc_{s}", f"{ds}_{s}_base"))
            res[(ds, s, "grid")] = parse(run_attack(f"{ds}_mcc_{s}_grid", f"{shadow}_mcc_{s}", f"{ds}_{s}_grid"))

    names = {**{str(i): f"Attack-{i}" for i in range(8)}, "link_pred": "link_pred"}
    csv = os.path.join(RES, "mcc_attack_auc.csv")
    with open(csv, "w") as f:
        f.write("dataset,setting,blackbox_acc,attack,auc_no_defense,auc_grid,delta\n")
        for ds in DIRS:
            for s in SETTINGS:
                b, g = res[(ds, s, "base")], res[(ds, s, "grid")]
                acc = box_acc(ds, s)
                print(f"\n=== {ds} / {s} (box acc {acc*100:.1f}%; no-defense -> GRID) ===", flush=True)
                for a in ATT + ["link_pred"]:
                    if a in b and a in g:
                        print(f"  {names[a]:10s} {b[a]:.3f} -> {g[a]:.3f}  ({g[a]-b[a]:+.3f})", flush=True)
                        f.write(f"{ds},{s},{acc:.4f},{names[a]},{b[a]:.4f},{g[a]:.4f},{g[a]-b[a]:+.4f}\n")
    # cleanup bulky attack datasets
    ADS = os.path.join(ATTACK, "data", "attack_dataset")
    for fn in os.listdir(ADS):
        if "_mcc_" in fn:
            os.remove(os.path.join(ADS, fn))
    print(f"\nwrote {csv}", flush=True)


if __name__ == "__main__":
    main()
