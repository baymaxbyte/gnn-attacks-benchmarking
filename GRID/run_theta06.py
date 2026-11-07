"""GRID at theta=0.6 on whole cora/citeseer (n=3, whole shadow). Reports attack
AUC (baseline vs GRID), and the BOX ACCURACY measured on the GRID-MODIFIED
posteriors (argmax vs true label on the benchmark's 60/20/20 test split) vs the
baseline posteriors -- to confirm/measure any utility change at theta=0.6.
"""
import os, sys, json, pickle, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ATTACK = os.path.join(HERE, "attack")
PRED = os.path.join(ATTACK, "data", "pred")
RES = os.path.join(HERE, "results", "theta06")
PY = sys.executable
sys.path.insert(0, HERE)
import grid_defense as G
from lsa.data import load_data

DIRS = {"cora": "citeseer", "citeseer": "cora"}
ATTACKS = [str(i) for i in range(8)]
THETA = 0.6
DP = os.path.join(ATTACK, "data", "dataset", "original") + os.sep


def make_split(labels, n_classes, seed=0):
    labels = np.asarray(labels); n = labels.shape[0]
    rng = np.random.RandomState(seed); test_idx = []
    for c in range(n_classes):
        idx = np.where(labels == c)[0]; idx = idx[rng.permutation(idx.shape[0])]; m = idx.shape[0]
        n_tr, n_va = int(round(0.6 * m)), int(round(0.2 * m))
        test_idx.append(idx[n_tr + n_va:])
    mask = np.zeros(n, dtype=bool); mask[np.concatenate(test_idx)] = True
    return mask


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
    boxacc = {}
    for ds in DIRS:
        _, _, _, _, y_test, *_ = load_data(DP, ds)
        labels = np.asarray(y_test).argmax(1); nC = np.asarray(y_test).shape[1]
        tmask = make_split(labels, nC, seed=0)
        V = G.load_pred(os.path.join(PRED, f"{ds}_kronbb_whole_pred.pkl"))
        _, nbrs = G.build_adjacency(ds)
        R, st = G.grid_posteriors(V, nbrs, dsname=f"{ds} theta0.6", theta=THETA, n_hop=3, verbose=False)
        accV = float((V.argmax(1)[tmask] == labels[tmask]).mean()) * 100
        accR = float((R.argmax(1)[tmask] == labels[tmask]).mean()) * 100
        lab_chg = int((V.argmax(1) != R.argmax(1)).sum())
        maxl1 = float(np.abs(R - V).sum(1).max())
        boxacc[ds] = dict(accV=accV, accR=accR, lab_chg=lab_chg, maxl1=maxl1,
                          adjb=st["adj_before"], adja=st["adj_after"], core=st["core"])
        pickle.dump(R, open(os.path.join(PRED, f"{ds}_kronbb_whole_t06_pred.pkl"), "wb"))
        build(ds, f"{ds}_kronbb_whole_pred.pkl", f"{ds}_kron_whole")
        build(ds, f"{ds}_kronbb_whole_t06_pred.pkl", f"{ds}_kron_whole_t06")

    res = {}
    for ds, shadow in DIRS.items():
        b = parse(attack(f"{ds}_kron_whole", f"{shadow}_kron_whole", f"{ds}_base"))
        g = parse(attack(f"{ds}_kron_whole_t06", f"{shadow}_kron_whole", f"{ds}_t06"))
        res[ds] = (b, g)

    print("\n=== BOX ACCURACY on modified posteriors (theta=0.6), 60/20/20 test split ===")
    for ds in DIRS:
        s = boxacc[ds]
        print(f"  {ds:9s} baseline_acc={s['accV']:.2f}%  GRID_modified_acc={s['accR']:.2f}%  "
              f"label_changes={s['lab_chg']}  max||s||1={s['maxl1']:.3f}  "
              f"adj_sim {s['adjb']:.3f}->{s['adja']:.3f}  core={s['core']}")
    for key in ["0", "1", "6"]:
        print(f"\n=== Attack-{key}: baseline -> GRID(theta=0.6) ===")
        for ds in DIRS:
            b, g = res[ds][0][key], res[ds][1][key]
            print(f"  {ds:9s} {b:.3f} -> {g:.3f}  ({g-b:+.3f})")
    csv = os.path.join(HERE, "results", "theta06.csv")
    with open(csv, "w") as f:
        f.write("dataset,attack,auc_base,auc_grid_t0.6,delta\n")
        for ds in DIRS:
            b, g = res[ds]
            for a in ATTACKS + ["link_pred"]:
                if a in b and a in g:
                    f.write(f"{ds},{a},{b[a]:.4f},{g[a]:.4f},{g[a]-b[a]:+.4f}\n")
    print("\nwrote", csv)


if __name__ == "__main__":
    main()
