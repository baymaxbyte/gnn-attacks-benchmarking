#!/usr/bin/env python3
"""Turn the Fisher-Rao geodesic move into an actual DEFENSE and measure its
effect on the 3 paper attacks (Attack-0/1/6), vs no-defense and vs GRID, on the
same soft-GCN target and identical evaluation (random-unlinked negatives, 5 runs).

FR defense = for each GRID core node, descend the hinge objective
max(sim_k - delta, 0) over its neighbours along the sqrt(p)-sphere GEODESIC,
preserving argmax (zero label change) -- the Fisher-Rao analogue of GRID's
L1-budgeted move, with no theta cap (it stops at the n-hop floor delta).
"""
import os, sys, json, pickle, subprocess
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDROOT = os.path.dirname(HERE)
ATTACK = os.path.join(HERE, "attack")
PRED = os.path.join(ATTACK, "data", "pred")
ADS = os.path.join(ATTACK, "data", "attack_dataset", "")
RES = os.path.join(HERE, "results", "fr_defense")
PY = sys.executable
sys.path.insert(0, GRIDROOT)
import grid_defense as G
torch.set_default_dtype(torch.float64)
os.makedirs(RES, exist_ok=True)

DIRS = {"cora": "citeseer", "citeseer": "cora"}
ATT = ["0", "1", "6"]
EPS = 1e-12


def fr_node(p, P, delta, steps=100, dt=0.05):
    c = int(np.argmax(p))
    Pt = torch.tensor(P)
    u = np.sqrt(np.clip(p, 0, None)); u = u / (np.linalg.norm(u) + EPS)
    for _ in range(steps):
        ut = torch.tensor(u, requires_grad=True)
        q = ut * ut; q = q / q.sum()
        loss = torch.clamp(G._sim_rows(q, Pt) - delta, min=0).sum()
        if float(loss) < 1e-7:
            break
        g, = torch.autograd.grad(loss, ut)
        g = g.detach().numpy(); g = g - (g @ u) * u
        ng = np.linalg.norm(g)
        if ng < 1e-9:
            break
        u = np.cos(dt) * u + np.sin(dt) * (-g / ng)
        u = np.clip(u, 0, None); u = u / (np.linalg.norm(u) + EPS)
        q = u * u; q = q / q.sum()
        if int(np.argmax(q)) != c:
            q = q.copy(); q[c] = q.max() * (1 + 1e-6); q = q / q.sum(); u = np.sqrt(q)
    q = u * u
    return q / q.sum()


def fr_defend(ds):
    V = np.asarray(pickle.load(open(os.path.join(PRED, f"{ds}_softgcn_whole_pred.pkl"), "rb")), float)
    V = V / V.sum(1, keepdims=True)
    _, nbrs = G.build_adjacency(ds)
    delta, _ = G.estimate_delta(V, nbrs, 3)
    core, _, _ = G.select_core_nodes(V, nbrs, delta)
    R = V.copy()
    for k, i in enumerate(core):
        P = V[[int(j) for j in nbrs[i]]]
        if len(P):
            R[i] = fr_node(V[i], P, delta)
        if (k + 1) % 400 == 0:
            print(f"    [{ds}] FR-defended {k+1}/{len(core)}", flush=True)
    lc = int((V.argmax(1) != R.argmax(1)).sum())
    print(f"  [{ds}] core={len(core)} delta={delta:.3f} label_changes={lc} (must be 0)", flush=True)
    pickle.dump(R, open(os.path.join(PRED, f"{ds}_softgcn_whole_frdef_pred.pkl"), "wb"))
    return lc


def sh(cmd):
    p = subprocess.run(cmd, cwd=ATTACK, capture_output=True, text=True)
    if p.returncode:
        sys.stderr.write(p.stdout[-1500:] + p.stderr[-1500:]); raise SystemExit("fail " + " ".join(cmd))


def build(ds, pred, name):
    sh([PY, "partial_graph_generation.py", "--dataset", ds, "--datapath", "data/dataset/original/",
        "--prediction_path", "data/pred/", "--saving_path", "data/attack_dataset/",
        "--target_pred", pred, "--reference_pred", f"{ds}_gkron_pred.pkl", "--name", name, "--seed", "1"])


def run_attack(target, shadow, tag):
    out = os.path.join(RES, f"{target}__{shadow}__{tag}.json")
    sh([PY, "run_attack.py", "--attack", "0,1,6", "--dataset", target, "--shadow", shadow,
        "--attack_dataset_path", "data/attack_dataset/", "--result_path", RES,
        "--runs", "5", "--tag", tag, "--link_prediction", "--quiet"])
    return out


def parse(p):
    d = json.load(open(p))
    out = {a: d["results"][a].get("auc_mean", d["results"][a].get("auc")) for a in ATT if a in d.get("results", {})}
    lp = d.get("link_prediction")
    if isinstance(lp, dict):
        out["link_pred"] = lp.get("auc_mean", lp.get("auc"))
    return out


def main():
    for ds in DIRS:
        fr_defend(ds)
    # build attack datasets: baseline (= no defense = shadow), GRID, FR
    for ds in DIRS:
        build(ds, f"{ds}_softgcn_whole_pred.pkl", f"{ds}_sb")
        build(ds, f"{ds}_softgcn_whole_grid_pred.pkl", f"{ds}_sg")
        build(ds, f"{ds}_softgcn_whole_frdef_pred.pkl", f"{ds}_sf")
    res = {}
    for ds, shadow in DIRS.items():
        nod = run_attack(f"{ds}_sb", f"{shadow}_sb", f"{ds}_nodef")
        grd = run_attack(f"{ds}_sg", f"{shadow}_sb", f"{ds}_grid")
        frd = run_attack(f"{ds}_sf", f"{shadow}_sb", f"{ds}_fr")
        res[ds] = dict(nodef=parse(nod), grid=parse(grd), fr=parse(frd))
    names = {"0": "Attack-0", "1": "Attack-1", "6": "Attack-6", "link_pred": "link_pred"}
    csv = os.path.join(RES, "fr_defense_auc.csv")
    with open(csv, "w") as f:
        f.write("dataset,attack,no_defense,grid,fisher_rao,delta_grid,delta_fr\n")
        for ds in DIRS:
            r = res[ds]
            print(f"\n=== {ds} (soft GCN; no-def -> GRID / Fisher-Rao) ===", flush=True)
            for a in ATT + ["link_pred"]:
                if a in r["nodef"]:
                    nd, gr, fr = r["nodef"][a], r["grid"].get(a), r["fr"].get(a)
                    print(f"  {names[a]:10s} nodef={nd:.3f}  GRID={gr:.3f}({gr-nd:+.3f})  FR={fr:.3f}({fr-nd:+.3f})", flush=True)
                    f.write(f"{ds},{names[a]},{nd:.4f},{gr:.4f},{fr:.4f},{gr-nd:+.4f},{fr-nd:+.4f}\n")
    # cleanup big attack datasets
    for fn in os.listdir(ADS):
        if any(t in fn for t in ("_sb_attack_", "_sg_attack_", "_sf_attack_")):
            os.remove(os.path.join(ADS, fn))
    print(f"\nwrote {csv}", flush=True)


if __name__ == "__main__":
    main()
