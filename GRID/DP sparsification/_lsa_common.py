#!/usr/bin/env python3
"""Shared LSA leakage runner for the DP-sparsification experiments (exp1/2/3).

Given target posteriors already saved in attack/data/pred/, this runs the He et al.
attacks (0-7 + link_pred), baseline vs GRID, directions A (cora target) + B (reverse),
and writes a CSV with a blackbox_acc column. Mirrors mcc_attack.py exactly, just
parametrised by the posterior file stems so each experiment can call it.

Conventions (consistent with the MCC / PrivGNN runs):
  * shadow = the OTHER dataset's box at the same setting (undefended)
  * reference g = the existing graph-free {ds}_gmcc reference (method-agnostic)
  * GRID uses the ORIGINAL graph adjacency (the defender defends the real links),
    theta=0.4 n_hop=3, argmax-preserving
  * attack ground-truth pairs come from the ORIGINAL graph (the private links)
"""
import os, sys, json, pickle, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))     # .../GRID/DP sparsification
GRIDROOT = os.path.dirname(HERE)                        # .../GRID
MCC = os.path.join(GRIDROOT, "MCC")
ATTACK = os.path.join(GRIDROOT, "attack")
PRED = os.path.join(ATTACK, "data", "pred")
PY = sys.executable
sys.path.insert(0, GRIDROOT); sys.path.insert(0, MCC)
import grid_defense as G
import mcc_boxes as M

DIRS = {"cora": "citeseer", "citeseer": "cora"}
ATT = [str(i) for i in range(8)]


def _sh(cmd):
    p = subprocess.run(cmd, cwd=ATTACK, capture_output=True, text=True)
    if p.returncode:
        sys.stderr.write(p.stdout[-1500:] + p.stderr[-1500:]); raise SystemExit("fail " + " ".join(cmd[:6]))
    return p.stdout


def grid_perturb(ds, stem):
    """Write {stem}_grid_pred.pkl from {stem}_pred.pkl via GRID on the ORIGINAL graph."""
    V = np.asarray(pickle.load(open(os.path.join(PRED, f"{stem}_pred.pkl"), "rb")), float)
    _, nbrs = G.build_adjacency(ds)
    R, st = G.grid_posteriors(V, nbrs, dsname=f"{stem}", theta=0.4, n_hop=3, verbose=False)
    pickle.dump(R, open(os.path.join(PRED, f"{stem}_grid_pred.pkl"), "wb"))
    print(f"  [grid] {stem} label_changes={st['label_changes']} adj {st['adj_before']:.3f}->{st['adj_after']:.3f}", flush=True)


def _pgg(ds, target_pred, ref_pred, name):
    _sh([PY, "partial_graph_generation.py", "--dataset", ds, "--datapath", "data/dataset/original/",
         "--prediction_path", "data/pred/", "--saving_path", "data/attack_dataset/",
         "--target_pred", target_pred, "--reference_pred", ref_pred, "--name", name, "--seed", "1"])


def _run_attack(target, shadow, tag, res_dir, runs):
    out = os.path.join(res_dir, f"{target}__{shadow}__{tag}.json")
    _sh([PY, "run_attack.py", "--attack", "all", "--dataset", target, "--shadow", shadow,
         "--attack_dataset_path", "data/attack_dataset/", "--result_path", res_dir, "--runs", str(runs),
         "--tag", tag, "--link_prediction", "--quiet"])
    return out


def _parse(p):
    d = json.load(open(p))
    o = {a: d["results"][a].get("auc_mean", d["results"][a].get("auc")) for a in ATT if a in d.get("results", {})}
    lp = d.get("link_prediction")
    if isinstance(lp, dict):
        o["link_pred"] = lp.get("auc_mean", lp.get("auc"))
    return o


_ACC = {}
def box_acc(ds, stem):
    if (ds, stem) not in _ACC:
        _, _, labels, Cn = M.load(ds)
        _, _, mte = M.split_6020(labels, Cn)
        post = np.asarray(pickle.load(open(os.path.join(PRED, f"{stem}_pred.pkl"), "rb")), float)
        _ACC[(ds, stem)] = float((post.argmax(1)[mte] == labels[mte]).mean())
    return _ACC[(ds, stem)]


def run_leakage(method, settings, target_stem, ref_stem, res_dir, runs=3, setting_col="setting", grid=True):
    """method: tag for attack_dataset naming/cleanup. settings: list of labels.
    target_stem(ds,s)->pkl stem (no _pred.pkl). ref_stem(ds)->reference g stem.
    grid=False -> baseline attack AUC only (no GRID defense)."""
    os.makedirs(res_dir, exist_ok=True)
    if grid:
        for ds in DIRS:
            for s in settings:
                grid_perturb(ds, target_stem(ds, s))
    for ds in DIRS:
        for s in settings:
            ts = target_stem(ds, s); rs = ref_stem(ds)
            _pgg(ds, f"{ts}_pred.pkl", f"{rs}_pred.pkl", f"{ds}_{method}_{s}")
            if grid:
                _pgg(ds, f"{ts}_grid_pred.pkl", f"{rs}_pred.pkl", f"{ds}_{method}_{s}_grid")
    res = {}
    for ds, shadow in DIRS.items():
        for s in settings:
            res[(ds, s, "base")] = _parse(_run_attack(f"{ds}_{method}_{s}", f"{shadow}_{method}_{s}", f"{ds}_{s}_base", res_dir, runs))
            if grid:
                res[(ds, s, "grid")] = _parse(_run_attack(f"{ds}_{method}_{s}_grid", f"{shadow}_{method}_{s}", f"{ds}_{s}_grid", res_dir, runs))
    names = {**{str(i): f"Attack-{i}" for i in range(8)}, "link_pred": "link_pred"}
    csv = os.path.join(res_dir, f"{method}_attack_auc.csv")
    with open(csv, "w") as f:
        if grid:
            f.write(f"dataset,{setting_col},blackbox_acc,attack,auc_no_defense,auc_grid,delta\n")
        else:
            f.write(f"dataset,{setting_col},blackbox_acc,attack,auc\n")
        for ds in DIRS:
            for s in settings:
                b = res[(ds, s, "base")]
                acc = box_acc(ds, target_stem(ds, s))
                if grid:
                    g = res[(ds, s, "grid")]
                    print(f"\n=== {ds} / {s} (box acc {acc*100:.1f}%; no-defense -> GRID) ===", flush=True)
                    for a in ATT + ["link_pred"]:
                        if a in b and a in g:
                            print(f"  {names[a]:10s} {b[a]:.3f} -> {g[a]:.3f}  ({g[a]-b[a]:+.3f})", flush=True)
                            f.write(f"{ds},{s},{acc:.4f},{names[a]},{b[a]:.4f},{g[a]:.4f},{g[a]-b[a]:+.4f}\n")
                else:
                    print(f"\n=== {ds} / {s} (box acc {acc*100:.1f}%; baseline AUC, no defense) ===", flush=True)
                    for a in ATT + ["link_pred"]:
                        if a in b:
                            print(f"  {names[a]:10s} {b[a]:.3f}", flush=True)
                            f.write(f"{ds},{s},{acc:.4f},{names[a]},{b[a]:.4f}\n")
    ADS = os.path.join(ATTACK, "data", "attack_dataset")
    for fn in os.listdir(ADS):
        if f"_{method}_" in fn:
            os.remove(os.path.join(ADS, fn))
    print(f"\nwrote {csv}", flush=True)
    return csv
