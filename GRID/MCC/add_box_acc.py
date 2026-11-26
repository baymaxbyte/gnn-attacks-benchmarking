#!/usr/bin/env python3
"""Add a blackbox_acc column to results/attacks/mcc_attack_auc.csv -- the MCC
box's held-out TEST accuracy, recomputed exactly from the saved box posteriors
(attack/data/pred/{ds}_mccbb_{setting}_pred.pkl) using the same split_6020 seed-0
split and Planetoid node order. GRID preserves argmax (label_changes=0) so the
accuracy is identical for base and GRID -> one value per (dataset, setting)."""
import os, sys, csv, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mcc_boxes as M

CSV = os.path.join(HERE, "results", "attacks", "mcc_attack_auc.csv")

_ds_cache = {}
def _split(ds):
    if ds not in _ds_cache:
        A, X, labels, Cn = M.load(ds)
        _, _, mte = M.split_6020(labels, Cn)
        _ds_cache[ds] = (labels, mte)
    return _ds_cache[ds]

_acc_cache = {}
def box_acc(ds, setting):
    key = (ds, setting)
    if key not in _acc_cache:
        labels, mte = _split(ds)
        post = np.asarray(pickle.load(open(os.path.join(M.PRED, f"{ds}_mccbb_{setting}_pred.pkl"), "rb")))
        _acc_cache[key] = float((post.argmax(1)[mte] == labels[mte]).mean())
    return _acc_cache[key]


def main():
    rows = list(csv.DictReader(open(CSV)))
    fields = ["dataset", "setting", "blackbox_acc", "attack", "auc_no_defense", "auc_grid", "delta"]
    for r in rows:
        r["blackbox_acc"] = f"{box_acc(r['dataset'], r['setting']):.4f}"
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print("MCC box held-out test accuracy:")
    for (ds, s), a in sorted(_acc_cache.items()):
        print(f"  {ds:9s} {s:6s} {a*100:5.1f}%")
    print(f"\nrewrote {CSV} with blackbox_acc column")


if __name__ == "__main__":
    main()
