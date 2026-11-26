#!/usr/bin/env python3
"""Finalize CiteSeer MCC hyperparameters: take the top candidates from the
200-epoch grid (dedup'd) PLUS the incumbent borrowed value, re-validate each at
the FULL 500-epoch box protocol (mcc_boxes.py verbatim), select by held-out
VALIDATION accuracy, and rebuild + save the winning box posteriors
(citeseer_mccbb_{setting}_pred.pkl). Prints the comparison table and winners."""
import os, sys, csv
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mcc_boxes as M

GRIDDIR = os.path.join(HERE, "results", "gridsearch")
SETTINGS = ["0.1", "0.3"]
INCUMBENT = {"0.1": (10.0, 100.0, 0.1), "0.3": (0.1, 100.0, 100.0)}   # borrowed cora values
CAND_N = 6        # distinct coarsenings from the grid to re-validate at full epochs
DS = "citeseer"


def top_candidates(setting):
    fn = os.path.join(GRIDDIR, f"{DS}_{setting}_grid.csv")
    rows = [r for r in csv.DictReader(open(fn)) if r["status"] == "ok" and r["best_val"]]
    rows.sort(key=lambda r: (float(r["best_val"]), float(r["test_acc"])), reverse=True)
    out, seen = [], set()
    for r in rows:
        sig = (r["train_supernodes"], round(float(r["best_val"]), 4), round(float(r["test_acc"]), 4))
        if sig in seen:
            continue
        seen.add(sig)
        out.append((float(r["alpha"]), float(r["beta"]), float(r["gamma"])))
        if len(out) >= CAND_N:
            break
    inc = INCUMBENT[setting]
    if inc not in out:
        out.append(inc)
    return out


def main():
    A, X, labels, Cn = M.load(DS)
    n, d = A.shape[0], X.shape[1]
    mtr, mva, mte = M.split_6020(labels, Cn)
    Xo = torch.tensor(X); eio = M._ei(A); ewo = torch.ones(eio.shape[1]); yt = torch.from_numpy(labels)
    print(f"[{DS}] n={n} d={d} C={Cn}  split {mtr.sum()}/{mva.sum()}/{mte.sum()}  FULL epochs={M.EPOCHS}")

    winners = {}
    for s in SETTINGS:
        r = float(s)
        cands = top_candidates(s)
        inc = INCUMBENT[s]
        print(f"\n=== {DS}/{s}: re-validating {len(cands)} candidates at {M.EPOCHS} epochs ===", flush=True)
        results = []
        for (al, be, ga) in cands:
            C, Ad = M.mcc_coarsen(A, r, al, be, ga)
            Xc, eic, ewc, Yc, keep = M.build_coarse(C, Ad, X, labels, mtr, Cn)
            ntr = int(keep.sum())
            if ntr < 2:
                print(f"  a={al:<6} b={be:<6} g={ga:<7} DEGENERATE (train_sn={ntr})", flush=True)
                continue
            post, bv = M.train_box(Xc, eic, ewc, Yc, keep, Xo, eio, ewo, labels, mva, d, Cn)
            ta = float((post.argmax(1)[mte] == labels[mte]).mean())
            tag = "  <-- incumbent" if (al, be, ga) == inc else ""
            print(f"  a={al:<6} b={be:<6} g={ga:<7} val={bv:.4f} test={ta:.4f} train_sn={ntr} k={C.shape[1]}{tag}", flush=True)
            results.append((bv, ta, al, be, ga, ntr, C.shape[1], post))
        results.sort(key=lambda x: (x[0], x[1]), reverse=True)
        bv, ta, al, be, ga, ntr, k, post = results[0]
        winners[s] = (al, be, ga, bv, ta, ntr, k)
        print(f"  WINNER {DS}/{s}: alpha={al} beta={be} gamma={ga}  val={bv:.4f} test={ta:.4f} "
              f"(train_sn={ntr}, k={k})", flush=True)
        M.save(f"{DS}_mccbb_{s}", post)   # overwrite the box posteriors used by the attack

    print("\n================ CiteSeer winners (full-epoch, val-selected) ================")
    for s in SETTINGS:
        al, be, ga, bv, ta, ntr, k = winners[s]
        iv = INCUMBENT[s]
        print(f"  {s}: alpha={al} beta={be} gamma={ga}  val={bv:.3f} test={ta:.3f}  "
              f"(was borrowed {iv}, train_sn={ntr}/{k})")


if __name__ == "__main__":
    main()
