#!/usr/bin/env python3
"""Pick MCC (alpha,beta,gamma) for new datasets by held-out VAL accuracy on the
ORIGINAL graph (same protocol as mcc_gridsearch -- no test peeking), from a small
candidate pool instead of the full 7x7x7 grid. Prints winners per (dataset, ratio)
to paste into coarsen_boxes.MCC_HP. Reuses mcc_gridsearch.fast_train so the scoring
box == the benchmark box (reduced epochs). Usage: python3 mcc_hp_select.py [datasets...]"""
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDROOT = os.path.dirname(os.path.dirname(HERE))
MCC = os.path.join(GRIDROOT, "MCC")
sys.path.insert(0, MCC)
import mcc_boxes as M
import mcc_gridsearch as GS

# known cora/citeseer winners + a few generic spreads
CANDS = [(0.1, 100., 100.), (10., 100., 0.1), (100., 1., 0.001), (100., 100., 0.01),
         (1., 1., 1.), (100., 100., 100.), (0.01, 100., 100.), (1., 100., 0.1)]
RATIOS = [0.3, 0.1]
EPOCHS, VAL_EVERY = 200, 5


def main():
    datasets = sys.argv[1:] or ["squirrel", "chameleon"]
    winners = {}
    for ds in datasets:
        A, X, labels, Cn = M.load_any(ds); n, d = A.shape[0], X.shape[1]
        mtr, mva, mte = M.split_6020(labels, Cn)
        Xo = torch.tensor(X); eio = M._ei(A); ewo = torch.ones(eio.shape[1])
        print(f"\n[{ds}] n={n} d={d} C={Cn} split {mtr.sum()}/{mva.sum()}/{mte.sum()}", flush=True)
        for r in RATIOS:
            best = None
            for (al, be, ga) in CANDS:
                try:
                    C, Ad = M.mcc_coarsen(A, r, al, be, ga)
                    Xc, eic, ewc, Yc, keep = M.build_coarse(C, Ad, X, labels, mtr, Cn)
                    ntr = int(keep.sum())
                    if ntr < 2:
                        print(f"  r={r} a={al} b={be} g={ga}: degenerate (train_sn={ntr})", flush=True); continue
                    bv, ta = GS.fast_train(Xc, eic, ewc, Yc, keep, Xo, eio, ewo, labels,
                                           mva, mte, d, Cn, EPOCHS, VAL_EVERY)
                    print(f"  r={r} a={al} b={be} g={ga}: val={bv:.3f} test={ta:.3f} (k={C.shape[1]}, train_sn={ntr})", flush=True)
                    if best is None or bv > best[0]:
                        best = (bv, ta, al, be, ga)
                except Exception as e:
                    print(f"  r={r} a={al} b={be} g={ga}: FAIL {type(e).__name__}: {e}", flush=True)
            if best:
                winners[(ds, r)] = (best[2], best[3], best[4])
                print(f"  >>> BEST ({ds}, {r}): ({best[2]}, {best[3]}, {best[4]})  val={best[0]:.3f} test={best[1]:.3f}", flush=True)
    print("\n# paste into MCC_HP:")
    for (ds, r), hp in winners.items():
        print(f'          ("{ds}", {r}): {hp},', flush=True)


if __name__ == "__main__":
    main()
