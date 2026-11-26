#!/usr/bin/env python3
"""Grid-search MCC coarsening hyperparameters (alpha,beta,gamma) for a given
(dataset, ratio), scoring by our KRON-consistent box's accuracy on the ORIGINAL
graph. Mirrors the author's 7x7x7 grid: alpha,beta,gamma in
{100,10,1,0.1,0.01,0.001,0.0001} (gamma = C-update step size, per multi_comp.py).

Selection is by VALIDATION accuracy (test is reported but NOT used to pick), so
no test-set peeking. The winner is re-built at full 500 epochs by mcc_boxes.py for
the actual attack benchmark. Fast ranking here uses reduced epochs + periodic
validation; coarsening (the part alpha/beta/gamma control) is run in full.

Reuses mcc_boxes.py verbatim (coarsen + box + split), so the search box == the
benchmark box, just fewer epochs. Writes results/gridsearch/{ds}_{setting}_grid.csv
incrementally and prints the top settings by held-out validation accuracy.
"""
import os, sys, argparse, itertools, time, csv
import numpy as np
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mcc_boxes as M

GRID = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]


def fast_train(xtr, eitr, ewtr, ytr, mtr, Xo, eio, ewo, labels, val_mask, test_mask,
               d, C, epochs, val_every, seed=0):
    """Same as mcc_boxes.train_box but fewer epochs + validate every val_every
    (the per-epoch original-graph forward is the cost). Returns (best_val, test_acc)."""
    torch.manual_seed(seed); model = M.GCN(d, C)
    opt = torch.optim.Adam(model.parameters(), lr=M.LR, weight_decay=M.WD)
    yt = torch.from_numpy(labels); vm = torch.from_numpy(val_mask)
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        Fn.nll_loss(model(xtr, eitr, ewtr)[mtr], ytr[mtr]).backward(); opt.step()
        if ep % val_every == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                va = float((model(Xo, eio, ewo).argmax(1)[vm] == yt[vm]).float().mean())
            if va > best:
                best, best_state = va, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred = model(Xo, eio, ewo).argmax(1).numpy()
    return best, float((pred[test_mask] == labels[test_mask]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--setting", required=True, help="coarsening ratio, e.g. 0.1")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="only first N combos (timing)")
    a = ap.parse_args()

    A, X, labels, Cn = M.load(a.dataset); n, d = A.shape[0], X.shape[1]
    mtr, mva, mte = M.split_6020(labels, Cn)
    Xo = torch.tensor(X); eio = M._ei(A); ewo = torch.ones(eio.shape[1])
    r = float(a.setting)
    combos = list(itertools.product(GRID, GRID, GRID))   # (alpha, beta, gamma)
    if a.limit:
        combos = combos[:a.limit]

    outdir = os.path.join(HERE, "results", "gridsearch"); os.makedirs(outdir, exist_ok=True)
    outcsv = os.path.join(outdir, f"{a.dataset}_{a.setting}_grid.csv")
    print(f"[{a.dataset}] n={n} d={d} C={Cn} r={r} k~={int(n*r)}  split {mtr.sum()}/{mva.sum()}/{mte.sum()}"
          f"  combos={len(combos)}  epochs={a.epochs} val_every={a.val_every}", flush=True)

    rows = []
    t0 = time.time()
    f = open(outcsv, "w", newline=""); w = csv.writer(f)
    w.writerow(["alpha", "beta", "gamma", "k", "train_supernodes", "best_val", "test_acc", "status"]); f.flush()
    for i, (al, be, ga) in enumerate(combos):
        try:
            C, Ad = M.mcc_coarsen(A, r, al, be, ga)
            Xc, eic, ewc, Yc, keep = M.build_coarse(C, Ad, X, labels, mtr, Cn)
            ntr = int(keep.sum())
            if ntr < 2:
                row = (al, be, ga, int(C.shape[1]), ntr, "", "", "degenerate")
            else:
                bv, ta = fast_train(Xc, eic, ewc, Yc, keep, Xo, eio, ewo, labels,
                                    mva, mte, d, Cn, a.epochs, a.val_every)
                row = (al, be, ga, int(C.shape[1]), ntr, round(bv, 4), round(ta, 4), "ok")
        except Exception as e:
            row = (al, be, ga, -1, -1, "", "", f"fail:{type(e).__name__}")
        rows.append(row); w.writerow(row); f.flush()
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(combos)}  elapsed={el:.0f}s  eta={el/(i+1)*(len(combos)-i-1):.0f}s", flush=True)
    f.close()

    ok = [x for x in rows if x[7] == "ok"]
    ok.sort(key=lambda x: (x[5], x[6]), reverse=True)   # by best_val, then test_acc
    print(f"\n=== {a.dataset} / {a.setting}: top 10 by held-out VAL acc ({len(ok)}/{len(rows)} ok) ===", flush=True)
    for x in ok[:10]:
        print(f"  alpha={x[0]:<7} beta={x[1]:<7} gamma={x[2]:<7}  k={x[3]} train_sn={x[4]}  "
              f"val={x[5]:.3f} test={x[6]:.3f}", flush=True)
    if ok:
        b = ok[0]
        print(f"\nBEST {a.dataset}/{a.setting}: alpha={b[0]} beta={b[1]} gamma={b[2]}  "
              f"val={b[5]:.3f} test={b[6]:.3f}  (k={b[3]}, train_supernodes={b[4]})", flush=True)
    print(f"wrote {outcsv}", flush=True)


if __name__ == "__main__":
    main()
