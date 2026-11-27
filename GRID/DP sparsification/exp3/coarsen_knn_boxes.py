#!/usr/bin/env python3
"""EXP3 (extended): replace the real adjacency with the feature-KNN(k=10) graph,
then coarsen THAT graph with MCC/FGC/KRON/UGC at each method's native ratios, train
the consistent 3-layer GCN on the coarse graph, serve on the KNN graph. Attack
ground-truth = ORIGINAL edges (GOREN/GCond handled in separate scripts).

Reuses exp1's coarse_box() verbatim, just passing the KNN graph as the base
adjacency instead of a DP-sparsified graph. Posteriors ->
attack/data/pred/{ds}_exp3{method}_{r}_pred.pkl
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DPS = os.path.dirname(HERE)                      # DP sparsification
EXP1 = os.path.join(DPS, "exp1")
GRIDROOT = os.path.dirname(DPS)
MCC = os.path.join(GRIDROOT, "MCC")
sys.path.insert(0, MCC); import mcc_boxes as M
sys.path.insert(0, HERE); from knn_boxes import knn_graph
sys.path.insert(0, EXP1); import coarsen_boxes as CB

RATIOS = CB.RATIOS   # {"mcc":[0.3,0.1], "fgc":[0.3,0.1,0.05], "kron":[0.3,0.5,0.7], "ugc":[0.3,0.5,0.7]}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    ap.add_argument("--methods", nargs="+", default=["mcc", "fgc", "kron", "ugc"])
    a = ap.parse_args()
    for ds in a.datasets:
        A, X, y, C = M.load_any(ds); mtr, mva, mte = M.split_6020(y, C)
        Aknn = knn_graph(X)
        print(f"\n[{ds}] KNN base: n={A.shape[0]} knn_edges={int(Aknn.nnz//2)}", flush=True)
        for method in a.methods:
            for r in RATIOS[method]:
                try:
                    post, acc, k, ksn, ce = CB.coarse_box(ds, method, r, Aknn, X, y, C, mtr, mva, mte)
                    M.save(f"{ds}_exp3{method}_{r}", post)
                    print(f"  {method:4s} r={r}: k={k} train_sn={ksn} coarse_edges={ce} acc={acc*100:.1f}%", flush=True)
                except Exception as e:
                    print(f"  {method:4s} r={r}: FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
