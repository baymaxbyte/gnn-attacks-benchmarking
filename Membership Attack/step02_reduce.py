"""Step 02 -- reduce each dataset's V_in with every method x ratio.

Produces one reduced training graph per cell (control + GCond, FGC, UGC, GOREN,
KRON at their ratios). The reduction is done ONCE per cell here; step03 trains
three black boxes on it. GCond and GOREN shell out to their own interpreters, so
this runs serially, GCond last.

    python step02_reduce.py                       # everything
    python step02_reduce.py --skip gcond          # fast methods only
    python step02_reduce.py --only cora --methods kron,ugc
"""
import argparse
import os
import time

import common as K

os.environ.update(K.single_thread_env())

import numpy as np

import config as C
import data as D
from reductions import control, fgc, gcond, goren, kron, ugc

MODULES = {"control": control, "kron": kron, "ugc": ugc, "fgc": fgc,
           "goren": goren, "gcond": gcond}
ORDER = ["control", "kron", "ugc", "fgc", "goren", "gcond"]   # gcond last (slow)


def cell_tag(ds, method, r, v):
    if r is None:
        return f"{ds}_{method}_whole"
    return f"{ds}_{method}_r{r}" + (f"_{v}" if v else "")


def cells_for(method):
    mod = MODULES[method]
    variants = getattr(mod, "VARIANTS", [None])
    for r in mod.RATIOS:
        for v in variants:
            yield r, v


def reduced_path(tag):
    return os.path.join(C.REDUCED, f"{tag}.npz")


def save_reduced(path, rd, dataset, method, ratio, variant):
    ew = rd["edge_weight"]
    np.savez_compressed(
        path,
        edge_index=rd["edge_index"].astype(np.int64),
        edge_weight=(np.zeros(0, np.float32) if ew is None
                     else np.asarray(ew, np.float32)),
        weighted=np.array([0 if ew is None else 1]),
        X=rd["X"].astype(np.float32),
        Y=rd["Y"].astype(np.int64),
        train_mask=np.asarray(rd["train_mask"], bool),
        k=np.array([rd["k"]]), feat_dim=np.array([rd["feat_dim"]]),
        serve_mode=np.array(rd["serve_mode"]),
        serve_alpha=np.array([rd.get("serve_alpha", 0.0)], np.float32),
        dataset=np.array(dataset), method=np.array(method),
        ratio=np.array([-1.0 if ratio is None else ratio], np.float64),
        variant=np.array(variant or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="restrict to one dataset")
    ap.add_argument("--methods", default=",".join(ORDER))
    ap.add_argument("--skip", default="", help="comma list of methods to skip")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    K.ensure_dirs()
    methods = [m for m in ORDER if m in args.methods.split(",")
               and m not in args.skip.split(",")]
    datasets = [d for d in C.DATASETS if args.only in (None, d)]

    print(f"reducing V_in: methods={methods}  datasets={datasets}\n")
    rows = []
    t_all = time.time()
    for ds in datasets:
        sp_ = D.load_split(ds)
        vin_adj, vin_feat, vin_labels = sp_["vin_adj"], sp_["vin_feat"], sp_["vin_labels"]
        m = D.load_whole(ds)
        n_classes = m["n_classes"]
        print(f"=== {ds}  V_in {vin_adj.shape[0]} nodes ===")
        for method in methods:
            for r, v in cells_for(method):
                tag = cell_tag(ds, method, r, v)
                path = reduced_path(tag)
                if os.path.exists(path) and not args.force:
                    print(f"  [cached] {tag}")
                    continue
                t0 = time.time()
                rd = MODULES[method].reduce(ds, vin_adj, vin_feat, vin_labels,
                                            n_classes, r, variant=v)
                save_reduced(path, rd, ds, method, r, v)
                dt = time.time() - t0
                ew = rd["edge_weight"]
                ews = "unweighted" if ew is None else f"w[{ew.min():.3g},{ew.max():.3g}]"
                rows.append({"tag": tag, "k": rd["k"], "feat_dim": rd["feat_dim"],
                             "serve": rd["serve_mode"], "seconds": round(dt, 1)})
                print(f"  {tag:26s} k={rd['k']:5d} feat_dim={rd['feat_dim']:5d} "
                      f"E={rd['edge_index'].shape[1] // 2:6d} "
                      f"train={int(np.asarray(rd['train_mask']).sum()):5d} "
                      f"{ews:22s} {rd['serve_mode']:10s} [{dt:.1f}s]")
        print()

    if rows:
        K.record(
            "step02_reduce",
            summary=(f"Reduced V_in into {len(rows)} training graphs across "
                     f"{len(methods)} methods and {len(datasets)} datasets"),
            inputs=["artifacts/splits/*_split.npz"],
            outputs=[K.rel(reduced_path(r["tag"])) for r in rows],
            numbers={r["tag"]: r["k"] for r in rows},
            notes=("One reduced graph per cell; step03 trains 3 boxes on each. "
                   "control=V_in un-reduced; GCond condenses to synthetic nodes "
                   "(serve on row-normalised features); UGC augments features with "
                   "the V_in adjacency (serve augments to V_in width); GOREN has bl "
                   "and goren edge-weight variants."))
    print(f"total {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
