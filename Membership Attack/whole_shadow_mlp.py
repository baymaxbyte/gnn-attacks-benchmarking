"""Add a WHOLE-GRAPH (uncoarsened) shadow variant for the membership shadow_mlp attack
ONLY -- the one membership attack that uses a shadow -- across all methods, both datasets.

The benchmark's coarsened shadow for a target cell <ds>_<method>_<setting> is the OTHER
dataset at the SAME reduction cell (<other>_<method>_<setting>). This recomputes the same
sklearn shadow MLP but with the shadow = the other dataset's CONTROL cell
(<other>_control_whole) -- i.e. a GNN trained on the uncoarsened shadow graph. It reuses
the shipped posteriors (no re-coarsening, no retraining): only which posterior file feeds
the shadow changes. Non-destructive: writes a new results/<ds>_membership_shadow_compare.csv;
nothing existing is modified.

    python whole_shadow_mlp.py
"""
import csv
import os

import numpy as np

import config as C
import data as D
from attacks import features as FE
from attacks import shadow_clf

METHOD_ORDER = ["control", "kron", "ugc", "fgc", "goren", "gcond"]


def parse(tag):
    p = tag.split("_")
    ds, method, rest = p[0], p[1], p[2:]
    if rest == ["whole"]:
        return ds, method, "", ""
    ratio = rest[0][1:] if rest and rest[0].startswith("r") else ""
    variant = "_".join(rest[1:]) if len(rest) > 1 else ""
    return ds, method, ratio, variant


def _post(tag, seed):
    p = os.path.join(C.POSTERIORS, f"{tag}_seed{seed}.npy")
    return np.load(p) if os.path.exists(p) else None


def _ms(a):
    return (round(float(np.mean(a)), 4), round(float(np.std(a)), 4)) if a else ("", "")


def main():
    splits = {ds: D.load_split(ds) for ds in C.DATASETS}
    labels = {ds: D.load_whole(ds)["labels"] for ds in C.DATASETS}
    mask = {ds: splits[ds]["member_mask"] for ds in C.DATASETS}

    tags = sorted({f.rsplit("_seed", 1)[0] for f in os.listdir(C.POSTERIORS)
                   if "_seed" in f})
    by_ds = {}
    for tag in tags:
        ds, method, ratio, variant = parse(tag)
        sds = C.SHADOW_OF[ds]
        co_shadow = sds + tag[len(ds):]            # coarsened, same reduction cell
        wh_shadow = f"{sds}_control_whole"          # whole / uncoarsened shadow
        co_aucs, wh_aucs = [], []
        for seed in C.TARGET_SEEDS:
            pt = _post(tag, seed)
            if pt is None:
                continue
            ft = FE.node_features(pt, labels[ds])
            ps_co, ps_wh = _post(co_shadow, seed), _post(wh_shadow, seed)
            if ps_co is not None:
                fs = FE.node_features(ps_co, labels[sds])
                co_aucs.append(shadow_clf.shadow_attack(fs, mask[sds], ft, mask[ds],
                                                        seed=seed))
            if ps_wh is not None:
                fsw = FE.node_features(ps_wh, labels[sds])
                wh_aucs.append(shadow_clf.shadow_attack(fsw, mask[sds], ft, mask[ds],
                                                        seed=seed))
        co_m, co_s = _ms(co_aucs)
        wh_m, wh_s = _ms(wh_aucs)
        delta = round(wh_m - co_m, 4) if co_m != "" and wh_m != "" else ""
        by_ds.setdefault(ds, []).append({
            "dataset": ds, "method": method, "ratio": ratio, "variant": variant,
            "cell": tag,
            "shadow_coarsened_cell": co_shadow, "shadow_mlp_coarsened_auc": co_m,
            "coarsened_std": co_s,
            "shadow_whole_cell": wh_shadow, "shadow_mlp_whole_auc": wh_m,
            "whole_std": wh_s,
            "delta_whole_minus_coarsened": delta,
        })

    os.makedirs(C.RESULTS, exist_ok=True)
    for ds, rows in by_ds.items():
        rows.sort(key=lambda r: (METHOD_ORDER.index(r["method"])
                                 if r["method"] in METHOD_ORDER else 99,
                                 r["ratio"], r["variant"]))
        path = os.path.join(C.RESULTS, f"{ds}_membership_shadow_compare.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {os.path.relpath(path, C.HERE)}  ({len(rows)} cells)")


if __name__ == "__main__":
    main()
