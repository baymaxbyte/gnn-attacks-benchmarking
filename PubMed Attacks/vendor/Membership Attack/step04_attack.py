"""Step 04 -- run the membership attacks against every box.

For each cell and seed: the unsupervised threshold attacks (confidence, entropy,
loss, correctness) on the target's own posteriors, plus the supervised shadow
classifier trained on the OTHER dataset at the same cell. Every attack is scored by
AUC at separating members (V_in) from non-members (V_out); 0.5 is no leakage.

    python step04_attack.py
    python step04_attack.py --only cora --methods gcond
"""
import argparse
import glob
import json
import os

import common as K

os.environ.update(K.single_thread_env())

import numpy as np

import config as C
import data as D
from attacks import features as FE
from attacks import shadow_clf, threshold


def _tags_present():
    tags = set()
    for p in glob.glob(os.path.join(C.POSTERIORS, "*_seed*.npy")):
        tags.add(os.path.basename(p).rsplit("_seed", 1)[0])
    return sorted(tags)


def _post(tag, seed):
    p = os.path.join(C.POSTERIORS, f"{tag}_seed{seed}.npy")
    return np.load(p) if os.path.exists(p) else None


def _split_cache():
    cache = {}
    for ds in C.DATASETS:
        sp_ = D.load_split(ds)
        m = D.load_whole(ds)
        mask = np.zeros(m["n"], dtype=bool)
        mask[sp_["member_ids"]] = True
        cache[ds] = {"member_mask": mask, "labels": m["labels"]}
    return cache


def attack_cell(tag, splits):
    ds = tag.split("_", 1)[0]
    shadow_ds = C.SHADOW_OF[ds]
    shadow_tag = shadow_ds + tag[len(ds):]
    tinfo, sinfo = splits[ds], splits[shadow_ds]

    per_seed = []
    for seed in C.TARGET_SEEDS:
        post_t = _post(tag, seed)
        if post_t is None:
            continue
        thr = threshold.threshold_aucs(post_t, tinfo["labels"],
                                       tinfo["member_mask"])
        row = {"seed": seed, **{f"thr_{k}": v for k, v in thr.items()}}

        post_s = _post(shadow_tag, seed)
        if post_s is not None:
            ft = FE.node_features(post_t, tinfo["labels"])
            fs = FE.node_features(post_s, sinfo["labels"])
            row["shadow_mlp"] = shadow_clf.shadow_attack(
                fs, sinfo["member_mask"], ft, tinfo["member_mask"], seed=seed)
        per_seed.append(row)

    if not per_seed:
        return None
    keys = [k for k in per_seed[0] if k != "seed"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in per_seed if k in r]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    best_thr = max((agg[k]["mean"] for k in agg if k.startswith("thr_")),
                   default=float("nan"))
    agg["best_threshold"] = {"mean": best_thr}
    return {"tag": tag, "shadow_tag": shadow_tag, "n_seeds": len(per_seed),
            "aucs": agg, "per_seed": per_seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--methods", default=None)
    args = ap.parse_args()

    K.ensure_dirs()
    os.makedirs(os.path.join(C.RESULTS, "attacks"), exist_ok=True)
    splits = _split_cache()

    tags = _tags_present()
    if args.only:
        tags = [t for t in tags if t.split("_")[0] == args.only]
    if args.methods:
        want = args.methods.split(",")
        tags = [t for t in tags if t.split("_")[1] in want]

    print(f"attacking {len(tags)} cells\n")
    print(f"{'cell':28s} {'best thr':>9s} {'shadow':>8s}  "
          f"{'conf':>6s} {'entropy':>7s} {'loss':>6s}")
    print("-" * 76)
    results = {}
    for tag in tags:
        r = attack_cell(tag, splits)
        if r is None:
            continue
        results[tag] = r
        a = r["aucs"]
        sh = a.get("shadow_mlp", {}).get("mean", float("nan"))
        print(f"{tag:28s} {a['best_threshold']['mean']:>9.4f} {sh:>8.4f}  "
              f"{a['thr_confidence']['mean']:>6.3f} "
              f"{a['thr_neg_entropy']['mean']:>7.3f} "
              f"{a['thr_neg_loss']['mean']:>6.3f}")
        with open(os.path.join(C.RESULTS, "attacks", f"{tag}.json"), "w") as f:
            json.dump(r, f, indent=2)

    K.record(
        "step04_attack",
        summary=(f"Ran threshold + cross-dataset shadow membership attacks against "
                 f"{len(results)} cells, {len(C.TARGET_SEEDS)} seeds each"),
        inputs=["artifacts/posteriors/*.npy", "artifacts/splits/*_split.npz"],
        outputs=[K.rel(os.path.join(C.RESULTS, "attacks", f"{t}.json"))
                 for t in results],
        numbers={"cells": len(results)},
        notes=("AUC separates members (V_in) from non-members (V_out); 0.5 = no "
               "leakage. Threshold attacks are unsupervised; shadow_mlp trains on "
               "the other dataset at the same cell. Features are dimension-agnostic "
               "scalar summaries so the shadow transfers across datasets."))
    print(f"\nattacked {len(results)} cells")


if __name__ == "__main__":
    main()
