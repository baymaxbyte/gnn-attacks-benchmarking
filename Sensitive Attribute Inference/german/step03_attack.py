"""Step 03 -- run the sensitive-attribute attack against every box.

For each (cell, sensitive feature): load the two-world Z of each seed, train the
attack MLP on a stratified node split, and score AUC at recovering the sensitive
bit. Averaged over the target seeds. AUC 0.5 = no leakage.

    python step03_attack.py
    python step03_attack.py --only cora --methods gcond
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
from attacks import sia


def _sensitive_labels(dataset, feat_idx):
    m = D.load_whole(dataset)
    col = np.asarray(m["features"][:, feat_idx].todense()).ravel()
    return (col > 0).astype(int)


def _cells_features():
    """(cell, feat_idx) -> sorted seeds present, from the saved Z files."""
    out = {}
    for p in glob.glob(os.path.join(C.ZDIR, "*.npy")):
        base = os.path.basename(p)[:-4]                 # {cell}_feat{idx}_seed{s}
        cell, rest = base.split("_feat", 1)
        feat_idx, seed = rest.split("_seed", 1)
        out.setdefault((cell, int(feat_idx)), []).append(int(seed))
    return {k: sorted(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--methods", default=None)
    args = ap.parse_args()
    K.ensure_dirs()
    os.makedirs(os.path.join(C.RESULTS, "attacks"), exist_ok=True)

    cf = _cells_features()
    keys = sorted(cf)
    if args.only:
        keys = [k for k in keys if k[0].split("_")[0] == args.only]
    if args.methods:
        want = args.methods.split(",")
        keys = [k for k in keys if k[0].split("_")[1] in want]

    label_cache = {}
    print(f"attacking {len(keys)} (cell, feature) pairs\n")
    print(f"{'cell':28s} {'feat':>5s} {'AUC':>14s} {'acc':>7s} {'maj':>6s} {'pos%':>6s}")
    print("-" * 72)
    results = {}
    for cell, feat_idx in keys:
        ds = cell.split("_")[0]
        if (ds, feat_idx) not in label_cache:
            label_cache[(ds, feat_idx)] = _sensitive_labels(ds, feat_idx)
        S = label_cache[(ds, feat_idx)]
        per_seed = []
        for seed in cf[(cell, feat_idx)]:
            Z = np.load(K.z_path(cell, feat_idx, seed))
            per_seed.append(sia.attack(Z, S, seed=seed))
        aucs = [r["auc"] for r in per_seed]
        accs = [r["accuracy"] for r in per_seed]
        rec = {"cell": cell, "dataset": ds, "feat_idx": feat_idx,
               "n_seeds": len(per_seed),
               "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
               "accuracy_mean": float(np.mean(accs)),
               "majority_baseline": per_seed[0]["majority_baseline"],
               "pos_rate": per_seed[0]["pos_rate"], "per_seed": per_seed}
        results[(cell, feat_idx)] = rec
        print(f"{cell:28s} {feat_idx:>5d} {rec['auc_mean']:.4f}+/-{rec['auc_std']:.4f} "
              f"{rec['accuracy_mean']:>7.3f} {rec['majority_baseline']:>6.3f} "
              f"{100*rec['pos_rate']:>5.1f}%")
        with open(os.path.join(C.RESULTS, "attacks", f"{cell}_feat{feat_idx}.json"),
                  "w") as f:
            json.dump(rec, f, indent=2)

    K.record("step03_attack",
             summary=(f"Ran the sensitive-attribute MLP attack on {len(results)} "
                      f"(cell, feature) pairs, {len(C.TARGET_SEEDS)} seeds each"),
             inputs=["artifacts/Z/*.npy", "german.npz"],
             outputs=[K.rel(os.path.join(C.RESULTS, "attacks",
                                         f"{c}_feat{f}.json")) for c, f in results],
             numbers={"pairs": len(results)},
             notes=("AUC recovers the sensitive bit from Z=[p0||p1] via a stratified "
                    "same-dataset node split. 0.5 = no leakage; accuracy is reported "
                    "against the majority-class baseline since the bit is imbalanced."))
    print(f"\nattacked {len(results)} (cell, feature) pairs")


if __name__ == "__main__":
    main()
