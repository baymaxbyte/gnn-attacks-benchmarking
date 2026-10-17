"""
Stage 2 -- build the attack dataset and split it the way the paper specifies.

Paper clause (p7):

    "Recall that several attacks require the knowledge of the target dataset's
     partial graph. To simulate and fairly evaluate different attacks, we
     construct an attack dataset which contains node pairs and labels
     representing whether they are linked or not. Specifically, we first select
     all node pairs that are linked. Then, we randomly sample the same number of
     node pairs that are not linked. ... Next, we split the attack dataset
     randomly by half into attack training dataset and attack testing dataset.
     We use the attack training dataset to train our attack models when the
     target dataset's partial graph is part of the adversary's knowledge. We use
     attack testing dataset to evaluate all our attacks. For the attacks that
     have a shadow dataset, we also construct an attack dataset on the shadow
     dataset to train the attack model. Note that we do not split this attack
     dataset because we do not use it for evaluation."

Three files are written per dataset:

    <ds>_attack_all.json     every pair, shuffled. This is the unsplit form the
                             paper prescribes when the dataset plays the role of
                             a SHADOW dataset.
    <ds>_attack_train.json   a random half. This is the partial graph A*.
    <ds>_attack_test.json    the other half. Every attack is evaluated on this,
                             including the two unsupervised ones.

The released code instead emits ten files per dataset for train ratios
0.1 ... 1.0 and every attack then hardcodes the ratio-0.9 file as its test set,
which makes the evaluation set 10% rather than the paper's 50%. Use
`--train_fraction` only for the Figure-7 training-set-size study; the headline
configuration is the default 0.5.

How "randomly by half" is realised: the positive list and the negative list are
each shuffled, then paired index-wise and cut at the halfway index. That keeps
both halves exactly 1:1 balanced, which matters because the paper's justification
for AUC assumes the balanced construction above. Splitting the merged pool
instead would leave the halves only approximately balanced.

Each record carries f's posteriors, g's posteriors and the raw attributes for
both endpoints, so a single file serves all 8 attacks; each attack then reads
only the fields its knowledge triple allows.
"""
import argparse
import json
import os
import pickle as pkl
import random
import time

import numpy as np

from lsa.data import load_data
from lsa.spec import ATTACK_TRAIN_FRACTION, NEGATIVE_SAMPLING_RATIO


def parse_args():
    p = argparse.ArgumentParser(
        description="Build the paper's attack dataset and 50/50 split")
    p.add_argument("--dataset", default="cora")
    p.add_argument("--datapath", default="data/dataset/original/")
    p.add_argument("--prediction_path", default="data/pred/")
    p.add_argument("--saving_path", default="data/attack_dataset/")
    p.add_argument("--train_fraction", type=float, default=ATTACK_TRAIN_FRACTION,
                   help="paper: 0.5. Other values only for the Figure-7 study.")
    p.add_argument("--seed", type=int, default=1,
                   help="seed for negative sampling and the random half split")
    p.add_argument("--target_pred", default=None,
                   help="posterior file for f; default <dataset>_gcn_pred.pkl. "
                        "Point this at a sage/gat/top-k file for Tables 11-12.")
    p.add_argument("--reference_pred", default=None,
                   help="posterior file for g; default <dataset>_dense_pred.pkl")
    p.add_argument("--name", default=None,
                   help="output filename stem; default <dataset>. Use this to "
                        "keep variants side by side, e.g. cora_gat")
    return p.parse_args()


def build_attack_dataset(adj, node_num, seed=1):
    """Positives = all linked pairs; negatives = an equal number of unlinked."""
    rng = random.Random(seed)

    link, existing = [], set()
    rows, cols = adj.nonzero()
    for r, c in zip(rows, cols):
        if r < c:                      # canonical order, so (u,v) == (v,u)
            link.append([int(r), int(c)])
            existing.add((int(r), int(c)))
    print(f"  linked pairs: {len(link)}  (from {len(rows)} directed entries)")

    n_neg = int(len(link) * NEGATIVE_SAMPLING_RATIO)
    unlink, seen_neg = [], set()
    t0 = time.time()
    while len(unlink) < n_neg:
        r = rng.randint(0, node_num - 1)
        c = rng.randint(0, node_num - 1)
        if r > c:
            r, c = c, r
        if r == c or (r, c) in existing or (r, c) in seen_neg:
            continue
        unlink.append([r, c])
        seen_neg.add((r, c))
    print(f"  unlinked pairs: {len(unlink)}  ({time.time()-t0:.1f}s)")

    # "split the attack dataset randomly by half"
    rng.shuffle(link)
    rng.shuffle(unlink)
    return link, unlink


def _record(label, i0, i1, gcn_pred, dense_pred, feature_arr):
    return {
        "label": label,
        "gcn_pred0": gcn_pred[i0], "gcn_pred1": gcn_pred[i1],
        "dense_pred0": dense_pred[i0], "dense_pred1": dense_pred[i1],
        "feature_arr0": feature_arr[i0], "feature_arr1": feature_arr[i1],
        "id_pair": [int(i0), int(i1)],
    }


def _dump(rows, path):
    with open(path, "w") as wf:
        for row in rows:
            wf.write(json.dumps(row) + "\n")
    return len(rows)


def main():
    args = parse_args()
    os.makedirs(args.saving_path, exist_ok=True)

    adj, features, *_ = load_data(args.datapath, args.dataset)
    feature_arr = (features if isinstance(features, np.ndarray)
                   else features.toarray()).tolist()

    ref_file = args.reference_pred or f"{args.dataset}_dense_pred.pkl"
    tgt_file = args.target_pred or f"{args.dataset}_gcn_pred.pkl"
    with open(os.path.join(args.prediction_path, ref_file), "rb") as f:
        dense_pred = pkl.loads(f.read()).tolist()
    with open(os.path.join(args.prediction_path, tgt_file), "rb") as f:
        gcn_pred = pkl.loads(f.read()).tolist()

    node_num = len(gcn_pred)
    print(f"dataset={args.dataset}  nodes={node_num}")
    print(f"  f <- {tgt_file}\n  g <- {ref_file}")
    link, unlink = build_attack_dataset(adj, node_num, seed=args.seed)

    n_train = int(len(link) * args.train_fraction)
    train, test, allrows = [], [], []
    for i in range(len(link)):
        pos = _record(1, link[i][0], link[i][1], gcn_pred, dense_pred, feature_arr)
        neg = _record(0, unlink[i][0], unlink[i][1], gcn_pred, dense_pred, feature_arr)
        allrows += [pos, neg]
        (train if i < n_train else test).extend([pos, neg])

    base = os.path.join(args.saving_path, args.name or args.dataset)
    n_all = _dump(allrows, f"{base}_attack_all.json")
    n_tr = _dump(train, f"{base}_attack_train.json")
    n_te = _dump(test, f"{base}_attack_test.json")

    print(f"  all  : {n_all:6d} records  -> {base}_attack_all.json")
    print(f"  train: {n_tr:6d} records  -> {base}_attack_train.json")
    print(f"  test : {n_te:6d} records  -> {base}_attack_test.json")
    print(f"  train_fraction={args.train_fraction}  "
          f"(paper: {ATTACK_TRAIN_FRACTION})")


if __name__ == "__main__":
    main()
