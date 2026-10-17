"""Step 03 -- the attacker's side: her reference model g, and the attack datasets.

Everything here lives on the adversary's side of the black-box boundary. She holds
the ORIGINAL, pre-coarsening node attributes -- she never sees X_c, A_c, the
assignment, or how many supernodes exist. Ground truth is the ORIGINAL edge set.

g mirrors the FGC target's hyperparameters, per the link-stealing paper
("Hyperparameters, including the number of neurons in the hidden layer, epochs, and
learning rate are the same as those of the target model"), but is graph-free: the
same GCNConv stack run with an empty edge list, so each node sees only its own
self-loop and the model reduces to a per-node MLP.

    python step03_attacker.py
    python step03_attacker.py --skip_g --settings whole,0.3
"""
import argparse
import os
import subprocess
import time

import common as K

os.environ.update(K.single_thread_env())

import numpy as np
import torch
import torch.nn.functional as Fn

import blackbox_fgc as B
import config as C
import data as D

SETTINGS = ["whole", "0.3", "0.1", "0.05"]

# g mirrors the target (blackbox_fgc): hidden 64, dropout 0.5, lr 0.01, wd 1e-4,
# 60 epochs, raw binary features -- the same attributes the target consumes.
LABEL_RATE = 0.10          # link-stealing paper section 5.1
LABEL_SEED = 123
PAIR_SEED = 1              # same seed as the GCond benchmark -> identical pairs
G_TAG = "gfgc"             # keeps GCond's <ds>_g_pred.pkl untouched
BB_TAG = "fgcbb"
SET_TAG = "fgc"


def train_reference_model(dataset, verbose=False):
    """Graph-free mirror of the FGC target, fitted on the adversary's 10%."""
    m = D.load(dataset)
    X = torch.as_tensor(np.asarray(m["features"].todense()), dtype=torch.float32)
    labels = torch.from_numpy(m["labels"])
    n = m["n"]

    rng = np.random.RandomState(LABEL_SEED)
    idx_known = np.sort(rng.permutation(n)[:int(n * LABEL_RATE)])
    print(f"  adversary knows {len(idx_known)} of {n} labels "
          f"({LABEL_RATE:.0%}, seed={LABEL_SEED})")

    empty_edges = torch.zeros((2, 0), dtype=torch.long)   # graph-free
    torch.manual_seed(0)
    g = B.Net(m["d"], m["n_classes"])
    opt = torch.optim.Adam(g.parameters(), lr=B.LR, weight_decay=B.WEIGHT_DECAY)
    idx = torch.from_numpy(idx_known)
    g.train()
    for ep in range(B.EPOCHS):
        opt.zero_grad()
        loss = Fn.nll_loss(g(X, empty_edges)[idx], labels[idx])
        loss.backward()
        opt.step()
        if verbose and ep % 20 == 0:
            print(f"    epoch {ep:3d}  loss {float(loss):.4f}")

    g.eval()
    with torch.no_grad():
        post = g(X, empty_edges).exp().numpy()
    post = post / post.sum(1, keepdims=True)

    # g must be a pure function of a node's own attributes: identical feature rows
    # must give identical posteriors.
    raw = np.asarray(m["features"].todense())
    _, first, counts = np.unique(raw, axis=0, return_index=True, return_counts=True)
    checked = 0
    for gi in first[counts > 1][:25]:
        same = np.flatnonzero((raw == raw[gi]).all(1))
        if len(same) > 1:
            assert np.allclose(post[same[0]], post[same[1]], atol=1e-6), (
                f"g is not graph-free: {same[0]} and {same[1]} share features "
                f"but differ in posterior")
            checked += 1

    lab = m["labels"]
    acc_known = float((post[idx_known].argmax(1) == lab[idx_known]).mean())
    held = np.setdiff1d(np.arange(n), idx_known)
    acc_held = float((post[held].argmax(1) == lab[held]).mean())
    print(f"  g accuracy: {acc_known:.4f} on its known labels, {acc_held:.4f} on "
          f"the rest   (graph-free verified on {checked} duplicate-feature pairs)")
    return post, dict(n_known=int(len(idx_known)), n_nodes=int(n),
                      acc_known=round(acc_known, 4), acc_heldout=round(acc_held, 4),
                      graphfree_pairs_checked=checked,
                      hidden=B.HIDDEN, epochs=B.EPOCHS, lr=B.LR,
                      weight_decay=B.WEIGHT_DECAY, dropout=B.DROPOUT)


def publish_blackbox_posteriors(dataset, setting):
    """Copy the Mode A posteriors step02 served into the attack code's pred dir."""
    src = os.path.join(C.ARTIFACTS, "posteriors", f"{dataset}_{setting}_orig.npy")
    if not os.path.exists(src):
        raise SystemExit(f"missing {src}; run step02_blackbox.py first")
    post = np.load(src)
    return K.save_attack_posteriors(f"{dataset}_{BB_TAG}_{setting}", post)


def build_attack_dataset(dataset, setting, force=False):
    name = f"{dataset}_{SET_TAG}_{setting}"
    probe = os.path.join(C.ATTACK_DATASET, f"{name}_attack_test.json")
    if os.path.exists(probe) and not force:
        print(f"  [cached] {name}_attack_*.json")
        return name
    cmd = [C.PYTHON, "partial_graph_generation.py",
           "--dataset", dataset,
           "--datapath", "data/dataset/original/",
           "--prediction_path", "data/pred/",
           "--saving_path", "data/attack_dataset/",
           "--target_pred", f"{dataset}_{BB_TAG}_{setting}_pred.pkl",
           "--reference_pred", f"{dataset}_{G_TAG}_pred.pkl",
           "--name", name,
           "--seed", str(PAIR_SEED)]
    print(f"\n>>> attack dataset {name}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=C.ATTACK_HOME, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:]); print(p.stderr[-2000:])
        raise SystemExit(f"failed building {name}")
    for line in p.stdout.splitlines():
        if any(k in line for k in ("linked pairs", "all  :", "train:", "test :")):
            print("   ", line.strip())
    print(f"    {time.time() - t0:.1f}s")
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_g", action="store_true")
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    settings = [s.strip() for s in args.settings.split(",")]

    K.ensure_dirs()
    torch.set_num_threads(4)
    numbers, outputs = {}, []
    t0 = time.time()

    if not args.skip_g:
        print("=" * 78)
        print(f"reference model g -- graph-free, hidden {B.HIDDEN}, "
              f"{B.EPOCHS} epochs, raw binary features")
        print("=" * 78)
        for ds in C.DATASETS:
            print(f"\n{ds}:")
            post, info = train_reference_model(ds)
            outputs.append(K.rel(K.save_attack_posteriors(f"{ds}_{G_TAG}", post)))
            numbers[f"g_{ds}"] = info

    print(f"\n{'=' * 78}")
    print("publishing black-box posteriors (Mode A) to the attack code")
    print("=" * 78)
    for ds in C.DATASETS:
        for s in settings:
            publish_blackbox_posteriors(ds, s)

    print(f"\n{'=' * 78}")
    print("attack datasets -- ORIGINAL edges, equal non-edges, random 50/50 split")
    print("=" * 78)
    for ds in C.DATASETS:
        for s in settings:
            build_attack_dataset(ds, s, force=args.force)

    K.record(
        "step03_attacker",
        summary=(f"Trained the adversary's reference model g for "
                 f"{len(C.DATASETS)} datasets and built "
                 f"{len(C.DATASETS) * len(settings)} attack datasets on ORIGINAL edges"),
        inputs=["artifacts/posteriors/*_orig.npy",
                "data/dataset/original/ind.{cora,citeseer}.*"],
        outputs=outputs,
        numbers=numbers,
        notes=(f"g mirrors the FGC target (hidden {B.HIDDEN}, {B.EPOCHS} epochs, "
               f"lr {B.LR}, wd {B.WEIGHT_DECAY}) but is graph-free, per the "
               f"link-stealing paper. Adversary holds ORIGINAL uncoarsened "
               f"attributes and a random {LABEL_RATE:.0%} of labels. Pair seed "
               f"{PAIR_SEED} matches the GCond benchmark, so both benchmarks "
               f"attack identical pairs with identical splits."))
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
