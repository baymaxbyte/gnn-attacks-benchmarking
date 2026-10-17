"""Step 03 -- the attacker's side: her reference model g, and the attack datasets.

The adversary holds the ORIGINAL node attributes, queries original node ids, and is
scored on recovering ORIGINAL edges. She never sees the reduction, which nodes were
eliminated, or the Schur-complement edge weights the target was trained on.

g mirrors the target's hyperparameters per the link-stealing paper, but is graph-free:
the same GCN stack run with an empty edge list, so each node sees only its own
self-loop and the model reduces to a per-node MLP.

    python step03_attacker.py
    python step03_attacker.py --skip_g --settings whole,0.7
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

import blackbox_kron as B
import config as C
import data as D

SETTINGS = ["whole"] + [str(r) for r in C.RATIOS]

LABEL_RATE = 0.10
LABEL_SEED = 123
PAIR_SEED = 1              # shared with the GCond, FGC, UGC and GOREN benchmarks
G_TAG = "gkron"
BB_TAG = "kronbb"
SET_TAG = "kron"


def train_reference_model(dataset, verbose=False):
    m = D.load(dataset)
    X = torch.as_tensor(np.asarray(m["features"].todense()), dtype=torch.float32)
    labels = torch.from_numpy(m["labels"])
    n = m["n"]

    rng = np.random.RandomState(LABEL_SEED)
    idx_known = np.sort(rng.permutation(n)[:int(n * LABEL_RATE)])
    print(f"  adversary knows {len(idx_known)} of {n} labels ({LABEL_RATE:.0%}, "
          f"seed={LABEL_SEED}); {X.shape[1]} original feature columns")

    empty_edges = torch.zeros((2, 0), dtype=torch.long)
    empty_w = torch.zeros(0)
    torch.manual_seed(0)
    g = B.GCN(m["d"], B.HIDDEN, m["n_classes"])
    opt = torch.optim.Adam(g.parameters(), lr=B.LR, weight_decay=B.WEIGHT_DECAY)
    idx = torch.from_numpy(idx_known)
    g.train()
    for ep in range(B.EPOCHS):
        opt.zero_grad()
        loss = Fn.nll_loss(g(X, empty_edges, empty_w)[idx], labels[idx])
        loss.backward()
        opt.step()
        if verbose and ep % 100 == 0:
            print(f"    epoch {ep:3d}  loss {float(loss):.4f}")

    g.eval()
    with torch.no_grad():
        post = g(X, empty_edges, empty_w).exp().numpy()
    post = post / post.sum(1, keepdims=True)

    raw = np.asarray(m["features"].todense())
    _, first, counts = np.unique(raw, axis=0, return_index=True, return_counts=True)
    checked = 0
    for gi in first[counts > 1][:25]:
        same = np.flatnonzero((raw == raw[gi]).all(1))
        if len(same) > 1:
            assert np.allclose(post[same[0]], post[same[1]], atol=1e-6), (
                f"g is not graph-free: {same[0]} and {same[1]} share features")
            checked += 1

    lab = m["labels"]
    acc_known = float((post[idx_known].argmax(1) == lab[idx_known]).mean())
    held = np.setdiff1d(np.arange(n), idx_known)
    acc_held = float((post[held].argmax(1) == lab[held]).mean())
    print(f"  g accuracy: {acc_known:.4f} on its known labels, {acc_held:.4f} on the "
          f"rest   (graph-free verified on {checked} duplicate-feature pairs)")
    return post, dict(n_known=int(len(idx_known)), n_nodes=int(n),
                      acc_known=round(acc_known, 4), acc_heldout=round(acc_held, 4),
                      graphfree_pairs_checked=checked, input_dim=int(X.shape[1]),
                      hidden=B.HIDDEN, epochs=B.EPOCHS, lr=B.LR,
                      weight_decay=B.WEIGHT_DECAY)


def publish(dataset, setting):
    src = os.path.join(C.POSTERIORS, f"{dataset}_{setting}.npy")
    if not os.path.exists(src):
        raise SystemExit(f"missing {src}; run step02_blackbox.py first")
    return K.save_attack_posteriors(f"{dataset}_{BB_TAG}_{setting}", np.load(src))


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
        print(f"reference model g -- graph-free, hidden {B.HIDDEN}, {B.EPOCHS} epochs, "
              f"original features")
        print("=" * 78)
        for ds in C.DATASETS:
            print(f"\n{ds}:")
            post, info = train_reference_model(ds)
            outputs.append(K.rel(K.save_attack_posteriors(f"{ds}_{G_TAG}", post)))
            numbers[f"g_{ds}"] = info

    print(f"\n{'=' * 78}")
    print("publishing black-box posteriors to the attack code")
    print("=" * 78)
    for ds in C.DATASETS:
        for s in settings:
            publish(ds, s)

    print(f"\n{'=' * 78}")
    print("attack datasets -- ORIGINAL edges, equal non-edges, random 50/50 split")
    print("=" * 78)
    for ds in C.DATASETS:
        for s in settings:
            build_attack_dataset(ds, s, force=args.force)

    K.record(
        "step03_attacker",
        summary=(f"Trained the adversary's reference model g for {len(C.DATASETS)} "
                 f"datasets and built {len(C.DATASETS) * len(settings)} attack "
                 f"datasets on ORIGINAL edges"),
        inputs=["artifacts/posteriors/*.npy",
                "attack/data/dataset/original/ind.{cora,citeseer}.*"],
        outputs=outputs, numbers=numbers,
        notes=(f"g mirrors the target's hyperparameters but is graph-free and runs on "
               f"the original features. The adversary cannot tell a reduced box from "
               f"the control. Pair seed {PAIR_SEED} matches the GCond, FGC, UGC and "
               f"GOREN benchmarks, so all five attack identical pairs."))
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
