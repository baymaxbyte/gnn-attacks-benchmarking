"""Step 03 -- the attacker's side: her reference model g, and the attack datasets.

Both live entirely on the adversary's side of the black-box boundary.

g mirrors the target's hyperparameters but is graph-free (identity adjacency).
The adversary knows a random 10% of labels; attack datasets are built on the
ORIGINAL edges.

    python step03_attacker.py
    python step03_attacker.py --skip_g
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import scipy.sparse as sp

import common as C

GCOND = C.GCOND
GCOND_PY = C.GCOND_PY

SETTINGS = ["whole", "0.25", "0.5", "1.0"]
DATASETS = ["cora", "citeseer"]

G_HIDDEN = 256
G_NLAYERS = 2
G_DROPOUT = 0.0
G_WEIGHT_DECAY = 5e-4
G_EPOCHS = 600
G_LR = 0.01

LABEL_RATE = 0.10
LABEL_SEED = 123
PAIR_SEED = 1


# ------------------------------------------------------------- reference model
def train_reference_model(dataset, device="cpu", verbose=False):
    """Graph-free MLP on the adversary's 10% of labels."""
    sys.path.insert(0, GCOND) if GCOND not in sys.path else None
    sys.path.insert(0, C.LSA) if C.LSA not in sys.path else None
    from lsa.data import load_data, preprocess_features

    cwd = os.getcwd()
    os.chdir(GCOND)
    try:
        import utils as gu
        from models.gcn import GCN
        data = gu.Transd2Ind(gu.get_dataset(dataset, normalize_features=True),
                             keep_ratio=1.0)
    finally:
        os.chdir(cwd)

    _, feats_raw, *_ = load_data(C.LSA_ORIGINAL, dataset)
    X = preprocess_features(feats_raw)
    n = X.shape[0]
    labels = np.asarray(data.labels_full).astype(np.int64)
    n_classes = int(labels.max()) + 1

    rng = np.random.RandomState(LABEL_SEED)
    idx_known = np.sort(rng.permutation(n)[:int(n * LABEL_RATE)])
    print(f"  adversary knows {len(idx_known)} of {n} labels "
          f"({LABEL_RATE:.0%}, seed={LABEL_SEED})")

    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    g = GCN(nfeat=X.shape[1], nhid=G_HIDDEN, nclass=n_classes,
            nlayers=G_NLAYERS, dropout=G_DROPOUT, lr=G_LR,
            weight_decay=G_WEIGHT_DECAY, device=device).to(device)

    eye = sp.eye(n, format="csr")            # identity => graph-free
    os.chdir(GCOND)
    try:
        g.fit(X, eye, labels, idx_known, idx_val=None, train_iters=G_EPOCHS,
              normalize=True, verbose=verbose)
        out = g.predict(X, eye)
    finally:
        os.chdir(cwd)

    post = np.exp(out.detach().cpu().numpy())    # forward returns log_softmax
    post = post / post.sum(1, keepdims=True)

    raw = feats_raw.toarray() if sp.issparse(feats_raw) else np.asarray(feats_raw)
    _, first, counts = np.unique(raw, axis=0, return_index=True,
                                 return_counts=True)
    dup_groups = first[counts > 1]
    checked = 0
    for gi in dup_groups[:25]:
        same = np.flatnonzero((raw == raw[gi]).all(1))
        if len(same) > 1:
            assert np.allclose(post[same[0]], post[same[1]], atol=1e-6), (
                f"g is not graph-free: nodes {same[0]} and {same[1]} have "
                f"identical features but different posteriors")
            checked += 1
    acc_known = float((post[idx_known].argmax(1) == labels[idx_known]).mean())
    held = np.setdiff1d(np.arange(n), idx_known)
    acc_held = float((post[held].argmax(1) == labels[held]).mean())
    print(f"  g accuracy: {acc_known:.4f} on its known labels, "
          f"{acc_held:.4f} on the rest   (graph-free verified on "
          f"{checked} duplicate-feature pairs)")
    return post, dict(n_known=int(len(idx_known)), n_nodes=int(n),
                      acc_known=round(acc_known, 4),
                      acc_heldout=round(acc_held, 4),
                      graphfree_pairs_checked=checked)


# ------------------------------------------------------------ attack datasets
def build_attack_dataset(dataset, setting):
    name = f"{dataset}_gc_{setting}"
    probe = os.path.join(C.LSA_ATTACK, f"{name}_attack_test.json")
    if os.path.exists(probe):
        print(f"  [cached] {name}_attack_*.json")
        return name, None
    cmd = [GCOND_PY, "partial_graph_generation.py",
           "--dataset", dataset,
           "--datapath", "data/dataset/original/",
           "--prediction_path", "data/pred/",
           "--saving_path", "data/attack_dataset/",
           "--target_pred", f"{dataset}_gcondbb_{setting}_pred.pkl",
           "--reference_pred", f"{dataset}_g_pred.pkl",
           "--name", name,
           "--seed", str(PAIR_SEED)]
    print(f"\n>>> attack dataset {name}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=C.LSA, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:]); print(p.stderr[-2000:])
        raise SystemExit(f"failed building {name}")
    for line in p.stdout.splitlines():
        if any(k in line for k in ("linked pairs", "all  :", "train:", "test :")):
            print("   ", line.strip())
    print(f"    {time.time() - t0:.1f}s")
    return name, p.stdout


def main():
    ap = argparse.ArgumentParser(description="Attacker-side setup")
    ap.add_argument("--skip_g", action="store_true")
    ap.add_argument("--settings", default=",".join(SETTINGS))
    args = ap.parse_args()
    settings = [s.strip() for s in args.settings.split(",")]

    C.ensure_dirs()
    numbers, outputs = {}, []
    t0 = time.time()

    if not args.skip_g:
        print("=" * 74)
        print("reference model g -- graph-free MLP, 256 hidden, 600 epochs")
        print("=" * 74)
        for ds in DATASETS:
            print(f"\n{ds}:")
            post, info = train_reference_model(ds)
            path = C.save_posteriors(f"{ds}_g", post)
            outputs.append(C.rel(path))
            numbers[f"g_{ds}"] = info

    print(f"\n{'=' * 74}")
    print("attack datasets -- original edges, random 50/50 split, shared seed")
    print("=" * 74)
    for ds in DATASETS:
        for s in settings:
            name, _ = build_attack_dataset(ds, s)
            outputs += [C.rel(os.path.join(C.LSA_ATTACK,
                                           f"{name}_attack_{k}.json"))
                        for k in ("all", "train", "test")]

    C.record(
        step="03_attacker_setup",
        summary=(f"Trained the adversary's reference model g for "
                 f"{len(DATASETS)} datasets and built "
                 f"{len(DATASETS) * len(settings)} attack datasets on ORIGINAL "
                 f"edges"),
        inputs=["Benchmarking/lsa_paper/data/pred/*_gcondbb_*_pred.pkl",
                "Benchmarking/lsa_paper/data/dataset/original/"],
        outputs=outputs,
        numbers=numbers,
        notes="g is graph-free and mirrors the target; one g per dataset, reused across all settings so the black box is the only thing that varies.")
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
