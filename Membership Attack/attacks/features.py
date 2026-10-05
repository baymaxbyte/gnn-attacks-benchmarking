"""Dimension-agnostic per-node features from a black box's posterior.

Every feature is a scalar summary of the posterior (or of it against the true
label), so the vector has a fixed length regardless of the number of classes. That
is what lets the shadow classifier train on one dataset (e.g. Citeseer, 6 classes)
and transfer to another (Cora, 7 classes).

Label-free features (confidence, entropy, gaps, top-k) assume only black-box output.
Label-based features (true-class probability, loss, correctness) additionally assume
the adversary knows the node's label, which is the stronger of He et al.'s threat
models.
"""
import numpy as np

EPS = 1e-12
FEATURE_NAMES = ["confidence", "entropy", "gap12", "top1", "top2", "top3",
                 "p_true", "loss", "correct"]


def node_features(post, labels):
    """(N, C) posteriors + (N,) labels -> (N, 9) fixed-width scalar features."""
    post = np.clip(np.asarray(post, dtype=np.float64), EPS, 1.0)
    n, c = post.shape
    conf = post.max(1)
    ent = -(post * np.log(post)).sum(1)
    s = -np.sort(-post, axis=1)
    top1 = s[:, 0]
    top2 = s[:, 1] if c > 1 else np.zeros(n)
    top3 = s[:, 2] if c > 2 else np.zeros(n)
    gap = top1 - top2

    labels = np.asarray(labels)
    idx = np.arange(n)
    p_true = post[idx, labels]
    loss = -np.log(p_true)
    correct = (post.argmax(1) == labels).astype(np.float64)

    feats = np.stack([conf, ent, gap, top1, top2, top3, p_true, loss, correct], 1)
    return feats.astype(np.float32)
