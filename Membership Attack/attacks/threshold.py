"""Unsupervised threshold membership attacks.

Each is a single scalar signal that should rank members above non-members; the AUC
of that ranking is the attack's score, so no threshold and no shadow are needed.
Members tend to be predicted with higher confidence, lower entropy, lower loss.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-12


def threshold_aucs(post, labels, member_mask):
    """AUC of each per-node signal at separating members (1) from non-members (0)."""
    post = np.clip(np.asarray(post, dtype=np.float64), EPS, 1.0)
    y = np.asarray(member_mask).astype(int)
    idx = np.arange(post.shape[0])
    labels = np.asarray(labels)

    conf = post.max(1)
    ent = -(post * np.log(post)).sum(1)
    p_true = post[idx, labels]
    loss = -np.log(p_true)
    correct = (post.argmax(1) == labels).astype(float)

    signals = {                     # higher value => predicted member
        "confidence": conf,
        "neg_entropy": -ent,
        "p_true": p_true,
        "neg_loss": -loss,
        "correct": correct,
    }
    return {k: float(roc_auc_score(y, v)) for k, v in signals.items()}
