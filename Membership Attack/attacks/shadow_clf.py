"""Supervised shadow-trained membership classifier.

The shadow is the OTHER dataset at the same reduction cell, whose member /
non-member labels the adversary knows. An MLP is trained on the shadow's per-node
features to separate members from non-members, then applied to the target. Because
the features are fixed-width scalars, the classifier transfers across datasets with
different class counts. Shadow training is class-balanced so the classifier is not
biased by the 80/20 base rate; the target is scored by AUC, which is base-rate
invariant.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def shadow_attack(shadow_feats, shadow_member, target_feats, target_member,
                  seed=0):
    shadow_member = np.asarray(shadow_member).astype(int)
    target_member = np.asarray(target_member).astype(int)

    pos = np.where(shadow_member == 1)[0]
    neg = np.where(shadow_member == 0)[0]
    rng = np.random.RandomState(seed)
    k = min(len(pos), len(neg))
    sel = np.concatenate([rng.choice(pos, k, replace=False),
                          rng.choice(neg, k, replace=False)])
    Xs, ys = shadow_feats[sel], shadow_member[sel]

    scaler = StandardScaler().fit(Xs)
    clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=seed)
    clf.fit(scaler.transform(Xs), ys)
    scores = clf.predict_proba(scaler.transform(target_feats))[:, 1]
    return float(roc_auc_score(target_member, scores))
