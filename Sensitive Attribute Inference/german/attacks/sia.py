"""The sensitive-attribute attack classifier.

Input per node is Z_u = [p0 || p1] -- the box's posteriors when the sensitive bit is
forced to 0 and to 1. An MLP is trained on a labelled subset of nodes (where the
adversary knows the true bit) to predict the bit, and evaluated on held-out nodes.
Same dataset throughout (no shadow); the split is stratified on the sensitive bit so
train and test share its class balance. Scored by AUC (0.5 = no leakage); accuracy
and the majority-class baseline are reported for context since the bit is imbalanced.
"""
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def attack(Z, S, seed=0, test_size=0.5):
    """Z: (n, 2C) probe features. S: (n,) binary sensitive labels. Returns metrics."""
    S = np.asarray(S).astype(int)
    idx = np.arange(len(S))
    tr, te = train_test_split(idx, test_size=test_size, stratify=S,
                              random_state=seed)
    scaler = StandardScaler().fit(Z[tr])
    clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=seed)
    clf.fit(scaler.transform(Z[tr]), S[tr])
    proba = clf.predict_proba(scaler.transform(Z[te]))[:, 1]
    auc = float(roc_auc_score(S[te], proba))
    acc = float(accuracy_score(S[te], (proba >= 0.5).astype(int)))
    maj = float(max(S[te].mean(), 1 - S[te].mean()))      # majority-class baseline
    return {"auc": auc, "accuracy": acc, "majority_baseline": maj,
            "n_test": int(len(te)), "pos_rate": float(S[te].mean())}
