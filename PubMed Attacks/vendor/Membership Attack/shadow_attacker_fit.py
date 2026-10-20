"""Measure the shadow-MLP attacker's OWN fit quality -- its train accuracy and a
held-out shadow-test accuracy -- for the membership shadow attack, across every cell
and both datasets.

The benchmark (step04_attack.py / shadow_clf.shadow_attack) trains the attacker MLP
on the ENTIRE balanced shadow (the other dataset at the same reduction cell) and
reports only the AUC on the TARGET. It never measures how well the attacker fits the
shadow itself. This script mirrors that attacker exactly -- the SAME LSA attack MLP
(shadow_clf.AttackMLP: 3xDense(32,ReLU)+Dropout(0.5)->Dense(2), Adam lr1e-3, 50
epochs, batch 128), same class balancing, same StandardScaler -- but holds out 30% of
the balanced shadow as a shadow-test set, so the attacker's own train and held-out
shadow-test accuracy can be reported. Because of the hold-out, the attacker here
trains on 70% of the balanced shadow (not 100% as in the pipeline), so
target_transfer_auc is a reference and may differ slightly from the pipeline
shadow_mlp_auc. Features are the current 8-dim set (confidence dropped).
Non-destructive: writes results/<ds>_shadow_attacker_fit.csv.

    python shadow_attacker_fit.py
"""
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config as C
import data as D
from attacks import features as FE
from attacks import shadow_clf as SC
from attacks.shadow_clf import AttackMLP

METHOD_ORDER = ["control", "kron", "ugc", "fgc", "goren", "gcond"]
TEST_FRACTION = 0.30


def parse(tag):
    p = tag.split("_")
    ds, method, rest = p[0], p[1], p[2:]
    if rest == ["whole"]:
        return ds, method, "", ""
    ratio = rest[0][1:] if rest and rest[0].startswith("r") else ""
    variant = "_".join(rest[1:]) if len(rest) > 1 else ""
    return ds, method, ratio, variant


def _post(tag, seed):
    p = os.path.join(C.POSTERIORS, f"{tag}_seed{seed}.npy")
    return np.load(p) if os.path.exists(p) else None


def _ms(a):
    return (round(float(np.mean(a)), 4), round(float(np.std(a)), 4)) if a else ("", "")


def attacker_fit(shadow_feats, shadow_member, target_feats, target_member, seed):
    """Train the LSA attack MLP (shadow_clf.AttackMLP) on a 70% split of the balanced
    shadow and measure its own train / held-out-shadow-test accuracy, plus the target
    transfer AUC.

    Returns (train_acc, test_acc, target_auc, n_balanced).
    """
    shadow_member = np.asarray(shadow_member).astype(int)
    target_member = np.asarray(target_member).astype(int)

    pos = np.where(shadow_member == 1)[0]
    neg = np.where(shadow_member == 0)[0]
    rng = np.random.RandomState(seed)
    k = min(len(pos), len(neg))
    sel = np.concatenate([rng.choice(pos, k, replace=False),
                          rng.choice(neg, k, replace=False)])
    Xs, ys = shadow_feats[sel], shadow_member[sel]

    # Hold out part of the balanced shadow as the attacker's own test set.
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, ys, test_size=TEST_FRACTION, random_state=seed, stratify=ys)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)
    Xtg_s = scaler.transform(target_feats)

    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr_t = torch.as_tensor(np.asarray(Xtr_s), dtype=torch.float32)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long)

    model = AttackMLP(Xtr_t.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=SC.ATTACK_LR, betas=(0.9, 0.999),
                           eps=1e-7)
    lossfn = nn.CrossEntropyLoss()
    n = Xtr_t.shape[0]
    for _ in range(SC.ATTACK_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, SC.ATTACK_BATCH):
            idx = perm[i:i + SC.ATTACK_BATCH]
            opt.zero_grad()
            loss = lossfn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        tr_pred = model(Xtr_t).argmax(1).numpy()
        te_pred = model(torch.as_tensor(np.asarray(Xte_s),
                                        dtype=torch.float32)).argmax(1).numpy()
        tg_scores = F.softmax(
            model(torch.as_tensor(np.asarray(Xtg_s), dtype=torch.float32)),
            dim=1).numpy()[:, 1]
    train_acc = float((tr_pred == ytr).mean())
    test_acc = float((te_pred == yte).mean())
    target_auc = float(roc_auc_score(target_member, tg_scores))
    return train_acc, test_acc, target_auc, int(2 * k)


def main():
    splits = {ds: D.load_split(ds) for ds in C.DATASETS}
    labels = {ds: D.load_whole(ds)["labels"] for ds in C.DATASETS}
    mask = {ds: splits[ds]["member_mask"] for ds in C.DATASETS}

    tags = sorted({f.rsplit("_seed", 1)[0] for f in os.listdir(C.POSTERIORS)
                   if "_seed" in f})
    by_ds = {}
    for tag in tags:
        ds, method, ratio, variant = parse(tag)
        sds = C.SHADOW_OF[ds]
        shadow_tag = sds + tag[len(ds):]          # coarsened same-cell shadow (the real one)
        tr_accs, te_accs, aucs = [], [], []
        nbal = ""
        for seed in C.TARGET_SEEDS:
            pt = _post(tag, seed)
            ps = _post(shadow_tag, seed)
            if pt is None or ps is None:
                continue
            ft = FE.node_features(pt, labels[ds])
            fs = FE.node_features(ps, labels[sds])
            tr, te, auc, nbal = attacker_fit(fs, mask[sds], ft, mask[ds], seed)
            tr_accs.append(tr)
            te_accs.append(te)
            aucs.append(auc)
        tr_m, tr_s = _ms(tr_accs)
        te_m, te_s = _ms(te_accs)
        au_m, au_s = _ms(aucs)
        gap = round(tr_m - te_m, 4) if tr_m != "" and te_m != "" else ""
        by_ds.setdefault(ds, []).append({
            "dataset": ds, "method": method, "ratio": ratio, "variant": variant,
            "cell": tag, "shadow_cell": shadow_tag, "n_shadow_balanced": nbal,
            "attacker_train_acc": tr_m, "train_std": tr_s,
            "attacker_test_acc": te_m, "test_std": te_s,
            "train_minus_test": gap,
            "target_transfer_auc": au_m, "auc_std": au_s,
        })

    os.makedirs(C.RESULTS, exist_ok=True)
    for ds, rows in by_ds.items():
        rows.sort(key=lambda r: (METHOD_ORDER.index(r["method"])
                                 if r["method"] in METHOD_ORDER else 99,
                                 r["ratio"], r["variant"]))
        path = os.path.join(C.RESULTS, f"{ds}_shadow_attacker_fit.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        tr = np.mean([r["attacker_train_acc"] for r in rows if r["attacker_train_acc"] != ""])
        te = np.mean([r["attacker_test_acc"] for r in rows if r["attacker_test_acc"] != ""])
        au = np.mean([r["target_transfer_auc"] for r in rows if r["target_transfer_auc"] != ""])
        print(f"{ds:9s} mean attacker train acc {tr:.3f}  shadow-test acc {te:.3f}  "
              f"target AUC {au:.3f}   ({len(rows)} cells)")
        print(f"          wrote {os.path.relpath(path, C.HERE)}")


if __name__ == "__main__":
    main()
