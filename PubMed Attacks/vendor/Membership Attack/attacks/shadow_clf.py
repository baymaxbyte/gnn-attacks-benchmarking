"""Supervised shadow-trained membership classifier.

The shadow is the OTHER dataset at the same reduction cell, whose member /
non-member labels the adversary knows. An MLP is trained on the shadow's per-node
features to separate members from non-members, then applied to the target. Because
the features are fixed-width scalars, the classifier transfers across datasets with
different class counts. Shadow training is class-balanced so the classifier is not
biased by the 80/20 base rate; the target is scored by AUC, which is base-rate
invariant.

The attack model is the Link-Stealing paper's attack MLP (He et al., "Stealing Links
from Graph Neural Networks", USENIX Security 2021, p8):

    Dense(32, ReLU) + Dropout(0.5)  x3   ->   Dense(2, softmax)
    Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-7), cross-entropy, 50 epochs, batch 128,
    Glorot-uniform kernels, zero biases.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Link-Stealing paper p8 attack-model spec.
ATTACK_HIDDEN_UNITS = 32
ATTACK_HIDDEN_LAYERS = 3
ATTACK_DROPOUT = 0.5
ATTACK_EPOCHS = 50
ATTACK_LR = 1e-3
ATTACK_BATCH = 128


class AttackMLP(nn.Module):
    """Dense(32, ReLU) + Dropout(0.5) x hidden_layers -> Dense(n_classes).

    Logits are returned; softmax is fused into nn.CrossEntropyLoss at train time and
    applied explicitly at inference.
    """

    def __init__(self, in_dim, n_classes=2, hidden=ATTACK_HIDDEN_UNITS,
                 dropout=ATTACK_DROPOUT, hidden_layers=ATTACK_HIDDEN_LAYERS):
        super().__init__()
        dims = [in_dim] + [hidden] * hidden_layers
        self.hidden = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(hidden_layers))
        self.out = nn.Linear(hidden, n_classes)
        self.dropout = dropout
        # Keras Dense defaults the LSA code reproduces: Glorot-uniform, zero bias.
        for layer in [*self.hidden, self.out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)


def shadow_attack(shadow_feats, shadow_member, target_feats, target_member,
                  seed=0):
    """Train the LSA attack MLP on the class-balanced shadow, score AUC on target."""
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
    Xs = scaler.transform(Xs)
    Xt = scaler.transform(target_feats)

    torch.manual_seed(seed)
    np.random.seed(seed)
    Xs_t = torch.as_tensor(np.asarray(Xs), dtype=torch.float32)
    ys_t = torch.as_tensor(ys, dtype=torch.long)
    Xt_t = torch.as_tensor(np.asarray(Xt), dtype=torch.float32)

    model = AttackMLP(Xs_t.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=ATTACK_LR, betas=(0.9, 0.999),
                           eps=1e-7)
    lossfn = nn.CrossEntropyLoss()
    n = Xs_t.shape[0]
    for _ in range(ATTACK_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, ATTACK_BATCH):
            idx = perm[i:i + ATTACK_BATCH]
            opt.zero_grad()
            loss = lossfn(model(Xs_t[idx]), ys_t[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        scores = F.softmax(model(Xt_t), dim=1).numpy()[:, 1]
    return float(roc_auc_score(target_member, scores))
