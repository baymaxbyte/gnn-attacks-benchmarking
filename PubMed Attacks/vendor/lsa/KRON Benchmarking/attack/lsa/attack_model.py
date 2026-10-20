"""
The attack model shared by the six supervised attacks (1, 3, 4, 5, 6, 7), plus
the evaluation helpers.

Paper clause (p8):

    "We use an MLP with 3 hidden layers as our attack model. The number of
     neurons for all hidden layers is 32. ReLU is adopted as the activation
     function for hidden layers and softmax is used as the output activation
     function. We append Dropout (the rate is 0.5) to each hidden layer to
     prevent overfitting. We train 50 epochs with a learning rate of 0.001. The
     loss function is cross-entropy and the optimizer is Adam."

=> Dense(32, ReLU) x 3, each followed by Dropout(0.5), then Dense(2, softmax).

Why 3 and not 2. The authors' released Keras code builds only two 32-unit hidden
layers before the softmax output, so "3 hidden layers" could conceivably be read
as counting the output layer (the paper does exactly that for the target GCN,
which it calls a 2-hidden-layer model while footnote 3 points at Kipf's
2-weight-layer reference implementation). Two sentences rule that reading out:

  * "The number of neurons for all hidden layers is 32" -- the output layer has
    2 neurons, so it cannot be one of the hidden layers.
  * Figure 3's caption describes "the attack model's last hidden layer's output
    (32-dimension)" -- the last hidden layer is 32-wide, so the 2-wide softmax
    layer sits after it.

This module therefore uses 3 hidden layers. This is the single largest
architectural difference between the paper and the released code.

Keras defaults that the paper does not restate but that the released code
inherits, and that are reproduced here:
    Glorot-uniform kernels, zero biases
    Adam with lr=1e-3, betas=(0.9, 0.999), eps=1e-7
    batch_size 128
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler

from .spec import (ATTACK_DROPOUT, ATTACK_EPOCHS, ATTACK_HIDDEN_LAYERS,
                   ATTACK_HIDDEN_UNITS, ATTACK_LR, ATTACK_N_CLASSES)

DEFAULT_BATCH_SIZE = 128


class AttackMLP(nn.Module):
    """Paper p8: 3 x Dense(32, ReLU) + Dropout(0.5), then Dense(2, softmax).

    Logits are returned; the softmax is applied by the caller so that
    `nn.CrossEntropyLoss` can fuse it.
    """

    def __init__(self, in_dim, n_classes=ATTACK_N_CLASSES,
                 hidden=ATTACK_HIDDEN_UNITS, dropout=ATTACK_DROPOUT,
                 hidden_layers=ATTACK_HIDDEN_LAYERS):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        dims = [in_dim] + [hidden] * hidden_layers
        self.hidden = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(hidden_layers)
        )
        self.out = nn.Linear(hidden, n_classes)
        self.dropout = dropout
        # Keras Dense defaults: Glorot-uniform kernel, zero bias.
        for layer in [*self.hidden, self.out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.out(self.hidden_representation(x))

    def hidden_representation(self, x):
        """Output of the LAST hidden layer (32-dimensional).

        Figure 3 projects exactly this with t-SNE: "the attack model's last
        hidden layer's output (32-dimension) for positive node pairs and
        negative node pairs". Dropout is applied only in training mode, so
        calling this under model.eval() gives the deterministic embedding.
        """
        for layer in self.hidden:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def n_hidden_layers(self):
        return len(self.hidden)


def scale_features(x_train, x_test, independent_test_fit=True):
    """Standardise the attack features.

    The paper does not mention feature scaling at all. The released code calls
    `StandardScaler().fit_transform` on the train set and, separately, on the
    test set -- justified in the authors' README by the adversary holding the
    whole batch of target pairs at once, so their statistics are available to
    her. That behaviour is kept as the default so numbers stay comparable with
    both the released code and the paper's tables.

    Pass independent_test_fit=False for the stricter `transform`, which uses
    only training statistics. Reported as a sensitivity check, not as the
    headline configuration.
    """
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    if independent_test_fit:
        x_test = StandardScaler().fit_transform(x_test)
    else:
        x_test = ss.transform(x_test)
    return x_train, x_test


def train_and_evaluate(x_train, y_train, x_test, y_test, *,
                       epochs=ATTACK_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                       lr=ATTACK_LR, device="cpu", seed=42, verbose=True,
                       hidden_layers=ATTACK_HIDDEN_LAYERS,
                       return_model=False):
    """Train the attack MLP for one run and score it on the attack test set.

    Set return_model=True to also get the trained model back, which Figure 3
    needs in order to read the last hidden layer's activations.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train = torch.as_tensor(np.asarray(x_train), dtype=torch.float32, device=device)
    y_train = torch.as_tensor(np.asarray(y_train), dtype=torch.long, device=device)
    x_test_t = torch.as_tensor(np.asarray(x_test), dtype=torch.float32, device=device)
    y_test_np = np.asarray(y_test)

    model = AttackMLP(x_train.shape[1], hidden_layers=hidden_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-7)
    lossfn = nn.CrossEntropyLoss()

    n = x_train.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = lossfn(model(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"      epoch {epoch+1:3d}/{epochs}  loss={total/n:.4f}")

    model.eval()
    with torch.no_grad():
        prob = F.softmax(model(x_test_t), dim=1).cpu().numpy()

    p_link = prob[:, 1]
    y_pred_label = (p_link >= 0.5).astype(int)

    res = {
        "accuracy": accuracy_score(y_test_np, y_pred_label),
        "precision": precision_score(y_test_np, y_pred_label, zero_division=0),
        "recall": recall_score(y_test_np, y_pred_label, zero_division=0),
        "f1": _f1(y_test_np, y_pred_label),
        "auc": roc_auc_score(y_test_np, p_link),
    }
    if return_model:
        res["model"] = model
    return res


def last_hidden_embedding(model, x, device="cpu"):
    """32-dimensional last-hidden-layer activations, for Figure 3's t-SNE."""
    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
        return model.hidden_representation(xt).cpu().numpy()


def _f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, zero_division=0)


def unsupervised_auc(labels, scores):
    """AUC for a single distance used directly as a link predictor.

    Distance is inversely related to link probability, so a raw distance gives
    AUC < 0.5. The released code flips any AUC below 0.5, which makes each
    metric sign-agnostic and is what lets Figure 1 report all 8 metrics on the
    same axis. Kept here for the same reason.
    """
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    auc = roc_auc_score(labels, scores)
    return 1 - auc if auc < 0.5 else auc
