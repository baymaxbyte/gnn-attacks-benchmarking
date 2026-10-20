"""
Target models f and the reference model g.

Paper clauses (p8):

    "The number of neurons in the hidden layer is set to 16. We adopt the
     frequently used ReLU and softmax as activation functions for the first
     hidden layer and the second hidden layer, respectively. Note that we
     append Dropout (the rate is 0.5) to the output of the hidden layer to
     prevent overfitting. We train 100 epochs with a learning rate of 0.01.
     Cross-entropy is adopted as the loss function and we use the Adam
     optimizer to update the model parameters. Our GNNs are implemented based
     on publicly available code."   [footnote 3: https://github.com/tkipf/gcn]

    "We use an MLP with 2 hidden layers as the reference model and the shadow
     reference model. Hyperparameters, including the number of neurons in the
     hidden layer, activation functions, loss function, optimizer, epochs, and
     learning rate are the same as those of the target model."

    p3: "Note that in most of the paper, we focus on two-layer GCN. Later, we
         show that our attack can be also performed on other types of GNNs,
         including GraphSAGE and GAT (see Section 5)."

Note on the paper's layer counting: for the GNNs and the reference model,
"2 hidden layers" includes the softmax output layer, since footnote 3 points at
Kipf's reference GCN, which has exactly two weight layers (in->16 ReLU,
16->C softmax). The paper uses the opposite convention for the attack model --
see attack_model.py. `n_layers` below therefore counts WEIGHT layers, and
n_layers=2 is the paper's default. Table 16 needs n_layers=3 for the shadow
target model.

Four architectures:

    GCN         the default target model f            (Table 4-10, 15-22)
    MLP         the reference model g; ignores the graph entirely
    GraphSAGE   alternative target model              (Table 11)
    GAT         alternative target model              (Table 11)

Implementation details carried over from the released TF1 code for GCN and MLP,
which the paper does not restate:
  * NO bias terms (the TF layers default to bias=False)
  * Glorot-uniform initialisation
  * dropout applied to the INPUT of every layer
  * weight decay 5e-4 on first-layer weights only, as 5e-4 * sum(W^2)/2
    (TF's tf.nn.l2_loss is sum(t**2)/2, not sum(t**2))
  * masked softmax cross-entropy == mean of the loss over the masked nodes

GraphSAGE and GAT are not in the released code at all; the paper cites the
authors' own repositories (footnotes 5 and 6), so those structures are followed
and the choice is recorded in each class docstring.
"""
import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import normalize_adj, sparse_to_torch
from .spec import TARGET_DROPOUT, TARGET_HIDDEN


def glorot(shape):
    """Glorot/Xavier uniform init, matching the original inits.py."""
    init_range = math.sqrt(6.0 / (shape[0] + shape[1]))
    return nn.Parameter(torch.empty(*shape).uniform_(-init_range, init_range))


# ============================================================== GCN / MLP
class GraphConvolution(nn.Module):
    """output = act( support @ (x @ W) ), no bias."""

    def __init__(self, in_dim, out_dim, dropout=0.0, act=F.relu):
        super().__init__()
        self.weight = glorot((in_dim, out_dim))
        self.dropout = dropout
        self.act = act

    def forward(self, x, support):
        x = F.dropout(x, p=self.dropout, training=self.training)
        pre_sup = torch.mm(x, self.weight)
        out = torch.sparse.mm(support, pre_sup)
        return self.act(out) if self.act is not None else out


class Dense(nn.Module):
    """output = act( x @ W ), no bias."""

    def __init__(self, in_dim, out_dim, dropout=0.0, act=F.relu):
        super().__init__()
        self.weight = glorot((in_dim, out_dim))
        self.dropout = dropout
        self.act = act

    def forward(self, x, support=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = torch.mm(x, self.weight)
        return self.act(out) if self.act is not None else out


def _widths(in_dim, n_classes, hidden, n_layers):
    """Weight-layer widths: in -> hidden -> ... -> n_classes."""
    if n_layers < 2:
        raise ValueError("n_layers counts weight layers and must be >= 2")
    return [in_dim] + [hidden] * (n_layers - 1) + [n_classes]


class GCN(nn.Module):
    """The default target model f. n_layers weight layers, hidden = 16.

    n_layers=2 is the paper's target model. Table 16 uses n_layers=3 for the
    shadow target model, to test whether the transferring attacks still work
    when the adversary's shadow architecture differs from the target's.
    """

    def __init__(self, in_dim, n_classes, hidden=TARGET_HIDDEN,
                 dropout=TARGET_DROPOUT, n_layers=2):
        super().__init__()
        w = _widths(in_dim, n_classes, hidden, n_layers)
        self.layers = nn.ModuleList(
            GraphConvolution(w[i], w[i + 1], dropout,
                             act=F.relu if i < len(w) - 2 else None)
            for i in range(len(w) - 1))

    @staticmethod
    def preprocess(adj):
        """GCN support: D^-1/2 (A + I) D^-1/2."""
        return sparse_to_torch(normalize_adj(adj + sp.eye(adj.shape[0])))

    def forward(self, x, support):
        for layer in self.layers:
            x = layer(x, support)
        return x                                # logits

    def first_layer_weights(self):
        return [self.layers[0].weight]


class MLP(nn.Module):
    """The reference model g. Same shape as f but ignores the graph entirely.

    Section 4: g is trained on the target dataset's node attributes and the same
    small set of known labels, WITHOUT the graph. Comparing f's and g's
    posteriors isolates the information the graph contributed, which is what
    Attack-2 and the F = Y attacks exploit.
    """

    def __init__(self, in_dim, n_classes, hidden=TARGET_HIDDEN,
                 dropout=TARGET_DROPOUT, n_layers=2):
        super().__init__()
        w = _widths(in_dim, n_classes, hidden, n_layers)
        self.layers = nn.ModuleList(
            Dense(w[i], w[i + 1], dropout,
                  act=F.relu if i < len(w) - 2 else None)
            for i in range(len(w) - 1))

    @staticmethod
    def preprocess(adj):
        return None                             # g never sees the graph

    def forward(self, x, support=None):
        for layer in self.layers:
            x = layer(x)
        return x                                # logits

    def first_layer_weights(self):
        return [self.layers[0].weight]


# =============================================================== GraphSAGE
class SageConv(nn.Module):
    """GraphSAGE layer with the mean aggregator.

        h_v = act( W . CONCAT(h_v, MEAN_{u in N(v)} h_u) )

    which is the `mean` aggregator of williamleif/GraphSAGE (paper footnote 5),
    written full-batch. `support` is the row-normalised adjacency D^-1 A with NO
    self-loops, so the self term stays separate from the neighbour mean, as in
    the original.
    """

    def __init__(self, in_dim, out_dim, dropout=0.0, act=F.relu):
        super().__init__()
        self.weight = glorot((2 * in_dim, out_dim))
        self.dropout = dropout
        self.act = act

    def forward(self, x, support):
        x = F.dropout(x, p=self.dropout, training=self.training)
        neigh = torch.sparse.mm(support, x)
        out = torch.mm(torch.cat([x, neigh], dim=1), self.weight)
        return self.act(out) if self.act is not None else out


class GraphSAGE(nn.Module):
    """Alternative target model for Table 11.

    Structure follows williamleif/GraphSAGE (mean aggregator, 2 layers); the
    width, dropout, epochs and learning rate stay at the paper's target-model
    values so that only the aggregation scheme differs.
    """

    def __init__(self, in_dim, n_classes, hidden=TARGET_HIDDEN,
                 dropout=TARGET_DROPOUT, n_layers=2):
        super().__init__()
        w = _widths(in_dim, n_classes, hidden, n_layers)
        self.layers = nn.ModuleList(
            SageConv(w[i], w[i + 1], dropout,
                     act=F.relu if i < len(w) - 2 else None)
            for i in range(len(w) - 1))

    @staticmethod
    def preprocess(adj):
        """Row-normalised D^-1 A, no self-loops (mean over true neighbours)."""
        adj = sp.coo_matrix(adj)
        rowsum = np.asarray(adj.sum(1), dtype=np.float64).flatten()
        inv = np.zeros_like(rowsum)
        np.divide(1.0, rowsum, out=inv, where=rowsum != 0)
        return sparse_to_torch(sp.diags(inv).dot(adj))

    def forward(self, x, support):
        for layer in self.layers:
            x = layer(x, support)
        return x

    def first_layer_weights(self):
        return [self.layers[0].weight]


# ===================================================================== GAT
class GATConv(nn.Module):
    """Multi-head graph attention, computed sparsely over the edge list.

        e_ij    = LeakyReLU_0.2( a_src . Wh_i + a_dst . Wh_j )
        alpha   = softmax_j over N(i) of e_ij
        h'_i    = sum_j alpha_ij Wh_j

    Structure follows PetarV-/GAT (paper footnote 6). Attention is computed on
    the edge list rather than an N x N matrix so Pubmed (19,717 nodes) stays
    tractable -- dense attention would need ~1.5 GB per head.
    """

    def __init__(self, in_dim, out_dim, heads=1, dropout=0.0, act=F.elu,
                 concat=True, negative_slope=0.2):
        super().__init__()
        self.heads, self.out_dim, self.concat = heads, out_dim, concat
        self.dropout, self.act, self.negative_slope = dropout, act, negative_slope
        self.weight = glorot((in_dim, heads * out_dim))
        self.att_src = glorot((heads, out_dim))
        self.att_dst = glorot((heads, out_dim))

    def forward(self, x, edge_index):
        dst, src = edge_index[0], edge_index[1]
        n = x.shape[0]

        x = F.dropout(x, p=self.dropout, training=self.training)
        h = torch.mm(x, self.weight).view(n, self.heads, self.out_dim)

        # unnormalised attention logits per edge, per head
        alpha = ((h * self.att_src).sum(-1)[dst]
                 + (h * self.att_dst).sum(-1)[src])
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # softmax over the incoming edges of each destination node
        amax = torch.full((n, self.heads), float("-inf"), device=x.device)
        amax = amax.scatter_reduce(0, dst.unsqueeze(-1).expand_as(alpha), alpha,
                                   reduce="amax", include_self=False)
        alpha = torch.exp(alpha - amax[dst])
        denom = torch.zeros((n, self.heads), device=x.device)
        denom.index_add_(0, dst, alpha)
        alpha = alpha / (denom[dst] + 1e-16)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros((n, self.heads, self.out_dim), device=x.device)
        out.index_add_(0, dst, h[src] * alpha.unsqueeze(-1))

        out = out.reshape(n, self.heads * self.out_dim) if self.concat \
            else out.mean(dim=1)
        return self.act(out) if self.act is not None else out


class GAT(nn.Module):
    """Alternative target model for Table 11.

    The paper cites only PetarV-/GAT, so that repository's Cora configuration is
    used: 8 attention heads of 8 features in the hidden layer, then a single
    averaged head for the output. Dropout, epochs and learning rate stay at the
    paper's target-model values. Override with --heads / --gat_head_dim.
    """

    def __init__(self, in_dim, n_classes, hidden=8, dropout=TARGET_DROPOUT,
                 heads=8, n_layers=2):
        super().__init__()
        if n_layers != 2:
            raise ValueError("GAT here is the 2-layer structure of PetarV-/GAT")
        self.layer0 = GATConv(in_dim, hidden, heads=heads, dropout=dropout,
                              act=F.elu, concat=True)
        self.layer1 = GATConv(hidden * heads, n_classes, heads=1,
                              dropout=dropout, act=None, concat=False)

    @staticmethod
    def preprocess(adj):
        """Edge list with self-loops added, as [2, E] with row 0 = destination."""
        a = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
        return torch.from_numpy(np.vstack((a.row, a.col)).astype(np.int64))

    def forward(self, x, edge_index):
        h = self.layer0(x, edge_index)
        return self.layer1(h, edge_index)       # logits

    def first_layer_weights(self):
        return [self.layer0.weight]


ARCHITECTURES = {"gcn": GCN, "dense": MLP, "sage": GraphSAGE, "gat": GAT}

# Which architectures are target models vs the reference model.
TARGET_ARCHITECTURES = ("gcn", "sage", "gat")
REFERENCE_ARCHITECTURE = "dense"


# ================================================================== losses
def masked_cross_entropy(logits, labels_onehot, mask):
    """Mean softmax cross-entropy over the masked nodes.

    Equivalent to the TF reference, which multiplies by mask / mean(mask) and
    then takes the global mean.
    """
    targets = labels_onehot[mask].argmax(dim=1)
    return F.cross_entropy(logits[mask], targets)


def masked_accuracy(logits, labels_onehot, mask):
    pred = logits[mask].argmax(dim=1)
    targets = labels_onehot[mask].argmax(dim=1)
    return (pred == targets).float().mean()


def l2_reg(model, weight_decay):
    """weight_decay * sum(W^2)/2 over first-layer weights only (TF l2_loss)."""
    reg = 0.0
    for w in model.first_layer_weights():
        reg = reg + weight_decay * torch.sum(w ** 2) / 2.0
    return reg


# ======================================================= top-k defense
def apply_topk(posteriors, k, renormalize=False):
    """Section 6 defense: the model reports only its k largest posteriors.

    Paper (Table 12): "we fix k = 2 and report the results for Attack-3 ...
    Experimental results in Table 12 show that this defense indeed reduces the
    performance of our attack. However, the performance drop is not very big."

    Entries outside the top k are set to 0, which is what the adversary sees
    when the API returns only k values. The paper does not say whether the
    remaining values are rescaled, so renormalisation is off by default and
    available as a flag.

    Exactly k entries survive per row. Selecting by index rather than by a
    value threshold matters: with a threshold, a row like [0.1, 0.1, 0.7, 0.1]
    and k=2 would keep all four values, since three of them tie at the cutoff.
    """
    p = np.array(posteriors, dtype=np.float64, copy=True)
    if k >= p.shape[1]:
        return p
    keep = np.argpartition(-p, k - 1, axis=1)[:, :k]
    mask = np.zeros(p.shape, dtype=bool)
    np.put_along_axis(mask, keep, True, axis=1)
    p[~mask] = 0.0
    if renormalize:
        s = p.sum(axis=1, keepdims=True)
        np.divide(p, s, out=p, where=s != 0)
    return p
