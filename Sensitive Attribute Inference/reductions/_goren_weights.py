"""GOREN's weight-assignment network, from Cai, Wang & Wang (ICLR 2021) and the
authors' released code.

GOREN never changes the coarse topology. It learns a map M_theta assigning a weight
to each coarse edge so the coarse Laplacian better matches the original's quadratic
form. Paper section 3: M_theta is a GIN, coarse edge attributes start at 1, node
features are the 5-dimensional Local Degree Profile, a final ReLU keeps weights
positive, Adam with lr 0.001.

The loss is taken from the authors' loss_util.loss_manager.quaratic_loss rather than
from the paper's equation, because the two differ and the code is what produced the
published numbers:

    X'      = P x            P = 1/|s| averaging  (loss_util.get_sparse_projection_mat)
    diff_i  = | x_i' L1 x_i  -  X'_i' L2 X'_i |   DIAGONAL only, one term per signal
    diff_i /= x_i' pi x_i                          pi = C^T C, the rayleigh option
    loss    = mean_i diff_i

Three things here that cost us several wrong attempts, recorded so they are not
repeated:

  * loss_util uses only the DIAGONAL of (quadL1 - quadL2). The older
    loss.py::random_vec_loss sums the whole matrix and carries the authors' own
    comment "important: this is wrong!".
  * signals are the BOTTOM-K LAPLACIAN EIGENVECTORS (loss_manager is constructed
    with signal='bottomk'), not random vectors.
  * the projection is the 1/|s| averaging matrix, while the Laplacian and the
    rayleigh denominator use C with 1/sqrt(|s|). loss_util keeps these apart
    deliberately; get_sparse_C's comment reads "the major differeence".

Cora and Citeseer are disconnected (78 and 438 components), so the Laplacian null
space is that large. Every null eigenvector has quadratic form 0 and constrains no
edge weight, so they are skipped -- the authors expose the same idea as args.offset.
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv

LDP_DIM = 5
EMB_DIM = 32
N_LAYERS = 2
LR = 0.001
EPOCHS = 400
N_BOTTOMK = 40            # loss_util precomputes k=40 signals
NULL_TOL = 1e-8


# ------------------------------------------------------------------ LDP features
def ldp(edge_index, n):
    """Local Degree Profile: [deg, min, max, mean, std] over neighbour degrees."""
    src, dst = np.asarray(edge_index)
    deg = np.bincount(src, minlength=n).astype(np.float32)
    feats = np.zeros((n, LDP_DIM), dtype=np.float32)
    feats[:, 0] = deg
    order = np.argsort(src, kind="stable")
    s_sorted, d_sorted = src[order], dst[order]
    bounds = np.searchsorted(s_sorted, np.arange(n + 1))
    for v in range(n):
        nbr = d_sorted[bounds[v]:bounds[v + 1]]
        if nbr.size:
            nd = deg[nbr]
            feats[v, 1:] = (nd.min(), nd.max(), nd.mean(), nd.std())
    return feats


# ------------------------------------------------------------- spectral machinery
def laplacian(adj):
    A = sp.csr_matrix(adj, dtype=float)
    return sp.diags(np.asarray(A.sum(1)).ravel()) - A


def _dense_eigh(L):
    """Full symmetric eigendecomposition, ascending.

    ARPACK with which='SM' does not converge on these Laplacians because the null
    space is 78- and 438-dimensional. At n ~ 3000 a dense solve is exact and cheap.
    """
    return np.linalg.eigh(np.asarray(sp.csr_matrix(L).todense(), dtype=np.float64))


def bottomk(adj, k=N_BOTTOMK):
    """The k smallest NON-NULL Laplacian eigenpairs of the original graph."""
    vals, vecs = _dense_eigh(laplacian(adj))
    take = np.where(vals > NULL_TOL)[0][:k]
    return vals[take].astype(np.float32), vecs[:, take].astype(np.float32)


def bottomk_vals(L_dense, k=N_BOTTOMK):
    vals, _ = _dense_eigh(sp.csr_matrix(L_dense))
    return vals[vals > NULL_TOL][:k]


def eigenerror(vals_orig, L_coarse_dense, k=N_BOTTOMK):
    """Relative eigenvalue error, the paper's Eigenerror."""
    vc = bottomk_vals(L_coarse_dense.detach().numpy(), k)
    m = min(len(vals_orig), len(vc))
    if m == 0:
        return float("nan")
    return float(np.mean(np.abs(vals_orig[:m] - vc[:m])
                         / np.maximum(vals_orig[:m], 1e-12)))


def projection_matrix(assignment, k, n):
    """(k, n) averaging matrix, P[s, v] = 1/|s|. loss_util.get_sparse_projection_mat."""
    assignment = np.asarray(assignment)
    sizes = np.bincount(assignment, minlength=k).astype(np.float64)
    vals = 1.0 / np.maximum(sizes[assignment], 1.0)
    return sp.csr_matrix((vals, (assignment, np.arange(n))), shape=(k, n))


def sqrt_C(assignment, k, n):
    """(k, n) matrix with 1/sqrt(|s|) entries. loss_util.get_sparse_C."""
    assignment = np.asarray(assignment)
    sizes = np.bincount(assignment, minlength=k).astype(np.float64)
    vals = 1.0 / np.sqrt(np.maximum(sizes[assignment], 1.0))
    return sp.csr_matrix((vals, (assignment, np.arange(n))), shape=(k, n))


def undirected_edges(edge_index):
    """Unique (u < v) pairs plus the index mapping back to both directions."""
    src, dst = np.asarray(edge_index)
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    keyed = lo.astype(np.int64) * (int(max(hi.max(), lo.max())) + 1) + hi
    uniq, inverse = np.unique(keyed, return_inverse=True)
    return inverse.astype(np.int64), int(uniq.shape[0])


def coarse_laplacian(edge_index, weight, k):
    """L = D - W from per-directed-edge weights, differentiable."""
    src, dst = edge_index
    W = torch.zeros(k, k, dtype=weight.dtype)
    W = W.index_put((src, dst), weight, accumulate=True)
    W = 0.5 * (W + W.T)
    return torch.diag(W.sum(1)) - W


# --------------------------------------------------------------------- the model
class GOREN(nn.Module):
    """GIN node embeddings on the coarse graph, then a symmetric edge-weight head."""

    def __init__(self, node_dim=LDP_DIM, emb=EMB_DIM, layers=N_LAYERS):
        super().__init__()
        self.convs = nn.ModuleList()
        d_in = node_dim
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(d_in, emb), nn.ReLU(), nn.Linear(emb, emb))
            self.convs.append(GINEConv(mlp, edge_dim=1))
            d_in = emb
        self.head = nn.Sequential(nn.Linear(emb, emb), nn.ReLU(), nn.Linear(emb, 1))
        # Start every weight positive. Without this the output ReLU can be born dead
        # for all edges at once, after which its gradient is exactly zero forever and
        # every learned weight sticks at 0.
        nn.init.constant_(self.head[-1].bias, 1.0)

    def forward(self, x, edge_index, edge_attr):
        h = x
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index, edge_attr))
        src, dst = edge_index
        return torch.relu(self.head(h[src] + h[dst])).squeeze(-1)


def train_weights(adj, edge_index, baseline_weight, assignment, k,
                  epochs=EPOCHS, lr=LR, n_bottomk=N_BOTTOMK, seed=0,
                  rayleigh=True, verbose=False):
    """Fit M_theta on one coarse graph. Returns learned weights and diagnostics."""
    torch.manual_seed(seed)
    n = int(np.asarray(assignment).shape[0])

    vals_o, vecs_o = bottomk(adj, n_bottomk)
    x_sig = torch.from_numpy(vecs_o)                                    # (n, k_sig)
    L1 = torch.from_numpy(np.asarray(laplacian(adj).todense(), dtype=np.float32))
    Pd = torch.from_numpy(np.asarray(projection_matrix(assignment, k, n).todense(),
                                     dtype=np.float32))
    Cs = sqrt_C(assignment, k, n)
    pi = torch.from_numpy(np.asarray((Cs.T @ Cs).todense(), dtype=np.float32))

    quad1 = torch.diag(x_sig.T @ (L1 @ x_sig)).detach()
    denom = torch.diag(x_sig.T @ (pi @ x_sig)).detach().clamp_min(1e-12)
    Xp = Pd @ x_sig

    def loss_of(L2):
        quad2 = torch.diag(Xp.T @ (L2 @ Xp))
        diff = (quad1 - quad2).abs()
        if rayleigh:
            diff = diff / denom
        return diff.mean()

    ei = torch.from_numpy(np.asarray(edge_index)).long()
    inv_idx, n_undirected = undirected_edges(edge_index)
    inv_t = torch.from_numpy(inv_idx)
    feats = torch.from_numpy(ldp(edge_index, k))
    edge_attr = torch.ones(ei.shape[1], 1)          # initialised to 1, per the paper

    L_bl = coarse_laplacian(ei, torch.from_numpy(baseline_weight.astype(np.float32)), k)
    loss_bl = float(loss_of(L_bl).detach())
    ee_bl = eigenerror(vals_o, L_bl, n_bottomk)

    model = GOREN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history, best = [], (float("inf"), None)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        w_dir = model(feats, ei, edge_attr)
        # one weight per UNDIRECTED edge, shared by both directions, as
        # eval.py does with pred.repeat_interleave(2)
        w_und = torch.zeros(n_undirected, dtype=w_dir.dtype).index_put(
            (inv_t,), w_dir, accumulate=True) / 2.0
        w_sym = w_und[inv_t]
        loss = loss_of(coarse_laplacian(ei, w_sym, k))
        loss.backward()
        opt.step()
        lv = float(loss.detach())
        history.append(lv)
        if lv < best[0]:
            best = (lv, w_sym.detach().clone())
        if verbose and ep % 50 == 0:
            print(f"    epoch {ep:4d}  loss {lv:.6f}")

    w_star = best[1]
    L_goren = coarse_laplacian(ei, w_star, k)
    ee_goren = eigenerror(vals_o, L_goren, n_bottomk)

    return {
        "weight": w_star.numpy().astype(np.float32),
        "quad_loss_baseline": loss_bl, "quad_loss_goren": best[0],
        "quad_reduction": (loss_bl - best[0]) / loss_bl if loss_bl > 0 else 0.0,
        "eigenerror_baseline": ee_bl, "eigenerror_goren": ee_goren,
        "eigenerror_reduction": (ee_bl - ee_goren) / ee_bl if ee_bl > 0 else 0.0,
        "epochs": epochs, "n_bottomk": n_bottomk, "rayleigh": rayleigh,
        "n_undirected_edges": n_undirected,
        "weight_stats": {"min": float(w_star.min()), "max": float(w_star.max()),
                         "mean": float(w_star.mean()),
                         "zeros": int((w_star == 0).sum())},
        "baseline_stats": {"min": float(baseline_weight.min()),
                           "max": float(baseline_weight.max()),
                           "mean": float(baseline_weight.mean())},
        "history": history[::20],
    }
