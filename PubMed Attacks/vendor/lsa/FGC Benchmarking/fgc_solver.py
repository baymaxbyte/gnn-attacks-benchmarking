"""FGC with sparsity regulariser (FGCS), re-extracted from the authors'
Node_classification_FGC_with_sparsity.ipynb.

This is the variant behind the paper's Table 4 ("FGC with sparsity regularizer").
It is NOT the numpy solver_v2 from FGC_experiment.ipynb -- that one has the beta
term commented out of both calc_f and grad_C and L2-normalises X_tilde, and it
produced the REE/DE numbers in Table 1.

Objective (paper eq. 7 plus the sparsity term of section 4.1):

    min  tr(X~' C' L C X~) - gamma*logdet(C'LC + J) + (alpha/2)||X - C X~||_F^2
         + (lambda/2)||C 1||^2 + beta*||C' L C||_F^2

Update sequence per iteration, verbatim from the notebook's `update`:
    C~  <- max(C - k * grad_C, 0), clamped at 1e-10, then L1 row-normalised
    X~  <- pinv(C'thetaC*(2/alpha) + C'C) C' X, then L1 row-normalised

Two deliberate deviations, both performance-only:
  * the notebook's per-row Python normalisation loops are vectorised
  * float32 throughout (the notebook's convertScipyToTensor yields FloatTensor)
"""
import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import rv_continuous

THRESH = 1e-10
DENSITY = 0.15          # notebook driver cells that call get_accuracy
ITERS = 50              # notebook: for i in tqdm(range(50))
INIT_SEED = 1           # notebook: random_state=1


class _Normal(rv_continuous):
    def _rvs(self, size=None, random_state=None):
        return random_state.standard_normal(size)


def _init_pair(n, k, d, density, seed, shared_stream=True):
    """Initial (C, X_tilde).

    The notebook creates ONE frozen CustomDistribution(seed=1) at module level and
    hands temp2.rvs to both scipy.sparse.random calls. scipy invokes
    data_rvs(size) without passing a random_state, so values come from that single
    advancing stream: X_tilde draws first, then C continues from wherever X_tilde
    left off. The sparsity patterns are independent of that and fixed by
    random_state=seed.

    shared_stream=False draws each block from its own stream starting at `seed`,
    which is NOT what the notebook does.
    """
    if shared_stream:
        rv = _Normal(seed=seed)()
        X_tilde = sp.random(k, d, density=density, random_state=seed, data_rvs=rv.rvs)
        C = sp.random(n, k, density=density, random_state=seed, data_rvs=rv.rvs)
    else:
        X_tilde = sp.random(k, d, density=density, random_state=seed,
                            data_rvs=_Normal(seed=seed)().rvs)
        C = sp.random(n, k, density=density, random_state=seed,
                      data_rvs=_Normal(seed=seed)().rvs)
    return C, X_tilde


def _l1_rows(M):
    """Row-wise L1 normalisation. Equivalent to the notebook's per-row loop."""
    return M / M.abs().sum(1, keepdim=True).clamp_min(THRESH)


def coarsen(theta, X, k, lam, alpha, gamma, beta,
            iters=ITERS, density=DENSITY, seed=INIT_SEED, track=False,
            step="notebook", variant="notebook"):
    """Run FGCS. Returns (C, X_tilde, history).

    theta   : (n, n) dense Laplacian, numpy or torch
    X       : (n, d) dense features, numpy or torch
    k       : number of supernodes
    step    : gradient step scale for the C update.
              "notebook" -> g * k. The notebook computes T2 = g / L with L = 1/k.
              "paper"    -> g / k. Appendix B.3 states a learning rate of 1/k,
                            which is the reciprocal of what the notebook applies.
              float      -> g * that value (solver_v2 uses a fixed 1e-5).
    variant : which block-update order to use.
              "notebook" -> X_tilde is recomputed from the OLD C, and C is
                            thresholded and row-normalised only afterwards. This is
                            literally what the notebook's `update` does (Jacobi).
              "paper"    -> X_tilde is recomputed from the NEW, normalised C
                            (Gauss-Seidel), which is what a BSUM alternating
                            minimisation normally means.
    """
    if step == "notebook":
        step_scale = float(k)
    elif step == "paper":
        step_scale = 1.0 / k
    else:
        step_scale = float(step)
    theta = torch.as_tensor(np.asarray(theta), dtype=torch.float32)
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    n, d = X.shape

    C0, Xt0 = _init_pair(n, k, d, density, seed)
    C = torch.as_tensor(C0.toarray(), dtype=torch.float32)
    X_tilde = torch.as_tensor(Xt0.toarray(), dtype=torch.float32)

    J = torch.full((k, k), 1.0 / k)
    ones = torch.ones(k, k)
    zeros = torch.zeros(n, k)
    history = []

    for it in range(iters):
        C_prev = C
        thetaC = theta @ C
        CT = C.T
        X_tildeT = X_tilde.T

        term_bracket = torch.linalg.pinv(CT @ thetaC + J)

        g = (-2.0 * gamma * (thetaC @ term_bracket)
             + alpha * (C @ X_tilde - X) @ X_tildeT
             + 2.0 * (thetaC @ X_tilde) @ X_tildeT
             + lam * (C @ ones)
             + 2.0 * beta * (thetaC @ CT @ thetaC))

        C_new = (C - g * step_scale).maximum(zeros)

        if variant == "notebook":
            # X_tilde from the OLD C, before C_new is thresholded/normalised
            A = torch.linalg.pinv(CT @ thetaC * (2.0 / alpha) + CT @ C)
            X_new = A @ CT @ X
            C_new = torch.where(C_new < THRESH, torch.full_like(C_new, THRESH), C_new)
            C = _l1_rows(C_new)
            X_tilde = _l1_rows(X_new)
        elif variant == "paper":
            C_new = torch.where(C_new < THRESH, torch.full_like(C_new, THRESH), C_new)
            C = _l1_rows(C_new)
            A = torch.linalg.pinv(C.T @ (theta @ C) * (2.0 / alpha) + C.T @ C)
            X_tilde = _l1_rows(A @ C.T @ X)
        else:
            raise ValueError(f"unknown variant: {variant}")

        if track:
            assign = C.argmax(1)
            history.append({
                "iter": it,
                "delta_C": float(torch.linalg.norm(C - C_prev)),
                "nonempty_supernodes": int(torch.unique(assign).numel()),
                "max_row_mass": float(C.max(1).values.mean()),
            })

    return C, X_tilde, history


# ------------------------------------------------------------------ coarse graph
def harden(C):
    """C_hard[i, argmax C[i]] = 1. The notebook's hardening step."""
    C = torch.as_tensor(C)
    n, k = C.shape
    H = torch.zeros(n, k, dtype=torch.float32)
    H[torch.arange(n), C.argmax(1)] = 1.0
    return H


def coarse_graph(C_hard, theta, X, labels, n_classes, edge_threshold=0.1):
    """Build (A_c, X_c, Y_c, assignment) exactly as the notebook's get_accuracy.

    A_c : binary. W_c = -offdiag(C'theta C), thresholded at 0.1, then binarised
          because the notebook computes edge_weight but never passes it to GCNConv.
    X_c : pinv(C_hard) @ X. For a one-hot C this is the per-supernode feature mean.
    Y_c : argmax(pinv(C_hard) @ onehot(Y)) -- the supernode majority label.
    """
    C_hard = torch.as_tensor(C_hard, dtype=torch.float32)
    theta = torch.as_tensor(np.asarray(theta), dtype=torch.float32)
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    k = C_hard.shape[1]

    Lc = C_hard.T @ theta @ C_hard
    Wc = (-1.0 * Lc) * (1.0 - torch.eye(k))
    Wc = torch.where(Wc < edge_threshold, torch.zeros_like(Wc), Wc)

    src, dst = torch.nonzero(Wc, as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)

    P = torch.linalg.pinv(C_hard)
    X_c = P @ X

    Y_onehot = torch.eye(n_classes)[torch.as_tensor(labels, dtype=torch.long)]
    Y_c = (P @ Y_onehot).argmax(1)

    assignment = C_hard.argmax(1)
    sizes = torch.bincount(assignment, minlength=k)

    return {
        "edge_index": edge_index,
        "X_c": X_c,
        "Y_c": Y_c,
        "assignment": assignment,
        "Wc_nnz": int(torch.count_nonzero(Wc)),
        "coarse_edges": int(edge_index.shape[1] // 2),
        "k": k,
        "nonempty_supernodes": int((sizes > 0).sum()),
        "max_supernode_size": int(sizes.max()),
        "mean_nonempty_size": float(sizes[sizes > 0].float().mean()),
        "singleton_supernodes": int((sizes == 1).sum()),
    }
