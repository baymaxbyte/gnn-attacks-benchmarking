"""UGC coarsening, reimplemented from the authors' UGC.py and BinWidthFinder.py.

UGC is locality-sensitive hashing, not an optimisation. Each node is projected onto
`n_hash` random vectors, the projections are floored into bins of width `bin_width`,
and a node's supernode id is the MODE of its bin indices across the projectors.
Nodes sharing a modal bin merge.

The defining feature of UGC is that it first replaces the feature matrix with an
augmented one:

    X_aug = [ (1-alpha) * X  |  alpha * A ]        shape (n, d + n)

so every node's input vector contains its own adjacency row. That augmented matrix
is what gets hashed, what the GNN trains on, and what it is served at inference.
Bin width is therefore a function of alpha.

Reference points in the authors' code:
    UGC.py:705-710    the augmentation
    UGC.py:116-143    hashed_values
    UGC.py:162-174    partition (floor, then mode across projectors)
    UGC.py:769-790    C_diag, P_hat, zero_list
    UGC.py:791-798    P = P_hat @ diag(C_diag^-1/2)
    UGC.py:865-874    cor_feat, coarse adjacency
    UGC.py:916-918    coarse labels from TRAIN labels only
    BinWidthFinder.py:63-80   the bin-width search, ratio = 1 - k/n
"""
import numpy as np
import scipy.sparse as sp
import torch

N_HASH = 1000                # run.sh default --number_of_projectors
DISTRIBUTION = "uniform"     # run.sh default --projectors_distribution
HASH_FUNCTION = "dot"        # run.sh default --hash_function


# --------------------------------------------------------------- augmentation
def augment(X, adj, alpha):
    """X_aug = [(1-alpha)X | alpha*A], dense float32. UGC.py:705-710."""
    X = torch.as_tensor(np.asarray(X.todense() if sp.issparse(X) else X),
                        dtype=torch.float32)
    A = torch.as_tensor(np.asarray(adj.todense() if sp.issparse(adj) else adj),
                        dtype=torch.float32)
    return torch.cat(((1.0 - alpha) * X, alpha * A), dim=1)


# ----------------------------------------------------------------- projections
def projectors(n_hash, feature_size, distribution=DISTRIBUTION, seed=0):
    g = torch.Generator().manual_seed(seed)
    if distribution == "normal":
        return torch.empty(n_hash, feature_size).normal_(0, 1, generator=g)
    if distribution == "VAEs":
        return torch.empty(n_hash, feature_size).normal_(-0.0017, 0.29, generator=g)
    return torch.empty(n_hash, feature_size).uniform_(0, 1, generator=g)


def hashed_values(X_aug, n_hash=N_HASH, distribution=DISTRIBUTION,
                  function=HASH_FUNCTION, seed=0):
    """Projections of every node. UGC.py:116-143."""
    Wl = projectors(n_hash, X_aug.shape[1], distribution, seed)
    if function == "L2-norm":
        return torch.cdist(X_aug, Wl, p=2)
    if function == "L1-norm":
        return torch.cdist(X_aug, Wl, p=1)
    return X_aug @ Wl.T                       # dot


def partition(bin_values, bin_width, seed=0):
    """Supernode id per node: mode of floored bin indices. UGC.py:162-174."""
    g = torch.Generator().manual_seed(seed + 1)
    n_hash = bin_values.shape[1]
    bias = (torch.rand(n_hash, generator=g) * 2 - 1) * bin_width
    binned = torch.floor((1.0 / bin_width) * (bin_values + bias))
    cluster, _ = torch.mode(binned, dim=1)
    return cluster.to(torch.long)


def achieved_ratio(assignment):
    """Fraction of nodes REMOVED, 1 - k/n. BinWidthFinder.py:78."""
    n = assignment.numel()
    return 1.0 - (torch.unique(assignment).numel() / n)


# ------------------------------------------------------------- bin-width search
def find_bin_width(X_aug, target_ratio, precision=0.0005, max_iters=80,
                   n_hash=N_HASH, distribution=DISTRIBUTION,
                   function=HASH_FUNCTION, seed=0, verbose=False):
    """Search a bin width achieving `target_ratio` (a fraction removed).

    Faithful to BinWidthFinder.Find_Binwidth: start at bw=1 and multiply by 0.5
    when the achieved ratio is too high, by 1.5 when too low. Two safeguards the
    original lacks, because that search can oscillate forever: an iteration cap,
    and we return the closest bin width seen rather than only the last one.
    """
    bin_values = hashed_values(X_aug, n_hash, distribution, function, seed)
    bw, ratio = 1.0, 1.0
    best = (float("inf"), bw, ratio)
    history = []
    for it in range(max_iters):
        if abs(ratio - target_ratio) <= precision:
            break
        bw = bw * 0.5 if ratio > target_ratio else bw * 1.5
        assign = partition(bin_values, bw, seed)
        ratio = achieved_ratio(assign)
        err = abs(ratio - target_ratio)
        history.append({"iter": it, "bin_width": bw, "ratio": ratio})
        if err < best[0]:
            best = (err, bw, ratio)
        if verbose:
            print(f"    it {it:3d}  bw {bw:.6g}  ratio {ratio:.4f}")
    return {"bin_width": best[1], "achieved_ratio": best[2], "error": best[0],
            "converged": best[0] <= precision, "iters": len(history),
            "history": history}


# ---------------------------------------------------------------- coarse graph
def coarse_graph(assignment, X_aug, adj, labels, n_classes, train_mask):
    """Build the coarse graph exactly as UGC.py:769-932.

    P_hat   hard one-hot assignment, (n, k)
    P       P_hat @ diag(C_diag^-1/2)   -- note the code divides by sqrt(size),
            not by size, despite the comment calling it an average
    X_c     P' @ X_aug
    A_c     P_hat' @ A @ P_hat, plus supernode sizes on the diagonal, minus I
    Y_c     argmax(P' @ Y) with non-training rows of Y zeroed first
    keep    supernodes containing at least one TRAINING node (the loss mask)
    """
    assignment = torch.as_tensor(assignment, dtype=torch.long)
    n = assignment.numel()
    uniq, remap = torch.unique(assignment, return_inverse=True)
    k = uniq.numel()

    P_hat = torch.zeros(n, k, dtype=torch.float32)
    P_hat[torch.arange(n), remap] = 1.0
    sizes = P_hat.sum(0)                                    # C_diag
    P = P_hat @ torch.diag(sizes.pow(-0.5))

    A = torch.as_tensor(np.asarray(adj.todense() if sp.issparse(adj) else adj),
                        dtype=torch.float32)
    A_c = P_hat.T @ A @ P_hat + torch.diag(sizes) - torch.eye(k)
    src, dst = torch.nonzero(A_c, as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = A_c[src, dst]

    X_c = P.T @ X_aug

    train_mask = torch.as_tensor(train_mask, dtype=torch.bool)
    Y = torch.eye(n_classes)[torch.as_tensor(labels, dtype=torch.long)]
    Y[~train_mask] = 0.0                                    # train labels only
    Y_c = (P.T @ Y).argmax(1)

    # a supernode is trainable iff it holds at least one training node
    keep = (P_hat[train_mask].sum(0) > 0)

    return {"P_hat": P_hat, "assignment": remap, "k": k,
            "X_c": X_c, "A_c_edge_index": edge_index, "A_c_edge_attr": edge_attr,
            "Y_c": Y_c, "keep": keep,
            "sizes": sizes,
            "coarse_edges": int(edge_index.shape[1] // 2),
            "achieved_ratio": 1.0 - k / n,
            "trainable_supernodes": int(keep.sum()),
            "max_supernode_size": int(sizes.max()),
            "mean_supernode_size": float(sizes.mean()),
            "singleton_supernodes": int((sizes == 1).sum())}
