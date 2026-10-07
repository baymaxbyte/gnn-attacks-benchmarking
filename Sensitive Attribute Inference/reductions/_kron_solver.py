"""Kron reduction of a graph (Dorfler & Bullo, arXiv 1102.2950).

Given the combinatorial Laplacian L of a weighted graph and a split of the nodes
into a BOUNDARY set alpha (kept) and an INTERIOR set beta (eliminated), the Kron
reduced Laplacian is the Schur complement of L with respect to the interior:

    L_red = L[a,a] - L[a,b] @ inv(L[b,b]) @ L[b,a]        (paper eq. 1.2)

The paper proves L_red is again a well-defined loopy Laplacian, so it induces a
weighted graph on alpha with A_red[i,j] = -L_red[i,j]. Our input adjacency has no
self-loops, so L is loop-less; then L @ 1 = 0 forces L_red @ 1 = 0 as well, meaning
the reduced graph has no shunt-to-ground either: A_red[i,j] = -L_red[i,j] >= 0 and
diag(L_red) is exactly the weighted degree. step01 asserts max|L_red @ 1| ~ 0 as a
correctness check.

Two implementation facts that keep the Schur complement well posed and cheap:

  * We choose the interior PER CONNECTED COMPONENT, keeping the highest-degree
    ceil((1-r) * size) nodes of each component (at least one). Because every
    component then retains a boundary node, each interior block L[b,b] is the
    grounded Laplacian of a connected graph, which is symmetric positive definite;
    the solve uses a Cholesky path.
  * L is block-diagonal across components, so we reduce each component on its own
    dense block and stitch the results, never forming a dense n x n matrix.
"""
import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve
from scipy.sparse.csgraph import connected_components

WEIGHT_EPS = 1e-8          # reduced edges with weight below this are dropped


def select_boundary(adj, r, selection="degree"):
    """Return a boolean mask over nodes: True = boundary (kept), False = interior.

    Keep a global budget of target_k = round((1-r) * n) nodes, chosen as the
    highest-degree nodes so the reduced graph is the well-connected backbone. Two
    constraints are then enforced so the Schur complement stays well posed and the
    node count still matches the target:

      * every connected component must retain at least one boundary node (otherwise
        that component's interior Laplacian block is singular). Components with none
        selected get their highest-degree node promoted.
      * promotions would push the count past target_k, so an equal number of the
        lowest-degree kept nodes are demoted back to the interior, never emptying a
        component. The result keeps exactly target_k nodes whenever that is >= the
        number of components.
    """
    if selection != "degree":
        raise ValueError(f"unknown selection: {selection}")
    A = sp.csr_matrix(adj)
    n = A.shape[0]
    deg = np.asarray(A.sum(1)).ravel()
    ncomp, comp = connected_components(A, directed=False)
    target_k = min(max(int(round((1.0 - r) * n)), ncomp), n)

    ranked = np.lexsort((np.arange(n), -deg))      # degree desc, id asc
    keep = np.zeros(n, dtype=bool)
    keep[ranked[:target_k]] = True

    kept_per = np.bincount(comp[keep], minlength=ncomp)
    for c in np.where(kept_per == 0)[0]:
        idx = np.where(comp == c)[0]
        best = idx[np.lexsort((idx, -deg[idx]))][0]
        keep[best] = True
        kept_per[c] += 1

    over = int(keep.sum()) - target_k
    if over > 0:
        kept_nodes = np.where(keep)[0]
        # lowest degree first, id desc, as demotion candidates
        cand = kept_nodes[np.lexsort((-kept_nodes, deg[kept_nodes]))]
        for v in cand:
            if over == 0:
                break
            c = comp[v]
            if kept_per[c] > 1:
                keep[v] = False
                kept_per[c] -= 1
                over -= 1
    return keep, comp, ncomp


def _component_laplacian(A, idx):
    """Dense combinatorial Laplacian of the subgraph on `idx`, in the order given."""
    sub = A[idx][:, idx]
    sub = np.asarray(sub.todense(), dtype=np.float64)
    return np.diag(sub.sum(1)) - sub


def kron_reduce(adj, r, selection="degree"):
    """Kron-reduce `adj` by eliminating a fraction r of nodes.

    Returns a dict with the reduced graph in LOCAL indexing over the kept nodes:
        kept_idx      original node ids of the boundary set, sorted ascending
        edge_index    (2, 2E) int64, both directions
        edge_weight   (2E,) float32, the Schur-complement conductances
    plus diagnostics (fill-in, weight range, Laplacian residual, component stats).
    """
    A = sp.csr_matrix(adj, dtype=np.float64)
    n = A.shape[0]
    keep, comp, ncomp = select_boundary(A, r, selection)

    kept_idx = np.where(keep)[0]
    k = kept_idx.shape[0]
    g2l = -np.ones(n, dtype=np.int64)          # global id -> local reduced id
    g2l[kept_idx] = np.arange(k)

    rows, cols, vals = [], [], []
    min_raw_weight = np.inf                    # most negative off-diagonal seen
    clamped = 0
    lap_residual = 0.0
    solved_components = 0
    max_interior = 0

    for c in range(ncomp):
        idx = np.where(comp == c)[0]
        comp_keep = idx[keep[idx]]
        comp_drop = idx[~keep[idx]]

        if comp_drop.size == 0:
            # nothing eliminated: carry the component's original edges through
            if comp_keep.size > 1:
                sub = sp.triu(A[comp_keep][:, comp_keep], k=1).tocoo()
                for a, b, w in zip(sub.row, sub.col, sub.data):
                    la, lb = g2l[comp_keep[a]], g2l[comp_keep[b]]
                    rows += [la, lb]
                    cols += [lb, la]
                    vals += [w, w]
            continue

        # order the component as [boundary, interior] and build its dense Laplacian
        order = np.concatenate([comp_keep, comp_drop])
        L = _component_laplacian(A, order)
        na = comp_keep.shape[0]
        Laa, Lab = L[:na, :na], L[:na, na:]
        Lbb, Lba = L[na:, na:], L[na:, :na]
        max_interior = max(max_interior, comp_drop.shape[0])

        try:
            X = solve(Lbb, Lba, assume_a="pos")
        except np.linalg.LinAlgError:
            X = solve(Lbb, Lba, assume_a="sym")
        L_red = Laa - Lab @ X
        solved_components += 1

        lap_residual = max(lap_residual, float(np.abs(L_red.sum(1)).max()))

        # off-diagonal of L_red -> reduced edges; A_red = -L_red
        iu, ju = np.triu_indices(na, k=1)
        w_off = -L_red[iu, ju]
        if w_off.size:
            min_raw_weight = min(min_raw_weight, float(w_off.min()))
        neg = w_off < 0
        clamped += int((neg & (w_off > -1e-6)).sum())
        w_off = np.where(neg, 0.0, w_off)
        sig = w_off > WEIGHT_EPS
        for a, b, w in zip(iu[sig], ju[sig], w_off[sig]):
            la, lb = g2l[comp_keep[a]], g2l[comp_keep[b]]
            rows += [la, lb]
            cols += [lb, la]
            vals += [w, w]

    if rows:
        edge_index = np.vstack([rows, cols]).astype(np.int64)
        edge_weight = np.asarray(vals, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_weight = np.zeros(0, dtype=np.float32)

    reduced_edges = edge_index.shape[1] // 2
    orig_edges = int(A.nnz // 2)
    return {
        "kept_idx": kept_idx,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "k": int(k),
        "n": int(n),
        "achieved_ratio": 1.0 - k / n,
        "keep_fraction": k / n,
        "reduced_edges": int(reduced_edges),
        "original_edges": orig_edges,
        "fill_in_ratio": (reduced_edges / orig_edges) if orig_edges else 0.0,
        "n_components": int(ncomp),
        "solved_components": int(solved_components),
        "max_interior_block": int(max_interior),
        "edge_weight_min": float(edge_weight.min()) if edge_weight.size else 0.0,
        "edge_weight_max": float(edge_weight.max()) if edge_weight.size else 0.0,
        "edge_weight_mean": float(edge_weight.mean()) if edge_weight.size else 0.0,
        "min_raw_offdiag": (0.0 if not np.isfinite(min_raw_weight)
                            else float(min_raw_weight)),
        "clamped_negatives": int(clamped),
        "laplacian_residual": float(lap_residual),
    }


def summary(rd):
    return (f"k={rd['k']:5d}  keep={rd['keep_fraction']:.4f}  "
            f"r*={rd['achieved_ratio']:.4f}  "
            f"edges {rd['original_edges']:5d}->{rd['reduced_edges']:6d} "
            f"(fill x{rd['fill_in_ratio']:.2f})  "
            f"w[{rd['edge_weight_min']:.3g},{rd['edge_weight_max']:.3g}]  "
            f"|L1|={rd['laplacian_residual']:.1e}")
