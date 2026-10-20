"""Shared conventions for the coarsen_leakage pipeline: artifact paths, file
formats and the manifest.

Coarse graphs are written in the .npz layout lsa_paper/lsa/data.py::
load_data_external already reads, so downstream stages need no new code.
"""
import json
import os
import pickle as pkl
import sys
import time

import numpy as np
import scipy.sparse as sp

from config import ATTACK_HOME, GCOND_HOME, GCOND_PYTHON

HERE = os.path.dirname(os.path.abspath(__file__))
LSA = ATTACK_HOME
REPO_ROOT = HERE
GCOND = GCOND_HOME
GCOND_PY = GCOND_PYTHON

if LSA not in sys.path:
    sys.path.insert(0, LSA)

DIRS = {
    "assignments":   os.path.join(HERE, "artifacts", "assignments"),
    "coarse_graphs": os.path.join(HERE, "artifacts", "coarse_graphs"),
    "target_models": os.path.join(HERE, "artifacts", "target_models"),
    "posteriors":    os.path.join(HERE, "artifacts", "posteriors"),
    "results":       os.path.join(HERE, "results"),
    "logs":          os.path.join(HERE, "logs"),
}
MANIFEST = os.path.join(HERE, "manifest.json")

LSA_PRED = os.path.join(LSA, "data", "pred")
LSA_ATTACK = os.path.join(LSA, "data", "attack_dataset")
LSA_ORIGINAL = os.path.join(LSA, "data", "dataset", "original") + os.sep


def ensure_dirs():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------- manifest
def record(step, summary, inputs=None, outputs=None, numbers=None, notes=None):
    """Append one step to manifest.json; a rerun replaces its entry."""
    ensure_dirs()
    entry = {
        "step": step,
        "summary": summary,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": inputs or [],
        "outputs": outputs or [],
        "numbers": numbers or {},
        "notes": notes or "",
    }
    log = []
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            try:
                log = json.load(f).get("steps", [])
            except json.JSONDecodeError:
                log = []
    log = [e for e in log if e["step"] != step] + [entry]
    log.sort(key=lambda e: str(e["step"]))
    with open(MANIFEST, "w") as f:
        json.dump({"pipeline": "coarsen_leakage", "steps": log}, f, indent=2)
    print(f"\n[manifest] recorded step {step}: {summary}")
    return entry


def manifest_steps():
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST) as f:
        return json.load(f).get("steps", [])


def rel(path):
    """Repo-relative, so manifest paths survive moves."""
    return os.path.relpath(path, REPO_ROOT)


# ------------------------------------------------------- original dataset
def load_original(dataset="cora", label_rate=0.10, seed=123):
    """(adj, features_dense, labels_int, n_classes) via lsa_paper's loader."""
    from lsa.data import load_data
    adj, feats, y_train, y_val, y_test, m_train, m_val, m_test = load_data(
        LSA_ORIGINAL, dataset, seed=seed, label_rate=label_rate)
    X = feats.toarray() if sp.issparse(feats) else np.asarray(feats)
    labels = y_test.argmax(1)
    return adj, np.asarray(X, dtype=np.float64), labels.astype(int), \
        int(y_test.shape[1])


def edge_count(adj):
    """Undirected edges as strict upper-triangle nonzeros (self-loop safe)."""
    coo = sp.coo_matrix(adj)
    return int(((coo.row < coo.col) & (coo.data != 0)).sum())


# ----------------------------------------------------------- artifact IO
def save_assignment(method, dataset, tag, assignment, n_coarse, extra=None):
    """assignment[u] = supernode index of original node u."""
    ensure_dirs()
    a = np.asarray(assignment, dtype=np.int64).ravel()
    if a.min() < 0 or a.max() >= n_coarse:
        raise ValueError(f"assignment out of range [0,{n_coarse})")
    payload = dict(assignment=a, n_coarse=np.int64(n_coarse),
                   n_original=np.int64(a.shape[0]))
    payload.update(extra or {})
    path = os.path.join(DIRS["assignments"], f"{method}_{dataset}_{tag}.npz")
    np.savez_compressed(path, **payload)
    sizes = np.bincount(a, minlength=n_coarse)
    print(f"  [saved] {rel(path)}")
    print(f"          {a.shape[0]} original -> {n_coarse} supernodes; "
          f"sizes min={sizes.min()} max={sizes.max()} "
          f"mean={sizes.mean():.2f} singletons={(sizes == 1).sum()}")
    return path


def load_assignment(method, dataset, tag):
    path = os.path.join(DIRS["assignments"], f"{method}_{dataset}_{tag}.npz")
    z = np.load(path)
    return z["assignment"], int(z["n_coarse"])


def save_coarse_graph(method, dataset, tag, edge_index, features, labels):
    """Written in lsa_paper's external .npz format so its loader reads it directly."""
    ensure_dirs()
    ei = np.asarray(edge_index, dtype=np.int64)
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64).ravel()
    if ei.shape[0] != 2:
        raise ValueError("edge_index must be (2, E)")
    n = X.shape[0]
    if y.shape[0] != n:
        raise ValueError(f"labels {y.shape[0]} != nodes {n}")
    path = os.path.join(DIRS["coarse_graphs"], f"{method}_{dataset}_{tag}.npz")
    np.savez_compressed(path, n_nodes=np.int64(n), edge_index=ei,
                        features=X, labels=y)
    und = {(min(a, b), max(a, b)) for a, b in ei.T.tolist() if a != b}
    print(f"  [saved] {rel(path)}")
    print(f"          {n} nodes, {len(und)} undirected edges, "
          f"{X.shape[1]} features, {int(y.max()) + 1} classes")
    return path


def save_posteriors(name, posteriors):
    """Served posteriors for every original node, into lsa_paper/data/pred/."""
    os.makedirs(LSA_PRED, exist_ok=True)
    p = np.asarray(posteriors, dtype=np.float64)
    path = os.path.join(LSA_PRED, f"{name}_pred.pkl")
    with open(path, "wb") as f:
        f.write(pkl.dumps(p))
    print(f"  [saved] {rel(path)}  shape={p.shape}")
    return path


def lift_posteriors(coarse_posteriors, assignment):
    """Serve original node u with its supernode's posterior: f(s(u))."""
    P = np.asarray(coarse_posteriors)
    a = np.asarray(assignment, dtype=np.int64).ravel()
    if a.max() >= P.shape[0]:
        raise ValueError(f"assignment references supernode {a.max()} but only "
                         f"{P.shape[0]} coarse posteriors were given")
    lifted = P[a]
    uniq = len(np.unique(lifted, axis=0))
    print(f"  lifted {P.shape[0]} coarse -> {lifted.shape[0]} original "
          f"posteriors; {uniq} distinct rows "
          f"({100 * (1 - uniq / lifted.shape[0]):.1f}% are collisions)")
    return lifted


def summarise_collisions(assignment, adj):
    """How many same-supernode pairs are genuinely edges in the original graph."""
    a = np.asarray(assignment).ravel()
    A = sp.csr_matrix(adj)
    order = np.argsort(a, kind="stable")
    same_pairs = 0
    same_edges = 0
    for s in np.unique(a):
        members = order[np.searchsorted(a[order], s):
                        np.searchsorted(a[order], s, side="right")]
        k = len(members)
        if k < 2:
            continue
        same_pairs += k * (k - 1) // 2
        sub = A[members][:, members]
        same_edges += int(sub.nnz // 2)
    return dict(same_supernode_pairs=int(same_pairs),
                of_which_are_edges=int(same_edges),
                edge_fraction=(same_edges / same_pairs) if same_pairs else 0.0)
