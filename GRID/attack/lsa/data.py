"""
Dataset loading and preprocessing.

Paper clause implemented here (p7):

    "For each dataset, we train a target model and a reference model. In
     particular, we randomly sample 10% nodes and use their ground truth labels
     to train the target model and the reference model."

So the labeled set is a UNIFORM RANDOM 10% of nodes. The authors' released
`utils.py` instead takes the first 10% of node IDs, which is not a random sample
because Planetoid node ordering is not arbitrary. This module implements the
paper's version only; `lsa_pytorch/lsa/data.py` keeps the released behaviour if
you need to compare.

Everything else follows the released preprocessing, which the paper does not
restate and which is standard Kipf-GCN practice:
  * features are L1 row-normalised
  * the GCN support is the symmetrically normalised D^-1/2 (A + I) D^-1/2

Legacy APIs from the TF1 original are replaced:
  scipy.sparse.linalg.eigen.arpack.eigsh -> scipy.sparse.linalg.eigsh
  np.bool                                -> bool
"""
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from .spec import LABEL_RATE

ORIGINAL_DATASETS = ["citeseer", "cora", "pubmed"]
TU_DATASETS = ["AIDS", "COX2", "DHFR", "ENZYMES", "PROTEINS_full"]

# Coarsened / condensed graphs converted by prepare_coarsened.py live as .npz
# files and are loaded by name, so every attack works on them unchanged.
EXTERNAL_KEYS = ("n_nodes", "edge_index", "features", "labels")


# ---------------------------------------------------------------- helpers
def parse_index_file(filename):
    return [int(line.strip()) for line in open(filename)]


def sample_mask(idx, l):
    """Boolean mask of length l, True at positions idx."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)          # np.bool removed in NumPy>=1.24


def make_split(labels, seed=123, label_rate=LABEL_RATE):
    """Paper p7: uniformly sample 10% of nodes as the labeled training set.

    The remaining 90% becomes the validation set (used only for the reference
    early-stopping rule). Posteriors are later produced for EVERY node, since
    the adversary may query any node pair, so the "test" mask covers all nodes.

    `label_rate` exists for condensed graphs. GCond's synthetic graph IS the
    training set -- every synthetic node carries a label and the GNN is fitted on
    all of them -- so labelling only 10% of 70 nodes (7 nodes) would model
    something nobody deploys. Pass label_rate=1.0 for those. It stays at the
    paper's 0.10 for real datasets.
    """
    n = labels.shape[0]
    rng = np.random.RandomState(seed)
    all_ids = rng.permutation(n).tolist()      # <-- random 10%, per the paper

    n_train = n if label_rate >= 1.0 else int(n * label_rate)
    idx_train = all_ids[:n_train]
    # With label_rate=1.0 there is no held-out validation set; reuse the train
    # mask so the early-stopping rule (if enabled) still has something to read.
    idx_val = all_ids[n_train:] if n_train < n else all_ids
    idx_test = list(range(n))

    train_mask = sample_mask(idx_train, n)
    val_mask = sample_mask(idx_val, n)
    test_mask = sample_mask(idx_test, n)

    y_train, y_val, y_test = (np.zeros(labels.shape) for _ in range(3))
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return y_train, y_val, y_test, train_mask, val_mask, test_mask


# ---------------------------------------------------------------- loaders
def load_data(datapath, dataset, seed=123, label_rate=LABEL_RATE):
    # A converted coarsened/condensed graph takes precedence, so a coarsened
    # variant can be named freely (e.g. ugc_cora_r50) without colliding.
    ext = os.path.join(datapath, f"{dataset}.npz")
    if os.path.exists(ext):
        return load_data_external(ext, seed=seed, label_rate=label_rate)
    if dataset in ORIGINAL_DATASETS:
        return load_data_original(datapath, dataset, seed=seed, label_rate=label_rate)
    if dataset in TU_DATASETS:
        return load_data_tu(datapath, dataset, seed=seed, label_rate=label_rate)
    raise ValueError(
        f"Invalid dataset: {dataset}. Not one of {ORIGINAL_DATASETS}, "
        f"{TU_DATASETS}, and no converted graph at {ext}")


def load_data_external(path, seed=123, label_rate=LABEL_RATE):
    """Load a converted coarsened / condensed graph written by
    prepare_coarsened.py.

    Expected npz keys: n_nodes, edge_index (2 x E), features (n x d),
    labels (n,) as integer class ids.

    The adjacency is symmetrised, binarised and stripped of self-loops:

      * binarised because link stealing asks whether an edge EXISTS, not how
        heavily two supernodes are connected. Coarse graphs carry multiplicities
        (UGC's Cora edge weights run 1-7) which are not part of the threat model.
      * self-loops removed because a supernode's internal edges are not a link
        between two distinct nodes, and because the GCN support re-adds the
        diagonal as A + I anyway.
    """
    z = np.load(path, allow_pickle=False)
    missing = [k for k in EXTERNAL_KEYS if k not in z]
    if missing:
        raise ValueError(f"{path} is missing keys: {missing}")

    n = int(z["n_nodes"])
    ei = z["edge_index"]
    adj = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    adj = ((adj + adj.T) > 0).astype(np.float64).tolil()
    adj.setdiag(0)
    adj = sp.csr_matrix(adj)
    adj.eliminate_zeros()

    features = sp.csr_matrix(np.asarray(z["features"], dtype=np.float64))
    ids = np.asarray(z["labels"]).astype(int).ravel()
    n_classes = int(ids.max()) + 1
    labels = np.eye(n_classes)[ids]

    return (adj, features, *make_split(labels, seed=seed, label_rate=label_rate))


def load_data_original(datapath, dataset, seed=123, label_rate=LABEL_RATE):
    """Planetoid-format loader (cora / citeseer / pubmed)."""
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for nm in names:
        with open(f"{datapath}ind.{dataset}.{nm}", "rb") as f:
            objects.append(pkl.load(f, encoding="latin1") if sys.version_info > (3, 0)
                           else pkl.load(f))
    x, y, tx, ty, allx, ally, graph = objects

    test_idx_reorder = parse_index_file(f"{datapath}ind.{dataset}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == "citeseer":
        # citeseer has isolated test nodes; pad them with zero rows
        full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_ext = sp.lil_matrix((len(full), x.shape[1]))
        tx_ext[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_ext
        ty_ext = np.zeros((len(full), y.shape[1]))
        ty_ext[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_ext

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    # networkx>=3 returns a sparse array; coerce to csr matrix for consistency
    adj = sp.csr_matrix(nx.adjacency_matrix(nx.from_dict_of_lists(graph)))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    return (adj, features, *make_split(labels, seed=seed, label_rate=label_rate))


def load_data_tu(datapath, dataset, seed=123, label_rate=LABEL_RATE):
    """TU-format loader (AIDS / COX2 / DHFR / ENZYMES / PROTEINS_full)."""
    objects = []
    for nm in ["attr", "graph"]:
        with open(f"{datapath}tu/DS_all/{dataset}/{dataset}_{nm}.pkl", "rb") as f:
            objects.append(pkl.load(f, encoding="latin1"))
    attr, graph = objects

    adj = sp.csr_matrix(nx.adjacency_matrix(nx.from_dict_of_lists(graph)))
    features = np.array([attr[i]["feature_vec"] for i in range(len(attr))])
    labels = np.array([attr[i]["label"] for i in range(len(attr))])

    return (adj, features, *make_split(labels, seed=seed, label_rate=label_rate))


# ---------------------------------------------------------- preprocessing
def preprocess_features(features):
    """L1 row-normalise the feature matrix. Returns a dense float32 ndarray."""
    rowsum = np.asarray(features.sum(1), dtype=np.float64)
    r_inv = np.zeros_like(rowsum, dtype=np.float64)
    np.divide(1.0, rowsum, out=r_inv, where=rowsum != 0)
    r_inv = r_inv.flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_inv[np.isnan(r_inv)] = 0.0
    features = sp.diags(r_inv).dot(features)
    if sp.issparse(features):
        features = features.toarray()
    return np.asarray(features, dtype=np.float32)


def normalize_adj(adj):
    """Symmetric normalisation D^-1/2 A D^-1/2."""
    adj = sp.coo_matrix(adj)
    rowsum = np.asarray(adj.sum(1), dtype=np.float64)
    d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float64)
    np.divide(1.0, np.sqrt(rowsum), out=d_inv_sqrt, where=rowsum != 0)
    d_inv_sqrt = d_inv_sqrt.flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0.0
    d = sp.diags(d_inv_sqrt)
    return adj.dot(d).transpose().dot(d).tocoo()


def preprocess_adj(adj):
    """GCN support: D^-1/2 (A + I) D^-1/2 as a torch sparse tensor."""
    return sparse_to_torch(normalize_adj(adj + sp.eye(adj.shape[0])))


def sparse_to_torch(mx):
    """scipy sparse -> torch.sparse_coo_tensor (float32, coalesced)."""
    mx = sp.coo_matrix(mx)
    idx = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    val = torch.from_numpy(mx.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, torch.Size(mx.shape)).coalesce()
