"""Build german.npz (whole-graph format) from the German credit graph.

Source: the NIFTY-format German credit dataset shipped with the link-stealing repo
(german.csv + german_edges.txt). We reproduce that repo's `load_nifty` feature/label
handling exactly, with one addition tailored to this benchmark:

  * features  = all csv columns EXCEPT GoodCustomer, OtherLoansAtStore, PurposeOfLoan
                (Gender mapped Female=1 / Male=0 and KEPT, so it lands at column 0)
  * each feature column is min-max normalised to [0, 1]; binary columns (including
    Gender) are unchanged, so the two-world probe injecting 0/1 into the sensitive
    column stays semantically Male-world vs Female-world
  * labels    = GoodCustomer, with the -1 class remapped to 0 (so classes are {0, 1})
  * adjacency = german_edges.txt, symmetrised, self-loops removed, binary

The output matches cora.npz / citeseer.npz: adj_* and feat_* CSR triples plus labels,
n and n_classes. Gender is the sensitive attribute at feature index 0 (see config.py).

    python build_german.py
"""
import csv
import os

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
# the German credit graph vendored under the link-stealing benchmark
SRC = os.path.abspath(os.path.join(
    HERE, "..", "..", "LSA_pytorch", "unfairness_link_stealing_attack-main",
    "data", "dataset", "german"))
CSV = os.path.join(SRC, "german.csv")
EDGES = os.path.join(SRC, "german_edges.txt")

DROP = ("GoodCustomer", "OtherLoansAtStore", "PurposeOfLoan")
LABEL = "GoodCustomer"
SENS = "Gender"


def load_rows():
    with open(CSV, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r]
    return header, rows


def main():
    header, rows = load_rows()
    col = {name: i for i, name in enumerate(header)}
    n = len(rows)

    # feature columns = csv order minus the dropped ones (Gender kept -> index 0)
    feat_cols = [name for name in header if name not in DROP]
    assert feat_cols[0] == SENS, f"expected Gender first, got {feat_cols[0]}"

    X = np.zeros((n, len(feat_cols)), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, name in enumerate(feat_cols):
            raw = r[col[name]]
            if name == SENS:                       # Female=1, Male=0
                X[i, j] = 1.0 if raw == "Female" else 0.0
            else:
                X[i, j] = float(raw)

    # min-max per column to [0,1]; binary columns (incl. Gender) are untouched
    cmin, cmax = X.min(0), X.max(0)
    span = cmax - cmin
    nz = span > 0
    X[:, nz] = (X[:, nz] - cmin[nz]) / span[nz]
    X[:, ~nz] = 0.0

    # labels: GoodCustomer in {1,-1} -> {1,0}
    labels = np.array([int(r[col[LABEL]]) for r in rows], dtype=np.int64)
    labels[labels == -1] = 0
    n_classes = int(labels.max() + 1)

    # adjacency: edge list -> symmetric binary, no self-loops
    e = np.genfromtxt(EDGES).astype(np.int64)
    if e.ndim == 1:
        e = e.reshape(-1, 2)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    adj = sp.coo_matrix((np.ones(src.shape[0], np.float32), (src, dst)),
                        shape=(n, n)).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1.0                              # binarise
    adj = adj.tocsr()

    feat = sp.csr_matrix(X)
    out = os.path.join(HERE, "german.npz")
    np.savez_compressed(
        out,
        adj_data=adj.data, adj_indices=adj.indices, adj_indptr=adj.indptr,
        adj_shape=np.array(adj.shape),
        feat_data=feat.data, feat_indices=feat.indices, feat_indptr=feat.indptr,
        feat_shape=np.array(feat.shape),
        labels=labels, n=np.array([n]), n_classes=np.array([n_classes]))

    gender = X[:, 0]
    print(f"wrote {out}")
    print(f"  n={n}  d={len(feat_cols)}  classes={n_classes}  "
          f"edges={adj.nnz // 2}")
    print(f"  label pos-rate (GoodCustomer=1): {labels.mean():.3f}")
    print(f"  sensitive col 0 = {feat_cols[0]}  binary={set(np.unique(gender).tolist()) <= {0.0, 1.0}}  "
          f"pos-rate(Female=1)={gender.mean():.3f}")
    print(f"  non-binary feature columns (min-max scaled): "
          f"{[feat_cols[j] for j in range(len(feat_cols)) if not set(np.unique(X[:, j]).tolist()) <= {0.0, 1.0}]}")


if __name__ == "__main__":
    main()
