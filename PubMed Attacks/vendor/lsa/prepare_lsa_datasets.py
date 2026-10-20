"""Write chameleon/squirrel into the LSA external-npz schema for every benchmark.

The LSA attack loader (`attack/lsa/data.py::load_data_external`) and the patched
benchmark-side `data.py` both read a `{dataset}.npz` with keys:

    n_nodes      int
    edge_index   (2, E) int64   -- both directions
    features     (n, d) float32
    labels       (n,)  int64

We build it from the membership benchmark's assembled whole graph
(`Membership Attack/{ds}.npz`, the PyG WikipediaNetwork geom-gcn graph), so the LSA
run uses the identical graph/node ordering as the membership run. The file is copied
into every benchmark's `attack/data/dataset/original/`, which is the datapath both
loaders use, guaranteeing the same node ordering on the benchmark side and the
attack side.

    python prepare_lsa_datasets.py                 # chameleon + squirrel, all benchmarks
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
MEMBERSHIP = os.path.abspath(os.path.join(HERE, "..", "Membership Attack"))
BENCHMARKS = ["KRON Benchmarking", "FGC Benchmarking", "UGC Benchmarking",
              "GOREN Benchmarking", "GCOND Benchmarking"]
DATASETS = ["chameleon", "squirrel"]


def _csr(z, prefix):
    return sp.csr_matrix((z[f"{prefix}_data"], z[f"{prefix}_indices"],
                          z[f"{prefix}_indptr"]), shape=tuple(z[f"{prefix}_shape"]))


def load_membership(ds):
    z = np.load(os.path.join(MEMBERSHIP, f"{ds}.npz"))
    adj = _csr(z, "adj")                         # symmetric, binary, no self-loops
    feat = _csr(z, "feat")
    labels = z["labels"].astype(np.int64)
    n = int(z["n"][0])
    return adj, feat, labels, n


def to_external(adj, feat, labels, n):
    coo = adj.tocoo()
    edge_index = np.vstack([coo.row, coo.col]).astype(np.int64)   # both directions
    features = np.asarray(feat.todense(), dtype=np.float32)
    return dict(n_nodes=np.array(n, dtype=np.int64),
                edge_index=edge_index, features=features,
                labels=labels.astype(np.int64))


def main():
    which = sys.argv[1:] or DATASETS
    for ds in which:
        adj, feat, labels, n = load_membership(ds)
        ext = to_external(adj, feat, labels, n)
        print(f"{ds}: n={n} d={ext['features'].shape[1]} "
              f"classes={int(labels.max() + 1)} E={ext['edge_index'].shape[1] // 2}")
        for b in BENCHMARKS:
            d = os.path.join(HERE, b, "attack", "data", "dataset", "original")
            if not os.path.isdir(d):
                print(f"  [skip] {b} (no {d})")
                continue
            path = os.path.join(d, f"{ds}.npz")
            np.savez_compressed(path, **ext)
            print(f"  wrote {b}/attack/data/dataset/original/{ds}.npz")


if __name__ == "__main__":
    main()
