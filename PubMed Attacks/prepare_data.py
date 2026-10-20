"""Download PubMed (and, if missing, Cora) from code and write them in BOTH npz
schemas the benchmarks use, so no manual data step is ever needed.

  - Membership schema  (CSR adj/feat + labels + n/n_classes/n_edges):
        <Membership Attack>/<ds>.npz
  - LSA external schema (n_nodes, edge_index, features, labels):
        <lsa>/<METHOD> Benchmarking/attack/data/dataset/original/<ds>.npz

PubMed is built once from PyG Planetoid, then the LSA file is derived from the
SAME assembled graph so node ordering is identical on both sides (this is exactly
how chameleon/squirrel were prepared -- see prepare_datasets.py / prepare_lsa_datasets.py).

Idempotent: existing files are kept unless --force. Cora is normally already
present (shipped cora.npz + ind.cora.* pickles); we only (re)build what's missing.

    python prepare_data.py                 # pubmed (+ cora if missing)
    python prepare_data.py --force         # rebuild
    python prepare_data.py --datasets pubmed,cora
"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp

import config as C

# PyG Planetoid canonical names.
PLANETOID = {"pubmed": "PubMed", "cora": "Cora", "citeseer": "CiteSeer"}
PYG_CACHE = os.path.join(C.HERE, "_pyg_cache")


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------- assemble (PyG)
def assemble_planetoid(ds):
    """Return (adj csr: symmetric/binary/no-self-loops, feat csr float32,
    labels int64, n) for a Planetoid dataset, matching the benchmarks'
    normalisation exactly."""
    from torch_geometric.datasets import Planetoid
    from torch_geometric.utils import to_scipy_sparse_matrix
    root = os.path.join(PYG_CACHE, ds)
    g = Planetoid(root=root, name=PLANETOID[ds])[0]
    n = int(g.num_nodes)
    adj = to_scipy_sparse_matrix(g.edge_index, num_nodes=n).tocsr()
    adj = adj.maximum(adj.T)          # symmetric
    adj.setdiag(0)                    # no self-loops
    adj.eliminate_zeros()
    adj.data[:] = 1.0                 # binary
    adj = adj.tocsr()
    feat = sp.csr_matrix(np.asarray(g.x.numpy(), dtype=np.float32))
    labels = np.asarray(g.y.numpy(), dtype=np.int64).ravel()
    return adj, feat, labels, n


# ------------------------------------------------------------- membership npz
def write_membership_npz(ds, adj, feat, labels, n, force=False):
    path = os.path.join(C.MEMBERSHIP_ROOT, f"{ds}.npz")
    if os.path.exists(path) and not force:
        log(f"  [keep] membership {ds}.npz (exists)")
        return path
    if not os.path.isdir(C.MEMBERSHIP_ROOT):
        log(f"  [skip] membership root missing: {C.MEMBERSHIP_ROOT}")
        return None
    np.savez_compressed(
        path,
        adj_data=adj.data, adj_indices=adj.indices, adj_indptr=adj.indptr,
        adj_shape=np.array(adj.shape),
        feat_data=feat.data, feat_indices=feat.indices, feat_indptr=feat.indptr,
        feat_shape=np.array(feat.shape),
        labels=labels, n=np.array([n]),
        n_classes=np.array([int(labels.max() + 1)]),
        n_edges=np.array([int(adj.nnz // 2)]))
    log(f"  wrote membership {ds}.npz")
    return path


# ------------------------------------------------------------- LSA external npz
def write_lsa_npz(ds, adj, feat, labels, n, force=False):
    coo = adj.tocoo()
    edge_index = np.vstack([coo.row, coo.col]).astype(np.int64)   # both directions
    features = np.asarray(feat.todense(), dtype=np.float32)
    ext = dict(n_nodes=np.array(n, dtype=np.int64),
               edge_index=edge_index, features=features,
               labels=labels.astype(np.int64))
    written = []
    for spec in C.LSA_SPECS.values():
        d = os.path.join(C.LSA_ROOT, spec["dir"], "attack", "data",
                         "dataset", "original")
        if not os.path.isdir(os.path.join(C.LSA_ROOT, spec["dir"])):
            log(f"  [skip] {spec['dir']} not present")
            continue
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{ds}.npz")
        if os.path.exists(path) and not force:
            log(f"  [keep] {spec['dir']}/.../{ds}.npz (exists)")
            written.append(path)
            continue
        np.savez_compressed(path, **ext)
        log(f"  wrote {spec['dir']}/attack/data/dataset/original/{ds}.npz")
        written.append(path)
    return written


# --------------------------------------------------------------- verification
def verify(ds):
    """Reload the membership npz and sanity-check shapes/counts."""
    path = os.path.join(C.MEMBERSHIP_ROOT, f"{ds}.npz")
    if not os.path.exists(path):
        return
    z = np.load(path)
    n = int(z["n"][0])
    adj = sp.csr_matrix((z["adj_data"], z["adj_indices"], z["adj_indptr"]),
                        shape=tuple(z["adj_shape"]))
    d = int(z["feat_shape"][1])
    nc = int(z["n_classes"][0])
    ne = int(z["n_edges"][0])
    assert adj.shape == (n, n), f"{ds}: adj shape {adj.shape} != {(n, n)}"
    assert adj.diagonal().sum() == 0, f"{ds}: self-loops present"
    assert (adj != adj.T).nnz == 0, f"{ds}: adjacency not symmetric"
    log(f"  verify {ds}: n={n} d={d} classes={nc} edges={ne}  OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="pubmed",
                    help="comma list; cora is auto-added if its files are missing")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(PYG_CACHE, exist_ok=True)
    want = [d.strip() for d in args.datasets.split(",") if d.strip()]

    # Ensure the whole-cora shadow data exists on both sides.
    cora_mem = os.path.join(C.MEMBERSHIP_ROOT, "cora.npz")
    cora_lsa = os.path.join(C.LSA_ROOT, "FGC Benchmarking", "attack", "data",
                            "dataset", "original", "ind.cora.graph")
    if "cora" not in want and (not os.path.exists(cora_mem) or not os.path.exists(cora_lsa)):
        log("cora data missing on one side -> adding cora to the build")
        want.append("cora")

    log("=" * 70)
    log(f"prepare_data  datasets={want}  force={args.force}")
    log(f"  PyG cache: {PYG_CACHE}")
    for ds in want:
        if ds not in PLANETOID:
            log(f"  [skip] {ds}: not a Planetoid dataset this script can build")
            continue
        log(f"\n[{ds}] downloading/assembling via PyG Planetoid ...")
        adj, feat, labels, n = assemble_planetoid(ds)
        log(f"  assembled: n={n} d={feat.shape[1]} classes={int(labels.max() + 1)} "
            f"edges={int(adj.nnz // 2)}")
        write_membership_npz(ds, adj, feat, labels, n, force=args.force)
        write_lsa_npz(ds, adj, feat, labels, n, force=args.force)
        verify(ds)
    log("\nprepare_data DONE")


if __name__ == "__main__":
    main()
