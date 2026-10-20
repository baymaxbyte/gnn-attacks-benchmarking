"""Step 01 -- build the member / non-member split for each dataset.

Uniform-random 20% of nodes become non-members (V_out); their edges are removed.
The 80% members (V_in) and their induced subgraph are saved for the reduction step.
The split is shared by every reduction method and, via the fixed seed, reproducible.

    python step01_split.py
"""
import os

import common as K

os.environ.update(K.single_thread_env())

import numpy as np
import scipy.sparse as sp

import config as C
import data as D
import splits as S


def build(dataset):
    m = D.load_whole(dataset)
    adj, feat, labels, n = m["adj"], m["features"], m["labels"], m["n"]
    orig_deg = np.asarray(adj.sum(1)).ravel().astype(np.int64)

    member, nonmember, member_mask = S.make_split(adj, C.SPLIT_SEED,
                                                   C.HOLDOUT_FRACTION)
    vin_adj, vin_feat, vin_labels = S.induced_subgraph(adj, feat, labels, member)
    vin_deg = np.asarray(vin_adj.sum(1)).ravel()

    path = K.split_path(dataset)
    np.savez_compressed(
        path,
        member_ids=member.astype(np.int64),
        nonmember_ids=nonmember.astype(np.int64),
        member_mask=member_mask,
        vin_adj_data=vin_adj.data, vin_adj_indices=vin_adj.indices,
        vin_adj_indptr=vin_adj.indptr, vin_adj_shape=np.array(vin_adj.shape),
        vin_feat_data=vin_feat.data, vin_feat_indices=vin_feat.indices,
        vin_feat_indptr=vin_feat.indptr, vin_feat_shape=np.array(vin_feat.shape),
        vin_labels=vin_labels.astype(np.int64),
        seed=np.array([C.SPLIT_SEED]), frac=np.array([C.HOLDOUT_FRACTION]))

    return {
        "dataset": dataset, "n": n, "members": len(member),
        "nonmembers": len(nonmember), "held_frac": len(nonmember) / n,
        "orig_edges": int(adj.nnz // 2), "vin_edges": int(vin_adj.nnz // 2),
        "vin_isolated": int((vin_deg == 0).sum()),
        "member_mean_deg": float(orig_deg[member].mean()),
        "nonmember_mean_deg": float(orig_deg[nonmember].mean()),
        "path": K.rel(path),
    }


def main():
    K.ensure_dirs()
    print(f"membership split: uniform-random {C.HOLDOUT_FRACTION:.0%} non-members, "
          f"seed {C.SPLIT_SEED}\n")
    rows, outputs = [], []
    for ds in C.DATASETS:
        r = build(ds)
        outputs.append(r["path"])
        rows.append(r)
        print(f"=== {ds} ===")
        print(f"  members / non-members : {r['members']} / {r['nonmembers']}  "
              f"({r['held_frac']:.3%})")
        print(f"  edges (orig -> V_in)  : {r['orig_edges']} -> {r['vin_edges']}")
        print(f"  isolated inside V_in  : {r['vin_isolated']}")
        print(f"  mean degree (orig)    : members {r['member_mean_deg']:.2f}  "
              f"non-members {r['nonmember_mean_deg']:.2f}  "
              f"(close = degree-representative)")
        print(f"  wrote {r['path']}\n")

    K.record(
        "step01_split",
        summary=(f"Built uniform-random {C.HOLDOUT_FRACTION:.0%} member/non-member "
                 f"split for {len(rows)} datasets (seed {C.SPLIT_SEED})"),
        inputs=[f"{ds}.npz" for ds in C.DATASETS],
        outputs=outputs,
        numbers={f"{r['dataset']}_members": r["members"] for r in rows},
        notes=("Non-members (V_out) and their edges are removed; V_in is the members' "
               "induced subgraph. Split is uniform-random and shared by every "
               "reduction method, so members/non-members are identical across "
               "GCond, FGC, UGC, GOREN, KRON and the control."))


if __name__ == "__main__":
    main()
