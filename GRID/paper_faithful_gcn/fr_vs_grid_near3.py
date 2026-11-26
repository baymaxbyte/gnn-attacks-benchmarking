#!/usr/bin/env python3
"""Fill the missing cell: Fisher-Rao defense evaluated under near3 negatives,
alongside no-defense and GRID, on identical pairs (same soft GCN target).
Gives a clean defense x eval matrix. Reuses step3_negmodes helpers."""
import os, sys, json, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step3_negmodes as S

RES = os.path.join(S.HERE, "results", "fr_vs_grid_near3")
os.makedirs(RES, exist_ok=True)
SEED = 1234
ATT = ["0", "1", "6"]


def load3(ds):
    adj, feat, *_ = S.load_data(S.DS, ds)
    feat = feat.toarray() if not isinstance(feat, np.ndarray) else feat
    def pk(n): return np.asarray(pickle.load(open(os.path.join(S.PRED, n), "rb"))).tolist()
    return (adj.tocsr(), S.nbrs_of(adj), feat.tolist(), pk(f"{ds}_gkron_pred.pkl"),
            pk(f"{ds}_softgcn_whole_pred.pkl"), pk(f"{ds}_softgcn_whole_grid_pred.pkl"),
            pk(f"{ds}_softgcn_whole_frdef_pred.pkl"))


def run_attack(target, shadow, tag):
    out = os.path.join(RES, f"{target}__{shadow}__{tag}.json")
    S.sh([S.PY, "run_attack.py", "--attack", "0,1,6", "--dataset", target, "--shadow", shadow,
          "--attack_dataset_path", "data/attack_dataset/", "--result_path", RES,
          "--runs", "5", "--tag", tag, "--link_prediction", "--quiet"])
    return out


def main():
    for ds in S.DIRS:
        adj, nbrs, feat, g, fb, fg, ff = load3(ds)
        N = adj.shape[0]
        links, negs = S.sample_pairs(adj, nbrs, N, "near3", SEED)
        print(f"[{ds}/near3] {len(links)} linked + {len(negs)} unlinked (dist-3)", flush=True)
        S.write_ds(f"{ds}_n3b", links, negs, fb, g, feat)
        S.write_ds(f"{ds}_n3g", links, negs, fg, g, feat)
        S.write_ds(f"{ds}_n3f", links, negs, ff, g, feat)
    res = {}
    for ds, shadow in S.DIRS.items():
        b = run_attack(f"{ds}_n3b", f"{shadow}_n3b", f"{ds}_nodef")
        gg = run_attack(f"{ds}_n3g", f"{shadow}_n3b", f"{ds}_grid")
        fr = run_attack(f"{ds}_n3f", f"{shadow}_n3b", f"{ds}_fr")
        res[ds] = (S.parse(b), S.parse(gg), S.parse(fr))
    names = {"0": "Attack-0", "1": "Attack-1", "6": "Attack-6", "link_pred": "link_pred"}
    csv = os.path.join(RES, "fr_vs_grid_near3.csv")
    with open(csv, "w") as f:
        f.write("dataset,attack,no_defense,grid,fisher_rao,delta_grid,delta_fr\n")
        for ds in S.DIRS:
            nd, gr, fr = res[ds]
            print(f"\n=== {ds} / near3 eval (no-def -> GRID / Fisher-Rao) ===", flush=True)
            for a in ATT + ["link_pred"]:
                if a in nd:
                    n, gv, fv = nd[a], gr.get(a), fr.get(a)
                    print(f"  {names[a]:10s} nodef={n:.3f}  GRID={gv:.3f}({gv-n:+.3f})  FR={fv:.3f}({fv-n:+.3f})", flush=True)
                    f.write(f"{ds},{names[a]},{n:.4f},{gv:.4f},{fv:.4f},{gv-n:+.4f},{fv-n:+.4f}\n")
    for fn in os.listdir(S.ADS):
        if any(t in fn for t in ("_n3b_attack_", "_n3g_attack_", "_n3f_attack_")):
            os.remove(os.path.join(S.ADS, fn))
    print(f"\nwrote {csv}", flush=True)


if __name__ == "__main__":
    main()
