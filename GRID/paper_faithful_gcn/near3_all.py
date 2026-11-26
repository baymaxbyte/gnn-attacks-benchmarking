#!/usr/bin/env python3
"""Run ALL 8 He et al. attacks (0-7) + link-pred under the near3 evaluation
(negatives = unlinked pairs at graph distance exactly 3), soft GCN, base vs GRID.
Reuses step3_negmodes' sampling so Attack-0/1/6 reproduce the earlier near3 run.
"""
import os, sys, json
import step3_negmodes as S

MODE = "near3"
ALL = [str(i) for i in range(8)]


def run_all(target, shadow, tag):
    out = os.path.join(S.RES, f"{target}__{shadow}__{tag}.json")
    S.sh([S.PY, "run_attack.py", "--attack", "all", "--dataset", target, "--shadow", shadow,
          "--attack_dataset_path", "data/attack_dataset/", "--result_path", S.RES,
          "--runs", "5", "--tag", tag, "--link_prediction", "--quiet"])
    return out


def parse_all(p):
    d = json.load(open(p))
    out = {a: S.auc(d["results"][a]) for a in ALL if a in d.get("results", {})}
    lp = d.get("link_prediction")
    if isinstance(lp, dict):
        out["link_pred"] = S.auc(lp)
    return out


def main():
    data = {ds: S.load_ds(ds) for ds in S.DIRS}
    for ds in S.DIRS:
        adj, nbrs, feat, g, fb, fg = data[ds]
        N = adj.shape[0]
        links, negs = S.sample_pairs(adj, nbrs, N, MODE, seed=hash((ds, MODE)) % (2**31))
        print(f"[{ds}/near3] {len(links)} linked + {len(negs)} unlinked", flush=True)
        S.write_ds(f"{ds}_{MODE}", links, negs, fb, g, feat)
        S.write_ds(f"{ds}_{MODE}_grid", links, negs, fg, g, feat)

    res = {}
    for ds, shadow in S.DIRS.items():
        b = run_all(f"{ds}_{MODE}", f"{shadow}_{MODE}", f"{ds}_{MODE}_allbase")
        gr = run_all(f"{ds}_{MODE}_grid", f"{shadow}_{MODE}", f"{ds}_{MODE}_allgrid")
        res[ds] = (parse_all(b), parse_all(gr))

    names = {**{str(i): f"Attack-{i}" for i in range(8)}, "link_pred": "link_pred"}
    csv = os.path.join(S.RES, "near3_all_attacks.csv")
    with open(csv, "w") as f:
        f.write("dataset,attack,auc_no_defense,auc_grid,delta\n")
        for ds in S.DIRS:
            base, grid = res[ds]
            print(f"\n=== {ds} / near3  (no-defense -> GRID) ===", flush=True)
            for a in ALL + ["link_pred"]:
                if a in base and a in grid:
                    print(f"  {names[a]:10s} {base[a]:.3f} -> {grid[a]:.3f}  ({grid[a]-base[a]:+.3f})", flush=True)
                    f.write(f"{ds},{names[a]},{base[a]:.4f},{grid[a]:.4f},{grid[a]-base[a]:+.4f}\n")
    S.rm_mode(MODE)
    print(f"\nwrote {csv}", flush=True)


if __name__ == "__main__":
    main()
