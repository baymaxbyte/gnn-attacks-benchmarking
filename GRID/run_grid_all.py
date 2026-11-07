#!/usr/bin/env python3
"""Full GRID sweep: every coarsening/condensation method x every ratio x
{cora, citeseer}, attack AUC with vs without GRID on the TARGET black box.

The shadow is always the WHOLE (uncoarsened / uncondensed) box of the other
dataset -- only the target's served posteriors are defended by GRID.

Posteriors are read from each method's published baseline pred pkl
({ds}_{bb}_{setting}_pred.pkl) -- exactly what the attacker queries -- so this
works uniformly for all methods (incl. GCOND, which has no .npy posteriors, and
FGC, whose pkl already holds the served `_orig` posteriors). Results -> one CSV.

Resumable: an attack whose result json already exists is skipped.
"""
import argparse, json, os, pickle, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
LSA = os.path.join(HERE, "..", "lsa")                 # Benchmarking/temp/lsa
RES = os.path.join(HERE, "results", "attacks_all")
PY = sys.executable
sys.path.insert(0, HERE)
import grid_defense as G  # noqa: E402

SPECS = {
    "fgc":   dict(dir="FGC Benchmarking",   st="fgc",   bb="fgcbb",   g="gfgc",
                  settings=["whole", "0.3", "0.1", "0.05"]),
    "ugc":   dict(dir="UGC Benchmarking",   st="ugc",   bb="ugcbb",   g="gugc",
                  settings=["whole", "30", "50", "70"]),
    "gcond": dict(dir="GCOND Benchmarking", st="gc",    bb="gcondbb", g="g",
                  settings=["whole", "0.25", "0.5", "1.0"]),
    "goren": dict(dir="GOREN Benchmarking", st="goren", bb="gorenbb", g="ggoren",
                  settings=["whole", "0.3_bl", "0.3_goren", "0.5_bl", "0.5_goren",
                            "0.7_bl", "0.7_goren"]),
    "kron":  dict(dir="KRON Benchmarking",  st="kron",  bb="kronbb",  g="gkron",
                  settings=["whole", "0.3", "0.5", "0.7"]),
}
DIRECTIONS = {"cora": "citeseer", "citeseer": "cora"}   # target -> whole shadow
ATTACKS = [str(i) for i in range(8)]


def sh(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        sys.stderr.write(p.stdout[-1200:] + "\n" + p.stderr[-1200:] + "\n")
    return p.returncode == 0


def ds_files(attack_dir, name):
    base = os.path.join(attack_dir, "data", "attack_dataset", name)
    return [f"{base}_attack_{k}.json" for k in ("all", "train", "test")]


def valid_ds(attack_dir, name):
    return all(os.path.exists(p) and os.path.getsize(p) > 0
               for p in ds_files(attack_dir, name))


def rm_ds(attack_dir, name):
    for p in ds_files(attack_dir, name):
        if os.path.exists(p):
            os.remove(p)


def build_ds(attack_dir, ds, target_pred, g_tag, name, force=False):
    # NB: require NON-EMPTY train+test+all (stale 0-byte files from old runs
    # must not be treated as cached).
    if valid_ds(attack_dir, name) and not force:
        return True
    rm_ds(attack_dir, name)
    return sh([PY, "partial_graph_generation.py", "--dataset", ds,
               "--datapath", "data/dataset/original/", "--prediction_path", "data/pred/",
               "--saving_path", "data/attack_dataset/", "--target_pred", target_pred,
               "--reference_pred", f"{ds}_{g_tag}_pred.pkl", "--name", name,
               "--seed", "1"], attack_dir)


def run_attack(attack_dir, target, shadow, tag, runs):
    out = os.path.join(RES, f"{target}__{shadow}__{tag}.json")
    if os.path.exists(out):
        return out
    ok = sh([PY, "run_attack.py", "--attack", "all", "--dataset", target,
             "--shadow", shadow, "--attack_dataset_path", "data/attack_dataset/",
             "--result_path", RES, "--runs", str(runs), "--tag", tag,
             "--link_prediction", "--quiet"], attack_dir)
    return out if ok and os.path.exists(out) else None


def auc_of(e):
    return e.get("auc_mean", e.get("auc"))


def parse(path):
    d = json.load(open(path))
    out = {a: auc_of(d["results"][a]) for a in ATTACKS if a in d.get("results", {})}
    lp = d.get("link_prediction")
    if isinstance(lp, dict):
        out["link_pred"] = auc_of(lp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=0.4)
    ap.add_argument("--nhop", type=int, default=3)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--methods", default=",".join(SPECS))
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    nbrs = {ds: G.build_adjacency(ds)[1] for ds in DIRECTIONS}   # adjacency once per dataset
    methods = [m.strip() for m in args.methods.split(",")]
    jobs = []
    for m in methods:
        sp = SPECS[m]
        attack_dir = os.path.join(LSA, sp["dir"], "attack")
        pred = os.path.join(attack_dir, "data", "pred")
        for ds, shadow in DIRECTIONS.items():
            # whole shadow dataset (clean, uncoarsened) built once per (method, shadow)
            build_ds(attack_dir, shadow, f"{shadow}_{sp['bb']}_whole_pred.pkl",
                     sp["g"], f"{shadow}_{sp['st']}_whole")
            for setting in sp["settings"]:
                base_name = f"{ds}_{sp['st']}_{setting}"
                grid_name = f"{base_name}_grid"
                base_json = os.path.join(RES, f"{base_name}__{shadow}_{sp['st']}_whole__"
                                              f"{m}_{ds}_{setting}_base.json")
                grid_json = os.path.join(RES, f"{grid_name}__{shadow}_{sp['st']}_whole__"
                                              f"{m}_{ds}_{setting}_grid.json")
                if os.path.exists(base_json) and os.path.exists(grid_json):
                    jobs.append((m, ds, setting, base_json, grid_json)); continue

                # 1) GRID-perturb the served posteriors (from the baseline pkl)
                vpath = os.path.join(pred, f"{ds}_{sp['bb']}_{setting}_pred.pkl")
                R, st = G.grid_posteriors(G.load_pred(vpath), nbrs[ds],
                                          dsname=f"{m}:{ds}/{setting}",
                                          theta=args.theta, n_hop=args.nhop, verbose=False)
                with open(os.path.join(pred, f"{ds}_{sp['bb']}_{setting}_grid_pred.pkl"),
                          "wb") as f:
                    pickle.dump(R, f)
                print(f"[{m}:{ds}/{setting}] core={st.get('core','?')} "
                      f"lbl_chg={st['label_changes']} adj {st['adj_before']:.3f}->"
                      f"{st['adj_after']:.3f} (floor {st['nhop_after']:.3f})", flush=True)

                # 2) attack datasets
                build_ds(attack_dir, ds, f"{ds}_{sp['bb']}_{setting}_pred.pkl",
                         sp["g"], base_name)
                build_ds(attack_dir, ds, f"{ds}_{sp['bb']}_{setting}_grid_pred.pkl",
                         sp["g"], grid_name, force=True)

                # 3) attacks: baseline vs GRID, both with the whole shadow
                run_attack(attack_dir, base_name, f"{shadow}_{sp['st']}_whole",
                           f"{m}_{ds}_{setting}_base", args.runs)
                run_attack(attack_dir, grid_name, f"{shadow}_{sp['st']}_whole",
                           f"{m}_{ds}_{setting}_grid", args.runs)
                # cap disk: the per-pair JSONs are large (cora ~0.3GB, citeseer
                # ~0.5GB/box). Drop the GRID set (single-use) and coarsened base
                # sets; keep the *_whole base sets (they double as shadows).
                rm_ds(attack_dir, grid_name)
                if not base_name.endswith("_whole"):
                    rm_ds(attack_dir, base_name)
                jobs.append((m, ds, setting, base_json, grid_json))

    # 4) aggregate
    names = {**{str(i): f"Attack-{i}" for i in range(8)}, "link_pred": "link_pred"}
    order = ATTACKS + ["link_pred"]
    csv = os.path.join(HERE, "results", f"grid_all_theta{args.theta}.csv")
    rows = 0
    with open(csv, "w") as f:
        f.write("dataset,method,setting,attack,auc_base_wholeshadow,auc_grid,delta\n")
        for m, ds, setting, bj, gj in jobs:
            if not (os.path.exists(bj) and os.path.exists(gj)):
                continue
            base, grid = parse(bj), parse(gj)
            for a in order:
                if a not in base or a not in grid:
                    continue
                b, g = base[a], grid[a]
                f.write(f"{ds},{m},{setting},{names[a]},{b:.4f},{g:.4f},{g-b:+.4f}\n")
                rows += 1
    print(f"\nwrote {os.path.relpath(csv, HERE)}  ({rows} rows)")


if __name__ == "__main__":
    main()
