"""Add a WHOLE-GRAPH shadow variant to every lsa-benchmarking method, non-destructively.

For each benchmark it (1) builds the attack_datasets from the shipped pred pkls (not
shipped themselves), (2) runs the 8 He attacks + link-pred with the shadow pointed at the
other dataset's UNCOARSENED (_whole) box, tagged '_wholeshadow', writing results into
results/attacks_wholeshadow/ (the coarsened results/attacks/ are left untouched), then
(3) deletes the giant attack_datasets to cap disk. Resumable: existing datasets and
existing result JSONs are skipped.

    python run_wholeshadow.py                      # all methods, runs=5
    python run_wholeshadow.py --methods fgc --runs 1
"""
import argparse
import datetime
import glob
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("LSA_PY",
                    "/Users/chitraansh.pandey/Downloads/GNN DP/GCond/GCond-main/.venv/bin/python3")

SPECS = {
    "fgc":   dict(dir="FGC Benchmarking",   set_tag="fgc",   bb="fgcbb",   g="gfgc",
                  settings=["whole", "0.3", "0.1", "0.05"]),
    "gcond": dict(dir="GCOND Benchmarking", set_tag="gc",    bb="gcondbb", g="g",
                  settings=["whole", "0.25", "0.5", "1.0"]),
    "goren": dict(dir="GOREN Benchmarking", set_tag="goren", bb="gorenbb", g="ggoren",
                  settings=["whole", "0.3_bl", "0.3_goren", "0.5_bl", "0.5_goren",
                            "0.7_bl", "0.7_goren"]),
    "kron":  dict(dir="KRON Benchmarking",  set_tag="kron",  bb="kronbb",  g="gkron",
                  settings=["whole", "0.3", "0.5", "0.7"]),
    "ugc":   dict(dir="UGC Benchmarking",   set_tag="ugc",   bb="ugcbb",   g="gugc",
                  settings=["whole", "30", "50", "70"]),
}
DIRECTIONS = {"A": ("cora", "citeseer"), "B": ("citeseer", "cora"),
              "C": ("chameleon", "squirrel"), "D": ("squirrel", "chameleon")}
DATASETS = ("cora", "citeseer", "chameleon", "squirrel")
# Dense graphs cap positive pairs so the attack datasets stay tractable; matches
# each benchmark's step03 MAX_POS. cora/citeseer (< cap) build uncapped.
MAX_POS = {"chameleon": 10000, "squirrel": 10000}

_logf = open(os.path.join(HERE, "wholeshadow.log"), "a")


def log(m):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


def _run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def build_datasets(spec, attack_dir):
    """Build attack_datasets for both datasets x all settings (targets + whole shadow)."""
    st, bb, g = spec["set_tag"], spec["bb"], spec["g"]
    for ds in DATASETS:
        for setting in spec["settings"]:
            name = f"{ds}_{st}_{setting}"
            adir = os.path.join(attack_dir, "data", "attack_dataset")
            if (os.path.exists(os.path.join(adir, f"{name}_attack_test.json"))
                    and os.path.exists(os.path.join(adir, f"{name}_attack_all.json"))):
                continue
            cmd = [PY, "partial_graph_generation.py",
                   "--dataset", ds, "--datapath", "data/dataset/original/",
                   "--prediction_path", "data/pred/", "--saving_path", "data/attack_dataset/",
                   "--target_pred", f"{ds}_{bb}_{setting}_pred.pkl",
                   "--reference_pred", f"{ds}_{g}_pred.pkl",
                   "--name", name, "--seed", "1"]
            if ds in MAX_POS:
                cmd += ["--max_pos", str(MAX_POS[ds])]
            rc, out, err = _run(cmd, attack_dir)
            if rc:
                log(f"  BUILD FAIL {name}: {err[-600:]}")
                raise SystemExit(1)
            log(f"  built {name}")


def run_attacks(spec, attack_dir, results_ws, runs):
    st = spec["set_tag"]
    for d, (target, shadow) in DIRECTIONS.items():
        for setting in spec["settings"]:
            tag = f"{st}_{d}_{setting}_wholeshadow"
            out = os.path.join(results_ws,
                               f"{target}_{st}_{setting}__{shadow}_{st}_whole__{tag}.json")
            if os.path.exists(out):
                log(f"  skip {target} {setting} (exists)")
                continue
            cmd = [PY, "run_attack.py", "--attack", "all",
                   "--dataset", f"{target}_{st}_{setting}",
                   "--shadow", f"{shadow}_{st}_whole",
                   "--attack_dataset_path", "data/attack_dataset/",
                   "--result_path", results_ws,
                   "--runs", str(runs), "--tag", tag,
                   "--link_prediction", "--quiet"]
            t0 = time.time()
            rc, o, e = _run(cmd, attack_dir)
            if rc:
                log(f"  ATTACK FAIL {tag}: {e[-800:]}")
                raise SystemExit(1)
            log(f"  attacked {target}/{setting}  shadow={shadow}_whole  {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(SPECS))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--keep_datasets", action="store_true")
    args = ap.parse_args()

    log("=" * 70)
    log(f"run_wholeshadow  methods={args.methods}  runs={args.runs}")
    for m in [x.strip() for x in args.methods.split(",")]:
        spec = SPECS[m]
        bench = os.path.join(HERE, spec["dir"])
        attack_dir = os.path.join(bench, "attack")
        results_ws = os.path.join(bench, "results", "attacks_wholeshadow")
        os.makedirs(results_ws, exist_ok=True)
        log(f"----- {m} ({spec['dir']}) -----")
        t0 = time.time()
        build_datasets(spec, attack_dir)
        run_attacks(spec, attack_dir, results_ws, args.runs)
        if not args.keep_datasets:
            n = 0
            for f in glob.glob(os.path.join(attack_dir, "data", "attack_dataset", "*.json")):
                os.remove(f)
                n += 1
            log(f"  cleaned {n} attack_dataset files")
        log(f"  {m} done in {(time.time()-t0)/60:.1f} min")
    log("ALL DONE")


if __name__ == "__main__":
    main()
