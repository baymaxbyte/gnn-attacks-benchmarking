"""End-to-end PubMed attack runner (LSA link-stealing + node membership inference).

ONE entry point. It drives the already-verified sibling benchmarks so PubMed is
produced with identical code/seeds/ratios/epochs as the other datasets, with the
WHOLE (uncoarsened) Cora graph as the shadow everywhere a shadow is used.

Pipeline:
  0. prepare_data.py                 download PubMed (+Cora if missing) -> npz
  1. LSA, per method:                step01 -> [step02a] -> step02 -> step03
     (scoped to pubmed), then build attack datasets (pubmed x all settings +
     cora whole) and run the 8 attacks + link-pred with --shadow cora_<tag>_whole
  2. Membership:                     cora control box (shadow) + pubmed
     step01..step04, whole-cora shadow MLP, step05 aggregate
  3. aggregate.py                    collect PubMed results into ./results/

Everything is resumable: each finished step is recorded in progress.json and the
sibling steps also skip their own cached artifacts. Logs stream live to
logs/pubmed_attacks.log and per-step logs under logs/.

    python run_all.py                         # default (in-repo methods + membership)
    python run_all.py --only lsa --lsa-methods fgc,kron,ugc
    python run_all.py --lsa-methods all --membership-methods all   # needs externals
    python run_all.py --skip-data            # data already prepared
"""
import argparse
import glob
import os
import sys

import config as C
import _runner as R

# step01 script name per LSA method (the rest are step02_blackbox / step03_attacker).
LSA_STEP01 = {"fgc": "step01_coarsen.py", "kron": "step01_reduce.py",
              "ugc": "step01_coarsen.py", "goren": "step01_coarsen.py",
              "gcond": "step01_condense.py"}
LSA_DIRECTION = "E"          # pubmed -> cora, continuing the A/B/C/D convention


# ===================================================================== LSA
def lsa_recipe(method):
    """Ordered list of (step_key_suffix, script) to build pubmed boxes+g+preds."""
    steps = [("step01", LSA_STEP01[method])]
    if method == "goren":
        steps.append(("step02a_weights", "step02a_goren_weights.py"))
    steps += [("step02", "step02_blackbox.py"), ("step03", "step03_attacker.py")]
    return steps


def cora_preds_present(attack_dir, spec):
    bb = os.path.join(attack_dir, "data", "pred", f"cora_{spec['bb']}_whole_pred.pkl")
    g = os.path.join(attack_dir, "data", "pred", f"cora_{spec['g']}_pred.pkl")
    return os.path.exists(bb) and os.path.exists(g)


def build_attack_dataset(attack_dir, ds, st, bb, g, setting, prog, key, cap):
    adir = os.path.join(attack_dir, "data", "attack_dataset")
    name = f"{ds}_{st}_{setting}"
    done_files = (os.path.join(adir, f"{name}_attack_test.json"),
                  os.path.join(adir, f"{name}_attack_all.json"))
    if all(os.path.exists(p) for p in done_files):
        R.log(f"  [cached] attack_dataset {name}")
        return True
    cmd = [C.PYTHON, "partial_graph_generation.py",
           "--dataset", ds, "--datapath", "data/dataset/original/",
           "--prediction_path", "data/pred/", "--saving_path", "data/attack_dataset/",
           "--target_pred", f"{ds}_{bb}_{setting}_pred.pkl",
           "--reference_pred", f"{ds}_{g}_pred.pkl",
           "--name", name, "--seed", "1"]
    if cap:
        cmd += ["--max_pos", str(C.MAX_POS)]
    rc, _ = R.run_cmd(cmd, attack_dir, R.build_env(), key, log_name=f"lsa_build_{name}")
    return rc == 0


def run_lsa_method(method, prog, runs, stop_on_fail=True):
    spec = C.LSA_SPECS[method]
    bench = os.path.join(C.LSA_ROOT, spec["dir"])
    attack_dir = os.path.join(bench, "attack")
    st, bb, g = spec["set_tag"], spec["bb"], spec["g"]
    results_ws = os.path.join(bench, "results", C.LSA_RESULTS_SUBDIR)
    os.makedirs(results_ws, exist_ok=True)

    ok, reason = C.env_ready(method, "lsa")
    if not ok:
        R.log(f"[LSA {method}] SKIPPED (external not configured): {reason}")
        prog.mark(f"lsa:{method}", "skipped", reason=reason)
        return True
    if not os.path.isdir(bench):
        R.log(f"[LSA {method}] SKIPPED: benchmark folder missing ({bench})")
        return True

    R.banner(f"LSA  {method}  ({spec['dir']})  settings={spec['settings']}")

    # --- 1. build pubmed boxes + reference g + published posteriors (scoped pubmed)
    env_pub = R.build_env(scope_datasets=["pubmed"])
    for suffix, script in lsa_recipe(method):
        key = f"lsa:{method}:{suffix}"
        if prog.done(key):
            R.log(f"  [skip] {key} (progress)")
            continue
        if not os.path.exists(os.path.join(bench, script)):
            R.log(f"  [warn] {script} not found in {spec['dir']}; skipping")
            prog.mark(key, "missing_script")
            continue
        rc, dt = R.run_cmd([C.PYTHON, script], bench, env_pub, key)
        if rc != 0:
            prog.mark(key, "failed", rc=rc)
            if stop_on_fail:
                raise SystemExit(f"[LSA {method}] {script} failed (rc={rc}); "
                                 f"see logs/, fix, and re-run to resume")
            return False
        prog.mark(key, "done", minutes=round(dt / 60, 2))

    # --- 2. ensure the whole-cora shadow box exists (pred pkls). Normally shipped.
    if not cora_preds_present(attack_dir, spec):
        R.log(f"  [LSA {method}] cora pred pkls missing -> building cora boxes")
        env_cora = R.build_env(scope_datasets=["cora"])
        for suffix, script in lsa_recipe(method):
            key = f"lsa:{method}:cora:{suffix}"
            if prog.done(key):
                R.log(f"  [skip] {key} (progress)")
                continue
            if not os.path.exists(os.path.join(bench, script)):
                continue
            rc, dt = R.run_cmd([C.PYTHON, script], bench, env_cora, key)
            if rc != 0:
                prog.mark(key, "failed", rc=rc)
                if stop_on_fail:
                    raise SystemExit(f"[LSA {method}] cora {script} failed (rc={rc})")
                return False
            prog.mark(key, "done", minutes=round(dt / 60, 2))

    # --- 3. build attack datasets: pubmed x all settings (capped) + cora whole
    build_attack_dataset(attack_dir, "cora", st, bb, g, "whole",
                          prog, f"lsa:{method}:ds:cora_whole", cap=False)
    for setting in spec["settings"]:
        build_attack_dataset(attack_dir, "pubmed", st, bb, g, setting,
                              prog, f"lsa:{method}:ds:pubmed_{setting}", cap=True)

    # --- 4. run the attacks: pubmed_<setting> vs shadow cora_<tag>_whole
    for setting in spec["settings"]:
        key = f"lsa:{method}:attack:{setting}"
        tag = f"{st}_{LSA_DIRECTION}_{setting}_wholeshadow"
        out = os.path.join(results_ws,
                           f"pubmed_{st}_{setting}__cora_{st}_whole__{tag}.json")
        if os.path.exists(out) or prog.done(key):
            R.log(f"  [skip] attack {method}/{setting} (exists)")
            prog.mark(key, "done", note="cached")
            continue
        cmd = [C.PYTHON, "run_attack.py", "--attack", "all",
               "--dataset", f"pubmed_{st}_{setting}",
               "--shadow", f"cora_{st}_whole",
               "--attack_dataset_path", "data/attack_dataset/",
               "--result_path", results_ws,
               "--runs", str(runs), "--tag", tag,
               "--link_prediction", "--quiet"]
        rc, dt = R.run_cmd(cmd, attack_dir, R.build_env(), key,
                           log_name=f"lsa_attack_{method}_{setting}")
        if rc != 0:
            prog.mark(key, "failed", rc=rc)
            if stop_on_fail:
                raise SystemExit(f"[LSA {method}] attack {setting} failed (rc={rc})")
            return False
        prog.mark(key, "done", minutes=round(dt / 60, 2))

    R.log(f"[LSA {method}] complete")
    return True


# ============================================================== MEMBERSHIP
def _mia(script, args, bench, env, prog, key, stop_on_fail):
    if prog.done(key):
        R.log(f"  [skip] {key} (progress)")
        return True
    if not os.path.exists(os.path.join(bench, script)):
        R.log(f"  [warn] {script} missing in Membership Attack; skipping {key}")
        prog.mark(key, "missing_script")
        return True
    rc, dt = R.run_cmd([C.PYTHON, script] + args, bench, env, key)
    if rc != 0:
        prog.mark(key, "failed", rc=rc)
        if stop_on_fail:
            raise SystemExit(f"[membership] {script} failed (rc={rc}); "
                             f"fix and re-run to resume")
        return False
    prog.mark(key, "done", minutes=round(dt / 60, 2))
    return True


def run_membership(methods, prog, stop_on_fail=True):
    bench = C.MEMBERSHIP_ROOT
    if not os.path.isdir(bench):
        R.log(f"[membership] SKIPPED: folder missing ({bench})")
        return True
    # Scope DATASETS to the two we touch: step01_split.py has no --only filter and
    # would otherwise try to load citeseer/chameleon/squirrel (absent in the
    # self-contained vendor). step02/03/04 additionally honour --only.
    env = R.build_env(scope_datasets=["cora", "pubmed"])
    ms = ",".join(methods)
    R.banner(f"MEMBERSHIP  methods={methods}  (shadow = whole cora)")

    # --- shadow prerequisite: cora control-whole posteriors (seeds 0,1,2)
    cora_post = os.path.join(bench, "artifacts", "posteriors",
                             "cora_control_whole_seed0.npy")
    if not os.path.exists(cora_post):
        R.log("  [membership] cora_control_whole posteriors missing -> building")
        _mia("step01_split.py", ["--only", "cora"], bench, env,
             prog, "mia:cora:split", stop_on_fail)
        _mia("step02_reduce.py", ["--only", "cora", "--methods", "control"],
             bench, env, prog, "mia:cora:reduce", stop_on_fail)
        _mia("step03_blackbox.py", ["--only", "cora", "--methods", "control"],
             bench, env, prog, "mia:cora:box", stop_on_fail)
    else:
        R.log("  [membership] cora_control_whole posteriors present (shadow ready)")

    # --- pubmed target pipeline
    _mia("step01_split.py", ["--only", "pubmed"], bench, env,
         prog, "mia:pubmed:split", stop_on_fail)
    _mia("step02_reduce.py", ["--only", "pubmed", "--methods", ms], bench, env,
         prog, "mia:pubmed:reduce", stop_on_fail)
    _mia("step03_blackbox.py", ["--only", "pubmed", "--methods", ms], bench, env,
         prog, "mia:pubmed:box", stop_on_fail)
    _mia("step04_attack.py", ["--only", "pubmed", "--methods", ms], bench, env,
         prog, "mia:pubmed:attack", stop_on_fail)
    # whole-cora shadow MLP (computes the <ds>_control_whole shadow AUC for pubmed)
    _mia("whole_shadow_mlp.py", [], bench, env,
         prog, "mia:pubmed:wholeshadow", stop_on_fail)
    # step05 writes one CSV per C.DATASETS; scope it to pubmed so it doesn't choke
    # on cora (shadow-only here, so it has no attack cells of its own).
    env_agg = R.build_env(scope_datasets=["pubmed"])
    _mia("step05_aggregate.py", [], bench, env_agg,
         prog, "mia:aggregate", stop_on_fail)
    R.log("[membership] complete")
    return True


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["both", "lsa", "membership"], default="both")
    ap.add_argument("--lsa-methods", default="default",
                    help="comma list / 'all' / 'default' (fgc,kron,ugc)")
    ap.add_argument("--membership-methods", default="default",
                    help="comma list / 'all' / 'default' (control,kron,ugc,fgc)")
    ap.add_argument("--runs", type=int, default=5, help="LSA attack-model repeats")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--no-stop-on-fail", action="store_true",
                    help="keep going past a failing step instead of stopping")
    args = ap.parse_args()

    os.makedirs(C.LOGS, exist_ok=True)
    os.makedirs(C.RESULTS, exist_ok=True)
    prog = R.Progress()
    stop = not args.no_stop_on_fail

    R.banner("PubMed attacks: end-to-end run")
    R.log(f"python={C.PYTHON}")
    R.log(f"only={args.only}  runs={args.runs}  skip_data={args.skip_data}")
    R.log(f"UGC alpha(pubmed)={C.UGC_ALPHA_PUBMED}  (only invented hyperparameter)")

    # --- 0. data
    if not args.skip_data and not prog.done("data"):
        rc, _ = R.run_cmd([C.PYTHON, "prepare_data.py"], C.HERE, R.build_env(),
                          "data", log_name="prepare_data")
        if rc != 0 and stop:
            raise SystemExit("prepare_data failed; see logs/prepare_data.log")
        prog.mark("data", "done" if rc == 0 else "failed", rc=rc)
    else:
        R.log("[data] skipped")

    # --- 1. LSA
    if args.only in ("both", "lsa"):
        methods = C.resolve_methods(args.lsa_methods, "lsa")
        R.log(f"LSA methods to run: {methods}")
        for m in methods:
            run_lsa_method(m, prog, args.runs, stop_on_fail=stop)

    # --- 2. membership
    if args.only in ("both", "membership"):
        methods = C.resolve_methods(args.membership_methods, "membership")
        if "control" not in methods:
            methods = ["control"] + methods        # control box is the reference
        R.log(f"membership methods to run: {methods}")
        run_membership(methods, prog, stop_on_fail=stop)

    # --- 3. collect
    if not prog.done("aggregate_collect"):
        rc, _ = R.run_cmd([C.PYTHON, "aggregate.py"], C.HERE, R.build_env(),
                          "aggregate_collect", log_name="aggregate")
        prog.mark("aggregate_collect", "done" if rc == 0 else "failed", rc=rc)

    R.banner("RUN COMPLETE (re-run the same command to resume anything unfinished)")


if __name__ == "__main__":
    main()
