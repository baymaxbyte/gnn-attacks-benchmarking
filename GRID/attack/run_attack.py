"""
Stage 3 -- run any of the 8 link stealing attacks under the paper's protocol.

Paper clauses that this script implements:

  p7  "we split the attack dataset randomly by half into attack training dataset
       and attack testing dataset. We use the attack training dataset to train
       our attack models when the target dataset's partial graph is part of the
       adversary's knowledge. We use attack testing dataset to evaluate all our
       attacks. For the attacks that have a shadow dataset, we also construct an
       attack dataset on the shadow dataset to train the attack model. Note that
       we do not split this attack dataset because we do not use it for
       evaluation."

  p8  "We run all experiments with this setting for 5 times and report the
       average value and the standard deviation of AUC scores. Note that for
       Attack-0 and Attack-2, the AUC scores keep the same since these two
       attacks are unsupervised."

Consequences, all of which differ from the released scripts:

  * Evaluation set = the target's attack TESTING half (50%), for every attack.
    The released code evaluates on a 10% suffix taken from its ratio-0.9 file.

  * A shadow dataset contributes its ENTIRE attack dataset to training, unsplit.

  * 5 repetitions with mean and standard deviation. The second sentence above
    pins down what varies: if the 50/50 split were re-drawn per repetition, the
    unsupervised AUCs would move too. So the split is drawn once in stage 2 and
    only the attack model's own randomness (initialisation, dropout, batch
    order) is repeated. Attack-0 and Attack-2 are therefore reported once, with
    no standard deviation.

Examples
--------
    # attacks that stay inside the target dataset
    python run_attack.py --attack 0 --dataset cora
    python run_attack.py --attack 3 --dataset cora

    # attacks with a shadow dataset: --dataset is the TARGET
    python run_attack.py --attack 7 --dataset cora --shadow citeseer

    # the full table
    python run_attack.py --attack all --dataset cora --shadow citeseer
"""
import argparse
import json
import os

import numpy as np

from lsa.attack_model import DEFAULT_BATCH_SIZE
from lsa.attacks import (ATTACK_FEATURE_GROUPS, ATTACK_FEATURE_GROUPS_TABLE3,
                         FEATURISERS, FEATURISERS_TABLE3, GROUP_ORDER,
                         KNOWLEDGE, NEEDS_SHADOW, TRAIN_SOURCES, UNSUPERVISED,
                         correlation_distance_buckets, knowledge_tag,
                         make_featuriser, prepare_supervised, read_jsonl,
                         run_supervised_once)
from lsa.baselines import link_prediction_baseline
from lsa.features import DISPLAY_NAMES
from lsa.spec import (ATTACK_EPOCHS, ATTACK_HIDDEN_LAYERS, ATTACK_LR, N_RUNS,
                      RESPLIT_PER_RUN)

METRICS = ["accuracy", "precision", "recall", "f1", "auc"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the link stealing attacks under the paper's protocol")
    p.add_argument("--attack", default="all",
                   help="0-7, a comma list like 0,3,7, or 'all'")
    p.add_argument("--dataset", default="cora", help="TARGET dataset")
    p.add_argument("--shadow", default=None,
                   help="shadow dataset; required for attacks 1, 4, 5, 7")
    p.add_argument("--attack_dataset_path", default="data/attack_dataset/")
    p.add_argument("--result_path", default="result/")
    p.add_argument("--operator", default="concate_all",
                   choices=["average", "hadamard", "weighted_l1",
                            "weighted_l2", "concate_all"],
                   help="Table 14 operator; the paper concatenates all four")
    p.add_argument("--metric_type", default="entropy",
                   choices=["entropy", "kl_divergence", "js_divergence"])
    p.add_argument("--epochs", type=int, default=ATTACK_EPOCHS)
    p.add_argument("--lr", type=float, default=ATTACK_LR)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                   help="not stated in the paper; Keras default is 128")
    p.add_argument("--hidden_layers", type=int, default=ATTACK_HIDDEN_LAYERS,
                   help="paper p8: 3 hidden layers of 32 neurons")
    p.add_argument("--runs", type=int, default=N_RUNS,
                   help="paper p8: 5. Unsupervised attacks always run once.")
    p.add_argument("--train_subsample", type=float, default=1.0,
                   help="fraction of the target training half to keep; only "
                        "for the Figure-7 training-set-size study")
    p.add_argument("--train_fraction", type=float, default=None,
                   help="Figure 7: re-split the target's attack dataset at this "
                        "fraction instead of using the stored 50/50 halves. "
                        "Reads <target>_attack_all.json and slices it, which "
                        "reproduces partial_graph_generation exactly because "
                        "that file is written in split order.")
    p.add_argument("--strict_scaling", action="store_true",
                   help="use StandardScaler.transform on the test features "
                        "instead of an independent fit (sensitivity check)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="paper", help="label written into results")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--feature_groups", default=None,
                   help="comma list of Table-3 group names to override the "
                        "attack's own set, e.g. 'd_f' or 'psi_f'. Used for the "
                        "Figure-5 ablation; applies to every supervised attack "
                        f"selected. Available: {','.join(GROUP_ORDER)}")
    p.add_argument("--link_prediction", action="store_true",
                   help="also run the traditional link-prediction baseline "
                        "(common neighbour, Jaccard, preferential attachment) "
                        "for the Figure-6 comparison")
    p.add_argument("--table3_strict", action="store_true",
                   help="use Table 3's literal feature list. Only differs for "
                        "Attack-6, where Table 3 also ticks Psi(Fu,Fv); that "
                        "adds 4D columns and moves the result away from the "
                        "paper's own Table 10 value. See lsa/attacks.py.")
    p.add_argument("--buckets", action="store_true",
                   help="also report Table 5: Attack-0 AUC per Correlation "
                        "distance level")
    return p.parse_args()


def path_for(base, dataset, split):
    return os.path.join(base, f"{dataset}_attack_{split}.json")


def require(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing {what}: {path}\n"
            f"Run:  python train_target.py --dataset <ds> --model gcn\n"
            f"      python train_target.py --dataset <ds> --model dense\n"
            f"      python partial_graph_generation.py --dataset <ds>")
    return path


def target_halves(args):
    """The target's (attack training half, attack testing half).

    Normally these are the stored 50/50 files. With --train_fraction set, the
    unsplit `_attack_all` file is sliced instead. That file is written pair by
    pair in split order (positive then negative for each index), so slicing at
    2 * floor(fraction * n_pairs) reproduces what partial_graph_generation would
    have written for that fraction -- which is what Figure 7 sweeps.
    """
    if args.train_fraction is None:
        train = read_jsonl(require(
            path_for(args.attack_dataset_path, args.dataset, "train"),
            f"target attack training half for '{args.dataset}'"))
        test = read_jsonl(require(
            path_for(args.attack_dataset_path, args.dataset, "test"),
            f"target attack testing half for '{args.dataset}'"))
        return train, test

    allrows = read_jsonl(require(
        path_for(args.attack_dataset_path, args.dataset, "all"),
        f"target attack dataset for '{args.dataset}'"))
    n_pairs = len(allrows) // 2
    cut = 2 * int(n_pairs * args.train_fraction)
    return allrows[:cut], allrows[cut:]


def load_training_rows(a, args, target_train):
    """Assemble the attack training set allowed by attack a's knowledge."""
    rows = []
    for src in TRAIN_SOURCES[a]:
        if src == "shadow_all":
            # p7: the shadow attack dataset is NOT split.
            p = require(path_for(args.attack_dataset_path, args.shadow, "all"),
                        f"shadow attack dataset for '{args.shadow}'")
            rows += read_jsonl(p)
        elif src == "target_train":
            # p7: the target's attack TRAINING half is the partial graph A*.
            tr = target_train
            if args.train_subsample < 1.0:
                tr = tr[:int(len(tr) * args.train_subsample)]
            rows += tr
    return rows


def aggregate(run_results, runs):
    """Mean and standard deviation over the repetitions (paper p8).

    The individual run values are kept as `<metric>_runs` as well. The paper
    only publishes mean and std, but without the raw five numbers you cannot
    tell a tight spread from an outlier-driven one, or run any test across
    configurations later.
    """
    out = {}
    for name in METRICS:
        vals = [float(r[name]) for r in run_results]
        out[name] = float(np.mean(vals))
        out[f"{name}_std"] = float(np.std(vals))
        out[f"{name}_runs"] = vals
    out["runs"] = runs
    out["n_features"] = run_results[0]["n_features"]
    return out


def main():
    args = parse_args()
    attacks = list(range(8)) if args.attack == "all" else \
        [int(a) for a in args.attack.split(",")]

    os.makedirs(args.result_path, exist_ok=True)

    target_train_rows, test_rows = target_halves(args)

    n_pos = sum(r["label"] for r in test_rows)
    frac = args.train_fraction if args.train_fraction is not None else 0.5
    print(f"target={args.dataset}  attack training = {len(target_train_rows)} "
          f"pairs, testing = {len(test_rows)} pairs "
          f"({n_pos} linked / {len(test_rows)-n_pos} unlinked), "
          f"train_fraction={frac}")
    if args.shadow:
        print(f"shadow={args.shadow}  (its attack dataset is used unsplit)")
    print(f"attack model = MLP with {args.hidden_layers} hidden layers x 32, "
          f"{args.epochs} epochs, lr={args.lr}, runs={args.runs}, "
          f"resplit_per_run={RESPLIT_PER_RUN}")

    all_results = {}
    for a in attacks:
        print(f"\n{'='*68}\nAttack-{a}   K = {knowledge_tag(a)}\n{'='*68}")

        if a in NEEDS_SHADOW and not args.shadow:
            print(f"  SKIPPED: attack {a} needs --shadow")
            continue

        if a in UNSUPERVISED:
            # p8: unsupervised, so the score is exact -- one run, no std.
            res = UNSUPERVISED[a](test_rows, seed=args.seed)
            res["runs"] = 1
            print(f"  selected metric : {res['selected_metric']}")
            print(f"  AUC             : {res['auc']:.4f}")
            print(f"  K-means (K=2)   : precision={res['precision']:.4f}  "
                  f"recall={res['recall']:.4f}  f1={res['f1']:.4f}")
            if a == 0 and args.buckets:
                res["distance_buckets"] = correlation_distance_buckets(test_rows)
                print("  Table 5, AUC per Correlation distance level:")
                for b in res["distance_buckets"]:
                    auc = "n/a" if b["auc"] is None else f"{b['auc']:.4f}"
                    print(f"    {b['range']}  n={b['n_pairs']:<7}"
                          f"pos={b['n_positive']:<7}AUC={auc}")
            if not args.quiet:
                print("  all metrics:")
                for name, auc in sorted(res["per_metric"].items(),
                                        key=lambda kv: -kv[1]):
                    pretty = DISPLAY_NAMES.get(name.split("_")[-1], name)
                    print(f"    {name:<28} {auc:.4f}   ({pretty})")
        else:
            train_rows = load_training_rows(a, args, target_train_rows)
            sources = " + ".join(TRAIN_SOURCES[a])
            print(f"  training data   : {len(train_rows)} pairs  [{sources}]")

            # Featurise once; the repetitions vary only the attack model.
            if args.feature_groups:
                groups = tuple(g.strip() for g in args.feature_groups.split(","))
                featuriser = make_featuriser(groups)
                print(f"  feature groups  : {','.join(featuriser.groups)} "
                      f"(override; Table 3 says "
                      f"{','.join(ATTACK_FEATURE_GROUPS[a])})")
            elif args.table3_strict:
                featuriser = FEATURISERS_TABLE3[a]
                print(f"  feature groups  : {','.join(featuriser.groups)}"
                      f"   [Table 3, literal]")
            else:
                featuriser = FEATURISERS[a]
                note = ("Table 3" if ATTACK_FEATURE_GROUPS[a]
                        == ATTACK_FEATURE_GROUPS_TABLE3[a]
                        else "Table 3 minus psi_F, to match Table 10")
                print(f"  feature groups  : {','.join(featuriser.groups)}"
                      f"   [{note}]")
            prepared = prepare_supervised(
                train_rows, test_rows, featuriser,
                operator=args.operator, metric_type=args.metric_type,
                independent_test_fit=not args.strict_scaling, verbose=True)

            run_results = []
            for run in range(args.runs):
                print(f"    run {run+1}/{args.runs}")
                run_results.append(run_supervised_once(
                    prepared, epochs=args.epochs, batch_size=args.batch_size,
                    device=args.device, seed=args.seed + run,
                    hidden_layers=args.hidden_layers,
                    verbose=not args.quiet))
            res = aggregate(run_results, args.runs)
            print(f"  features        : {res['n_features']}")
            print(f"  AUC             : {res['auc']:.4f} +/- {res['auc_std']:.4f}")
            print(f"  accuracy        : {res['accuracy']:.4f} +/- "
                  f"{res['accuracy_std']:.4f}")
            print(f"  precision/recall/f1 : {res['precision']:.4f} / "
                  f"{res['recall']:.4f} / {res['f1']:.4f}")

        all_results[a] = res

    # ------------------------------------------- link prediction comparison
    baseline = None
    if args.link_prediction:
        print(f"\n{'='*68}\nLink prediction baseline   "
              f"[common neighbour, Jaccard, preferential attachment]\n{'='*68}")
        baseline = link_prediction_baseline(
            target_train_rows, test_rows, runs=args.runs,
            epochs=args.epochs, batch_size=args.batch_size,
            hidden_layers=args.hidden_layers, device=args.device,
            seed=args.seed, independent_test_fit=not args.strict_scaling)
        print(f"  features        : {baseline['n_features']} "
              f"({', '.join(baseline['feature_names'])})")
        print(f"  AUC             : {baseline['auc']:.4f} +/- "
              f"{baseline['auc_std']:.4f}")
        print(f"  precision/recall/f1 : {baseline['precision']:.4f} / "
              f"{baseline['recall']:.4f} / {baseline['f1']:.4f}")

    # ------------------------------------------------------------- reporting
    if len(all_results) > 1:
        print(f"\n{'='*68}\nSUMMARY   target={args.dataset}"
              f"   shadow={args.shadow}\n{'='*68}")
        print(f"  {'attack':<9}{'K=(F,A*,D)':<14}{'AUC':<10}{'std':<9}{'runs':<6}")
        for a in sorted(all_results):
            r = all_results[a]
            std = r.get("auc_std")
            print(f"  Attack-{a:<2} {knowledge_tag(a):<14}"
                  f"{r['auc']:<10.4f}"
                  f"{('-' if std is None else f'{std:.4f}'):<9}"
                  f"{r['runs']:<6}")
        if baseline:
            print(f"  {'link-pred':<9} {'(baseline)':<14}"
                  f"{baseline['auc']:<10.4f}{baseline['auc_std']:<9.4f}"
                  f"{baseline['runs']:<6}")
            beaten = sum(1 for r in all_results.values()
                         if r["auc"] > baseline["auc"])
            print(f"\n  {beaten}/{len(all_results)} attacks beat the "
                  f"link-prediction baseline")

    out = os.path.join(args.result_path,
                       f"{args.dataset}__{args.shadow or 'none'}__{args.tag}.json")
    payload = {
        "target": args.dataset,
        "shadow": args.shadow,
        "config": {
            "hidden_layers": args.hidden_layers, "epochs": args.epochs,
            "lr": args.lr, "batch_size": args.batch_size, "runs": args.runs,
            "operator": args.operator, "metric_type": args.metric_type,
            "train_subsample": args.train_subsample,
            "train_fraction": frac,
            "strict_scaling": args.strict_scaling, "seed": args.seed,
            "n_test_pairs": len(test_rows),
            "feature_groups_override": args.feature_groups,
            "table3_strict": args.table3_strict,
            "attack_dataset_path": args.attack_dataset_path,
        },
        "results": {str(k): {kk: vv for kk, vv in v.items() if kk != "per_metric"}
                    for k, v in all_results.items()},
        "per_metric": {str(k): v["per_metric"] for k, v in all_results.items()
                       if "per_metric" in v},
        "feature_groups": {str(a): list(ATTACK_FEATURE_GROUPS[a])
                           for a in all_results if a in ATTACK_FEATURE_GROUPS},
        "link_prediction": baseline,
    }
    with open(out, "w") as wf:
        json.dump(payload, wf, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
