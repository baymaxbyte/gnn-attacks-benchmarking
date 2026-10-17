"""Step 05 -- compile the UGC attack results into CSVs, figures and a writeup.

Column names match the GCond and FGC benchmarks' summary CSVs, so all three
concatenate and can be compared directly: they attack identical node pairs with
identical 50/50 splits under the shared pair seed.

    python step05_aggregate.py
    python step05_aggregate.py --no_figures
"""
import argparse
import csv
import glob
import json
import os

import common as K

os.environ.update(K.single_thread_env())

import numpy as np

import blackbox_ugc as B
import config as C

SET_TAG = "ugc"
ORDER = ["whole"] + [str(r) for r in C.RATIOS]
DIRECTION = {"cora": "A", "citeseer": "B"}
FIGURES = os.path.join(C.RESULTS, "figures")

KNOW = {0: ("x", "x", "x"), 1: ("x", "x", "Y"), 2: ("Y", "x", "x"),
        3: ("x", "Y", "x"), 4: ("x", "Y", "Y"), 5: ("Y", "x", "Y"),
        6: ("Y", "Y", "x"), 7: ("Y", "Y", "Y")}
UNSUPERVISED = (0, 2)
METRICS = ("auc", "accuracy", "precision", "recall", "f1")
DISTANCES = ("correlation", "cosine", "euclidean", "sqeuclidean",
             "braycurtis", "cityblock", "chebyshev", "canberra")


def load():
    attacks = {}
    for f in glob.glob(os.path.join(C.RESULTS, "attacks", "*.json")):
        d = json.load(open(f))
        ds, setting = d["target"].split(f"_{SET_TAG}_")
        attacks[(ds, setting)] = d
    boxes = {}
    vp = os.path.join(C.RESULTS, "step02_validation.json")
    if os.path.exists(vp):
        for b in json.load(open(vp))["boxes"]:
            key = (b["dataset"], "whole" if b["ratio"] is None else str(b["ratio"]))
            boxes[key] = b
    return attacks, boxes


def _box_cols(ds, setting, boxes):
    b = boxes.get((ds, setting))
    if not b:
        return dict(alpha="", k="", bin_width="", box_acc_mean="", box_acc_std="",
                    box_acc_paper="", box_acc_delta="", box_acc_runs="",
                    augmented_dim="")
    sp = b["split"]
    return dict(
        alpha=b["alpha"],
        k=b["k"] or (sp["train"] + sp["val"] + sp["test"]),
        bin_width=(round(b["bin_width"], 6) if b["bin_width"] else ""),
        box_acc_mean=round(b["test_acc_mean"], 4),
        box_acc_std=round(b["test_acc_std"], 4),
        box_acc_paper=(b["paper_mean"] if b["paper_mean"] else ""),
        box_acc_delta=(round(b["delta_vs_paper"], 4)
                       if b["delta_vs_paper"] is not None else ""),
        box_acc_runs=";".join(f"{v:.4f}" for v in b["test_acc_runs"]),
        augmented_dim=b["feature_dim_augmented"])


def write_csvs(attacks, boxes):
    written = []
    for ds in C.DATASETS:
        shadow = "citeseer" if ds == "cora" else "cora"
        present = [s for s in ORDER if (ds, s) in attacks]
        det_rows, sum_rows = [], []
        for setting in present:
            d = attacks[(ds, setting)]
            cfg = d["config"]
            base = dict(target=ds, shadow=shadow, direction=DIRECTION[ds],
                        box=f"{ds}_{setting}", setting=setting,
                        **_box_cols(ds, setting, boxes),
                        n_test_pairs=cfg["n_test_pairs"])
            entries = [(str(a), a) for a in range(8)] + [("link_prediction", None)]
            for key, a in entries:
                r = d["results"][key] if a is not None else d["link_prediction"]
                if a is None:
                    name, know, sup = "link_pred", ("-", "Y", "-"), True
                else:
                    name, know = f"attack-{a}", KNOW[a]
                    sup = a not in UNSUPERVISED
                common = dict(base, attack=name, knowledge="/".join(know),
                              F=know[0], A_star=know[1], D_shadow=know[2],
                              supervised=sup,
                              selected_metric=r.get("selected_metric", ""),
                              n_features=r.get("n_features", ""))
                n_runs = r.get("runs", 1)
                for i in range(n_runs):
                    row = dict(common, run=i + 1)
                    for m in METRICS:
                        runs = r.get(f"{m}_runs")
                        if runs and i < len(runs):
                            row[m] = round(float(runs[i]), 6)
                        elif m in r and n_runs == 1:
                            row[m] = round(float(r[m]), 6)
                        else:
                            row[m] = ""
                    det_rows.append(row)
                srow = dict(common, n_runs=n_runs)
                for m in METRICS:
                    runs = r.get(f"{m}_runs")
                    vals = ([float(v) for v in runs] if runs
                            else ([float(r[m])] if m in r else []))
                    if vals:
                        srow[f"{m}_mean"] = round(float(np.mean(vals)), 6)
                        srow[f"{m}_std"] = round(float(np.std(vals)), 6)
                        srow[f"{m}_min"] = round(float(np.min(vals)), 6)
                        srow[f"{m}_max"] = round(float(np.max(vals)), 6)
                    else:
                        for suf in ("mean", "std", "min", "max"):
                            srow[f"{m}_{suf}"] = ""
                sum_rows.append(srow)

        if "whole" in present:
            ref = {r["attack"]: r["auc_mean"] for r in sum_rows
                   if r["setting"] == "whole" and r["auc_mean"] != ""}
            for r in sum_rows:
                b = ref.get(r["attack"])
                r["auc_delta_vs_whole"] = (round(r["auc_mean"] - b, 6)
                                           if b not in (None, "")
                                           and r["auc_mean"] != "" else "")
        for rows, kind in ((det_rows, "detailed"), (sum_rows, "summary")):
            if not rows:
                continue
            path = os.path.join(C.RESULTS, f"{ds}_attacks_{kind}.csv")
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            print(f"  {K.rel(path)}   {len(rows)} rows")
            written.append(path)
    return written


def figures(attacks, boxes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGURES, exist_ok=True)
    out = []
    for ds in C.DATASETS:
        present = [s for s in ORDER if (ds, s) in attacks]
        if not present:
            continue
        xs = np.arange(len(present))
        labels = [f"{s}\n({_box_cols(ds, s, boxes)['k']})" for s in present]

        fig, ax = plt.subplots(figsize=(8, 5.2))
        for a in range(8):
            y = [attacks[(ds, s)]["results"][str(a)]["auc"] for s in present]
            ax.plot(xs, y, marker="o", lw=1.4, ms=5,
                    label=f"attack-{a} ({'/'.join(KNOW[a])})")
        y = [attacks[(ds, s)]["link_prediction"]["auc"] for s in present]
        ax.plot(xs, y, marker="s", lw=2, ls="--", color="k",
                label="link-pred baseline")
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("coarsening ratio, percent of nodes removed (supernodes)")
        ax.set_ylabel("AUC")
        alpha = _box_cols(ds, present[0], boxes)["alpha"]
        ax.set_title(f"Link-stealing AUC vs UGC coarsening — target {ds} "
                     f"(alpha={alpha})\nknowledge as F/A*/D_shadow")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.10 * (hi - lo), hi)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=3, loc="lower center", framealpha=.95)
        fig.tight_layout()
        p = os.path.join(FIGURES, f"{ds}_auc_vs_setting.png")
        fig.savefig(p, dpi=150); plt.close(fig); out.append(p)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        for m in DISTANCES:
            y = [attacks[(ds, s)]["per_metric"]["0"].get(m) for s in present]
            if any(v is None for v in y):
                continue
            st = (dict(lw=2.4, marker="o", ms=6) if m == "correlation"
                  else dict(lw=1.2, marker=".", alpha=.8))
            ax.plot(xs, y, label=m + (" (selected)" if m == "correlation" else ""), **st)
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("coarsening ratio (supernodes)"); ax.set_ylabel("AUC")
        ax.set_title(f"Attack-0 by distance metric — target {ds}")
        ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        p = os.path.join(FIGURES, f"{ds}_attack0_metrics.png")
        fig.savefig(p, dpi=150); plt.close(fig); out.append(p)

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for s in present:
            bc = _box_cols(ds, s, boxes)
            if bc["box_acc_mean"] == "":
                continue
            aucs = [attacks[(ds, s)]["results"][str(a)]["auc"] for a in range(8)]
            ax.scatter(bc["box_acc_mean"], np.mean(aucs), s=90)
            ax.annotate(f"{s}\n({bc['k']})", (bc["box_acc_mean"], np.mean(aucs)),
                        textcoords="offset points", xytext=(7, -4), fontsize=8)
        ax.set_xlabel("black-box held-out test accuracy (%)")
        ax.set_ylabel("mean attack AUC over the 8 attacks")
        ax.set_title(f"Utility vs leakage — target {ds}")
        ax.grid(alpha=.3); fig.tight_layout()
        p = os.path.join(FIGURES, f"{ds}_utility_vs_leakage.png")
        fig.savefig(p, dpi=150); plt.close(fig); out.append(p)

    for p in out:
        print(f"  {K.rel(p)}")
    return out


def _other_benchmark(results_dir, tag_order):
    """Mean AUC per setting from a companion benchmark's summary CSVs."""
    got = {}
    for ds in C.DATASETS:
        p = os.path.join(results_dir, f"{ds}_attacks_summary.csv")
        if not os.path.exists(p):
            continue
        per = {}
        with open(p) as f:
            for row in csv.DictReader(f):
                name = row["attack"].replace("_", "-")
                if not name.startswith("attack-") or not row["auc_mean"]:
                    continue
                per.setdefault(row["setting"], []).append(float(row["auc_mean"]))
        order = [s for s in tag_order if s in per]
        if len(order) >= 2:
            got[ds] = {"first": (order[0], float(np.mean(per[order[0]]))),
                       "last": (order[-1], float(np.mean(per[order[-1]])))}
    return got


def markdown(attacks, boxes, figs):
    L = ["# UGC leakage benchmark — results", "",
         "Do the 8 link-stealing attacks of He et al. (USENIX Security 2021) get "
         "weaker when the target GNN is trained on a graph coarsened by UGC "
         "(Universal Graph Coarsening, NeurIPS 2024)?", "",
         "UGC differs from the other coarsening methods we benchmarked in one "
         "structural way: it replaces the feature matrix with "
         "`[(1-alpha)X | alpha*A]`, so every node's input vector contains its own "
         "adjacency row. That augmented matrix is what the model trains on and what "
         "it is served at inference, and inference runs over the original graph. The "
         "graph therefore reaches the model twice -- as input features and through "
         "message passing.", ""]

    for ds in C.DATASETS:
        present = [s for s in ORDER if (ds, s) in attacks]
        if not present:
            continue
        shadow = "citeseer" if ds == "cora" else "cora"
        alpha = _box_cols(ds, present[0], boxes)["alpha"]
        L += [f"## Target {ds}, shadow {shadow} (alpha = {alpha})", "",
              "| attack | F/A*/D' | " + " | ".join(present) + " | delta |",
              "|---|---|" + "---|" * (len(present) + 1)]
        for a in range(8):
            y = [attacks[(ds, s)]["results"][str(a)]["auc"] for s in present]
            L.append(f"| attack-{a} | {'/'.join(KNOW[a])} | "
                     + " | ".join(f"{v:.4f}" for v in y)
                     + f" | {y[-1] - y[0]:+.4f} |")
        y = [attacks[(ds, s)]["link_prediction"]["auc"] for s in present]
        L.append("| link-pred | – | " + " | ".join(f"{v:.4f}" for v in y)
                 + f" | {y[-1] - y[0]:+.4f} |")
        m = [float(np.mean([attacks[(ds, s)]["results"][str(a)]["auc"]
                            for a in range(8)])) for s in present]
        L.append("| **mean of 8** | | " + " | ".join(f"**{v:.4f}**" for v in m)
                 + f" | **{m[-1] - m[0]:+.4f}** |")
        L += ["", "Black-box utility, held-out test accuracy on a 60/20/20 split:", "",
              "| ratio | supernodes | bin width | test accuracy | paper | delta |",
              "|---|---|---|---|---|---|"]
        for s in present:
            bc = _box_cols(ds, s, boxes)
            if bc["box_acc_mean"] == "":
                continue
            delta = bc["box_acc_delta"]
            delta_str = f"{delta:+.2f}" if delta != "" else "–"
            paper_str = f"{bc['box_acc_paper']}" if bc["box_acc_paper"] != "" else "–"
            bw_str = f"{bc['bin_width']}" if bc["bin_width"] != "" else "–"
            L.append(f"| {s} | {bc['k']} | {bw_str} | "
                     f"{bc['box_acc_mean']:.2f} ± {bc['box_acc_std']:.2f} | "
                     f"{paper_str} | {delta_str} |")
        L.append("")

    # Optional. Point GCOND_RESULTS / FGC_RESULTS at the companion benchmarks'
    # results/ directories to get the cross-benchmark table; without them the
    # section is simply omitted.
    others = [
        ("GCond", os.environ.get("GCOND_RESULTS", os.path.abspath(os.path.join(
            C.HERE, "..", "push_benchmark", "results"))),
         ["whole", "0.25", "0.5", "1.0"]),
        ("FGC", os.environ.get("FGC_RESULTS", os.path.abspath(os.path.join(
            C.HERE, "..", "push_fgc_benchmark", "results"))),
         ["whole", "0.3", "0.1", "0.05"]),
    ]
    rows = []
    for ds in C.DATASETS:
        present = [s for s in ORDER if (ds, s) in attacks]
        if present:
            m = [float(np.mean([attacks[(ds, s)]["results"][str(a)]["auc"]
                                for a in range(8)])) for s in present]
            rows.append(("UGC", ds, present[0], m[0], present[-1], m[-1]))
    for name, d, order in others:
        got = _other_benchmark(d, order)
        for ds, v in got.items():
            rows.append((name, ds, v["first"][0], v["first"][1],
                         v["last"][0], v["last"][1]))
    if len(rows) > len([r for r in rows if r[0] == "UGC"]):
        L += ["## Compared with the GCond and FGC benchmarks", "",
              "All three benchmarks attack identical node pairs with identical "
              "50/50 splits under the same pair seed, using the same attack code, so "
              "the mean AUCs are directly comparable. The reduction axes differ: "
              "GCond synthesises 35-140 nodes, FGC merges into 135-998 supernodes, "
              "UGC hashes into 812-2329.", "",
              "| benchmark | target | no reduction | strongest reduction | change |",
              "|---|---|---|---|---|"]
        for name, ds, s0, v0, s1, v1 in sorted(rows, key=lambda r: (r[1], r[0])):
            L.append(f"| {name} | {ds} | {v0:.4f} ({s0}) | {v1:.4f} ({s1}) | "
                     f"{v1 - v0:+.4f} |")
        L.append("")

    L += ["## Attack setup", "", "| element | value |", "|---|---|",
          "| attack model | MLP, 3 hidden layers x 32, ReLU, dropout, softmax |",
          "| optimiser | Adam, lr 0.001, cross-entropy, batch 128, 50 epochs |",
          "| repetitions | 5 (attacks 0 and 2 are unsupervised, 1 run) |",
          "| adversary's attributes | **original, unaugmented** (1433 Cora / 3703 Citeseer) |",
          "| adversary's labels | random 10%, seed 123 |",
          f"| reference model g | graph-free mirror of the target: hidden {B.HIDDEN}, "
          f"{B.EPOCHS} epochs, lr {B.LR}, wd {B.WEIGHT_DECAY} |",
          "| shadow target | the other dataset at the same coarsening ratio |",
          "| ground truth | original edges: 5278 Cora, 4552 Citeseer |",
          "| pair sampling | all edges + equal non-edges, random 50/50 split |", ""]

    L += ["## Black box", "", "| element | value | source |", "|---|---|---|",
          "| model | GCN, 3 GCNConv layers: (d+n) -> 64 -> 64 -> classes | GCN.py |",
          "| optimiser | Adam, lr 0.003, decay 0.0005, 500 epochs, NLLLoss | Table 8 |",
          "| features | `[(1-alpha)X | alpha*A]`, dim d+n | UGC.py |",
          "| alpha | Cora 0.19, Citeseer 0.26, fixed across ratios | Table 6, run.sh |",
          "| split | random 60/20/20, best-validation weights | paper |",
          "| coarsening | LSH: project, floor into bins, merge on modal bin | UGC.py |",
          "| ratio | percent of nodes REMOVED | BinWidthFinder.py |", ""]

    if figs:
        L += ["## Figures", ""]
        L += [f"![{os.path.basename(p)}](figures/{os.path.basename(p)})\n"
              for p in figs]

    L += ["## Reproduction notes", "",
          "Cora at ratios 30 and 50 reproduce the paper's Table 9 to within half a "
          "point. Ratio 70 falls short by 4.26, and it is the one setting for which "
          "the authors shipped no bin width, so ours is entirely self-calibrated. "
          "Citeseer has no published UGC accuracy anywhere in the paper, so all four "
          "of its boxes are our own measurements with no target. Five of the six bin "
          "widths were calibrated here; for the one the authors did ship (Cora, "
          "ratio 50) our 0.004171 against their 0.004 gives a closer match to a true "
          "50% reduction. Per-box numbers are in `step02_validation.json` and the "
          "calibration in `step01_calibration.json`.", ""]

    path = os.path.join(C.RESULTS, "UGC_RESULTS.md")
    with open(path, "w") as f:
        f.write("\n".join(L))
    print(f"  {K.rel(path)}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_figures", action="store_true")
    args = ap.parse_args()
    K.ensure_dirs()
    attacks, boxes = load()
    if not attacks:
        raise SystemExit("no results/attacks/*.json; run step04_attacks.py first")
    print(f"loaded {len(attacks)} attack configurations, {len(boxes)} boxes\n")
    print("CSVs:")
    csvs = write_csvs(attacks, boxes)
    figs = []
    if not args.no_figures:
        print("\nfigures:")
        figs = figures(attacks, boxes)
    print("\nwriteup:")
    md = markdown(attacks, boxes, figs)
    K.record(
        "step05_aggregate",
        summary=(f"Compiled {len(csvs)} CSVs, {len(figs)} figures and the writeup "
                 f"from {len(attacks)} attack configurations"),
        inputs=["results/attacks/*.json", "results/step02_validation.json"],
        outputs=[K.rel(p) for p in csvs + figs + [md]],
        numbers={"csvs": len(csvs), "figures": len(figs),
                 "configurations": len(attacks)},
        notes=("Column names match the GCond and FGC benchmarks so all three "
               "summary CSVs concatenate."))


if __name__ == "__main__":
    main()
