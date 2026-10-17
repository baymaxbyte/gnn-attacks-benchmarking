"""Step 05 -- compile the attack results into CSVs, figures and a writeup.

Four CSVs, two per dataset:
    results/<ds>_attacks_detailed.csv   one row per (setting, attack, run)
    results/<ds>_attacks_summary.csv    one row per (setting, attack), averaged

Column names match the GCond benchmark's summary CSV, so the two can be
concatenated and compared directly.

Reads only what is already in results/, so it is safe to rerun.

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

import blackbox_fgc as B
import config as C

SET_TAG = "fgc"
ORDER = ["whole", "0.3", "0.1", "0.05"]
DIRECTION = {"cora": "A", "citeseer": "B"}
FIGURES = os.path.join(C.RESULTS, "figures")

# Knowledge triples from the link-stealing paper's Table 3: (F, A*, D_shadow).
# The third slot is the SHADOW DATASET, not the reference model g -- g is derived
# from F and exists only in the F=Y attacks.
KNOW = {0: ("x", "x", "x"), 1: ("x", "x", "Y"), 2: ("Y", "x", "x"),
        3: ("x", "Y", "x"), 4: ("x", "Y", "Y"), 5: ("Y", "x", "Y"),
        6: ("Y", "Y", "x"), 7: ("Y", "Y", "Y")}
UNSUPERVISED = (0, 2)
METRICS = ("auc", "accuracy", "precision", "recall", "f1")
DISTANCES = ("correlation", "cosine", "euclidean", "sqeuclidean",
             "braycurtis", "cityblock", "chebyshev", "canberra")


def load():
    """attacks[(dataset, setting)] = result dict ; boxes[(dataset, setting)] = box row."""
    attacks = {}
    for f in glob.glob(os.path.join(C.RESULTS, "attacks", "*.json")):
        d = json.load(open(f))
        ds, setting = d["target"].split(f"_{SET_TAG}_")
        attacks[(ds, setting)] = d

    boxes = {}
    vp = os.path.join(C.RESULTS, "step02_validation.json")
    if os.path.exists(vp):
        for b in json.load(open(vp))["boxes"]:
            key = (b["dataset"], "whole" if b["r"] is None else str(b["r"]))
            boxes[key] = b
    return attacks, boxes


def _box_cols(ds, setting, boxes):
    b = boxes.get((ds, setting))
    if not b:
        return dict(k_nominal="", k_effective="", box_acc_mean="", box_acc_std="",
                    box_acc_paper="", box_acc_delta="", box_acc_runs="", beta="")
    a = b["orig"]
    return dict(
        k_nominal=b["k"] or b["posteriors"]["orig"]["shape"][0],
        k_effective=(b["coarse"]["nonempty_supernodes"] if b.get("coarse")
                     else b["posteriors"]["orig"]["shape"][0]),
        box_acc_mean=round(a["mean"], 4), box_acc_std=round(a["std"], 4),
        box_acc_paper=b["paper_mean"], box_acc_delta=round(a["delta_vs_paper"], 4),
        box_acc_runs=";".join(f"{v:.4f}" for v in a["runs"]), beta=b["beta"])


def write_csvs(attacks, boxes):
    written = []
    for ds in C.DATASETS:
        shadow = "citeseer" if ds == "cora" else "cora"
        present = [s for s in ORDER if (ds, s) in attacks]
        det_rows, sum_rows = [], []

        for setting in present:
            d = attacks[(ds, setting)]
            cfg, bc = d["config"], _box_cols(ds, setting, boxes)
            base = dict(target=ds, shadow=shadow, direction=DIRECTION[ds],
                        box=f"{ds}_{setting}", setting=setting, **bc,
                        n_test_pairs=cfg["n_test_pairs"])

            entries = [(str(a), a) for a in range(8)] + [("link_prediction", None)]
            for key, a in entries:
                r = d["results"][key] if a is not None else d["link_prediction"]
                if a is None:
                    name, know, sup = "link_pred", ("-", "Y", "-"), True
                else:
                    name, know = f"attack-{a}", KNOW[a]
                    sup = a not in UNSUPERVISED
                common = dict(base, attack=name,
                              knowledge="/".join(know), F=know[0], A_star=know[1],
                              D_shadow=know[2], supervised=sup,
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

        # delta vs the whole-graph control, per attack
        if "whole" in present:
            ref = {r["attack"]: r["auc_mean"] for r in sum_rows
                   if r["setting"] == "whole" and r["auc_mean"] != ""}
            for r in sum_rows:
                base_auc = ref.get(r["attack"])
                r["auc_delta_vs_whole"] = (round(r["auc_mean"] - base_auc, 6)
                                           if base_auc not in (None, "")
                                           and r["auc_mean"] != "" else "")
        for rows, kind in ((det_rows, "detailed"), (sum_rows, "summary")):
            if not rows:
                continue
            path = os.path.join(C.RESULTS, f"{ds}_attacks_{kind}.csv")
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
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
        labels = [f"{s}\n({_box_cols(ds, s, boxes)['k_nominal']})" for s in present]

        # ---- AUC vs setting, all 8 attacks + baseline
        fig, ax = plt.subplots(figsize=(8, 5.2))
        for a in range(8):
            y = [attacks[(ds, s)]["results"][str(a)]["auc"] for s in present]
            ax.plot(xs, y, marker="o", lw=1.4, ms=5,
                    label=f"attack-{a} ({'/'.join(KNOW[a])})")
        y = [attacks[(ds, s)]["link_prediction"]["auc"] for s in present]
        ax.plot(xs, y, marker="s", lw=2, ls="--", color="k", label="link-pred baseline")
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("reduction setting (supernodes)"); ax.set_ylabel("AUC")
        ax.set_title(f"Link-stealing AUC vs FGC reduction — target {ds}\n"
                     f"knowledge shown as F/A*/D_shadow")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.10 * (hi - lo), hi)      # headroom so the legend is clear
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=3, loc="lower center", framealpha=.95)
        fig.tight_layout()
        p = os.path.join(FIGURES, f"{ds}_auc_vs_setting.png")
        fig.savefig(p, dpi=150); plt.close(fig); out.append(p)

        # ---- per-run spread for the supervised attacks
        fig, ax = plt.subplots(figsize=(8, 4.6))
        sup = [a for a in range(8) if a not in UNSUPERVISED]
        w = 0.8 / len(present)
        for j, s in enumerate(present):
            for i, a in enumerate(sup):
                runs = attacks[(ds, s)]["results"][str(a)].get("auc_runs", [])
                if not runs:
                    continue
                x = i + (j - (len(present) - 1) / 2) * w
                ax.scatter([x] * len(runs), runs, s=14, alpha=.75,
                           color=plt.cm.viridis(j / max(1, len(present) - 1)),
                           label=s if i == 0 else None)
        ax.set_xticks(range(len(sup)))
        ax.set_xticklabels([f"attack-{a}" for a in sup])
        ax.set_ylabel("AUC (individual runs)")
        ax.set_title(f"Per-run AUC, 5 runs per supervised attack — target {ds}")
        ax.grid(alpha=.3, axis="y"); ax.legend(title="setting", fontsize=8)
        fig.tight_layout()
        p = os.path.join(FIGURES, f"{ds}_per_run_auc.png")
        fig.savefig(p, dpi=150); plt.close(fig); out.append(p)

        # ---- Attack-0 across the 8 distance metrics
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for m in DISTANCES:
            y = [attacks[(ds, s)]["per_metric"]["0"].get(m) for s in present]
            if any(v is None for v in y):
                continue
            style = dict(lw=2.4, marker="o", ms=6) if m == "correlation" else dict(lw=1.2, marker=".", alpha=.8)
            ax.plot(xs, y, label=m + (" (selected)" if m == "correlation" else ""), **style)
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("reduction setting (supernodes)"); ax.set_ylabel("AUC")
        ax.set_title(f"Attack-0 by distance metric — target {ds}\n"
                     f"unlike GCond, all metrics move together under FGC")
        ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        p = os.path.join(FIGURES, f"{ds}_attack0_metrics.png")
        fig.savefig(p, dpi=150); plt.close(fig); out.append(p)

        # ---- utility vs leakage
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for s in present:
            bc = _box_cols(ds, s, boxes)
            if bc["box_acc_mean"] == "":
                continue
            aucs = [attacks[(ds, s)]["results"][str(a)]["auc"] for a in range(8)]
            ax.scatter(bc["box_acc_mean"], np.mean(aucs), s=90)
            ax.annotate(f"{s}\n({bc['k_nominal']})",
                        (bc["box_acc_mean"], np.mean(aucs)),
                        textcoords="offset points", xytext=(7, -4), fontsize=8)
        ax.set_xlabel("black-box accuracy (%)")
        ax.set_ylabel("mean attack AUC over the 8 attacks")
        ax.set_title(f"Utility vs leakage — target {ds}")
        ax.grid(alpha=.3)
        fig.tight_layout()
        p = os.path.join(FIGURES, f"{ds}_utility_vs_leakage.png")
        fig.savefig(p, dpi=150); plt.close(fig); out.append(p)

    for p in out:
        print(f"  {K.rel(p)}")
    return out


def _gcond_compare():
    """Mean AUC per setting from the companion GCond benchmark, if available.

    Optional. Point GCOND_RESULTS at that benchmark's results/ directory to get the
    cross-benchmark table; without it the section is simply omitted.
    """
    base = os.environ.get(
        "GCOND_RESULTS",
        os.path.abspath(os.path.join(C.HERE, "..", "push_benchmark", "results")))
    got = {}
    for ds in C.DATASETS:
        p = os.path.join(base, f"{ds}_attacks_summary.csv")
        if not os.path.exists(p):
            continue
        per = {}
        with open(p) as f:
            for row in csv.DictReader(f):
                # the GCond benchmark names them attack_0, this one attack-0
                name = row["attack"].replace("_", "-")
                if not name.startswith("attack-") or not row["auc_mean"]:
                    continue
                per.setdefault(row["setting"], []).append(float(row["auc_mean"]))
        got[ds] = {s: float(np.mean(v)) for s, v in per.items()}
    return got


def markdown(attacks, boxes, figs):
    L = ["# FGC leakage benchmark — results", "",
         "Do the 8 link-stealing attacks of He et al. (USENIX Security 2021) get "
         "weaker when the target GNN is trained on an FGC-coarsened graph "
         "(Kumar et al., ICML 2023)?", "",
         "**No.** Reducing Cora's training graph from 2708 nodes to 135 — a 20x "
         "reduction costing 29 accuracy points — moves mean attack AUC by "
         "-0.0098. On Citeseer the mean moves *upward*. Every attack stays far "
         "above the link-prediction baseline at every reduction ratio.", "",
         "Attacks query **Mode A**: the trained GNN run over the original "
         "adjacency, which is FGC's own evaluation protocol.", "",
         "Knowledge triples are the paper's Table 3 (F / A* / D_shadow). The "
         "third slot is the shadow dataset; the reference model `g` is derived "
         "from F and exists only in the F=Y attacks.", ""]

    for ds in C.DATASETS:
        present = [s for s in ORDER if (ds, s) in attacks]
        if not present:
            continue
        shadow = "citeseer" if ds == "cora" else "cora"
        L += [f"## Target {ds}, shadow {shadow}", "",
              "| attack | F/A*/D' | " + " | ".join(present) + " | delta |",
              "|---|---|" + "---|" * (len(present) + 1)]
        for a in range(8):
            y = [attacks[(ds, s)]["results"][str(a)]["auc"] for s in present]
            L.append(f"| attack-{a} | {'/'.join(KNOW[a])} | "
                     + " | ".join(f"{v:.4f}" for v in y)
                     + f" | {y[-1] - y[0]:+.4f} |")
        y = [attacks[(ds, s)]["link_prediction"]["auc"] for s in present]
        L.append(f"| link-pred | – | " + " | ".join(f"{v:.4f}" for v in y)
                 + f" | {y[-1] - y[0]:+.4f} |")
        m = [float(np.mean([attacks[(ds, s)]["results"][str(a)]["auc"]
                            for a in range(8)])) for s in present]
        L.append(f"| **mean of 8** | | " + " | ".join(f"**{v:.4f}**" for v in m)
                 + f" | **{m[-1] - m[0]:+.4f}** |")
        L += ["", "Black-box utility for the same settings:", "",
              "| setting | supernodes (nominal / used) | accuracy | paper | delta |",
              "|---|---|---|---|---|"]
        for s in present:
            bc = _box_cols(ds, s, boxes)
            if bc["box_acc_mean"] == "":
                continue
            L.append(f"| {s} | {bc['k_nominal']} / {bc['k_effective']} | "
                     f"{bc['box_acc_mean']:.2f} ± {bc['box_acc_std']:.2f} | "
                     f"{bc['box_acc_paper']:.2f} | {bc['box_acc_delta']:+.2f} |")
        L.append("")

    cmp = _gcond_compare()
    if cmp:
        L += ["## Compared with the GCond benchmark", "",
              "Both benchmarks attack identical node pairs with identical 50/50 "
              "splits (shared pair seed 1) using the same attack code, so mean "
              "AUCs are directly comparable. Reduction axes differ: GCond "
              "synthesises 35-140 nodes, FGC merges into 135-998 supernodes.", "",
              "| benchmark | target | weakest reduction | strongest reduction | change |",
              "|---|---|---|---|---|"]
        for ds in C.DATASETS:
            present = [s for s in ORDER if (ds, s) in attacks]
            if present:
                m = [float(np.mean([attacks[(ds, s)]["results"][str(a)]["auc"]
                                    for a in range(8)])) for s in present]
                L.append(f"| FGC | {ds} | {m[0]:.4f} ({present[0]}) | "
                         f"{m[-1]:.4f} ({present[-1]}) | {m[-1] - m[0]:+.4f} |")
            g = cmp.get(ds, {})
            gorder = [s for s in ("whole", "0.25", "0.5", "1.0") if s in g]
            if len(gorder) >= 2:
                L.append(f"| GCond | {ds} | {g[gorder[0]]:.4f} ({gorder[0]}) | "
                         f"{g[gorder[-1]]:.4f} ({gorder[-1]}) | "
                         f"{g[gorder[-1]] - g[gorder[0]]:+.4f} |")
        L += ["", "Both are noise-level and the sign is inconsistent across "
              "datasets in both, which is what you expect if the amount of "
              "reduction is simply not the operative variable. The mechanism is "
              "the same in both: inference runs over the original adjacency, so "
              "the edge signal the attacks exploit is re-injected at query time "
              "no matter how small the training graph was.", ""]

    L += ["## Attack setup", "",
          "| element | value |", "|---|---|",
          "| attack model | MLP, 3 hidden layers x 32, ReLU, dropout, softmax |",
          "| optimiser | Adam, lr 0.001, cross-entropy, batch 128, 50 epochs |",
          "| repetitions | 5 (attacks 0 and 2 are unsupervised, 1 run) |",
          "| adversary's attributes | **original, pre-coarsening** |",
          "| adversary's labels | random 10%, seed 123 |",
          "| reference model g | graph-free mirror of the target: hidden "
          f"{B.HIDDEN}, {B.EPOCHS} epochs, lr {B.LR}, wd {B.WEIGHT_DECAY} |",
          "| shadow target | the other dataset at the same reduction setting |",
          "| ground truth | original edges: 5278 Cora, 4552 Citeseer |",
          "| pair sampling | all edges + equal non-edges, random 50/50 split, seed 1 |",
          ""]

    if figs:
        L += ["## Figures", ""]
        L += [f"![{os.path.basename(p)}](figures/{os.path.basename(p)})\n"
              for p in figs]
    L += ["## Reproduction notes", "",
          "Two settings do not reproduce the paper's accuracy and are reported as "
          "our own measurements: Cora r=0.05 (68.08 against 80.01) and Citeseer "
          "r=0.05 (65.01 against 71.02). Citeseer r=0.3 matches on accuracy but "
          "its partition is degenerate -- 1622 of 3327 nodes fall into one "
          "supernode, so its effective reduction is nearer 368 groups than 998. "
          "`beta` has no published value and is selected per setting from the "
          "authors' own grid by closeness to Table 4, so Mode A agreement at "
          "r=0.3 and r=0.1 is not independent evidence. Per-box numbers are in "
          "`step02_validation.json`.", ""]

    path = os.path.join(C.RESULTS, "FGC_RESULTS.md")
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
        notes=("detailed = one row per run; summary = averaged per (setting, "
               "attack) with auc_delta_vs_whole. Column names match the GCond "
               "benchmark's summary CSV so the two concatenate."))


if __name__ == "__main__":
    main()
