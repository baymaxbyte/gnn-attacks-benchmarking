"""Step 05 -- compile the KRON attack results into CSVs, figures and a writeup.

Each reduced box is compared against its own un-reduced control: within a dataset
the only thing that changes across settings is how much of the graph the target was
trained on, so any AUC movement is attributable to the Kron reduction.

Column names match the GCond, FGC, UGC and GOREN benchmarks' summary CSVs, so all
five concatenate: they attack identical node pairs under a shared pair seed.

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

import blackbox_kron as B
import config as C

SET_TAG = "kron"
ORDER = ["whole"] + [str(r) for r in C.RATIOS]
DIRECTION = {"cora": "A", "citeseer": "B"}
FIGURES = os.path.join(C.RESULTS, "figures")

KNOW = {0: ("x", "x", "x"), 1: ("x", "x", "Y"), 2: ("Y", "x", "x"),
        3: ("x", "Y", "x"), 4: ("x", "Y", "Y"), 5: ("Y", "x", "Y"),
        6: ("Y", "Y", "x"), 7: ("Y", "Y", "Y")}
UNSUP = (0, 2)
METRICS = ("auc", "accuracy", "precision", "recall", "f1")


def load():
    attacks = {}
    for f in glob.glob(os.path.join(C.RESULTS, "attacks", "*.json")):
        d = json.load(open(f))
        ds, setting = d["target"].split(f"_{SET_TAG}_", 1)
        attacks[(ds, setting)] = d
    boxes = {}
    vp = os.path.join(C.RESULTS, "step02_validation.json")
    if os.path.exists(vp):
        for b in json.load(open(vp))["boxes"]:
            boxes[(b["dataset"], b["tag"].split("_", 1)[1])] = b
    return attacks, boxes


def _split_setting(s):
    if s == "whole":
        return None, None
    return float(s), "kron"


def _box_cols(ds, setting, boxes):
    b = boxes.get((ds, setting))
    if not b:
        return dict(ratio="", variant="", k="", box_acc_mean="", box_acc_std="",
                    box_acc_runs="", edge_weight_mean="", reduced_edges="")
    r, v = _split_setting(setting)
    c = b.get("reduced") or {}
    sp = b["split"]
    return dict(ratio=("" if r is None else r), variant=(v or "none"),
                k=b["k"] or (sp["train"] + sp["val"] + sp["test"]),
                box_acc_mean=round(b["test_acc_mean"], 4),
                box_acc_std=round(b["test_acc_std"], 4),
                box_acc_runs=";".join(f"{x:.4f}" for x in b["test_acc_runs"]),
                edge_weight_mean=(round(c["edge_weight_mean"], 4) if c else ""),
                reduced_edges=(c.get("reduced_edges", "") if c else ""))


def write_csvs(attacks, boxes):
    written = []
    for ds in C.DATASETS:
        shadow = "citeseer" if ds == "cora" else "cora"
        present = [s for s in ORDER if (ds, s) in attacks]
        det, summ = [], []
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
                    name, know, sup = f"attack-{a}", KNOW[a], a not in UNSUP
                common = dict(base, attack=name, knowledge="/".join(know),
                              F=know[0], A_star=know[1], D_shadow=know[2],
                              supervised=sup,
                              selected_metric=r.get("selected_metric", ""),
                              n_features=r.get("n_features", ""))
                nr = r.get("runs", 1)
                for i in range(nr):
                    row = dict(common, run=i + 1)
                    for m in METRICS:
                        runs = r.get(f"{m}_runs")
                        row[m] = (round(float(runs[i]), 6) if runs and i < len(runs)
                                  else (round(float(r[m]), 6)
                                        if m in r and nr == 1 else ""))
                    det.append(row)
                srow = dict(common, n_runs=nr)
                for m in METRICS:
                    runs = r.get(f"{m}_runs")
                    vals = ([float(x) for x in runs] if runs
                            else ([float(r[m])] if m in r else []))
                    if vals:
                        srow[f"{m}_mean"] = round(float(np.mean(vals)), 6)
                        srow[f"{m}_std"] = round(float(np.std(vals)), 6)
                        srow[f"{m}_min"] = round(float(np.min(vals)), 6)
                        srow[f"{m}_max"] = round(float(np.max(vals)), 6)
                    else:
                        for suf in ("mean", "std", "min", "max"):
                            srow[f"{m}_{suf}"] = ""
                summ.append(srow)

        if "whole" in present:
            ref = {x["attack"]: x["auc_mean"] for x in summ
                   if x["setting"] == "whole" and x["auc_mean"] != ""}
            for x in summ:
                b = ref.get(x["attack"])
                x["auc_delta_vs_whole"] = (round(x["auc_mean"] - b, 6)
                                           if b not in (None, "")
                                           and x["auc_mean"] != "" else "")
        for rows, kind in ((det, "detailed"), (summ, "summary")):
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

        # mean AUC over the 8 attacks vs reduction, against the control
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        rs = [r for r in C.RATIOS if (ds, str(r)) in attacks]
        xs = np.arange(len(rs))
        y = [float(np.mean([attacks[(ds, str(r))]["results"][str(a)]["auc"]
                            for a in range(8)])) for r in rs]
        ax.plot(xs, y, lw=2, marker="s", label="Kron-reduced box")
        if (ds, "whole") in attacks:
            w = float(np.mean([attacks[(ds, "whole")]["results"][str(a)]["auc"]
                               for a in range(8)]))
            ax.axhline(w, ls=":", color="k", lw=1.5, label="no reduction (control)")
        lp = attacks[(ds, present[0])]["link_prediction"]["auc"]
        ax.axhline(lp, ls="--", color="gray", lw=1.5, label="link-pred baseline")
        ax.set_xticks(xs); ax.set_xticklabels([f"r={r}" for r in rs])
        ax.set_xlabel("Kron reduction ratio (fraction of nodes removed)")
        ax.set_ylabel("mean attack AUC over the 8 attacks")
        ax.set_title(f"Link-stealing leakage vs Kron reduction — target {ds}")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(FIGURES, f"{ds}_auc_vs_reduction.png")
        fig.savefig(p, dpi=150); plt.close(fig); out.append(p)

        # per-attack AUC across all settings
        fig, ax = plt.subplots(figsize=(9, 5))
        xs = np.arange(len(present))
        for a in range(8):
            y = [attacks[(ds, s)]["results"][str(a)]["auc"] for s in present]
            ax.plot(xs, y, marker="o", lw=1.3, ms=4,
                    label=f"attack-{a} ({'/'.join(KNOW[a])})")
        y = [attacks[(ds, s)]["link_prediction"]["auc"] for s in present]
        ax.plot(xs, y, marker="s", lw=2, ls="--", color="k", label="link-pred")
        ax.set_xticks(xs)
        ax.set_xticklabels([s.replace("_", "\n") for s in present], fontsize=8)
        ax.set_ylabel("AUC")
        ax.set_title(f"Link-stealing AUC across settings — target {ds}")
        lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.10 * (hi - lo), hi)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=3, loc="lower center", framealpha=.95)
        fig.tight_layout()
        p = os.path.join(FIGURES, f"{ds}_auc_vs_setting.png")
        fig.savefig(p, dpi=150); plt.close(fig); out.append(p)

    for p in out:
        print(f"  {K.rel(p)}")
    return out


def markdown(attacks, boxes, figs):
    L = ["# KRON leakage benchmark — results", "",
         "Do the 8 link-stealing attacks of He et al. (USENIX Security 2021) get "
         "weaker when the target GNN is trained on a Kron-reduced graph (Dorfler & "
         "Bullo, arXiv 1102.2950)?", "",
         "Kron reduction eliminates a set of interior nodes by taking the Schur "
         "complement of the graph Laplacian with respect to them. The surviving "
         "boundary nodes are a subset of the ORIGINAL nodes and keep their own "
         "features and labels, so unlike the coarsening benchmarks nothing is merged "
         "into a synthetic supernode. Here the interior is the lowest-degree "
         "n*r nodes, chosen per connected component so every component keeps at least "
         "one boundary node.", "",
         "Each reduced box is trained on the reduced graph and evaluated on the "
         "original graph, so the only difference from its own control is the training "
         "graph. The paper never runs node classification, so there is no published "
         "accuracy for any dataset and it uses neither Cora nor Citeseer. Every "
         "accuracy here is our own measurement; the one paper-grounded check is that "
         "the reduced matrix is a valid Laplacian (max|L_red @ 1| ~ 1e-14, step 01).",
         ""]

    for ds in C.DATASETS:
        present = [s for s in ORDER if (ds, s) in attacks]
        if not present:
            continue
        shadow = "citeseer" if ds == "cora" else "cora"
        L += [f"## Target {ds}, shadow {shadow}", "",
              "| attack | F/A*/D' | " + " | ".join(present) + " |",
              "|---|---|" + "---|" * len(present)]
        for a in range(8):
            y = [attacks[(ds, s)]["results"][str(a)]["auc"] for s in present]
            L.append(f"| attack-{a} | {'/'.join(KNOW[a])} | "
                     + " | ".join(f"{v:.4f}" for v in y) + " |")
        y = [attacks[(ds, s)]["link_prediction"]["auc"] for s in present]
        L.append("| link-pred | – | " + " | ".join(f"{v:.4f}" for v in y) + " |")
        m = [float(np.mean([attacks[(ds, s)]["results"][str(a)]["auc"]
                            for a in range(8)])) for s in present]
        L.append("| **mean of 8** | | " + " | ".join(f"**{v:.4f}**" for v in m) + " |")
        L.append("")

    L += ["## Reduction effect (mean AUC over the 8 attacks vs the control)", "",
          "| setting | box acc | mean attack AUC | AUC vs control |",
          "|---|---|---|---|"]
    for ds in C.DATASETS:
        if (ds, "whole") not in attacks:
            continue
        wa = float(np.mean([attacks[(ds, "whole")]["results"][str(a)]["auc"]
                            for a in range(8)]))
        for s in [x for x in ORDER if (ds, x) in attacks]:
            a8 = float(np.mean([attacks[(ds, s)]["results"][str(a)]["auc"]
                                for a in range(8)]))
            box = boxes.get((ds, s))
            acc = f"{box['test_acc_mean']:.2f}" if box else "–"
            delta = "control" if s == "whole" else f"{a8 - wa:+.4f}"
            L.append(f"| {ds} {s} | {acc} | {a8:.4f} | {delta} |")
    L.append("")

    L += ["## Attack setup", "", "| element | value |", "|---|---|",
          "| attack model | MLP, 3 hidden layers x 32, ReLU, dropout, softmax |",
          "| optimiser | Adam, lr 0.001, cross-entropy, batch 128, 50 epochs |",
          "| repetitions | 5 (attacks 0 and 2 are unsupervised, 1 run) |",
          "| adversary's attributes | original node attributes |",
          "| adversary's labels | random 10%, seed 123 |",
          f"| reference model g | graph-free mirror of the target: hidden {B.HIDDEN}, "
          f"{B.EPOCHS} epochs, lr {B.LR}, wd {B.WEIGHT_DECAY} |",
          "| shadow target | the other dataset at the same setting |",
          "| ground truth | original edges: 5278 Cora, 4552 Citeseer |",
          "| pair sampling | all edges + equal non-edges, random 50/50 split |", "",
          "## Black box", "", "| element | value | source |", "|---|---|---|",
          "| reduction | Kron / Schur complement of the Laplacian | arXiv 1102.2950 |",
          "| interior nodes | lowest-degree n*r, one boundary node per component "
          "minimum | ours |",
          "| ratio | fraction of nodes REMOVED | matches UGC and GOREN |",
          "| reduced weights | Schur-complement conductances, threaded through the "
          "GCN | Schur complement |",
          "| classifier | GCN, 3 GCNConv: d -> 64 -> 64 -> classes, edge weights "
          "threaded through | ours |",
          "| optimiser | Adam, lr 0.003, decay 0.0005, 500 epochs | ours |",
          "| split | random 60/20/20, best-validation weights | ours |",
          "| inference | original graph, unit weights | ours |", ""]

    if figs:
        L += ["## Figures", ""]
        L += [f"![{os.path.basename(p)}](figures/{os.path.basename(p)})\n"
              for p in figs]

    L += ["## Notes", "",
          "The Kron reduction reaches the black box only through the trained "
          "parameters, never at query time, because inference runs over the original "
          "graph with unit weights. The reduced graph's Schur-complement weights "
          "shape what the model learns, but the adversary always queries the same "
          "original topology, so any leakage change is a training-time effect.", "",
          "Kept nodes are original nodes with their own features and labels, so this "
          "is the only benchmark of the five where the training set is a literal "
          "subset of the real nodes rather than synthetic or merged supernodes.", ""]

    path = os.path.join(C.RESULTS, "KRON_RESULTS.md")
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
        notes=("summary CSVs carry auc_delta_vs_whole. Column names match the GCond, "
               "FGC, UGC and GOREN benchmarks so all five concatenate on identical "
               "node pairs."))


if __name__ == "__main__":
    main()
