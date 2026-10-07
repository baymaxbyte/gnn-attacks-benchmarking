"""Step 04 -- compile the sensitive-attribute attack results.

Produces per-dataset CSVs and a writeup of the attack AUC per method, ratio and
sensitive feature, against the no-reduction control. AUC 0.5 = no leakage.

    python step04_aggregate.py
    python step04_aggregate.py --no_figures
"""
import argparse
import csv
import glob
import json
import os

import common as K

os.environ.update(K.single_thread_env())

import numpy as np

import config as C

METHOD_ORDER = ["control", "kron", "ugc", "fgc", "goren", "gcond"]


def parse_cell(cell):
    p = cell.split("_")
    ds, method, rest = p[0], p[1], p[2:]
    if rest == ["whole"]:
        return ds, method, None, ""
    ratio = rest[0][1:] if rest and rest[0].startswith("r") else ""
    variant = rest[1] if len(rest) > 1 else ""
    return ds, method, ratio, variant


def sort_key(cell):
    ds, method, ratio, variant = parse_cell(cell)
    mi = METHOD_ORDER.index(method) if method in METHOD_ORDER else 99
    rr = -1.0 if ratio in (None, "") else float(ratio)
    return (mi, rr, variant)


def load_attacks():
    out = {}
    for p in glob.glob(os.path.join(C.RESULTS, "attacks", "*.json")):
        d = json.load(open(p))
        out[(d["cell"], d["feat_idx"])] = d
    return out


def load_box_acc():
    p = os.path.join(C.RESULTS, "step02_boxes.json")
    acc = {}
    if os.path.exists(p):
        agg = {}
        for b in json.load(open(p))["boxes"]:
            agg.setdefault(b["tag"], []).append(b["box_acc"])
        acc = {t: float(np.mean(v)) for t, v in agg.items()}
    return acc


def write_csvs(attacks, box_acc):
    written = []
    for ds in C.DATASETS:
        feats = C.SENSITIVE[ds]
        cells = sorted({c for (c, f) in attacks if parse_cell(c)[0] == ds},
                       key=sort_key)
        ctrl = next((c for c in cells if parse_cell(c)[1] == "control"), None)

        det = []
        for cell in cells:
            _, method, ratio, variant = parse_cell(cell)
            for f in feats:
                a = attacks.get((cell, f))
                if not a:
                    continue
                ca = attacks.get((ctrl, f)) if ctrl else None
                row = {"dataset": ds, "method": method, "ratio": ratio or "",
                       "variant": variant, "cell": cell, "feat_idx": f,
                       "pos_rate": round(a["pos_rate"], 4),
                       "box_acc": round(box_acc.get(cell, float("nan")), 2),
                       "auc_mean": round(a["auc_mean"], 4),
                       "auc_std": round(a["auc_std"], 4),
                       "accuracy_mean": round(a["accuracy_mean"], 4),
                       "majority_baseline": round(a["majority_baseline"], 4)}
                if ca:
                    row["auc_vs_control"] = round(a["auc_mean"] - ca["auc_mean"], 4)
                det.append(row)
        path = os.path.join(C.RESULTS, f"{ds}_sia_detailed.csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(det[0].keys()))
            w.writeheader(); w.writerows(det)
        print(f"  {K.rel(path)}   {len(det)} rows")
        written.append(path)

        # summary: per (method,ratio) averaged over the 3 features + per-feature AUC
        summ = []
        for cell in cells:
            _, method, ratio, variant = parse_cell(cell)
            per = {f: attacks[(cell, f)]["auc_mean"] for f in feats
                   if (cell, f) in attacks}
            if not per:
                continue
            row = {"dataset": ds, "method": method, "ratio": ratio or "",
                   "variant": variant, "cell": cell,
                   "box_acc": round(box_acc.get(cell, float("nan")), 2),
                   "auc_mean_over_feats": round(float(np.mean(list(per.values()))), 4)}
            for f in feats:
                row[f"auc_feat{f}"] = round(per[f], 4) if f in per else ""
            summ.append(row)
        path = os.path.join(C.RESULTS, f"{ds}_sia_summary.csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
            w.writeheader(); w.writerows(summ)
        print(f"  {K.rel(path)}   {len(summ)} rows")
        written.append(path)
    return written


def figures(attacks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = os.path.join(C.RESULTS, "figures")
    os.makedirs(figdir, exist_ok=True)
    out = []
    for ds in C.DATASETS:
        feats = C.SENSITIVE[ds]
        cells = sorted({c for (c, f) in attacks if parse_cell(c)[0] == ds},
                       key=sort_key)
        if not cells:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        xs = range(len(cells))
        for f in feats:
            ys = [attacks[(c, f)]["auc_mean"] if (c, f) in attacks else np.nan
                  for c in cells]
            ax.plot(xs, ys, marker="o", label=f"feat {f}")
        ax.axhline(0.5, ls="--", color="gray", label="no leakage (0.5)")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([c.replace(f"{ds}_", "") for c in cells], rotation=90,
                           fontsize=6)
        ax.set_ylabel("sensitive-attribute AUC")
        ax.set_title(f"Sensitive-attribute leakage per cell and feature -- {ds}")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout()
        p = os.path.join(figdir, f"{ds}_sia.png")
        fig.savefig(p, dpi=150); plt.close(fig); out.append(p)
        print(f"  {K.rel(p)}")
    return out


def markdown(attacks, box_acc, figs):
    L = ["# Sensitive-attribute inference benchmark -- results", "",
         "Can an adversary recover a hidden binary feature of a node from a black box "
         "trained on a reduced graph, given all other features? For each node the box "
         "is probed with the sensitive bit forced to 0 and to 1 (`Z=[p0||p1]`), and an "
         "MLP maps `Z` to the true bit, scored by AUC (0.5 = no leakage). Cora/Citeseer "
         "have no native sensitive attribute, so the most class-balanced binary "
         "features stand in; three per dataset.", ""]
    for ds in C.DATASETS:
        feats = C.SENSITIVE[ds]
        cells = sorted({c for (c, f) in attacks if parse_cell(c)[0] == ds},
                       key=sort_key)
        L += [f"## {ds}", "",
              "AUC per cell (rows) and sensitive feature (cols). Pos-rate of each "
              "feature in the header.", "",
              "| method/ratio | box acc | " + " | ".join(
                  f"feat {f} ({100*attacks[(cells[0],f)]['pos_rate']:.0f}% ones)"
                  for f in feats if (cells[0], f) in attacks) + " |",
              "|---|---|" + "---|" * len(feats)]
        for c in cells:
            _, method, ratio, variant = parse_cell(c)
            lab = method + ("" if ratio in (None, "") else f" r{ratio}") + (
                f"/{variant}" if variant else "")
            cells_auc = " | ".join(
                f"{attacks[(c,f)]['auc_mean']:.3f}" if (c, f) in attacks else "-"
                for f in feats)
            L.append(f"| {lab} | {box_acc.get(c, float('nan')):.1f} | {cells_auc} |")
        L.append("")
    L += ["## Reading it", "",
          "- **control** is the leakage with no reduction.",
          "- Leakage is dominated by *which* feature is sensitive (some bits are far "
          "more predictable from the graph and the other features than others), and "
          "is comparatively flat across reduction methods and ratios.",
          "- AUC near 0.5 means the two-world probe reveals nothing about the bit.", ""]
    if figs:
        L += ["## Figures", ""]
        L += [f"![{os.path.basename(p)}](figures/{os.path.basename(p)})\n"
              for p in figs]
    path = os.path.join(C.RESULTS, "SENSITIVE_RESULTS.md")
    with open(path, "w") as f:
        f.write("\n".join(L))
    print(f"  {K.rel(path)}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_figures", action="store_true")
    args = ap.parse_args()
    K.ensure_dirs()
    attacks = load_attacks()
    box_acc = load_box_acc()
    if not attacks:
        raise SystemExit("no results/attacks/*.json; run step03_attack.py first")
    print(f"loaded {len(attacks)} (cell,feature) attack results\n\nCSVs:")
    csvs = write_csvs(attacks, box_acc)
    figs = []
    if not args.no_figures:
        print("\nfigures:")
        figs = figures(attacks)
    print("\nwriteup:")
    md = markdown(attacks, box_acc, figs)
    K.record("step04_aggregate",
             summary=(f"Compiled {len(csvs)} CSVs, {len(figs)} figures and the writeup "
                      f"from {len(attacks)} (cell, feature) attack results"),
             inputs=["results/attacks/*.json", "results/step02_boxes.json"],
             outputs=[K.rel(p) for p in csvs + figs + [md]],
             numbers={"pairs": len(attacks)},
             notes="Sensitive-attribute AUC per method/ratio/feature vs the control.")


if __name__ == "__main__":
    main()
