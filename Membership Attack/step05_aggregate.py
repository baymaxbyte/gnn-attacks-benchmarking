"""Step 05 -- compile the membership-attack results into CSVs, figures, a writeup.

The headline is membership-inference AUC per method and ratio, compared against the
no-reduction control, so the question "does graph reduction reduce membership
leakage?" can be read off directly. AUC 0.5 = no leakage.

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

import config as C

METHOD_ORDER = ["control", "kron", "ugc", "fgc", "goren", "gcond"]
THR_KEYS = ["thr_confidence", "thr_neg_entropy", "thr_p_true", "thr_neg_loss",
            "thr_correct"]


def parse_tag(tag):
    parts = tag.split("_")
    ds, method, rest = parts[0], parts[1], parts[2:]
    if rest == ["whole"]:
        return ds, method, None, ""
    ratio = rest[0][1:] if rest and rest[0].startswith("r") else ""
    variant = rest[1] if len(rest) > 1 else ""
    return ds, method, ratio, variant


def load_attacks():
    out = {}
    for p in glob.glob(os.path.join(C.RESULTS, "attacks", "*.json")):
        d = json.load(open(p))
        out[d["tag"]] = d
    return out


def load_boxes():
    p = os.path.join(C.RESULTS, "step03_boxes.json")
    boxes = {}
    if os.path.exists(p):
        agg = {}
        for b in json.load(open(p))["boxes"]:
            agg.setdefault(b["tag"], []).append(b)
        for tag, rs in agg.items():
            boxes[tag] = {
                "member_acc": float(np.mean([r["member_acc"] for r in rs])),
                "nonmember_acc": float(np.mean([r["nonmember_acc"] for r in rs])),
                "acc_gap": float(np.mean([r["acc_gap"] for r in rs])),
                "k_train": rs[0]["k_train"],
            }
    return boxes


def _mean(a, key):
    return a.get(key, {}).get("mean", "")


def sort_key(tag):
    ds, method, ratio, variant = parse_tag(tag)
    mi = METHOD_ORDER.index(method) if method in METHOD_ORDER else 99
    rr = -1.0 if ratio in (None, "") else float(ratio)
    return (mi, rr, variant)


def write_csvs(attacks, boxes):
    written = []
    for ds in C.DATASETS:
        tags = sorted([t for t in attacks if parse_tag(t)[0] == ds], key=sort_key)
        ctrl = next((t for t in tags if parse_tag(t)[1] == "control"), None)
        ctrl_best = _mean(attacks[ctrl]["aucs"], "best_threshold") if ctrl else None
        ctrl_shadow = _mean(attacks[ctrl]["aucs"], "shadow_mlp") if ctrl else None

        summ, det = [], []
        for tag in tags:
            _, method, ratio, variant = parse_tag(tag)
            a = attacks[tag]["aucs"]
            bx = boxes.get(tag, {})
            best_thr = _mean(a, "best_threshold")
            shadow = _mean(a, "shadow_mlp")
            row = {
                "dataset": ds, "method": method, "ratio": ratio or "",
                "variant": variant, "cell": tag,
                "shadow_cell": attacks[tag]["shadow_tag"],
                "k_train": bx.get("k_train", ""),
                "member_acc": round(bx.get("member_acc", float("nan")), 2),
                "nonmember_acc": round(bx.get("nonmember_acc", float("nan")), 2),
                "acc_gap": round(bx.get("acc_gap", float("nan")), 2),
                "best_threshold_auc": round(best_thr, 4) if best_thr != "" else "",
                "shadow_mlp_auc": round(shadow, 4) if shadow != "" else "",
            }
            for k in THR_KEYS:
                row[k + "_auc"] = round(_mean(a, k), 4) if _mean(a, k) != "" else ""
            if ctrl_best is not None and best_thr != "":
                row["best_thr_vs_control"] = round(best_thr - ctrl_best, 4)
            if ctrl_shadow is not None and shadow != "":
                row["shadow_vs_control"] = round(shadow - ctrl_shadow, 4)
            summ.append(row)

            for k in ["best_threshold", "shadow_mlp"] + THR_KEYS:
                det.append({"dataset": ds, "method": method, "ratio": ratio or "",
                            "variant": variant, "cell": tag, "attack": k,
                            "auc_mean": round(_mean(a, k), 6) if _mean(a, k) != "" else "",
                            "auc_std": round(a.get(k, {}).get("std", float("nan")), 6)
                            if k in a else ""})

        for rows, kind in ((summ, "summary"), (det, "detailed")):
            path = os.path.join(C.RESULTS, f"{ds}_membership_{kind}.csv")
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            print(f"  {K.rel(path)}   {len(rows)} rows")
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
        tags = [t for t in attacks if parse_tag(t)[0] == ds]
        ctrl = next((t for t in tags if parse_tag(t)[1] == "control"), None)
        ctrl_v = _mean(attacks[ctrl]["aucs"], "best_threshold") if ctrl else 0.5

        fig, ax = plt.subplots(figsize=(8, 5))
        for method in METHOD_ORDER:
            if method == "control":
                continue
            cells = sorted([t for t in tags if parse_tag(t)[1] == method],
                           key=sort_key)
            if not cells:
                continue
            xs = [parse_tag(t)[2] + (("/" + parse_tag(t)[3]) if parse_tag(t)[3] else "")
                  for t in cells]
            ys = [_mean(attacks[t]["aucs"], "best_threshold") for t in cells]
            ax.plot(range(len(cells)), ys, marker="o", label=method)
            for i, t in enumerate(cells):
                ax.annotate(xs[i], (i, ys[i]), fontsize=6,
                            textcoords="offset points", xytext=(0, 5))
        ax.axhline(ctrl_v, ls=":", color="k", label="control (no reduction)")
        ax.axhline(0.5, ls="--", color="gray", label="no leakage (0.5)")
        ax.set_ylabel("membership AUC (best threshold attack)")
        ax.set_xlabel("reduction cell (per method, increasing ratio)")
        ax.set_title(f"Membership leakage vs reduction -- {ds}")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout()
        p = os.path.join(figdir, f"{ds}_membership_by_method.png")
        fig.savefig(p, dpi=150); plt.close(fig); out.append(p)
        print(f"  {K.rel(p)}")
    return out


def markdown(attacks, boxes, figs):
    L = ["# Node membership-inference benchmark -- results", "",
         "Does reducing the training graph lower a GNN's node membership leakage? "
         "Each dataset is split into members (V_in, 80%) and non-members (V_out, "
         "20%, nodes and edges removed). Every method reduces V_in only; a uniform "
         "3-layer GCN trains on the reduction and is served on the original graph. "
         "The 8-attack link-stealing battery does not apply here -- membership is a "
         "per-node question -- so the attacks are: unsupervised thresholds "
         "(confidence, entropy, loss, correctness) and a shadow MLP trained on the "
         "other dataset at the same cell. AUC 0.5 = no leakage.", ""]
    for ds in C.DATASETS:
        tags = sorted([t for t in attacks if parse_tag(t)[0] == ds], key=sort_key)
        L += [f"## {ds}", "",
              "| method | ratio | k | acc gap | best threshold AUC | shadow AUC |",
              "|---|---|---|---|---|---|"]
        for tag in tags:
            _, method, ratio, variant = parse_tag(tag)
            a = attacks[tag]["aucs"]
            bx = boxes.get(tag, {})
            rv = (ratio or "whole") + (f"/{variant}" if variant else "")
            L.append(f"| {method} | {rv} | {bx.get('k_train','')} | "
                     f"{bx.get('acc_gap',float('nan')):+.2f} | "
                     f"{_mean(a,'best_threshold'):.4f} | "
                     f"{_mean(a,'shadow_mlp'):.4f} |")
        L.append("")
    L += ["## Reading it", "",
          "- **control** is the reference leakage of a plain GNN on V_in.",
          "- A method/ratio with AUC near 0.5 hides membership; above the control "
          "means it leaks more, below means less.",
          "- GCond trains on synthetic nodes (no real node in training), so it is "
          "expected to leak least; KRON keeps real nodes, so it should track the "
          "control most closely.", ""]
    if figs:
        L += ["## Figures", ""]
        L += [f"![{os.path.basename(p)}](figures/{os.path.basename(p)})\n"
              for p in figs]
    path = os.path.join(C.RESULTS, "MEMBERSHIP_RESULTS.md")
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
    boxes = load_boxes()
    if not attacks:
        raise SystemExit("no results/attacks/*.json; run step04_attack.py first")
    print(f"loaded {len(attacks)} attacked cells, {len(boxes)} boxes\n\nCSVs:")
    csvs = write_csvs(attacks, boxes)
    figs = []
    if not args.no_figures:
        print("\nfigures:")
        figs = figures(attacks)
    print("\nwriteup:")
    md = markdown(attacks, boxes, figs)
    K.record(
        "step05_aggregate",
        summary=(f"Compiled {len(csvs)} CSVs, {len(figs)} figures and the writeup "
                 f"from {len(attacks)} attacked cells"),
        inputs=["results/attacks/*.json", "results/step03_boxes.json"],
        outputs=[K.rel(p) for p in csvs + figs + [md]],
        numbers={"cells": len(attacks)},
        notes="Membership AUC per method/ratio vs the no-reduction control.")


if __name__ == "__main__":
    main()
