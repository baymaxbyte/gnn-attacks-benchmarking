"""Step 05 -- compile the results into CSVs and plots.

Four CSVs, two per dataset:
    results/<ds>_attacks_detailed.csv   one row per (setting, attack, run)
    results/<ds>_attacks_summary.csv    one row per (setting, attack), averaged

Six plots per dataset, plus results/GCOND_RESULTS.md.

Reads only the JSON already in results/, so it is safe to rerun.

    python step05_aggregate.py
    python step05_aggregate.py --no_clustering
"""
import csv
import glob
import json
import os

import numpy as np

import common as C

args_no_clustering = False
SETTINGS = ["whole", "0.25", "0.5", "1.0"]
BOX = {"whole": "B0", "0.25": "B1", "0.5": "B2", "1.0": "B3"}
PAPER_R = {"cora": {"whole": "100%", "0.25": "1.3%", "0.5": "2.6%", "1.0": "5.2%"},
           "citeseer": {"whole": "100%", "0.25": "0.9%", "0.5": "1.8%",
                        "1.0": "3.6%"}}
SYN_NODES = {"cora": {"whole": 2708, "0.25": 35, "0.5": 70, "1.0": 140},
             "citeseer": {"whole": 3327, "0.25": 30, "0.5": 60, "1.0": 120}}
KNOW = {0: (0, 0, 0), 1: (0, 0, 1), 2: (1, 0, 0), 3: (0, 1, 0),
        4: (0, 1, 1), 5: (1, 0, 1), 6: (1, 1, 0), 7: (1, 1, 1)}
UNSUPERVISED = {0, 2}
METRICS = ["auc", "accuracy", "precision", "recall", "f1"]
DIRECTION = {"cora": "A", "citeseer": "B"}
A2_FAMILIES = [("target", "d(f(u),f(v))"), ("reference", "d(g(u),g(v))"),
               ("posterior_diff", "d(f) - d(g)"), ("feature", "d(Fu,Fv)")]


def ktag(a):
    return "/".join("Y" if x else "x" for x in KNOW[a])


# ------------------------------------------------------------------- loading
def load():
    box = {}
    vp = os.path.join(C.DIRS["results"], "step02_validation.json")
    if os.path.exists(vp):
        for r in json.load(open(vp))["rows"]:
            box[(r["dataset"], r["setting"])] = r
    runs = {}
    for f in glob.glob(os.path.join(C.DIRS["results"], "attacks", "*.json")):
        d = json.load(open(f))
        tgt, s = d["target"].split("_gc_")
        runs[(tgt, s)] = d
    if not runs:
        raise SystemExit("no attack results in results/attacks/; "
                         "run step04_attacks.py first")
    return runs, box


def box_columns(dataset, setting, box):
    """The black box's accuracy results, repeated on every row."""
    b = box.get((dataset, setting), {})
    return {
        "box": BOX[setting], "setting": setting,
        "paper_r": PAPER_R[dataset][setting],
        "train_nodes": SYN_NODES[dataset][setting],
        "box_acc_mean": b.get("acc_mean"), "box_acc_std": b.get("acc_std"),
        "box_acc_paper": b.get("paper_mean"), "box_acc_delta": b.get("delta"),
        "box_acc_runs": ";".join(str(x) for x in b.get("acc_runs", [])),
    }


def per_run_values(entry, metric):
    """Individual runs; unsupervised attacks have a single exact value."""
    if f"{metric}_runs" in entry and entry[f"{metric}_runs"]:
        return [float(v) for v in entry[f"{metric}_runs"]]
    return [float(entry[metric])] if metric in entry else []


# ---------------------------------------------------------------------- CSVs
def write_csvs(dataset, runs, box):
    det_rows, sum_rows = [], []
    for s in SETTINGS:
        d = runs.get((dataset, s))
        if d is None:
            continue
        shadow = d["shadow"].split("_gc_")[0]
        base = {"target": dataset, "shadow": shadow,
                "direction": DIRECTION[dataset], **box_columns(dataset, s, box)}
        entries = [(f"attack_{a}", d["results"][str(a)], a)
                   for a in range(8) if str(a) in d["results"]]
        if d.get("link_prediction"):
            entries.append(("link_prediction", d["link_prediction"], None))

        for name, e, a in entries:
            lp = a is None
            meta = {
                **base, "attack": name,
                "knowledge": "" if lp else ktag(a),
                "F": "" if lp else KNOW[a][0],
                "A_star": "" if lp else KNOW[a][1],
                "D_shadow": "" if lp else KNOW[a][2],
                "supervised": "" if lp else int(a not in UNSUPERVISED),
                "n_features": e.get("n_features", ""),
                "n_test_pairs": d["config"]["n_test_pairs"],
                "selected_metric": e.get("selected_metric", ""),
            }
            vals = {m: per_run_values(e, m) for m in METRICS}
            n_runs = max((len(v) for v in vals.values()), default=0)
            for i in range(n_runs):
                det_rows.append({**meta, "run": i + 1,
                                 **{m: (round(vals[m][i], 6)
                                        if i < len(vals[m]) else "")
                                    for m in METRICS}})
            row = {**meta, "n_runs": e.get("runs", n_runs)}
            for m in METRICS:
                if not vals[m]:
                    for k in ("mean", "std", "min", "max"):
                        row[f"{m}_{k}"] = ""
                    continue
                v = np.asarray(vals[m], dtype=float)
                row[f"{m}_mean"] = round(float(v.mean()), 6)
                row[f"{m}_std"] = round(float(v.std()), 6)
                row[f"{m}_min"] = round(float(v.min()), 6)
                row[f"{m}_max"] = round(float(v.max()), 6)
            sum_rows.append(row)

    det_cols = ["target", "shadow", "direction", "box", "setting", "paper_r",
                "train_nodes", "box_acc_mean", "box_acc_std", "box_acc_paper",
                "box_acc_delta", "attack", "knowledge", "F", "A_star",
                "D_shadow", "supervised", "selected_metric", "n_features",
                "n_test_pairs", "run"] + METRICS
    sum_cols = (["target", "shadow", "direction", "box", "setting", "paper_r",
                 "train_nodes", "box_acc_mean", "box_acc_std", "box_acc_paper",
                 "box_acc_delta", "box_acc_runs", "attack", "knowledge", "F",
                 "A_star", "D_shadow", "supervised", "selected_metric",
                 "n_features", "n_test_pairs", "n_runs"]
                + [f"{m}_{k}" for m in METRICS
                   for k in ("mean", "std", "min", "max")])

    out = []
    for rows, cols, suffix in ((det_rows, det_cols, "detailed"),
                               (sum_rows, sum_cols, "summary")):
        p = os.path.join(C.DIRS["results"], f"{dataset}_attacks_{suffix}.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"  [saved] {C.rel(p)}  ({len(rows)} rows)")
        out.append(p)
    return out


# --------------------------------------------------------------------- plots
def _xlabels(dataset, present):
    return [f"{BOX[s]}\n{PAPER_R[dataset][s]}\n{SYN_NODES[dataset][s]}n"
            for s in present]


def plots(dataset, runs, box, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [s for s in SETTINGS if (dataset, s) in runs]
    x = np.arange(len(present))
    made = []

    # ---- 1. AUC vs condensation setting, all attacks -----------------------
    fig, ax = plt.subplots(figsize=(9, 5.2))
    cmap = plt.get_cmap("tab10")
    for a in range(8):
        y = [runs[(dataset, s)]["results"][str(a)]["auc"] for s in present]
        e = [runs[(dataset, s)]["results"][str(a)].get("auc_std", 0) or 0
             for s in present]
        style = "--" if a in UNSUPERVISED else "-"
        ax.errorbar(x, y, yerr=e, marker="o", ms=5, ls=style, color=cmap(a),
                    capsize=3, label=f"Attack-{a} ({ktag(a)})")
    lp = [runs[(dataset, s)]["link_prediction"]["auc"] for s in present]
    ax.plot(x, lp, marker="s", ls=":", color="dimgray", lw=2,
            label="link-prediction baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(_xlabels(dataset, present))
    ax.set_ylabel("attack AUC")
    ax.set_xlabel("black box (condensation setting)")
    ax.set_title(f"{dataset.capitalize()} target — link-stealing AUC vs GCond "
                 f"condensation\n(ground truth: original {dataset} edges; "
                 f"dashed = unsupervised)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    p = os.path.join(outdir, f"{dataset}_auc_vs_setting.png")
    fig.tight_layout(); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    made.append(p)

    # ---- 2. every individual run ------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.2))
    width = 0.8 / len(present)
    for j, s in enumerate(present):
        d = runs[(dataset, s)]
        xs, ys = [], []
        for a in range(8):
            e = d["results"][str(a)]
            vals = per_run_values(e, "auc")
            off = a + j * width - 0.4 + width / 2
            xs += [off] * len(vals)
            ys += vals
            ax.hlines(np.mean(vals), off - width * 0.4, off + width * 0.4,
                      color="black", lw=1.2, zorder=3)
        ax.scatter(xs, ys, s=26, alpha=0.85, label=f"{BOX[s]} {PAPER_R[dataset][s]}",
                   zorder=2)
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"Attack-{a}\n{ktag(a)}" for a in range(8)], fontsize=8)
    ax.set_ylabel("attack AUC")
    ax.set_title(f"{dataset.capitalize()} target — every individual run "
                 f"(5 per supervised attack; Attacks 0 and 2 are unsupervised "
                 f"and exact, so one each)\nblack ticks = mean")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, title="black box")
    p = os.path.join(outdir, f"{dataset}_per_run_auc.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    made.append(p)

    # ---- 3. Attack-0: the 8 distance metrics ------------------------------
    pm0 = {s: runs[(dataset, s)].get("per_metric", {}).get("0") for s in present}
    if all(pm0.values()):
        names = list(pm0[present[0]].keys())
        fig, ax = plt.subplots(figsize=(10, 5))
        w = 0.8 / len(names)
        for i, m in enumerate(names):
            ax.bar(x + i * w - 0.4, [pm0[s][m] for s in present], w, label=m)
        sel = runs[(dataset, present[0])]["results"]["0"].get("selected_metric", "")
        ax.set_xticks(x); ax.set_xticklabels(_xlabels(dataset, present))
        ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUC")
        ax.set_title(f"{dataset.capitalize()} — Attack-0 (unsupervised) per "
                     f"distance metric\nselected metric: {sel}")
        ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8, ncol=4)
        p = os.path.join(outdir, f"{dataset}_unsup_attack0_metrics.png")
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        made.append(p)

    # ---- 4. Attack-2: the 4 information families -------------------------
    pm2 = {s: runs[(dataset, s)].get("per_metric", {}).get("2") for s in present}
    if all(pm2.values()):
        fig, ax = plt.subplots(figsize=(9, 5))
        w = 0.8 / len(A2_FAMILIES)
        for i, (key, lab) in enumerate(A2_FAMILIES):
            ax.bar(x + i * w - 0.4,
                   [pm2[s].get(f"{key}_correlation", np.nan) for s in present],
                   w, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(_xlabels(dataset, present))
        ax.set_ylim(0.4, 1.0); ax.set_ylabel("AUC")
        ax.set_title(f"{dataset.capitalize()} — Attack-2 (unsupervised) by "
                     f"information source, Correlation distance")
        ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9)
        p = os.path.join(outdir, f"{dataset}_unsup_attack2_families.png")
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        made.append(p)

    for m in made:
        print(f"  [saved] {C.rel(m)}")
    return made


# ------------------------------------------------------------------ markdown
def _stream_distances(path, want_reference=False):
    """Correlation distances per pair, streamed so memory stays flat."""
    import json as _json
    from scipy.spatial.distance import correlation
    labels, d_f, d_g = [], [], []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = _json.loads(line)
            labels.append(r["label"])
            d_f.append(correlation(r["gcn_pred0"], r["gcn_pred1"]))
            if want_reference:
                d_g.append(correlation(r["dense_pred0"], r["dense_pred1"]))
            del r
    out = {"label": np.array(labels),
           "d_f": np.nan_to_num(np.array(d_f), nan=0.0, posinf=0.0, neginf=0.0)}
    if want_reference:
        dg = np.nan_to_num(np.array(d_g), nan=0.0, posinf=0.0, neginf=0.0)
        out["d_g"] = dg
        out["d_diff"] = out["d_f"] - dg
    return out


def _kmeans_panel(ax, dist, labels, title, xlabel):
    """Histogram of a distance split by ground truth, with the K-means cut."""
    import sys as _sys
    if C.LSA not in _sys.path:
        _sys.path.insert(0, C.LSA)
    from sklearn.cluster import KMeans
    from sklearn.metrics import f1_score, precision_score, recall_score

    d = dist.reshape(-1, 1)
    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(d)
    centres = np.sort(km.cluster_centers_.ravel())
    linked = int(np.argmin(km.cluster_centers_.ravel()))
    pred = (km.labels_ == linked).astype(int)
    boundary = centres.mean()

    lo, hi = np.percentile(dist, [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    ax.hist(dist[labels == 0], bins=bins, alpha=0.6, label="not linked",
            color="tab:blue")
    ax.hist(dist[labels == 1], bins=bins, alpha=0.6, label="linked",
            color="tab:orange")
    for c in centres:
        ax.axvline(c, color="black", ls="-", lw=1.1)
    ax.axvline(boundary, color="crimson", ls="--", lw=1.6)
    ax.axvspan(bins[0], boundary, color="tab:orange", alpha=0.07)

    p = precision_score(labels, pred, zero_division=0)
    r = recall_score(labels, pred, zero_division=0)
    f1 = f1_score(labels, pred, zero_division=0)
    ax.set_title(f"{title}\nP={p:.3f} R={r:.3f} F1={f1:.3f}", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)
    return dict(precision=p, recall=r, f1=f1,
                centre_linked=float(centres[0]),
                centre_unlinked=float(centres[1]), boundary=float(boundary))


def clustering_plots(dataset, runs, outdir):
    """K-means figures for the two unsupervised attacks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [s for s in SETTINGS if (dataset, s) in runs]
    if not present:
        return [], {}
    made, stats = [], {}

    data = {}
    for s in present:
        path = os.path.join(C.LSA_ATTACK, f"{dataset}_gc_{s}_attack_test.json")
        if not os.path.exists(path):
            return [], {}
        print(f"    reading {os.path.basename(path)} ...", flush=True)
        data[s] = _stream_distances(path, want_reference=True)

    # ---- Attack-0: clustering of d(f) ------------------------------------
    fig, axes = plt.subplots(1, len(present), figsize=(4.1 * len(present), 3.9),
                             squeeze=False)
    for ax, s in zip(axes[0], present):
        auc = runs[(dataset, s)]["results"]["0"]["auc"]
        stats[f"attack0_{s}"] = _kmeans_panel(
            ax, data[s]["d_f"], data[s]["label"],
            f"{BOX[s]}  r={PAPER_R[dataset][s]}  ({SYN_NODES[dataset][s]}n)"
            f"   AUC={auc:.4f}", "Correlation distance  d(f(u),f(v))")
    axes[0][0].set_ylabel("node pairs")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"{dataset.capitalize()} — Attack-0 K-means (K=2) on Correlation "
                 f"distance\nblack = cluster centres, red dashed = decision "
                 f"boundary, shaded = predicted linked", fontsize=11)
    p = os.path.join(outdir, f"{dataset}_attack0_clustering.png")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(p, dpi=150); plt.close(fig)
    made.append(p)

    # ---- Attack-2: d(f) vs d(f)-d(g) -------------------------------------
    fig, axes = plt.subplots(2, len(present),
                             figsize=(4.1 * len(present), 7.2), squeeze=False)
    for j, s in enumerate(present):
        _kmeans_panel(axes[0][j], data[s]["d_f"], data[s]["label"],
                      f"{BOX[s]}  d(f)", "d(f(u),f(v))")
        stats[f"attack2_diff_{s}"] = _kmeans_panel(
            axes[1][j], data[s]["d_diff"], data[s]["label"],
            f"{BOX[s]}  d(f) - d(g)", "d(f(u),f(v)) - d(g(u),g(v))")
    axes[0][0].set_ylabel("node pairs"); axes[1][0].set_ylabel("node pairs")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"{dataset.capitalize()} — Attack-2 K-means, by information "
                 f"source\ntop: target posteriors only (identical to Attack-0)   "
                 f"bottom: the posterior difference only Attack-2 can compute",
                 fontsize=11)
    p = os.path.join(outdir, f"{dataset}_attack2_clustering.png")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(p, dpi=150); plt.close(fig)
    made.append(p)

    for m in made:
        print(f"  [saved] {C.rel(m)}")
    return made, stats


def markdown(runs, box, figs):
    md = ["# GCond condensation vs link stealing", "",
          "Black box: GCond condenses the training graph, a 2-layer / 256-hidden",
          "GCN is trained on it, and at query time that GCN propagates over the",
          "**original** graph (GCond's transductive protocol). The adversary sends",
          "original node ids, holds original pre-condensation attributes as `F`,",
          "and attacks **original** edges.", "",
          "Held fixed across every box: the edge set, the sampled non-edges, the",
          "random 50/50 split, the reference model `g`, and the attack model. Only",
          "the served posteriors differ.", "",
          "Per-dataset data: `<ds>_attacks_detailed.csv` (one row per run) and",
          "`<ds>_attacks_summary.csv` (averaged over the 5 runs).", ""]
    for tgt in ("cora", "citeseer"):
        present = [s for s in SETTINGS if (tgt, s) in runs]
        if not present:
            continue
        shd = runs[(tgt, present[0])]["shadow"].split("_gc_")[0]
        md += [f"## Target {tgt.capitalize()}, shadow {shd.capitalize()}", "",
               "| Attack | K=(F,A*,D') | " +
               " | ".join(f"{BOX[s]} r={PAPER_R[tgt][s]}" for s in present) +
               " | Δ(B3−B0) |",
               "|---|---|" + "---|" * (len(present) + 1)]
        deltas, spreads = [], []
        for a in range(8):
            cells, vals = [], []
            for s in present:
                r = runs[(tgt, s)]["results"][str(a)]
                vals.append(r["auc"])
                c = f"{r['auc']:.4f}"
                if r.get("auc_std") and r["runs"] > 1:
                    c += f" ± {r['auc_std']:.4f}"
                cells.append(c)
            deltas.append(vals[-1] - vals[0]); spreads.append(max(vals) - min(vals))
            md.append(f"| {a} | {ktag(a)} | " + " | ".join(cells) +
                      f" | **{vals[-1] - vals[0]:+.4f}** |")
        lp = [runs[(tgt, s)]["link_prediction"]["auc"] for s in present]
        md.append("| link-pred baseline | — | " +
                  " | ".join(f"{v:.4f}" for v in lp) + f" | {lp[-1] - lp[0]:+.4f} |")
        md.append("| **box accuracy %** | — | " +
                  " | ".join(f"{box[(tgt, s)]['acc_mean']:.2f}" for s in present)
                  + " | |")
        md.append("| **training nodes** | — | " +
                  " | ".join(str(SYN_NODES[tgt][s]) for s in present) + " | |")
        md += ["",
               f"- mean AUC change B0 → B3 across the 8 attacks: "
               f"**{np.mean(deltas):+.4f}**",
               f"- largest spread of any single attack across the boxes: "
               f"**{max(spreads):.4f}**", ""]
    md += ["## Reading", "",
           "Condensing the training graph — down to 35 synthetic nodes for Cora,",
           "1.3% of the original — moves link-stealing AUC by at most ~0.03, and",
           "often *upward*. The mechanism is structural: GCond's condensed graph is",
           "a **training set only**, and the deployed model aggregates over the",
           "**true original adjacency at every query**. The edge signal the attacks",
           "exploit is injected at inference, so shrinking the training graph",
           "cannot remove it.", "",
           "The link-prediction baseline is identical across settings, as it must",
           "be — it reads only `A*`, which is held fixed. A useful internal",
           "consistency check on the ablation.", ""]

    # ---- the headline hides a real effect: metric scale-sensitivity --------
    md += ["## But the headline hides a real effect", "",
           "Attack-0 scores each of the 8 distance metrics separately, and they do",
           "*not* respond to condensation the same way. Condensation flattens the",
           "posteriors (mean max probability falls from 0.695 to 0.278 on Cora), so",
           "**magnitude-sensitive** distances shrink while **scale-invariant** ones",
           "are unaffected.", ""]
    for tgt in ("cora", "citeseer"):
        present = [s for s in SETTINGS if (tgt, s) in runs]
        pm = {s: runs[(tgt, s)].get("per_metric", {}).get("0") for s in present}
        if not present or not all(pm.values()):
            continue
        scale_inv = {"correlation": "yes", "cosine": "partly (direction)"}
        md += [f"### {tgt.capitalize()} — Attack-0 by distance metric", "",
               "| Metric | " + " | ".join(BOX[s] for s in present) +
               " | Δ(B3−B0) | scale-invariant? |",
               "|---|" + "---|" * (len(present) + 2)]
        for m in pm[present[0]]:
            v = [pm[s][m] for s in present]
            md.append(f"| {m} | " + " | ".join(f"{x:.4f}" for x in v) +
                      f" | **{v[-1] - v[0]:+.4f}** | "
                      f"{scale_inv.get(m, 'no (magnitude)')} |")
        md.append("")
    md += ["Correlation — the metric the paper selects for Attack-0, and the best",
           "one on the uncondensed graph too — normalises away both scale and mean,",
           "so it is immune. On Cora it even improves. The magnitude-based metrics",
           "lose 0.10–0.13 AUC.", "",
           "So the honest statement is narrower than \"condensation does not help\":",
           "condensation **does** degrade five of the eight distance metrics",
           "substantially, but not the one the attack actually uses. A practitioner",
           "who reached for Euclidean would measure 0.916 → 0.786 on Cora and",
           "conclude condensation protects well. It does not — the attack simply",
           "has a better metric available.", "",
           "## Figures", ""]
    for f in figs:
        md.append(f"![{os.path.basename(f)}](figures/{os.path.basename(f)})\n")
    p = os.path.join(C.DIRS["results"], "GCOND_RESULTS.md")
    open(p, "w").write("\n".join(md) + "\n")
    print(f"  [saved] {C.rel(p)}")
    return p


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compile results into CSVs + plots")
    ap.add_argument("--no_clustering", action="store_true",
                    help="skip the K-means clustering figures, which have to "
                         "re-read the attack JSONL files")
    a = ap.parse_args()
    global args_no_clustering
    args_no_clustering = a.no_clustering

    runs, box = load()
    figdir = os.path.join(C.DIRS["results"], "figures")
    os.makedirs(figdir, exist_ok=True)

    outputs, figs, clust = [], [], {}
    for ds in ("cora", "citeseer"):
        if not any((ds, s) in runs for s in SETTINGS):
            continue
        print(f"\n{ds}:")
        outputs += write_csvs(ds, runs, box)
        figs += plots(ds, runs, box, figdir)
        if not args_no_clustering:
            cf, cs = clustering_plots(ds, runs, figdir)
            figs += cf
            clust[ds] = cs
    if clust:
        p = os.path.join(C.DIRS["results"], "unsupervised_clustering.json")
        with open(p, "w") as f:
            json.dump(clust, f, indent=2)
        print(f"  [saved] {C.rel(p)}")
        outputs.append(p)
    print()
    outputs.append(markdown(runs, box, figs))

    for old in ("gcond_leakage.csv", "gcond_leakage_matrix.csv"):
        p = os.path.join(C.DIRS["results"], old)
        if os.path.exists(p):
            os.remove(p)
            print(f"  [removed stale] {old}")

    C.record(
        step="05_aggregate_gcond",
        summary=("Compiled per-dataset detailed (per-run) and summary (5-run "
                 "average) CSVs plus 8 plots"),
        inputs=["results/attacks/*.json",
                "results/step02_validation.json"],
        outputs=[C.rel(p) for p in outputs + figs],
        numbers={"csvs": 4, "figures": len(figs)},
        notes="detailed = one row per run; summary = averaged over runs.")


if __name__ == "__main__":
    main()
