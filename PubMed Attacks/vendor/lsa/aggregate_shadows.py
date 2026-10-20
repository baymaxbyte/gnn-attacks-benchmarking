"""Aggregate the whole-graph-shadow attack results into step05-style CSVs, and a
coarsened-vs-whole comparison. Non-destructive: the shipped coarsened
<ds>_attacks_summary.csv files are NOT overwritten.

Per benchmark it writes into results/:
  <ds>_attacks_summary_wholeshadow.csv   one row per (method,setting,attack); shadow col
                                         shows <shadow>_<set_tag>_whole for attacks 1/4/5/7
                                         and '-' for the shadow-free attacks (0/2/3/6/link)
  <ds>_attacks_detailed_wholeshadow.csv  one row per run
  <ds>_shadow_compare.csv                attacks 1/4/5/7 only: coarsened_auc vs
                                         wholeshadow_auc side by side (the effect of the
                                         shadow choice)

    python aggregate_shadows.py
"""
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SPECS = {
    "fgc":   dict(dir="FGC Benchmarking",   set_tag="fgc"),
    "gcond": dict(dir="GCOND Benchmarking", set_tag="gc"),
    "goren": dict(dir="GOREN Benchmarking", set_tag="goren"),
    "kron":  dict(dir="KRON Benchmarking",  set_tag="kron"),
    "ugc":   dict(dir="UGC Benchmarking",   set_tag="ugc"),
}
KNOW = {0: ("x", "x", "x"), 1: ("x", "x", "Y"), 2: ("Y", "x", "x"), 3: ("x", "Y", "x"),
        4: ("x", "Y", "Y"), 5: ("Y", "x", "Y"), 6: ("Y", "Y", "x"), 7: ("Y", "Y", "Y")}
USES_SHADOW = {1, 4, 5, 7}
METRICS = ("accuracy", "precision", "recall", "f1")


def _num(x):
    return round(float(x), 6) if x is not None and x != "" else ""


def _parse(target_field, set_tag):
    """'cora_fgc_0.3' -> ('cora','0.3'); handles goren '0.3_bl'."""
    ds, _, setting = target_field.partition(f"_{set_tag}_")
    return ds, setting


def _load(attacks_dir, set_tag):
    """{(ds, setting): {attack_name: rec}} from a results dir of run_attack JSONs.
    rec carries auc/auc_std/auc_runs + accuracy/precision/recall/f1 (+std), shadow, n_test."""
    cells = {}
    for p in glob.glob(os.path.join(attacks_dir, "*.json")):
        d = json.load(open(p))
        ds, setting = _parse(d["target"], set_tag)
        shadow = d.get("shadow") or ""
        n_test = d.get("config", {}).get("n_test_pairs", "")
        bucket = cells.setdefault((ds, setting), {})
        for a in range(8):
            r = d.get("results", {}).get(str(a))
            if r:
                bucket[f"attack-{a}"] = dict(r, _shadow=shadow, _ntest=n_test, _a=a)
        lp = d.get("link_prediction")
        if lp:
            bucket["link_pred"] = dict(lp, _shadow=shadow, _ntest=n_test, _a=None)
    return cells


def _rows(cells, set_tag):
    """(summary_rows, detailed_rows) for one variant, ordered control/whole first."""
    order = ["link_pred"] + [f"attack-{i}" for i in range(8)]
    settings = sorted(cells, key=lambda ks: (ks[0], ks[1] != "whole", ks[1]))
    whole_ref = {}                      # (ds, attack) -> auc of the whole setting
    for (ds, setting), bucket in cells.items():
        if setting == "whole":
            for name, rec in bucket.items():
                whole_ref[(ds, name)] = rec.get("auc")
    summ, det = [], []
    for ds, setting in settings:
        bucket = cells[(ds, setting)]
        for name in [f"attack-{i}" for i in range(8)] + ["link_pred"]:
            rec = bucket.get(name)
            if not rec:
                continue
            a = rec["_a"]
            uses = (a in USES_SHADOW)
            know = KNOW.get(a, ("-", "Y", "-")) if a is not None else ("-", "-", "-")
            runs_list = rec.get("auc_runs") or ([rec["auc"]] if "auc" in rec else [])
            ref = whole_ref.get((ds, name))
            row = {
                "dataset": ds, "setting": setting, "attack": name,
                "knowledge": "/".join(know),
                "F": know[0], "A_star": know[1], "D_shadow": know[2],
                "supervised": (a not in (0, 2)) if a is not None else True,
                "shadow": (rec["_shadow"] if uses else "-"),
                "n_runs": rec.get("runs", ""), "n_test_pairs": rec.get("_ntest", ""),
                "auc_mean": _num(rec.get("auc")),
                "auc_std": _num(rec.get("auc_std")),
                "auc_min": _num(min(runs_list)) if runs_list else "",
                "auc_max": _num(max(runs_list)) if runs_list else "",
            }
            for mm in METRICS:
                row[f"{mm}_mean"] = _num(rec.get(mm))
                row[f"{mm}_std"] = _num(rec.get(f"{mm}_std"))
            row["auc_delta_vs_whole"] = (_num(rec["auc"] - ref)
                                         if ref not in (None, "") and "auc" in rec else "")
            summ.append(row)
            for i, v in enumerate(runs_list):
                det.append({"dataset": ds, "setting": setting, "attack": name,
                            "knowledge": row["knowledge"], "shadow": row["shadow"],
                            "run": i + 1, "auc": _num(v)})
    return summ, det


def _write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    for m, spec in SPECS.items():
        bench = os.path.join(HERE, spec["dir"])
        res = os.path.join(bench, "results")
        ws_dir = os.path.join(res, "attacks_wholeshadow")
        co_dir = os.path.join(res, "attacks")
        if not os.path.isdir(ws_dir) or not glob.glob(os.path.join(ws_dir, "*.json")):
            print(f"{m}: no whole-shadow results yet, skipping")
            continue
        ws_cells = _load(ws_dir, spec["set_tag"])
        co_cells = _load(co_dir, spec["set_tag"]) if os.path.isdir(co_dir) else {}

        ws_sum, ws_det = _rows(ws_cells, spec["set_tag"])
        for ds in sorted({r["dataset"] for r in ws_sum}):
            _write(os.path.join(res, f"{ds}_attacks_summary_wholeshadow.csv"),
                   [r for r in ws_sum if r["dataset"] == ds])
            _write(os.path.join(res, f"{ds}_attacks_detailed_wholeshadow.csv"),
                   [r for r in ws_det if r["dataset"] == ds])

        # coarsened-vs-whole comparison for the shadow-dependent attacks only
        co_sum, _ = _rows(co_cells, spec["set_tag"])
        co_idx = {(r["dataset"], r["setting"], r["attack"]): r for r in co_sum}
        cmp_rows = []
        for r in ws_sum:
            if r["attack"] not in (f"attack-{i}" for i in USES_SHADOW):
                continue
            co = co_idx.get((r["dataset"], r["setting"], r["attack"]))
            cmp_rows.append({
                "dataset": r["dataset"], "setting": r["setting"], "attack": r["attack"],
                "knowledge": r["knowledge"],
                "auc_coarsened_shadow": co["auc_mean"] if co else "",
                "auc_whole_shadow": r["auc_mean"],
                "delta_whole_minus_coarsened":
                    (_num(r["auc_mean"] - co["auc_mean"])
                     if co and co["auc_mean"] != "" and r["auc_mean"] != "" else ""),
            })
        for ds in sorted({r["dataset"] for r in cmp_rows}):
            _write(os.path.join(res, f"{ds}_shadow_compare.csv"),
                   [r for r in cmp_rows if r["dataset"] == ds])
        print(f"{m}: wrote wholeshadow summaries + shadow_compare "
              f"({len(ws_sum)} rows) into {spec['dir']}/results/")


if __name__ == "__main__":
    main()
