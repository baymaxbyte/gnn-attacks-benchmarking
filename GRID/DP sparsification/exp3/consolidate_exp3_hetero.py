#!/usr/bin/env python3
"""Consolidate exp3 (KNN-graph coarsening) on squirrel/chameleon.
Reads results/attacks/exp3coarsenhet_attack_auc.csv (dataset,method_ratio,
blackbox_acc,attack,auc); baseline = exp3knnhet_attack_auc.csv (plain KNN-GCN).
Writes exp3hetero_consolidated_auc.csv and prints a per-(dataset,method) summary
vs the plain KNN-GCN baseline. Mirrors consolidate_exp3coarsen.py."""
import csv, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "attacks")
KEPT = {"mcc", "fgc"}; REMOVED = {"kron", "ugc"}
meaning = lambda m: "kept" if m in KEPT else ("removed" if m in REMOVED else "train_frac")

rows = list(csv.DictReader(open(os.path.join(RES, "exp3coarsenhet_attack_auc.csv"))))
out = []
for r in rows:
    m, ra = r["method_ratio"].split("_", 1)
    out.append((r["dataset"], m, ra, meaning(m), r["blackbox_acc"], r["attack"], r["auc"]))

csvp = os.path.join(RES, "exp3hetero_consolidated_auc.csv")
with open(csvp, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "method", "ratio", "ratio_meaning", "blackbox_acc", "attack", "auc"])
    for row in out:
        w.writerow(row)
methods = sorted(set(r[1] for r in out))
print(f"wrote {csvp}: {len(out)} rows, methods={methods}")

# plain KNN-GCN baseline from exp3knnhet_attack_auc.csv (grid=True -> auc_no_defense)
base = collections.defaultdict(dict)
bp = os.path.join(RES, "exp3knnhet_attack_auc.csv")
if os.path.exists(bp):
    for r in csv.DictReader(open(bp)):
        col = "auc_no_defense" if "auc_no_defense" in r else "auc"
        base[r["dataset"]][r["attack"]] = float(r[col])
        if r["attack"] == "Attack-0":
            base[r["dataset"]]["acc"] = float(r["blackbox_acc"])

S = collections.defaultdict(lambda: collections.defaultdict(list))
for ds, m, ra, cv, acc, at, au in out:
    if at in ("Attack-0", "Attack-1", "Attack-6"):
        S[(ds, m)][at].append(float(au))
    if at == "Attack-0":
        S[(ds, m)]["acc"].append(float(acc))
    if at == "link_pred":
        S[(ds, m)]["lp"].append(float(au))

print(f"\n{'dataset':9} {'method':8} {'acc%':>6} {'A0':>6} {'A1':>6} {'A6':>6} {'lp':>6}   (mean over ratios)")
for ds in ("squirrel", "chameleon"):
    b = base.get(ds, {})
    if b:
        print(f"{ds:9} {'KNN-GCN':8} {b.get('acc',0)*100:6.1f} "
              f"{b.get('Attack-0',0):6.3f} {b.get('Attack-1',0):6.3f} {b.get('Attack-6',0):6.3f} {b.get('link_pred',0):6.3f}   <- baseline (no coarsening)")
    for (d, m) in sorted(k for k in S if k[0] == ds):
        d2 = S[(d, m)]
        mean = lambda k: sum(d2[k]) / len(d2[k]) if d2[k] else float("nan")
        print(f"{d:9} {m:8} {mean('acc')*100:6.1f} {mean('Attack-0'):6.3f} {mean('Attack-1'):6.3f} {mean('Attack-6'):6.3f} {mean('lp'):6.3f}")
    print()
