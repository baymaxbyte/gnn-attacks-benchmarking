#!/usr/bin/env python3
"""Consolidate the 6-method exp3-coarsen attack results into one table + summary.

exp3-coarsen = build feature-KNN(k=10) graph (replacing real adjacency), then
coarsen/condense it with all 6 methods at native ratios, serve a black-box GCN on
the KNN graph, and run He et al. attacks (NO GRID). Ground truth = ORIGINAL edges.

Input : results/attacks/exp3coarsen_attack_auc.csv  (dataset,method_ratio,blackbox_acc,attack,auc)
Output: results/attacks/exp3coarsen_consolidated_auc.csv
        + compact per-method summary vs the plain-KNN-GCN baseline (exp3knn).

Ratio conventions differ by method:
  mcc/fgc = fraction KEPT ; kron/ugc/goren = fraction REMOVED ; gcond = train_frac.
"""
import csv, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "attacks")
KEPT = {"mcc", "fgc"}; REMOVED = {"kron", "ugc", "goren"}
meaning = lambda m: "kept" if m in KEPT else ("removed" if m in REMOVED else "train_frac")

rows = list(csv.DictReader(open(os.path.join(RES, "exp3coarsen_attack_auc.csv"))))
out = []
for r in rows:
    mr = r["method_ratio"]
    m, ra = mr.split("_", 1)
    out.append((r["dataset"], m, ra, meaning(m), r["blackbox_acc"], r["attack"], r["auc"]))

csvp = os.path.join(RES, "exp3coarsen_consolidated_auc.csv")
with open(csvp, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "method", "ratio", "ratio_meaning", "blackbox_acc", "attack", "auc"])
    for row in out:
        w.writerow(row)
methods = sorted(set(r[1] for r in out))
print(f"wrote {csvp}: {len(out)} rows, methods={methods}")

# plain-KNN-GCN baseline (no coarsening), from exp3knn_attack_auc.csv -> auc_no_defense
base = collections.defaultdict(dict)
bp = os.path.join(RES, "exp3knn_attack_auc.csv")
if os.path.exists(bp):
    for r in csv.DictReader(open(bp)):
        base[r["dataset"]][r["attack"]] = float(r["auc_no_defense"])
        if r["attack"] == "Attack-0":
            base[r["dataset"]]["acc"] = float(r["blackbox_acc"])

# compact summary: mean Attack-0/1/6 across ratios, per dataset x method
S = collections.defaultdict(lambda: collections.defaultdict(list))
for ds, m, ra, cv, acc, at, au in out:
    if at in ("Attack-0", "Attack-1", "Attack-6"):
        S[(ds, m)][at].append(float(au))
    if at == "Attack-0":
        S[(ds, m)]["acc"].append(float(acc))

print(f"\n{'dataset':9} {'method':7} {'acc%':>6} {'A0':>6} {'A1':>6} {'A6':>6}   (mean over ratios)")
for ds in ("cora", "citeseer"):
    b = base.get(ds, {})
    if b:
        print(f"{ds:9} {'KNN-GCN':7} {b.get('acc',0)*100:6.1f} "
              f"{b.get('Attack-0',0):6.3f} {b.get('Attack-1',0):6.3f} {b.get('Attack-6',0):6.3f}   <- baseline (no coarsening)")
    for (d, m) in sorted(k for k in S if k[0] == ds):
        d2 = S[(d, m)]
        mean = lambda k: sum(d2[k]) / len(d2[k])
        print(f"{d:9} {m:7} {mean('acc')*100:6.1f} {mean('Attack-0'):6.3f} {mean('Attack-1'):6.3f} {mean('Attack-6'):6.3f}")
    print()
