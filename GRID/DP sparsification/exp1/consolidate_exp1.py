#!/usr/bin/env python3
"""Merge the 6 coarsening methods' exp1 attack CSVs into one consolidated table.
Columns: dataset, method, ratio, ratio_meaning, eps, blackbox_acc, attack, auc.
Ratio conventions differ: FGC/MCC = fraction KEPT; KRON/UGC/GOREN = fraction REMOVED;
GCond = reduction_rate (fraction of the train set)."""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "attacks")
KEPT = {"mcc", "fgc"}; REMOVED = {"kron", "ugc", "goren"}
meaning = lambda m: "kept" if m in KEPT else ("removed" if m in REMOVED else "train_frac")

out = []
# 4 in-process methods: setting = "<method>_<ratio>_e<eps>"
for r in csv.DictReader(open(os.path.join(RES, "exp1coarsen_attack_auc.csv"))):
    m, ra, ep = r["method_ratio_eps"].split("_"); ep = ep[1:]
    out.append((r["dataset"], m, ra, meaning(m), ep, r["blackbox_acc"], r["attack"], r["auc"]))
# GOREN / GCond: setting = "<ratio>_e<eps>"
for fn, m, col in [("exp1goren_attack_auc.csv", "goren", "ratio_eps"),
                   ("exp1gcond_attack_auc.csv", "gcond", "rrate_eps")]:
    for r in csv.DictReader(open(os.path.join(RES, fn))):
        ra, ep = r[col].split("_"); ep = ep[1:]
        out.append((r["dataset"], m, ra, meaning(m), ep, r["blackbox_acc"], r["attack"], r["auc"]))

csvp = os.path.join(RES, "exp1_consolidated_auc.csv")
with open(csvp, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "method", "ratio", "ratio_meaning", "eps", "blackbox_acc", "attack", "auc"])
    for row in out:
        w.writerow(row)
methods = sorted(set(r[1] for r in out))
print(f"wrote {csvp}: {len(out)} rows, methods={methods}")

# compact summary: mean Attack-0/1/6 across ratios, per method x eps x dataset
import collections
S = collections.defaultdict(lambda: collections.defaultdict(list))
for ds, m, ra, cv, ep, acc, at, au in out:
    if at in ("Attack-0", "Attack-1", "Attack-6"):
        S[(ds, m, ep)][at].append(float(au))
    if at == "Attack-0":
        S[(ds, m, ep)]["acc"].append(float(acc))
print(f"\n{'dataset':8} {'method':6} {'eps':4} {'acc%':>6} {'A0':>6} {'A1':>6} {'A6':>6}  (mean over ratios)")
for (ds, m, ep) in sorted(S):
    d = S[(ds, m, ep)]
    mean = lambda k: sum(d[k]) / len(d[k])
    print(f"{ds:8} {m:6} {ep:4} {mean('acc')*100:6.1f} {mean('Attack-0'):6.3f} {mean('Attack-1'):6.3f} {mean('Attack-6'):6.3f}")
