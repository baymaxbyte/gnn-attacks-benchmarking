#!/usr/bin/env python3
"""Consolidate exp1 on the heterophilous datasets (squirrel/chameleon).
Reads results/attacks/exp1coarsenhet_attack_auc.csv (dataset,method_ratio_eps,
blackbox_acc,attack,auc), writes exp1hetero_consolidated_auc.csv and prints a
per-(dataset,method,eps) summary: mean Attack-0/1/6 over ratios. Mirrors
consolidate_exp1.py but for the single no-GRID hetero CSV."""
import csv, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "attacks")
KEPT = {"mcc", "fgc"}; REMOVED = {"kron", "ugc"}
meaning = lambda m: "kept" if m in KEPT else ("removed" if m in REMOVED else "train_frac")

src = os.path.join(RES, "exp1coarsenhet_attack_auc.csv")
out = []
for r in csv.DictReader(open(src)):
    m, ra, ep = r["method_ratio_eps"].split("_"); ep = ep[1:]
    out.append((r["dataset"], m, ra, meaning(m), ep, r["blackbox_acc"], r["attack"], r["auc"]))

csvp = os.path.join(RES, "exp1hetero_consolidated_auc.csv")
with open(csvp, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "method", "ratio", "ratio_meaning", "eps", "blackbox_acc", "attack", "auc"])
    for row in out:
        w.writerow(row)
methods = sorted(set(r[1] for r in out))
print(f"wrote {csvp}: {len(out)} rows, methods={methods}")

S = collections.defaultdict(lambda: collections.defaultdict(list))
for ds, m, ra, cv, ep, acc, at, au in out:
    if at in ("Attack-0", "Attack-1", "Attack-6"):
        S[(ds, m, ep)][at].append(float(au))
    if at == "Attack-0":
        S[(ds, m, ep)]["acc"].append(float(acc))
    if at == "link_pred":
        S[(ds, m, ep)]["lp"].append(float(au))
print(f"\n{'dataset':9} {'method':5} {'eps':4} {'acc%':>6} {'A0':>6} {'A1':>6} {'A6':>6} {'lp':>6}  (mean over ratios)")
for (ds, m, ep) in sorted(S):
    d = S[(ds, m, ep)]
    mean = lambda k: sum(d[k]) / len(d[k]) if d[k] else float("nan")
    print(f"{ds:9} {m:5} {ep:4} {mean('acc')*100:6.1f} {mean('Attack-0'):6.3f} {mean('Attack-1'):6.3f} {mean('Attack-6'):6.3f} {mean('lp'):6.3f}")
