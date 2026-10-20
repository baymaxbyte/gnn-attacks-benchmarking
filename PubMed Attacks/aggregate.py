"""Collect PubMed results from the sibling benchmarks into ./results/.

LSA: runs lsa/aggregate_shadows.py (dataset-agnostic -> emits
     pubmed_attacks_summary_wholeshadow.csv per method), then copies those here and
     builds one combined CSV across methods (method,setting,attack,auc_mean,auc_std,...).
Membership: copies pubmed_membership_{summary,detailed,shadow_compare}.csv here.

Safe to run repeatedly; only reads/copies. Missing pieces (from a partial run) are
reported, not fatal.

    python aggregate.py
"""
import csv
import glob
import os
import shutil
import subprocess

import config as C

OUT = C.RESULTS
OUT_LSA = os.path.join(OUT, "lsa")
OUT_MIA = os.path.join(OUT, "membership")


def _copy(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  copied {os.path.relpath(src, C.TEMP)} -> {os.path.relpath(dst, C.HERE)}")
        return True
    print(f"  [missing] {os.path.relpath(src, C.TEMP)}")
    return False


def collect_lsa():
    print("\n[LSA] compiling whole-shadow summaries via aggregate_shadows.py ...")
    agg = os.path.join(C.LSA_ROOT, "aggregate_shadows.py")
    if os.path.exists(agg):
        subprocess.run([C.PYTHON, "aggregate_shadows.py"], cwd=C.LSA_ROOT)
    os.makedirs(OUT_LSA, exist_ok=True)

    combined, header = [], None
    for method, spec in C.LSA_SPECS.items():
        src = os.path.join(C.LSA_ROOT, spec["dir"], "results",
                           "pubmed_attacks_summary_wholeshadow.csv")
        dst = os.path.join(OUT_LSA, f"{method}_pubmed_wholeshadow.csv")
        if not _copy(src, dst):
            continue
        with open(src) as f:
            rows = list(csv.DictReader(f))
        if rows and header is None:
            header = ["method"] + list(rows[0].keys())
        for r in rows:
            combined.append(dict(method=method, **r))
    if combined:
        path = os.path.join(OUT, "lsa_pubmed_wholeshadow_ALL.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(combined)
        print(f"  -> combined {len(combined)} rows across methods: "
              f"{os.path.relpath(path, C.HERE)}")
    else:
        print("  [LSA] no pubmed whole-shadow summaries found yet")


def collect_membership():
    print("\n[membership] collecting pubmed CSVs ...")
    res = os.path.join(C.MEMBERSHIP_ROOT, "results")
    for name in ("pubmed_membership_summary.csv",
                 "pubmed_membership_detailed.csv",
                 "pubmed_membership_shadow_compare.csv"):
        _copy(os.path.join(res, name), os.path.join(OUT_MIA, name))


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 70)
    print("Collecting PubMed results into", os.path.relpath(OUT, C.HERE))
    collect_lsa()
    collect_membership()
    print("\naggregate DONE.  Key files:")
    print("  results/lsa_pubmed_wholeshadow_ALL.csv         (LSA, all methods)")
    print("  results/membership/pubmed_membership_summary.csv")
    print("  results/membership/pubmed_membership_shadow_compare.csv (whole-cora shadow)")


if __name__ == "__main__":
    main()
