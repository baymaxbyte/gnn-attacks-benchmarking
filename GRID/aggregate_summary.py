import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(HERE, "results", "grid_all_theta0.4.csv")
rows = list(csv.DictReader(open(path)))
D = {}
for r in rows:
    D[(r["dataset"], r["method"], r["setting"], r["attack"])] = (
        float(r["auc_base_wholeshadow"]), float(r["auc_grid"]), float(r["delta"]))

methods = ["fgc", "ugc", "gcond", "goren", "kron"]
sets = {"fgc": ["whole","0.3","0.1","0.05"], "ugc": ["whole","30","50","70"],
        "gcond": ["whole","0.25","0.5","1.0"],
        "goren": ["whole","0.3_bl","0.3_goren","0.5_bl","0.5_goren","0.7_bl","0.7_goren"],
        "kron": ["whole","0.3","0.5","0.7"]}
post_only = ["Attack-0","Attack-1","Attack-2","Attack-5"]
attr_topo = ["Attack-3","Attack-4","Attack-6","Attack-7"]

for ds in ["cora","citeseer"]:
    print(f"\n================ {ds}  (Attack-1 base->GRID; mean delta over attack groups) ============")
    print(f"{'method':6s} {'ratio':9s} {'A1_base':>8s} {'A1_grid':>8s} {'dA1':>7s} "
          f"{'mean_d(post)':>12s} {'mean_d(attr)':>12s}")
    for m in methods:
        for s in sets[m]:
            if (ds,m,s,"Attack-1") not in D: continue
            a1b,a1g,a1d = D[(ds,m,s,"Attack-1")]
            dp = [D[(ds,m,s,a)][2] for a in post_only if (ds,m,s,a) in D]
            da = [D[(ds,m,s,a)][2] for a in attr_topo if (ds,m,s,a) in D]
            mp = sum(dp)/len(dp) if dp else 0
            ma = sum(da)/len(da) if da else 0
            print(f"{m:6s} {s:9s} {a1b:8.3f} {a1g:8.3f} {a1d:+7.3f} {mp:+12.3f} {ma:+12.3f}")

print("\n================ biggest Attack-1 drops (top 10) ================")
items = [(k, v) for k,v in D.items() if k[3]=="Attack-1"]
items.sort(key=lambda kv: kv[1][2])
for (ds,m,s,a),(b,g,d) in items[:10]:
    print(f"  {ds:9s} {m:6s} {s:9s}  {b:.3f} -> {g:.3f}  ({d:+.3f})")
