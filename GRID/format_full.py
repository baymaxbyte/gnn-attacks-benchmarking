import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "results", "grid_all_theta0.4.csv"))))
D = {}
for r in rows:
    D[(r["dataset"], r["method"], r["setting"], r["attack"])] = (
        float(r["auc_base_wholeshadow"]), float(r["auc_grid"]))

methods = ["fgc", "ugc", "gcond", "goren", "kron"]
sets = {"fgc": ["whole","0.3","0.1","0.05"], "ugc": ["whole","30","50","70"],
        "gcond": ["whole","0.25","0.5","1.0"],
        "goren": ["whole","0.3_bl","0.3_goren","0.5_bl","0.5_goren","0.7_bl","0.7_goren"],
        "kron": ["whole","0.3","0.5","0.7"]}
atks = [f"Attack-{i}" for i in range(8)] + ["link_pred"]
cols = [f"A{i}" for i in range(8)] + ["link_pred"]

for ds in ["cora", "citeseer"]:
    for m in methods:
        print(f"\n#### {ds} — {m.upper()}")
        print("| Ratio | " + " | ".join(cols) + " |")
        print("|" + "---|" * (len(cols) + 1))
        for s in sets[m]:
            cells = []
            for a in atks:
                if (ds, m, s, a) in D:
                    b, g = D[(ds, m, s, a)]
                    cells.append(f"{b:.3f}→{g:.3f}")
                else:
                    cells.append("—")
            print(f"| {s} | " + " | ".join(cells) + " |")
