import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "results", "grid_all_theta0.4.csv"))))
rows.sort(key=lambda r: float(r["delta"]))
print("biggest 5 AUC drops (most negative delta):")
for r in rows[:5]:
    b = float(r["auc_base_wholeshadow"]); g = float(r["auc_grid"]); d = float(r["delta"])
    print(f"  {r['dataset']:9s} {r['method']:6s} {r['setting']:9s} {r['attack']:9s}  "
          f"{b:.3f} -> {g:.3f}  (delta {d:+.3f})")
