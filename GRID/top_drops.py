import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "results", "grid_all_theta0.4.csv"))))
rows.sort(key=lambda r: float(r["delta"]))
print(f"{'#':>2s}  {'dataset':9s} {'method':6s} {'ratio':9s} {'attack':9s} "
      f"{'base':>6s} {'grid':>6s} {'drop':>7s}")
for i, r in enumerate(rows[:20], 1):
    b, g, d = float(r["auc_base_wholeshadow"]), float(r["auc_grid"]), float(r["delta"])
    print(f"{i:>2d}  {r['dataset']:9s} {r['method']:6s} {r['setting']:9s} {r['attack']:9s} "
          f"{b:6.3f} {g:6.3f} {d:+7.3f}")
