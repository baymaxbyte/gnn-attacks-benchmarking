import csv, os, openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = "/Users/chitraansh.pandey/Downloads/attack_results_summary.xlsx"
CSV = os.path.join(HERE, "results", "grid_all_theta0.4.csv")

# box accuracy (target model test acc) from the Link Stealing sheet
wb = openpyxl.load_workbook(XLSX, data_only=True)
ws = wb["Link Stealing"]
box = {}
for r in list(ws.iter_rows(values_only=True))[1:]:
    ds, meth, ratio, atk, bacc, auc = r
    box[(ds, meth, str(ratio))] = bacc


def to_excel(method, setting):
    if method == "fgc":   return ("FGC", setting)
    if method == "ugc":   return ("UGC", setting)
    if method == "gcond": return ("GCond", setting)
    if method == "kron":  return ("KRON", setting)
    if method == "goren":
        if setting == "whole": return ("GOREN", "whole")
        ratio, var = setting.rsplit("_", 1)
        return ("GOREN-bl" if var == "bl" else "GOREN-goren", ratio)


rows = list(csv.DictReader(open(CSV)))
missing = set()
out = []
for r in rows:
    em, er = to_excel(r["method"], r["setting"])
    b = box.get((r["dataset"], em, er))
    if b is None:
        missing.add((r["dataset"], em, er))
    out.append((r, b))

if missing:
    raise SystemExit(f"MISSING box acc for: {sorted(missing)}")
print("box acc found for all", len(rows), "rows")

# rewrite CSV with box_acc_pct inserted after 'setting'
# (NB: GRID preserves argmax, so box accuracy is identical with/without defence)
fields = ["dataset", "method", "setting", "box_acc_pct", "attack",
          "auc_base_wholeshadow", "auc_grid", "delta"]
with open(CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r, b in out:
        r["box_acc_pct"] = round(float(b), 2)
        w.writerow({k: r[k] for k in fields})
print("wrote", os.path.relpath(CSV, HERE), "with box_acc_pct column")
