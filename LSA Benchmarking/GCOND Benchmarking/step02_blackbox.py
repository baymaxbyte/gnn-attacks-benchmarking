"""Step 02 -- build the GCond black boxes and validate them against the paper.

This is the gate: no attack runs until the boxes reproduce GCond's Table 2.

    python step02_blackbox.py
    python step02_blackbox.py --datasets cora,citeseer --repeats 10
"""
import argparse
import json
import os
import time

import numpy as np

import common as C
from blackbox_gcond import EVAL_REPEATS, GCondBlackBox

SETTINGS = ["whole", "0.25", "0.5", "1.0"]
BOX_NAMES = {"whole": "B0", "0.25": "B1", "0.5": "B2", "1.0": "B3"}


def main():
    ap = argparse.ArgumentParser(description="Build + validate GCond black boxes")
    ap.add_argument("--datasets", default="cora")
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--repeats", type=int, default=EVAL_REPEATS)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    C.ensure_dirs()
    datasets = [d.strip() for d in args.datasets.split(",")]
    settings = [s.strip() for s in args.settings.split(",")]

    rows, outputs = [], []
    t0 = time.time()
    for dataset in datasets:
        for setting in settings:
            box = BOX_NAMES.get(setting, setting)
            print(f"\n{'=' * 74}")
            print(f"{box}  {dataset}  setting={setting}")
            print("=" * 74, flush=True)
            bb = GCondBlackBox(dataset, setting, device=args.device)
            res = bb.validate(repeats=args.repeats)

            post = res.pop("served_posteriors")
            name = f"{dataset}_gcondbb_{setting}"
            path = C.save_posteriors(name, post)
            outputs.append(C.rel(path))

            s = post.sum(1)
            assert np.allclose(s, 1.0, atol=1e-5), \
                f"served posteriors are not normalised: sum range " \
                f"[{s.min():.6f}, {s.max():.6f}]"

            res["box"] = box
            res["posterior_file"] = os.path.basename(path)
            res["n_served"] = int(post.shape[0])
            rows.append(res)

            ref = (f"paper {res['paper_mean']:.1f} +-{res['paper_std']:.1f}"
                   if res["paper_mean"] is not None else "paper n/a")
            print(f"  -> ours {res['acc_mean']:.2f} +-{res['acc_std']:.2f}   "
                  f"{ref}   delta {res['delta']:+.2f}")

    print(f"\n{'=' * 74}\nVALIDATION GATE vs GCond Table 2\n{'=' * 74}")
    hdr = (f"  {'box':<5}{'dataset':<10}{'setting':<9}{'paper r':<9}"
           f"{'train nodes':<13}{'ours':<17}{'paper':<14}{'delta':<8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    worst = 0.0
    for r in rows:
        ours = f"{r['acc_mean']:.2f} +-{r['acc_std']:.2f}"
        paper = (f"{r['paper_mean']:.1f} +-{r['paper_std']:.1f}"
                 if r["paper_mean"] is not None else "n/a")
        d = f"{r['delta']:+.2f}" if r["delta"] is not None else "n/a"
        print(f"  {r['box']:<5}{r['dataset']:<10}{r['setting']:<9}"
              f"{r['paper_r']:<9}{r['n_train_nodes']:<13}{ours:<17}"
              f"{paper:<14}{d:<8}")
        if r["delta"] is not None:
            worst = max(worst, abs(r["delta"]))
    print(f"\n  largest absolute deviation from the paper: {worst:.2f} points")

    out = os.path.join(C.DIRS["results"], "step02_validation.json")
    with open(out, "w") as f:
        json.dump({"eval_spec": {
            "model": "2-layer GCN", "hidden": 256, "dropout": 0.0,
            "weight_decay": 5e-4, "epochs": 600, "lr": 0.01,
            "repeats": args.repeats,
            "source": "paper Appendix A.2 + tester_other_arcs.py"},
            "rows": rows}, f, indent=2)
    print(f"\n[saved] {C.rel(out)}")

    C.record(
        step="02_blackbox_gcond",
        summary=(f"Built and validated {len(rows)} GCond black boxes against "
                 f"Table 2; largest deviation {worst:.2f} points"),
        inputs=["artifacts/condensed/*.pt",
                "GCond/GCond-main/models/gcn.py",
                "2110.07580v4.cleaned.pdf (Appendix A.2 evaluation spec)"],
        outputs=outputs + [C.rel(out)],
        numbers={f"{r['dataset']}_{r['setting']}":
                 {"ours_mean": r["acc_mean"], "ours_std": r["acc_std"],
                  "paper_mean": r["paper_mean"], "delta": r["delta"],
                  "train_nodes": r["n_train_nodes"]} for r in rows},
        notes="Validated against the published GCond Table 2 accuracies.")
    print(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
