#!/usr/bin/env python3
"""Fisher-Rao "room" diagnostic (todo.md direction A, first step).

Question: GRID budgets its noise in the EUCLIDEAN L1 ball and stalls on confident
posteriors. If instead we move along the simplex's intrinsic FISHER-RAO geometry
(a geodesic on the sqrt(p) sphere), do we get materially more room to cut
adjacent-node similarity -- subject to the SAME true utility constraint GRID uses,
namely argmax (label) preservation?

Both methods minimise the SAME objective -- mean GRID similarity (corr+cos) of a
core node to its graph neighbours -- preserving argmax. They differ only in HOW
they move:

  * L1 / GRID : additive projected-gradient inside ||q-p||_1 <= theta
                (exactly grid_defense.constraint_check), swept over theta.
  * Fisher-Rao: geodesic descent on the unit sphere u = sqrt(q). The Riemannian
                gradient (Euclidean grad projected off the radial direction) does
                NOT vanish at a vertex, so -- unlike additive/mirror steps -- it
                can move away from a near-one-hot posterior.

We trace each method's achieved similarity vs move size (L1 displacement, the
unit GRID itself budgets in), and report the LOWEST adjacent similarity each can
reach under argmax preservation, vs the n-hop floor delta GRID targets.

NOTE on targets: both argmax-preserving, so by the paper's own utility definition
(label/box accuracy) BOTH are zero-cost. The real question is reachability.
We run on the SOFT GCN posteriors (match the paper's confidence, non-degenerate)
and also report the saturated KRON box for contrast.
"""
import os, sys, json, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDROOT = os.path.dirname(HERE)
sys.path.insert(0, GRIDROOT)
import grid_defense as G

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
EPS = 1e-12
torch.set_default_dtype(torch.float64)

SOFT_PRED = os.path.join(GRIDROOT, "paper_faithful_gcn", "attack", "data", "pred")
THETAS = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
FR_STEPS, FR_DT = 70, 0.06
L1_ITERS, L1_LR = 40, 0.25
MAX_CORE = 150
N_HOP = 3
L1GRID = np.linspace(0, 1.4, 60)        # common L1-displacement axis


def kl(q, p):
    q = np.clip(q, EPS, 1); p = np.clip(p, EPS, 1); q = q / q.sum()
    return float((q * (np.log(q) - np.log(p))).sum())


def l1d(q, p):
    return float(np.abs(q - p).sum())


def meansim_t(q, P):
    return G._sim_rows(q, P).mean()


def meansim_np(q, P):
    return float(G._sim_rows(torch.tensor(q), torch.tensor(P)).mean())


# ----------------------------------------------------------------- L1 (GRID)
def l1_point(p, P, theta):
    v = torch.tensor(p); Pt = torch.tensor(P)
    s = torch.zeros_like(v, requires_grad=True)
    for _ in range(L1_ITERS):
        g, = torch.autograd.grad(meansim_t(v + s, Pt), s)
        with torch.no_grad():
            s = G.constraint_check(s - L1_LR * g, v, theta).detach().requires_grad_(True)
    q = np.clip((v + s).detach().numpy(), 0, None)
    return q / q.sum()


# ------------------------------------------------- Fisher-Rao sphere geodesic
def fr_traj(p, P):
    """Geodesic descent on the sqrt(p) sphere; returns [(L1, KL, sim), ...]."""
    c = int(np.argmax(p))
    Pt = torch.tensor(P)
    u = np.sqrt(np.clip(p, 0, None)); u = u / (np.linalg.norm(u) + EPS)
    out = []
    for _ in range(FR_STEPS):
        ut = torch.tensor(u, requires_grad=True)
        q = ut * ut
        q = q / q.sum()
        g, = torch.autograd.grad(meansim_t(q, Pt), ut)     # dJ/du
        g = g.detach().numpy()
        g = g - (g @ u) * u                                 # Riemannian (tangent) grad
        ng = np.linalg.norm(g)
        if ng < 1e-9:
            break
        d = -g / ng                                         # unit descent direction
        u = np.cos(FR_DT) * u + np.sin(FR_DT) * d           # geodesic step
        u = np.clip(u, 0, None); u = u / (np.linalg.norm(u) + EPS)
        q = u * u; q = q / q.sum()
        if int(np.argmax(q)) != c:                          # keep the label
            q = q.copy(); q[c] = q.max() * (1 + 1e-6); q = q / q.sum()
            u = np.sqrt(q)
        out.append((l1d(q, p), kl(q, p), meansim_np(q, P)))
    return out


def frontier(points, xgrid):
    """points: (x=L1, sim). Return min-sim-achieved-by-x interpolated on xgrid."""
    pts = sorted(points)
    xs = np.array([a for a, _ in pts]); ss = np.array([b for _, b in pts])
    best = np.minimum.accumulate(ss)
    return np.interp(xgrid, xs, best, left=ss[0], right=best[-1])


def load_V(ds, source):
    if source == "soft":
        V = np.asarray(pickle.load(open(os.path.join(SOFT_PRED, f"{ds}_softgcn_whole_pred.pkl"), "rb")), float)
        Rg = np.asarray(pickle.load(open(os.path.join(SOFT_PRED, f"{ds}_softgcn_whole_grid_pred.pkl"), "rb")), float)
    else:
        V = np.load(os.path.join(G.POSTERIORS, f"{ds}_whole.npy")).astype(float)
        Rg = G.load_pred(os.path.join(G.PRED, f"{ds}_kronbb_whole_grid_pred.pkl"))
    return V / V.sum(1, keepdims=True), Rg / Rg.sum(1, keepdims=True)


def run(ds, source):
    V, Rg = load_V(ds, source)
    _, nbrs = G.build_adjacency(ds)
    delta, _ = G.estimate_delta(V, nbrs, N_HOP)
    core, _, _ = G.select_core_nodes(V, nbrs, delta)
    rng = np.random.RandomState(0)
    sample = [int(i) for i in (rng.permutation(core)[:MAX_CORE] if len(core) > MAX_CORE else core)]

    l1_curves, fr_curves = [], []
    l1_min, fr_min, sim0s, tops = [], [], [], []
    grid_l1, grid_sim = [], []
    for i in sample:
        P = V[[int(j) for j in nbrs[i]]]
        if len(P) == 0:
            continue
        sim0 = meansim_np(V[i], P); sim0s.append(sim0); tops.append(float(V[i].max()))
        l1_pts = [(0.0, sim0)] + [(l1d(q := l1_point(V[i], P, th), V[i]), meansim_np(q, P)) for th in THETAS]
        fr_pts = [(0.0, sim0)] + [(a, s) for a, _, s in fr_traj(V[i], P)]
        l1_curves.append(frontier(l1_pts, L1GRID)); fr_curves.append(frontier(fr_pts, L1GRID))
        l1_min.append(min(s for _, s in l1_pts)); fr_min.append(min(s for _, s in fr_pts))
        grid_l1.append(l1d(Rg[i], V[i])); grid_sim.append(meansim_np(Rg[i], P))
    return dict(ds=ds, source=source, delta=delta, n_core=len(core), n=len(sim0s),
                sim0=float(np.mean(sim0s)), L1=np.mean(l1_curves, 0), FR=np.mean(fr_curves, 0),
                l1_min=float(np.mean(l1_min)), fr_min=float(np.mean(fr_min)),
                grid_l1=float(np.mean(grid_l1)), grid_sim=float(np.mean(grid_sim)),
                tops=np.array(tops), l1_minA=np.array(l1_min), fr_minA=np.array(fr_min),
                sim0A=np.array(sim0s))


def main():
    summ = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, ds in zip(axes, ["cora", "citeseer"]):
        r = run(ds, "soft")
        print(f"\n===== {ds} (SOFT GCN) =====")
        print(f"  core={r['n_core']} sampled={r['n']}  sim0(corr+cos)={r['sim0']:.3f}  n-hop floor delta={r['delta']:.3f}")
        print(f"  GRID theta=0.4 operating point: L1={r['grid_l1']:.3f}  sim={r['grid_sim']:.3f}  (drop {r['sim0']-r['grid_sim']:.3f})")
        print(f"  LOWEST sim reachable (argmax-preserving):  L1/GRID={r['l1_min']:.3f}   Fisher-Rao={r['fr_min']:.3f}   (floor delta={r['delta']:.3f})")
        print(f"  total drop to min:  L1={r['sim0']-r['l1_min']:.3f}   FR={r['sim0']-r['fr_min']:.3f}   (FR/L1 = {(r['sim0']-r['fr_min'])/max(r['sim0']-r['l1_min'],1e-9):.2f}x)")
        conf = r["tops"] >= 0.8
        if conf.any():
            print(f"  confident nodes top>=0.8 (n={int(conf.sum())}): drop L1={np.mean(r['sim0A'][conf]-r['l1_minA'][conf]):.3f}  FR={np.mean(r['sim0A'][conf]-r['fr_minA'][conf]):.3f}")
        rk = run(ds, "kron")
        print(f"  [contrast] KRON (saturated): sim0={rk['sim0']:.3f} min L1={rk['l1_min']:.3f} FR={rk['fr_min']:.3f} delta={rk['delta']:.3f} (mean top={rk['tops'].mean():.3f})")
        summ += [{k: (v if not isinstance(v, np.ndarray) else None) for k, v in r.items() if k not in ("L1", "FR", "tops", "l1_minA", "fr_minA", "sim0A")},
                 {k: (v if not isinstance(v, np.ndarray) else None) for k, v in rk.items() if k not in ("L1", "FR", "tops", "l1_minA", "fr_minA", "sim0A")}]
        ax.plot(L1GRID, r["L1"], "-", color="#dd8452", label="L1 / GRID geometry")
        ax.plot(L1GRID, r["FR"], "-", color="#4c72b0", label="Fisher-Rao geodesic")
        ax.scatter([r["grid_l1"]], [r["grid_sim"]], color="k", zorder=5, label=f"GRID theta=0.4 point")
        ax.axhline(r["delta"], color="green", ls=":", lw=1, label=f"n-hop floor delta={r['delta']:.2f}")
        ax.set_xlabel("move size  ||q - p||_1"); ax.set_ylabel("mean adjacent similarity (corr+cos)")
        ax.set_title(f"{ds} (soft GCN)"); ax.legend(fontsize=7.5)
    fig.suptitle("Fisher-Rao room diagnostic: lowest adjacent similarity reachable vs move size\n"
                 "(both argmax-preserving; lower = better defense; green = GRID's target floor)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(OUT, "fr_room_frontier.png"); fig.savefig(p, dpi=135)
    json.dump(summ, open(os.path.join(OUT, "fr_room_summary.json"), "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
