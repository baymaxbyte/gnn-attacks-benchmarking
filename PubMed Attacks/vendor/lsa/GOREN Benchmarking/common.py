"""Shared conventions: artifact paths, the manifest, and CPU thread policy.

Thread policy: every workload in this pipeline is small (2-layer GCNs on graphs of
a few thousand nodes, 3x32 attack MLPs). Measured on this machine, going from 1 to
10 torch threads makes them SLOWER, and MPS is slower still. So each worker runs
single-threaded and parallelism comes from running independent configurations in
separate processes.
"""
import json
import os
import time

import config as C

DIRS = (C.COARSENED, C.RESULTS, C.LOGS)


def ensure_dirs():
    for d in DIRS:
        os.makedirs(d, exist_ok=True)


def single_thread_env():
    """Env vars that must be set BEFORE numpy/torch are imported.

    OpenBLAS reads these at import time; torch.set_num_threads(1) alone does not
    constrain the BLAS thread pool.
    """
    return {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1"}


def pin_single_thread():
    """Pin to one thread -- but only inside a worker process.

    In the serial fallback this would otherwise throttle the main process to one
    thread for no reason, so it is a no-op there.
    """
    import multiprocessing as mp

    import torch
    if mp.current_process().name != "MainProcess":
        torch.set_num_threads(1)


def main_is_importable():
    """True if a spawn-started child can re-import the parent's __main__.

    The spawn start method re-executes __main__ by file path. When the parent was
    launched from stdin (python - <<'EOF'), from -c, or from a REPL, that path does
    not exist: every worker raises FileNotFoundError during startup, Pool replaces
    it, and the cycle repeats forever. The parent never finishes and has to be
    killed by hand. Detect it up front instead.
    """
    import __main__
    path = getattr(__main__, "__file__", None)
    return bool(path) and os.path.exists(path)


def parallel_map(fn, jobs, workers=8, chunksize=1, serial=False, quiet=False):
    """Map fn over jobs in separate processes, serially when that is not safe.

    fn must be a module-level function (spawn pickles it by qualified name).
    """
    jobs = list(jobs)
    if not jobs:
        return []

    reason = None
    if serial:
        reason = "--serial requested"
    elif workers <= 1 or len(jobs) == 1:
        reason = "single worker or single job"
    elif not main_is_importable():
        reason = ("__main__ has no importable file path, so spawned workers cannot "
                  "start (launched from stdin, -c or a REPL)")

    if reason:
        if not quiet and not serial:
            print(f"[parallel_map] running serially: {reason}")
        return [fn(j) for j in jobs]

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(workers, len(jobs))) as pool:
        return pool.map(fn, jobs, chunksize=chunksize)


def rel(path):
    """Path relative to the benchmark root, for manifest readability."""
    return os.path.relpath(path, C.HERE)


def record(step, summary, inputs=None, outputs=None, numbers=None, notes=None):
    """Append one step to manifest.json. Idempotent per step name."""
    entry = {
        "step": step,
        "summary": summary,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": list(inputs or []),
        "outputs": list(outputs or []),
        "numbers": dict(numbers or {}),
    }
    if notes:
        entry["notes"] = notes

    log = []
    if os.path.exists(C.MANIFEST):
        with open(C.MANIFEST) as f:
            log = json.load(f).get("steps", [])
    log = [e for e in log if e.get("step") != step]
    log.append(entry)
    log.sort(key=lambda e: str(e["step"]))
    with open(C.MANIFEST, "w") as f:
        json.dump({"pipeline": "goren_leakage", "steps": log}, f, indent=2)
    print(f"\n[manifest] recorded step {step}: {summary}")
    return entry


# ------------------------------------------------------------ FGC conventions
def coarse_path(dataset, r, tag="goren"):
    return os.path.join(C.COARSENED, f"{tag}_{dataset}_ratio{r}.npz")


def setting_label(dataset, r, n=None):
    """Human label for a reduction setting, e.g. 'r=0.1 (270 nodes)'."""
    if r is None:
        return "whole"
    return f"r={r}" + (f" ({int(n * r)} nodes)" if n else "")


def save_attack_posteriors(name, posteriors):
    """Write posteriors where the attack code expects them: data/pred/<name>_pred.pkl.

    Format matches what the attack implementation already reads: a pickled numpy
    array of shape (n_nodes, n_classes), dtype float64.
    """
    import pickle as pkl

    import numpy as np
    os.makedirs(C.ATTACK_PRED, exist_ok=True)
    p = np.asarray(posteriors, dtype=np.float64)
    path = os.path.join(C.ATTACK_PRED, f"{name}_pred.pkl")
    with open(path, "wb") as f:
        f.write(pkl.dumps(p))
    print(f"  [saved] data/pred/{name}_pred.pkl  shape={p.shape}")
    return path
