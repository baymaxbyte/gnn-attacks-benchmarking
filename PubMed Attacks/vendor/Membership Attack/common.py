"""Shared conventions: artifact paths, the manifest, and CPU thread policy.

Same pattern as the link-stealing benchmarks: each worker runs single-threaded and
parallelism comes from running independent configurations in separate processes.
"""
import json
import os
import time

import config as C

DIRS = (C.SPLITS, C.REDUCED, C.POSTERIORS, C.RESULTS, C.LOGS)


def ensure_dirs():
    for d in DIRS:
        os.makedirs(d, exist_ok=True)


def single_thread_env():
    return {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1"}


def pin_single_thread():
    import multiprocessing as mp

    import torch
    if mp.current_process().name != "MainProcess":
        torch.set_num_threads(1)


def main_is_importable():
    import __main__
    path = getattr(__main__, "__file__", None)
    return bool(path) and os.path.exists(path)


def parallel_map(fn, jobs, workers=8, chunksize=1, serial=False, quiet=False):
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
    return os.path.relpath(path, C.HERE)


def record(step, summary, inputs=None, outputs=None, numbers=None, notes=None):
    entry = {
        "step": step, "summary": summary,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": list(inputs or []), "outputs": list(outputs or []),
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
        json.dump({"pipeline": "membership_inference", "steps": log}, f, indent=2)
    print(f"\n[manifest] recorded step {step}: {summary}")
    return entry


def split_path(dataset):
    return os.path.join(C.SPLITS, f"{dataset}_split.npz")
