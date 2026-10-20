"""Shared execution engine: live logging, resumable progress manifest, and a
subprocess runner that streams a child's output to the log AS IT RUNS.

Design goals (from the build spec):
  * lots of prints + a log file that updates live while a command executes
  * crash-safe / resumable: every finished step is recorded in progress.json and
    skipped on the next run (the sibling steps also skip their own cached
    artifacts, so re-running is always safe)
"""
import datetime
import json
import os
import subprocess
import sys
import time

import config as C


# --------------------------------------------------------------------- logging
def _stamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg, also_file=True):
    """Timestamped line to stdout and (appended, flushed) to the main log."""
    line = f"{_stamp()}  {msg}"
    print(line, flush=True)
    if also_file:
        os.makedirs(C.LOGS, exist_ok=True)
        with open(C.MAIN_LOG, "a") as f:
            f.write(line + "\n")
            f.flush()


def banner(msg):
    log("=" * 78)
    log(msg)
    log("=" * 78)


# ------------------------------------------------------------------- progress
class Progress:
    """A tiny JSON manifest of completed step keys, for resume."""

    def __init__(self, path=C.PROGRESS):
        self.path = path
        self.data = {"started": _stamp(), "steps": {}}
        if os.path.exists(path):
            try:
                self.data = json.load(open(path))
            except Exception:
                pass
        self.data.setdefault("steps", {})

    def done(self, key):
        return self.data["steps"].get(key, {}).get("status") == "done"

    def mark(self, key, status, **info):
        self.data["steps"][key] = dict(status=status, when=_stamp(), **info)
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)       # atomic, crash-safe


# ------------------------------------------------------------------- env setup
def build_env(scope_datasets=None, extra=None):
    """A child env: inherit the parent, point every per-method interpreter at our
    PYTHON, optionally scope the sibling DATASETS, and pass through any external
    checkout vars the user set (GCOND_HOME, GOREN_COARSEN_PYTHON, ...)."""
    env = dict(os.environ)
    for var in ("FGC_PYTHON", "KRON_PYTHON", "UGC_PYTHON", "GOREN_PYTHON",
                "MIA_PYTHON", "LSA_PY"):
        env.setdefault(var, C.PYTHON)
    # GOREN_COARSEN_PYTHON / GCOND_PYTHON / GCOND_HOME are left exactly as the
    # user set them (external checkouts); only default the plain interpreters.
    if scope_datasets:
        env["PUBMED_ATTACKS_DATASETS"] = ",".join(scope_datasets)
    env.setdefault("PYTHONUNBUFFERED", "1")        # so child output streams live
    if extra:
        env.update(extra)
    return env


# ------------------------------------------------------------- subprocess run
def run_cmd(cmd, cwd, env, step_key, log_name=None, timeout=None):
    """Run `cmd`, streaming combined stdout+stderr live to the main log and a
    per-step log file. Returns (returncode, seconds). Never raises on non-zero;
    the caller decides what to do."""
    os.makedirs(C.LOGS, exist_ok=True)
    step_log = os.path.join(C.LOGS, (log_name or step_key.replace(":", "_")) + ".log")
    pretty = " ".join(str(c) for c in cmd)
    log(f"[run] {step_key}")
    log(f"      cwd={cwd}")
    log(f"      cmd={pretty}")
    log(f"      step-log -> {os.path.relpath(step_log, C.HERE)}")
    t0 = time.time()
    with open(step_log, "a") as lf:
        lf.write(f"\n{'='*78}\n{_stamp()}  {step_key}\n{pretty}\n{'='*78}\n")
        lf.flush()
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                # Echo child lines to our console too, prefixed, so the live
                # terminal shows progress; the main log keeps a clean record.
                sys.stdout.write(f"    | {line}")
                sys.stdout.flush()
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            lf.write("\n[TIMEOUT] killed\n")
            log(f"[TIMEOUT] {step_key} exceeded {timeout}s -> killed")
            return 124, time.time() - t0
    dt = time.time() - t0
    rc = proc.returncode
    log(f"[{'ok' if rc == 0 else 'FAIL'}] {step_key}  rc={rc}  {dt/60:.1f} min")
    return rc, dt
