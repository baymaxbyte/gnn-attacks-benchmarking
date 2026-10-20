# PubMed attacks — end-to-end LSA + membership inference

Runs **both** privacy attacks on **PubMed** across every graph-reduction method and
ratio, with the **whole (uncoarsened) Cora graph as the shadow** everywhere a shadow
is used, with settings **identical** to the existing cora/citeseer/chameleon/squirrel
runs. One command, resumable, heavy logging, data auto-downloaded. Built for CPU
(SageMaker `ml.m6i.8xlarge`, 32 vCPU / 128 GiB).

- **LSA link-stealing** (He et al., USENIX 2021): 8 attacks + link-prediction baseline, per method/ratio, shadow = `cora_<tag>_whole`.
- **Node membership inference**: 4 threshold attacks + shadow-MLP per method/ratio, shadow = `cora_control_whole`.

## How it works (architecture)

This folder is an **orchestration layer**, not a re-implementation. It drives the
already-verified sibling benchmarks so PubMed is produced with byte-identical code,
seeds, ratios and epochs:

```
Benchmarking/temp/
  lsa/<METHOD> Benchmarking/   # FGC, KRON, UGC, GOREN, GCOND link-stealing pipelines
  Membership Attack/           # node membership-inference pipeline
  pubmed_attacks/              # <- THIS folder: config + data + orchestrator
```

**It therefore requires those sibling folders to be present** (upload all of
`Benchmarking/temp/` to SageMaker). The only edits made to the siblings are
additive and non-destructive: `DATASETS` became env-overridable (default
unchanged), `pubmed` was added to the membership config, and a `pubmed` UGC alpha
was added. PubMed is scoped into each run via the `PUBMED_ATTACKS_DATASETS` env var
so the other datasets are never recomputed.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # torch 2.8.0, torch-geometric 2.6.1, ...
```

## Run

```bash
# default: in-repo methods only (no external checkouts) + membership
python run_all.py

# everything (needs the external checkouts below)
python run_all.py --lsa-methods all --membership-methods all

# one family / subset
python run_all.py --only membership --membership-methods control,kron,ugc,fgc
python run_all.py --only lsa --lsa-methods fgc,kron,ugc
python run_all.py --skip-data              # data already prepared
```

`caffeinate -s python run_all.py` is recommended on long macOS runs. On SageMaker
run it inside `tmux`/`nohup` so a dropped connection doesn't kill it.

**Resume:** just re-run the same command. Every finished step is recorded in
`progress.json` and skipped; the sibling steps also skip their own cached
artifacts. A crash loses at most the step in flight.

### Methods and the external-checkout requirement

| group | LSA | membership | needs |
|---|---|---|---|
| in-repo (default) | fgc, kron, ugc | control, kron, ugc, fgc | nothing extra |
| external (opt-in) | goren, gcond | goren, gcond | a separate checkout/venv |

External methods are **skipped with a logged reason** unless their env is set
(same requirement as the original runs):

```bash
export GOREN_COARSEN_PYTHON=/path/to/graph-coarsening-venv/bin/python   # GOREN
export GCOND_HOME=/path/to/GCond/checkout                                # GCOND
export GCOND_PYTHON=/path/to/gcond-venv/bin/python
```

## Outputs

```
results/lsa_pubmed_wholeshadow_ALL.csv              # LSA: all methods, one row per (method,setting,attack): auc_mean/auc_std
results/lsa/<method>_pubmed_wholeshadow.csv         # per-method LSA summary
results/membership/pubmed_membership_summary.csv    # membership: per cell, threshold + shadow AUCs + box acc gap
results/membership/pubmed_membership_shadow_compare.csv   # whole-cora shadow-MLP AUC (the headline shadow result)
logs/pubmed_attacks.log                             # live master log
logs/<step>.log                                     # per-step child output (streamed live)
progress.json                                       # resume manifest
```

The raw per-config JSONs live under each benchmark's
`results/attacks_wholeshadow/` (LSA) and `results/attacks/` (membership).

## Consistency matrix (inherited unchanged from the siblings)

| method | reduction settings | black box |
|---|---|---|
| FGC (LSA) | whole, 0.3, 0.1, 0.05 | GCN h64 do0.5 lr0.01 wd1e-4 **ep60**; β-grid {.001,.01,.1,1,10}, 3 reps |
| KRON (LSA) | whole, 0.3, 0.5, 0.7 | GCN h64 do0.5 lr0.003 wd5e-4 **ep500**, split 60/20/20 seed0 |
| UGC (LSA) | whole, 30, 50, 70 | same GCN; **alpha(pubmed)=0.2** (see below) |
| GOREN (LSA) | whole + {0.3,0.5,0.7}×{bl,goren} | same GCN; weights ep400 |
| GCOND (LSA) | whole, 0.25, 0.5, 1.0 | eval ep600 rep10; condense ep600 seed1 |
| membership (all) | control; kron/ugc {.3,.5,.7}; fgc {.3,.1,.05}; goren ×{bl,goren}; gcond {.013,.026,.052} | 3-layer GCN d-64-64-C ep500 lr0.003 wd5e-4; 3 target seeds (0,1,2); split 80/20 seed42 |

Shared LSA attack: label-rate 0.10, **5 runs**, attack MLP 3×32 ep50 lr1e-3 batch128,
pair-seed 1, canonical box seed 0. PubMed positive-pair cap = 10000 (it has ~44k
edges), matching chameleon/squirrel.

### The one invented hyperparameter

UGC augments features as `[(1-alpha)X | alpha*A]` with a dataset-tuned `alpha`. The
reproductions use cora 0.19, citeseer 0.26, and 0.2 for datasets with no published
value (chameleon, squirrel). **PubMed has no published UGC alpha**, so we use the
same **0.2** default. This is the only setting not inherited verbatim. Change it in
`../lsa/UGC Benchmarking/config.py` and `../Membership Attack/reductions/ugc.py`
(both already have the `pubmed` key).

## Shadow = whole Cora (why)

The link-stealing signal is homophily-driven; Cora's homophily (~0.81) matches
PubMed's (~0.80), and both are clean citation graphs, so the attack transfers.
Chameleon/squirrel are heterophilous and would transfer poorly. Cora's attack
features are class-count-agnostic, so Cora(7 classes) → PubMed(3 classes) is fine.

## Runtime / memory (CPU)

The long pole is FGC (dense n×n solver) and GCOND condensation. On 128 GiB the
default worker counts are memory-safe (KRON/FGC build ~3–5 GB dense arrays per
cell). If you ever hit memory pressure, run the reduction steps with fewer workers.
Membership + the in-repo LSA methods finish in a few hours; FGC β-search and GCOND
are the slow parts.

## What was smoke-tested here vs on SageMaker

Validated locally: data download + both npz schemas; PubMed loads on the benchmark
and attack sides of all methods; config scoping; the orchestrator plumbing (live
logs, progress, resume); and a **full end-to-end membership `control` run** (PubMed
box acc 88.7/87.9, whole-Cora shadow AUC 0.503). The full multi-hour run (all LSA
coarsenings, GCOND/GOREN externals) is intended for SageMaker.
