"""
The 8 link stealing attacks of He et al. (USENIX Security 2021), Section 4.

Threat model (Section 3). The adversary's knowledge is a triple
K = (F, A*, D'):

    F   nodes' attributes of the target dataset
    A*  a partial target graph (a subset of true links)
    D'  a shadow dataset with its own attributes and graph

Table 2 enumerates all 8 combinations:

    Attack   F   A*   D'    type          trained on
    ------   -   --   --    -----------   -------------------------------
    0        x   x    x     unsupervised  -
    1        x   x    Y     supervised    shadow attack dataset
    2        Y   x    x     unsupervised  -
    3        x   Y    x     supervised    target attack training half
    4        x   Y    Y     supervised    shadow + target training half
    5        Y   x    Y     supervised    shadow attack dataset
    6        Y   Y    x     supervised    target attack training half
    7        Y   Y    Y     supervised    shadow + target training half

Two problems Section 4 calls out, and how they are solved:

  * Node pair order. The prediction must not depend on whether the pair is
    given as (u,v) or (v,u). Handled by (a) canonicalising every pair to r < c
    when the attack dataset is built, and (b) using only symmetric functions --
    p5: "all distance metrics we adopt are symmetric ... this naturally solves
    the problem of node pair order" -- with the entropy pair folded through the
    symmetric Table-14 operators.

  * Dimension mismatch. A shadow dataset has a different number of classes and
    attributes than the target, so raw posteriors cannot transfer. The attacks
    that use D' (1, 4, 5, 7) therefore use ONLY scalar features: the 8 distances
    and the entropy-derived values, both dimension-independent. Attacks 3 and 6
    stay inside one dataset and may additionally use raw posteriors.

Which model's posteriors each attack may read follows from F. Footnote 1 on p7:
"We do not train the reference model for attacks when F is unavailable." So the
reference model g and the raw attributes appear only in the F = Y attacks
(2, 5, 6, 7).
"""
import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, precision_score, recall_score

from .attack_model import (DEFAULT_BATCH_SIZE, scale_features,
                           train_and_evaluate, unsupervised_auc)
from .features import (SIMILARITY_NAMES, get_metrics, operator_func,
                       pairwise_distances)
from .spec import ATTACK0_SELECTED_METRIC, KMEANS_K


# ------------------------------------------------------------------ io
def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _vecs(row):
    """The three vector types for both endpoints of a pair."""
    return (np.array(row["gcn_pred0"]), np.array(row["gcn_pred1"]),        # f
            np.array(row["dense_pred0"]), np.array(row["dense_pred1"]),    # g
            np.array(row["feature_arr0"]), np.array(row["feature_arr1"]))  # F


# ================================================== UNSUPERVISED ATTACKS
# p8: "Note that for Attack-0 and Attack-2, the AUC scores keep the same since
#      these two attacks are unsupervised."
# There is nothing to train and nothing random, so each returns one exact value
# per metric, with no standard deviation.

def kmeans_link_predictions(distances, seed=42):
    """Paper p8 threshold-estimation method.

        "we propose to use clustering, such as K-means. Concretely, we obtain a
         set of node pairs' distances, and perform K-means on these distances
         with K being set to 2. The cluster with lower (higher) average distance
         value is considered as the set of positive (negative) node pairs."

    Returns a 0/1 vector, 1 = predicted linked. This is what turns an
    unsupervised AUC into the concrete Precision / Recall / F1 of Table 15.
    """
    d = np.nan_to_num(np.asarray(distances, dtype=np.float64),
                      nan=0.0, posinf=0.0, neginf=0.0).reshape(-1, 1)
    km = KMeans(n_clusters=KMEANS_K, random_state=seed, n_init=10)
    cluster = km.fit_predict(d)
    # 1-D K-means centres ARE the per-cluster mean distances, so the smaller
    # centre is the "lower average distance value" cluster.
    linked_cluster = int(np.argmin(km.cluster_centers_.reshape(-1)))
    return (cluster == linked_cluster).astype(int)


def attack_0(test_rows, *, seed=42, **_):
    """K = (x, x, x). Only the target posteriors f(u), f(v) are available.

    Section 4: "if there is a link between two nodes, then their posteriors
    obtained from the target model should be closer". With no training data the
    adversary can only measure distance, so each of the 8 metrics is used
    directly as a score.

    Figure 1 reports all 8. Correlation is the metric the paper selects
    ("Correlation distance achieves the best performance followed by Cosine
    distance across all datasets"), and Table 15 reports Attack-0's
    Precision / Recall / F1 / AUC under it.
    """
    labels = [r["label"] for r in test_rows]
    dists = np.array([pairwise_distances(np.array(r["gcn_pred0"]),
                                         np.array(r["gcn_pred1"]))
                      for r in test_rows])

    per_metric = {name: unsupervised_auc(labels, dists[:, j])
                  for j, name in enumerate(SIMILARITY_NAMES)}

    sel = ATTACK0_SELECTED_METRIC
    pred = kmeans_link_predictions(dists[:, SIMILARITY_NAMES.index(sel)], seed=seed)

    return {
        "per_metric": per_metric,
        "selected_metric": sel,
        "auc": per_metric[sel],
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "unsupervised": True,
    }


def correlation_distance_buckets(test_rows, edges=(0.00, 0.01, 0.02, 0.03, 0.04)):
    """Table 5: "AUC in different Correlation distance levels for Attack-0".

    The paper reports Pubmed with buckets 0.00-0.01, 0.01-0.02, 0.02-0.03 and
    0.03-0.04. Within a narrow band of Correlation distance the metric loses its
    discriminative power (the reported AUCs fall from 0.608 to 0.399, i.e. toward
    and past chance), which is the paper's argument for why a single global
    threshold is hard to pick and why K-means clustering is used instead.

    Each bucket is scored on its own pairs only, so a bucket with no positives
    or no negatives is reported as None rather than a misleading number.
    """
    labels = np.array([r["label"] for r in test_rows])
    corr = np.array([correlation_of(r) for r in test_rows])
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (corr >= lo) & (corr < hi)
        n, npos = int(sel.sum()), int(labels[sel].sum())
        entry = {"range": f"{lo:.2f}-{hi:.2f}", "n_pairs": n,
                 "n_positive": npos, "n_negative": n - npos, "auc": None}
        if n and 0 < npos < n:
            entry["auc"] = float(unsupervised_auc(labels[sel], corr[sel]))
        out.append(entry)
    return out


def correlation_of(row):
    """Correlation distance between the pair's target posteriors."""
    j = SIMILARITY_NAMES.index(ATTACK0_SELECTED_METRIC)
    return pairwise_distances(np.array(row["gcn_pred0"]),
                              np.array(row["gcn_pred1"]))[j]


def attack_2(test_rows, *, seed=42, **_):
    """K = (Y, x, x). Adds node attributes, so the adversary can also build the
    reference model g on the target attributes plus the known labels.

    Section 4 defines the posterior distance difference as

        d(f(u), f(v)) - d(g(u), g(v))

    i.e. TARGET minus REFERENCE. The intuition is that g sees the attributes but
    not the graph, so subtracting it removes the attribute-similarity component
    and leaves the part of the posterior similarity that the graph caused. The
    released code computes `reference - target`, the opposite sign; AUC is
    invariant because sub-0.5 scores get flipped, but the feature itself is
    negated, so the paper's orientation is used here.

    Figure 4 scores four information families separately, each with all 8
    metrics: the posterior difference, the target posteriors, the reference
    posteriors, and the attributes.
    """
    labels = [r["label"] for r in test_rows]
    tgt, ref, feat = [], [], []
    for r in test_rows:
        t0, t1, r0, r1, f0, f1 = _vecs(r)
        tgt.append(pairwise_distances(t0, t1))
        ref.append(pairwise_distances(r0, r1))
        feat.append(pairwise_distances(f0, f1))
    tgt, ref, feat = np.array(tgt), np.array(ref), np.array(feat)

    diff = tgt - ref                       # paper Section 4 orientation

    per_metric = {}
    for group, mat in (("posterior_diff", diff), ("target", tgt),
                       ("reference", ref), ("feature", feat)):
        for j, name in enumerate(SIMILARITY_NAMES):
            per_metric[f"{group}_{name}"] = unsupervised_auc(labels, mat[:, j])

    sel = f"target_{ATTACK0_SELECTED_METRIC}"
    pred = kmeans_link_predictions(
        tgt[:, SIMILARITY_NAMES.index(ATTACK0_SELECTED_METRIC)], seed=seed)

    return {
        "per_metric": per_metric,
        "selected_metric": sel,
        "auc": per_metric[sel],
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "unsupervised": True,
    }


# ============================================================ FEATURISERS
#
# Table 3 lists exactly which of eight feature groups each supervised attack
# uses. Its columns, in order:
#
#     d(f(u),f(v))      Psi(f(u),f(v))      Psi(e(f(u)),e(f(v)))
#     d(g(u),g(v))      Psi(g(u),g(v))      Psi(e(g(u)),e(g(v)))
#     d(Fu,Fv)          Psi(Fu,Fv)
#
# and the caption: "the features used in these attack models include all the
# distance metrics and pairwise vector operations", i.e. every d(.,.) group is
# all 8 metrics of Table 13 and every Psi(.,.) group is all 4 operators of
# Table 14.
#
# Building the featurisers by composing these named groups (rather than writing
# each attack's vector by hand) means Table 3 can be asserted directly, and the
# Figure 5 ablation is just a subset of group names.
#
# C = number of target classes, D = number of node attributes.

FEATURE_GROUPS = {
    # name        (extractor, width expression)
    "d_f":     (lambda t0, t1, r0, r1, f0, f1, op, mt:
                pairwise_distances(t0, t1), "8"),
    "psi_f":   (lambda t0, t1, r0, r1, f0, f1, op, mt:
                operator_func(op, t0, t1), "4C"),
    "psi_e_f": (lambda t0, t1, r0, r1, f0, f1, op, mt:
                get_metrics(t0, t1, mt, op), "4"),
    "d_g":     (lambda t0, t1, r0, r1, f0, f1, op, mt:
                pairwise_distances(r0, r1), "8"),
    "psi_g":   (lambda t0, t1, r0, r1, f0, f1, op, mt:
                operator_func(op, r0, r1), "4C"),
    "psi_e_g": (lambda t0, t1, r0, r1, f0, f1, op, mt:
                get_metrics(r0, r1, mt, op), "4"),
    "d_F":     (lambda t0, t1, r0, r1, f0, f1, op, mt:
                pairwise_distances(f0, f1), "8"),
    "psi_F":   (lambda t0, t1, r0, r1, f0, f1, op, mt:
                operator_func(op, f0, f1), "4D"),
}

# Table 3, transcribed. Order follows the table's columns.
GROUP_ORDER = ("d_f", "psi_f", "psi_e_f",
               "d_g", "psi_g", "psi_e_g",
               "d_F", "psi_F")

ATTACK_FEATURE_GROUPS_TABLE3 = {
    1: ("d_f", "psi_e_f"),                                      # 12
    3: ("d_f", "psi_f", "psi_e_f"),                             # 4C + 12
    4: ("d_f", "psi_e_f"),                                      # 12
    5: ("d_f", "psi_e_f", "d_g", "psi_e_g", "d_F"),             # 32
    6: GROUP_ORDER,                                             # 8C + 4D + 32
    7: ("d_f", "psi_e_f", "d_g", "psi_e_g", "d_F"),             # 32
}

# Table 3 and Table 10 cannot both be right about Attack-6.
#
# Table 3 ticks Psi(Fu,Fv) for Attack-6, which adds 4D columns -- 5,732 of them
# on Cora, against 5,278 training pairs. Measured on Cora, 3 runs each:
#
#     Table 3 literally (8C + 4D + 32 = 5820 features)   AUC 0.9368
#     without Psi(Fu,Fv) (8C + 32 = 88 features)          AUC 0.9617
#     Table 10's published value                         AUC 0.964
#
# With more features than training samples the attack model overfits, and the
# result moves AWAY from the number the paper reports. The released attack_6.py
# computes d(Fu,Fv) but never applies the operators to raw attributes, so the
# 88-feature set is what the authors actually ran. We therefore treat the
# Psi(Fu,Fv) tick as a documentation error in Table 3 and default to the feature
# set that reproduces Table 10, while keeping the literal reading one flag away
# (`--table3_strict`) so the discrepancy stays visible and testable.
ATTACK_FEATURE_GROUPS = dict(ATTACK_FEATURE_GROUPS_TABLE3)
ATTACK_FEATURE_GROUPS[6] = tuple(g for g in GROUP_ORDER if g != "psi_F")

# Attacks 1, 4, 5 and 7 transfer from a shadow dataset, so their feature vector
# must not depend on C or D. Every group they use is a fixed-width scalar block,
# which is exactly why Table 3 leaves Psi(f), Psi(g) and Psi(F) unchecked for
# them -- those are the C- and D-dependent ones.
DIMENSION_DEPENDENT_GROUPS = {"psi_f", "psi_g", "psi_F"}


def make_featuriser(groups):
    """Compose a featuriser from Table-3 group names, in Table-3 order."""
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")
    ordered = [g for g in GROUP_ORDER if g in set(groups)]

    def featurise(row, operator, metric_type):
        t0, t1, r0, r1, f0, f1 = _vecs(row)
        return np.concatenate([
            FEATURE_GROUPS[g][0](t0, t1, r0, r1, f0, f1, operator, metric_type)
            for g in ordered])

    featurise.groups = tuple(ordered)
    return featurise


def build_xy(rows, featuriser, operator, metric_type):
    x = np.array([np.nan_to_num(featuriser(r, operator, metric_type))
                  for r in rows], dtype=np.float32)
    y = np.array([r["label"] for r in rows])
    return x, y


# ==================================================== SUPERVISED ATTACKS
def prepare_supervised(train_rows, test_rows, featuriser, *, operator,
                       metric_type, independent_test_fit, verbose=True):
    """Featurise and scale once.

    The paper's 5 repetitions vary only the attack model ("the AUC scores keep
    the same" for the unsupervised attacks, which share this data), so the
    feature matrices are built once and reused. This is both faithful and about
    5x cheaper, since the distance computations dominate the runtime.
    """
    x_train, y_train = build_xy(train_rows, featuriser, operator, metric_type)
    x_test, y_test = build_xy(test_rows, featuriser, operator, metric_type)
    if verbose:
        print(f"    features={x_train.shape[1]}  "
              f"train={len(x_train)}  test={len(x_test)}")
    x_train, x_test = scale_features(x_train, x_test, independent_test_fit)
    return x_train, y_train, x_test, y_test


def run_supervised_once(prepared, *, epochs, batch_size, device, seed,
                        hidden_layers, verbose=True):
    """One repetition: fresh initialisation, fresh dropout and batch order."""
    x_train, y_train, x_test, y_test = prepared
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    res = train_and_evaluate(x_train, y_train, x_test, y_test,
                             epochs=epochs, batch_size=batch_size,
                             device=device, seed=seed, verbose=verbose,
                             hidden_layers=hidden_layers)
    res["n_features"] = int(np.asarray(x_train).shape[1])
    return res


def _supervised(train_rows, test_rows, featuriser, *, operator, metric_type,
                epochs, batch_size, device, seed, independent_test_fit,
                hidden_layers, verbose=True):
    """Single-shot convenience wrapper: featurise, then train once."""
    prepared = prepare_supervised(train_rows, test_rows, featuriser,
                                  operator=operator, metric_type=metric_type,
                                  independent_test_fit=independent_test_fit,
                                  verbose=verbose)
    return run_supervised_once(prepared, epochs=epochs, batch_size=batch_size,
                               device=device, seed=seed,
                               hidden_layers=hidden_layers, verbose=verbose)


# One featuriser per supervised attack.
FEATURISERS = {a: make_featuriser(g) for a, g in ATTACK_FEATURE_GROUPS.items()}
FEATURISERS_TABLE3 = {a: make_featuriser(g)
                      for a, g in ATTACK_FEATURE_GROUPS_TABLE3.items()}


def _make_attack(a):
    def run(train_rows, test_rows, **kw):
        return _supervised(train_rows, test_rows, FEATURISERS[a], **kw)
    run.__name__ = f"attack_{a}"
    run.__doc__ = (f"K = {_tag(a)}. Table-3 feature groups: "
                   f"{', '.join(ATTACK_FEATURE_GROUPS[a])}.")
    return run


# ---------------------------------------------------------------- registry
# Table 2 knowledge triples. Declared before the attack wrappers so their
# docstrings can quote them.
KNOWLEDGE = {
    0: dict(F=False, A=False, D=False),
    1: dict(F=False, A=False, D=True),
    2: dict(F=True,  A=False, D=False),
    3: dict(F=False, A=True,  D=False),
    4: dict(F=False, A=True,  D=True),
    5: dict(F=True,  A=False, D=True),
    6: dict(F=True,  A=True,  D=False),
    7: dict(F=True,  A=True,  D=True),
}


def _tag(a):
    k = KNOWLEDGE[a]
    return f"({'Y' if k['F'] else 'x'},{'Y' if k['A'] else 'x'}," \
           f"{'Y' if k['D'] else 'x'})"


attack_1 = _make_attack(1)
attack_3 = _make_attack(3)
attack_4 = _make_attack(4)
attack_5 = _make_attack(5)
attack_6 = _make_attack(6)
attack_7 = _make_attack(7)

UNSUPERVISED = {0: attack_0, 2: attack_2}

SUPERVISED = {1: attack_1, 3: attack_3, 4: attack_4,
              5: attack_5, 6: attack_6, 7: attack_7}

# What each supervised attack trains on, per p7.
#   shadow_all   = the ENTIRE shadow attack dataset; the paper does not split it
#                  ("we do not split this attack dataset because we do not use
#                  it for evaluation")
#   target_train = the target's attack TRAINING half, i.e. the partial graph A*
TRAIN_SOURCES = {
    1: ("shadow_all",),
    3: ("target_train",),
    4: ("shadow_all", "target_train"),
    5: ("shadow_all",),
    6: ("target_train",),
    7: ("shadow_all", "target_train"),
}

NEEDS_SHADOW = {1, 4, 5, 7}

knowledge_tag = _tag


# ------------------------------------------------- Figure 5 feature ablation
# "Average AUC for Attack-3 on all the 8 datasets with different set of
#  features" -- the legend lists four bars:
#      Psi(e(f(u)),e(f(v)))    Psi(f(u),f(v))    d(f(u),f(v))    All
ABLATION_FIGURE5 = {
    "psi_e_f": ("psi_e_f",),
    "psi_f": ("psi_f",),
    "d_f": ("d_f",),
    "all": ATTACK_FEATURE_GROUPS[3],
}

ABLATION_LABELS = {
    "psi_e_f": "Psi(e(f(u)),e(f(v)))",
    "psi_f": "Psi(f(u),f(v))",
    "d_f": "d(f(u),f(v))",
    "all": "All",
}
