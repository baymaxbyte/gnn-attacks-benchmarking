"""
The feature vocabulary shared by all 8 attacks: 8 distance metrics (Table 13),
4 pairwise operators (Table 14), and the entropy/divergence summaries.

Paper clause (p5):

    "we experimented with 8 common distance metrics: Cosine distance, Euclidean
     distance, Correlation distance, Chebyshev distance, Braycurtis distance,
     Canberra distance, Manhattan distance, and Square-euclidean distance.
     Their formal definitions are in Table 13 in Appendix. It is worth noting
     that all distance metrics we adopt are symmetric, i.e.
     d(f(u),f(v)) = d(f(v),f(u)), this naturally solves the problem of node
     pair order."

ORDERING NOTE. The paper is internally inconsistent about the order of the last
three metrics:

    p5 body text     ... Braycurtis, Canberra, Manhattan, Square-euclidean
    Figure 1 legend  ... Braycurtis, Canberra, Manhattan, Sqeuclidean
    Table 13 (p18)   ... Braycurtis, Manhattan, Canberra, Sqeuclidean

We follow the body text and Figure 1, which agree with each other and with the
released code. Table 13 only lists the formulas, so its ordering carries no
experimental meaning. DISTANCE_ORDER in spec.py is the single definition, and
`SIMILARITY_NAMES` is asserted against it at import time.

Manhattan is scipy's `cityblock`; the name is kept as `cityblock` so it matches
the function actually called.
"""
import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
                                    correlation, cosine, euclidean,
                                    sqeuclidean)

from .spec import DISTANCE_ORDER

# The 8 distance metrics, in the paper's Figure 1 / body-text order.
SIMILARITY_LIST = [cosine, euclidean, correlation, chebyshev,
                   braycurtis, canberra, cityblock, sqeuclidean]

SIMILARITY_NAMES = ["cosine", "euclidean", "correlation", "chebyshev",
                    "braycurtis", "canberra", "cityblock", "sqeuclidean"]

assert tuple(SIMILARITY_NAMES) == DISTANCE_ORDER, \
    "distance ordering drifted from lsa.spec.DISTANCE_ORDER"
assert len(SIMILARITY_LIST) == len(SIMILARITY_NAMES) == 8

# Human-readable labels, for tables that should read like the paper's.
DISPLAY_NAMES = {"cosine": "Cosine", "euclidean": "Euclidean",
                 "correlation": "Correlation", "chebyshev": "Chebyshev",
                 "braycurtis": "Braycurtis", "canberra": "Canberra",
                 "cityblock": "Manhattan", "sqeuclidean": "Sqeuclidean"}

EPSILON = 0.00001          # smoothing constant, as in the released code


# ------------------------------------------------- divergence / entropy
def kl_divergence(P, Q):
    P = np.asarray(P, dtype=np.float64) + EPSILON
    Q = np.asarray(Q, dtype=np.float64) + EPSILON
    return float(np.sum(P * np.log(P / Q)))


def js_divergence(P, Q):
    # scipy's jensenshannon returns the DISTANCE (sqrt of the divergence).
    # The released code calls it the same way, so the behaviour is kept.
    return float(distance.jensenshannon(P, Q, 2.0))


def entropy(P):
    P = np.asarray(P, dtype=np.float64) + EPSILON
    return float(-np.sum(P * np.log(P)))


# --------------------------------------------------- pairwise operators
# Table 14 (p18): "Pairwise vector operations ... Note that these operations can
# be applied to nodes' attributes and entropies summarized from posteriors as
# well."
#   Average     (f_i(u) + f_i(v)) / 2        Weighted-L1  |f_i(u) - f_i(v)|
#   Hadamard    f_i(u) . f_i(v)              Weighted-L2  |f_i(u) - f_i(v)|^2
def average(a, b):
    return (a + b) / 2


def hadamard(a, b):
    return a * b


def weighted_l1(a, b):
    return abs(a - b)


def weighted_l2(a, b):
    return abs((a - b) * (a - b))


def concate_all(a, b):
    """All four Table-14 operators concatenated.

    Section 4 applies every operator and concatenates the results, so this is
    the default used for every supervised attack.
    """
    return np.concatenate((average(a, b), hadamard(a, b),
                           weighted_l1(a, b), weighted_l2(a, b)))


OPERATORS = {
    "average": average,
    "hadamard": hadamard,
    "weighted_l1": weighted_l1,
    "weighted_l2": weighted_l2,
    "concate_all": concate_all,
}


def operator_func(operator, a, b):
    if operator not in OPERATORS:
        raise ValueError(f"Unknown operator: {operator}")
    return OPERATORS[operator](np.asarray(a, dtype=np.float64),
                               np.asarray(b, dtype=np.float64))


# ----------------------------------------------------- metric features
def get_metrics(a, b, metric_type, operator):
    """Entropy / divergence summaries of a node pair, folded through `operator`.

    Table 14's note ("these operations can be applied to ... entropies
    summarized from posteriors as well") is what licenses passing the two
    scalars through the same operators as the posteriors. Folding them through
    symmetric operators is also what keeps the pair order irrelevant.

    The released code had a latent bug in the kl/js branches -- `np.array(...)`
    around a Python float instead of `np.array([...])` -- which produced a 0-d
    array and made `np.concatenate` raise, so only `entropy` was usable. The
    scalars are wrapped consistently here; `entropy` reproduces the released
    behaviour exactly.
    """
    if metric_type == "kl_divergence":
        s1 = np.array([kl_divergence(a, b)])
        s2 = np.array([kl_divergence(b, a)])
    elif metric_type == "js_divergence":
        s1 = np.array([js_divergence(a, b)])
        s2 = np.array([js_divergence(b, a)])
    elif metric_type == "entropy":
        s1 = np.array([entropy(a)])
        s2 = np.array([entropy(b)])
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")
    return operator_func(operator, s1, s2)


def pairwise_distances(v0, v1):
    """All 8 distances between two vectors, in DISTANCE_ORDER."""
    v0 = np.asarray(v0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    return np.array([fn(v0, v1) for fn in SIMILARITY_LIST])
