"""
The traditional link-prediction baseline the paper compares against.

Paper clause (p12):

    "We further compare all our attacks with a traditional link prediction
     method. More specifically, we build an MLP with features summarized from
     the target model's partial graph, including Common neighbor, Jaccard index,
     and Preferential attachment. As we can see from Figure 6, most of our
     attacks outperforms the link prediction method. For instance, on the COX2
     dataset, all our 8 attacks outperform the link prediction model, the best
     attack (Attack-6) achieves more than 20% performance gain."

Three heuristics, computed for a node pair (u, v) from the neighbour sets
Gamma(u), Gamma(v):

    Common neighbour        |Gamma(u) INTERSECT Gamma(v)|
    Jaccard index           |Gamma(u) INTERSECT Gamma(v)| / |Gamma(u) UNION Gamma(v)|
    Preferential attachment |Gamma(u)| . |Gamma(v)|

The important constraint is that this baseline gets the SAME knowledge as the
attacks with A*: the neighbour sets must come from the partial graph only, i.e.
the edges in the attack TRAINING half. Building them from the full graph would
let the baseline read the answer for the test pairs straight out of its own
features -- a common-neighbour count computed on the full graph already knows
whether u and v are adjacent. The adjacency used here is therefore reconstructed
from the training-half positive pairs alone.

The classifier is the same MLP as the attacks (paper p8: 3 hidden layers of 32),
so the comparison isolates the features rather than the model.
"""
import numpy as np

from .attack_model import (DEFAULT_BATCH_SIZE, scale_features,
                           train_and_evaluate)
from .spec import ATTACK_EPOCHS, ATTACK_HIDDEN_LAYERS

FEATURE_NAMES = ("common_neighbour", "jaccard_index", "preferential_attachment")


def neighbours_from_rows(rows):
    """Neighbour sets of the partial graph A*, from positive training pairs.

    Only rows with label 1 contribute an edge. This is exactly the knowledge
    A* gives the adversary.
    """
    nbr = {}
    n_edges = 0
    for r in rows:
        if r["label"] != 1:
            continue
        u, v = r["id_pair"]
        nbr.setdefault(u, set()).add(v)
        nbr.setdefault(v, set()).add(u)
        n_edges += 1
    return nbr, n_edges


def pair_features(u, v, nbr):
    gu, gv = nbr.get(u, ()), nbr.get(v, ())
    su, sv = set(gu), set(gv)
    inter = len(su & sv)
    union = len(su | sv)
    return np.array([
        float(inter),                                  # common neighbour
        inter / union if union else 0.0,               # Jaccard index
        float(len(su) * len(sv)),                      # preferential attachment
    ])


def build_xy(rows, nbr):
    x = np.array([pair_features(*r["id_pair"], nbr) for r in rows],
                 dtype=np.float32)
    y = np.array([r["label"] for r in rows])
    return x, y


def link_prediction_baseline(train_rows, test_rows, *, runs=5,
                             epochs=ATTACK_EPOCHS,
                             batch_size=DEFAULT_BATCH_SIZE,
                             hidden_layers=ATTACK_HIDDEN_LAYERS,
                             device="cpu", seed=42, independent_test_fit=True,
                             verbose=True):
    """Train the 3-feature MLP baseline and score it on the attack test half.

    `train_rows` must be the TARGET's attack training half, since the partial
    graph is what this method is given.
    """
    nbr, n_edges = neighbours_from_rows(train_rows)
    if verbose:
        print(f"    partial graph: {n_edges} edges over {len(nbr)} nodes")

    x_train, y_train = build_xy(train_rows, nbr)
    x_test, y_test = build_xy(test_rows, nbr)
    x_train, x_test = scale_features(x_train, x_test, independent_test_fit)

    out = []
    for run in range(runs):
        out.append(train_and_evaluate(
            x_train, y_train, x_test, y_test, epochs=epochs,
            batch_size=batch_size, device=device, seed=seed + run,
            verbose=False, hidden_layers=hidden_layers))
    res = {}
    for k in ("accuracy", "precision", "recall", "f1", "auc"):
        vals = [r[k] for r in out]
        res[k] = float(np.mean(vals))
        res[f"{k}_std"] = float(np.std(vals))
        res[f"{k}_runs"] = [float(v) for v in vals]
    res.update({"runs": runs, "n_features": len(FEATURE_NAMES),
                "partial_graph_edges": n_edges,
                "feature_names": list(FEATURE_NAMES)})
    return res
