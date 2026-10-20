"""Build the inductive member / non-member split.

Plain uniform-random: a fraction (default 20%) of nodes are drawn at random and
become non-members (V_out). Their nodes and all their incident edges are removed;
the remaining 80% form V_in (members), whose induced subgraph is what every
reduction method is later applied to.

No cascade handling and no isolation guard: a uniform V_out is degree-representative,
so members and non-members share the same degree distribution and a membership attack
cannot separate them by degree alone. A few nodes left isolated inside V_in are
harmless to train on.
"""
import numpy as np
import scipy.sparse as sp


def make_split(adj, seed, frac):
    """Return member/non-member ids over the original node set.

    member_ids and nonmember_ids are sorted ascending; member_mask is over all n.
    """
    n = adj.shape[0]
    k = int(round(frac * n))
    perm = np.random.RandomState(seed).permutation(n)
    nonmember = np.sort(perm[:k])
    member = np.sort(perm[k:])
    member_mask = np.ones(n, dtype=bool)
    member_mask[nonmember] = False
    return member, nonmember, member_mask


def induced_subgraph(adj, features, labels, member_ids):
    """V_in: edges among members only (non-members and their edges dropped),
    relabelled 0..m-1 in ascending original-id order."""
    vin_adj = adj[member_ids][:, member_ids].tocsr()
    vin_adj.eliminate_zeros()
    vin_feat = features[member_ids].tocsr()
    vin_labels = np.asarray(labels)[member_ids]
    return vin_adj, vin_feat, vin_labels
