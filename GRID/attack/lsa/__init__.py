"""
A paper-faithful PyTorch implementation of the 8 link stealing attacks from:

    Xinlei He, Jinyuan Jia, Michael Backes, Neil Zhenqiang Gong, Yang Zhang.
    "Stealing Links from Graph Neural Networks."
    30th USENIX Security Symposium (USENIX Security '21), pp. 2669-2686.
    https://www.usenix.org/conference/usenixsecurity21/presentation/he-xinlei

Every experimental choice in this package is traceable to a sentence in the
paper. Where the authors' released TensorFlow code disagrees with the paper,
this package follows the PAPER. See README.md for the clause-by-clause map and
PAPER_SPEC.md for the verbatim quotes each choice is derived from.

The sibling package `lsa_pytorch` reproduces the released code instead, and is
kept so the two can be compared directly.
"""

__all__ = ["data", "models", "features", "attack_model", "attacks", "spec"]
