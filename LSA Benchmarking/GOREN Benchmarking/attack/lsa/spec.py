"""
Single source of truth for every hyperparameter the paper states explicitly.

Each constant carries the paper clause it comes from, so `verify_paper_spec.py`
can assert the code against this module and this module against the paper.

Page numbers refer to the USENIX proceedings PDF (sec21-he-xinlei.pdf).
"""

# ---------------------------------------------------------------- datasets
# p7: "we randomly sample 10% nodes and use their ground truth labels to train
#      the target model and the reference model."
LABEL_RATE = 0.10
LABEL_SPLIT_IS_RANDOM = True

# p7: "we first select all node pairs that are linked. Then, we randomly sample
#      the same number of node pairs that are not linked."
NEGATIVE_SAMPLING_RATIO = 1.0

# p7: "we split the attack dataset randomly by half into attack training dataset
#      and attack testing dataset."
ATTACK_TRAIN_FRACTION = 0.5

# p7: "For the attacks that have a shadow dataset, we also construct an attack
#      dataset on the shadow dataset to train the attack model. Note that we do
#      not split this attack dataset because we do not use it for evaluation."
SHADOW_ATTACK_DATASET_IS_SPLIT = False

# ------------------------------------------------------- target / reference
# p8: "The number of neurons in the hidden layer is set to 16. We adopt the
#      frequently used ReLU and softmax as activation functions for the first
#      hidden layer and the second hidden layer, respectively. Note that we
#      append Dropout (the rate is 0.5) to the output of the hidden layer ...
#      We train 100 epochs with a learning rate of 0.01. Cross-entropy is
#      adopted as the loss function and we use the Adam optimizer ..."
# Footnote 3 points at https://github.com/tkipf/gcn, i.e. the standard 2-weight-
# layer GCN, so the paper's "2 hidden layers" counts the softmax output layer.
TARGET_HIDDEN = 16
TARGET_DROPOUT = 0.5
TARGET_EPOCHS = 100
TARGET_LR = 0.01
TARGET_WEIGHT_LAYERS = 2

# p8: "We use an MLP with 2 hidden layers as the reference model and the shadow
#      reference model. Hyperparameters ... are the same as those of the target
#      model."
REFERENCE_WEIGHT_LAYERS = 2

# p7 footnote 1: "We do not train the reference model for attacks when F is
#                 unavailable."

# ------------------------------------------------------------ attack model
# p8: "We use an MLP with 3 hidden layers as our attack model. The number of
#      neurons for all hidden layers is 32. ReLU is adopted as the activation
#      function for hidden layers and softmax is used as the output activation
#      function. We append Dropout (the rate is 0.5) to each hidden layer to
#      prevent overfitting. We train 50 epochs with a learning rate of 0.001.
#      The loss function is cross-entropy and the optimizer is Adam."
#
# "all hidden layers is 32" excludes the 2-unit output layer from the count,
# and Figure 3's caption ("the attack model's last hidden layer's output
# (32-dimension)") confirms the 32-unit layer is the last HIDDEN layer. So the
# attack model is Dense(32) x 3 -> Dense(2), i.e. 4 weight layers.
ATTACK_HIDDEN_LAYERS = 3
ATTACK_HIDDEN_UNITS = 32
ATTACK_DROPOUT = 0.5
ATTACK_EPOCHS = 50
ATTACK_LR = 0.001
ATTACK_N_CLASSES = 2

# ------------------------------------------------------------- evaluation
# p8: "We run all experiments with this setting for 5 times and report the
#      average value and the standard deviation of AUC scores. Note that for
#      Attack-0 and Attack-2, the AUC scores keep the same since these two
#      attacks are unsupervised."
#
# The second sentence pins down what "5 times" varies: if the 50/50 attack
# split were re-drawn each time, the unsupervised AUCs would move too. So the
# split is drawn ONCE and only the attack model's own randomness is repeated.
N_RUNS = 5
RESPLIT_PER_RUN = False

# p8: "we propose to use clustering, such as K-means. Concretely, we obtain a
#      set of node pairs' distances, and perform K-means on these distances
#      with K being set to 2. The cluster with lower (higher) average distance
#      value is considered as the set of positive (negative) node pairs."
KMEANS_K = 2

# p8, Attack-0 discussion: "Correlation distance achieves the best performance
# followed by Cosine distance across all datasets." Table 15 reports Attack-0's
# Precision/Recall/F1/AUC, and Correlation is the metric it selects.
ATTACK0_SELECTED_METRIC = "correlation"

# p5: "we experimented with 8 common distance metrics: Cosine distance,
#      Euclidean distance, Correlation distance, Chebyshev distance, Braycurtis
#      distance, Canberra distance, Manhattan distance, and Square-euclidean
#      distance."
# This ordering is also the legend order of Figure 1 (p8). Appendix Table 13
# (p18) lists Manhattan before Canberra; that table only gives formulas, so the
# discrepancy is presentational. We follow the body text / Figure 1.
DISTANCE_ORDER = ("cosine", "euclidean", "correlation", "chebyshev",
                  "braycurtis", "canberra", "cityblock", "sqeuclidean")

# Table 14 (p18): Average, Hadamard, Weighted-L1, Weighted-L2.
PAIRWISE_OPERATORS = ("average", "hadamard", "weighted_l1", "weighted_l2")
