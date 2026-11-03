"""
The Superhuman Fairness baseline.

This is a port of the reference implementation published alongside
"Superhuman Fairness" (Memarrast, Vu, Ziebart; ICML 2023):

    https://github.com/omidMemari/superhumn-fairness

specifically of `Super_human.update_model()` in that repository's `main.py`
and of every method it calls (`sample_superhuman()`,
`get_samples_demo_indexed()`, `get_sample_loss()`, `compute_exp_phi_X_Y()`,
`feature_matching()`, `compute_grad_theta()`, `compute_alpha()`,
`eval_model()`).

The algorithm itself is unchanged. Only the three things that have to change
for it to run inside this project are different:

1.  Data plumbing. The original writes its dataset splits, demonstrations and
    models to CSV/pickle files and reads them back. This version is handed the
    data in memory by `frwo.experiment_utils`, so that the baseline sees
    exactly the dataset, the label bias and the train/validation/test split
    that the Fairness Reward Weight Optimization technique sees.

2.  Feature encoding. The original operates on an already-numeric
    `dataset_ref.csv`. This version runs the project's own
    `sklearn_clf_pipeline()` preprocessing to turn the project's mixed
    boolean/categoric/continuous columns into the numeric design matrix its
    logistic regression base model needs. The protected attribute `z` stays in
    the design matrix, exactly as the sensitive column stays in the original's
    `X`.

3.  Vectorization. Three of the original's loops compute nothing but a
    matrix-vector product or an element-wise maximum, and are written as such
    here: per-row sampling (`sample_superhuman()`), the `phi(X, Y)` sums
    (`compute_exp_phi_X_Y()`/`feature_matching()`), and the inner
    per-objective loop of `compute_grad_theta()`. These produce the same
    values as the original loops; they just make the baseline tractable on
    datasets of this project's size. See the comments at each site.

Two quirks of the reference implementation are reproduced deliberately, and
are flagged where they occur:

* `compute_exp_phi_X_Y()` normalizes `phi(X, Y)` by the size of the whole
  training pool while `feature_matching()` normalizes it by the size of one
  demonstration, so the two differ by a constant factor.
* `Super_human.gamma_superhuman` is initialized to zeros and never updated,
  which makes the first of the two early-stopping tests unreachable.

Both are left as they are upstream: reproducing the published baseline
faithfully matters more than fixing it.

The first quirk has a large effect, though. With each demonstration covering
half the pool, it makes `phi_j - E[phi]` equal to half the mean of `phi` --
the same for every demonstration -- plus the demonstration-specific signal,
and the shared half dominates the gradient. Since the subdominance weights
multiplying it sum to a positive number, every step pushes all of the logits
down, so the model drifts toward predicting every label 0 rather than
learning from the demonstrations. `fix_normalization=True` (the "Superhuman
Fairness Fixed" technique) normalizes `E[phi]` like `phi_j`, which leaves only
the demonstration-specific signal. See `_compute_feature_matching()`.

The `NN` base model is ported separately, from the reference implementation's
`reorg_current` branch, in `frwo.sh.superhuman_fairness_nn`.

Not ported: the original's own demonstrators (`-b pp` and `-b fair_logloss`),
since the demonstrations here are this project's expert's (see
`build_expert_demo_list()`), and its COMPAS-only initialization of `theta`
from a fair log-loss classifier.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_negative_rate_difference,
    false_positive_rate_difference,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, zero_one_loss
from sklearn.model_selection import train_test_split

from frwo.rl.objectives import OBJ_LOOKUP_BY_NAME, ObjectiveSet
from frwo.utils import sklearn_clf_pipeline

# The original's `logi_params` (main.py). `penalty="l2"` is spelled here as
# `l1_ratio=0.0`, which is the same L2 penalty in every scikit-learn version
# this project supports, without the `penalty` deprecation warning that
# scikit-learn >= 1.8 raises.
SH_DEFAULT_LOGI_PARAMS = {
    "C": 100,
    "l1_ratio": 0.0,
    "solver": "newton-cg",
    "max_iter": 1000,
}

# The original's defaults, from `default_args` and the module-level constants
# of main.py.
SH_DEFAULT_LR_THETA = 0.01
SH_DEFAULT_ITERS = 30
SH_DEFAULT_LAMDA = 0.001
# `alpha` in main.py: the fraction of the data the base model is fit on.
SH_BASE_MODEL_TRAIN_FRAC = 0.5


def compute_alphas(raw_demos_feat_loss, clf_demos_feat_loss, lamda=SH_DEFAULT_LAMDA):
    """
    Compute the per-objective `alpha` scaling of the subdominance measure.

    Original code taken from
    https://github.com/omidMemari/superhumn-fairness/blob/main/main.py#L672
    (`Super_human.compute_alpha()`), with the demonstration and sample losses
    passed in as arrays instead of read off `self`.

    Parameters
    ----------
    raw_demos_feat_loss : numpy.ndarray, shape (n_demos, n_features)
        The feature losses of the reference (expert/demonstration) decisions.
    clf_demos_feat_loss : numpy.ndarray, shape (n_demos, n_features)
        The feature losses of the classifier being scored, on the same demos.
    lamda : float, default 0.001
        The original's `self.lamda`.

    Returns
    -------
    alphas : numpy.ndarray, shape (n_features,)
    """
    alphas = np.ones(raw_demos_feat_loss.shape[1])

    for k in range(raw_demos_feat_loss.shape[1]):  # for each feature
        sorted_demos = []
        for j in range(raw_demos_feat_loss.shape[0]):  # for each demo
            sample_loss = clf_demos_feat_loss[j][
                k
            ]  # Contains the various feature expectations for each classifier demo
            demo_loss = raw_demos_feat_loss[j][
                k
            ]  # Contains the various feature expectations for each human demo
            sorted_demos.append((demo_loss, sample_loss))

        sorted_demos.sort(
            key=lambda x: x[0]
        )  # dominated_demos.sort(key = lambda x: x[0], reverse=True)   # sort based on demo loss
        sorted_demos = np.array(sorted_demos)
        alphas[k] = (
            100  # max(self.alpha) #np.mean(self.alpha) # default value in case it didn't change using previous alpha values
        )
        for m, demo in enumerate(sorted_demos):
            if demo[0] > demo[1]:
                alphas[k] = min(
                    100, 1.0 / (demo[0] - demo[1])
                )  ### limit max alpha to 100
            if (demo[1] + lamda) <= np.mean(
                [x[0] for x in sorted_demos[0 : m + 1]]
            ):  # if (demo[2]) <= np.mean([x[1] for x in dominated_demos[0:m+1]] and demo[0] > 0):
                break

    return alphas


##
# Feature (loss) definitions.
#
# The "features" of the Superhuman Fairness algorithm are the performance and
# fairness measures it tries to beat the demonstrations on, expressed as
# losses where lower is better -- the `-f` flag of the original's CLI.
#
# Two vocabularies are accepted:
#
#   * This project's objective names (`OBJ_LOOKUP_BY_NAME`, e.g. "Acc",
#     "DemPar"), inverted to losses the same way `compute_relevant_feat_loss()`
#     inverts them for the subdominance metric. Using these makes the baseline
#     optimize exactly what both techniques are then evaluated on.
#   * The original paper's metric names ("inacc", "dp", "eqodds", "prp", ...),
#     which reproduce `util.get_metrics_df()` of the reference implementation.
#
# A configuration may mix the two; names are resolved per entry.
##


def _positive_predictive_value(y_true, y_pred):
    """`util.positive_predictive_value_helper()` of the reference repo."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    if tp == 0 and fp == 0:
        return 0
    return tp / (tp + fp)


def _negative_predictive_value(y_true, y_pred):
    """`util.negative_predictive_value_helper()` of the reference repo."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    if tn == 0 and fn == 0:
        return 0
    return tn / (tn + fn)


def _between_group_difference(metric_fn, demo):
    return MetricFrame(
        metrics=metric_fn,
        y_true=demo["y"],
        y_pred=demo["yhat"],
        sensitive_features=demo["z"],
    ).difference(method="between_groups")


def _predictive_value_difference(demo):
    """`util.predictive_value()`: the larger of the PPV and NPV differences."""
    prp = MetricFrame(
        metrics={
            "ppv": _positive_predictive_value,
            "npv": _negative_predictive_value,
        },
        y_true=demo["y"],
        y_pred=demo["yhat"],
        sensitive_features=demo["z"],
    )
    return max(prp.difference(method="between_groups"))


# The reference repo's `feature_expand_dict` (util.py), as functions of a
# demonstration frame with `y`, `yhat` and `z` columns.
SH_PAPER_FEATURE_LOSSES = {
    "inacc": lambda demo: zero_one_loss(demo["y"], demo["yhat"]),
    "dp": lambda demo: demographic_parity_difference(
        demo["y"], demo["yhat"], sensitive_features=demo["z"]
    ),
    "eqodds": lambda demo: equalized_odds_difference(
        demo["y"], demo["yhat"], sensitive_features=demo["z"]
    ),
    "prp": _predictive_value_difference,
    "eqopp": lambda demo: false_negative_rate_difference(
        demo["y"], demo["yhat"], sensitive_features=demo["z"]
    ),
    "fnr": lambda demo: false_negative_rate_difference(
        demo["y"], demo["yhat"], sensitive_features=demo["z"]
    ),
    "fpr": lambda demo: false_positive_rate_difference(
        demo["y"], demo["yhat"], sensitive_features=demo["z"]
    ),
    "ppv": lambda demo: _between_group_difference(_positive_predictive_value, demo),
    "npv": lambda demo: _between_group_difference(_negative_predictive_value, demo),
    "error_rate_diff": lambda demo: _between_group_difference(
        balanced_accuracy_score, demo
    ),
}


def objective_feature_losses(demo, obj_names):
    """
    Compute this project's objectives on a demonstration, as losses.

    The objectives are "goodness" measures in [0, 1] (e.g. accuracy, or
    `1 - |demographic parity difference|`), so they are inverted with
    `-mu + 1` to get losses where lower is better. This is the single
    definition of "feature loss" that both the subdominance metric
    (`compute_relevant_feat_loss()`) and this baseline use, so that the two
    techniques are scored on identical quantities.

    Parameters
    ----------
    demo : pandas.DataFrame
        Demonstration frame with at least `y`, `yhat` and `z` columns.
    obj_names : sequence<str>
        Keys of `OBJ_LOOKUP_BY_NAME`.

    Returns
    -------
    losses : numpy.ndarray, shape (len(obj_names),)
    """
    obj_set = ObjectiveSet([OBJ_LOOKUP_BY_NAME[name]() for name in obj_names])
    return -np.array(obj_set.compute_demo_feature_exp(demo)) + 1


def validate_feature_names(feature_names, context="feature"):
    """
    Raise if any name is neither one of this project's objectives nor one of
    the original paper's metrics.
    """
    unknown = [
        name
        for name in feature_names
        if name not in OBJ_LOOKUP_BY_NAME and name not in SH_PAPER_FEATURE_LOSSES
    ]
    if unknown:
        raise ValueError(
            f"Unrecognized {context} name(s): {unknown}."
            f" Valid names are this project's objectives"
            f" {sorted(OBJ_LOOKUP_BY_NAME)} and the original paper's metrics"
            f" {sorted(SH_PAPER_FEATURE_LOSSES)}."
        )


def feature_losses_from_demo(demo, feature_names):
    """
    Compute a demonstration's loss on each named performance/fairness measure.

    This is the one definition of "feature loss" in the project: the
    subdominance metric (`compute_relevant_feat_loss()`), the Superhuman
    Fairness baseline and the expert demonstrations recorded for plotting all
    come through here, so every one of them is measuring the same quantity.

    Two vocabularies are accepted, and may be mixed:

    * This project's objective names (`OBJ_LOOKUP_BY_NAME`, e.g. `"Acc"`,
      `"DemPar"`). These are "goodness" measures in [0, 1], so they are
      inverted with `-mu + 1` to get a loss where lower is better.
    * The original paper's metric names (`SH_PAPER_FEATURE_LOSSES`, e.g.
      `"inacc"`, `"dp"`, `"prp"`), which are already losses, and which
      reproduce `util.get_metrics_df()` of the reference implementation.

    Parameters
    ----------
    demo : pandas.DataFrame
        Demonstration frame with at least `y`, `yhat` and `z` columns.
    feature_names : sequence<str>
        The measures to compute, in the order they are returned.

    Returns
    -------
    losses : numpy.ndarray, shape (len(feature_names),)
    """
    validate_feature_names(feature_names)

    # This project's objectives are computed in one batch, since an
    # ObjectiveSet evaluates all of its objectives in a single pass.
    obj_names = [name for name in feature_names if name in OBJ_LOOKUP_BY_NAME]
    obj_losses = (
        dict(zip(obj_names, objective_feature_losses(demo, obj_names)))
        if obj_names
        else {}
    )

    return np.array(
        [
            (
                obj_losses[name]
                if name in obj_losses
                else SH_PAPER_FEATURE_LOSSES[name](demo)
            )
            for name in feature_names
        ],
        dtype=float,
    )


def make_feature_loss_fn(feature_names):
    """
    Build the loss function the baseline optimizes and is measured with.

    Parameters
    ----------
    feature_names : sequence<str>
        Each entry is either one of this project's objective names
        (`OBJ_LOOKUP_BY_NAME`) or one of the original paper's metric names
        (`SH_PAPER_FEATURE_LOSSES`).

    Returns
    -------
    loss_fn : callable(y_true, y_pred, z) -> numpy.ndarray
        The losses of the given decisions, in the order of `feature_names`.
    """
    validate_feature_names(feature_names, context="Superhuman Fairness feature")

    def loss_fn(y_true, y_pred, z):
        # Every objective and every paper metric is a function of `y`, `yhat`
        # and `z` alone, so the demo frame only has to carry those three
        # columns. `yhat` is cast the way `generate_demo()` casts it, so that
        # sampled decisions (which are floats) score identically to the
        # evaluation pipeline's predictions.
        demo = pd.DataFrame(
            {
                "y": np.asarray(y_true).astype(np.int64),
                "yhat": np.asarray(y_pred).astype(np.int64),
                "z": np.asarray(z).astype(np.int64),
            }
        )

        return feature_losses_from_demo(demo, feature_names)

    return loss_fn


@dataclass
class SuperhumanDemo:
    """
    One reference decision-maker ("demonstration") the baseline imitates.

    Corresponds to one entry of `Super_human.demo_list` in the reference
    implementation: a set of held-out rows the demonstrator made decisions on,
    and the losses those decisions incurred.

    Attributes
    ----------
    idx : numpy.ndarray<int>
        Positional indices, into the baseline's training pool, of the rows this
        demonstration covers. The original's `data_demo.idx_test`.
    metric : numpy.ndarray<float>
        The demonstration's loss on each feature, in the order of the
        configured feature names. The original's `data_demo.metric`.
    """

    idx: np.ndarray
    metric: np.ndarray


@dataclass
class SuperhumanTrainingHistory:
    """
    Per-iteration diagnostics of one `SuperhumanFairness.fit()` call.

    These are the quantities the reference implementation prints and pickles
    while training (`subdom_tensor_sum_arr`, `self.eval`,
    `self.gamma_superhuman_arr`, `self.alpha`). They are recorded so that the
    experiment can report the baseline's training curve to W&B.

    Attributes
    ----------
    subdom_sum : list<float>
        The summed subdominance tensor of each iteration.
    feature_loss : list<numpy.ndarray>
        The model's loss on each feature, on the training pool, per iteration.
    gamma_superhuman : list<numpy.ndarray>
        The fraction of demonstrations the model matched or beat, per feature,
        per iteration.
    alphas : list<numpy.ndarray>
        The `alpha` values of each iteration.
    n_iterations : int
        How many iterations actually ran before the early stopping tests fired.
    learning_rate : list<float>
        The optimizer learning rate each iteration's step was taken with, for
        models whose learning rate changes during training (the neural network
        variant). Empty otherwise.
    """

    subdom_sum: list = field(default_factory=list)
    feature_loss: list = field(default_factory=list)
    gamma_superhuman: list = field(default_factory=list)
    alphas: list = field(default_factory=list)
    n_iterations: int = 0
    learning_rate: list = field(default_factory=list)


class SuperhumanFairness:
    """
    The Superhuman Fairness classifier.

    Learns a logistic regression whose sampled decisions dominate a set of
    reference decisions on every configured performance/fairness measure as
    often as possible, by descending the subdominance of its samples against
    those references. This is `Super_human` of the reference implementation,
    restricted to its `LR` base model.

    Exposes `predict()` so that the rest of this project's pipeline --
    `generate_demo()`, `_evaluate_policy()`, the subdominance metric -- can
    treat it exactly like a learned FRWO policy.

    Parameters
    ----------
    feature_types : dict<str, list>
        Mapping of column names to their type of feature, used to build the
        project's preprocessing pipeline.
    feature_names : sequence<str>
        The performance/fairness measures to optimize. See
        `make_feature_loss_fn()`.
    loss_fn : callable(y_true, y_pred, z) -> numpy.ndarray
        The loss function for those measures, from `make_feature_loss_fn()`.
    lr_theta : float, default 0.01
        Gradient step size on `theta`. The original's `lr_theta`.
    iters : int, default 30
        Maximum number of gradient iterations. The original's `iters`.
    lamda : float, default 0.001
        The `alpha` search tolerance. The original's `lamda`.
    logi_params : dict, Optional
        Parameters of the logistic regression base model. Defaults to
        `SH_DEFAULT_LOGI_PARAMS`, the original's `logi_params`.
    rng : numpy.random.Generator, Optional
        Source of randomness for the half of the data the base model is fit
        on and for decision sampling. A dedicated generator is used (rather than the global numpy random state) so that enabling this
        baseline cannot shift the random draws of the FRWO technique running
        alongside it.
    fix_normalization : bool, default False
        Normalize `E[phi(X, Y)]` by each demonstration's size, like
        `phi(X, Y)` itself, instead of by the whole pool's size as upstream
        does. See `_compute_feature_matching()`.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        Preprocessing plus the logistic regression whose `coef_` is `theta`.
    threshold_ : float
        The decision threshold, `np.mean(Y_train)`, as in `eval_model()`.
    alpha_ : numpy.ndarray
        The current per-feature `alpha`.
    history_ : SuperhumanTrainingHistory
        Per-iteration diagnostics of the last `fit()`.
    selected_iteration_ : int
        The index, into `history_`, of the iteration whose model `fit()`
        returns: always the last one here, as upstream keeps training `theta`
        in place.
    """

    def __init__(
        self,
        feature_types,
        feature_names,
        loss_fn,
        lr_theta=SH_DEFAULT_LR_THETA,
        iters=SH_DEFAULT_ITERS,
        lamda=SH_DEFAULT_LAMDA,
        logi_params=None,
        rng=None,
        fix_normalization=False,
    ):
        self.feature_types = feature_types
        self.feature_names = list(feature_names)
        self.num_of_features = len(self.feature_names)
        self.loss_fn = loss_fn
        self.lr_theta = lr_theta
        self.iters = iters
        self.lamda = lamda
        self.logi_params = dict(logi_params or SH_DEFAULT_LOGI_PARAMS)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.fix_normalization = fix_normalization

        self.pipeline_ = None
        self.threshold_ = None
        self.alpha_ = np.array([1.0 for _ in range(self.num_of_features)])
        # `Super_human.gamma_superhuman`. Upstream initializes it to zeros and
        # never assigns to it again, which makes the "was superhuman on every
        # feature last iteration and no longer is" stopping test below
        # unreachable. Kept as-is so the stopping behavior matches.
        self.gamma_superhuman_ = np.zeros(self.num_of_features)
        self.history_ = SuperhumanTrainingHistory()
        self.selected_iteration_ = None

    ##
    # Training
    ##

    def fit(self, X, y, demo_list):
        """
        Learn the classifier.

        Parameters
        ----------
        X : pandas.DataFrame
            The training pool's input columns. The rows `demo.idx` index into.
        y : pandas.Series
            The training pool's labels, aligned with `X`.
        demo_list : list<SuperhumanDemo>
            The reference decisions to dominate.

        Returns
        -------
        self
        """
        self.X_ = X
        self.y_ = np.asarray(y).astype(np.int64)
        self.z_ = np.asarray(X["z"]).astype(np.int64)
        self.demo_list_ = demo_list
        self.num_of_demos_ = len(demo_list)
        self.demo_losses_ = np.array([demo.metric for demo in demo_list], dtype=float)

        self._fit_base_model(X, y)

        # The design matrix the logistic regression actually sees. Cached
        # because both the per-iteration sampling and the phi(X, Y) sums run
        # over the whole pool on every iteration; the original re-reads its
        # training CSV instead.
        self.design_ = np.asarray(
            self.pipeline_.named_steps["preprocessor"].transform(X), dtype=float
        )
        self.num_of_attributs_ = self.design_.shape[1]

        self._update_model()

        return self

    def _fit_base_model(self, X, y):
        """
        Fit the initial logistic regression, i.e. the starting `theta`.

        `Super_human.base_model()`: a logistic regression fit on a stratified
        half of the training pool.
        """
        X_base, _, y_base, _ = train_test_split(
            X,
            y,
            test_size=1 - SH_BASE_MODEL_TRAIN_FRAC,
            # The original hard-codes 12345; drawn from this model's generator
            # instead, so that the half differs between trials and restarts.
            random_state=int(self.rng.integers(0, 2**31 - 1)),
            stratify=y,
        )

        self.pipeline_ = sklearn_clf_pipeline(
            feature_types=self.feature_types,
            clf_inst=LogisticRegression(**self.logi_params),
        )
        self.pipeline_.fit(X_base, y_base)

        # `eval_model()` thresholds at the mean of the training labels.
        self.threshold_ = float(np.mean(np.asarray(y)))

    def _get_model_theta(self):
        return self.pipeline_.named_steps["classifier"].coef_[0]

    def _update_model_theta(self, new_theta):
        self.pipeline_.named_steps["classifier"].coef_ = np.asarray([new_theta])

    def _sample_superhuman(self):
        """
        Draw one set of decisions per demonstration from the current model.

        `Super_human.sample_superhuman()` plus `sample_from_prob()`. The
        original loops over the rows of the training pool, calling
        `predict_proba()` on one row at a time and then
        `np.random.choice([0.0, 1.0], num_of_demos, True, [p0, p1])`. That is
        `num_of_demos` independent Bernoulli(p1) draws per row, which is what
        the vectorized form below produces, in one `predict_proba()` call
        instead of one per row.

        Returns
        -------
        sample_matrix : numpy.ndarray, shape (num_of_demos, n_pool)
        """
        start_time = time.time()

        p1 = self.pipeline_.named_steps["classifier"].predict_proba(self.design_)[:, 1]
        sample_matrix = (
            self.rng.random((self.num_of_demos_, len(p1))) < p1[None, :]
        ).astype(float)

        logging.debug(
            f"\t\t--- {time.time() - start_time}s end of sample_superhuman ---"
        )
        return sample_matrix

    def _get_samples_demo_indexed(self, sample_matrix):
        """
        Restrict demonstration `i`'s samples to the rows it covers.

        `Super_human.get_samples_demo_indexed()`.
        """
        return [
            sample_matrix[i, :][self.demo_list_[i].idx]
            for i in range(self.num_of_demos_)
        ]

    def _get_sample_loss(self, samples_demo_indexed):
        """
        Score each demonstration's samples on every feature.

        `Super_human.get_sample_loss()`. As upstream, the labels used are the
        training pool's own, not any (possibly noisy) copy the demonstration
        carries.

        Returns
        -------
        sample_loss : numpy.ndarray, shape (num_of_demos, num_of_features)
        """
        start_time = time.time()

        sample_loss = np.zeros((self.num_of_demos_, self.num_of_features))
        for demo_index, demo in enumerate(self.demo_list_):
            sample_loss[demo_index, :] = self.loss_fn(
                self.y_[demo.idx],
                samples_demo_indexed[demo_index],
                self.z_[demo.idx],
            )

        logging.debug(f"\t\t--- {time.time() - start_time}s end of get_sample_loss ---")
        return sample_loss

    def _compute_feature_matching(self, samples_demo_indexed):
        """
        The `phi(X, Y) - E[phi(X, Y)]` term of the gradient, per demonstration.

        `Super_human.compute_exp_phi_X_Y()` and `feature_matching()`. Both
        compute `sum_i(Y_i * X_i)` over a demonstration's rows, which is the
        matrix-vector product written here.

        The two normalize that sum differently -- `compute_exp_phi_X_Y()`
        divides by the size of the whole training pool, `feature_matching()`
        by the size of the demonstration -- so the expectation subtracted here
        is on a different scale than the term it is subtracted from. That is
        how the reference implementation computes it, so it is reproduced
        exactly by default.

        With `fix_normalization`, the expectation is normalized by each
        demonstration's size as well, making it the mean of the `phi(X, Y)`
        terms themselves. Each row returned is then only that demonstration's
        deviation from the mean, i.e. the demonstration-specific signal,
        without the component shared by every demonstration that the
        mismatched normalization adds.

        Returns
        -------
        feature_matching : numpy.ndarray, shape (num_of_demos, n_attributes)
        """
        n_pool = self.design_.shape[0]

        # compute_exp_phi_X_Y(): normalized by the whole pool's size, or with
        # `fix_normalization` by the demonstration's size, like phi_X_Y.
        exp_phi_X_Y = np.zeros(self.num_of_attributs_)
        # feature_matching(): normalized by the demonstration's size.
        phi_X_Y = np.zeros((self.num_of_demos_, self.num_of_attributs_))

        for i, demo in enumerate(self.demo_list_):
            design_demo = self.design_[demo.idx]
            weighted_sum = design_demo.T @ samples_demo_indexed[i]
            exp_phi_X_Y += weighted_sum / (
                design_demo.shape[0] if self.fix_normalization else n_pool
            )
            phi_X_Y[i] = weighted_sum / design_demo.shape[0]

        exp_phi_X_Y /= self.num_of_demos_

        return phi_X_Y - exp_phi_X_Y

    def _compute_grad_theta(self, sample_loss, samples_demo_indexed):
        """
        The subdominance tensor and the gradient of the loss w.r.t. `theta`.

        `Super_human.compute_grad_theta()`, with two changes that do not affect
        the values produced:

        * The per-demonstration `feature_matching()` term is computed once per
          demonstration instead of once per (demonstration, feature). Upstream
          recomputes the same vector inside the inner loop; the subdominance
          constant only changes between demonstrations, so summing the row's
          subdominance and scaling the term once is equivalent.
        * The inner per-feature loop is written as an element-wise maximum.

        The subdominance constant is still the mean of the partially filled
        tensor as of the start of each demonstration, zeros included, which is
        what `get_subdom_constant()` returns (its `self.c` is never set, so the
        mean is recomputed every time).

        Returns
        -------
        subdom_tensor_sum : float
        grad_theta : numpy.ndarray, shape (n_attributes,)
        """
        start_time = time.time()

        feature_matching = self._compute_feature_matching(samples_demo_indexed)

        subdom_tensor = np.zeros((self.num_of_demos_, self.num_of_features))
        grad_theta = np.zeros(self.num_of_attributs_)

        for j in range(self.num_of_demos_):
            if j == 0:
                subdom_constant = 0.0
            else:
                subdom_constant = np.mean(subdom_tensor)

            # subtract constant c to optimize for useful demonstation instead
            # of avoiding from noisy ones
            subdom_tensor[j, :] = (
                np.maximum(
                    self.alpha_ * (sample_loss[j, :] - self.demo_losses_[j, :]) + 1,
                    0,
                )
                - subdom_constant
            )
            grad_theta += subdom_tensor[j, :].sum() * feature_matching[j]

        subdom_tensor_sum = np.sum(subdom_tensor)
        logging.debug(
            f"\t\t--- {time.time() - start_time}s end of compute_grad_theta ---"
        )
        logging.debug(f"\t\tsubdom tensor sum: {subdom_tensor_sum}")

        return subdom_tensor_sum, grad_theta

    def _eval_model(self):
        """
        The model's loss on each feature, over the training pool.

        `Super_human.eval_model(mode="train")`.
        """
        return self.loss_fn(self.y_, self._predict_pool(), self.z_)

    def _predict_pool(self):
        scores = self.pipeline_.named_steps["classifier"].predict_proba(self.design_)[
            :, 1
        ]
        return (scores >= self.threshold_) * 1

    def _find_gamma_superhuman(self, model_loss):
        """
        Per feature, the fraction of demonstrations the model matches or beats.

        `util.find_gamma_superhuman()`.
        """
        return np.array(
            [
                np.mean(model_loss[i] <= self.demo_losses_[:, i])
                for i in range(self.num_of_features)
            ]
        )

    def _update_model(self):
        """
        The training loop.

        `Super_human.update_model()`, restricted to its `LR` base model branch.
        """
        gamma_superhuman_arr = []
        gamma_degrade = 0

        for i in range(self.iters):
            # find sample loss and store it, we will use it for computing
            # grad_theta and grad_alpha
            sample_matrix = self._sample_superhuman()
            samples_demo_indexed = self._get_samples_demo_indexed(sample_matrix)
            sample_loss = self._get_sample_loss(samples_demo_indexed)

            # get the current theta
            theta = self._get_model_theta()

            # computer gradient of loss w.r.t theta by sampling from our model
            subdom_tensor_sum, grad_theta = self._compute_grad_theta(
                sample_loss, samples_demo_indexed
            )
            # update theta using the gradinet values
            new_theta = theta - self.lr_theta * grad_theta

            # find new alpha
            new_alpha = compute_alphas(self.demo_losses_, sample_loss, self.lamda)

            self._update_model_theta(new_theta)
            self.alpha_ = new_alpha

            # eval model
            model_loss = self._eval_model()
            gamma_superhuman = self._find_gamma_superhuman(model_loss)
            gamma_superhuman_arr.append(gamma_superhuman)

            self.history_.subdom_sum.append(float(subdom_tensor_sum))
            self.history_.feature_loss.append(model_loss)
            self.history_.gamma_superhuman.append(gamma_superhuman)
            self.history_.alphas.append(new_alpha)
            self.history_.n_iterations = i + 1
            self.selected_iteration_ = i

            logging.info(
                f"\t\t SH iter {i + 1}/{self.iters}:"
                f" subdom_sum={subdom_tensor_sum:.5f},"
                f" gamma_superhuman={np.round(gamma_superhuman, 3)},"
                f" loss={np.round(model_loss, 4)}"
            )

            # if last iter every feature was 1-superhuman and this iter
            # changes --> break
            if (
                sum(self.gamma_superhuman_) == self.num_of_features
                and sum(gamma_superhuman) < self.num_of_features
            ):
                break
            # look back if it has improved for the last 3 iterations, if
            # not --> break
            if len(gamma_superhuman_arr) > 10 and sum(gamma_superhuman) < sum(
                gamma_superhuman_arr[-2]
            ):
                gamma_degrade += 1
            else:
                gamma_degrade = 0
            # peformance degrades for 3 cosecutive iterations.
            if gamma_degrade == 3:
                break

    ##
    # Prediction
    ##

    def predict(self, X, y=None):
        """
        Predict labels for `X`.

        `Super_human.eval_model()`: the positive-class score thresholded at the
        mean of the training labels.

        `y` is accepted and ignored, so that this classifier is callable
        exactly like a `ClassificationMDPPolicy` from the evaluation pipeline.
        """
        scores = self.pipeline_.predict_proba(X)[:, 1]
        return (scores >= self.threshold_) * 1

    def predict_proba(self, X):
        return self.pipeline_.predict_proba(X)


##
# Demonstration sources
##


def build_expert_demo_list(demo_df, group_idxs, loss_fn):
    """
    Build the demonstration list from this project's expert demonstrations.

    Each entry covers one of the groups `generate_subdominance_groups()`
    sampled from the expert's demonstrations -- the same groups the
    subdominance metric scores every learned policy against. Using them here
    means the baseline imitates the same expert, on the same rows, that the
    Fairness Reward Weight Optimization technique does, so the two are directly
    comparable.

    This changes where the demonstrations come from, not what the algorithm
    does with them: they are still a list of held-out decision sets with a loss
    per feature, exactly like `Super_human.prepare_test_pp()` produces.

    Parameters
    ----------
    demo_df : pandas.DataFrame
        The expert's demonstrations over the training pool, with `yhat`, `y`
        and `z` columns. Its row order defines the positional indices.
    group_idxs : list<numpy.ndarray>
        Positional indices of each group within `demo_df`, from
        `generate_subdominance_groups()`.
    loss_fn : callable(y_true, y_pred, z) -> numpy.ndarray

    Returns
    -------
    demo_list : list<SuperhumanDemo>
    """
    yhat = np.asarray(demo_df["yhat"])
    y = np.asarray(demo_df["y"])
    z = np.asarray(demo_df["z"])

    return [
        SuperhumanDemo(idx=idx, metric=loss_fn(y[idx], yhat[idx], z[idx]))
        for idx in group_idxs
    ]
