"""
The fair-classification baselines the Superhuman Fairness paper compares
against.

Ports of the baselines `Super_human.eval_model_baseline()` evaluates in the
reference implementation published with "Superhuman Fairness" (Memarrast et
al., ICML 2023):

    https://github.com/omidMemari/superhumn-fairness

* **Post-processing** (Hardt et al., 2016), with demographic parity or
  equalized odds as the constraint -- that function's `baseline="pp"` branch.
* **Robust fair log-loss** (Rezaei et al., 2020), with demographic parity or
  equalized odds as the constraint -- its `baseline="fair_logloss"` branch,
  running the classifiers vendored in `frwo.sh.fair_logloss`.

Each is unchanged from the reference implementation. What differs is only what
had to, for them to run here:

* Data plumbing. The original reads its splits from the CSVs it writes itself;
  these are handed the training split in memory, so every baseline sees exactly
  the dataset, the label bias and the split that the Fairness Reward Weight
  Optimization technique sees.
* Feature encoding. The original operates on an already-numeric
  `dataset_ref.csv`. These run the project's own `sklearn_clf_pipeline()`
  preprocessing, so the columns they see are the columns every other model here
  sees. The protected attribute `z` stays among them, exactly as the sensitive
  column stays in the original's `X`.
* The original shifts its protected attribute by one (`A_train - 1`) for the
  datasets whose sensitive column is coded {1, 2}. This project's `z` is
  already coded {0, 1}, which is what those branches were shifting *to*, so
  there is nothing to shift.

Every baseline exposes `predict()`, so that `generate_demo()` and
`_evaluate_policy()` can score it exactly like a learned FRWO policy.
"""

import logging

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.linear_model import LogisticRegression

from frwo.sh.fair_logloss import (
    DP_fair_logloss_classifier,
    EODD_fair_logloss_classifier,
    EOPP_fair_logloss_classifier,
)
from frwo.utils import sklearn_clf_pipeline

# The original's `logi_params`. See `SH_DEFAULT_LOGI_PARAMS` in
# `frwo.sh.superhuman_fairness` for why `penalty="l2"` is spelled as
# `l1_ratio=0.0` here.
BASELINE_LOGI_PARAMS = {
    "C": 100,
    "l1_ratio": 0.0,
    "solver": "newton-cg",
    "max_iter": 1000,
}

# The `C` the original passes to every fair log-loss classifier it evaluates.
FAIR_LOGLOSS_C = 0.005

# The `random_state` hard-coded in the original's class balancing. Only the
# default here: `PostProcessingBaseline` draws its own from its generator, so
# that the subsample differs between trials and restarts.
BALANCE_RANDOM_STATE = 1234


def balanced_train_index(X_train, y_train, random_state=BALANCE_RANDOM_STATE):
    """
    The class-balanced subsample the post-processor is fit on.

    `Super_human.eval_model_baseline()`: "Balanced data set is obtained by
    sampling the same number of points from the majority class (Y=0) as there
    are points in the minority class (Y=1)".

    The sample size is clamped to how many `Y=0` rows there actually are, since
    this project's datasets are not all majority-negative the way the
    original's are; on a majority-negative dataset the clamp never binds and
    this is the original's index exactly (with the original's `random_state`).

    Returns
    -------
    index : pandas.Index
        Labels of the balanced subsample, within `X_train`/`y_train`.
    """
    balanced_idx1 = X_train[y_train == 1].index
    n_negatives = int((y_train == 0).sum())
    return balanced_idx1.union(
        y_train[y_train == 0]
        .sample(
            n=min(balanced_idx1.size, n_negatives),
            random_state=random_state,
        )
        .index
    )


def fit_post_processing_model(
    X_train,
    y_train,
    feature_types,
    constraints="demographic_parity",
    logi_params=None,
    balance_random_state=BALANCE_RANDOM_STATE,
):
    """
    Fit the post-processing model of Hardt et al. (2016).

    `Super_human.eval_model_baseline()` with `baseline="pp"`, and the model
    the `PostProcDemo` expert uses: a logistic regression, wrapped in a
    fairlearn `ThresholdOptimizer` that is fit on a class-balanced subsample of
    the same data.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training inputs, including the protected attribute `z`.
    y_train : pandas.Series
        Training labels.
    feature_types : dict<str, list>
        Mapping of column names to their type of feature, for the pipeline.
    constraints : str, default "demographic_parity"
        The fairness constraint, i.e. the original's `mode`. The paper's
        baselines use "demographic_parity" and "equalized_odds".
    logi_params : dict, Optional
        Defaults to `BASELINE_LOGI_PARAMS`.
    balance_random_state : int, default BALANCE_RANDOM_STATE
        Draws the class-balanced subsample (see `balanced_train_index()`).

    Returns
    -------
    postprocess_est : fairlearn.postprocessing.ThresholdOptimizer
        Fitted, and ready to `predict(X, sensitive_features=X["z"])`.
    """
    logi_params = dict(logi_params or BASELINE_LOGI_PARAMS)

    model_logi = sklearn_clf_pipeline(
        feature_types=feature_types,
        clf_inst=LogisticRegression(**logi_params),
    )
    model_logi.fit(X_train, y_train)

    # Post-processing
    postprocess_est = ThresholdOptimizer(
        estimator=model_logi,
        constraints=constraints,
        predict_method="auto",
        prefit=True,
    )

    pp_train_idx = balanced_train_index(X_train, y_train, balance_random_state)
    X_train_balanced = X_train.loc[pp_train_idx, :]
    y_train_balanced = y_train.loc[pp_train_idx]

    # Post-process fitting
    postprocess_est.fit(
        X_train_balanced,
        y_train_balanced,
        sensitive_features=X_train_balanced["z"],
    )

    return postprocess_est


class PostProcessingBaseline:
    """
    The post-processing model of Hardt et al. (2016) as a baseline technique.

    Wraps `fit_post_processing_model()` so that the fitted post-processor can
    be handed to this project's evaluation pipeline like any other model.

    Parameters
    ----------
    feature_types : dict<str, list>
        Mapping of column names to their type of feature.
    constraints : str, default "demographic_parity"
        The fairness constraint the post-processor enforces.
    logi_params : dict, Optional
        Parameters of the underlying logistic regression.
    rng : numpy.random.Generator, Optional
        Source of this model's randomness: the class-balanced subsample it is
        fit on, and the randomization the `ThresholdOptimizer` applies when it
        predicts. A dedicated generator is used (rather than the global numpy
        random state) so that enabling this baseline cannot shift the random
        draws of the FRWO technique running alongside it.

    Attributes
    ----------
    model_ : fairlearn.postprocessing.ThresholdOptimizer
    """

    def __init__(
        self,
        feature_types,
        constraints="demographic_parity",
        logi_params=None,
        rng=None,
    ):
        self.feature_types = feature_types
        self.constraints = constraints
        self.logi_params = dict(logi_params or BASELINE_LOGI_PARAMS)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.model_ = None

    def fit(self, X, y):
        self.model_ = fit_post_processing_model(
            X,
            y,
            feature_types=self.feature_types,
            constraints=self.constraints,
            logi_params=self.logi_params,
            balance_random_state=int(self.rng.integers(0, 2**31 - 1)),
        )
        return self

    def predict(self, X, y=None):
        """
        Predict labels for `X`.

        `y` is accepted and ignored, so that this classifier is callable
        exactly like a `ClassificationMDPPolicy` from the evaluation pipeline.
        """
        preds = self.model_.predict(
            X,
            sensitive_features=X["z"],
            random_state=int(self.rng.integers(0, 2**31 - 1)),
        )
        return np.asarray(preds).astype(np.int64)


# The original's `mode` values, mapped to the classifier each selects.
FAIR_LOGLOSS_CLASSIFIERS = {
    "demographic_parity": DP_fair_logloss_classifier,
    "equalized_odds": EODD_fair_logloss_classifier,
    "equalized_opportunity": EOPP_fair_logloss_classifier,
}


class FairLogLossBaseline:
    """
    The robust fair-log-loss model of Rezaei et al. (2020) as a baseline.

    `Super_human.eval_model_baseline()` with `baseline="fair_logloss"`: fit one
    of the vendored `frwo.sh.fair_logloss` classifiers on the training
    split's features, labels and protected attribute, then predict by rounding
    its probabilities, with any NaN prediction taken to be 1 -- exactly the
    `baseline_preds[np.isnan(baseline_preds)] = 1` of the original.

    The features are this project's preprocessed design matrix rather than the
    original's raw `dataset_ref` columns. Like the original's own
    `eval_model_baseline()` branch (and unlike its `run_demo_baseline()` one),
    no additional standardization is applied.

    Parameters
    ----------
    feature_types : dict<str, list>
        Mapping of column names to their type of feature.
    mode : str, default "demographic_parity"
        The fairness constraint, i.e. the original's `mode`. One of
        `FAIR_LOGLOSS_CLASSIFIERS`.
    C : float, default 0.005
        The regularization the original passes to these classifiers.
    random_initialization : bool, default True
        As the original passes. Draws the starting `theta` at random.
    seed : int, Optional
        Seeds that random start. The vendored classifier draws it from numpy's
        global random state, so `fit()` seeds that state and restores it
        afterwards, leaving the global stream exactly as it found it. This
        keeps the baseline's draws reproducible without shifting a single draw
        the FRWO technique makes.

    Attributes
    ----------
    model_ : frwo.sh.fair_logloss.fair_logloss_classifier
    preprocessor_ : sklearn.compose.ColumnTransformer
        This project's preprocessing, fit on the training split.
    training_loss_ : float
        The objective the classifier minimized, at the `theta` it ended at, on
        the training data. Restarts of this baseline are compared on it.
    """

    def __init__(
        self,
        feature_types,
        mode="demographic_parity",
        C=FAIR_LOGLOSS_C,
        random_initialization=True,
        seed=None,
    ):
        if mode not in FAIR_LOGLOSS_CLASSIFIERS:
            raise ValueError(
                f"Unrecognized fair log-loss mode: {mode!r}."
                f" Valid modes are {sorted(FAIR_LOGLOSS_CLASSIFIERS)}."
            )
        self.feature_types = feature_types
        self.mode = mode
        self.C = C
        self.random_initialization = random_initialization
        self.seed = seed

        self.model_ = None
        self.preprocessor_ = None
        self.training_loss_ = None

    def _design(self, X):
        return np.asarray(self.preprocessor_.transform(X), dtype=float)

    def fit(self, X, y):
        # The preprocessing is taken from a throwaway pipeline so that the
        # design matrix is byte-for-byte the one every other model here is
        # trained on.
        pipeline = sklearn_clf_pipeline(
            feature_types=self.feature_types,
            clf_inst=LogisticRegression(**BASELINE_LOGI_PARAMS),
        )
        self.preprocessor_ = pipeline.named_steps["preprocessor"]
        self.preprocessor_.fit(X, y)

        design = self._design(X)
        y_values = np.asarray(y).astype(float)
        # The original's `A`: the protected attribute column of the training
        # data, coded {0, 1}.
        z_values = np.asarray(X["z"]).astype(float)

        self.model_ = FAIR_LOGLOSS_CLASSIFIERS[self.mode](
            C=self.C,
            random_initialization=self.random_initialization,
            verbose=False,
        )

        # The vendored classifier's random start draws from numpy's global
        # random state. Seed it, then put it back exactly as it was, so that
        # this baseline is reproducible and yet consumes none of the random
        # draws the rest of the experiment makes.
        state = np.random.get_state()
        try:
            if self.seed is not None:
                np.random.seed(self.seed % (2**32))
            self.model_.fit(design, y_values, z_values)
        finally:
            np.random.set_state(state)

        # The vendored `fit()` minimizes `compute_loss_grad()` on the design
        # matrix with an intercept column appended, and keeps only `theta`.
        design_intercept = np.hstack((design, np.ones((len(design), 1))))
        self.training_loss_ = float(
            self.model_.compute_loss_grad(
                self.model_.theta, design_intercept, y_values
            )[0]
        )
        return self

    def predict(self, X, y=None):
        """
        Predict labels for `X`.

        `y` is accepted and ignored, so that this classifier is callable
        exactly like a `ClassificationMDPPolicy` from the evaluation pipeline.
        """
        preds = self.model_.predict(self._design(X), np.asarray(X["z"]).astype(float))
        preds = np.asarray(preds, dtype=float)
        # As the original does, right after predicting.
        preds[np.isnan(preds)] = 1
        return preds.astype(np.int64)
