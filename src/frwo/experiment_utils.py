import contextlib
import copy
import datetime
import hashlib
import logging
import os
import random
import tempfile
import uuid
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
import sklearn.base
import optuna
import optunahub
import wandb
import pybobyqa
from catboost import CatBoostClassifier
from ConfigSpace import Configuration
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_negative_rate,
    true_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import (
    BoundedGroupLoss,
    DemographicParity,
    EqualizedOdds,
    ExponentiatedGradient,
    TruePositiveRateParity,
    ZeroOneLoss,
)
from line_profiler import LineProfiler
from scipy.spatial.distance import cosine
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import normalize

from frwo.irl.frwo import *
from frwo.rl.clf_mdp import *
from frwo.rl.clf_mdp_policy import *
from frwo.rl.objectives import *
from frwo.utils import *

from .datasets import *
from .sh.baselines import (
    FAIR_LOGLOSS_C,
    FairLogLossBaseline,
    PostProcessingBaseline,
)
from .sh.superhuman_fairness import (
    SH_DEFAULT_ITERS,
    SH_DEFAULT_LAMDA,
    SH_DEFAULT_LOGI_PARAMS,
    SH_DEFAULT_LR_THETA,
    SuperhumanFairness,
    build_expert_demo_list,
    compute_alphas,
    make_feature_loss_fn,
    feature_losses_from_demo,
    objective_feature_losses,
    validate_feature_names,
)
from .sh.superhuman_fairness_nn import (
    SH_NN_DEFAULT_BASE_FIT_EPOCHS,
    SH_NN_DEFAULT_HIDDEN_NODES,
    SH_NN_DEFAULT_LR_BOOST_FACTOR,
    SH_NN_DEFAULT_LR_DECAY_GAMMA_FRAC,
    SH_NN_DEFAULT_LR_THETA,
    SuperhumanFairnessNN,
)

# The commit of the OptunaHub registry to load the SMAC sampler from.
# `optunahub.load_module()` otherwise downloads and runs whatever is on `main`.
SMAC_SAMPLER_REF = "9be3365fb8167dbd3fd927f560a5682059dda818"

# `OBJ_LOOKUP_BY_NAME` now lives in `frwo.rl.objectives`, next to the
# objectives it names, so that modules which cannot import this one (e.g.
# `frwo.sh.superhuman_fairness`) can still resolve objective names. It is
# star-imported above and so is still available as
# `experiment_utils.OBJ_LOOKUP_BY_NAME`.


def _negative_predictive_value(y_true, y_pred):
    """
    P(y=0 | yhat=0) = TN / (TN + FN). Not available in fairlearn/sklearn, so
    it's implemented manually to be used as a `MetricFrame` metric function.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    denom = tn + fn
    return tn / denom if denom > 0 else np.nan


def _group_rate_loss(metric_fn, z_value):
    """
    Builds a `demo`-loss function for a metric evaluated on a single
    sensitive-feature group (e.g. TPR_Z0), matching the `-mu + 1` inversion
    used by `compute_relevant_feat_loss()`.
    """

    def _loss(demo):
        mf = MetricFrame(
            metrics=metric_fn,
            y_true=demo["y"],
            y_pred=demo["yhat"],
            sensitive_features=demo["z"],
        )
        return 1 - mf.by_group[z_value]

    return _loss


def _pairwise_metric_diff_loss(metric_fn):
    """
    Builds a `demo`-loss function for the absolute between-group difference
    of a metric (e.g. Predictive Parity), matching the `-mu + 1` inversion
    used by `compute_relevant_feat_loss()`.
    """

    def _loss(demo):
        mf = MetricFrame(
            metrics=metric_fn,
            y_true=demo["y"],
            y_pred=demo["yhat"],
            sensitive_features=demo["z"],
        )
        return mf.difference(method="between_groups")

    return _loss


# Fairlearn/sklearn-based equivalent of OBJ_LOOKUP_BY_NAME: maps each obj_name
# to a function of `demo` that reproduces the corresponding entry of
# `demo_feat_exp` as computed in `compute_relevant_feat_loss()`.
FAIRLEARN_OBJ_LOOKUP_BY_NAME = {
    "Acc": lambda demo: 1 - accuracy_score(demo["y"], demo["yhat"]),
    "AccPar": _pairwise_metric_diff_loss(accuracy_score),
    "DemPar": lambda demo: demographic_parity_difference(
        demo["y"], demo["yhat"], sensitive_features=demo["z"]
    ),
    "EqOpp": lambda demo: equal_opportunity_difference(
        demo["y"], demo["yhat"], sensitive_features=demo["z"]
    ),
    "FPRPar": _pairwise_metric_diff_loss(false_positive_rate),
    "EqOdds": lambda demo: equalized_odds_difference(
        demo["y"], demo["yhat"], sensitive_features=demo["z"]
    ),
    "TNRPar": _pairwise_metric_diff_loss(true_negative_rate),
    "FNRPar": _pairwise_metric_diff_loss(false_negative_rate),
    "PredPar": _pairwise_metric_diff_loss(precision_score),
    "NegPredPar": _pairwise_metric_diff_loss(_negative_predictive_value),
    "PR_Z0": _group_rate_loss(selection_rate, 0),
    "PR_Z1": _group_rate_loss(selection_rate, 1),
    "NR_Z0": _group_rate_loss(lambda yt, yp: 1 - selection_rate(yt, yp), 0),
    "NR_Z1": _group_rate_loss(lambda yt, yp: 1 - selection_rate(yt, yp), 1),
    "TPR_Z0": _group_rate_loss(true_positive_rate, 0),
    "TPR_Z1": _group_rate_loss(true_positive_rate, 1),
    "TNR_Z0": _group_rate_loss(true_negative_rate, 0),
    "TNR_Z1": _group_rate_loss(true_negative_rate, 1),
    "FPR_Z0": _group_rate_loss(false_positive_rate, 0),
    "FPR_Z1": _group_rate_loss(false_positive_rate, 1),
    "FNR_Z0": _group_rate_loss(false_negative_rate, 0),
    "FNR_Z1": _group_rate_loss(false_negative_rate, 1),
}


class FairLearnSkLearnWrapper:
    """
    Wrapper around scikit-learn classifiers to make them compatible with
    fairlearn classifiers, which require an additional `sensitive_features`
    attribute to be passed in when calling `fit` and `predict`.

    Attributes
    ----------
    initial_clf : sklearn base estimator
        A clone of the initial `clf`. Used to create new clones later on
        without any of the attributes updated.
    has_access_to_sensitive_features : bool, default True
        If False, sensitive features are not available at prediction time. E.g.
        for fairlearn reductions (as opposed to threshold optimizers).
    clone_on_fit : bool, default False
        If True, resets the base classifier.
    """

    def __init__(
        self,
        clf,
        sensitive_features,
        has_access_to_sensitive_features=True,
        clone_on_fit=False,
    ):
        self.clf = clf
        self.initial_clf = sklearn.base.clone(self.clf)
        self.sensitive_features = sensitive_features
        self.has_access_to_sensitive_features = has_access_to_sensitive_features
        self.clone_on_fit = clone_on_fit

    def fit(self, X, y, sample_weight=None, **kwargs):
        if self.clone_on_fit:
            self.clf = sklearn.base.clone(self.initial_clf)

        self.clf.fit(
            X,
            y,
            sensitive_features=X[self.sensitive_features],
            **kwargs,
        )
        return self

    def predict(self, X, sample_weight=None, **kwargs):
        if self.has_access_to_sensitive_features:
            preds = self.clf.predict(
                X,
                sensitive_features=X[self.sensitive_features],
                **kwargs,
            )
        else:
            preds = self.clf.predict(
                X,
                **kwargs,
            )

        return preds


class UnfairNoisyClassifier:
    """
    Wrapper around scikit-learn classifiers to intentionally make them slightly
    less performant. This is useful when generating initial policies where
    having reasonably fair and accurate (but not optimal) classifiers helps set
    the weights in the right directions.

    It randomplly flips negative predictions to positive predictions, based on
    the probabilities defined in the input.
    """

    def __init__(self, clf, prob):
        self.clf = clf
        self.prob = prob

    def fit(self, X, y, **kwargs):
        self.clf.fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        preds = self.clf.predict(X, **kwargs)

        z0_override_indexes = np.argwhere(
            np.random.rand(len(X)) < self.prob[0]
        ).flatten()

        z1_override_indexes = np.argwhere(
            np.random.rand(len(X)) < self.prob[1]
        ).flatten()

        for idx in z0_override_indexes:
            z = X.iloc[idx]["z"]
            if z == 0:
                preds[idx] = 1 - preds[idx]

        for idx in z1_override_indexes:
            z = X.iloc[idx]["z"]
            if z == 1:
                preds[idx] = 1 - preds[idx]

        return preds


def generate_expert_algo_lookup(feature_types, exp_info=None):
    """
    Parameters
    ----------
    feature_types : dict<str, array-like>
        Mapping of column names to their type of feature. Used to when
        constructing the sklearn pipeline.
    exp_info : dict, Optional
        Experiment parameters. Only the `"PostProcDemo"` expert reads them --
        for its fairness constraint and its random seed -- so leaving this out
        builds that expert with the defaults of `SH_DEFAULTS`.

    Returns
    -------
    expert_algo_lookup : dict<str, sklearn.pipeline>
        The expert algo lookup dictionary that maps the string name for an
        algorithm to the actual implementation.
    """
    exp_info = exp_info if exp_info is not None else {}
    # OptAcc
    opt_acc_pipe = sklearn_clf_pipeline(
        feature_types,
        # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
        # clf_inst=DecisionTreeClassifier(min_samples_leaf=5, max_depth=10),
        # RandomForestClassifier(),
        CatBoostClassifier(allow_writing_files=False, logging_level="Silent"),
    )

    catboost_opt_acc_pipe = sklearn_clf_pipeline(
        feature_types,
        # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
        # clf_inst=DecisionTreeClassifier(min_samples_leaf=5, max_depth=10),
        # RandomForestClassifier(),
        CatBoostClassifier(allow_writing_files=False, logging_level="Silent"),
    )

    xgboost_opt_acc_pipe = sklearn_clf_pipeline(
        feature_types,
        # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
        # clf_inst=DecisionTreeClassifier(min_samples_leaf=5, max_depth=10),
        # RandomForestClassifier(),
        XGBClassifier(),
    )

    # HardtDemPar
    dem_par_thresh_opt = ThresholdOptimizer(
        constraints="demographic_parity",
        predict_method="predict",
        prefit=False,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(), # Messes up DemPar. Why?
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    dem_par_wrapper = FairLearnSkLearnWrapper(
        clf=dem_par_thresh_opt,
        sensitive_features="z",
    )

    # HardtEqOpp
    eq_opp_thresh_opt = ThresholdOptimizer(
        constraints="true_positive_rate_parity",
        predict_method="predict",
        prefit=False,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    eq_opp_wrapper = FairLearnSkLearnWrapper(
        clf=eq_opp_thresh_opt,
        sensitive_features="z",
    )

    # HardtEqOdds
    eq_odds_thresh_opt = ThresholdOptimizer(
        constraints="equalized_odds",
        predict_method="predict",
        prefit=False,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10000, max_depth=20),
        ),
    )
    eq_odds_thresh_wrapper = FairLearnSkLearnWrapper(
        clf=eq_odds_thresh_opt,
        sensitive_features="z",
    )

    # Demographic Parity Reduction with difference_bound=0.01
    dem_par = DemographicParity(difference_bound=0.01)
    dem_par_exp_grad = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    dem_par_red_wrapper = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.01
    dem_par_1 = DemographicParity(difference_bound=0.01)
    dem_par_exp_grad_1 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_1,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    dem_par_red_wrapper_1 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_1,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.02
    dem_par_2 = DemographicParity(difference_bound=0.02)
    dem_par_exp_grad_2 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_2,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    dem_par_red_wrapper_2 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_2,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.03
    dem_par_3 = DemographicParity(difference_bound=0.03)
    dem_par_exp_grad_3 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_3,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    dem_par_red_wrapper_3 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_3,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.04
    dem_par_4 = DemographicParity(difference_bound=0.04)
    dem_par_exp_grad_4 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_4,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    dem_par_red_wrapper_4 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_4,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.05
    dem_par_5 = DemographicParity(difference_bound=0.05)
    dem_par_exp_grad_5 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_5,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    dem_par_red_wrapper_5 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_5,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.06
    dem_par_6 = DemographicParity(difference_bound=0.06)
    dem_par_exp_grad_6 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_6,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    dem_par_red_wrapper_6 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_6,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.07
    dem_par_7 = DemographicParity(difference_bound=0.07)
    dem_par_exp_grad_7 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_7,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    dem_par_red_wrapper_7 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_7,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.08
    dem_par_8 = DemographicParity(difference_bound=0.08)
    dem_par_exp_grad_8 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_8,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    dem_par_red_wrapper_8 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_8,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.09
    dem_par_9 = DemographicParity(difference_bound=0.09)
    dem_par_exp_grad_9 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_9,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    dem_par_red_wrapper_9 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_9,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.1
    dem_par_10 = DemographicParity(difference_bound=0.1)
    dem_par_exp_grad_10 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_10,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    dem_par_red_wrapper_10 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_10,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Equal Opportunity Reduction
    eq_opp = TruePositiveRateParity(difference_bound=0.01)
    eq_opp_exp_grad = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=eq_opp,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    eq_opp_red_wrapper = FairLearnSkLearnWrapper(
        clf=eq_opp_exp_grad,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Equal Odds Reduction
    eq_odds = EqualizedOdds(difference_bound=0.01)
    eq_odds_exp_grad = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=eq_odds,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    eq_odds_red_wrapper = FairLearnSkLearnWrapper(
        clf=eq_odds_exp_grad,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Bounded Group Loss Reduction
    bgl = BoundedGroupLoss(ZeroOneLoss(), upper_bound=0.03)
    bgl_exp_grad = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=bgl,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=5, max_depth=10),
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    bgl_wrapper = FairLearnSkLearnWrapper(
        clf=bgl_exp_grad,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # False Positive Rate Noisy
    fpr_thresh_opt = ThresholdOptimizer(
        constraints="false_positive_rate_parity",
        predict_method="predict",
        prefit=False,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    fpr_wrapper = FairLearnSkLearnWrapper(
        clf=fpr_thresh_opt,
        sensitive_features="z",
    )

    # True Negative Rate Noisy
    tnr_thresh_opt = ThresholdOptimizer(
        constraints="true_negative_rate_parity",
        predict_method="predict",
        prefit=False,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    tnr_wrapper = FairLearnSkLearnWrapper(
        clf=tnr_thresh_opt,
        sensitive_features="z",
    )

    # False Positive Rate Noisy
    fnr_thresh_opt = ThresholdOptimizer(
        constraints="false_negative_rate_parity",
        predict_method="predict",
        prefit=False,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
        ),
    )
    fnr_wrapper = FairLearnSkLearnWrapper(
        clf=fnr_thresh_opt,
        sensitive_features="z",
    )

    dummy_pipe = DummyClassifier(strategy="uniform")

    compas_score_high = ManualClassifier(
        # lambda row: int(row['score_text'] == 'High')
        lambda row: int(row["decile_score"] >= 6)
    )

    # PostProcDemo: a logistic regression post-processed by a fairlearn
    # ThresholdOptimizer that is fit on a class-balanced subsample (see
    # `fit_post_processing_model()`). Its fairness constraint is
    # `SH_DEMO_CONSTRAINTS`.
    #
    # Unlike the `Hardt*` experts, whose ThresholdOptimizers draw their
    # prediction-time randomization from the global numpy random state, this
    # one draws from a generator seeded from the experiment's own
    # `RANDOM_SEED`, so its demonstrations are reproducible.
    post_proc_demo_expert = PostProcessingBaseline(
        feature_types=feature_types,
        constraints=_superhuman_config(exp_info)["SH_DEMO_CONSTRAINTS"],
        rng=np.random.default_rng(_baseline_seed(exp_info, "PostProcDemo", (), 0)),
    )

    # OptClfMDPPol: optimal classifier policy (see compute_optimal_policy() in
    # frwo.py), with reward weights split equally (and positively) across
    # every feature-expectation objective. Its ClassificationMDPPolicy is
    # actually built per-fold in generate_non_overfit_demos() since it depends
    # on the objective set and training fold, neither of which is available
    # here yet.
    opt_clf_mdp_pol_expert = OptClfMDPPolicyExpert(feature_types)

    expert_algo_lookup = {
        # Experts
        "OptAcc": opt_acc_pipe,
        "CatBoostOptAcc": catboost_opt_acc_pipe,
        "XGBoostOptAcc": xgboost_opt_acc_pipe,
        "HardtDemPar": dem_par_wrapper,
        "HardtEqOpp": eq_opp_wrapper,
        "HardtEqOdds": eq_odds_thresh_wrapper,
        "HardtFPRPar": fpr_wrapper,
        "HardtTNRPar": tnr_wrapper,
        "HardtFNRPar": fnr_wrapper,
        "Dummy": dummy_pipe,
        "DemParRed": dem_par_red_wrapper,
        "DemParRed_0.01": dem_par_red_wrapper_1,
        "DemParRed_0.02": dem_par_red_wrapper_2,
        "DemParRed_0.03": dem_par_red_wrapper_3,
        "DemParRed_0.04": dem_par_red_wrapper_4,
        "DemParRed_0.05": dem_par_red_wrapper_5,
        "DemParRed_0.06": dem_par_red_wrapper_6,
        "DemParRed_0.07": dem_par_red_wrapper_7,
        "DemParRed_0.08": dem_par_red_wrapper_8,
        "DemParRed_0.09": dem_par_red_wrapper_9,
        "DemParRed_0.1": dem_par_red_wrapper_10,
        "EqOppRed": eq_opp_red_wrapper,
        "EqOddsRed": eq_odds_red_wrapper,
        "BoundedGroupLoss": bgl_wrapper,
        "COMPAS": compas_score_high,
        "OptClfMDPPol": opt_clf_mdp_pol_expert,
        "PostProcDemo": post_proc_demo_expert,
        # Initial policies
        "OptAccNoisy": UnfairNoisyClassifier(clf=opt_acc_pipe, prob=[0.15, 0.25]),
        "HardtDemParNoisy": UnfairNoisyClassifier(
            clf=dem_par_wrapper, prob=[0.05, 0.25]
        ),
        "HardtEqOppNoisy": UnfairNoisyClassifier(clf=eq_opp_wrapper, prob=[0.05, 0.25]),
        "HardtFPRNoisy": UnfairNoisyClassifier(clf=fpr_wrapper, prob=[0.05, 0.05]),
        "HardtTNRNoisy": UnfairNoisyClassifier(clf=tnr_wrapper, prob=[0.05, 0.05]),
        "DummyNoisy": UnfairNoisyClassifier(clf=dummy_pipe, prob=[0.05, 0.05]),
    }

    return expert_algo_lookup


# ---------------------------------------------------------------------------
# Weights & Biases experiment tracking
# ---------------------------------------------------------------------------
# Experiment configuration, learned-policy metrics and trial results are all
# reported to W&B rather than written out as CSV/JSON files. The server address and credentials come from the standard
# `wandb` configuration (the `WANDB_BASE_URL` environment variable or
# `~/.config/wandb/settings`), so pointing the experiments at a different
# server needs no code change here.

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "frwo")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or None

# The four aggregations of the subdominance metric, in the order
# `compute_iteration_subdominance()` returns them.
SUBDOMINANCE_AGGREGATIONS = (
    ("max", "abs"),
    ("sum", "abs"),
    ("max", "rel"),
    ("sum", "rel"),
)

# The data splits every subdominance measurement is computed on.
SUBDOMINANCE_SPLITS = ("train", "val", "test")

# The 12 subdominance measurements returned, in this order, by
# `_evaluate_policy()` for every learned policy.
SUBDOMINANCE_KEYS = tuple(
    f"{agg}_{kind}_subdominance_{split}"
    for split in SUBDOMINANCE_SPLITS
    for agg, kind in SUBDOMINANCE_AGGREGATIONS
)

# The techniques `run_experiment_trial()` can train and evaluate. Which of them
# a run of the experiment script actually covers is set per experiment, as
# `exp_info["ALGORITHMS"]`.
#
# Superhuman Fairness Fixed is Superhuman Fairness with the normalization of
# its gradient's feature-matching term fixed, Superhuman Fairness Neural
# Network is the neural network version of the reference implementation's
# `reorg_current` branch, and Superhuman Fairness Neural Network Fixed is that
# with its training loss fixed; see `SUPERHUMAN_VARIANTS`.
#
# The four names after them are the fair-classification baselines the
# Superhuman Fairness paper compares itself against, ported in
# `frwo.sh.baselines`.
ALGORITHM_FRWO = "Fairness Reward Weight Optimization"
ALGORITHM_SUPERHUMAN = "Superhuman Fairness"
ALGORITHM_SUPERHUMAN_FIXED = "Superhuman Fairness Fixed"
ALGORITHM_SUPERHUMAN_NN = "Superhuman Fairness Neural Network"
ALGORITHM_SUPERHUMAN_NN_FIXED = "Superhuman Fairness Neural Network Fixed"
ALGORITHM_POST_PROC_DP = "Post Proc DP"
ALGORITHM_POST_PROC_EQODDS = "Post Proc EqOdds"
ALGORITHM_FAIR_LOGLOSS_DP = "Fair LogLoss DP"
ALGORITHM_FAIR_LOGLOSS_EQODDS = "Fair LogLoss EqOdds"
ALGORITHMS = (
    ALGORITHM_FRWO,
    ALGORITHM_SUPERHUMAN,
    ALGORITHM_SUPERHUMAN_FIXED,
    ALGORITHM_SUPERHUMAN_NN,
    ALGORITHM_SUPERHUMAN_NN_FIXED,
    ALGORITHM_POST_PROC_DP,
    ALGORITHM_POST_PROC_EQODDS,
    ALGORITHM_FAIR_LOGLOSS_DP,
    ALGORITHM_FAIR_LOGLOSS_EQODDS,
)

# How to build each of the fair-classification baselines. Every one of them is
# an ordinary classifier: fit on the training split, then scored by
# `_evaluate_and_finalize()` like everything else here, so one runner
# (`_run_baseline_trial()`) covers them all.
#
# `constraints`/`mode` are the `mode` argument the original's
# `eval_model_baseline()` is called with for that baseline.
BASELINE_SPECS = {
    ALGORITHM_POST_PROC_DP: {
        "builder": PostProcessingBaseline,
        "kwargs": {"constraints": "demographic_parity"},
    },
    ALGORITHM_POST_PROC_EQODDS: {
        "builder": PostProcessingBaseline,
        "kwargs": {"constraints": "equalized_odds"},
    },
    ALGORITHM_FAIR_LOGLOSS_DP: {
        "builder": FairLogLossBaseline,
        "kwargs": {"mode": "demographic_parity"},
        "config_prefix": "FAIR_LOGLOSS_DP_",
    },
    ALGORITHM_FAIR_LOGLOSS_EQODDS: {
        "builder": FairLogLossBaseline,
        "kwargs": {"mode": "equalized_odds"},
        "config_prefix": "FAIR_LOGLOSS_EQODDS_",
    },
}

# Defaults for the fair log-loss baselines. Each baseline has its own copy of
# these settings, so that each can be tuned separately: the `exp_info` key of a
# setting is its baseline's `config_prefix` in `BASELINE_SPECS` followed by the
# setting's name here, e.g. `FAIR_LOGLOSS_DP_C`. Both defaults are what the
# original's `eval_model_baseline()` passes.
FAIR_LOGLOSS_DEFAULTS = {
    "C": FAIR_LOGLOSS_C,
    "RANDOM_INIT": True,
}

# Defaults for the Superhuman Fairness baseline's own configuration. Each is
# overridable through the correspondingly named `exp_info` key; see
# `_superhuman_config()`.
SH_DEFAULTS = {
    # The performance/fairness measures the baseline optimizes, as objective
    # names (e.g. "Acc", "DemPar") or the original paper's metric names (e.g.
    # "inacc", "dp"). `None` uses this experiment's subdominance metrics, so
    # that the baseline optimizes exactly what both techniques are scored on.
    "SH_FEATURES": None,
    # How many demonstrations to imitate. `None` uses every group the
    # subdominance metric already sampled.
    "SH_NUM_DEMOS": None,
    "SH_ITERS": SH_DEFAULT_ITERS,
    "SH_LR_THETA": SH_DEFAULT_LR_THETA,
    "SH_LAMDA": SH_DEFAULT_LAMDA,
    # Superhuman Fairness Fixed's own `iters`, `lr_theta` and `lamda`, so that
    # it can be tuned separately. It shares every other setting here.
    "SH_FIXED_ITERS": SH_DEFAULT_ITERS,
    "SH_FIXED_LR_THETA": SH_DEFAULT_LR_THETA,
    "SH_FIXED_LAMDA": SH_DEFAULT_LAMDA,
    # Superhuman Fairness Neural Network's own `iters`, `lr_theta` and
    # `lamda`. Its `lr_theta` is the network's Adam learning rate, whose
    # default is the reference implementation's 1e-5.
    "SH_NN_ITERS": SH_DEFAULT_ITERS,
    "SH_NN_LR_THETA": SH_NN_DEFAULT_LR_THETA,
    "SH_NN_LAMDA": SH_DEFAULT_LAMDA,
    # Its network and schedule; see `SuperhumanFairnessNN`.
    "SH_NN_HIDDEN_NODES": SH_NN_DEFAULT_HIDDEN_NODES,
    "SH_NN_BASE_FIT_EPOCHS": SH_NN_DEFAULT_BASE_FIT_EPOCHS,
    "SH_NN_LR_BOOST_FACTOR": SH_NN_DEFAULT_LR_BOOST_FACTOR,
    "SH_NN_LR_DECAY_GAMMA_FRAC": SH_NN_DEFAULT_LR_DECAY_GAMMA_FRAC,
    # The torch device it trains on; `None` uses a GPU if torch sees one.
    "SH_NN_DEVICE": None,
    # Superhuman Fairness Neural Network Fixed's own counterparts of every
    # SH_NN_* setting above, so that it can be tuned separately.
    "SH_NN_FIXED_ITERS": SH_DEFAULT_ITERS,
    "SH_NN_FIXED_LR_THETA": SH_NN_DEFAULT_LR_THETA,
    "SH_NN_FIXED_LAMDA": SH_DEFAULT_LAMDA,
    "SH_NN_FIXED_HIDDEN_NODES": SH_NN_DEFAULT_HIDDEN_NODES,
    "SH_NN_FIXED_BASE_FIT_EPOCHS": SH_NN_DEFAULT_BASE_FIT_EPOCHS,
    "SH_NN_FIXED_LR_BOOST_FACTOR": SH_NN_DEFAULT_LR_BOOST_FACTOR,
    "SH_NN_FIXED_LR_DECAY_GAMMA_FRAC": SH_NN_DEFAULT_LR_DECAY_GAMMA_FRAC,
    "SH_NN_FIXED_DEVICE": None,
    # The fairness constraint the PostProcDemo expert satisfies.
    "SH_DEMO_CONSTRAINTS": "demographic_parity",
}


def _build_superhuman_lr(common, sh_config, fix_normalization):
    """A logistic regression `SuperhumanFairness`."""
    return SuperhumanFairness(**common, fix_normalization=fix_normalization)


def _build_superhuman_nn(common, sh_config, prefix, fix_gradient):
    """
    A neural network `SuperhumanFairnessNN`, configured from the `SH_DEFAULTS`
    keys starting with `prefix`.
    """
    return SuperhumanFairnessNN(
        **common,
        hidden_nodes=sh_config[f"{prefix}HIDDEN_NODES"],
        base_fit_epochs=sh_config[f"{prefix}BASE_FIT_EPOCHS"],
        lr_boost_factor=sh_config[f"{prefix}LR_BOOST_FACTOR"],
        lr_decay_gamma_frac=sh_config[f"{prefix}LR_DECAY_GAMMA_FRAC"],
        device=sh_config[f"{prefix}DEVICE"],
        fix_gradient=fix_gradient,
    )


# The Superhuman Fairness techniques, which share `_run_superhuman_trial()`:
# the same demonstrations, features and evaluation. Each has its own
# `SH_DEFAULTS` keys for `iters`, `lr_theta` and `lamda`, and `build`s its own
# model from them (passed in `common`) and from `SH_DEFAULTS`. `resolved` is
# recorded in its runs' config.
#
# Superhuman Fairness Fixed differs from Superhuman Fairness only in
# `fix_normalization`: see `SuperhumanFairness._compute_feature_matching()`.
# Superhuman Fairness Neural Network is `SuperhumanFairnessNN`, and Superhuman
# Fairness Neural Network Fixed differs from it only in `fix_gradient`: see
# `SuperhumanFairnessNN._fixed_loss()`.
SUPERHUMAN_VARIANTS = {
    ALGORITHM_SUPERHUMAN: {
        "iters": "SH_ITERS",
        "lr_theta": "SH_LR_THETA",
        "lamda": "SH_LAMDA",
        "build": partial(_build_superhuman_lr, fix_normalization=False),
        "resolved": {"SH_FIX_NORMALIZATION_RESOLVED": False},
    },
    ALGORITHM_SUPERHUMAN_FIXED: {
        "iters": "SH_FIXED_ITERS",
        "lr_theta": "SH_FIXED_LR_THETA",
        "lamda": "SH_FIXED_LAMDA",
        "build": partial(_build_superhuman_lr, fix_normalization=True),
        "resolved": {"SH_FIX_NORMALIZATION_RESOLVED": True},
    },
    ALGORITHM_SUPERHUMAN_NN: {
        "iters": "SH_NN_ITERS",
        "lr_theta": "SH_NN_LR_THETA",
        "lamda": "SH_NN_LAMDA",
        "build": partial(_build_superhuman_nn, prefix="SH_NN_", fix_gradient=False),
        "resolved": {"SH_NN_FIX_GRADIENT_RESOLVED": False},
    },
    ALGORITHM_SUPERHUMAN_NN_FIXED: {
        "iters": "SH_NN_FIXED_ITERS",
        "lr_theta": "SH_NN_FIXED_LR_THETA",
        "lamda": "SH_NN_FIXED_LAMDA",
        "build": partial(
            _build_superhuman_nn, prefix="SH_NN_FIXED_", fix_gradient=True
        ),
        "resolved": {"SH_NN_FIX_GRADIENT_RESOLVED": True},
    },
}


def algorithm_slug(algorithm):
    """
    Build the short, filename-safe label of an algorithm.

    Used as part of the W&B run name and as a run tag, so that the runs of the
    two techniques can be told apart at a glance.
    """
    return str(algorithm).lower().replace(" ", "_")


def selected_algorithms(exp_info):
    """
    The algorithms one experiment covers, validated and in a fixed order.

    Defaults to just the Fairness Reward Weight Optimization technique, so that
    an `exp_info` written before this baseline existed behaves exactly as it
    did then.

    Returns
    -------
    algorithms : list<str>
        The selected entries of `ALGORITHMS`, always in `ALGORITHMS` order so
        that the FRWO technique runs first regardless of how the config
        lists them.
    """
    selected = exp_info.get("ALGORITHMS") or [ALGORITHM_FRWO]

    unknown = [name for name in selected if name not in ALGORITHMS]
    if unknown:
        raise ValueError(
            f"Unrecognized entries in exp_info['ALGORITHMS']: {unknown}."
            f" Valid entries are {list(ALGORITHMS)}."
        )

    return [name for name in ALGORITHMS if name in selected]


def _superhuman_config(exp_info):
    """`SH_DEFAULTS`, overridden by whatever `exp_info` sets."""
    return {
        key: exp_info.get(key, default) if exp_info.get(key) is not None else default
        for key, default in SH_DEFAULTS.items()
    }


def _baseline_seed(exp_info, algorithm, bias_type, trial_i):
    """
    Derive a baseline's random seed from what makes its run unique.

    Every baseline gets its own deterministic seed rather than drawing from the
    global numpy random state. That way enabling a baseline cannot shift a
    single random draw the FRWO technique makes, so a session that runs
    several techniques produces exactly the FRWO results a session that runs
    only FRWO would -- and each baseline is itself reproducible.
    """
    key = "|".join(
        [
            str(exp_info.get("RANDOM_SEED")),
            str(exp_info.get("EXPERIMENT_NAME")),
            str(exp_info.get("DATASET")),
            str(exp_info.get("EXPERT_ALGO")),
            str(algorithm),
            bias_type_name(bias_type),
            str(trial_i),
        ]
    )
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")


def _baseline_rng(exp_info, algorithm, bias_type, trial_i):
    """The random generator one baseline draws from. See `_baseline_seed()`."""
    return np.random.default_rng(_baseline_seed(exp_info, algorithm, bias_type, trial_i))


def _json_safe(value):
    """
    Convert numpy scalars/arrays and tuples into plain Python types so they can
    be stored in a W&B run config.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def weight_adjusts_name(weight_adjust):
    """
    Build a short, human readable label for a weight adjustment configuration.

    Used as the W&B `job_type` and as part of the run name, so that runs which
    differ only in how the learned weights were adjusted can be grouped and
    compared against each other.

    Parameters
    ----------
    weight_adjusts : tuple
        One entry of `exp_info["WEIGHT_ADJUST_LIST"]`, or `()` for the
        unadjusted weights.

    Returns
    -------
    name : str
        E.g. `"unadjusted"` or `"opt_debias_optuna_GP"`.
    """
    if not weight_adjust:
        return "unadjusted"

    names = []
    for weight_adjust_component in weight_adjust:
        if isinstance(weight_adjust_component, str):
            names.append(weight_adjust_component)
        else:
            names.append(str(weight_adjust_component))

    return "_".join(names)


def bias_type_name(bias_type):
    """
    Build a short, human readable label for a bias type configuration.

    Used as part of the W&B run name and as a run tag, so that the runs of the
    bias types one trial covers can be told apart from each other.

    Parameters
    ----------
    bias_type : tuple
        One entry of `exp_info["DATASET_BIAS_TYPE_LIST"]` or
        `exp_info["DEMO_BIAS_TYPE_LIST"]`, or `()` for the unbiased
        demonstrations.

    Returns
    -------
    name : str
        E.g. `"unbiased"`, `"dataset_balanced_redlining_0.2"` or
        `"demo_noise_bias_0.1"`.
    """
    if not bias_type:
        return "unbiased"

    return "_".join(str(component) for component in bias_type)


# Fairness Reward Weight Optimization's training adjustments, by kind: the
# `exp_info` list of the kind's entries, and the key each run records its entry
# as (`()` for a run without one) in its W&B config. Every run has at most one
# training adjustment, as a `(kind, entry)` pair, where `entry` is one entry of
# the kind's list: a tuple of `(name, value)` pairs (see
# `frwo.config.TRAINING_ADJUST_LIST_KEYS`).
#   "alpha"          multiplies the `alpha` scaling of each subdominance
#                    metric in the `opt_debias` objective.
#   "subdom_weight"  multiplies the weight of each subdominance metric in the
#                    sum aggregation of the `opt_debias` objective.
#   "manual_weight"  adds a term to each reward weight of the unadjusted
#                    weights or of those `opt_debias` learned.
TRAINING_ADJUST_KINDS = {
    "alpha": ("ALPHA_ADJUST_LIST", "ALPHA_ADJUST"),
    "subdom_weight": ("SUBDOM_WEIGHT_ADJUST_LIST", "SUBDOM_WEIGHT_ADJUST"),
    "manual_weight": ("MANUAL_WEIGHT_ADJUST_LIST", "MANUAL_WEIGHT_ADJUST"),
}
# The training adjustment of a run without one.
NO_TRAINING_ADJUST = (None, ())


def training_adjusts(exp_info, kind):
    """The entries of `exp_info`'s training adjustment list of `kind`."""
    return tuple(exp_info.get(TRAINING_ADJUST_KINDS[kind][0]) or ())


def training_adjust_name(training_adjust):
    """
    Build a short, human readable label for a training adjustment.

    Parameters
    ----------
    training_adjust : tuple
        A `(kind, entry)` pair (see `TRAINING_ADJUST_KINDS`), or
        `NO_TRAINING_ADJUST`.

    Returns
    -------
    name : str
        E.g. `"alpha_adjust_EqOpp_2.0"`, or `""` for `NO_TRAINING_ADJUST`.
    """
    kind, entry = training_adjust
    if not entry:
        return ""
    return f"{kind}_adjust_" + "_".join(f"{name}_{value}" for name, value in entry)


def _adjust_vector(names, entry, default):
    """
    A training adjustment entry as a vector over `names`, in their order, with
    `default` for every name the entry leaves out.
    """
    values = dict(entry)
    return np.array([float(values.get(name, default)) for name in names])


def validate_training_adjusts(exp_info):
    """
    Check every entry of Fairness Reward Weight Optimization's training
    adjustment lists, so that a misconfigured run fails before anything
    expensive is done: that it names only metrics (or, for the manual weight
    adjustments, reward objectives) this experiment has, with finite values,
    non-negative for the multipliers, and that the adjustments that skew the
    `opt_debias` objective have an `opt_debias` run to skew.
    """
    metric_names = subdominance_metric_names(exp_info)
    objective_names = list(exp_info["FEAT_EXP_OBJECTIVE_NAMES"])
    for kind, (list_key, _) in TRAINING_ADJUST_KINDS.items():
        entries = training_adjusts(exp_info, kind)
        is_multiplier = kind != "manual_weight"
        if entries and is_multiplier and not exp_info["RUN_OPT_DEBIAS"]:
            raise ValueError(
                f"{list_key} skews the subdominance the `opt_debias` weight"
                " search minimizes, so it needs RUN_OPT_DEBIAS on."
            )
        names = metric_names if is_multiplier else objective_names
        for entry in entries:
            if not entry:
                raise ValueError(f"{list_key} has an empty entry, which adjusts nothing.")
            for name, value in entry:
                if name not in names:
                    raise ValueError(
                        f"{list_key} entry {dict(entry)!r}: {name!r} is not one of"
                        f" {names}."
                    )
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    or (is_multiplier and value < 0)
                ):
                    raise ValueError(
                        f"{list_key} entry {dict(entry)!r}: the value of {name!r}"
                        " must be a finite number"
                        + (" >= 0." if is_multiplier else ".")
                    )


# Callables notified of every run's outcome, as `listener(run, summary)`:
# `summary` is the dict `_finalize_trial()` stored in the run summary, or `None`
# when `_report_run_failure()` marked the run as not converged. The experiment
# script registers one to hand its runs' results to a W&B sweep trial, which
# cannot otherwise see them (see `frwo.sweep`).
RUN_RESULT_LISTENERS = []


def new_session_id():
    """
    Build the identifier shared by every W&B run of one execution of the
    experiment script.

    A single execution spreads its runs across many `run_bias_experiment()`
    calls, each of which gets its own group; this id is what ties them back
    together, so that the plotting notebook can select exactly one execution's
    results instead of mixing several.

    Returns
    -------
    session_id : str
        E.g. `"20260826-010125-3f9ab2"`. The random suffix keeps two executions
        started within the same second from being treated as one session.
    """
    return f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def start_wandb_run(
    exp_info,
    algorithm,
    bias_type,
    weight_adjust,
    trial_i,
    group,
    session_id,
    training_adjust=NO_TRAINING_ADJUST,
):
    """
    Start the W&B run that records one trial of one algorithm, one bias type,
    one weight adjustment and at most one training adjustment.

    Every algorithm reports through this same function, so that the runs of the
    Fairness Reward Weight Optimization technique and of the Superhuman
    Fairness baseline carry the same config keys, the same metric names and the
    same summary keys, and differ only by their `ALGORITHM`.

    Parameters
    ----------
    exp_info : dict
        Experiment parameters. Logged in full as the run's config, replacing
        the exp_info JSON files this pipeline used to write. It holds the
        listed bias types of the trial as `DATASET_BIAS_TYPE_LIST` and
        `DEMO_BIAS_TYPE_LIST`; the single one this run covers -- which may be
        the unbiased `()` that is always run and so is not listed there -- is
        recorded separately as `BIAS_TYPE`, and split by variety into
        `DATASET_BIAS_TYPE` and `DEMO_BIAS_TYPE` (`()` for the other).
    algorithm : str
        The technique this run covers, i.e. one entry of
        `exp_info["ALGORITHMS"]`. Recorded as `ALGORITHM` so that the two
        techniques' runs can be told apart and compared.
    bias_type : tuple
        The bias type this run covers, a dataset or a demo bias type; `()`
        for no bias.
    weight_adjusts : tuple
        The weight adjustment this run covers; `()` for the unadjusted
        weights.
    trial_i : int
        Index of the trial within the experiment.
    group : str
        W&B group shared by every run of the same experiment, so that trials,
        bias types and weight adjustments of one experiment stay together in
        the UI.
    session_id : str
        Identifies the execution of the experiment script this run belongs to.
        Recorded in the config so that the plotting notebook can select one
        execution's runs and never mix results from several.
    training_adjust : tuple, default NO_TRAINING_ADJUST
        The training adjustment of Fairness Reward Weight Optimization this
        run covers, as a `(kind, entry)` pair. Its entry is recorded as the
        kind's key of `TRAINING_ADJUST_KINDS` (e.g. `ALPHA_ADJUST`), as a list
        of `[name, value]` pairs, and every other kind's key as `[]`.

    Returns
    -------
    run : wandb.sdk.wandb_run.Run
        The started run. The caller is responsible for calling `finish()`.
    """
    bias_name = bias_type_name(bias_type)
    adjust_name = weight_adjusts_name(weight_adjust)
    training_kind, training_entry = training_adjust
    training_name = training_adjust_name(training_adjust)
    # The run name holds the whole training adjustment, but the job type and
    # tags, which W&B limits to 64 characters, only its kind.
    run_adjust_name = job_type = adjust_name
    adjust_tags = [adjust_name]
    if training_name:
        run_adjust_name = f"{adjust_name}__{training_name}"
        job_type = f"{adjust_name}__{training_kind}_adjust"
        adjust_tags.append(f"{training_kind}_adjust")
    kind = bias_type_kind(bias_type)
    dataset_bias_type = bias_type if kind == "dataset" else ()
    demo_bias_type = bias_type if kind == "demo" else ()
    algorithm_name = algorithm_slug(algorithm)

    config = {key: _json_safe(value) for key, value in exp_info.items()}
    config["ALGORITHM"] = algorithm
    config["ALGORITHM_NAME"] = algorithm_name
    config["BIAS_TYPE"] = _json_safe(bias_type)
    config["BIAS_TYPE_NAME"] = bias_name
    config["DATASET_BIAS_TYPE"] = _json_safe(dataset_bias_type)
    config["DATASET_BIAS_TYPE_NAME"] = bias_type_name(dataset_bias_type)
    config["DEMO_BIAS_TYPE"] = _json_safe(demo_bias_type)
    config["DEMO_BIAS_TYPE_NAME"] = bias_type_name(demo_bias_type)
    config["WEIGHT_ADJUST"] = _json_safe(weight_adjust)
    config["WEIGHT_ADJUST_NAME"] = adjust_name
    for adjust_kind, (_, record_key) in TRAINING_ADJUST_KINDS.items():
        entry = training_entry if adjust_kind == training_kind else ()
        config[record_key] = _json_safe(entry)
        config[f"{record_key}_NAME"] = training_adjust_name((adjust_kind, entry))
    config["TRIAL"] = trial_i
    config["SESSION_ID"] = session_id

    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group=group,
        job_type=job_type,
        name=f"{group}__{algorithm_name}__{bias_name}__{run_adjust_name}__trial{trial_i}",
        config=config,
        tags=[
            str(exp_info["DATASET"]),
            str(exp_info["EXPERT_ALGO"]),
            str(exp_info["IRL_METHOD"]),
            algorithm_name,
            bias_name,
            *adjust_tags,
        ],
    )


@dataclass
class ExpertDemos:
    """
    The expert demonstrations and feature expectations for one data split.

    Groups everything `_generate_expert_demonstrations()` produces for a split
    so that the train/validation/test expert data can be passed around as three
    objects instead of nine separate arrays. Any bias of the bias type these
    demonstrations belong to is already carried by them -- a dataset bias type
    by the dataset they were generated from, a demo bias type by their own
    `yhat` -- so there are no separate "unbiased" counterparts here.

    Attributes
    ----------
    mu : array-like<float>, shape(n_expert_demos, n_objectives)
        Feature expectations of the expert demonstrations.
    demo : pandas.DataFrame
        The expert demonstrations themselves.
    mu_perf : array-like<float>, shape(1, n_perf_objectives)
        Performance measures of the expert demonstrations.
    """

    mu: np.ndarray
    demo: pd.DataFrame
    mu_perf: np.ndarray


@dataclass
class BiasedDatasetDemos:
    """
    The biased dataset, data split and expert demonstrations of one bias type.

    `run_experiment_trial()` runs the unbiased dataset, every dataset bias
    type of `exp_info["DATASET_BIAS_TYPE_LIST"]` -- each one biased copy of
    the same dataset -- and every demo bias type of
    `exp_info["DEMO_BIAS_TYPE_LIST"]` -- each the unbiased dataset, with its
    expert's demonstrations biased -- all split along the same randomized
    indices. This groups everything that is specific to a single one of those
    bias types, so that the trial can iterate over one list instead of
    threading a dozen parallel lists through its loop.

    Attributes
    ----------
    bias_type : tuple
        The dataset or demo bias type these demonstrations were biased with;
        `()` for the unbiased dataset and demonstrations.
    X_train, X_val, X_test : pandas.DataFrame
        The input columns of each data split. Identical across bias types,
        except under a `dataset_noise_bias` with protected attribute noise.
    y_train, y_val, y_test : pandas.Series
        The (for a dataset bias type, biased) labels of each data split.
    expert_train, expert_val, expert_test : ExpertDemos
        The expert demonstrations of each data split.
    subdom_groups_train, subdom_groups_val, subdom_groups_test : tuple
        The subdominance groups of each split's expert demos, as
        `(group_idxs, raw_demos, raw_demos_feat_loss)` triples produced by
        `generate_subdominance_groups()`. They are sampled once and passed to
        every subsequent `compute_iteration_subdominance()` call, so that all
        subdominance computations of this bias type share the same groups and
        the same expert feature losses.
    expert_subdominance : dict<str, float>
        Each split's expert scored on its own subdominance groups, mapping each
        name in `SUBDOMINANCE_KEYS` to its value. Produced by
        `compute_expert_subdominance()`, which documents what these values mean.
        Computed here, once per bias type, since every run of this bias type is
        compared against the same experts.
    data_split_imbalance : float
        How unevenly the data split divides this bias type's dataset, as
        `_data_split_imbalance()` measures it.
    muE_split_cosine_dist : float
        How far apart the experts' feature expectations of the data splits
        are, as `_muE_split_cosine_dist()` measures it.
    """

    bias_type: tuple
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    expert_train: ExpertDemos
    expert_val: ExpertDemos
    expert_test: ExpertDemos
    subdom_groups_train: tuple
    subdom_groups_val: tuple
    subdom_groups_test: tuple
    expert_subdominance: dict
    data_split_imbalance: float
    muE_split_cosine_dist: float


class PolicyResults:
    """
    The evaluation of one learned policy.

    Owns every quantity `_evaluate_policy()` produces for a single set of
    reward weights, so that `run_experiment_trial()` and
    `_build_trial_summary()` exchange one object instead of thirty values.
    `record()` stores an `_evaluate_policy()` result and reports it to the
    active W&B run.

    Attributes
    ----------
    feat_obj_set_cols : list<str>
        Names of the feature expectation objectives, in order.
    perf_obj_set_cols : list<str>
        Names of the performance measure objectives, in order.
    weights : array-like<float>
        The reward weights the policy was learned from. Set by the caller
        before the policy is computed, not by `record()`.
    runtime : float
        Seconds spent learning and evaluating this policy.
    subdominance : dict<str, float>
        Maps each name in `SUBDOMINANCE_KEYS` to its value.
    """

    def __init__(self, feat_obj_set_cols, perf_obj_set_cols):
        self.feat_obj_set_cols = feat_obj_set_cols
        self.perf_obj_set_cols = perf_obj_set_cols

        self.weights = None
        self.runtime = None
        self.demo_train = None
        self.demo_val = None
        self.demo_test = None
        self.muL_train = None
        self.muL_val = None
        self.muL_test = None
        self.muL_perf_train = None
        self.muL_perf_val = None
        self.muL_perf_test = None
        self.t_train = None
        self.t_val = None
        self.t_test = None
        self.muL_delta_l2_train = None
        self.muL_delta_l2_val = None
        self.muL_delta_l2_test = None
        self.muL_delta_abs_l2_train = None
        self.muL_delta_abs_l2_val = None
        self.muL_delta_abs_l2_test = None
        self.subdominance = {key: None for key in SUBDOMINANCE_KEYS}

    def record(self, evaluate_policy_result, start, run):
        """
        Store an `_evaluate_policy()` result, then log it to the console and
        to W&B.

        Parameters
        ----------
        evaluate_policy_result : tuple
            The return value of `_evaluate_policy()`.
        start : datetime.datetime
            When this policy started being learned, used to report its runtime.
        run : wandb.sdk.wandb_run.Run or None
            The W&B run to log to. `None` disables W&B logging.
        """
        (
            self.demo_train,
            self.demo_val,
            self.demo_test,
            self.muL_train,
            self.muL_val,
            self.muL_test,
            self.muL_perf_train,
            self.muL_perf_val,
            self.muL_perf_test,
            muL_delta_train,
            self.muL_delta_l2_train,
            self.muL_delta_abs_l2_train,
            self.t_train,
            _muL_delta_val,
            self.muL_delta_l2_val,
            self.muL_delta_abs_l2_val,
            self.t_val,
            _muL_delta_test,
            self.muL_delta_l2_test,
            self.muL_delta_abs_l2_test,
            self.t_test,
            *subdominance,
        ) = evaluate_policy_result

        for key, value in zip(SUBDOMINANCE_KEYS, subdominance):
            self.subdominance[key] = value

        self.runtime = (datetime.datetime.now() - start).total_seconds()

        self._log(muL_delta_train, run)

    def _log(self, muL_delta_train, run):
        """Print the policy's stats and send them to the W&B run."""
        logging.info(
            f"\t\t muL_train \t\t= {str(np.round(self.muL_train, 2)).replace('0.', '.')}"
        )
        logging.debug(f"\t\t muL_val = {np.round(self.muL_val, 2)}")
        logging.debug(f"\t\t muL_test = {np.round(self.muL_test, 2)}")
        logging.info(f"\t\t t_train \t\t= {self.t_train:.5f}")
        logging.info(f"\t\t muL_delta_l2_train \t= {self.muL_delta_l2_train:.5f}")
        logging.info(
            f"\t\t muL_delta_abs_l2_train \t= {self.muL_delta_abs_l2_train:.5f}"
        )
        logging.info(
            f"\t\t muL_delta_train \t= {str(np.round(muL_delta_train, 2)).replace('0.', '.')}"
        )
        logging.info(
            f"\t\t weights \t= {str(np.round(self.weights, 2)).replace('0.', '.')}"
        )
        logging.info(f"\t\t Runtime for this policy: {self.runtime}")

        if run is None:
            return

        metrics = {
            "policy/t_train": self.t_train,
            "policy/t_val": self.t_val,
            "policy/t_test": self.t_test,
            "policy/mu_delta_l2_train": self.muL_delta_l2_train,
            "policy/mu_delta_l2_val": self.muL_delta_l2_val,
            "policy/mu_delta_l2_test": self.muL_delta_l2_test,
            "policy/mu_delta_abs_l2_train": self.muL_delta_abs_l2_train,
            "policy/mu_delta_abs_l2_val": self.muL_delta_abs_l2_val,
            "policy/mu_delta_abs_l2_test": self.muL_delta_abs_l2_test,
            "policy/runtime": self.runtime,
        }

        for key, value in self.subdominance.items():
            metrics[f"subdominance/{key}"] = value

        for j, col in enumerate(self.feat_obj_set_cols):
            metrics[f"weight/{col}"] = self.weights[j]
            metrics[f"muL_train/{col}"] = self.muL_train[j]
            metrics[f"muL_val/{col}"] = self.muL_val[j]
            metrics[f"muL_test/{col}"] = self.muL_test[j]
            metrics[f"muL_delta_train/{col}"] = muL_delta_train[j]

        for j, col in enumerate(self.perf_obj_set_cols):
            metrics[f"muL_perf_train/{col}"] = self.muL_perf_train[j]
            metrics[f"muL_perf_val/{col}"] = self.muL_perf_val[j]
            metrics[f"muL_perf_test/{col}"] = self.muL_perf_test[j]

        run.log(metrics)


def _build_trial_summary(
    feat_obj_set,
    perf_obj_set,
    bias_demos,
    unbiased_demos,
    results,
    trial_runtime,
    trial_inputsize,
):
    """
    Summarise one trial's learned policy as a flat metric dict.

    Most of the keys used here are the column names the results CSV used to
    have, so the same quantities are reported as before, addressed by name
    instead of by position in a result row. The `expert/` and `subdominance/`
    keys are newer and have no CSV counterpart.

    Parameters
    ----------
    feat_obj_set : frwo.irl.frwo.ObjectiveSet
        The set of objectives for the feature expectations.
    perf_obj_set : frwo.irl.frwo.ObjectiveSet
        The set of objectives for the performance measures.
    bias_demos : BiasedDatasetDemos
        The dataset, expert demonstrations and bias type this trial's policy
        was learned from.
    unbiased_demos : BiasedDatasetDemos
        The unbiased dataset and its expert demonstrations, reported alongside
        `bias_demos` so that a biased run's expert can be compared against the
        unbiased expert of the same data split.
    results : PolicyResults
        The evaluation of this trial's learned policy.
    trial_runtime : float
        The runtime to complete the trial, in seconds.
    trial_inputsize : int or None
        The size of the input space. Specifically `mdp.n_states_` (includes y).

    Returns
    -------
    summary : dict<str, numeric>
        The trial's results, ready to be stored in a W&B run summary.
    """
    summary = {}

    expert_train = bias_demos.expert_train
    expert_val = bias_demos.expert_val
    expert_test = bias_demos.expert_test

    expert_mus = (
        ("muE_train", expert_train.mu),
        ("muE_val", expert_val.mu),
        ("muE_test", expert_test.mu),
        ("muE_train_unbiased", unbiased_demos.expert_train.mu),
        ("muE_val_unbiased", unbiased_demos.expert_val.mu),
        ("muE_test_unbiased", unbiased_demos.expert_test.mu),
    )
    for prefix, muE in expert_mus:
        for i, obj in enumerate(feat_obj_set.objectives):
            summary[f"{prefix}_{obj.name}_mean"] = np.mean(muE[:, i])
            summary[f"{prefix}_{obj.name}_std"] = np.std(muE[:, i])

    # How alike the data splits are (see `_draw_data_split()`).
    summary["data_split_imbalance"] = bias_demos.data_split_imbalance
    summary["muE_split_cosine_dist"] = bias_demos.muE_split_cosine_dist

    for i, obj in enumerate(feat_obj_set.objectives):
        summary[f"wL_{obj.name}"] = results.weights[i]

    split_mus = (
        ("train", expert_train.mu, results.muL_train),
        ("val", expert_val.mu, results.muL_val),
        ("test", expert_test.mu, results.muL_test),
    )
    for split, muE, muL in split_mus:
        for i, obj in enumerate(feat_obj_set.objectives):
            summary[f"muL_{split}_{obj.name}"] = muL[i]
            summary[f"muL_{split}_err_{obj.name}"] = muL[i] - np.mean(muE[:, i])

    for key, value in results.subdominance.items():
        summary[key] = value

    # The expert's own subdominance, and how far the learned policy improved on
    # it. The delta is `expert - policy`, so that a positive value means the
    # policy is less subdominant -- better -- than the expert it imitated, in
    # the same direction as every other "improvement" reported here.
    for key, expert_value in bias_demos.expert_subdominance.items():
        summary[f"expert/{key}"] = expert_value
        summary[f"subdominance/delta_{key}"] = expert_value - results.subdominance[key]

    summary["muL_train_err_l2"] = results.muL_delta_l2_train
    summary["muL_val_err_l2"] = results.muL_delta_l2_val
    summary["muL_test_err_l2"] = results.muL_delta_l2_test
    summary["mu_delta_abs_l2_train"] = results.muL_delta_abs_l2_train
    summary["mu_delta_abs_l2_val"] = results.muL_delta_abs_l2_val
    summary["mu_delta_abs_l2_test"] = results.muL_delta_abs_l2_test

    # Training errors of the learned policy
    summary["t_train"] = results.t_train
    summary["t_val"] = results.t_val
    summary["t_test"] = results.t_test

    # Performance measures of the expert demos and of the learned policy
    expert_perf_mus = (
        ("muE_perf_train", expert_train.mu_perf),
        ("muE_perf_train_unbiased", unbiased_demos.expert_train.mu_perf),
        ("muE_perf_val", expert_val.mu_perf),
        ("muE_perf_val_unbiased", unbiased_demos.expert_val.mu_perf),
        ("muE_perf_test", expert_test.mu_perf),
        ("muE_perf_test_unbiased", unbiased_demos.expert_test.mu_perf),
    )
    split_perf_mus = (
        ("train", results.muL_perf_train),
        ("val", results.muL_perf_val),
        ("test", results.muL_perf_test),
    )
    for i, obj in enumerate(perf_obj_set.objectives):
        for prefix, muE_perf in expert_perf_mus:
            summary[f"{prefix}_{obj.name}"] = np.mean(muE_perf[:, i])
        for split, muL_perf in split_perf_mus:
            summary[f"muL_perf_{split}_{obj.name}"] = muL_perf[i]

    summary["trial_runtime"] = trial_runtime
    summary["policy_runtime"] = results.runtime
    summary["trial_inputsize"] = trial_inputsize

    return summary


def compute_subdominance(
    exp_info,
    alphas,
    raw_demos_feat_loss,
    clf_demos_feat_loss,
    relative=True,
    sum_agg=True,
    metric_weights=None,
):
    # Computation based on https://proceedings.mlr.press/v162/ziebart22a/ziebart22a.pdf eq. 5-8 and def. 5
    beta = 1.0
    if metric_weights is None:
        metric_weights = subdominance_metric_weights(exp_info)
    # compute the average across subdom for each demo
    total_subdom_list = []
    for demo_idx in range(raw_demos_feat_loss.shape[0]):
        subdom_list = []
        for feature_idx in range(raw_demos_feat_loss.shape[1]):
            alpha = alphas[feature_idx]
            raw_demo_loss = raw_demos_feat_loss[demo_idx][feature_idx]
            clf_demo_loss = clf_demos_feat_loss[demo_idx][feature_idx]
            if relative:
                epsilon = 1e-10  # to avoid divide by zero
                if raw_demo_loss < epsilon:
                    raw_demo_loss = epsilon
                subdom = np.maximum(
                    (alpha * ((clf_demo_loss / raw_demo_loss) - 1)) + beta, 0
                )
            else:
                subdom = np.maximum((alpha * (clf_demo_loss - raw_demo_loss)) + beta, 0)
            subdom_list.append(subdom)

        if sum_agg:
            # CUSTOM MODIFICATION FOR OUR USE CASE:
            # If we are aggregating by sum, weight each metric's subdominance
            # by `metric_weights` (every metric equally, unless skewed)
            total_subdom = sum(
                subdom * weight for subdom, weight in zip(subdom_list, metric_weights)
            )
        else:
            total_subdom = max(subdom_list)
        total_subdom_list.append(total_subdom)

    final_subdom = sum(total_subdom_list) / len(total_subdom_list)
    # Since we are using subdominance as an evaluation metric (not a training objective), we don't need to add regularization.
    # final_subdom += (lamda / 2.0) * (np.linalg.norm(alphas) ** 2)
    return final_subdom


def subdominance_metric_weights(exp_info, subdom_weight_adjust=()):
    """
    The weight of each subdominance metric in sum-aggregated subdominance, in
    the order of `subdominance_metric_names()`: every metric equally, each
    multiplied by its multiplier in `subdom_weight_adjust`, an entry of
    SUBDOM_WEIGHT_ADJUST_LIST (or `()` for none).
    """
    names = subdominance_metric_names(exp_info)
    return _adjust_vector(names, subdom_weight_adjust, 1.0) / len(names)


# Helper to sample the subdominance groups of a reference (expert) demo set once,
# so that every subsequent compute_iteration_subdominance() call against that
# reference reuses the same groups and the same reference feature losses.
def generate_subdominance_groups(exp_info, raw_demo_ref, group_idxs=None):
    """Sample the subdominance demo groups for one expert demo set.

    Parameters
    ----------
    exp_info : dict
        Metadata about the experiment.
    raw_demo_ref : pandas.DataFrame
        The expert (raw) demos the learned policy is scored against.
    group_idxs : list<numpy.ndarray>, optional
        The positional indices of each group, to use instead of sampling new
        ones; e.g. another demo set's, for one with the same rows.

    Returns
    -------
    group_idxs : list<numpy.ndarray>
        The positional indices of each subdominance group. Sampled once here so
        that every learned policy is compared against the expert on the exact
        same groups.
    raw_demos : list<pandas.DataFrame>
        The expert demos of each subdominance group.
    raw_demos_feat_loss : numpy.ndarray
        The feature losses of each expert demo group. Shape is
        (n_subdominance_groups, n_subdominance_metrics).
    """
    raw_demo = raw_demo_ref.copy()
    # OLD IMPLEMENTATION: Split the raw and clf demos into subdominance groups, where there are no repeated demos in each group.
    # raw_demos = np.array_split(raw_demo, exp_info["N_SUBDOMINANCE_GROUPS"])
    # clf_demos = np.array_split(clf_demo, exp_info["N_SUBDOMINANCE_GROUPS"])
    # NEW IMPLEMENTATION: Split the raw and clf demos into subdominance groups, where each group contains half of the demos,
    # sampled randomly without replacement. This way, each group contains a large and diverse set of demos, from which we can
    # more accurately and robustly compute the feature losses. This will help to improve the robustness of the subdominance
    # metric, especially when the number of demos is small.
    n_demos = len(raw_demo)
    group_size = n_demos // 2
    if group_idxs is None:
        group_idxs = [
            np.random.choice(n_demos, size=group_size, replace=False)
            for _ in range(exp_info["N_SUBDOMINANCE_GROUPS"])
        ]
    raw_demos = [raw_demo.iloc[group_idx] for group_idx in group_idxs]

    raw_demos_feat_loss = np.array(
        [
            compute_relevant_feat_loss(exp_info, raw_demo_group)
            for raw_demo_group in raw_demos
        ]
    )
    return group_idxs, raw_demos, raw_demos_feat_loss


def _subdominance_demo_groups(demo, group_idxs):
    """Select the rows of each subdominance group from `demo`.

    The feature losses only ever read the `y`, `yhat` and `z` columns, so when
    those are plain numpy-typed columns each group is just those three columns
    as numpy arrays. Every objective's `compute_feat_exp()` computes exactly
    the same values from them as from the full DataFrame rows, without the
    overhead of slicing the whole frame and running pandas operations on each
    group. Any other column type falls back to the DataFrame rows themselves.
    """
    cols = ("y", "yhat", "z")
    if all(
        col in demo.columns and isinstance(demo[col].dtype, np.dtype) for col in cols
    ):
        arrays = {}
        for col in cols:
            arr = demo[col].to_numpy()
            # Labels are small integers. Holding them as int8 leaves every
            # comparison and count the objectives make unchanged, but makes
            # gathering the rows of every group several times faster.
            int8 = np.iinfo(np.int8)
            if (
                arr.dtype.kind in "iu"
                and arr.size > 0
                and arr.min() >= int8.min
                and arr.max() <= int8.max
            ):
                arr = arr.astype(np.int8)
            arrays[col] = arr
        return [
            {col: arr[group_idx] for col, arr in arrays.items()}
            for group_idx in group_idxs
        ]
    return [demo.iloc[group_idx] for group_idx in group_idxs]


# Helper to compute subdominance for a specific set
def compute_iteration_subdominance(
    exp_info,
    group_idxs,
    raw_demos,
    raw_demos_feat_loss,
    clf_demo_cur,
    subdominance_type="all",
    alpha_lamda=SH_DEFAULT_LAMDA,
    alpha_adjust=(),
    subdom_weight_adjust=(),
):
    """Compute the subdominance metric of a learned policy's demos.

    `alpha_lamda` is the regularization of the per-measure `alpha` scaling (see
    `compute_alphas()`). Every subdominance reported as a result uses the
    default; only Fairness Reward Weight Optimization's `opt_debias` weight
    search passes its own, `OPT_DEBIAS_LAMDA`, for the objective it minimizes.

    `alpha_adjust` and `subdom_weight_adjust` are an entry of
    ALPHA_ADJUST_LIST and of SUBDOM_WEIGHT_ADJUST_LIST, which multiply the
    `alpha` scaling and the sum-aggregation weight of the metrics they name.
    Like `alpha_lamda`, only the `opt_debias` weight search passes them, and
    only in the runs of those training adjustments; every subdominance reported
    as a result is unadjusted.

    `group_idxs`, `raw_demos` and `raw_demos_feat_loss` come from
    `generate_subdominance_groups()`, which is called once per expert demo set in
    `_split_dataset_and_generate_expert_demos()`. Reusing them here keeps the
    subdominance groups (and the expert feature losses computed from them)
    identical across every call, so the subdominance values of different learned
    policies are directly comparable.

    `raw_demos` is not needed to compute the metric itself -- it is carried
    alongside its feature losses so callers hold the groups those losses came
    from.
    """
    # Compute subdominance metric for the learned policy, using the same
    # subdominance groups that the expert feature losses were computed on.
    clf_demos = _subdominance_demo_groups(clf_demo_cur, group_idxs)

    clf_demos_feat_loss = np.array(
        [
            compute_relevant_feat_loss(exp_info, clf_demo_group)
            for clf_demo_group in clf_demos
        ]
    )
    metric_names = subdominance_metric_names(exp_info)
    alphas = compute_alphas(
        raw_demos_feat_loss, clf_demos_feat_loss, lamda=alpha_lamda
    ) * _adjust_vector(metric_names, alpha_adjust, 1.0)
    metric_weights = subdominance_metric_weights(exp_info, subdom_weight_adjust)

    if subdominance_type == "max_abs" or subdominance_type == "all":
        # Compute max-aggregated absolute subdominance:
        max_abs_subdom = compute_subdominance(
            exp_info,
            alphas,
            raw_demos_feat_loss,
            clf_demos_feat_loss,
            relative=False,
            sum_agg=False,
        )
    if subdominance_type == "sum_abs" or subdominance_type == "all":
        # Compute sum-aggregated absolute subdominance:
        sum_abs_subdom = compute_subdominance(
            exp_info,
            alphas,
            raw_demos_feat_loss,
            clf_demos_feat_loss,
            relative=False,
            sum_agg=True,
            metric_weights=metric_weights,
        )
    if subdominance_type == "max_rel" or subdominance_type == "all":
        # Compute max-aggregated relative subdominance:
        max_rel_subdom = compute_subdominance(
            exp_info,
            alphas,
            raw_demos_feat_loss,
            clf_demos_feat_loss,
            relative=True,
            sum_agg=False,
        )
    if subdominance_type == "sum_rel" or subdominance_type == "all":
        # Compute sum-aggregated relative subdominance:
        sum_rel_subdom = compute_subdominance(
            exp_info,
            alphas,
            raw_demos_feat_loss,
            clf_demos_feat_loss,
            relative=True,
            sum_agg=True,
            metric_weights=metric_weights,
        )
    if subdominance_type == "max_abs":
        return max_abs_subdom
    elif subdominance_type == "sum_abs":
        return sum_abs_subdom
    elif subdominance_type == "max_rel":
        return max_rel_subdom
    elif subdominance_type == "sum_rel":
        return sum_rel_subdom
    elif subdominance_type == "all":
        return max_abs_subdom, sum_abs_subdom, max_rel_subdom, sum_rel_subdom
    else:
        raise ValueError(
            f"Invalid subdominance_type: {subdominance_type}. Must be one of ['max_abs', 'sum_abs', 'max_rel', 'sum_rel', 'all']"
        )


def compute_expert_subdominance(exp_info, splits):
    """Compute every subdominance measurement of the expert demonstrations.

    Each split's expert is scored exactly the way a learned policy is: its
    demonstrations are passed to `compute_iteration_subdominance()` as the
    decisions being measured, against the subdominance groups
    `generate_subdominance_groups()` sampled from those same demonstrations.
    This is the reference point the learned policies are compared against, so
    that a run's summary says how far its policy improved on the expert it
    imitated.

    Scoring the expert against its own groups is 1.0 in the ordinary case: the
    groups are built from the expert's own demonstrations, so the measured
    feature losses equal the reference feature losses in every group, and every
    aggregation of `compute_subdominance()` reduces to its `beta` margin of
    1.0. That makes 1.0 the "matches the expert" line -- a learned policy
    scoring below it is superhuman on that measurement, and one scoring above
    it is worse than the expert.

    The two relative measures can come out slightly below 1.0, though, which is
    why these values are computed here rather than hardcoded. A group whose
    expert feature loss is exactly 0 (a fairness metric that group satisfies
    perfectly, say) is clamped to `epsilon` in the denominator of
    `compute_subdominance()` but not in the numerator, so that term's ratio is
    0 instead of 1 and its subdominance is clipped to 0, pulling the group's
    relative measures under 1.0. The same clamp makes a learned policy's
    relative measures blow up on those groups, so `max_rel` and `sum_rel` are
    the shakiest of the four to read a delta from.

    Parameters
    ----------
    exp_info : dict
        Metadata about the experiment.
    splits : iterable<tuple>
        One `(split, expert, subdom_groups)` triple per data split, where
        `split` names the split ("train", "val" or "test"), `expert` is its
        `ExpertDemos` and `subdom_groups` is the
        `generate_subdominance_groups()` result of that expert.

    Returns
    -------
    subdominance : dict<str, float>
        Maps each name in `SUBDOMINANCE_KEYS` to the expert's value for it.
    """
    subdominance = {}
    for split, expert, subdom_groups in splits:
        values = compute_iteration_subdominance(
            exp_info, *subdom_groups, clf_demo_cur=expert.demo
        )
        for (agg, kind), value in zip(SUBDOMINANCE_AGGREGATIONS, values):
            subdominance[f"{agg}_{kind}_subdominance_{split}"] = value

    return subdominance


def _build_objective_sets(exp_info):
    feat_obj_set = ObjectiveSet(
        [OBJ_LOOKUP_BY_NAME[name]() for name in exp_info["FEAT_EXP_OBJECTIVE_NAMES"]]
    )
    feat_obj_set.reset()

    perf_obj_set = ObjectiveSet(
        [OBJ_LOOKUP_BY_NAME[name]() for name in exp_info["PERF_MEAS_OBJECTIVE_NAMES"]]
    )
    perf_obj_set.reset()

    return feat_obj_set, perf_obj_set


def _load_or_generate_dataset(exp_info, X, y, feature_types):
    if X is None or y is None or feature_types is None:
        return generate_dataset(
            exp_info["DATASET"],
            n_samples=exp_info["N_DATASET_SAMPLES"],
        )
    return X, y, feature_types


def _generate_expert_demonstrations(
    exp_info,
    X,
    y,
    expert_algo_lookup,
    feat_obj_set,
    perf_obj_set,
):
    """Generate the expert demonstrations for one data split.

    Whatever bias a dataset bias type applies is already carried by `X` and
    `y` (see `_split_dataset_and_generate_expert_demos()`), so the expert is
    fit on the biased dataset and its demonstrations are scored against it.
    Demo bias types are not generated here: see `_bias_expert_demonstrations()`.

    Returns
    -------
    expert : ExpertDemos
        The demonstrations, feature expectations and performance measures of
        this split.
    """
    clf = copy.deepcopy(expert_algo_lookup[exp_info["EXPERT_ALGO"]])
    muE, demoE = generate_mu_and_demos(
        exp_info,
        X=X,
        y=y,
        clf=clf,
        obj_set=feat_obj_set,
        n_demos=exp_info["N_EXPERT_DEMOS"],
    )
    return ExpertDemos(
        mu=muE,
        demo=demoE,
        mu_perf=np.array([perf_obj_set.compute_demo_feature_exp(demoE)]),
    )


def _bias_expert_demonstrations(
    exp_info,
    unbiased_experts,
    demo_bias_type,
    feature_types,
    feat_obj_set,
    perf_obj_set,
):
    """Apply a demo bias type to the unbiased expert demonstrations of every
    split.

    The expert is not refit: its unbiased demonstrations are reused and only
    their decisions (`yhat`) are biased, so the demonstrations of a demo bias
    type differ from the unbiased ones by exactly the bias. Their feature
    expectations and performance measures are those of the biased
    demonstrations, computed as `generate_mu_and_demos()` and
    `_generate_expert_demonstrations()` compute them.

    Most demo bias types bias each split's demonstrations independently.
    `demo_corruption_bias` instead biases them all together, so that every
    split is biased by one corrupted expert, as `dataset_corruption_bias`
    relabels the whole dataset with one corrupted classifier before it is
    split.

    Parameters
    ----------
    unbiased_experts : sequence<ExpertDemos>
        The unbiased expert demonstrations of each split.

    Returns
    -------
    experts : list<ExpertDemos>
        The biased expert demonstrations of each split, in the order of
        `unbiased_experts`.
    """
    unbiased_demos = [expert.demo for expert in unbiased_experts]
    if demo_bias_type[0] == "demo_corruption_bias":
        # Split the biased demonstrations back apart by position, since the
        # dataset's index needn't be unique.
        biased = add_demo_bias(
            pd.concat(unbiased_demos),
            demo_bias_type,
            dataset=exp_info["DATASET"],
            feature_types=feature_types,
        )
        bounds = np.cumsum([0] + [len(demo) for demo in unbiased_demos])
        demos = [
            biased.iloc[start:end].copy() for start, end in zip(bounds, bounds[1:])
        ]
    else:
        demos = [
            add_demo_bias(
                demo,
                demo_bias_type,
                dataset=exp_info["DATASET"],
                feature_types=feature_types,
            )
            for demo in unbiased_demos
        ]
    return [
        ExpertDemos(
            mu=np.array([feat_obj_set.compute_demo_feature_exp(demo)]),
            demo=demo,
            mu_perf=np.array([perf_obj_set.compute_demo_feature_exp(demo)]),
        )
        for demo in demos
    ]


def _muE_split_cosine_dist(muE_train, muE_val, muE_test):
    """The largest cosine distance between the training split's expert feature
    expectations and the validation or test split's.

    Only reported, never used to choose the split: these feature expectations
    are those of experts fit on each split, so their distance also reflects
    how each expert's fit happened to turn out, which re-splitting cannot fix
    and should not select for. See `_draw_data_split()` for what the split is
    chosen by instead.
    """
    return max(
        cosine(muE_train[demo_i], muE_other[0])
        for demo_i in range(len(muE_train))
        for muE_other in (muE_val, muE_test)
    )


def _bias_dataset(exp_info, X, y, feature_types, bias_type):
    """Apply one bias type to a dataset.

    Only dataset bias types change the dataset; `()` (the unbiased "bias
    type") and demo bias types return it unchanged.

    Returns
    -------
    X_biased : pandas.DataFrame
        `X`, or a biased copy of it (`dataset_noise_bias` with protected
        attribute noise).
    y_biased : pandas.Series
        A copy of `y` with `bias_type` applied, indexed and named like `y`.
    """
    X_biased, y_biased = add_dataset_bias(
        X, y, feature_types, dataset_bias_type=bias_type, dataset=exp_info["DATASET"]
    )

    if bias_type_kind(bias_type) == "dataset":
        logging.info(f"Dataset bias type added: {bias_type}")
        pct_unchanged = (np.asarray(y) == np.asarray(y_biased)).mean() * 100.0
        logging.info(f"Percent of y unchanged with added bias: {pct_unchanged}%")
        if X_biased is not X:
            pct_unchanged = (
                np.asarray(X["z"]) == np.asarray(X_biased["z"])
            ).mean() * 100.0
            logging.info(f"Percent of z unchanged with added bias: {pct_unchanged}%")

    return X_biased, y_biased


# The (train, val, test) fractions of the dataset split, unless
# exp_info["DATA_SPLIT_FRACTIONS"] overrides them.
DEFAULT_DATA_SPLIT_FRACTIONS = (0.60, 0.20, 0.20)


def _data_split_seed(exp_info, trial_i):
    """
    Derive the seed of a trial's data split from what makes the trial unique.

    Like `_baseline_seed()`, this keeps the split off the global numpy random
    state, so however many candidate splits `_draw_data_split()` draws, no
    later random draw shifts. The expert is deliberately left out of the key:
    every expert of a trial is split along the same indices, so the experts
    are compared on the same data.
    """
    key = "|".join(
        [
            str(exp_info.get("RANDOM_SEED")),
            str(exp_info.get("EXPERIMENT_NAME")),
            str(exp_info.get("DATASET")),
            "data_split",
            str(trial_i),
        ]
    )
    # `np.random.RandomState` takes 32-bit seeds.
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=4).digest(), "big")


@contextlib.contextmanager
def _bias_type_random_state(exp_info, bias_type, trial_i, stage):
    """
    Make a bias type's random draws from the global numpy and `random`
    generators from a seed of its own, then put both back as they were.

    The unbiased run draws from the global generators, whose state carries
    from one trial to the next. If a bias type drew from them too -- to bias
    the dataset or the demonstrations, sample its subdominance groups, or train
    on it -- every later unbiased run would draw differently, so an experiment
    config's unbiased results would depend on which bias types it lists. With
    this, they are those of a config that lists none, as long as both draw the
    same data split (a dataset bias type can change which one is chosen; see
    `_draw_data_split()`). Each bias type's own draws are likewise independent
    of which other bias types are listed. `()` is left on the global
    generators.

    Like `_data_split_seed()`, the seed leaves the expert out, so that every
    expert's demonstrations are biased alike.

    Parameters
    ----------
    stage : str
        Which of a bias type's steps is drawing, so that each step gets a seed
        of its own.
    """
    if bias_type == ():
        yield
        return

    key = "|".join(
        [
            str(exp_info.get("RANDOM_SEED")),
            str(exp_info.get("EXPERIMENT_NAME")),
            str(exp_info.get("DATASET")),
            "bias_type",
            bias_type_name(bias_type),
            stage,
            str(trial_i),
        ]
    )
    with _seeded_global_random_state(key):
        yield


@contextlib.contextmanager
def _training_adjust_random_state(
    exp_info, bias_type, weight_adjust, training_adjust, trial_i
):
    """
    Make the random draws of one training adjustment's run of Fairness Reward
    Weight Optimization from a seed of its own, then put the global numpy and
    `random` generators back as they were.

    Like `_bias_type_random_state()` does for the bias types, this keeps the
    runs without a training adjustment drawing exactly what they would if no
    training adjustment were listed, and each training adjustment's run
    independent of which others are.
    """
    key = "|".join(
        [
            str(exp_info.get("RANDOM_SEED")),
            str(exp_info.get("EXPERIMENT_NAME")),
            str(exp_info.get("DATASET")),
            "training_adjust",
            bias_type_name(bias_type),
            weight_adjusts_name(weight_adjust),
            training_adjust_name(training_adjust),
            str(trial_i),
        ]
    )
    with _seeded_global_random_state(key):
        yield


@contextlib.contextmanager
def _seeded_global_random_state(key):
    """
    Seed the global numpy and `random` generators from `key`, a string, then
    put both back as they were.
    """
    # `np.random.seed()` takes 32-bit seeds.
    seed = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=4).digest(), "big")
    np_state, py_state = np.random.get_state(), random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


def _data_split_stratify_labels(exp_info, X, y):
    """
    The labels `exp_info["DATA_SPLIT_STRATIFY"]` stratifies the split by.

    `False` (the default) doesn't stratify, `True` or `"y"` stratifies by the
    unbiased label, and `"zy"` by each combination of the protected attribute
    and the unbiased label.

    Returns
    -------
    labels : numpy.ndarray or None
        One label per row of `X`, or None to not stratify.
    """
    stratify = exp_info.get("DATA_SPLIT_STRATIFY", False)
    if stratify is False or stratify is None:
        return None
    if stratify is True or stratify == "y":
        return np.asarray(y)
    if stratify == "zy":
        _, labels = np.unique(
            np.column_stack([np.asarray(X["z"]), np.asarray(y)]),
            axis=0,
            return_inverse=True,
        )
        return labels.ravel()
    raise ValueError(
        "exp_info['DATA_SPLIT_STRATIFY'] must be false, true, 'y' or 'zy', got"
        f" {stratify!r}."
    )


def _split_group_rates(z, y, idxs, z_values):
    """
    `P(z = v)` and `P(y = 1 | z = v)` of the rows `idxs`, for every value `v`
    of the protected attribute: the rates the feature expectations of a
    split's demonstrations depend on.

    Returns
    -------
    rates : numpy.ndarray or None
        The rates, or None if some group has no rows among `idxs`.
    """
    zs, ys = z[idxs], y[idxs]
    rates = []
    for z_value in z_values:
        in_group = zs == z_value
        if not in_group.any():
            return None
        rates += [in_group.mean(), ys[in_group].mean()]
    return np.array(rates)


def _data_split_imbalance(datasets, split_idxs):
    """
    How differently the data is distributed across the splits.

    Parameters
    ----------
    datasets : iterable<tuple>
        `(X, y)` pairs of the datasets split along `split_idxs`.
    split_idxs : tuple<numpy.ndarray>
        The row indices of each split.

    Returns
    -------
    imbalance : float
        The largest spread across the splits (max - min) of any
        `_split_group_rates()` rate, over every dataset. Infinite if some
        split lacks one of the groups altogether.
    """
    imbalance = 0.0
    for X_biased, y_biased in datasets:
        z, y_values = np.asarray(X_biased["z"]), np.asarray(y_biased)
        z_values = np.unique(z)
        rates = [_split_group_rates(z, y_values, idxs, z_values) for idxs in split_idxs]
        if any(split_rates is None for split_rates in rates):
            return np.inf
        imbalance = max(imbalance, np.ptp(rates, axis=0).max())
    return float(imbalance)


def _draw_data_split(exp_info, X, y, datasets, trial_i):
    """
    Draw the train/validation/test split every bias type of a trial shares.

    So that a policy scoring worse on the test split than on the training
    split reflects the policy rather than a difference between the splits,
    the split is chosen for how evenly the protected groups and their labels
    are distributed across it, in every dataset it splits (see
    `_data_split_imbalance()`). Up to `exp_info["DATA_SPLIT_MAX_TRIES"]`
    candidate splits are drawn, and the first whose imbalance is at most
    `exp_info["DATA_SPLIT_MAX_IMBALANCE"]` is taken -- or, if none is, the
    most balanced one. This only needs the data, so no expert is fit for a
    rejected split.

    The candidates are drawn from a generator seeded by `_data_split_seed()`,
    so the split is reproducible and leaves the global random state alone.

    Parameters
    ----------
    X : pandas.DataFrame
        The unbiased dataset's inputs.
    y : pandas.Series
        The unbiased dataset's labels.
    datasets : iterable<tuple>
        The `(X, y)` pairs whose imbalance the split is chosen by.
    trial_i : int
        Index of this trial within the experiment.

    Returns
    -------
    split_idxs : tuple<numpy.ndarray>
        The `(train_idxs, val_idxs, test_idxs)` row indices of the split.
    imbalance : float
        Its `_data_split_imbalance()`.
    """
    train_frac, val_frac, test_frac = exp_info.get(
        "DATA_SPLIT_FRACTIONS", DEFAULT_DATA_SPLIT_FRACTIONS
    )
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError(
            "exp_info['DATA_SPLIT_FRACTIONS'] must sum to 1, got"
            f" {(train_frac, val_frac, test_frac)}."
        )
    max_tries = exp_info.get("DATA_SPLIT_MAX_TRIES", 1)
    if max_tries < 1:
        raise ValueError(
            f"exp_info['DATA_SPLIT_MAX_TRIES'] must be at least 1, got {max_tries}."
        )
    max_imbalance = exp_info.get("DATA_SPLIT_MAX_IMBALANCE", np.inf)
    stratify_labels = _data_split_stratify_labels(exp_info, X, y)
    random_state = np.random.RandomState(_data_split_seed(exp_info, trial_i))

    idxs = np.arange(len(X))
    best_split_idxs, best_imbalance = None, np.inf
    for try_i in range(max_tries):
        train_idxs, val_test_idxs = train_test_split(
            idxs,
            train_size=train_frac,
            stratify=stratify_labels,
            random_state=random_state,
        )
        val_idxs, test_idxs = train_test_split(
            val_test_idxs,
            test_size=test_frac / (val_frac + test_frac),
            stratify=(
                None if stratify_labels is None else stratify_labels[val_test_idxs]
            ),
            random_state=random_state,
        )
        split_idxs = (train_idxs, val_idxs, test_idxs)
        imbalance = _data_split_imbalance(datasets, split_idxs)
        if best_split_idxs is None or imbalance < best_imbalance:
            best_split_idxs, best_imbalance = split_idxs, imbalance
        if imbalance <= max_imbalance:
            logging.info(
                f"Data split {try_i + 1} of at most {max_tries} accepted, with"
                f" group rate imbalance {imbalance:.4f}."
            )
            break
    else:
        logging.info(
            f"No data split of {max_tries} had a group rate imbalance of at most"
            f" {max_imbalance}; using the most balanced one, with imbalance"
            f" {best_imbalance:.4f}."
        )
    return best_split_idxs, best_imbalance


def _split_dataset_and_generate_expert_demos(
    exp_info,
    expert_algo_lookup,
    feat_obj_set,
    perf_obj_set,
    X,
    y,
    feature_types,
    bias_type_list,
    trial_i=0,
):
    """Bias the dataset once per bias type, then split each biased dataset and
    generate expert demonstrations from its splits.

    A dataset bias type is applied to the dataset -- its `y` values, or for
    `dataset_noise_bias` optionally its protected attribute `z` -- before the
    data is split, so the expert of that bias type is fit on the biased
    dataset and every metric of that bias type -- the expert's own feature
    expectations, its subdominance groups, and the learned policies scored
    against them -- is computed against it.

    A demo bias type leaves the dataset unchanged. It reuses the unbiased
    expert demonstrations of every split, and their subdominance groups, and
    biases only their decisions (`yhat`), so that it differs from the unbiased
    run by exactly that bias; its expert's feature expectations, subdominance
    groups' losses and every metric computed against them are those of the
    biased demonstrations.

    Every dataset is split along the same randomized indices, so the bias
    types differ from each other only by their bias and their results stay
    comparable. Those indices are chosen by `_draw_data_split()`, for how
    evenly every dataset is distributed across them.

    Returns
    -------
    demos_by_bias_type : list<BiasedDatasetDemos>
        The biased data split, expert demonstrations and subdominance groups of
        each bias type, in the order of `bias_type_list`. Since
        `bias_type_list` starts with `()`, the first entry always holds the
        unbiased dataset and its demonstrations.
    """
    # One biased copy of the dataset per dataset bias type. `()` and the demo
    # bias types leave it untouched.
    logging.info("Applying each dataset bias type to the dataset...")
    biased_datasets = []
    for bias_type in bias_type_list:
        with _bias_type_random_state(exp_info, bias_type, trial_i, "dataset"):
            biased_datasets.append(
                _bias_dataset(exp_info, X, y, feature_types, bias_type)
            )

    # One randomized split, shared by every bias type: the biased datasets
    # differ from each other only by their bias, so splitting them all along
    # the same indices keeps their results comparable. It is chosen for how
    # evenly it splits the unbiased dataset and every dataset bias type's
    # biased copy; the demo bias types' datasets are the unbiased one.
    #
    # DATA_SPLIT_FRACTIONS sets the (train, val, test) fractions,
    # DATA_SPLIT_STRATIFY what both splits are stratified by, and
    # DATA_SPLIT_MAX_TRIES and DATA_SPLIT_MAX_IMBALANCE how hard
    # _draw_data_split() looks for an even split.
    logging.info("Drawing the train/validation/test split...")
    (train_idxs, val_idxs, test_idxs), _ = _draw_data_split(
        exp_info,
        X,
        y,
        [
            dataset
            for bias_type, dataset in zip(bias_type_list, biased_datasets)
            if bias_type_kind(bias_type) != "demo"
        ],
        trial_i,
    )

    # The demo bias types reuse the unbiased run's demonstrations, so it must
    # come first.
    assert bias_type_list[0] == ()
    demos_by_bias_type = []
    for bias_type, (X_biased, y_biased) in zip(bias_type_list, biased_datasets):
        with _bias_type_random_state(exp_info, bias_type, trial_i, "demos"):
            logging.info(f"Generating expert demonstrations for bias type: {bias_type}")

            X_train = X_biased.iloc[train_idxs]
            X_val = X_biased.iloc[val_idxs]
            X_test = X_biased.iloc[test_idxs]
            y_train = y_biased.iloc[train_idxs]
            y_val = y_biased.iloc[val_idxs]
            y_test = y_biased.iloc[test_idxs]

            if bias_type_kind(bias_type) == "demo":
                # A demo bias type biases the unbiased run's demonstrations
                # rather than refitting the expert, and reuses its subdominance
                # groups, so that it differs from the unbiased run by exactly its
                # bias.
                unbiased = demos_by_bias_type[0]
                logging.info("Biasing the unbiased expert demonstrations...")
                expert_train, expert_val, expert_test = _bias_expert_demonstrations(
                    exp_info,
                    (unbiased.expert_train, unbiased.expert_val, unbiased.expert_test),
                    bias_type,
                    feature_types,
                    feat_obj_set,
                    perf_obj_set,
                )
                group_idxs_train, group_idxs_val, group_idxs_test = (
                    unbiased.subdom_groups_train[0],
                    unbiased.subdom_groups_val[0],
                    unbiased.subdom_groups_test[0],
                )
            else:
                logging.info(
                    "Generating expert demonstrations and feature expectations for training set..."
                )
                expert_train = _generate_expert_demonstrations(
                    exp_info,
                    X_train,
                    y_train,
                    expert_algo_lookup,
                    feat_obj_set,
                    perf_obj_set,
                )

                logging.info(
                    "Generating expert demonstrations and feature expectations for validation set..."
                )
                expert_val = _generate_expert_demonstrations(
                    exp_info,
                    X_val,
                    y_val,
                    expert_algo_lookup,
                    feat_obj_set,
                    perf_obj_set,
                )

                logging.info(
                    "Generating expert demonstrations and feature expectations for test set..."
                )
                expert_test = _generate_expert_demonstrations(
                    exp_info,
                    X_test,
                    y_test,
                    expert_algo_lookup,
                    feat_obj_set,
                    perf_obj_set,
                )
                group_idxs_train = group_idxs_val = group_idxs_test = None

            # How evenly the split divides this bias type's dataset, and how close
            # its experts' feature expectations came out, for every run of this
            # bias type to report.
            data_split_imbalance = _data_split_imbalance(
                [(X_biased, y_biased)], (train_idxs, val_idxs, test_idxs)
            )
            muE_split_cosine_dist = _muE_split_cosine_dist(
                expert_train.mu, expert_val.mu, expert_test.mu
            )
            logging.info(
                f"Data split group rate imbalance: {data_split_imbalance:.4f};"
                f" muE cosine distance across splits: {muE_split_cosine_dist:.6f}"
            )

            # Sample the subdominance groups (and compute the expert feature losses
            # on them) once per split and bias type, so that every
            # compute_iteration_subdominance() call of that bias type reuses the
            # same groups. A demo bias type takes the unbiased run's groups.
            logging.info("Generating subdominance groups for the expert demonstrations...")
            subdom_groups_train = generate_subdominance_groups(
                exp_info, expert_train.demo, group_idxs_train
            )
            subdom_groups_val = generate_subdominance_groups(
                exp_info, expert_val.demo, group_idxs_val
            )
            subdom_groups_test = generate_subdominance_groups(
                exp_info, expert_test.demo, group_idxs_test
            )

            # Score the experts on their own groups, so that every run of this bias
            # type can report how far its learned policy improved on the expert it
            # imitated.
            logging.info("Computing the subdominance of the expert demonstrations...")
            expert_subdominance = compute_expert_subdominance(
                exp_info,
                (
                    ("train", expert_train, subdom_groups_train),
                    ("val", expert_val, subdom_groups_val),
                    ("test", expert_test, subdom_groups_test),
                ),
            )

            demos_by_bias_type.append(
                BiasedDatasetDemos(
                    bias_type=bias_type,
                    X_train=X_train,
                    X_val=X_val,
                    X_test=X_test,
                    y_train=y_train,
                    y_val=y_val,
                    y_test=y_test,
                    expert_train=expert_train,
                    expert_val=expert_val,
                    expert_test=expert_test,
                    subdom_groups_train=subdom_groups_train,
                    subdom_groups_val=subdom_groups_val,
                    subdom_groups_test=subdom_groups_test,
                    expert_subdominance=expert_subdominance,
                    data_split_imbalance=data_split_imbalance,
                    muE_split_cosine_dist=muE_split_cosine_dist,
                )
            )

    return demos_by_bias_type


def _compute_errors_and_metrics(
    wi,
    muE_train,
    muE_val,
    muE_test,
    muL_train,
    muL_val,
    muL_test,
    exp_info,
):
    """Compute the learned policy's feature expectation error metrics for the
    train/validation/test sets."""
    (
        muL_delta_train,
        muL_delta_l2_train,
        muL_delta_abs_l2_train,
        ti_train,
    ) = policy_error(
        wi,
        muE_train,
        muL_train,
        dot_weights_feat_exp=exp_info["DOT_WEIGHTS_FEAT_EXP"],
    )
    (
        muL_delta_val,
        muL_delta_l2_val,
        muL_delta_abs_l2_val,
        ti_val,
    ) = policy_error(
        wi,
        muE_val,
        muL_val,
        dot_weights_feat_exp=exp_info["DOT_WEIGHTS_FEAT_EXP"],
    )
    (
        muL_delta_test,
        muL_delta_l2_test,
        muL_delta_abs_l2_test,
        ti_test,
    ) = policy_error(
        wi,
        muE_test,
        muL_test,
        dot_weights_feat_exp=exp_info["DOT_WEIGHTS_FEAT_EXP"],
    )

    return (
        muL_delta_train,
        muL_delta_l2_train,
        muL_delta_abs_l2_train,
        ti_train,
        muL_delta_val,
        muL_delta_l2_val,
        muL_delta_abs_l2_val,
        ti_val,
        muL_delta_test,
        muL_delta_l2_test,
        muL_delta_abs_l2_test,
        ti_test,
    )


# Defaults for Fairness Reward Weight Optimization's own settings, overridable
# through the correspondingly named `exp_info` keys.
#
# The FRWO_CLF_* settings configure the random forest that predicts `y` from
# `x` for the policies Fairness Reward Weight Optimization learns. Since those
# policies cannot observe `y`, it decides which MDP state each row is in. The
# defaults are scikit-learn's. The expert has its own, separately fit random
# forest (see `OptClfMDPPolicyExpert`), which these leave untouched.
#
# OPT_DEBIAS_LAMDA is the regularization of the `alpha` scaling of the
# subdominance the `opt_debias` weight search minimizes -- the counterpart of
# the Superhuman Fairness baselines' SH_*LAMDA. It does not change the
# subdominance reported as a result, which always uses `SH_DEFAULT_LAMDA`.
FRWO_DEFAULTS = {
    "FRWO_CLF_N_ESTIMATORS": 100,
    "FRWO_CLF_MAX_DEPTH": None,
    "FRWO_CLF_MIN_SAMPLES_LEAF": 1,
    "OPT_DEBIAS_LAMDA": SH_DEFAULT_LAMDA,
}


def _frwo_setting(exp_info, key):
    """`exp_info[key]`, or its `FRWO_DEFAULTS` value if unset."""
    return exp_info[key] if key in exp_info else FRWO_DEFAULTS[key]


def opt_debias_lamda(exp_info):
    """The `alpha` regularization of the `opt_debias` objective."""
    return _frwo_setting(exp_info, "OPT_DEBIAS_LAMDA")


def opt_debias_training_adjust(exp_info, kind):
    """
    The entry of the training adjustment of `kind` ("alpha" or
    "subdom_weight") that skews the `opt_debias` objective, or `()` for none.

    `exp_info` holds it under the kind's key of `TRAINING_ADJUST_KINDS` (e.g.
    ALPHA_ADJUST) only in the copy `_run_frwo_policy()` hands the `opt_debias`
    weight search of that training adjustment's run, so that nothing else --
    in particular no reported subdominance -- is skewed by it.
    """
    return exp_info.get(TRAINING_ADJUST_KINDS[kind][1], ())


def apply_manual_weight_adjust(wi, manual_weight_adjust, objective_names):
    """
    Add the terms of `manual_weight_adjust`, an entry of
    MANUAL_WEIGHT_ADJUST_LIST, to the reward weights `wi` they name, and L1
    normalize the result.

    Parameters
    ----------
    wi : numpy.ndarray
        L1 normalized reward weights, in the order of `objective_names`.
    manual_weight_adjust : tuple
        `(objective name, term)` pairs.
    objective_names : list<str>
        The reward objective each weight of `wi` is for.

    Returns
    -------
    wi : numpy.ndarray
        The adjusted weights, L1 normalized. If the terms cancel every weight
        out, every weight is set to 1 before normalizing, i.e. the weights
        become equal.
    """
    adjusted = np.asarray(wi, dtype=float) + _adjust_vector(
        objective_names, manual_weight_adjust, 0.0
    )
    if not np.any(adjusted):
        adjusted = np.ones_like(adjusted)
    return normalize(adjusted.reshape(1, -1), norm="l1").flatten()


def _fit_clf_and_demo_df(exp_info, feature_types, X_train, y_train):
    """Fit a y|x predictor and build the demo DataFrame used by compute_optimal_policy."""
    clf = sklearn_clf_pipeline(
        feature_types=feature_types,
        clf_inst=RandomForestClassifier(
            n_estimators=_frwo_setting(exp_info, "FRWO_CLF_N_ESTIMATORS"),
            max_depth=_frwo_setting(exp_info, "FRWO_CLF_MAX_DEPTH"),
            min_samples_leaf=_frwo_setting(exp_info, "FRWO_CLF_MIN_SAMPLES_LEAF"),
        ),
    )
    clf.fit(X_train, y_train)
    demo_df = pd.DataFrame(X_train)
    demo_df["y"] = y_train
    return clf, demo_df


def _apply_weight_adjustments(
    wi,
    weight_adjust,
    feat_obj_set,
    demo_df,
    clf,
    x_cols,
    exp_info,
    X,
    y,
    can_observe_y,
    subdom_groups,
    run=None,
):
    """Apply the weight adjustment `weight_adjust` to wi and return the result.

    `weight_adjust` is `()` for the unadjusted weights, or an `opt_debias`
    entry of `exp_info["WEIGHT_ADJUST_LIST"]`, which optimizes them.

    `run` is the W&B run of the policy these weights are being learned for, or
    `None` to disable W&B logging. Only `opt_debias` reports anything to it.
    """
    if weight_adjust:
        library = weight_adjust[1]
        optimizer = weight_adjust[2]
        n_steps = weight_adjust[3] if 3 < len(weight_adjust) else 200
        wi = optimize_weights(
            normalize(np.asarray(wi, dtype=float).reshape(1, -1), norm="l1").flatten(),
            feat_obj_set,
            demo_df,
            clf,
            x_cols,
            exp_info,
            X,
            y,
            can_observe_y,
            subdom_groups,
            library,
            optimizer,
            n_steps,
            step_logger=_OptDebiasStepLogger(run),
        )
    wi = normalize(wi.reshape(1, -1), norm="l1").flatten()
    return wi


class _OptDebiasStepLogger:
    """Report every optimizer step of `optimize_weights()` to the console and W&B.

    Every optimizer library `optimize_weights()` supports evaluates a candidate
    weight set by calling `subdominance_of_weights()` exactly once per step, so
    that function hands each step's weights to `record()`. That keeps the
    reporting independent of which library and optimizer are used.

    In W&B, every `opt_debias_step/` metric is plotted against
    `opt_debias_step/step`, so each graph holds one point per step.
    """

    def __init__(self, run):
        self.run = run
        self.step = 0
        self.best_wi = None
        self.best_subdom = np.inf
        if run is not None:
            run.define_metric("opt_debias_step/step")
            run.define_metric("opt_debias_step/*", step_metric="opt_debias_step/step")

    def record(self, wi, subdom):
        """Report one optimizer step.

        Parameters
        ----------
        wi : numpy.ndarray
            The L1 normalized weights evaluated at this step.
        subdom : float
            The sum-aggregated absolute subdominance of those weights' demos.
        """
        if subdom < self.best_subdom:
            self.best_subdom = subdom
            self.best_wi = wi

        logging.info(f"\t\t opt_debias step {self.step}")
        logging.info(
            f"\t\t\t current weights \t= {str(np.round(wi, 2)).replace('0.', '.')}"
            f" \t sum_abs_subdominance = {subdom:.5f}"
        )
        logging.info(
            f"\t\t\t best weights \t\t= "
            f"{str(np.round(self.best_wi, 2)).replace('0.', '.')}"
            f" \t sum_abs_subdominance = {self.best_subdom:.5f}"
        )

        if self.run is not None:
            self.run.log(
                {
                    "opt_debias_step/step": self.step,
                    "opt_debias_step/sum_abs_subdominance": subdom,
                    "opt_debias_step/best_sum_abs_subdominance": self.best_subdom,
                }
            )

        self.step += 1


def optimize_weights(
    wi,
    feat_obj_set,
    demo_df,
    clf,
    x_cols,
    exp_info,
    X,
    y,
    can_observe_y,
    subdom_groups,
    library,
    optimizer,
    n_steps,
    step_logger=None,
):
    # Everything subdominance_of_weights() computes that doesn't depend on the
    # weights, computed once rather than on every one of its n_steps calls.
    prepared = prepare_subdominance_of_weights(
        feat_obj_set, demo_df, clf, x_cols, exp_info, X, can_observe_y
    )
    if library == "optuna":
        objective = partial(
            optuna_objective,
            feat_obj_set=feat_obj_set,
            demo_df=demo_df,
            clf=clf,
            x_cols=x_cols,
            exp_info=exp_info,
            X=X,
            y=y,
            can_observe_y=can_observe_y,
            subdom_groups=subdom_groups,
            prepared=prepared,
            step_logger=step_logger,
        )
        n_weights = len(feat_obj_set.objectives)
        x0 = np.clip(np.asarray(wi, dtype=float), -1.0, 1.0)
        x0_params = {f"unnormalized_w{j}": float(x0[j]) for j in range(n_weights)}
        if optimizer == "GP":
            sampler = optuna.samplers.GPSampler(
                seed=exp_info["RANDOM_SEED"],
                # subdominance_of_weights() always returns the same loss for the
                # same weights
                deterministic_objective=True,
            )
        elif optimizer == "TPE":
            sampler = optuna.samplers.TPESampler(seed=exp_info["RANDOM_SEED"])
        elif optimizer == "SMAC":
            sampler = make_smac_sampler(x0_params, n_steps, exp_info["RANDOM_SEED"])
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            # optuna.create_study() otherwise generates a random
            # UUID study name (via uuid.uuid4(), independent of
            # any seed), which is a source of non-determinism.
            study_name=f"opt_debias_{exp_info['RANDOM_SEED']}",
        )
        if optimizer in ("GP", "TPE"):
            # GPSampler and TPESampler have no x0 parameter; evaluating
            # the initial weights as the first trial gives them the same
            # starting point the other optimizers get.
            study.enqueue_trial(x0_params)
        study.optimize(objective, n_trials=n_steps)
        unnormalized_wi = np.array(
            [study.best_params[f"unnormalized_w{j}"] for j in range(n_weights)]
        )
    elif library == "pybobyqa":
        objective = partial(
            pybobyqa_objective,
            feat_obj_set=feat_obj_set,
            demo_df=demo_df,
            clf=clf,
            x_cols=x_cols,
            exp_info=exp_info,
            X=X,
            y=y,
            can_observe_y=can_observe_y,
            subdom_groups=subdom_groups,
            prepared=prepared,
            step_logger=step_logger,
        )
        n_weights = len(feat_obj_set.objectives)
        lower_bounds = -1.0 * np.ones(n_weights)
        upper_bounds = 1.0 * np.ones(n_weights)
        x0 = np.clip(np.asarray(wi, dtype=float), lower_bounds, upper_bounds)
        if optimizer == "Multi-Start BOBYQA":
            # Py-BOBYQA has no seed parameter of its own; it draws
            # its multi-start restarts from the global numpy RNG
            soln = pybobyqa.solve(
                objective,
                x0,
                bounds=(lower_bounds, upper_bounds),
                seek_global_minimum=True,
                maxfun=n_steps,
            )
        unnormalized_wi = np.array(soln.x)
        logging.info(f"unnormalized_wi: {unnormalized_wi}")
    return normalize(unnormalized_wi.reshape(1, -1), norm="l1").flatten()


def make_smac_sampler(x0_params, n_steps, seed):
    """Makes the OptunaHub SMAC sampler for `optimize_weights()`.

    Parameters
    ----------
    x0_params : dict
        The initial value of each parameter of the study, by name. The search
        space is these parameters, each in [-1, 1].
    n_steps : int
        The number of trials of the study.
    seed : int

    Returns
    -------
    sampler : optuna.samplers.BaseSampler
    """
    # Configuration.__hash__ otherwise hashes a string, which differs between
    # processes unless PYTHONHASHSEED is set, and SMAC's results depend on it.
    Configuration.__hash__ = lambda self: int(
        hashlib.sha1(repr(self).encode()).hexdigest()[:16], 16
    )
    module = optunahub.load_module("samplers/smac_sampler", ref=SMAC_SAMPLER_REF)
    search_space = {
        name: optuna.distributions.FloatDistribution(-1.0, 1.0) for name in x0_params
    }
    # Constructing the sampler replaces the root logger's handlers and level
    # with SMAC's own, and there is no argument to turn that off.
    root_logger = logging.getLogger()
    root_handlers, root_level = root_logger.handlers[:], root_logger.level
    with tempfile.TemporaryDirectory() as output_directory:
        sampler = module.SMACSampler(
            search_space,
            n_trials=n_steps,
            # The sampler treats a seed of 0 as no seed at all
            seed=seed or 1,
            output_directory=output_directory,
        )
    root_logger.handlers[:] = root_handlers
    root_logger.setLevel(root_level)
    # The random forest otherwise starts a thread pool on every prediction,
    # which only makes it slower. This doesn't change its results.
    sampler.smac._model._rf_opts["n_jobs"] = 1
    # study.enqueue_trial() doesn't work here: the sampler can only be told the
    # result of a trial it was asked for. So this makes the initial weights the
    # first trial SMAC asks for, the same starting point the other optimizers
    # get.
    sampler.smac._config_selector._initial_design_configs.insert(
        0, Configuration(sampler._cs, values=x0_params)
    )
    return sampler


def optuna_objective(
    trial,
    feat_obj_set,
    demo_df,
    clf,
    x_cols,
    exp_info,
    X,
    y,
    can_observe_y,
    subdom_groups,
    prepared=None,
    step_logger=None,
):
    """Objective function for Optuna optimization of weights."""
    n_weights = len(feat_obj_set.objectives)
    unnormalized_weights = np.array(
        [trial.suggest_float(f"unnormalized_w{j}", -1.0, 1.0) for j in range(n_weights)]
    )
    # Evaluate the weights using the provided evaluation function
    subdominance_loss = subdominance_of_weights(
        unnormalized_weights,
        feat_obj_set,
        demo_df,
        clf,
        x_cols,
        exp_info,
        X,
        y,
        can_observe_y,
        subdom_groups,
        prepared=prepared,
        step_logger=step_logger,
    )
    return subdominance_loss


def pybobyqa_objective(
    unnormalized_weights,
    feat_obj_set,
    demo_df,
    clf,
    x_cols,
    exp_info,
    X,
    y,
    can_observe_y,
    subdom_groups,
    prepared=None,
    step_logger=None,
):
    """Objective function for Py-BOBYQA optimization of weights."""
    unnormalized_weights = np.array(unnormalized_weights)
    # Evaluate the weights using the provided evaluation function
    subdominance_loss = subdominance_of_weights(
        unnormalized_weights,
        feat_obj_set,
        demo_df,
        clf,
        x_cols,
        exp_info,
        X,
        y,
        can_observe_y,
        subdom_groups,
        prepared=prepared,
        step_logger=step_logger,
    )
    return subdominance_loss


@dataclass
class PreparedSubdominanceOfWeights:
    """The parts of `subdominance_of_weights()` that don't depend on the weights.

    Built by `prepare_subdominance_of_weights()`.

    Attributes
    ----------
    state_mdp : ClassificationMDP
        The MDP every weight set's optimal policy is computed in, fit to
        everything but the reward weights (see `fit_state_mdp()`).
    demo_states : numpy.ndarray or None
        The MDP state of each row of `X` the policies' demos are generated on,
        or None if the policies can observe `y` (and so don't look them up).
    """

    state_mdp: ClassificationMDP
    demo_states: np.ndarray | None


def prepare_subdominance_of_weights(
    feat_obj_set, demo_df, clf, x_cols, exp_info, X, can_observe_y
):
    """Computes everything `subdominance_of_weights()` needs that doesn't depend
    on the weights, so that it can be computed once for any number of calls
    with the same arguments. The parameters are those of
    `subdominance_of_weights()`.

    Returns
    -------
    prepared : PreparedSubdominanceOfWeights
    """
    state_mdp = fit_state_mdp(
        clf_df=demo_df,
        x_cols=x_cols,
        obj_set=feat_obj_set,
        min_freq_fill_pct=exp_info["MIN_FREQ_FILL_PCT"],
        restrict_y=exp_info["RESTRICT_Y_ACTION"],
    )
    demo_states = None
    if not can_observe_y:
        # The state of each row depends only on the MDP and on `clf`'s
        # predictions of `y`, not on which policy acts in it.
        demo_states = ClassificationMDPPolicy(
            mdp=state_mdp, pi=None, clf=clf
        ).lookup_states(X)
    return PreparedSubdominanceOfWeights(state_mdp=state_mdp, demo_states=demo_states)


def subdominance_of_weights(
    unnormalized_weights,
    feat_obj_set,
    demo_df,
    clf,
    x_cols,
    exp_info,
    X,
    y,
    can_observe_y,
    subdom_groups,
    prepared=None,
    step_logger=None,
):
    """Computes the subdominance loss for a given set of weights. Uses the weights to
    compute the optimal policy, uses that policy to generate demos, uses the demos to
    compute feature expectations, and then computes the subdominance of the generated
    demos against the expert demos. Returns the subdominance loss metric for that weight set.

    `prepared` is the result of `prepare_subdominance_of_weights()` for these
    same arguments. Passing it in avoids recomputing everything that doesn't
    depend on the weights when calling this repeatedly (e.g. from an
    optimizer); it is computed here when not given.

    `step_logger`, if given, is an `_OptDebiasStepLogger` that this call is
    reported to as one optimizer step.
    """
    if prepared is None:
        prepared = prepare_subdominance_of_weights(
            feat_obj_set, demo_df, clf, x_cols, exp_info, X, can_observe_y
        )
    # Start by L1 normalizing the weights to ensure they sum to 1
    weights = normalize(unnormalized_weights.reshape(1, -1), norm="l1").flatten()
    # Compute the optimal policy for the given weights
    reward_weights = {
        obj.name: weights[j] for j, obj in enumerate(feat_obj_set.objectives)
    }
    clf_pol = compute_optimal_policy_of_state_mdp(
        state_mdp=prepared.state_mdp,
        clf=clf,
        reward_weights=reward_weights,
        skip_error_terms=True,
        method=exp_info["METHOD"],
    )
    # Generate demos from the optimal policy
    yhat = None
    if prepared.demo_states is not None:
        yhat = clf_pol.actions_for_states(prepared.demo_states)
    demo = generate_demo(clf_pol, X, y, can_observe_y=can_observe_y, yhat=yhat)
    # Compute the subdominance of the generated demos against the expert demos
    # lp = LineProfiler()
    # lp.add_function(compute_iteration_subdominance)
    # lp.enable_by_count()
    group_idxs, raw_demos, raw_demos_feat_loss = subdom_groups
    sum_abs_subdom = compute_iteration_subdominance(
        exp_info,
        group_idxs,
        raw_demos,
        raw_demos_feat_loss,
        demo,
        subdominance_type="sum_abs",
        alpha_lamda=opt_debias_lamda(exp_info),
        alpha_adjust=opt_debias_training_adjust(exp_info, "alpha"),
        subdom_weight_adjust=opt_debias_training_adjust(exp_info, "subdom_weight"),
    )
    # lp.disable_by_count()
    # lp.print_stats()
    if step_logger is not None:
        step_logger.record(weights, sum_abs_subdom)
    return sum_abs_subdom


def _evaluate_policy(
    clf_pol,
    wi,
    feat_obj_set,
    perf_obj_set,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    subdom_groups_train,
    subdom_groups_val,
    subdom_groups_test,
    muE_train,
    muE_val,
    muE_test,
    exp_info,
    can_observe_y,
):
    """Generate demos from clf_pol, evaluate errors via _compute_errors_and_metrics,
    and compute subdominance via compute_iteration_subdominance for train/val/test sets.
    """
    logging.debug("\tGenerating learned demostration...")
    demo_train = generate_demo(clf_pol, X_train, y_train, can_observe_y=can_observe_y)
    demo_val = generate_demo(clf_pol, X_val, y_val, can_observe_y=False)
    demo_test = generate_demo(clf_pol, X_test, y_test, can_observe_y=False)

    muL_train = feat_obj_set.compute_demo_feature_exp(demo_train)
    muL_val = feat_obj_set.compute_demo_feature_exp(demo_val)
    muL_test = feat_obj_set.compute_demo_feature_exp(demo_test)
    muL_perf_train = perf_obj_set.compute_demo_feature_exp(demo_train)
    muL_perf_val = perf_obj_set.compute_demo_feature_exp(demo_val)
    muL_perf_test = perf_obj_set.compute_demo_feature_exp(demo_test)

    (
        muL_delta_train,
        muL_delta_l2_train,
        muL_delta_abs_l2_train,
        ti_train,
        muL_delta_val,
        muL_delta_l2_val,
        muL_delta_abs_l2_val,
        ti_val,
        muL_delta_test,
        muL_delta_l2_test,
        muL_delta_abs_l2_test,
        ti_test,
    ) = _compute_errors_and_metrics(
        wi,
        muE_train,
        muE_val,
        muE_test,
        muL_train,
        muL_val,
        muL_test,
        exp_info,
    )

    (
        max_abs_subdominance,
        sum_abs_subdominance,
        max_rel_subdominance,
        sum_rel_subdominance,
    ) = compute_iteration_subdominance(
        exp_info, *subdom_groups_train, clf_demo_cur=demo_train
    )

    (
        max_abs_subdom_v,
        sum_abs_subdom_v,
        max_rel_subdom_v,
        sum_rel_subdom_v,
    ) = compute_iteration_subdominance(
        exp_info, *subdom_groups_val, clf_demo_cur=demo_val
    )

    (
        max_abs_subdom_t,
        sum_abs_subdom_t,
        max_rel_subdom_t,
        sum_rel_subdom_t,
    ) = compute_iteration_subdominance(
        exp_info, *subdom_groups_test, clf_demo_cur=demo_test
    )

    return (
        demo_train,
        demo_val,
        demo_test,
        muL_train,
        muL_val,
        muL_test,
        muL_perf_train,
        muL_perf_val,
        muL_perf_test,
        muL_delta_train,
        muL_delta_l2_train,
        muL_delta_abs_l2_train,
        ti_train,
        muL_delta_val,
        muL_delta_l2_val,
        muL_delta_abs_l2_val,
        ti_val,
        muL_delta_test,
        muL_delta_l2_test,
        muL_delta_abs_l2_test,
        ti_test,
        max_abs_subdominance,
        sum_abs_subdominance,
        max_rel_subdominance,
        sum_rel_subdominance,
        max_abs_subdom_v,
        sum_abs_subdom_v,
        max_rel_subdom_v,
        sum_rel_subdom_v,
        max_abs_subdom_t,
        sum_abs_subdom_t,
        max_rel_subdom_t,
        sum_rel_subdom_t,
    )


def _finalize_trial(
    exp_info,
    run,
    results,
    feat_obj_set,
    perf_obj_set,
    bias_demos,
    unbiased_demos,
    clf_pol,
    trial_runtime,
):
    """Book-keeping for one (bias type, weight adjustment) pair of a trial:
    report the learned policy's results to W&B, where they are stored in the
    run summary.

    Alongside the scalar results, the summary also records the expert's
    demonstrations as the individual subdominance groups every policy is scored
    against (see `_expert_demo_losses_summary()`), so that plots can show the
    spread of the expert's decisions rather than only their mean.
    """
    # Compare the learned policy with the expert demonstrations
    logging.info(
        f"Learned Policy yhat (not real yhat since doesn't factor mu0): {results.demo_train['yhat'].mean():.3f}"
    )
    logging.info(f"weights:\t {np.round(results.weights, 3)}")

    trial_inputsize = None
    if hasattr(clf_pol, "mdp"):
        trial_inputsize = clf_pol.mdp.n_states_

    logging.debug("Experiment Summary")

    summary = _build_trial_summary(
        feat_obj_set,
        perf_obj_set,
        bias_demos,
        unbiased_demos,
        results,
        trial_runtime,
        trial_inputsize,
    )
    summary.update(_expert_demo_losses_summary(exp_info, bias_demos))
    summary.update(_model_feat_loss_summary(exp_info, results))
    run.summary["converged"] = True
    run.summary.update(summary)

    for listener in RUN_RESULT_LISTENERS:
        listener(run, summary)


def _model_feat_loss_summary(exp_info, results):
    """
    The learned model's loss on each subdominance metric, for the run summary.

    The scalar results already report every objective of
    `PERF_MEAS_OBJECTIVE_NAMES` as a feature expectation, but the subdominance
    metrics may be named in the Superhuman Fairness paper's vocabulary (e.g.
    `"prp"`, `"error_rate_diff"`), which has no objective behind it and so no
    feature expectation. These keys report the model on exactly the measures
    the subdominance metric -- and the `opt_debias` weight search that
    minimizes it -- is computed over, whichever vocabulary names them, so that
    the two are always directly comparable and the plots can read either.

    Returns
    -------
    summary : dict<str, float>
        `model_feat_loss_{split}_{metric}` for each split and metric.
    """
    metric_names = subdominance_metric_names(exp_info)
    summary = {}
    split_demos = (
        ("train", results.demo_train),
        ("val", results.demo_val),
        ("test", results.demo_test),
    )
    for split, demo in split_demos:
        losses = feature_losses_from_demo(demo, metric_names)
        for name, loss in zip(metric_names, losses):
            summary[f"model_feat_loss_{split}_{name}"] = float(loss)
    return summary


def _expert_demo_losses_summary(exp_info, bias_demos):
    """
    The expert's per-demonstration losses, ready for a W&B run summary.

    The expert's demonstrations of each split are sampled into
    `N_SUBDOMINANCE_GROUPS` groups by `generate_subdominance_groups()`, and
    each group is scored on the subdominance metrics as a loss (lower is
    better). Each group is one "demonstration" in the sense of the Superhuman
    Fairness paper -- they are exactly the demonstrations that baseline
    imitates -- so these are the points its `plot_features()` figures scatter.

    The losses are stored as nested lists rather than as scalar keys: they are
    one value per group, not one value per run, so they have no meaningful
    average across trials and must not be folded into the scalar results.

    Returns
    -------
    summary : dict
        `expert_demo_feat_loss_metrics`: the metric names, in column order.
        `expert_demo_feat_loss_{split}`: shape (n_groups, n_metrics), for each
        of the train, val and test splits.
    """
    summary = {
        "expert_demo_feat_loss_metrics": subdominance_metric_names(exp_info),
    }
    split_groups = (
        ("train", bias_demos.subdom_groups_train),
        ("val", bias_demos.subdom_groups_val),
        ("test", bias_demos.subdom_groups_test),
    )
    for split, (_group_idxs, _raw_demos, raw_demos_feat_loss) in split_groups:
        summary[f"expert_demo_feat_loss_{split}"] = _json_safe(
            np.asarray(raw_demos_feat_loss, dtype=float)
        )
    return summary


def _build_superhuman_demo_list(sh_config, bias_demos, loss_fn):
    """
    Build the reference decisions the Superhuman Fairness baseline imitates,
    together with the training pool they index into.

    These are this project's own expert demonstrations for the training split,
    grouped exactly the way `generate_subdominance_groups()` grouped them for
    the subdominance metric. The baseline then imitates the same expert, on the
    same rows, that the FRWO technique learns from, which is what makes the
    two directly comparable.

    Parameters
    ----------
    sh_config : dict
        The resolved baseline configuration, from `_superhuman_config()`.
    bias_demos : BiasedDatasetDemos
        The biased data split, expert demonstrations and subdominance groups of
        the bias type being run.
    loss_fn : callable(y_true, y_pred, z) -> numpy.ndarray
        The feature losses a demonstration is scored with, from
        `make_feature_loss_fn()`.

    Returns
    -------
    pool_X : pandas.DataFrame
        The training pool the demonstrations' indices are positions into.
    pool_y : pandas.Series
        That pool's labels.
    demo_list : list<SuperhumanDemo>
    """
    num_demos = sh_config["SH_NUM_DEMOS"]

    # The expert's demonstrations cover every row of the training split
    # exactly once, in their own order, and `subdom_groups_train` holds
    # positional indices into that order.
    expert_demo_df = bias_demos.expert_train.demo
    group_idxs = bias_demos.subdom_groups_train[0]
    if num_demos is not None:
        group_idxs = group_idxs[:num_demos]

    pool_X = expert_demo_df[bias_demos.X_train.columns]
    pool_y = expert_demo_df["y"]
    demo_list = build_expert_demo_list(expert_demo_df, group_idxs, loss_fn)

    return pool_X, pool_y, demo_list


def _log_superhuman_training(run, sh_model):
    """Report the baseline's per-iteration training curve to its W&B run."""
    if run is None:
        return

    history = sh_model.history_
    for i in range(history.n_iterations):
        metrics = {
            "superhuman/iteration": i,
            "superhuman/subdom_sum": history.subdom_sum[i],
            "superhuman/gamma_superhuman_mean": float(
                np.mean(history.gamma_superhuman[i])
            ),
        }
        for j, name in enumerate(sh_model.feature_names):
            metrics[f"superhuman/loss/{name}"] = history.feature_loss[i][j]
            metrics[f"superhuman/gamma_superhuman/{name}"] = history.gamma_superhuman[
                i
            ][j]
            metrics[f"superhuman/alpha/{name}"] = history.alphas[i][j]
        if history.learning_rate:
            metrics["superhuman/learning_rate"] = history.learning_rate[i]
        run.log(metrics)


def _report_run_failure(run, algorithm, error):
    """
    Record that one technique's run failed, so the trial can carry on.

    A technique that raises would otherwise take down the whole execution, and
    with it every result of every dataset still to be run. The fair log-loss
    classifiers in particular can fail to converge on a split whose groups are
    small or degenerate, which is the original implementation's behavior too --
    it just has nothing else running alongside it to protect.

    The failed run is left marked `converged = False`, which is exactly what
    the plotting notebook filters on, so a failure can never quietly pass for a
    result.
    """
    logging.exception(
        f"{algorithm} failed on this bias type."
        " Its W&B run is marked as not converged; the trial continues with the"
        " remaining techniques."
    )
    if run is not None:
        run.summary["converged"] = False
        run.summary["error"] = f"{type(error).__name__}: {error}"

        for listener in RUN_RESULT_LISTENERS:
            listener(run, None)


def _evaluate_and_finalize(
    exp_info,
    run,
    model,
    weights,
    bias_demos,
    unbiased_demos,
    feat_obj_set,
    perf_obj_set,
    policy_start,
    trial_start,
):
    """
    Score a trained model and write its results into its W&B run.

    Every technique that is not the FRWO weight-adjustment loop -- the
    Superhuman Fairness baseline and the four fair-classification baselines it
    is compared against -- finishes its run through this one function, which is
    the same `_evaluate_policy()`/`PolicyResults`/`_finalize_trial()` sequence
    the FRWO technique goes through. That is what makes every model's
    metrics directly comparable: they are produced by the same code, from the
    same splits of the same biased dataset.

    Parameters
    ----------
    model : object
        Anything with a `predict(X)`, which is all `generate_demo()` needs.
    weights : numpy.ndarray
        The reward weights the weighted metrics are computed with. None of
        these techniques learns reward weights, so they are all scored with the
        same equal, positive weights the FRWO technique starts from; that
        keeps `t_train`/`t_val`/`t_test` and `wL_*` comparable across all of
        them.

    Returns
    -------
    results : PolicyResults
    """
    results = PolicyResults(
        [obj.name for obj in feat_obj_set.objectives],
        [obj.name for obj in perf_obj_set.objectives],
    )
    results.weights = weights

    evaluate_policy_result = _evaluate_policy(
        model,
        weights,
        feat_obj_set,
        perf_obj_set,
        bias_demos.X_train,
        bias_demos.X_val,
        bias_demos.X_test,
        bias_demos.y_train,
        bias_demos.y_val,
        bias_demos.y_test,
        bias_demos.subdom_groups_train,
        bias_demos.subdom_groups_val,
        bias_demos.subdom_groups_test,
        bias_demos.expert_train.mu,
        bias_demos.expert_val.mu,
        bias_demos.expert_test.mu,
        exp_info,
        # These baselines always predict `y` from `X`; none of them observes it.
        can_observe_y=False,
    )

    results.record(evaluate_policy_result, policy_start, run)

    trial_runtime = (datetime.datetime.now() - trial_start).total_seconds()

    _finalize_trial(
        exp_info,
        run,
        results,
        feat_obj_set,
        perf_obj_set,
        bias_demos,
        unbiased_demos,
        model,
        trial_runtime,
    )

    return results


def _run_superhuman_trial(
    exp_info,
    bias_demos,
    unbiased_demos,
    feat_obj_set,
    perf_obj_set,
    feature_types,
    weights,
    trial_i,
    group,
    session_id,
    trial_start,
    algorithm=ALGORITHM_SUPERHUMAN,
):
    """
    Train and evaluate one Superhuman Fairness technique for one bias type.

    Reports one W&B run, built by the same `start_wandb_run()` and finished by
    the same `_evaluate_and_finalize()` every other technique here goes
    through, so that all of their runs hold the same metrics, computed the same
    way, on the same train/validation/test splits of the same biased dataset.

    Parameters
    ----------
    weights : numpy.ndarray
        The reward weights the weighted metrics are computed with. See
        `_evaluate_and_finalize()`.
    algorithm : str, default ALGORITHM_SUPERHUMAN
        Which of `SUPERHUMAN_VARIANTS` to run.
    """
    sh_config = _superhuman_config(exp_info)
    variant = SUPERHUMAN_VARIANTS[algorithm]
    feature_names = sh_config["SH_FEATURES"] or subdominance_metric_names(exp_info)
    # Every variant draws from the generator seeded for Superhuman Fairness, so
    # that on the same trial the two logistic regression variants start from
    # exactly the same model, and differ only by the normalization fix and
    # their own hyperparameters; likewise the two neural network variants
    # (given the same network settings), which differ only by the loss fix.
    rng = _baseline_rng(
        exp_info, ALGORITHM_SUPERHUMAN, bias_demos.bias_type, trial_i
    )

    with start_wandb_run(
        exp_info,
        algorithm,
        bias_demos.bias_type,
        (),
        trial_i,
        group,
        session_id,
    ) as run:
        policy_start = datetime.datetime.now()

        try:
            run.config.update(
                {
                    f"{key}_RESOLVED": _json_safe(value)
                    for key, value in sh_config.items()
                }
                | {"SH_FEATURES_RESOLVED": _json_safe(list(feature_names))}
                | variant["resolved"],
                allow_val_change=True,
            )

            loss_fn = make_feature_loss_fn(feature_names)

            logging.info("Building Superhuman Fairness demonstrations...")
            pool_X, pool_y, demo_list = _build_superhuman_demo_list(
                sh_config, bias_demos, loss_fn
            )
            logging.info(
                f"\t {len(demo_list)} demonstrations over {len(pool_X)} training rows,"
                f" features {list(feature_names)}"
            )

            logging.info(f"Training {algorithm}...")
            sh_model = variant["build"](
                {
                    "feature_types": feature_types,
                    "feature_names": feature_names,
                    "loss_fn": loss_fn,
                    "lr_theta": sh_config[variant["lr_theta"]],
                    "iters": sh_config[variant["iters"]],
                    "lamda": sh_config[variant["lamda"]],
                    "rng": rng,
                },
                sh_config,
            )
            sh_model.fit(pool_X, pool_y, demo_list)

            _log_superhuman_training(run, sh_model)

            # Measure and record the error of the learned classifier. This is the
            # exact same evaluation the FRWO policies go through.
            _evaluate_and_finalize(
                exp_info,
                run,
                sh_model,
                weights,
                bias_demos,
                unbiased_demos,
                feat_obj_set,
                perf_obj_set,
                policy_start,
                trial_start,
            )

            run.summary["superhuman_iterations"] = sh_model.history_.n_iterations
            # The iteration whose model was kept and evaluated: the last one,
            # except for the neural network, which keeps its best.
            run.summary["superhuman_selected_iteration"] = sh_model.selected_iteration_
            if isinstance(sh_model, SuperhumanFairnessNN):
                # What its SH_NN_DEVICE / SH_NN_FIXED_DEVICE resolved to.
                run.summary["superhuman_nn_device"] = str(sh_model.device)
            run.summary["superhuman_num_demos"] = len(demo_list)
            for j, name in enumerate(feature_names):
                run.summary[f"superhuman_gamma_{name}"] = (
                    sh_model.history_.gamma_superhuman[sh_model.selected_iteration_][j]
                )
                # The selected model's alpha, `model_params["alpha"]` upstream,
                # whose reciprocal is the margin the Superhuman Fairness
                # plots draw around the model.
                run.summary[f"superhuman_alpha_{name}"] = sh_model.alpha_[j]
        except Exception as error:
            _report_run_failure(run, algorithm, error)


def _fair_logloss_config(exp_info, prefix):
    """
    `FAIR_LOGLOSS_DEFAULTS`, overridden by whatever `exp_info` sets for the
    fair log-loss baseline whose keys start with `prefix`.
    """
    return {
        name: (
            exp_info[prefix + name]
            if exp_info.get(prefix + name) is not None
            else default
        )
        for name, default in FAIR_LOGLOSS_DEFAULTS.items()
    }


def _build_baseline_model(exp_info, algorithm, feature_types, seed):
    """
    Build one of the fair-classification baselines, unfitted.

    Parameters
    ----------
    algorithm : str
        A key of `BASELINE_SPECS`.
    feature_types : dict<str, list>
        Mapping of column names to their type of feature.
    seed : int
        This run's seed, from `_baseline_seed()`.

    Returns
    -------
    model : object
        Ready for `fit(X, y)`, and then `predict(X)`.
    config : dict<str, object>
        The resolved parameters the model was built with, for the run config.
    """
    spec = BASELINE_SPECS[algorithm]
    kwargs = dict(spec["kwargs"])

    if spec["builder"] is PostProcessingBaseline:
        # Only the ThresholdOptimizer's prediction-time randomization needs a
        # generator; everything else about this baseline is deterministic.
        model = PostProcessingBaseline(
            feature_types=feature_types,
            rng=np.random.default_rng(seed),
            **kwargs,
        )
        return model, kwargs

    fll_config = _fair_logloss_config(exp_info, spec["config_prefix"])
    kwargs["C"] = fll_config["C"]
    kwargs["random_initialization"] = fll_config["RANDOM_INIT"]
    model = FairLogLossBaseline(feature_types=feature_types, seed=seed, **kwargs)
    return model, kwargs


def _run_baseline_trial(
    exp_info,
    algorithm,
    bias_demos,
    unbiased_demos,
    feat_obj_set,
    perf_obj_set,
    feature_types,
    weights,
    trial_i,
    group,
    session_id,
    trial_start,
):
    """
    Train and evaluate one fair-classification baseline for one bias type.

    Covers every entry of `BASELINE_SPECS`: the post-processing model of Hardt
    et al. (2016) and the robust fair-log-loss model of Rezaei et al. (2020),
    each with demographic parity or equalized odds as its constraint. All four
    are ordinary classifiers, so this fits one on the bias type's training
    split and hands it to `_evaluate_and_finalize()`, which scores it exactly
    the way every other technique here is scored.

    Reports one W&B run.
    """
    seed = _baseline_seed(exp_info, algorithm, bias_demos.bias_type, trial_i)

    with start_wandb_run(
        exp_info,
        algorithm,
        bias_demos.bias_type,
        (),
        trial_i,
        group,
        session_id,
    ) as run:
        policy_start = datetime.datetime.now()

        try:
            model, model_config = _build_baseline_model(
                exp_info, algorithm, feature_types, seed
            )
            run.config.update(
                {
                    f"BASELINE_{key.upper()}": _json_safe(value)
                    for key, value in model_config.items()
                },
                allow_val_change=True,
            )

            logging.info(f"Training {algorithm} on the training split...")
            model.fit(bias_demos.X_train, bias_demos.y_train)

            _evaluate_and_finalize(
                exp_info,
                run,
                model,
                weights,
                bias_demos,
                unbiased_demos,
                feat_obj_set,
                perf_obj_set,
                policy_start,
                trial_start,
            )
        except Exception as error:
            _report_run_failure(run, algorithm, error)


def run_experiment_trial(
    exp_info,
    X=None,
    y=None,
    feature_types=None,
    trial_i=0,
    group=None,
    session_id=None,
):
    """
    Runs 1 trial to learn an optimal classifier.

    X, y, feature_types don't need to be passed. If they are, then
    `generate_dataset()` is not invoked.

    Parameters
    ----------
    exp_info : dict
        Metadata about the experiment.
    X : pandas.DataFrame, Optional
        The X (including z) columns.
    y : pandas.Series, Optional
        Just the y column.
    feature_types : dict<str, array-like>, Optional
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines.
    trial_i : int, default 0
        Index of this trial within the experiment. Recorded in W&B.
    group : str, Optional
        W&B group shared by every run of this experiment.
    session_id : str, Optional
        Identifies the execution of the experiment script these runs belong to.
        Defaults to a fresh id, so that calling this directly still produces a
        self-contained session.

    Reports
    -------
    One W&B run per (algorithm, bias type, weight adjustment, training
    adjustment) combination:
    first the unbiased dataset, then one per entry in
    exp_info["DATASET_BIAS_TYPE_LIST"], then one per entry in
    exp_info["DEMO_BIAS_TYPE_LIST"] (neither of which is expected to contain
    "()" itself), and within each of those, every algorithm in
    exp_info["ALGORITHMS"]. Each bias type is run on its own: a dataset bias
    type with the expert's demonstrations otherwise unbiased, a demo bias type
    on the unbiased dataset.

    The Fairness Reward Weight Optimization technique reports one run for the
    unadjusted weights, then, if exp_info["RUN_OPT_DEBIAS"], one per
    `opt_debias` entry in exp_info["WEIGHT_ADJUST_LIST"] (which is not expected
    to contain "()" itself either), each derived from the unadjusted weights
    via `_apply_weight_adjustments()`. Each of those runs, the unadjusted one
    included, is followed by one run per training adjustment of it, from
    exp_info["MANUAL_WEIGHT_ADJUST_LIST"] and, for the `opt_debias` runs,
    exp_info["ALPHA_ADJUST_LIST"] and exp_info["SUBDOM_WEIGHT_ADJUST_LIST"]
    (see `_run_frwo_trials()`). The Superhuman
    Fairness baseline and the four fair-classification baselines have no reward
    weights to adjust and so each report a single run per bias type.

    Every run holds this trial's config, the metrics of the model it learned,
    and a summary of that model, all produced by the same evaluation code, so
    that the two techniques' numbers are directly comparable.

    Every dataset bias type is applied to the dataset before the data is
    split, and the datasets are then all split along the same indices, so that
    the only difference between them is their bias. Each bias type fits its
    own `y|x` predictor on its dataset and learns its policies in the MDP built
    from it, so a policy is computed in, demonstrated against, and scored
    against the same biased dataset its expert was. A demo bias type keeps the
    unbiased dataset and biases its expert's demonstrations instead, so its
    policies learn from, and are scored against, biased demonstrations of
    unbiased data.
    """
    trial_start = datetime.datetime.now()

    if session_id is None:
        session_id = new_session_id()

    # The unbiased dataset every bias type is derived from always gets its own
    # set of runs, so `()` is prepended here and neither DATASET_BIAS_TYPE_LIST
    # nor DEMO_BIAS_TYPE_LIST is expected to contain it itself. Prepending it
    # also puts the unbiased runs first, ahead of every listed bias type, and
    # makes the first entry of `demos_by_bias_type` the unbiased one. Each
    # listed bias type is run on its own, never combined with another.
    dataset_bias_type_list = tuple(exp_info["DATASET_BIAS_TYPE_LIST"])
    demo_bias_type_list = tuple(exp_info.get("DEMO_BIAS_TYPE_LIST", ()))
    validate_bias_types(dataset_bias_type_list, demo_bias_type_list)
    validate_training_adjusts(exp_info)
    bias_type_list = ((),) + dataset_bias_type_list + demo_bias_type_list
    weight_adjust_list = ((),)
    if exp_info["RUN_OPT_DEBIAS"]:
        weight_adjust_list += tuple(exp_info["WEIGHT_ADJUST_LIST"])

    # Which techniques this trial trains and evaluates. Always in ALGORITHMS
    # order, so the FRWO technique runs first and its random draws are
    # unaffected by whether the baseline is enabled.
    algorithms = selected_algorithms(exp_info)
    logging.info(f"ALGORITHMS: {algorithms}")

    feat_obj_set, perf_obj_set = _build_objective_sets(exp_info)
    can_observe_y = "FO" in exp_info["IRL_METHOD"]
    X, y, feature_types = _load_or_generate_dataset(exp_info, X, y, feature_types)

    expert_algo_lookup = generate_expert_algo_lookup(feature_types, exp_info)

    demos_by_bias_type = _split_dataset_and_generate_expert_demos(
        exp_info,
        expert_algo_lookup,
        feat_obj_set,
        perf_obj_set,
        X,
        y,
        feature_types,
        bias_type_list,
        trial_i,
    )

    # `bias_type_list` starts with `()`, so the first entry is the unbiased
    # dataset and its demonstrations. It is what everything that needs
    # unbiased data uses -- currently just the `*_unbiased` trial summary
    # metrics, which report every bias type's expert alongside the unbiased
    # one.
    unbiased_demos = demos_by_bias_type[0]

    x_cols = input_columns(feature_types)
    x_cols.remove("z")

    # The expert is an optimal classifier policy whose reward weights are
    # equal and positive across every objective, so those same weights are the
    # unadjusted ones every weight adjustment is derived from.
    n_objs = len(feat_obj_set.objectives)
    unadjusted_weight = np.full(n_objs, 1.0 / n_objs)

    # The unbiased dataset, then every bias type of this trial, each against
    # the same data split, differing only by its bias. Within a bias type,
    # every selected algorithm is trained and evaluated on that same data: the
    # FRWO technique gets one run for the unadjusted weights and then, if
    # RUN_OPT_DEBIAS is on, one per `opt_debias` weight adjustment (the
    # unadjusted weights always get their own W&B run regardless), and the
    # Superhuman Fairness baseline, which
    # has no reward weights to adjust, gets a single run.
    for bias_demos in demos_by_bias_type:
        with _bias_type_random_state(
            exp_info, bias_demos.bias_type, trial_i, "train"
        ):
            expert_train = bias_demos.expert_train
            expert_val = bias_demos.expert_val
            expert_test = bias_demos.expert_test

            logging.info(f"BIAS TYPE: {bias_demos.bias_type}")

            logging.info(f"muE_train:\n{expert_train.mu}")
            logging.info(f"muE_val:\n{expert_val.mu}")
            logging.info(f"muE_test:\n{expert_test.mu}")
            logging.info(f"muE_perf_train:\n{expert_train.mu_perf}")
            logging.info(f"muE_perf_val:\n{expert_val.mu_perf}")
            logging.info(f"muE_perf_test:\n{expert_test.mu_perf}")

            if ALGORITHM_FRWO in algorithms:
                logging.info(f"ALGORITHM: {ALGORITHM_FRWO}")
                _run_frwo_trials(
                    exp_info,
                    bias_demos,
                    unbiased_demos,
                    feat_obj_set,
                    perf_obj_set,
                    feature_types,
                    x_cols,
                    weight_adjust_list,
                    unadjusted_weight,
                    can_observe_y,
                    trial_i,
                    group,
                    session_id,
                    trial_start,
                )

            for algorithm in algorithms:
                if algorithm not in SUPERHUMAN_VARIANTS:
                    continue
                logging.info(f"ALGORITHM: {algorithm}")
                _run_superhuman_trial(
                    exp_info,
                    bias_demos,
                    unbiased_demos,
                    feat_obj_set,
                    perf_obj_set,
                    feature_types,
                    unadjusted_weight.copy(),
                    trial_i,
                    group,
                    session_id,
                    trial_start,
                    algorithm=algorithm,
                )

            for algorithm in algorithms:
                if algorithm not in BASELINE_SPECS:
                    continue
                logging.info(f"ALGORITHM: {algorithm}")
                _run_baseline_trial(
                    exp_info,
                    algorithm,
                    bias_demos,
                    unbiased_demos,
                    feat_obj_set,
                    perf_obj_set,
                    feature_types,
                    unadjusted_weight.copy(),
                    trial_i,
                    group,
                    session_id,
                    trial_start,
                )


def _run_frwo_trials(
    exp_info,
    bias_demos,
    unbiased_demos,
    feat_obj_set,
    perf_obj_set,
    feature_types,
    x_cols,
    weight_adjust_list,
    unadjusted_weight,
    can_observe_y,
    trial_i,
    group,
    session_id,
    trial_start,
):
    """
    Train and evaluate the Fairness Reward Weight Optimization technique for
    one bias type.

    One W&B run per entry of `weight_adjust_list`, each followed by one per
    training adjustment of it (see `TRAINING_ADJUST_KINDS`), each of those
    adjusting that one run and nothing else:
      * every entry of MANUAL_WEIGHT_ADJUST_LIST, applied to that run's
        weights -- the unadjusted ones, or those an `opt_debias` run learned
        -- without optimizing again;
      * after an `opt_debias` run only, every entry of ALPHA_ADJUST_LIST and
        then SUBDOM_WEIGHT_ADJUST_LIST, each an `opt_debias` run of its own,
        minimizing the subdominance the entry skews.
    Lifted out of `run_experiment_trial()` when the Superhuman Fairness
    baseline was added, so that each technique's trial reads as one block.
    """
    expert_train = bias_demos.expert_train
    expert_val = bias_demos.expert_val
    expert_test = bias_demos.expert_test

    feat_obj_set_cols = [obj.name for obj in feat_obj_set.objectives]
    perf_obj_set_cols = [obj.name for obj in perf_obj_set.objectives]

    # The `y|x` predictor -- and the MDP built from it -- of every policy
    # this bias type learns. It is fit on this bias type's own biased
    # labels, so the MDP its policies are computed in is the world its
    # expert demonstrated in.
    logging.debug("Fitting `y|x` predictor for clf policy...")
    clf, demo_df = _fit_clf_and_demo_df(
        exp_info,
        feature_types, bias_demos.X_train, bias_demos.y_train
    )

    def run_policy(weight_adjust, training_adjust=NO_TRAINING_ADJUST, base=None):
        """
        Learn, evaluate and report the policy of one W&B run. `base` is the
        `(weights, run id)` of the run a manual weight adjustment adjusts.

        Returns the run's weights and its W&B run id.
        """
        kind, entry = training_adjust
        with start_wandb_run(
            exp_info,
            ALGORITHM_FRWO,
            bias_demos.bias_type,
            weight_adjust,
            trial_i,
            group,
            session_id,
            training_adjust=training_adjust,
        ) as run:
            policy_start = datetime.datetime.now()

            results = PolicyResults(feat_obj_set_cols, perf_obj_set_cols)

            if kind == "manual_weight":
                base_wi, base_run_id = base
                wi = apply_manual_weight_adjust(base_wi, entry, feat_obj_set_cols)
                run.summary.update(
                    {
                        f"weight_before_manual_adjust/{col}": base_wi[j]
                        for j, col in enumerate(feat_obj_set_cols)
                    }
                )
                run.summary["manual_weight_adjust_base_run_id"] = base_run_id
            else:
                # An alpha or subdominance weight adjustment skews the
                # objective of this run's `opt_debias` weight search, and
                # nothing else (see `opt_debias_training_adjust()`).
                opt_debias_exp_info = exp_info
                if entry:
                    opt_debias_exp_info = {
                        **exp_info,
                        TRAINING_ADJUST_KINDS[kind][1]: entry,
                    }
                wi = _apply_weight_adjustments(
                    unadjusted_weight.copy(),
                    weight_adjust,
                    feat_obj_set,
                    demo_df,
                    clf,
                    x_cols,
                    opt_debias_exp_info,
                    bias_demos.X_train,  # could switch to bias_demos.X_val
                    bias_demos.y_train,  # could switch to bias_demos.y_val
                    can_observe_y,
                    # could switch to bias_demos.subdom_groups_val
                    bias_demos.subdom_groups_train,
                    run=run,
                )

            # Learn a policy that maximizes the reward function.
            results.weights = wi

            reward_weights = {
                obj.name: wi[j] for j, obj in enumerate(feat_obj_set.objectives)
            }
            clf_pol = compute_optimal_policy(
                clf_df=demo_df,
                clf=clf,
                x_cols=x_cols,
                obj_set=feat_obj_set,
                reward_weights=reward_weights,
                skip_error_terms=True,
                method=exp_info["METHOD"],
                min_freq_fill_pct=exp_info["MIN_FREQ_FILL_PCT"],
                restrict_y=exp_info["RESTRICT_Y_ACTION"],
            )

            ##
            # Measure and record the error of the learned policy.
            ##

            evaluate_policy_result = _evaluate_policy(
                clf_pol,
                wi,
                feat_obj_set,
                perf_obj_set,
                bias_demos.X_train,
                bias_demos.X_val,
                bias_demos.X_test,
                bias_demos.y_train,
                bias_demos.y_val,
                bias_demos.y_test,
                bias_demos.subdom_groups_train,
                bias_demos.subdom_groups_val,
                bias_demos.subdom_groups_test,
                expert_train.mu,
                expert_val.mu,
                expert_test.mu,
                exp_info,
                can_observe_y,
            )

            results.record(evaluate_policy_result, policy_start, run)

            trial_runtime = (datetime.datetime.now() - trial_start).total_seconds()

            _finalize_trial(
                exp_info,
                run,
                results,
                feat_obj_set,
                perf_obj_set,
                bias_demos,
                unbiased_demos,
                clf_pol,
                trial_runtime,
            )

            return wi, run.id

    for weight_adjust in weight_adjust_list:
        wi, run_id = run_policy(weight_adjust)

        training_adjust_runs = [
            (("manual_weight", entry), (wi, run_id))
            for entry in training_adjusts(exp_info, "manual_weight")
        ]
        # Every entry of `weight_adjust_list` but `()` is an `opt_debias` one.
        if weight_adjust:
            training_adjust_runs += [
                ((kind, entry), None)
                for kind in ("alpha", "subdom_weight")
                for entry in training_adjusts(exp_info, kind)
            ]
        for training_adjust, base in training_adjust_runs:
            logging.info(f"TRAINING ADJUSTMENT: {training_adjust_name(training_adjust)}")
            with _training_adjust_random_state(
                exp_info, bias_demos.bias_type, weight_adjust, training_adjust, trial_i
            ):
                run_policy(weight_adjust, training_adjust, base)


def subdominance_metric_names(exp_info):
    """
    The performance/fairness measures the subdominance metric is computed over,
    in order: the perf ones first, then the fair ones.

    Each may be named in either of the two vocabularies
    `feature_losses_from_demo()` accepts -- this project's objectives (e.g.
    `"Acc"`, `"DemPar"`) or the original Superhuman Fairness paper's metrics
    (e.g. `"inacc"`, `"dp"`, `"prp"`) -- and the two may be mixed.
    """
    names = list(exp_info["SUBDOMINANCE_PERF_METRICS_LIST"]) + list(
        exp_info["SUBDOMINANCE_FAIR_METRICS_LIST"]
    )
    validate_feature_names(names, context="subdominance metric")
    return names


def compute_relevant_feat_loss(exp_info, demo):
    # Generate the demo's loss on each subdominance metric, where lower is
    # better. `feature_losses_from_demo()` is the shared definition of that
    # loss, so that the Superhuman Fairness baseline optimizes and is scored on
    # exactly the quantities used here, whichever vocabulary names them.
    demo_feat_exp = feature_losses_from_demo(demo, subdominance_metric_names(exp_info))

    # Now compute the same feature expectations using Fairlearn's metric functions
    # obj_names = (
    #     exp_info["SUBDOMINANCE_PERF_METRICS_LIST"]
    #     + exp_info["SUBDOMINANCE_FAIR_METRICS_LIST"]
    # )
    # demo_feat_exp_fairlearn = np.array(
    #     [FAIRLEARN_OBJ_LOOKUP_BY_NAME[obj_name](demo) for obj_name in obj_names]
    # )
    # These should be the same, but we check to make sure they are. If not, raise an error.
    # assert np.allclose(demo_feat_exp, demo_feat_exp_fairlearn), (
    #     f"Fairlearn-based feature loss {demo_feat_exp_fairlearn} does not match "
    #     f"the original feature loss {demo_feat_exp} for obj_names {obj_names}"
    # )
    return demo_feat_exp


def run_bias_experiment(
    exp_info,
    source_X=None,
    source_y=None,
    source_feature_types=None,
    session_id=None,
    first_trial=0,
    after_trial=None,
):
    """
    Runs experiment for source domain based on the parameters in `exp_info`.

    Parameters
    ----------
    exp_info : dict
        Experiment parameters.
    source_X : pandas.DataFrame, Optional
        The X (including z) columns for the source domain.
    source_y : pandas.Series, Optional
        Just the y column for the source domain.
    source_feature_types : dict<str, array-like>, Optional
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines for the source domain.
    session_id : str, Optional
        Identifies the execution of the experiment script this experiment
        belongs to. One execution calls this function once per experiment, so
        passing the same id to each call is what marks their runs as one set of
        results. Defaults to a fresh id, so calling this on its own still
        produces a self-contained session.
    first_trial : int, default 0
        Index of the first trial to run. A resumed experiment (see
        `frwo.pause`) skips the trials it ran before it was paused.
    after_trial : callable, Optional
        Called with the trial's index after each trial finishes. This is where
        the experiment script pauses.

    Reports
    -------
    Every trial is reported to W&B as one run per (bias type, weight
    adjustment) pair, all sharing a `group` unique to this experiment and a
    `SESSION_ID` shared with every other experiment of the same execution. Each
    run's config holds `exp_info`, its history holds the metrics of the policy
    it learned, and its summary holds that policy's results.
    """
    logging.info(f"exp_info: {exp_info}")

    if session_id is None:
        session_id = new_session_id()

    # Shared by every run of this experiment so that its trials, bias types and
    # weight adjustments stay grouped together in the W&B UI. Ending the group with
    # the session id means every group of one execution carries the same
    # visible suffix.
    group = "__".join(
        [
            str(exp_info["EXPERIMENT_NAME"]),
            str(exp_info["EXPERT_ALGO"]),
            session_id,
        ]
    )
    logging.info(f"W&B run group: {group}")

    for trial_i in range(first_trial, exp_info["N_TRIALS"]):
        # Run trials to learn weights on source domain
        run_experiment_trial(
            exp_info,
            X=source_X,
            y=source_y,
            feature_types=source_feature_types,
            trial_i=trial_i,
            group=group,
            session_id=session_id,
        )
        if after_trial is not None:
            after_trial(trial_i)
