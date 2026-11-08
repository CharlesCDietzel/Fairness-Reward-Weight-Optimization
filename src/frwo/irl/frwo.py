import copy
import json
import logging
import numpy as np
import os
import tempfile
import pandas as pd
from frwo.rl.clf_mdp import *
from frwo.rl.clf_mdp_policy import *
from frwo.rl.objectives import *
from frwo.utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from frwo.utils import single_threaded_blas


@single_threaded_blas
def compute_optimal_policy(
    clf_df,
    clf,
    x_cols,
    obj_set,
    reward_weights,
    skip_error_terms=True,
    method="highs",
    gamma=1e-9,
    min_freq_fill_pct=0,
    restrict_y=True,
):
    """
    Learns the optimal policies from the provided reward weights.

    The `clf_df` is used to "fit" the ClfMDP parameters (e.g. b_eq, A_eq, etc)
    as well as the classifier that predicts `y` from `X`.

    The other classification parameters (`feature_types`, and `clf` are
    used only on the classifier that predicts the `y` value from the `X`
    values.

    Parameters
    ----------
    clf_df : pandas.DataFrame
        Classification dataset. Required columns:
            'z' : int. Binary protected attribute.
            'y' : int. Binary target variable.
    clf : sklearn.pipeline.Pipeline
        Sklearn classifier instance.
    x_cols : list<str>
        The columns that are used in the state (along with `z` and `y`).
    obj_set : ObjectiveSet
        The objective set.
    reward_weights : dict<str, float>
        Keys are objective identifiers. Values are their respective reward
        weights.
    skip_error_terms : bool, default True
        If true, doesn't try and find all solutions and instead just invokes
        the solver on the input terms.
    method : str, default 'highs'
        The HiGHS solver to use (see `frwo.rl.clf_mdp._ReducedLP`). Options
        are 'highs' (default), 'highs-ds', 'highs-ipm'.
    gamma : float, range, [0, 1)
        The MDP discount factor. Should be close to zero since this is a
        classification mdp, but cannot be zero exactly otherwise the linear
        program won't converge.
    min_freq_fill_pct, float, range[0, 1), default 0
        Minimum frequency for each input variable to not get replaced by a
        default value.
    restrict_y : bool, deafult True
        If True, policy must have same action for any x,z combo, regardless
        of y.

    Returns
    -------
    opt_pol : frwo.rl.clf_mdp.ClassificationMDPPolicy
        The optimal policy. If there are multiple, it randomly selects one.
    """
    clf_mdp = ClassificationMDP(
        gamma=gamma,
        obj_set=obj_set,
        x_cols=x_cols,
    )

    # Construct the mdp, including optimization problems
    clf_mdp.fit(
        reward_weights=reward_weights,
        clf_df=clf_df,
        min_freq_fill_pct=min_freq_fill_pct,
        restrict_y=restrict_y,
    )

    return _sample_optimal_policy(clf_mdp, clf, skip_error_terms, method)


def fit_state_mdp(
    clf_df,
    x_cols,
    obj_set,
    gamma=1e-9,
    min_freq_fill_pct=0,
    restrict_y=True,
):
    """
    Fits the part of the MDP `compute_optimal_policy()` builds that doesn't
    depend on the reward weights, so that the optimal policies of many reward
    weights can be computed from it with `compute_optimal_policy_of_state_mdp()`
    without refitting it for each of them.

    The parameters are those of `compute_optimal_policy()`. `obj_set` is
    copied rather than fit in place, so that the returned MDP is unaffected by
    whatever else `obj_set` is later fit to.

    Returns
    -------
    state_mdp : frwo.rl.clf_mdp.ClassificationMDP
        MDP fit with `fit_states()`, whose objectives are fit to it.
    """
    state_mdp = ClassificationMDP(
        gamma=gamma,
        obj_set=copy.deepcopy(obj_set),
        x_cols=x_cols,
    )
    state_mdp.fit_states(
        clf_df=clf_df,
        min_freq_fill_pct=min_freq_fill_pct,
        restrict_y=restrict_y,
    )
    state_mdp.obj_set.fit_objectives(state_mdp.ldf_)
    return state_mdp


def compute_optimal_policy_of_state_mdp(
    state_mdp,
    clf,
    reward_weights,
    skip_error_terms=True,
    method="highs",
):
    """
    Equivalent to `compute_optimal_policy()` with the parameters `state_mdp`
    was fit with by `fit_state_mdp()`, but without refitting any of the MDP.

    Parameters
    ----------
    state_mdp : frwo.rl.clf_mdp.ClassificationMDP
        The MDP returned by `fit_state_mdp()`. It is not modified.
    clf, reward_weights, skip_error_terms, method
        See `compute_optimal_policy()`.

    Returns
    -------
    opt_pol : frwo.rl.clf_mdp.ClassificationMDPPolicy
        The optimal policy. If there are multiple, it randomly selects one.
    """
    clf_mdp = state_mdp.with_reward_weights(reward_weights)
    return _sample_optimal_policy(clf_mdp, clf, skip_error_terms, method)


def _sample_optimal_policy(clf_mdp, clf, skip_error_terms, method):
    """
    Solves the fitted `clf_mdp` and returns one of its optimal policies, picked
    at random if there are multiple.
    """
    # Compute the optimal policy(s). This does NOT fit the classifier that
    # predicts Y from Z, X. That occurs on the generate_demo() call. This
    # `compute_optimal_policies` computes the optimal policy, assuming the
    # state is known.
    optimal_policies = clf_mdp.compute_optimal_policies(
        skip_error_terms=skip_error_terms,
        method=method,
    )

    # Pick from one of the policies (if there are multiple).
    logging.info(f"\n\t\tFound {len(optimal_policies)} optimal policies.")
    sampled_policy = optimal_policies[np.random.choice(len(optimal_policies))]

    clf_pol = ClassificationMDPPolicy(
        mdp=clf_mdp,
        pi=sampled_policy,
        clf=clf,
    )

    return clf_pol


class OptClfMDPPolicyExpert:
    """
    Configuration holder for the "OptClfMDPPol" expert algo (see
    `experiment_utils.generate_expert_algo_lookup()`).

    Unlike the other entries in `expert_algo_lookup`, a `ClassificationMDPPolicy`
    can't be built until the feature-expectation objective set (`obj_set`) and
    a concrete training fold (`X_train`, `y_train`) are known, neither of which
    is available yet when `expert_algo_lookup` is constructed. This class just
    holds `feature_types` and defers the actual `compute_optimal_policy()` call
    to `build_policy()`, which `generate_non_overfit_demos()` invokes once per
    fold instead of the usual `clf.fit()`/`clf.predict()`.

    Parameters
    ----------
    feature_types : dict<str, list>
        Mapping of column names to their type of feature.
    """

    def __init__(self, feature_types):
        self.feature_types = feature_types

    def build_policy(self, X_train, y_train, obj_set, exp_info):
        """
        Fits the `y|x` predictor on `(X_train, y_train)` and computes the
        optimal classifier policy for that data, using equal, positive reward
        weights (summing to 1) across every objective in `obj_set`.

        Returns
        -------
        clf_pol : ClassificationMDPPolicy
        """
        x_cols = input_columns(self.feature_types)
        x_cols.remove("z")

        inner_clf = sklearn_clf_pipeline(
            feature_types=self.feature_types,
            clf_inst=RandomForestClassifier(),
        )
        inner_clf.fit(X_train, y_train)

        clf_df = pd.DataFrame(X_train).copy()
        clf_df["y"] = y_train

        n_objs = len(obj_set.objectives)
        reward_weights = {obj.name: 1.0 / n_objs for obj in obj_set.objectives}

        return compute_optimal_policy(
            clf_df=clf_df,
            clf=inner_clf,
            x_cols=x_cols,
            obj_set=obj_set,
            reward_weights=reward_weights,
            skip_error_terms=True,
            method=exp_info["METHOD"],
            min_freq_fill_pct=exp_info["MIN_FREQ_FILL_PCT"],
            restrict_y=exp_info["RESTRICT_Y_ACTION"],
        )


def generate_demo(clf, X_test, y_test, can_observe_y=False, yhat=None):
    """
    Create demonstration dataframe (columns are '**X', 'yhat', 'y') from a
    fitted classifer `clf`.

    Parameters
    ----------
    clf : fitted sklearn classifier
    X_test : pandas.DataFrame
    y_test : pandas.Series
    can_observe_y : bool, default False
        Whether the policy can "see" y or if it needs to predict it from X.
    yhat : array-like, optional
        `clf`'s predictions on `X_test`, if they are already known. Used
        instead of calling `clf.predict(X_test)` when `can_observe_y` is False.

    Returns
    -------
    demo : pandas.DataFrame
        Demonstrations. Each demonstration represents an iteration of a
        trained classifier and its predictions on a hold-out set. Columns are
            **`X` columns : all input columns (i.e. `X`)
            yhat : predictions
            y : ground truth targets
    """
    demo = pd.DataFrame(X_test).copy()

    if can_observe_y:
        # yhat = clf.predict(X_test, y_test)
        if "y" in X_test.columns:
            demo["yhat"] = demo["y"].copy()
        else:
            demo["yhat"] = y_test
    elif yhat is None:
        demo["yhat"] = clf.predict(X_test).astype(np.int64)
    else:
        demo["yhat"] = np.asarray(yhat).astype(np.int64)

    demo["y"] = y_test.copy()

    return demo


##
# Bias types.
#
# Every bias type is a tuple whose first element names it and whose remaining
# elements are its parameters, e.g. `("dataset_balanced_redlining", 0.2)`.
# There are two varieties:
#
#   Dataset bias types (DATASET_BIAS_TYPE_LIST) bias the dataset itself --
#       its labels, or for `dataset_noise_bias` optionally its protected
#       attribute -- before it is split, so the expert is fit on, and everything
#       is scored against, the biased dataset. See `add_dataset_bias()`.
#   Demo bias types (DEMO_BIAS_TYPE_LIST) leave the dataset unchanged and bias
#       the expert's demonstrations instead: the `yhat` decisions the
#       EXPERT_ALGO makes. See `add_demo_bias()`.
#
# The redlining bias types of the two varieties share their implementation,
# `_apply_redlining()`, applied to the dataset's labels or to the expert's
# decisions respectively; so do the two noise bias types, `_flip_values()`,
# and the two corruption bias types, `_corrupt_and_relabel()`.
##

# The three redlining processes, each available as a dataset and a demo bias
# type: the bias type's name minus its "dataset_" / "demo_" prefix.
REDLINING_KINDS = (
    "unbalanced_redlining",
    "balanced_redlining",
    "perfectly_balanced_redlining",
)

DATASET_BIAS_TYPES = (
    *(f"dataset_{kind}" for kind in REDLINING_KINDS),
    "dataset_noise_bias",
    "dataset_corruption_bias",
)

DEMO_BIAS_TYPES = (
    *(f"demo_{kind}" for kind in REDLINING_KINDS),
    "demo_noise_bias",
    "demo_corruption_bias",
)

# The dataset bias types' names before the "dataset_" prefix was introduced,
# for the error a configuration that still uses them gets.
RENAMED_BIAS_TYPES = {
    "unbalanced_redlining": "dataset_unbalanced_redlining",
    "balanced_redlining": "dataset_balanced_redlining",
    "perfectly_balanced_redlining": "dataset_perfectly_balanced_redlining",
    "corruption_bias": "dataset_corruption_bias",
}


def bias_type_kind(bias_type):
    """
    Which variety a bias type is: "dataset", "demo", or `None` for `()`, the
    unbiased "bias type".
    """
    if not bias_type:
        return None
    if bias_type[0] in DATASET_BIAS_TYPES:
        return "dataset"
    if bias_type[0] in DEMO_BIAS_TYPES:
        return "demo"
    raise ValueError(f"Unrecognized bias type: {bias_type!r}.")


def validate_bias_types(dataset_bias_type_list, demo_bias_type_list):
    """
    Check that every entry of DATASET_BIAS_TYPE_LIST and DEMO_BIAS_TYPE_LIST
    names a bias type of its own variety, with the right parameters, so that a
    misconfigured run fails before anything expensive is done.
    """
    expected_params = {
        "dataset_noise_bias": "[dataset_noise_bias, noise_ratio, label_noise, protected_noise]",
        "dataset_corruption_bias": "[dataset_corruption_bias, clf_type, percent, noise_type, magnitude]",
        "demo_noise_bias": "[demo_noise_bias, noise_ratio]",
        "demo_corruption_bias": "[demo_corruption_bias, clf_type, percent, noise_type, magnitude]",
    }
    n_params = {
        "dataset_noise_bias": 3,
        "dataset_corruption_bias": 4,
        "demo_noise_bias": 1,
        "demo_corruption_bias": 4,
    }
    for list_name, bias_types, valid in (
        ("DATASET_BIAS_TYPE_LIST", dataset_bias_type_list, DATASET_BIAS_TYPES),
        ("DEMO_BIAS_TYPE_LIST", demo_bias_type_list, DEMO_BIAS_TYPES),
    ):
        for bias_type in bias_types:
            name = bias_type[0] if bias_type else None
            if name in RENAMED_BIAS_TYPES:
                raise ValueError(
                    f"{list_name} entry {bias_type!r}: the bias type {name!r} has"
                    f" been renamed to {RENAMED_BIAS_TYPES[name]!r}."
                )
            if name not in valid:
                raise ValueError(
                    f"{list_name} entry {bias_type!r} is not one of its bias"
                    f" types: {list(valid)}."
                )
            params = len(bias_type) - 1
            if name in n_params and params != n_params[name]:
                raise ValueError(
                    f"{list_name} entry {bias_type!r} should look like"
                    f" {expected_params[name]}."
                )
            if name.endswith("_redlining") and params > 1:
                raise ValueError(
                    f"{list_name} entry {bias_type!r} should look like"
                    f" [{name}, percent] or [{name}]."
                )


def _redlining_outcomes(dataset):
    """
    Which group a redlining bias discriminates against, and which outcome is
    the "bad" one, for a dataset.

    Z = 0 is discriminated against, Y = 0 is "bad" outcome (except for COMPAS
    where Y=1 is "bad" outcome)

    Returns
    -------
    rz, ry, nrz, nry : int
        The redlined Z value, the redline Y outcome, the non-redlined Z value
        and the non-redline Y outcome.
    """
    if dataset == "Adult":
        return 0, 0, 1, 1
    elif dataset == "Boston":
        return 0, 0, 1, 1
    elif "ACSIncome__" in dataset:  # All ACSIncome datasets have same strtucture
        return 0, 0, 1, 1
    elif dataset == "COMPAS":  # COMPAS Y redlining is reversed
        return 0, 1, 1, 0
    raise ValueError(f"Redlining bias is not defined for dataset {dataset!r}.")


def _apply_redlining(z, values, kind, percent, dataset):
    """
    Apply one redlining process to a vector of outcomes.

    The implementation every redlining bias type shares: the dataset bias
    types apply it to the dataset's labels `y`, and the demo bias types to the
    expert's decisions `yhat`.

    Parameters
    ----------
    z : array-like
        The protected attribute of each row.
    values : array-like
        The outcome of each row, `y` or `yhat`, aligned with `z`.
    kind : str
        One of `REDLINING_KINDS`.
    percent : float
        The fraction of the redlined rows whose outcome is changed.
    dataset : str
        Name of the dataset; see `_redlining_outcomes()`.

    Returns
    -------
    values_biased : numpy.ndarray
        A copy of `values` with the bias applied, in the same row order.
    """
    rz, ry, nrz, nry = _redlining_outcomes(dataset)

    # Work on a positionally indexed frame of just the columns the bias needs,
    # so that the caller's own index (which needn't be unique or ordered)
    # can't interfere with the sampling below.
    df = pd.DataFrame({"z": np.asarray(z), "y": np.asarray(values)})

    match kind:
        case "unbalanced_redlining":
            # wherever Z==rz, set y = ry with 20% probability. Otherwise, keep y the same
            # Count how many rows have z == rz
            rz_count = (df["z"] == rz).sum()
            # Multiply that by 20% to get the number of rows to redline
            n = int(rz_count * percent)
            # Step 3: Get n indices where z == rz
            rz_indices = df[df["z"] == rz].sample(n=n).index
            # Step 4: Set y to ry for sampled rows
            df.loc[rz_indices, "y"] = ry
        case "balanced_redlining":
            # wherever Z==rz, set y = ry with 20% probability. Otherwise, keep y the same. Also, whereever Z==nrz, randomly set an equal number of y = nry
            # Count how many rows have z == rz
            rz_count = (df["z"] == rz).sum()
            # Multiply that by 20% to get the number of rows to redline
            n = int(rz_count * percent)
            # Edge case check: If there are fewer z == nrz than n, set n = nrz_count instead
            nrz_count = (df["z"] == nrz).sum()
            n = min(nrz_count, n)
            # Get the n indices where z == rz
            rz_indices = df[df["z"] == rz].sample(n=n).index
            # Set y to ry for sampled rows
            df.loc[rz_indices, "y"] = ry
            # Get the n indices where z == nrz
            nrz_indices = df[df["z"] == nrz].sample(n=n).index
            # Set y to nry for sampled rows
            df.loc[nrz_indices, "y"] = nry
        case "perfectly_balanced_redlining":
            # wherever Z==rz and y==nry, set y = ry with 20% probability. Also, whereever Z==nrz and y==ry, randomly set an equal number of y = nry
            # Count how many rows have z == rz AND y == nry
            rz_count = ((df["z"] == rz) & (df["y"] == nry)).sum()
            # Multiply that by 20% to get the number of rows to flip
            n = int(rz_count * percent)
            # Edge case check: If there are fewer z == nrz and y == ry than n, set n = nrz_count instead
            nrz_count = ((df["z"] == nrz) & (df["y"] == ry)).sum()
            n = min(nrz_count, n)
            # Get the n indices where z == rz AND y == nry
            rz_indices = df[(df["z"] == rz) & (df["y"] == nry)].sample(n=n).index
            # Get the n indices where z == nrz AND y == ry
            nrz_indices = df[(df["z"] == nrz) & (df["y"] == ry)].sample(n=n).index
            # Set y to ry for sampled rows
            df.loc[rz_indices, "y"] = ry
            # Set y to nry for sampled rows
            df.loc[nrz_indices, "y"] = nry
        case _:
            raise ValueError(f"Unrecognized redlining kind: {kind!r}.")

    return df["y"].to_numpy()


def _redlining_params(bias_type, prefix):
    """The redlining kind and percent of a `<prefix>*_redlining` bias type."""
    kind = bias_type[0][len(prefix) :]
    percent = bias_type[1] if len(bias_type) > 1 else 0.2
    return kind, percent


def _random_fraction_positions(n, ratio):
    """
    `int(ratio * n)` distinct random positions among `n`, drawn as the
    Superhuman Fairness reference implementation's `add_noise()` and
    `add_noise_new()` draw the rows they flip.
    """
    return np.random.permutation(range(n))[: int(ratio * n)]


def _flip_values(values, positions):
    """
    A copy of a binary vector with the entries at `positions` flipped, 0 <-> 1.

    The implementation both noise bias types share.
    """
    values = np.asarray(values).copy()
    values[positions] = 1 - values[positions]
    return values


def add_redlining_bias(X, y, dataset_bias_type=(), dataset=None):
    """
    Apply a redlining dataset bias type to a dataset's labels.

    The bias is applied to the dataset itself, before it is split and before
    any expert is fit on it, so that every downstream use of the labels -- the
    expert's training targets, the `y` column of its demonstrations, and every
    metric computed against them -- sees the same biased labels.

    Parameters
    ----------
    X : pandas.DataFrame
        The dataset's input columns. Only the protected attribute `z` is used.
    y : pandas.Series
        The dataset's labels.
    dataset_bias_type : tuple, default ()
        The bias to apply, `()` (or any non-redlining bias type) leaving the
        labels untouched. The first element names the bias (one of the
        "dataset_*_redlining" bias types); the second, if given, is the
        percent of the redlined group whose label is flipped.
    dataset : str
        Name of the dataset, which decides which `z` group is redlined and
        which `y` outcome counts as the "bad" one.

    Returns
    -------
    y_biased : pandas.Series
        A copy of `y` with the bias applied, indexed and named like `y`.
    """
    name = dataset_bias_type[0] if dataset_bias_type else None
    if name not in (f"dataset_{kind}" for kind in REDLINING_KINDS):
        return y.copy()

    kind, percent = _redlining_params(dataset_bias_type, "dataset_")
    y_biased = _apply_redlining(X["z"], y, kind, percent, dataset)
    return pd.Series(y_biased, index=y.index, name=y.name)


def add_noise_bias(X, y, dataset_bias_type=()):
    """
    Apply the `dataset_noise_bias` dataset bias type to a dataset.

    Works like `add_noise_new()` of the Superhuman Fairness reference
    implementation: two independent random subsets of `noise_ratio` of the
    rows are drawn, and the labels of the first and the protected attribute
    `z` of the second are flipped. Unlike `add_noise_new()`, each of the two
    is optional, and -- as for every dataset bias type -- the noise lands on
    the dataset itself rather than on one demonstrator's training labels and
    test groups, so the expert is fit on, and everything is scored against,
    the noisy dataset.

    Only `z` is flipped, not any column of the dataset the protected attribute
    came from, just as `add_noise_new()` flips the protected attribute without
    touching the model inputs.

    Parameters
    ----------
    dataset_bias_type : tuple
        `("dataset_noise_bias", noise_ratio, label_noise, protected_noise)`:
        the fraction of rows to flip, and whether to flip their labels and
        whether to flip their protected attribute. Anything else leaves the
        dataset untouched.

    Returns
    -------
    X_biased : pandas.DataFrame
        `X`, or a copy of it with `z` flipped on the selected rows.
    y_biased : pandas.Series
        A copy of `y`, flipped on the selected rows if `label_noise`.
    """
    name = dataset_bias_type[0] if dataset_bias_type else None
    if name != "dataset_noise_bias":
        return X, y.copy()

    noise_ratio, label_noise, protected_noise = dataset_bias_type[1:]
    n = len(y)
    # Both subsets are always drawn, in `add_noise_new()`'s order, so that
    # turning either noise on or off leaves the other's rows unchanged.
    label_positions = _random_fraction_positions(n, noise_ratio)
    protected_positions = _random_fraction_positions(n, noise_ratio)

    y_biased = pd.Series(
        _flip_values(y, label_positions) if label_noise else np.asarray(y),
        index=y.index,
        name=y.name,
    )
    X_biased = X
    if protected_noise:
        X_biased = X.copy()
        X_biased["z"] = _flip_values(X["z"], protected_positions)
    return X_biased, y_biased


def add_dataset_bias(X, y, feature_types, dataset_bias_type=(), dataset=None):
    """
    Apply one dataset bias type to a dataset.

    Demo bias types, and `()`, leave the dataset unchanged: they bias the
    expert's demonstrations instead (see `add_demo_bias()`).

    Returns
    -------
    X_biased : pandas.DataFrame
        `X`, or for `dataset_noise_bias` with protected attribute noise, a
        copy of it with `z` biased.
    y_biased : pandas.Series
        A copy of `y` with the bias applied, indexed and named like `y`.
    """
    name = dataset_bias_type[0] if dataset_bias_type else None
    if name == "dataset_corruption_bias":
        return X, add_corruption_bias(X, y, feature_types, dataset_bias_type)
    if name == "dataset_noise_bias":
        return add_noise_bias(X, y, dataset_bias_type)
    return X, add_redlining_bias(X, y, dataset_bias_type, dataset)


def add_demo_bias(demos, demo_bias_type=(), dataset=None, feature_types=None):
    """
    Apply one demo bias type to an expert's demonstrations.

    The dataset is left unchanged; only the decisions the expert made, the
    demonstrations' `yhat`, are biased:

    `demo_unbalanced_redlining`, `demo_balanced_redlining`,
    `demo_perfectly_balanced_redlining`
        The dataset bias types of the same names minus "dataset_", applied to
        `yhat` in place of `y` (`[name, percent]`, percent defaulting to 0.2).
    `demo_noise_bias`
        `[demo_noise_bias, noise_ratio]`: flips the decisions of a random
        `noise_ratio` of the rows, like `add_noise()` of the Superhuman
        Fairness reference implementation does to its demonstrator's
        predictions, but keeping the demonstrations' own index rather than
        resetting it to 0..n-1.
    `demo_corruption_bias`
        `[demo_corruption_bias, clf_type, percent, noise_type, magnitude]`:
        the dataset bias type `dataset_corruption_bias` with the expert's
        decisions in place of the labels -- a `clf_type` classifier is fit to
        imitate `yhat` from the demonstrations' inputs, its parameters are
        corrupted, and its predictions replace `yhat` wherever the
        uncorrupted classifier matched it (see `_corrupt_and_relabel()`).
        Every row of `demos` is relabeled by the same corrupted classifier,
        so to bias several sets of demonstrations with one corrupted expert,
        pass them in together.

    Dataset bias types, and `()`, leave the demonstrations unchanged.

    Parameters
    ----------
    demos : pandas.DataFrame
        The demonstrations, with at least `z` and `yhat` columns.
    demo_bias_type : tuple, default ()
    dataset : str
        Name of the dataset, for the redlining bias types.
    feature_types : dict<str, list>
        Mapping of column names to their type of feature, for
        `demo_corruption_bias`.

    Returns
    -------
    demos_biased : pandas.DataFrame
        A copy of `demos` with `yhat` biased, indexed like `demos`.
    """
    name = demo_bias_type[0] if demo_bias_type else None
    if name not in DEMO_BIAS_TYPES:
        return demos

    demos = demos.copy()
    if name == "demo_noise_bias":
        (noise_ratio,) = demo_bias_type[1:]
        yhat = _flip_values(
            demos["yhat"], _random_fraction_positions(len(demos), noise_ratio)
        )
    elif name == "demo_corruption_bias":
        # The classifier imitates the expert from the inputs the expert saw:
        # never from the true labels `y`.
        yhat = _corrupt_and_relabel(
            demos.drop(columns=["yhat", "y"]),
            demos["yhat"],
            feature_types,
            *demo_bias_type[1:],
        )
    else:
        kind, percent = _redlining_params(demo_bias_type, "demo_")
        yhat = _apply_redlining(demos["z"], demos["yhat"], kind, percent, dataset)
    pct_unchanged = (np.asarray(yhat) == demos["yhat"].to_numpy()).mean() * 100.0
    logging.info(f"Demo bias type added: {demo_bias_type}")
    logging.info(f"Percent of yhat unchanged with added bias: {pct_unchanged}%")
    demos["yhat"] = np.asarray(yhat).astype(np.int64)
    return demos


def _make_corruption_clf(clf_type, feature_types):
    """Return a fresh, unfitted sklearn pipeline for the configured classifier type."""
    if clf_type == "MLP":
        clf_inst = MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
        )
    elif clf_type == "CatBoost":
        clf_inst = CatBoostClassifier(allow_writing_files=False, logging_level="Silent")
    elif clf_type == "XGBoost":
        clf_inst = XGBClassifier(eval_metric="logloss", verbosity=0)
    elif clf_type == "SVM":
        clf_inst = SVC(kernel="rbf")
    else:
        raise ValueError(f"Unknown ML_WEIGHT_ADJUST_CLF_TYPE: {clf_type}")
    return sklearn_clf_pipeline(feature_types, clf_inst)


def _ml_xgb_get_leaf_values(clf_inner):
    """Extract XGBoost leaf node values as a flat numpy array via JSON serialisation."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name
    try:
        clf_inner.get_booster().save_model(temp_path)
        with open(temp_path) as f:
            model_dict = json.load(f)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    leaves = []
    for tree in (
        model_dict.get("learner", {})
        .get("gradient_booster", {})
        .get("model", {})
        .get("trees", [])
    ):
        left_children = tree.get("left_children", [])
        base_weights = tree.get("base_weights", [])
        for j, lc in enumerate(left_children):
            # -1 or very large positive value signals a leaf (no left child)
            if (lc == -1 or lc > 1_000_000_000) and j < len(base_weights):
                leaves.append(float(base_weights[j]))
    return np.array(leaves)


def _ml_xgb_set_leaf_values(clf_inner, new_values):
    """Overwrite XGBoost leaf values in-place via JSON save/load."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name
    try:
        clf_inner.get_booster().save_model(temp_path)
        with open(temp_path) as f:
            model_dict = json.load(f)

        leaf_idx = 0
        for tree in (
            model_dict.get("learner", {})
            .get("gradient_booster", {})
            .get("model", {})
            .get("trees", [])
        ):
            left_children = tree.get("left_children", [])
            for j, lc in enumerate(left_children):
                if (lc == -1 or lc > 1_000_000_000) and leaf_idx < len(new_values):
                    lv = float(new_values[leaf_idx])
                    if j < len(tree.get("base_weights", [])):
                        tree["base_weights"][j] = lv
                    if j < len(tree.get("split_conditions", [])):
                        tree["split_conditions"][j] = lv
                    leaf_idx += 1

        with open(temp_path, "w") as f:
            json.dump(model_dict, f)
        clf_inner.get_booster().load_model(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _get_flat_clf_params(clf_pipeline, clf_type):
    """Return the learnable parameters of the inner classifier as a flat numpy array."""
    inner = clf_pipeline.named_steps["classifier"]
    if clf_type == "MLP":
        return np.concatenate(
            [c.flatten() for c in inner.coefs_]
            + [b.flatten() for b in inner.intercepts_]
        )
    elif clf_type == "CatBoost":
        return inner.get_leaf_values().copy()
    elif clf_type == "XGBoost":
        return _ml_xgb_get_leaf_values(inner)
    elif clf_type == "SVM":
        return np.concatenate([inner.dual_coef_.flatten(), inner.intercept_.flatten()])
    raise ValueError(f"Unknown clf_type: {clf_type}")


def _set_flat_clf_params(clf_pipeline, clf_type, biased_params):
    """Return a deep copy of clf_pipeline with biased_params installed."""
    new_clf = copy.deepcopy(clf_pipeline)
    orig_inner = clf_pipeline.named_steps["classifier"]
    new_inner = new_clf.named_steps["classifier"]

    if clf_type == "MLP":
        idx = 0
        for coef in new_inner.coefs_:
            n = coef.size
            coef[:] = biased_params[idx : idx + n].reshape(coef.shape)
            idx += n
        for intercept in new_inner.intercepts_:
            n = intercept.size
            intercept[:] = biased_params[idx : idx + n].reshape(intercept.shape)
            idx += n
    elif clf_type == "CatBoost":
        # copy.deepcopy() leaves the model in a non-"solid" internal state that
        # CatBoost refuses to modify in-place, so round-trip through disk first.
        with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as f:
            temp_path = f.name
        try:
            new_inner.save_model(temp_path)
            new_inner.load_model(temp_path)
            new_inner.set_leaf_values(biased_params)
        finally:
            os.unlink(temp_path)
    elif clf_type == "XGBoost":
        _ml_xgb_set_leaf_values(new_inner, biased_params)
    elif clf_type == "SVM":
        n_dual = orig_inner.dual_coef_.size
        new_inner.dual_coef_[:] = biased_params[:n_dual].reshape(
            orig_inner.dual_coef_.shape
        )
        new_inner.intercept_[:] = biased_params[n_dual:].reshape(
            orig_inner.intercept_.shape
        )
    return new_clf


def _apply_corruption(params, percent, noise_type, magnitude):
    """Return a biased copy of the flat parameter array according to the selected process."""
    params = params.copy()
    if len(params) == 0:
        return params
    if noise_type == "uniform":
        n_select = max(1, int(len(params) * percent))
        idx = np.random.choice(len(params), size=n_select, replace=False)
        params[idx] += np.random.uniform(-magnitude, magnitude, size=n_select)
        return params
    elif noise_type == "gaussian":
        n_select = max(1, int(len(params) * percent))
        idx = np.random.choice(len(params), size=n_select, replace=False)
        params[idx] += np.random.normal(0, magnitude, size=n_select)
        return params
    raise ValueError(f"Unknown noise_type: {noise_type}")


def _corrupt_and_relabel(
    X, target, feature_types, clf_type, percent, noise_type, magnitude
):
    """
    Relabel `target` with the predictions of a classifier whose learned
    parameters were corrupted.

    The implementation both corruption bias types share: the dataset bias type
    relabels the dataset's labels `y`, and the demo bias type the expert's
    decisions `yhat`.

    A `clf_type` classifier is fit on `(X, target)`, a `percent` of its
    learned parameters are perturbed by `noise_type` noise of the given
    `magnitude` (see `_apply_corruption()`), and its predictions on `X` replace
    `target` -- except wherever the uncorrupted classifier already disagreed
    with `target`, which is kept as is.

    Returns
    -------
    target_biased : pandas.Series
        A copy of `target` with the bias applied, indexed and named like
        `target`.
    """
    clf = _make_corruption_clf(clf_type, feature_types)
    clf.fit(X, target)
    unbiased_demo = generate_demo(
        clf,
        X,
        target,
        can_observe_y=False,
    )
    pred_unbiased = pd.Series(
        unbiased_demo["yhat"].to_numpy(), index=target.index, name=target.name
    )
    flat_params = _get_flat_clf_params(clf, clf_type)
    biased_params = _apply_corruption(flat_params, percent, noise_type, magnitude)
    biased_clf = _set_flat_clf_params(clf, clf_type, biased_params)
    demo = generate_demo(
        biased_clf,
        X,
        target,
        can_observe_y=False,
    )
    pred_biased = pd.Series(
        demo["yhat"].to_numpy(), index=target.index, name=target.name
    )
    # We want to preserve any bias that was already present in the original target.
    # Whenever the original target could not be predicted correctly by the unbiased classifier,
    # that indicates that the original target was likely already biased. In that case, we want
    # to keep the original target as is, and not apply the corruption bias.
    return pred_biased.where(pred_unbiased == target, target)


def add_corruption_bias(X, y, feature_types, dataset_bias_type=()):
    """
    Apply the `dataset_corruption_bias` dataset bias type to a dataset's labels.

    A classifier is fit on the dataset, its learned parameters are corrupted,
    and its predictions on the dataset become the biased labels (see
    `_corrupt_and_relabel()`). As with `add_redlining_bias()`, the bias lands
    on the dataset itself rather than on an expert's predictions, so
    everything downstream is fit on and scored against the biased labels.

    Returns
    -------
    y_biased : pandas.Series
        A copy of `y` with the bias applied, indexed and named like `y`, or
        an unchanged copy if `dataset_bias_type` is not
        `dataset_corruption_bias`.
    """
    name = dataset_bias_type[0] if dataset_bias_type else None
    if name != "dataset_corruption_bias":
        return y.copy()

    return _corrupt_and_relabel(X, y, feature_types, *dataset_bias_type[1:])


def generate_non_overfit_demos(
    exp_info,
    X,
    y,
    clf,
    n_demos=2,
    obj_set=None,
):
    """
    Generate the demonstrations of every fold.

    Returns
    -------
    demos : pandas.DataFrame
        The demonstrations of all the folds combined.
    """
    demos_folds = []

    k_fold = KFold(n_demos)
    for train, test in k_fold.split(X, y):
        X_train, y_train = (
            X.iloc[train],
            y.iloc[train],
        )
        X_test, y_test = (
            X.iloc[test],
            y.iloc[test],
        )

        if isinstance(clf, OptClfMDPPolicyExpert):
            # A ClassificationMDPPolicy can't be fit like a regular
            # classifier: it must be rebuilt (via compute_optimal_policy())
            # from this fold's own training data instead.
            clf_fold = clf.build_policy(X_train, y_train, obj_set, exp_info)
        else:
            clf_fold = copy.deepcopy(clf)
            clf_fold.fit(X_train, y_train)
        demos_folds.append(generate_demo(clf_fold, X_test, y_test))

    # Combine all the demos into one dataframe.
    # Preserves the original data sample order from the dataset.
    return pd.concat(demos_folds)


def generate_mu_and_demos(
    exp_info,
    X,
    y,
    clf,
    obj_set,
    n_demos=2,
):
    """
    Improved implementation of `generate_demos_k_folds` that uses k-folds to generate
    demonstrations for all data samples without overlap and also without overfitting.
    Returns demos as the combined demonstrations from all folds, and mu as the feature
    expectations of all demos.

    Any dataset bias has already been applied to `X` and `y` by the caller
    (see `add_dataset_bias()`), so the demonstrations generated here carry the
    bias of the dataset they were generated from. Demo bias types are applied
    to these demonstrations afterwards, by the caller (see `add_demo_bias()`).

    Parameters
    ----------
    X : numpy.ndarray
        The X training data reserved for generating the demonstrations.
    y : array-like
        The y training data reserved for generating the demonstrations.
    clf : sklearn.pipeline.Pipeline, ClassifierMixin
        Sklearn classifier pipeline.
    obj_set : ObjectiveSet
        The objective set.
    n_demos : int, default 2
        The number of demonstrations to generate (also the k in k folds).

    Returns
    -------
    mu : numpy.ndarray, shape(1, len(obj_set.objectives))
        The feature expectations across all demonstrations.
    demos : pandas.DataFrame
        A dataframe of all the demonstrations across the K-folds.
    """
    mu = np.zeros((1, len(obj_set.objectives)))  # demo feat exp

    if n_demos < 2:
        n_demos = 2  # KFold requires at least 2 splits

    demos = generate_non_overfit_demos(
        exp_info,
        X,
        y,
        clf,
        n_demos=n_demos,
        obj_set=obj_set,
    )

    # Compute mu for the combined demos
    mu[0] = obj_set.compute_demo_feature_exp(demos)
    logging.info(f"mu[0] = {mu[0]}")

    return mu, demos


def policy_error(
    w,
    muE,
    muL,
    dot_weights_feat_exp=True,
):
    """
    Computes how far a learned policy's feature expectations are from the
    expert's, i.e. wT(muE-muL).

    Parameters
    ----------
    w : np.array<float>
        Weights.
    muE : array-like<np.array<float>>, shape (n_expert_demos, len(len(objective_set.objectives)))
        Array of all expert feature expectations.
    muL : array-like<float>
        The learned policy's feature expectations.
    dot_weights_feat_exp : bool, default True
        If True, error = l2_norm( (muE - muL).dot(w) )
        If False, error = l2_norm(muE - muL) * l2_norm(w)
        Use the latter if all feat exp should be non-zero weights. Helps avoid
        the issue of nulling out feat exp componets that are hard to match.

    Returns
    -------
    mu_deltas : array<float>
        The array of differences between muE and muL.
    l2_mu_delta : float
        The l2 norm of the muE and muL deltas.
    abs_l2_mu_delta : float
        The l2 norm of the muE and muL deltas, without normalizing by muE.
    err : float
        The error of the policy.
    """
    # Trying this out. Make mu delta errors RELATIVE to their magnitude. So
    # adding the muE.mean(axis=0) as a denominator
    mu_deltas = (muE.mean(axis=0) - muL) / muE.mean(axis=0)

    abs_mu_deltas = muE.mean(axis=0) - muL

    if dot_weights_feat_exp:
        err = np.linalg.norm(w * mu_deltas, ord=2)
    else:
        err = np.linalg.norm(mu_deltas) * np.linalg.norm(w, ord=2)

    l2_mu_delta = np.linalg.norm(mu_deltas, ord=2)
    abs_l2_mu_delta = np.linalg.norm(abs_mu_deltas, ord=2)

    return mu_deltas, l2_mu_delta, abs_l2_mu_delta, err
