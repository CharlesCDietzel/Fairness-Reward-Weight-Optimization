import itertools
import logging
import numpy as np
import pandas as pd


class OptimizationProblem:
    """
    One linear program of an MDP: minimize `c @ x` subject to
    `A_eq @ x == b_eq`, `A_ub @ x <= b_ub` and `x >= 0`.

    Attributes
    ----------
    objective_costs : dict<str, numpy.ndarray> or None
        Each objective's unweighted contribution to `c` in this problem (the
        cost vector of the one split of it the problem uses), by objective
        name. With `objective_weights`, lets a solver tell how `c` depends on
        the reward weights (see `frwo.rl.reduced_lp`).
    objective_weights : dict<str, float> or None
        The reward weight of each objective in `objective_costs`, so that
        `c == sum(objective_weights[k] * objective_costs[k] for k in ...)`.
    """

    def __init__(
        self,
        name,
        c,
        A_eq=None,
        b_eq=None,
        b_ub=None,
        A_ub=None,
        objective_costs=None,
        objective_weights=None,
    ):
        self.name = name
        self.c = c
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.b_ub = b_ub
        self.A_ub = A_ub
        self.objective_costs = objective_costs
        self.objective_weights = objective_weights


class ObjectiveSplit:

    def __init__(self, name, parent, c, b_ub=None, A_ub=None):
        self.name = name
        self.parent = parent
        self.c = c
        self.b_ub = b_ub
        self.A_ub = A_ub


class Objective:
    def __init__(self, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs

    def fit(self, ldf):
        raise NotImplementedError()

    def to_splits(self):
        raise NotImplementedError()

    def compute_feat_exp(self, demo):
        raise NotImplementedError()


class LinearObjective(Objective):

    def __init__(self):
        self.n_splits = 1
        super().__init__()

    def fit(self, ldf):
        self.b_ub_row_ = None
        self.c_ = self._construct_reward(ldf)
        return self

    def to_splits(self):
        """
        Returns objective as one or more splits.

        Parameters
        ----------
        None

        Returns
        -------
        array<ObjectiveSplit>
        """
        split = ObjectiveSplit(
            name=self.name,
            parent=self,
            c=self.c_,
            A_ub=None,
            b_ub=None,
        )
        return [split]

    def _construct_reward(self, ldf):
        raise NotImplementedError()


class AbsoluteValueObjective(Objective):

    def __init__(self, n_splits=None):
        # Backwards compatible default: existing objectives still have 2 splits.
        # Subclasses can either pass n_splits=4 or set self.n_splits before super().
        if n_splits is None:
            n_splits = getattr(self, "n_splits", 2)

        self.n_splits = n_splits
        super().__init__()

    def fit(self, ldf):
        """
        Fits all LP sign splits for an absolute-value-style objective.

        Existing 2-split objectives remain compatible because they implement:
            _compute_A_ub_row__split1
            _compute_A_ub_row__split2
            _construct_reward__split1
            _construct_reward__split2

        New objectives can implement more splits, e.g. split1 ... split4.
        """
        for split_idx in range(1, self.n_splits + 1):
            A_func_name = f"_compute_A_ub_row__split{split_idx}"
            c_func_name = f"_construct_reward__split{split_idx}"

            if not hasattr(self, A_func_name):
                raise NotImplementedError(
                    f"{self.__class__.__name__} must implement {A_func_name}"
                )

            if not hasattr(self, c_func_name):
                raise NotImplementedError(
                    f"{self.__class__.__name__} must implement {c_func_name}"
                )

            A_ub = getattr(self, A_func_name)(ldf)
            c = getattr(self, c_func_name)(ldf)
            b_ub = self._compute_b_ub(ldf, split_idx)

            # Preserve the old attribute naming convention.
            setattr(self, f"A_ub_row__split{split_idx}_", A_ub)
            setattr(self, f"c__split{split_idx}_", c)
            setattr(self, f"b_ub__split{split_idx}_", b_ub)

        # Backwards compatibility with existing code that expects this attr.
        self.b_ub_row_ = getattr(self, "b_ub__split1_", 0)

        return self

    def to_splits(self):
        """
        Returns objective as one or more splits.

        Returns
        -------
        array<ObjectiveSplit>
        """
        splits = []

        for split_idx in range(1, self.n_splits + 1):
            split = ObjectiveSplit(
                name=f"{self.name} Split{split_idx}",
                parent=self,
                c=getattr(self, f"c__split{split_idx}_"),
                b_ub=getattr(self, f"b_ub__split{split_idx}_", self.b_ub_row_),
                A_ub=getattr(self, f"A_ub_row__split{split_idx}_"),
            )
            splits.append(split)

        return splits

    def _compute_b_ub(self, ldf, split_idx):
        """
        Default b_ub for one-row constraints.

        Existing objectives use one inequality per split, so scalar 0 remains
        correct. Objectives with multiple constraints per split, such as
        EqualizedOddsObjective, can override this and return e.g. np.zeros(2).
        """
        method_name = f"_compute_b_ub__split{split_idx}"

        if hasattr(self, method_name):
            return getattr(self, method_name)(ldf)

        return np.array([0.0])

    def _compute_A_ub_row__split1(self, ldf):
        raise NotImplementedError()

    def _compute_A_ub_row__split2(self, ldf):
        raise NotImplementedError()

    def _construct_reward__split1(self, ldf):
        raise NotImplementedError()

    def _construct_reward__split2(self, ldf):
        raise NotImplementedError()


class AccuracyObjective(LinearObjective):

    def __init__(self):
        self.name = "Acc"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        mu = np.mean(demo["yhat"] == demo["y"])

        return mu

    def _construct_reward(self, ldf):
        """
        Constructs the reward function when the objective is accuracy.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        ------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        p_all = ldf["mu0"].sum() / 2
        ldf["r"] = (ldf["yhat"] == ldf["y"]).astype(float) / p_all
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)


class AccuracyParityObjective(AbsoluteValueObjective):

    def __init__(self):
        self.name = "AccPar"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_y_giv_z_eq_0 = (
            (demo["yhat"] == demo["y"]) & (demo["z"] == 0)
        ).sum() / (demo["z"] == 0).sum()
        p_yhat_eq_y_giv_z_eq_1 = (
            (demo["yhat"] == demo["y"]) & (demo["z"] == 1)
        ).sum() / (demo["z"] == 1).sum()
        mu = 1 - max(
            [
                p_yhat_eq_y_giv_z_eq_0 - p_yhat_eq_y_giv_z_eq_1,
                p_yhat_eq_y_giv_z_eq_1 - p_yhat_eq_y_giv_z_eq_0,
            ]
        )
        if np.isnan(mu):
            mu = 1

        return mu

    def _compute_A_ub_row__split1(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=y|z=0) >= P(yhat=y|z=1)
            ```
        which is Accuracy Parity opt_problem 1.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        ldf = ldf.copy()
        filt__yhaty_z0 = (ldf["z"] == 0) & (ldf["yhat"] == ldf["y"])
        filt__yhaty_z1 = (ldf["z"] == 1) & (ldf["yhat"] == ldf["y"])
        p_z0 = ldf[ldf["z"] == 0]["mu0"].sum() / 2
        p_z1 = ldf[ldf["z"] == 1]["mu0"].sum() / 2
        ldf["A_ub"] = 0.0
        ldf.loc[filt__yhaty_z0, "A_ub"] = -1 / p_z0
        ldf.loc[filt__yhaty_z1, "A_ub"] = 1 / p_z1

        return ldf["A_ub"].to_numpy()

    def _compute_A_ub_row__split2(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=y|z=0) <= P(yhat=y|z=1)
            ```
        which is Accuracy Parity opt_problem 1.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        ldf = ldf.copy()
        filt__yhaty_z0 = (ldf["z"] == 0) & (ldf["yhat"] == ldf["y"])
        filt__yhaty_z1 = (ldf["z"] == 1) & (ldf["yhat"] == ldf["y"])
        p_z0 = ldf[ldf["z"] == 0]["mu0"].sum() / 2
        p_z1 = ldf[ldf["z"] == 1]["mu0"].sum() / 2
        ldf["A_ub"] = 0.0
        ldf.loc[filt__yhaty_z0, "A_ub"] = 1 / p_z0
        ldf.loc[filt__yhaty_z1, "A_ub"] = -1 / p_z1

        return ldf["A_ub"].to_numpy()

    def _construct_reward__split1(self, ldf):
        """
        Constructs the reward function for Accuracy Parity  opt_problem 1.
        opt_problem 1 is when we constrain P(yhat=y|z=0) >= P(yhat=y|z=1), in
        which case the reward penalizes giving the Z=0 group the correct
        prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhaty_z0 = (ldf["z"] == 0) & (ldf["yhat"] == ldf["y"])
        filt__yhaty_z1 = (ldf["z"] == 1) & (ldf["yhat"] == ldf["y"])
        p_z0 = ldf[ldf["z"] == 0]["mu0"].sum() / 2
        p_z1 = ldf[ldf["z"] == 1]["mu0"].sum() / 2
        ldf["r"] = np.zeros(len(ldf))
        ldf.loc[filt__yhaty_z0, "r"] = -1 / p_z0
        ldf.loc[filt__yhaty_z1, "r"] = 1 / p_z1
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)

    def _construct_reward__split2(self, ldf):
        """
        Constructs the reward function for Accuracy Parity  opt_problem 1.
        opt_problem 1 is when we constrain P(yhat=y|z=0) <= P(yhat=y|z=1), in
        which case the reward penalizes giving the Z=1 group the correct
        prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhaty_z0 = (ldf["z"] == 0) & (ldf["yhat"] == ldf["y"])
        filt__yhaty_z1 = (ldf["z"] == 1) & (ldf["yhat"] == ldf["y"])
        p_z0 = ldf[ldf["z"] == 0]["mu0"].sum() / 2
        p_z1 = ldf[ldf["z"] == 1]["mu0"].sum() / 2
        ldf["r"] = np.zeros(len(ldf))
        ldf.loc[filt__yhaty_z0, "r"] = 1 / p_z0
        ldf.loc[filt__yhaty_z1, "r"] = -1 / p_z1
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)


class DemographicParityObjective(AbsoluteValueObjective):

    def __init__(self):
        self.name = "DemPar"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_1_giv_z_eq_0 = ((demo["yhat"] == 1) & (demo["z"] == 0)).sum() / (
            demo["z"] == 0
        ).sum()
        p_yhat_eq_1_giv_z_eq_1 = ((demo["yhat"] == 1) & (demo["z"] == 1)).sum() / (
            demo["z"] == 1
        ).sum()
        mu = 1 - max(
            [
                p_yhat_eq_1_giv_z_eq_0 - p_yhat_eq_1_giv_z_eq_1,
                p_yhat_eq_1_giv_z_eq_1 - p_yhat_eq_1_giv_z_eq_0,
            ]
        )

        if np.isnan(mu):
            mu = 1

        return mu

    def _compute_A_ub_row__split1(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|z=0) >= P(yhat=1|z=1)
            ```
        which is Demographic Parity opt_problem 1.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        ldf = ldf.copy()
        filt__yhat1_z0 = (ldf["z"] == 0) & (ldf["yhat"] == 1)
        filt__yhat1_z1 = (ldf["z"] == 1) & (ldf["yhat"] == 1)
        p_z0 = ldf[ldf["z"] == 0]["mu0"].sum() / 2
        p_z1 = ldf[ldf["z"] == 1]["mu0"].sum() / 2
        ldf["A_ub"] = 0.0
        ldf.loc[filt__yhat1_z0, "A_ub"] = -1 / p_z0
        ldf.loc[filt__yhat1_z1, "A_ub"] = 1 / p_z1

        return ldf["A_ub"].to_numpy()

    def _compute_A_ub_row__split2(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|z=1) >= P(yhat=1|z=0)
            ```
        which is Demographic Parity opt_problem 2.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.


        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        ldf = ldf.copy()
        filt__yhat1_z0 = (ldf["z"] == 0) & (ldf["yhat"] == 1)
        filt__yhat1_z1 = (ldf["z"] == 1) & (ldf["yhat"] == 1)
        p_z0 = ldf[ldf["z"] == 0]["mu0"].sum() / 2
        p_z1 = ldf[ldf["z"] == 1]["mu0"].sum() / 2
        ldf["A_ub"] = 0.0
        ldf.loc[filt__yhat1_z0, "A_ub"] = 1 / p_z0
        ldf.loc[filt__yhat1_z1, "A_ub"] = -1 / p_z1

        return ldf["A_ub"].to_numpy()

    def _construct_reward__split1(self, ldf):
        """
        Constructs the reward function for Demographic Parity opt_problem 1.
        opt_problem 1 is when we constrain P(yhat=1|z=0) >= P(yhat=1|z=1), in
        which case the reward penalizes giving the Z=0 group the positive
        prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_z0 = (ldf["z"] == 0) & (ldf["yhat"] == 1)
        filt__yhat1_z1 = (ldf["z"] == 1) & (ldf["yhat"] == 1)
        p_z0 = ldf[ldf["z"] == 0]["mu0"].sum() / 2
        p_z1 = ldf[ldf["z"] == 1]["mu0"].sum() / 2
        ldf["r"] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_z0, "r"] = -1 / p_z0
        ldf.loc[filt__yhat1_z1, "r"] = 1 / p_z1
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)

    def _construct_reward__split2(self, ldf):
        """
        Constructs the reward function for Demographic Parity opt_problem 2.
        opt_problem 2 is when we constrain P(yhat=1|z=1) >= P(yhat=1|z=0), in
        which case the reward penalizes giving the Z=1 group the positive
        prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_z0 = (ldf["z"] == 0) & (ldf["yhat"] == 1)
        filt__yhat1_z1 = (ldf["z"] == 1) & (ldf["yhat"] == 1)
        p_z0 = ldf[ldf["z"] == 0]["mu0"].sum() / 2
        p_z1 = ldf[ldf["z"] == 1]["mu0"].sum() / 2
        ldf["r"] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_z0, "r"] = 1 / p_z0
        ldf.loc[filt__yhat1_z1, "r"] = -1 / p_z1
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)


class EqualOpportunityObjective(AbsoluteValueObjective):

    def __init__(self):
        self.name = "EqOpp"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_1_giv_y_eq_1_z_eq_0 = (
            (demo["yhat"] == 1) & (demo["y"] == 1) & (demo["z"] == 0)
        ).sum() / ((demo["y"] == 1) & (demo["z"] == 0)).sum()
        p_yhat_eq_1_giv_y_eq_1_z_eq_1 = (
            (demo["yhat"] == 1) & (demo["y"] == 1) & (demo["z"] == 1)
        ).sum() / ((demo["y"] == 1) & (demo["z"] == 1)).sum()
        mu = 1 - max(
            [
                p_yhat_eq_1_giv_y_eq_1_z_eq_0 - p_yhat_eq_1_giv_y_eq_1_z_eq_1,
                p_yhat_eq_1_giv_y_eq_1_z_eq_1 - p_yhat_eq_1_giv_y_eq_1_z_eq_0,
            ]
        )
        if np.isnan(mu):
            mu = 1

        return mu

    def _compute_A_ub_row__split1(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|y=1,z=0) >= P(yhat=1|y=1,z=1)
            ```
        which is Equal Opportunity opt_problem 1.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        ldf = ldf.copy()
        filt__yhat1_y1_z0 = (ldf["z"] == 0) & (ldf["y"] == 1) & (ldf["yhat"] == 1)
        filt__yhat1_y1_z1 = (ldf["z"] == 1) & (ldf["y"] == 1) & (ldf["yhat"] == 1)
        p_z0_y1 = ldf[(ldf["z"] == 0) & (ldf["y"] == 1)]["mu0"].sum() / 2
        p_z1_y1 = ldf[(ldf["z"] == 1) & (ldf["y"] == 1)]["mu0"].sum() / 2
        ldf["A_ub"] = 0.0
        ldf.loc[filt__yhat1_y1_z0, "A_ub"] = -1 / p_z0_y1
        ldf.loc[filt__yhat1_y1_z1, "A_ub"] = 1 / p_z1_y1

        return ldf["A_ub"].to_numpy()

    def _compute_A_ub_row__split2(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|y=1,z=1) >= P(yhat=1|y=1,z=0)
            ```
        which is Equal Opportunity opt_problem 2.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        ldf = ldf.copy()
        filt__yhat1_y1_z0 = (ldf["z"] == 0) & (ldf["y"] == 1) & (ldf["yhat"] == 1)
        filt__yhat1_y1_z1 = (ldf["z"] == 1) & (ldf["y"] == 1) & (ldf["yhat"] == 1)
        p_z0_y1 = ldf[(ldf["z"] == 0) & (ldf["y"] == 1)]["mu0"].sum() / 2
        p_z1_y1 = ldf[(ldf["z"] == 1) & (ldf["y"] == 1)]["mu0"].sum() / 2
        ldf["A_ub"] = 0.0
        ldf.loc[filt__yhat1_y1_z0, "A_ub"] = 1 / p_z0_y1
        ldf.loc[filt__yhat1_y1_z1, "A_ub"] = -1 / p_z1_y1

        return ldf["A_ub"].to_numpy()

    def _construct_reward__split1(self, ldf):
        """
        Constructs the reward function for Equal Opportunity opt_problem 1.
        opt_problem 1 is when we constrain P(yhat=1|y=1,z=0) >= P(yhat=1|y=1,z=1), in
        which case the reward penalizes giving the Z=0 group the positive
        prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_y1_z0 = (ldf["z"] == 0) & (ldf["y"] == 1) & (ldf["yhat"] == 1)
        filt__yhat1_y1_z1 = (ldf["z"] == 1) & (ldf["y"] == 1) & (ldf["yhat"] == 1)
        p_y1_z0 = ldf[(ldf["y"] == 1) & (ldf["z"] == 0)]["mu0"].sum() / 2
        p_y1_z1 = ldf[(ldf["y"] == 1) & (ldf["z"] == 1)]["mu0"].sum() / 2
        ldf["r"] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_y1_z0, "r"] = -1 / p_y1_z0
        ldf.loc[filt__yhat1_y1_z1, "r"] = 1 / p_y1_z1
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)

    def _construct_reward__split2(self, ldf):
        """
        Constructs the reward function for Equal Opportunity opt_problem 2.
        opt_problem 2 is when we constrain P(yhat=1|y=1,z=1) >= P(yhat=1|y=1,z=0), in
        which case the reward penalizes giving the Z=1 group the positive
        prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_y1_z0 = (ldf["z"] == 0) & (ldf["y"] == 1) & (ldf["yhat"] == 1)
        filt__yhat1_y1_z1 = (ldf["z"] == 1) & (ldf["y"] == 1) & (ldf["yhat"] == 1)
        p_y1_z0 = ldf[(ldf["y"] == 1) & (ldf["z"] == 0)]["mu0"].sum() / 2
        p_y1_z1 = ldf[(ldf["y"] == 1) & (ldf["z"] == 1)]["mu0"].sum() / 2
        ldf["r"] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_y1_z0, "r"] = 1 / p_y1_z0
        ldf.loc[filt__yhat1_y1_z1, "r"] = -1 / p_y1_z1
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)


class FalsePositiveRateParityObjective(AbsoluteValueObjective):

    def __init__(self):
        self.name = "FPRPar"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_1_giv_y_eq_0_z_eq_0 = (
            (demo["yhat"] == 1) & (demo["y"] == 0) & (demo["z"] == 0)
        ).sum() / ((demo["y"] == 0) & (demo["z"] == 0)).sum()
        p_yhat_eq_1_giv_y_eq_0_z_eq_1 = (
            (demo["yhat"] == 1) & (demo["y"] == 0) & (demo["z"] == 1)
        ).sum() / ((demo["y"] == 0) & (demo["z"] == 1)).sum()
        mu = 1 - max(
            [
                p_yhat_eq_1_giv_y_eq_0_z_eq_0 - p_yhat_eq_1_giv_y_eq_0_z_eq_1,
                p_yhat_eq_1_giv_y_eq_0_z_eq_1 - p_yhat_eq_1_giv_y_eq_0_z_eq_0,
            ]
        )
        if np.isnan(mu):
            mu = 1

        return mu

    def _compute_A_ub_row__split1(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|y=0,z=0) >= P(yhat=1|y=0,z=1)
            ```
        which is False Positive Rate Parity opt_problem 1.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        ldf = ldf.copy()
        filt__yhat1_y0_z0 = (ldf["z"] == 0) & (ldf["y"] == 0) & (ldf["yhat"] == 1)
        filt__yhat1_y0_z1 = (ldf["z"] == 1) & (ldf["y"] == 0) & (ldf["yhat"] == 1)
        p_z0_y0 = ldf[(ldf["z"] == 0) & (ldf["y"] == 0)]["mu0"].sum() / 2
        p_z1_y0 = ldf[(ldf["z"] == 1) & (ldf["y"] == 0)]["mu0"].sum() / 2
        ldf["A_ub"] = 0.0
        ldf.loc[filt__yhat1_y0_z0, "A_ub"] = -1 / p_z0_y0
        ldf.loc[filt__yhat1_y0_z1, "A_ub"] = 1 / p_z1_y0

        return ldf["A_ub"].to_numpy()

    def _compute_A_ub_row__split2(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=1|y=0,z=1) >= P(yhat=1|y=0,z=0)
            ```
        which is False Positive Rate Parity opt_problem 2.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        ldf = ldf.copy()
        filt__yhat1_y0_z0 = (ldf["z"] == 0) & (ldf["y"] == 0) & (ldf["yhat"] == 1)
        filt__yhat1_y0_z1 = (ldf["z"] == 1) & (ldf["y"] == 0) & (ldf["yhat"] == 1)
        p_z0_y0 = ldf[(ldf["z"] == 0) & (ldf["y"] == 0)]["mu0"].sum() / 2
        p_z1_y0 = ldf[(ldf["z"] == 1) & (ldf["y"] == 0)]["mu0"].sum() / 2
        ldf["A_ub"] = 0.0
        ldf.loc[filt__yhat1_y0_z0, "A_ub"] = 1 / p_z0_y0
        ldf.loc[filt__yhat1_y0_z1, "A_ub"] = -1 / p_z1_y0

        return ldf["A_ub"].to_numpy()

    def _construct_reward__split1(self, ldf):
        """
        Constructs the reward function for False Positive Rate Parity,
        opt_problem 1, which is when we constrain
        P(yhat=1|y=0,z=0) >= P(yhat=1|y=0,z=1), in which case the reward
        penalizes giving the Z=0 group the positive prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_y0_z0 = (ldf["z"] == 0) & (ldf["y"] == 0) & (ldf["yhat"] == 1)
        filt__yhat1_y0_z1 = (ldf["z"] == 1) & (ldf["y"] == 0) & (ldf["yhat"] == 1)
        p_y0_z0 = ldf[(ldf["y"] == 0) & (ldf["z"] == 0)]["mu0"].sum() / 2
        p_y0_z1 = ldf[(ldf["y"] == 0) & (ldf["z"] == 1)]["mu0"].sum() / 2
        ldf["r"] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_y0_z0, "r"] = -1 / p_y0_z0
        ldf.loc[filt__yhat1_y0_z1, "r"] = 1 / p_y0_z1
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)

    def _construct_reward__split2(self, ldf):
        """
        Constructs the reward function for False Positive Rate Parity,
        opt_problem 2, which is when we constrain
        P(yhat=1|y=0,z=0) <= P(yhat=1|y=0,z=1), in which case the reward
        penalizes giving the Z=1 group the positive prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_y0_z0 = (ldf["z"] == 0) & (ldf["y"] == 0) & (ldf["yhat"] == 1)
        filt__yhat1_y0_z1 = (ldf["z"] == 1) & (ldf["y"] == 0) & (ldf["yhat"] == 1)
        p_y0_z0 = ldf[(ldf["y"] == 0) & (ldf["z"] == 0)]["mu0"].sum() / 2
        p_y0_z1 = ldf[(ldf["y"] == 0) & (ldf["z"] == 1)]["mu0"].sum() / 2
        ldf["r"] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_y0_z0, "r"] = 1 / p_y0_z0
        ldf.loc[filt__yhat1_y0_z1, "r"] = -1 / p_y0_z1
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)


class EqualizedOddsObjective(AbsoluteValueObjective):

    def __init__(self):
        self.name = "EqOdds"
        super().__init__(n_splits=4)

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of Equalized Odds.

        Equalized Odds requires both:
            TPR parity:
                P(yhat=1 | y=1, z=0) == P(yhat=1 | y=1, z=1)

            FPR parity:
                P(yhat=1 | y=0, z=0) == P(yhat=1 | y=0, z=1)

        This returns:
            1 - max(|TPR_z0 - TPR_z1|, |FPR_z0 - FPR_z1|)
        """

        def cond_pos_rate(y_value, z_value):
            denom = ((demo["y"] == y_value) & (demo["z"] == z_value)).sum()

            if denom == 0:
                return np.nan

            numer = (
                (demo["yhat"] == 1) & (demo["y"] == y_value) & (demo["z"] == z_value)
            ).sum()

            return numer / denom

        tpr_z0 = cond_pos_rate(y_value=1, z_value=0)
        tpr_z1 = cond_pos_rate(y_value=1, z_value=1)
        fpr_z0 = cond_pos_rate(y_value=0, z_value=0)
        fpr_z1 = cond_pos_rate(y_value=0, z_value=1)

        # np.max (not the built-in max) is required here: Python's builtin
        # max() does not reliably propagate NaN when comparing two floats,
        # since the result depends on argument order (max(nan, 5) == nan,
        # but max(5, nan) == 5). np.max propagates NaN regardless of order,
        # which is required for the `np.isnan(mu)` fallback below to work
        # consistently when only one of the two rates is undefined.
        mu = 1 - np.max([abs(tpr_z0 - tpr_z1), abs(fpr_z0 - fpr_z1)])

        if np.isnan(mu):
            mu = 1

        return mu

    def _compute_b_ub(self, ldf, split_idx):
        """
        Each Equalized Odds split has three inequality constraints (see
        `_compute_A_ub_row_for_dominant_term`), so A_ub is 3 x n and b_ub
        must have length 3.
        """
        return np.zeros(3)

    def _signed_rate_diff_row(self, ldf, y_value, z0_ge_z1):
        """
        Constructs one inequality row.

        If z0_ge_z1 is True, construct:

            P(yhat=1 | y=y_value, z=1)
            -
            P(yhat=1 | y=y_value, z=0)

        which is <= 0 exactly when:

            P(yhat=1 | y=y_value, z=0)
            >=
            P(yhat=1 | y=y_value, z=1)

        If z0_ge_z1 is False, construct the opposite sign:

            P(yhat=1 | y=y_value, z=0)
            -
            P(yhat=1 | y=y_value, z=1)
        """
        ldf = ldf.copy()

        filt_yhat1_y_z0 = (ldf["z"] == 0) & (ldf["y"] == y_value) & (ldf["yhat"] == 1)

        filt_yhat1_y_z1 = (ldf["z"] == 1) & (ldf["y"] == y_value) & (ldf["yhat"] == 1)

        p_z0_y = ldf[(ldf["z"] == 0) & (ldf["y"] == y_value)]["mu0"].sum() / 2
        p_z1_y = ldf[(ldf["z"] == 1) & (ldf["y"] == y_value)]["mu0"].sum() / 2

        inv_z0_y = 1 / p_z0_y
        inv_z1_y = 1 / p_z1_y

        ldf["A_ub"] = 0.0

        if z0_ge_z1:
            # rate_z1 - rate_z0
            ldf.loc[filt_yhat1_y_z0, "A_ub"] = -inv_z0_y
            ldf.loc[filt_yhat1_y_z1, "A_ub"] = inv_z1_y
        else:
            # rate_z0 - rate_z1
            ldf.loc[filt_yhat1_y_z0, "A_ub"] = inv_z0_y
            ldf.loc[filt_yhat1_y_z1, "A_ub"] = -inv_z1_y

        return ldf["A_ub"].to_numpy()

    def _rate_diff_terms(self, ldf):
        """
        Returns the four candidate linear terms whose max equals
        max(|TPR_z0 - TPR_z1|, |FPR_z0 - FPR_z1|):

            0: TPR_z0 - TPR_z1
            1: TPR_z1 - TPR_z0   (== -term 0)
            2: FPR_z0 - FPR_z1
            3: FPR_z1 - FPR_z0   (== -term 2)

        Since max(a, -a, b, -b) == max(|a|, |b|), minimizing the max of
        these four terms over the policy is exactly equivalent to
        minimizing max(|TPR_z0 - TPR_z1|, |FPR_z0 - FPR_z1|).
        """
        return [
            self._signed_rate_diff_row(
                ldf=ldf, y_value=1, z0_ge_z1=False
            ),  # TPR_z0 - TPR_z1
            self._signed_rate_diff_row(
                ldf=ldf, y_value=1, z0_ge_z1=True
            ),  # TPR_z1 - TPR_z0
            self._signed_rate_diff_row(
                ldf=ldf, y_value=0, z0_ge_z1=False
            ),  # FPR_z0 - FPR_z1
            self._signed_rate_diff_row(
                ldf=ldf, y_value=0, z0_ge_z1=True
            ),  # FPR_z1 - FPR_z0
        ]

    def _compute_A_ub_row_for_dominant_term(self, ldf, dominant_idx):
        """
        Constructs the 3-row A_ub matrix for the split in which
        `_rate_diff_terms(ldf)[dominant_idx]` is the largest of the four
        candidate terms, i.e. it equals max(|TPR diff|, |FPR diff|).

        For each of the other three terms `other`, this adds the
        constraint `other - dominant <= 0` (dominant >= other).
        """
        terms = self._rate_diff_terms(ldf)
        dominant = terms[dominant_idx]
        others = [term for i, term in enumerate(terms) if i != dominant_idx]

        return np.vstack([other - dominant for other in others])

    def _construct_reward_for_dominant_term(self, ldf, dominant_idx):
        """
        Constructs the linear reward for the split in which
        `_rate_diff_terms(ldf)[dominant_idx]` is the dominant (largest)
        term, i.e. equal to max(|TPR diff|, |FPR diff|) in that region.

        Minimizing max(|TPR diff|, |FPR diff|) within this region is just
        minimizing the dominant term itself, so `c` is that term's row
        directly (scipy-style LP solvers minimize c^T x).
        """
        dominant = self._rate_diff_terms(ldf)[dominant_idx]

        return (
            dominant.to_numpy()
            if hasattr(dominant, "to_numpy")
            else np.asarray(dominant)
        )

    # ------------------------------------------------------------------
    # Four "dominant term" splits
    # ------------------------------------------------------------------
    #
    # max(|TPR_z0 - TPR_z1|, |FPR_z0 - FPR_z1|)
    #   == max(TPR_z0 - TPR_z1, TPR_z1 - TPR_z0, FPR_z0 - FPR_z1, FPR_z1 - FPR_z0)
    #
    # Each split fixes which of those four terms is the largest (the
    # "dominant" term) and minimizes it subject to that term actually
    # being >= the other three. Taking the best (highest) reward across
    # all four splits recovers min_policy max(|TPR diff|, |FPR diff|),
    # exactly matching the aggregation used by `compute_feat_exp`.
    #
    # Split1: TPR_z0 - TPR_z1 dominates.
    # Split2: TPR_z1 - TPR_z0 dominates.
    # Split3: FPR_z0 - FPR_z1 dominates.
    # Split4: FPR_z1 - FPR_z0 dominates.
    # ------------------------------------------------------------------

    def _compute_A_ub_row__split1(self, ldf):
        return self._compute_A_ub_row_for_dominant_term(ldf=ldf, dominant_idx=0)

    def _compute_A_ub_row__split2(self, ldf):
        return self._compute_A_ub_row_for_dominant_term(ldf=ldf, dominant_idx=1)

    def _compute_A_ub_row__split3(self, ldf):
        return self._compute_A_ub_row_for_dominant_term(ldf=ldf, dominant_idx=2)

    def _compute_A_ub_row__split4(self, ldf):
        return self._compute_A_ub_row_for_dominant_term(ldf=ldf, dominant_idx=3)

    def _construct_reward__split1(self, ldf):
        return self._construct_reward_for_dominant_term(ldf=ldf, dominant_idx=0)

    def _construct_reward__split2(self, ldf):
        return self._construct_reward_for_dominant_term(ldf=ldf, dominant_idx=1)

    def _construct_reward__split3(self, ldf):
        return self._construct_reward_for_dominant_term(ldf=ldf, dominant_idx=2)

    def _construct_reward__split4(self, ldf):
        return self._construct_reward_for_dominant_term(ldf=ldf, dominant_idx=3)


class TrueNegativeRateParityObjective(AbsoluteValueObjective):

    def __init__(self):
        self.name = "TNRPar"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_0_giv_y_eq_0_z_eq_0 = (
            (demo["yhat"] == 0) & (demo["y"] == 0) & (demo["z"] == 0)
        ).sum() / ((demo["y"] == 0) & (demo["z"] == 0)).sum()
        p_yhat_eq_0_giv_y_eq_0_z_eq_1 = (
            (demo["yhat"] == 0) & (demo["y"] == 0) & (demo["z"] == 1)
        ).sum() / ((demo["y"] == 0) & (demo["z"] == 1)).sum()
        mu = 1 - max(
            [
                p_yhat_eq_0_giv_y_eq_0_z_eq_0 - p_yhat_eq_0_giv_y_eq_0_z_eq_1,
                p_yhat_eq_0_giv_y_eq_0_z_eq_1 - p_yhat_eq_0_giv_y_eq_0_z_eq_0,
            ]
        )
        if np.isnan(mu):
            mu = 1

        return mu

    def _compute_A_ub_row__split1(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=0|y=0,z=0) >= P(yhat=0|y=0,z=1)
            ```
        which is True Negative Rate Parity opt_problem 1.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        ldf = ldf.copy()
        filt__yhat0_y0_z0 = (ldf["z"] == 0) & (ldf["y"] == 0) & (ldf["yhat"] == 0)
        filt__yhat0_y0_z1 = (ldf["z"] == 1) & (ldf["y"] == 0) & (ldf["yhat"] == 0)
        p_z0_y0 = ldf[(ldf["z"] == 0) & (ldf["y"] == 0)]["mu0"].sum() / 2
        p_z1_y0 = ldf[(ldf["z"] == 1) & (ldf["y"] == 0)]["mu0"].sum() / 2
        ldf["A_ub"] = 0.0
        ldf.loc[filt__yhat0_y0_z0, "A_ub"] = -1 / p_z0_y0
        ldf.loc[filt__yhat0_y0_z1, "A_ub"] = 1 / p_z1_y0

        return ldf["A_ub"].to_numpy()

    def _compute_A_ub_row__split2(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=0|y=0,z=1) >= P(yhat=0|y=0,z=0)
            ```
        which is True Negative Rate Parity opt_problem 2.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        ldf = ldf.copy()
        filt__yhat0_y0_z0 = (ldf["z"] == 0) & (ldf["y"] == 0) & (ldf["yhat"] == 0)
        filt__yhat0_y0_z1 = (ldf["z"] == 1) & (ldf["y"] == 0) & (ldf["yhat"] == 0)
        p_z0_y0 = ldf[(ldf["z"] == 0) & (ldf["y"] == 0)]["mu0"].sum() / 2
        p_z1_y0 = ldf[(ldf["z"] == 1) & (ldf["y"] == 0)]["mu0"].sum() / 2
        ldf["A_ub"] = 0.0
        ldf.loc[filt__yhat0_y0_z0, "A_ub"] = 1 / p_z0_y0
        ldf.loc[filt__yhat0_y0_z1, "A_ub"] = -1 / p_z1_y0

        return ldf["A_ub"].to_numpy()

    def _construct_reward__split1(self, ldf):
        """
        Constructs the reward function for True Negative Rate Parity,
        opt_problem 1, which is when we constrain
        P(yhat=0|y=0,z=0) >= P(yhat=0|y=0,z=1), in which case the reward
        penalizes giving the Z=0 group the positive prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat0_y0_z0 = (ldf["z"] == 0) & (ldf["y"] == 0) & (ldf["yhat"] == 0)
        filt__yhat0_y0_z1 = (ldf["z"] == 1) & (ldf["y"] == 0) & (ldf["yhat"] == 0)
        p_y0_z0 = ldf[(ldf["y"] == 0) & (ldf["z"] == 0)]["mu0"].sum() / 2
        p_y0_z1 = ldf[(ldf["y"] == 0) & (ldf["z"] == 1)]["mu0"].sum() / 2
        ldf["r"] = np.zeros(len(ldf))
        ldf.loc[filt__yhat0_y0_z0, "r"] = -1 / p_y0_z0
        ldf.loc[filt__yhat0_y0_z1, "r"] = 1 / p_y0_z1
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)

    def _construct_reward__split2(self, ldf):
        """
        Constructs the reward function for True Negative Rate Parity,
        opt_problem 2, which is when we constrain
        P(yhat=0|y=0,z=0) <= P(yhat=0|y=0,z=1), in which case the reward
        penalizes giving the Z=1 group the positive prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat0_y0_z0 = (ldf["z"] == 0) & (ldf["y"] == 0) & (ldf["yhat"] == 0)
        filt__yhat0_y0_z1 = (ldf["z"] == 1) & (ldf["y"] == 0) & (ldf["yhat"] == 0)
        p_y0_z0 = ldf[(ldf["y"] == 0) & (ldf["z"] == 0)]["mu0"].sum() / 2
        p_y0_z1 = ldf[(ldf["y"] == 0) & (ldf["z"] == 1)]["mu0"].sum() / 2
        ldf["r"] = np.zeros(len(ldf))
        ldf.loc[filt__yhat0_y0_z0, "r"] = 1 / p_y0_z0
        ldf.loc[filt__yhat0_y0_z1, "r"] = -1 / p_y0_z1
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)


class FalseNegativeRateParityObjective(AbsoluteValueObjective):

    def __init__(self):
        self.name = "FNRPar"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_0_giv_y_eq_1_z_eq_0 = (
            (demo["yhat"] == 0) & (demo["y"] == 1) & (demo["z"] == 0)
        ).sum() / ((demo["y"] == 1) & (demo["z"] == 0)).sum()
        p_yhat_eq_0_giv_y_eq_1_z_eq_1 = (
            (demo["yhat"] == 0) & (demo["y"] == 1) & (demo["z"] == 1)
        ).sum() / ((demo["y"] == 1) & (demo["z"] == 1)).sum()
        mu = 1 - max(
            [
                p_yhat_eq_0_giv_y_eq_1_z_eq_0 - p_yhat_eq_0_giv_y_eq_1_z_eq_1,
                p_yhat_eq_0_giv_y_eq_1_z_eq_1 - p_yhat_eq_0_giv_y_eq_1_z_eq_0,
            ]
        )
        if np.isnan(mu):
            mu = 1

        return mu

    def _compute_A_ub_row__split1(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=0|y=1,z=0) >= P(yhat=0|y=1,z=1)
            ```
        which is False Negative Rate Parity opt_problem 1.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        ldf = ldf.copy()
        filt__yhat0_y1_z0 = (ldf["z"] == 0) & (ldf["y"] == 1) & (ldf["yhat"] == 0)
        filt__yhat0_y1_z1 = (ldf["z"] == 1) & (ldf["y"] == 1) & (ldf["yhat"] == 0)
        p_z0_y1 = ldf[(ldf["z"] == 0) & (ldf["y"] == 1)]["mu0"].sum() / 2
        p_z1_y1 = ldf[(ldf["z"] == 1) & (ldf["y"] == 1)]["mu0"].sum() / 2
        ldf["A_ub"] = 0.0
        ldf.loc[filt__yhat0_y1_z0, "A_ub"] = -1 / p_z0_y1
        ldf.loc[filt__yhat0_y1_z1, "A_ub"] = 1 / p_z1_y1

        return ldf["A_ub"].to_numpy()

    def _compute_A_ub_row__split2(self, ldf):
        """
        Constructs the linear equation for the constraint that
            ```
            P(yhat=0|y=1,z=1) >= P(yhat=0|y=1,z=0)
            ```
        which is False Negative Rate opt_problem 2.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        ldf['A_ub'] : pandas.Series<float>
        """
        ldf = ldf.copy()
        filt__yhat0_y1_z0 = (ldf["z"] == 0) & (ldf["y"] == 1) & (ldf["yhat"] == 0)
        filt__yhat0_y1_z1 = (ldf["z"] == 1) & (ldf["y"] == 1) & (ldf["yhat"] == 0)
        p_z0_y1 = ldf[(ldf["z"] == 0) & (ldf["y"] == 1)]["mu0"].sum() / 2
        p_z1_y1 = ldf[(ldf["z"] == 1) & (ldf["y"] == 1)]["mu0"].sum() / 2
        ldf["A_ub"] = 0.0
        ldf.loc[filt__yhat0_y1_z0, "A_ub"] = 1 / p_z0_y1
        ldf.loc[filt__yhat0_y1_z1, "A_ub"] = -1 / p_z1_y1

        return ldf["A_ub"].to_numpy()

    def _construct_reward__split1(self, ldf):
        """
        Constructs the reward function for False Negative Rate Parity,
        opt_problem 1, which is when we constrain
        P(yhat=0|y=1,z=0) >= P(yhat=0|y=1,z=1), in which case the reward
        penalizes giving the Z=0 group the negative prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat0_y1_z0 = (ldf["z"] == 0) & (ldf["y"] == 1) & (ldf["yhat"] == 0)
        filt__yhat0_y1_z1 = (ldf["z"] == 1) & (ldf["y"] == 1) & (ldf["yhat"] == 0)
        p_y1_z0 = ldf[(ldf["y"] == 1) & (ldf["z"] == 0)]["mu0"].sum() / 2
        p_y1_z1 = ldf[(ldf["y"] == 1) & (ldf["z"] == 1)]["mu0"].sum() / 2
        ldf["r"] = np.zeros(len(ldf))
        ldf.loc[filt__yhat0_y1_z0, "r"] = -1 / p_y1_z0
        ldf.loc[filt__yhat0_y1_z1, "r"] = 1 / p_y1_z1
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)

    def _construct_reward__split2(self, ldf):
        """
        Constructs the reward function for False Negative Rate Parity,
        opt_problem 2, which is when we constrain
        P(yhat=0|y=1,z=0) <= P(yhat=0|y=1,z=1), in which case the reward
        penalizes giving the Z=1 group the positive prediction.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        -------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat0_y1_z0 = (ldf["z"] == 0) & (ldf["y"] == 1) & (ldf["yhat"] == 0)
        filt__yhat0_y1_z1 = (ldf["z"] == 1) & (ldf["y"] == 1) & (ldf["yhat"] == 0)
        p_y1_z0 = ldf[(ldf["y"] == 1) & (ldf["z"] == 0)]["mu0"].sum() / 2
        p_y1_z1 = ldf[(ldf["y"] == 1) & (ldf["z"] == 1)]["mu0"].sum() / 2
        ldf["r"] = np.zeros(len(ldf))
        ldf.loc[filt__yhat0_y1_z0, "r"] = 1 / p_y1_z0
        ldf.loc[filt__yhat0_y1_z1, "r"] = -1 / p_y1_z1
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)


class PredictiveParityObjective(Objective):
    """
    Not intended for optimization, just feature expectations. Can't optimize
    this with LP, since it's not a linear measure.
    """

    def __init__(self):
        self.name = "PredPar"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_y_eq_1_giv_yhat_eq_1_z_eq_0 = (
            (demo["yhat"] == 1) & (demo["y"] == 1) & (demo["z"] == 0)
        ).sum() / ((demo["yhat"] == 1) & (demo["z"] == 0)).sum()
        p_y_eq_1_giv_yhat_eq_1_z_eq_1 = (
            (demo["yhat"] == 1) & (demo["y"] == 1) & (demo["z"] == 1)
        ).sum() / ((demo["yhat"] == 1) & (demo["z"] == 1)).sum()
        mu = 1 - max(
            [
                p_y_eq_1_giv_yhat_eq_1_z_eq_0 - p_y_eq_1_giv_yhat_eq_1_z_eq_1,
                p_y_eq_1_giv_yhat_eq_1_z_eq_1 - p_y_eq_1_giv_yhat_eq_1_z_eq_0,
            ]
        )
        if np.isnan(mu):
            mu = 1

        return mu


class NegativePredictiveParityObjective(Objective):
    """
    Not intended for optimization, just feature expectations. Can't optimize
    this with LP, since it's not a linear measure.
    """

    def __init__(self):
        self.name = "NegPredPar"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_y_eq_0_giv_yhat_eq_0_z_eq_0 = (
            (demo["yhat"] == 0) & (demo["y"] == 0) & (demo["z"] == 0)
        ).sum() / ((demo["yhat"] == 0) & (demo["z"] == 0)).sum()
        p_y_eq_0_giv_yhat_eq_0_z_eq_1 = (
            (demo["yhat"] == 0) & (demo["y"] == 0) & (demo["z"] == 1)
        ).sum() / ((demo["yhat"] == 0) & (demo["z"] == 1)).sum()
        mu = 1 - max(
            [
                p_y_eq_0_giv_yhat_eq_0_z_eq_0 - p_y_eq_0_giv_yhat_eq_0_z_eq_1,
                p_y_eq_0_giv_yhat_eq_0_z_eq_1 - p_y_eq_0_giv_yhat_eq_0_z_eq_0,
            ]
        )
        if np.isnan(mu):
            mu = 1

        return mu


class GroupPositiveRateZ0Objective(LinearObjective):

    def __init__(self, z=0):
        self.z = z
        self.name = f"PR_Z{self.z}"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_1_giv_z_eq_z = ((demo["yhat"] == 1) & (demo["z"] == self.z)).sum() / (
            demo["z"] == self.z
        ).sum()

        mu = p_yhat_eq_1_giv_z_eq_z

        if np.isnan(mu):
            mu = 1

        return mu

    def _construct_reward(self, ldf):
        """
        Constructs the reward function component for this objective.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        ------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_giv_z = (ldf["z"] == self.z) & (ldf["yhat"] == 1)
        # Divide by 2 since ldf has two duplicate rows per state
        p_z = ldf[ldf["z"] == self.z]["mu0"].sum() / 2
        ldf["r"] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_giv_z, "r"] = 1 / p_z
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)


class GroupPositiveRateZ1Objective(GroupPositiveRateZ0Objective):

    def __init__(self):
        super().__init__(z=1)


class GroupNegativeRateZ0Objective(LinearObjective):

    def __init__(self, z=0):
        self.z = z
        self.name = f"NR_Z{self.z}"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_0_giv_z_eq_z = ((demo["yhat"] == 0) & (demo["z"] == self.z)).sum() / (
            demo["z"] == self.z
        ).sum()

        mu = p_yhat_eq_0_giv_z_eq_z

        if np.isnan(mu):
            mu = 1

        return mu

    def _construct_reward(self, ldf):
        """
        Constructs the reward function component for this objective.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        ------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat0_giv_z = (ldf["z"] == self.z) & (ldf["yhat"] == 0)
        # Divide by 2 since ldf has two duplicate rows per state
        p_z = ldf[ldf["z"] == self.z]["mu0"].sum() / 2
        ldf["r"] = np.zeros(len(ldf))
        ldf.loc[filt__yhat0_giv_z, "r"] = 1 / p_z
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)


class GroupNegativeRateZ1Objective(GroupNegativeRateZ0Objective):

    def __init__(self):
        super().__init__(z=1)


class GroupTruePositiveRateZ0Objective(LinearObjective):

    def __init__(self, z=0):
        self.z = z
        self.name = f"TPR_Z{self.z}"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_1_giv_y_eq_1_z_eq_z = (
            (demo["yhat"] == 1) & (demo["y"] == 1) & (demo["z"] == self.z)
        ).sum() / ((demo["y"] == 1) & (demo["z"] == self.z)).sum()

        mu = p_yhat_eq_1_giv_y_eq_1_z_eq_z

        if np.isnan(mu):
            mu = 1

        return mu

    def _construct_reward(self, ldf):
        """
        Constructs the reward function component for this objective.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        ------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_giv_y1_z = (
            (ldf["z"] == self.z) & (ldf["y"] == 1) & (ldf["yhat"] == 1)
        )
        # Divide by 2 since ldf has two duplicate rows per state
        p_y1_z = ldf[(ldf["y"] == 1) & (ldf["z"] == self.z)]["mu0"].sum() / 2
        ldf["r"] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_giv_y1_z, "r"] = 1 / p_y1_z
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)


class GroupTruePositiveRateZ1Objective(GroupTruePositiveRateZ0Objective):

    def __init__(self):
        super().__init__(z=1)


class GroupTrueNegativeRateZ0Objective(LinearObjective):

    def __init__(self, z=0):
        self.z = z
        self.name = f"TNR_Z{self.z}"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_0_giv_y_eq_0_z_eq_z = (
            (demo["yhat"] == 0) & (demo["y"] == 0) & (demo["z"] == self.z)
        ).sum() / ((demo["y"] == 0) & (demo["z"] == self.z)).sum()

        mu = p_yhat_eq_0_giv_y_eq_0_z_eq_z

        if np.isnan(mu):
            mu = 1

        return mu

    def _construct_reward(self, ldf):
        """
        Constructs the reward function component for this objective.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        ------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat0_giv_y0_z = (
            (ldf["z"] == self.z) & (ldf["y"] == 0) & (ldf["yhat"] == 0)
        )
        p_y0_z = ldf[(ldf["y"] == 0) & (ldf["z"] == self.z)]["mu0"].sum() / 2
        ldf["r"] = np.zeros(len(ldf))
        ldf.loc[filt__yhat0_giv_y0_z, "r"] = 1 / p_y0_z
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)


class GroupTrueNegativeRateZ1Objective(GroupTrueNegativeRateZ0Objective):

    def __init__(self):
        super().__init__(z=1)


class GroupFalsePositiveRateZ0Objective(LinearObjective):

    def __init__(self, z=0):
        self.z = z
        self.name = f"FPR_Z{self.z}"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_1_giv_y_eq_0_z_eq_z = (
            (demo["yhat"] == 1) & (demo["y"] == 0) & (demo["z"] == self.z)
        ).sum() / ((demo["y"] == 0) & (demo["z"] == self.z)).sum()

        mu = p_yhat_eq_1_giv_y_eq_0_z_eq_z

        if np.isnan(mu):
            mu = 1

        return mu

    def _construct_reward(self, ldf):
        """
        Constructs the reward function component for this objective.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        ------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat1_giv_y0_z = (
            (ldf["z"] == self.z) & (ldf["y"] == 0) & (ldf["yhat"] == 1)
        )
        p_y0_z = ldf[(ldf["y"] == 0) & (ldf["z"] == self.z)]["mu0"].sum() / 2
        ldf["r"] = np.zeros(len(ldf))
        ldf.loc[filt__yhat1_giv_y0_z, "r"] = 1 / p_y0_z
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)


class GroupFalsePositiveRateZ1Objective(GroupFalsePositiveRateZ0Objective):

    def __init__(self):
        super().__init__(z=1)


class GroupFalseNegativeRateZ0Objective(LinearObjective):

    def __init__(self, z=0):
        self.z = z
        self.name = f"FNR_Z{self.z}"
        super().__init__()

    def compute_feat_exp(self, demo):
        """
        Computes the feature expectation representation of the objective on
        the provided demonstration.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns:
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets
        """
        p_yhat_eq_0_giv_y_eq_1_z_eq_z = (
            (demo["yhat"] == 0) & (demo["y"] == 1) & (demo["z"] == self.z)
        ).sum() / ((demo["y"] == 1) & (demo["z"] == self.z)).sum()

        mu = p_yhat_eq_0_giv_y_eq_1_z_eq_z

        if np.isnan(mu):
            mu = 1

        return mu

    def _construct_reward(self, ldf):
        """
        Constructs the reward function component for this objective.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Returns
        ------
        c : np.array<float>, len(2*len(df))
            The objective function for the linear program.
        """
        ldf = ldf.copy()
        filt__yhat0_giv_y1_z = (
            (ldf["z"] == self.z) & (ldf["y"] == 1) & (ldf["yhat"] == 0)
        )
        p_y1_z = ldf[(ldf["y"] == 1) & (ldf["z"] == self.z)]["mu0"].sum() / 2
        ldf["r"] = np.zeros(len(ldf))
        ldf.loc[filt__yhat0_giv_y1_z, "r"] = 1 / p_y1_z
        c = -1 * ldf["r"]  # Negative since maximizing not minimizing

        return c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)


class GroupFalseNegativeRateZ1Objective(GroupFalseNegativeRateZ0Objective):

    def __init__(self):
        super().__init__(z=1)


class ObjectiveSet:
    """
    The set of all objectives that make up the space of possible objectives in
    the reward function. An obj_set gets paired with a set of reward
    weights to form the full reward function.

    Parameters
    ----------
    objectives : array-like<Objective>
        Array of all objectives.

    Attributes
    ----------
    opt_problems_ : list<OptimizationProblem>
        Optimization problems.
    """

    def __init__(self, objectives):
        self.objectives = objectives
        self.opt_problems_ = None
        self.fitted_ldf_ = None

    def compute_demo_feature_exp(self, demo):
        """
        Computes the feature expectations for a set of demonstrations.
        Does NOT need to be fit for this method to work.

        Parameters
        ----------
        demo : pandas.DataFrame
            Demonstrations. Each demonstration represents an iteration of a
            trained classifier and its predictions on a hold-out set. Columns are
                **`X` columns : all input columns (i.e. `X`)
                yhat : predictions
                y : ground truth targets

        Returns
        -------
        mu : array<float>, len(len(self.objectives))
            The feature expectations.
        """
        mu = [obj.compute_feat_exp(demo) for obj in self.objectives]
        return mu

    def fit(self, reward_weights, ldf, A_eq, b_eq):
        """
        Constructs all optimization problems for the given objectives,
        reward weights, and MDP (represented by ldf, A_eq, b_eq).

        Parameters
        ----------
        reward_weights : dict<str, float>
            Keys are objective identifiers. Values are their respective reward
            weights.
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.
        A_eq : 2-D array
            The equality constraint matrix. Each row of ``A_eq`` specifies the
            coefficients of a linear equality constraint on ``x``.
        b_eq : 1-D array
            The equality constraint vector. Each element of ``A_eq @ x`` must
            equal the corresponding element of ``b_eq``.

        Sets
        ----
        opt_problems_ : list<OptimizationProblem>
            Optimization problems.

        Returns
        -------
        self
        """
        self.fit_objectives(ldf)
        return self.build_opt_problems(reward_weights, ldf, A_eq, b_eq)

    def fit_objectives(self, ldf):
        """
        Fits every objective to the MDP represented by `ldf`. The first half of
        `fit()`: nothing it computes depends on the reward weights, so
        optimization problems for any number of reward weights can then be
        built from it with `build_opt_problems()`.

        Parameters
        ----------
        ldf : pandas.DataFrame
            "Lambda dataframe". One row for each state and action combination.

        Sets
        ----
        fitted_ldf_ : pandas.DataFrame
            `ldf`, which the objectives are now fit to.

        Returns
        -------
        self
        """
        self.objectives = [obj.fit(ldf) for obj in self.objectives]
        self.fitted_ldf_ = ldf
        return self

    def build_opt_problems(self, reward_weights, ldf, A_eq, b_eq):
        """
        Constructs all optimization problems for the given reward weights from
        objectives already fit to `ldf` by `fit_objectives()`. The second half
        of `fit()`.

        Parameters
        ----------
        See `fit()`.

        Sets
        ----
        opt_problems_ : list<OptimizationProblem>
            Optimization problems.

        Returns
        -------
        self
        """
        assert self.fitted_ldf_ is ldf, (
            "the objectives must be fit to this `ldf` (with fit_objectives()) "
            "before building its optimization problems"
        )

        abs_val_splits = []
        linear_splits = []
        abs_val_split_generator = []
        counter = 0
        for obj in self.objectives:
            if obj.n_splits == 1:
                linear_splits += obj.to_splits()
            else:
                # Handles 2-split objectives AND multi-split objectives (e.g. EqOdds with 4).
                splits = obj.to_splits()
                abs_val_splits += splits
                abs_val_split_generator.append(
                    list(range(counter, counter + obj.n_splits))
                )
                counter += obj.n_splits

        # itertools.product over an empty list of iterables yields a single empty tuple,
        # but we make this explicit so the case of "no abs-val objectives" is unambiguous.
        abs_val_splits_perms = list(itertools.product(*abs_val_split_generator))
        if not abs_val_splits_perms:
            abs_val_splits_perms = [()]

        opt_problems = []
        for split_indexes in abs_val_splits_perms:
            _abs_val_splits = [abs_val_splits[idx] for idx in split_indexes]
            all_splits = linear_splits + _abs_val_splits
            name = ",".join([split.name for split in all_splits])
            # vstack correctly stacks both 1-D rows (2-split objectives)
            # and 2-D row-blocks (e.g. EqualizedOdds, which contributes 2 rows
            # per split: one TPR constraint + one FPR constraint).
            if _abs_val_splits:
                A_ub = np.vstack(
                    [
                        np.atleast_2d(np.asarray(split.A_ub, dtype=float))
                        for split in _abs_val_splits
                    ]
                )
                # concatenate handles per-split b_ub vectors of unequal length
                # (e.g. length 1 for normal objectives, length 2 for EqOdds).
                b_ub = np.concatenate(
                    [
                        np.atleast_1d(np.asarray(split.b_ub, dtype=float))
                        for split in _abs_val_splits
                    ]
                )

                # Sanity check: one bound per inequality row.
                assert (
                    A_ub.shape[0] == b_ub.shape[0]
                ), f"A_ub has {A_ub.shape[0]} rows but b_ub has {b_ub.shape[0]} entries"
            else:
                A_ub = None
                b_ub = None
            # Initialize a zero vector of the correct length, then accumulate the
            # reward-weighted contribution of every split (linear + selected
            # absolute-value splits).
            if not all_splits:
                raise ValueError("ObjectiveSet.fit requires at least one objective.")

            c_len = np.asarray(all_splits[0].c, dtype=float).shape[0]
            c = np.zeros(c_len, dtype=float)
            objective_costs = {}
            for split in all_splits:
                split_c = np.asarray(split.c, dtype=float)
                c = c + reward_weights[split.parent.name] * split_c
                parent = split.parent.name
                objective_costs[parent] = (
                    objective_costs[parent] + split_c
                    if parent in objective_costs
                    else split_c
                )

            opt_problem = OptimizationProblem(
                name=name,
                A_eq=A_eq,
                b_eq=b_eq,
                A_ub=A_ub,
                b_ub=b_ub,
                c=c,
                objective_costs=objective_costs,
                objective_weights={
                    parent: reward_weights[parent] for parent in objective_costs
                },
            )
            opt_problems.append(opt_problem)

        self.opt_problems_ = opt_problems
        return self

    def reset(self):
        """
        Re-initializes each of the objectives and self.opt_problems_.

        Parameters
        ----------
        N/A

        Unsets
        ------
        opt_problems_
        fitted_ldf_

        Returns
        -------
        None
        """
        for obj in self.objectives:
            obj.__init__(*obj._init_args, **obj._init_kwargs)

        self.opt_problems_ = None
        self.fitted_ldf_ = None


# Objective lookup. Defined here, alongside the objectives themselves, so that
# any module can resolve an objective name without having to import
# `frwo.experiment_utils` (which imports this module).
OBJ_LOOKUP_BY_NAME = {
    "Acc": AccuracyObjective,
    "AccPar": AccuracyParityObjective,
    "DemPar": DemographicParityObjective,
    "EqOpp": EqualOpportunityObjective,
    "FPRPar": FalsePositiveRateParityObjective,
    "EqOdds": EqualizedOddsObjective,
    "TNRPar": TrueNegativeRateParityObjective,
    "FNRPar": FalseNegativeRateParityObjective,
    "PredPar": PredictiveParityObjective,
    "NegPredPar": NegativePredictiveParityObjective,
    "PR_Z0": GroupPositiveRateZ0Objective,
    "PR_Z1": GroupPositiveRateZ1Objective,
    "NR_Z0": GroupNegativeRateZ0Objective,
    "NR_Z1": GroupNegativeRateZ1Objective,
    "TPR_Z0": GroupTruePositiveRateZ0Objective,
    "TPR_Z1": GroupTruePositiveRateZ1Objective,
    "TNR_Z0": GroupTrueNegativeRateZ0Objective,
    "TNR_Z1": GroupTrueNegativeRateZ1Objective,
    "FPR_Z0": GroupFalsePositiveRateZ0Objective,
    "FPR_Z1": GroupFalsePositiveRateZ1Objective,
    "FNR_Z0": GroupFalseNegativeRateZ0Objective,
    "FNR_Z1": GroupFalseNegativeRateZ1Objective,
}
