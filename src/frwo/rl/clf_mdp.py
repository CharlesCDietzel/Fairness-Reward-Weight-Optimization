import copy
import itertools
import logging
import numbers
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csc_array, issparse

from frwo.rl.reduced_lp import ReducedLPSolver

# from line_profiler import LineProfiler


class ClassificationMDP:
    """
    Parameters
    ----------
    gamma : float, range, [0, 1)
        The discount factor. Should be close to zero since this is a
        classification mdp, but cannot be zero exactly otherwise the linear
        program won't converge.
    obj_set : ObjectiveSet
        The objective set.
    x_cols : list<str>
        The columns that are used in the state (along with `z` and `y`).

    Attributes
    ----------
    b_eq_ : numpy.array<float>
        A.k.a. "mu0". Initial state probabilities.
    A_eq_ : scipy.sparse.csc_array, shape (n_rows, 2*n_states_)
        The state-action transition matrix, followed (if `restrict_y`) by the
        action equality constraints. Entries with magnitude <= HiGHS'
        `small_matrix_value` are omitted (see `_compute_A_eq`).
    A_eq_solve_ : scipy.sparse.csc_array
        `A_eq_` as it is passed to the LP solver (see `_to_solver_matrix`):
        the same object as `A_eq_`.
    lp_solver_ : frwo.rl.reduced_lp.ReducedLPSolver or None
        Solves the optimization problems in reduced form, warm-started, with
        ties broken the same way every time (see `frwo.rl.reduced_lp`). None
        when the reduction doesn't apply (see `_build_lp_solver`). Shared by
        the copies `with_reward_weights()` makes, so that their solves
        warm-start each other.
    n_states_ : int
        Number of states.
    state_reducer_ : dict<str, dict<?, ?>>
        Specifies which state input columns and which values get replaced with
        default values. Used to reduce the state space by replacing infrequent
        state values with default values.
    reduced_state_df_ : pandas.DataFrame
        Index is state index, columns are features, mu0, and optimal policy
        actions.
    reduced_state_lookup_ : dict<tuple, int>
        Maps classification features to its MDP state index.
    ldf_ : pandas.DataFrame
        "Lambda dataframe". One row for each state and action combination.
        Columns are **x_cols, z, y, yhat.
    b_ub_dem_par_ : np.array<float>, len(n_states_)
        The uppber bound `b` for Demographic Parity Split1 and Split2.
    opt_problems_ : array<dict>
        Array of each linear optimization "sub" problem to solve. Each
        opt_problem is a dict with the following structure:
            'A_eq': self.A_eq_,
            'b_eq': self.b_eq_,
            'A_ub': numpy.ndarray<float>, len(num feat exp with abs val)
                The upper bound linear equation constraints for the split that
                occurs as the result of absolute value signs in the feat exp.
            'b_ub': np.array<float>, len(num feat exp with abs val)
                The uppber bound `b` for the linear equation constraint for the
                split that occurs as a result of absolute value signs in the
                feat exp.
            'c': array-like<float>, len(n_states*n_actions)
                Reward vector for the opt_problem.
    """

    def __init__(self, gamma, obj_set, x_cols):
        self.gamma = gamma
        self.obj_set = obj_set
        self.x_cols = x_cols
        self.state_reducer_ = {}
        self.reduced_state_df_ = None
        self.reduced_state_lookup_ = None
        self.n_states = None
        self.A_eq_ = None
        self.A_eq_solve_ = None
        self.lp_solver_ = None
        self.b_eq_ = None
        self.ldf_ = None
        self.opt_problems_ = None

    def fit(self, reward_weights, clf_df, min_freq_fill_pct=0, restrict_y=True):
        """
        Fits the MDP to `clf_df` (`fit_states()`), then builds its optimization
        problems for `reward_weights` (`fit_rewards()`).

        Parameters
        ----------
        reward_weights : dict<str, float>
            Keys are objective identifiers. Values are their respective reward
            weights.
        clf_df : pandas.DataFrame
            Classification dataset. Required columns:
                'z' : int. Binary protected attribute.
                'y' : int. Binary target variable.

        min_freq_fill_pct, float, range[0, 1), default 0
            Minimum frequency for each input variable to not get replaced by a
            default value.
        restrict_y : bool, deafult True
            If True, policy must have same action for any x,z combo, regardless
            of y.

        Sets Attributes
        ---------------
        b_eq_
        A_eq_
        state_reducer_
        reduced_state_lookup_
        reduced_state_df_
        n_states_
        ldf_
        opt_problems_

        Returns
        -------
        self
        """
        self.fit_states(
            clf_df, min_freq_fill_pct=min_freq_fill_pct, restrict_y=restrict_y
        )
        self.fit_rewards(reward_weights)
        return None

    def fit_states(self, clf_df, min_freq_fill_pct=0, restrict_y=True):
        """
        Fits the states, transitions and constraints of the MDP to `clf_df`:
        everything `fit()` sets that doesn't depend on the reward weights.

        Parameters
        ----------
        See `fit()`.

        Sets Attributes
        ---------------
        b_eq_
        A_eq_
        A_eq_solve_
        lp_solver_
        state_reducer_
        reduced_state_lookup_
        reduced_state_df_
        n_states_
        ldf_

        Returns
        -------
        None
        """
        clf_df = clf_df.copy()

        # Set the min frequency replacement values to reduce state
        # min_freq_fill_pcts : dict<str, (float, range [0,1], ?)>, default {}
        #     Dictionary with col_name -> (min_freq_pct, default_val)
        #     that specifies the minimum frequency to replace a each input
        #     attribute with a default value.  It's a hacky equivalent to what
        #     the `min_frequency` parameter does for the scikitlearn
        #     OneHotEncoder.
        min_freq_fill_pcts = {}
        for x in self.x_cols:
            min_freq_fill_pcts[x] = (
                min_freq_fill_pct,
                -555,  # Distinct value for too infrequent values.
                # clf_df[x].value_counts().sort_values().index[-1],  # most freq.
            )
        logging.debug("\nmin_freq_fill_pcts:")
        logging.debug(min_freq_fill_pcts)

        # Generate the state_reducer_ object that replaces infrequent state
        # values with defaults.
        logging.debug("\nFitting state_reducer_ ...")
        for x in min_freq_fill_pcts.keys():
            min_freq, default_val = min_freq_fill_pcts[x]
            freq = clf_df.groupby(x).size() / len(clf_df)
            mean_target_rel_delta = (
                clf_df.groupby(x)["y"].mean() - clf_df["y"].mean()
            ) / clf_df["y"].mean()
            # NEW as of 01/07/2024
            # Only use default override for categorical variables if it has a
            # relative mean target encoding > MIN_TARGET_MEAN_ENC_REL_DELTA and
            # its freq > min_freq, OR if its freq is 3x more than min_freq.
            # This is intending to make it so that I can add more categorical
            # variables in datasets without it taking so long.
            for x_val in (freq[freq < 1 * min_freq]).index:
                # MIN_TARGET_MEAN_ENC_REL_DELTA = .1
                # if mean_target_rel_delta.loc[x_val] > MIN_TARGET_MEAN_ENC_REL_DELTA and freq[x_val] > min_freq:
                #     logging.debug(f"\n\t\tBypassing category override; {x}: {x_val}, freq={freq[x_val]:.3f}, y_rel_delta={mean_target_rel_delta[x_val]:.3f}")
                #     continue
                try:
                    if x not in self.state_reducer_:
                        self.state_reducer_[x] = {}

                    self.state_reducer_[x][x_val] = default_val
                    mask = clf_df[x] == x_val
                    try:
                        clf_df.loc[mask, x] = default_val
                    except TypeError:
                        # Casting to object matches pandas <3 behavior,
                        # which silently upcast columns (e.g. bool, str)
                        # to object when assigning an incompatible value.
                        clf_df[x] = clf_df[x].astype(object)
                        clf_df.loc[mask, x] = default_val
                except KeyError:
                    continue

        logging.debug(f"\nstate_reducer_: \n{self.state_reducer_}")
        # Generate the reduced_state_df_ and the X,y -> state mapping
        self.reduced_state_df_ = (
            clf_df.groupby(self.x_cols + ["z", "y"])
            .size()
            .reset_index()
            .rename(columns={0: "count"})
        )

        # Keys are built column-wise rather than with iterrows(), which is much
        # slower; they hash and compare equal to iterrows()' `tuple(row)` keys.
        state_cols = self.reduced_state_df_.iloc[:, :-1]
        self.reduced_state_lookup_ = {}
        state_counter = 0
        for key in zip(*(state_cols[col].tolist() for col in state_cols.columns)):
            self.reduced_state_lookup_[key] = state_counter
            state_counter += 1

        # Cache n_states since frequently used in computations
        self.n_states_ = state_counter

        # Compute `mu0` (initial state probabilities)
        logging.debug("Computing b_eq ...")
        self.reduced_state_df_["mu0"] = (
            self.reduced_state_df_["count"] / self.reduced_state_df_["count"].sum()
        )
        self.reduced_state_df_ = self.reduced_state_df_.drop(columns="count")
        logging.debug(f"n_states: {len(self.reduced_state_df_)}")

        # Generate the lambda dataframe (state-action indexed)
        logging.debug("Generating the lambda dataframe ...")
        self.ldf_ = self._generate_lambda_linear_equations(self.reduced_state_df_)

        # Compute b_eq, which consists of two parts:
        #   1: mu0
        #   2: zeros (for action equality for same y-values)
        self.b_eq_ = self.reduced_state_df_["mu0"]
        if restrict_y:
            self.b_eq_ = np.concatenate(
                [
                    self.b_eq_,
                    np.zeros(
                        self.ldf_.groupby(self.x_cols + ["z", "yhat"]).size().shape[0]
                    ),  # action equality for y
                ]
            )

        # Compute transition matrix linear equations `A_eq`, which consists of
        # two parts:
        #   1: Transition matrix
        #   2: Action equality for same y-values
        logging.debug("Computing transition matrix linear equations A_eq ...")

        # profiler = LineProfiler()
        # profiler.add_function(self._compute_A_eq)
        # profiler.enable()

        self.A_eq_ = self._compute_A_eq(
            mu0=self.reduced_state_df_["mu0"],
            ldf=self.ldf_,
            x_cols=self.x_cols,
            restrict_y=restrict_y,
        )

        # profiler.disable()
        # profiler.print_stats()

        # `A_eq_` as the LP solver is given it. `A_eq_` is already built in
        # that form, so this is the same object.
        self.A_eq_solve_ = _to_solver_matrix(self.A_eq_)

        self.lp_solver_ = self._build_lp_solver(restrict_y)

        return None

    def _build_lp_solver(self, restrict_y):
        """
        The `ReducedLPSolver` of this MDP, or None if its LP can't be reduced:
        when some off-diagonal `-gamma * mu0` term of `A_eq_` is kept (gamma
        too large), or some y-pair constraint coefficient `mu0` is dropped
        (see `_compute_A_eq`).
        """
        mu0 = self.reduced_state_df_["mu0"].to_numpy(dtype=float)
        if self.gamma * mu0.max() > _HIGHS_SMALL_MATRIX_VALUE or (
            restrict_y and mu0.min() <= _HIGHS_SMALL_MATRIX_VALUE
        ):
            logging.debug("The LP can't be reduced; solving it in full.")
            return None

        if restrict_y:
            # Ldf rows 2s and 2s+1 are state s's (see _compute_A_eq). Its
            # y-pair constraints tie together the states of the same x_cols+z.
            states = self.ldf_.iloc[0::2]
            group_of_state = (
                states.groupby(self.x_cols + ["z"], sort=False, dropna=False)
                .ngroup()
                .to_numpy()
            )
        else:
            group_of_state = np.arange(self.n_states_)

        # The diagonal entries of A_eq_'s transition rows are
        # 1 - gamma * mu0, so each state's occupancy sums to this.
        solver = ReducedLPSolver(
            group_of_state=group_of_state,
            occupancy=mu0 / (1 - self.gamma * mu0),
        )
        # Every reduced solution is a solution of the LP...
        solver.check_parametrization(self.A_eq_, self.b_eq_)
        # ... and no solution of the LP is lost: its solutions have one degree
        # of freedom per state, less one per y-pair its constraints tie
        # together (two non-empty rows each, one per action), so as many as
        # there are groups.
        row_nnz = np.diff(self.A_eq_.tocsr().indptr)
        n_pairs = np.count_nonzero(row_nnz[self.n_states_ :]) // 2
        assert solver.n_groups == self.n_states_ - n_pairs, (
            f"{solver.n_groups} groups of states, but the LP's constraints tie"
            f" together {n_pairs} pairs of its {self.n_states_} states"
        )
        return solver

    def fit_rewards(self, reward_weights, refit_objectives=True):
        """
        Builds the optimization problems of the MDP for `reward_weights`. The
        MDP must already be fit with `fit_states()`.

        Parameters
        ----------
        reward_weights : dict<str, float>
            Keys are objective identifiers. Values are their respective reward
            weights.
        refit_objectives : bool, default True
            If False, `obj_set`'s objectives must already be fit to this MDP
            (with `obj_set.fit_objectives(self.ldf_)`), and are not refit.

        Sets Attributes
        ---------------
        opt_problems_

        Returns
        -------
        None
        """
        logging.debug("Fitting objectives ...")
        if refit_objectives:
            self.obj_set.fit_objectives(self.ldf_)
            # Its models were built from the previous objective fits.
            if self.lp_solver_ is not None:
                self.lp_solver_.clear_models()
        self.obj_set.build_opt_problems(
            reward_weights=reward_weights,
            ldf=self.ldf_,
            A_eq=self.A_eq_,
            b_eq=self.b_eq_,
        )
        self.opt_problems_ = self.obj_set.opt_problems_

        return None

    def with_reward_weights(self, reward_weights):
        """
        Returns a copy of this MDP with its optimization problems built for
        `reward_weights`, without refitting anything that doesn't depend on
        them.

        This MDP must be fit with `fit_states()`, and its `obj_set`'s objectives
        fit to it with `obj_set.fit_objectives(self.ldf_)`. The copy shares
        this MDP's (read-only) states, constraints and fitted objectives, but
        has its own `reduced_state_df_`, `obj_set` and `opt_problems_`, so that
        solving it leaves this MDP untouched and reusable for other weights.
        It also shares `lp_solver_`, so that solving it warm-starts the solves
        of the copies made after it (which doesn't change their solutions).

        Parameters
        ----------
        reward_weights : dict<str, float>
            Keys are objective identifiers. Values are their respective reward
            weights.

        Returns
        -------
        clf_mdp : ClassificationMDP
        """
        clf_mdp = copy.copy(self)
        clf_mdp.reduced_state_df_ = self.reduced_state_df_.copy()
        clf_mdp.obj_set = copy.copy(self.obj_set)
        clf_mdp.fit_rewards(reward_weights, refit_objectives=False)
        return clf_mdp

    def compute_optimal_policies(self, skip_error_terms=True, method="highs"):
        """
        Computes the optimal policies for the classification MDP.

        Parameters
        ----------
        skip_error_terms : bool, default False
            If true, doesn't try and find all solutions, and solves each
            optimization problem once. When the LP can be reduced (see
            `lp_solver_`), it is then solved in reduced form, and its one
            policy depends only on the reward weights: ties between equally
            optimal solutions are broken the same way every time.
        method : str, default 'highs'
            The HiGHS solver to use. Options are 'highs' (default),
            'highs-ds', 'highs-ipm', as for `scipy.optimize.linprog`.

        Returns
        -------
        opt_pols : list<np.array>
            Optimal policies.
        """
        # Find the best policy/reward of all the opt_problems
        best_policies_best_rewards = []
        for subprob in self.opt_problems_:
            if (
                skip_error_terms
                and self.lp_solver_ is not None
                and subprob.A_eq is self.A_eq_
                and subprob.objective_costs is not None
            ):
                solution = self.lp_solver_.solve(subprob, method=method)
                opt_pols, opt_rew = (
                    ([solution[0]], solution[1])
                    if solution is not None
                    else ([], -1 * np.inf)
                )
                best_policies_best_rewards.append(
                    {"policies": opt_pols, "reward": np.round(opt_rew, decimals=6)}
                )
                continue

            opt_pols, opt_rew = _find_all_solutions_lp(
                n_states=self.n_states_,
                c=subprob.c,
                A_eq=subprob.A_eq,
                b_eq=subprob.b_eq,
                A_ub=subprob.A_ub,
                b_ub=subprob.b_ub,
                skip_error_terms=skip_error_terms,
                method=method,
                A_eq_solve=(self.A_eq_solve_ if subprob.A_eq is self.A_eq_ else None),
            )
            best_policies_best_rewards.append(
                {
                    "policies": list(opt_pols),
                    "reward": np.round(opt_rew, decimals=6),
                }
            )

        opt_pols, opt_rew = _find_best_policies_from_multiple_opt_problems(
            best_policies_best_rewards,
        )

        # Append optimal policy actions to reduced_state_df_ attribute
        for i, pi in enumerate(opt_pols):
            self.reduced_state_df_[f"pi_{i}"] = pi

        return opt_pols

    def _compute_A_eq(self, mu0, ldf, x_cols, restrict_y):
        """
        Constructs two sets of linear equation constraints.

        1st set: transition matrix
        --------------------------
        Constructs a set of linear equations representing the state-action
        transition probabilities  where all "states" (classification dataset
        samples) have the initial state probability, regardless of the action
        taken.

        2nd set: require equal actions for same `y`
        -------------------------------------------
        # Construct constraints that require equal actions for the same `y`
        # value. This constraint helps the optimizer learn policies that are
        # robust to incorrect predictions of `y`. This was causing problems
        # when trying to optimize for fairness constraints like EqOpp that
        # require knowledge of the `y` value.

        Parameters
        ----------
        mu0 : 1-D array
            Initial state probabilities.
        ldf : pd.DataFrame
            "Lambda dataframe". One row for each state and action combination.
        x_cols : list<str>
            The columns that are used in the state (along with `z` and `y`).
        restrict_y : bool
            If True, policy must have same action for any x,z combo, regardless
            of y.

        Returns
        -------
        A_eq_ : scipy.sparse.csc_array
            Exactly the matrix `_to_solver_matrix()` makes of the dense
            constraint matrix: entries with magnitude <= HiGHS'
            `small_matrix_value` (e.g. every `-gamma * mu0` term when gamma is
            ~1e-9) are omitted, since HiGHS drops them anyway.
        """
        # The matrix is built directly as the sparse matrix HiGHS solves with
        # (see `_to_solver_matrix`), from its (row, col, value) entries. The
        # dense matrix has ~2*n_states^2 entries (tens of GB for the larger
        # datasets), almost all of which HiGHS would drop anyway.
        #
        # Construct constraints that correspond to transition matrix.
        # A_eq[s, sp*n_actions + a] = (1 if s == sp else 0) - gamma * mu0[sp]
        # Every row has the same "-gamma*mu0" pattern (repeated across
        # actions), plus +1 on its own diagonal block.
        n_states = len(mu0)
        n_actions = 2
        n_cols = n_states * n_actions
        mu0_arr = np.asarray(mu0, dtype=float)

        base_row = -self.gamma * np.repeat(mu0_arr, n_actions)
        state_of_col = np.repeat(np.arange(n_states), n_actions)
        # Diagonal-block entries, one per column.
        rows = [state_of_col]
        cols = [np.arange(n_cols)]
        vals = [base_row + 1]
        # Off-diagonal "-gamma*mu0" entries, only for the columns where they
        # exceed HiGHS' small_matrix_value (none when gamma is ~1e-9).
        for j in np.flatnonzero(np.abs(base_row) > _HIGHS_SMALL_MATRIX_VALUE):
            r = np.delete(np.arange(n_states), state_of_col[j])
            rows.append(r)
            cols.append(np.full(len(r), j))
            vals.append(np.full(len(r), base_row[j]))
        n_rows = n_states

        if restrict_y:
            # Construct constraints that require equal actions for the same `y`
            # value. This constraint helps the optimizer learn policies that are
            # robust to incorrect predictions of `y`. This was causing problems
            # when trying to optimize for fairness constraints like EqOpp that
            # require knowledge of the `y` value.

            # For each, x, a combination:
            #   Add constraint that x,y0,a == x,y1,a
            #
            # `ldf` is sorted by x_cols+z+y+mu0 (see
            # _generate_lambda_linear_equations), so within a group of
            # matching x_cols+z+yhat, the y=0 row always appears before the
            # y=1 row. Each row already carries its own `mu0` value, so no
            # dataframe-wide lookup is needed (unlike the previous
            # implementation, which re-searched the whole frame per group).
            # The assertions below enforce the structural invariants this
            # relies on instead of the (much slower) per-group verification
            # the old lookup-based implementation performed implicitly.
            required_cols = set(x_cols) | {"z", "y", "yhat", "mu0"}
            missing_cols = required_cols - set(ldf.columns)
            assert not missing_cols, f"ldf is missing required columns: {missing_cols}"
            assert set(ldf["yhat"].unique()) <= {0, 1}, (
                "expected exactly 2 actions ('yhat' in {0, 1}); n_actions is "
                "hardcoded to 2 in this function"
            )
            assert len(ldf) == n_states * n_actions, (
                "expected ldf to have exactly n_actions rows per state "
                f"({n_states} states * {n_actions} actions), got {len(ldf)} rows"
            )

            # A_eq2's columns (ldf row positions) must refer to the same
            # (state, action) pairs, in the same order, as A_eq's columns
            # (ordered by mu0). This holds iff every consecutive block of
            # n_actions rows in `ldf` corresponds to the same state, in the
            # same order as `mu0` -- verify directly rather than relying on
            # it silently.
            ldf_mu0_arr = ldf["mu0"].to_numpy()
            assert all(
                np.allclose(ldf_mu0_arr[a::n_actions], mu0_arr)
                for a in range(n_actions)
            ), "mu0 and ldf are not aligned to the same state ordering"

            group_cols = x_cols + ["z", "yhat"]
            group_indices = ldf.groupby(group_cols, sort=False).indices
            n_constr = len(group_indices)
            mu0_vals = ldf["mu0"].to_numpy()
            y_vals = ldf["y"].to_numpy()

            locy0s = np.empty(n_constr, dtype=np.intp)
            locy1s = np.empty(n_constr, dtype=np.intp)
            row_i = 0
            for idx_arr in group_indices.values():
                assert len(idx_arr) <= 2
                if len(idx_arr) == 1:
                    continue

                locy0, locy1 = idx_arr[0], idx_arr[1]
                assert y_vals[locy0] == 0 and y_vals[locy1] == 1, (
                    "expected the y=0 row to precede the y=1 row within each "
                    "x_cols+z+yhat group (relies on ldf's sort order)"
                )
                locy0s[row_i] = locy0
                locy1s[row_i] = locy1
                row_i += 1

            # Each constraint row: mu0[y1] * x[y0] - mu0[y0] * x[y1] == 0
            locy0s, locy1s = locy0s[:row_i], locy1s[:row_i]
            constr_rows = n_states + np.arange(row_i)
            rows += [constr_rows, constr_rows]
            cols += [locy0s, locy1s]
            vals += [mu0_vals[locy1s], -mu0_vals[locy0s]]
            # Groups with a single row keep their (all-zero) constraint row.
            n_rows += n_constr

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        keep = np.abs(vals) > _HIGHS_SMALL_MATRIX_VALUE
        A_eq = csc_array((vals[keep], (rows[keep], cols[keep])), shape=(n_rows, n_cols))
        # Canonical (sorted) row indices within each column, as
        # `_to_solver_matrix` produces.
        A_eq.sort_indices()
        return A_eq

    def _generate_lambda_linear_equations(self, reduced_state_df):
        """
        TODO
        """
        state_df = reduced_state_df.copy()
        # Set every two rows the same. One for each action.
        ldf = pd.concat([state_df, state_df], axis=0).reset_index(drop=True)
        ldf = ldf.sort_values(list(ldf.columns))
        yhat = np.zeros(len(ldf), dtype=int)
        yhat[1::2] = 1  # Makes 'a' 0, 1 repeating sequence
        ldf["yhat"] = yhat
        ldf = ldf.reset_index(drop=True)
        return ldf


def _find_best_policies_from_multiple_opt_problems(best_policies_best_rewards):
    """
    Parameters
    ----------
    best_policies_best_rewards : tuple<dict>
        Example:
        ```
        best_policies_best_rewards =
            {
                'policies': list(opt_pols_dem_par_split1_adult),
                'reward': np.round(opt_rew_dem_par_split1_adult, decimals=6)
            },
            {
                'policies': list(opt_pols_dem_par_split2_adult),
                'reward': np.round(opt_rew_dem_par_split2_adult, decimals=6)
            },
        )
        ```

    Returns
    -------
    best_of_best_pols : list<numpy.array>
        The unique list of optimal policies from all opt_problems.
    best_of_best_reward : float
        The best reward of all opt_problems.
    """
    rewards = [bpbr["reward"] for bpbr in best_policies_best_rewards]
    best_idx = np.argwhere(rewards == np.amax(rewards)).flatten().tolist()
    logging.debug(f"best_idx: {best_idx}")
    best_of_best_pols = []
    # For each opt_problem index where the reward is the best reward
    for idx in best_idx:
        # Get all the policies from that opt_problem (all have same reward)
        pols = best_policies_best_rewards[idx]["policies"]
        # For each of these policies, check add it to the list of
        # best_of_best_pols if it's not already in it.
        for pol in pols:
            logging.debug(f"pol {pol}")
            pol_in_best = False
            for bpol in best_of_best_pols:
                logging.debug("\tbpol {bpol}")
                if np.allclose(pol, bpol, atol=1e-5):
                    logging.debug("\t\t" + f"{pol} already in best_of_best_pols")
                    pol_in_best = True
                    break
            if not pol_in_best:
                logging.debug(f"appending {pol} to best_of_best_pols")
                best_of_best_pols.append(pol)
    best_of_best_reward = rewards[best_idx[0]]
    return best_of_best_pols, best_of_best_reward


def _round_occupancy(x, n_states, n_actions=2):
    """
    Rounds a state-action occupancy measure `x` to a deterministic policy.

    Each state takes its argmax action (the first one, on ties), and all of the
    state's occupancy mass is moved onto that action.

    Returns
    -------
    pi_opt : numpy.array<int>, len(n_states)
        The action of each state.
    x_det : numpy.array<float>, len(x)
        The occupancy measure of the rounded policy.
    """
    state_slices = x[: n_states * n_actions].reshape(n_states, n_actions)
    pi_opt = state_slices.argmax(axis=1).astype(int)
    x_det = np.zeros_like(x)
    x_det[np.arange(n_states) * n_actions + pi_opt] = state_slices.sum(axis=1)
    return pi_opt, x_det


def _round_and_verify_policy(
    res, n_states, c, A_eq, b_eq, A_ub, b_ub, feasibility_atol=1e-6
):
    """
    Rounds an LP solution to a deterministic policy (argmax action per
    state) and verifies that the rounded policy is actually valid, instead
    of trusting the rounding blindly.

    `res.x` is a state-action occupancy measure, which is only guaranteed
    to decompose into a clean "one action gets all the mass" pattern per
    state when the LP has a unique, non-degenerate optimum. When the
    optimal face is degenerate -- which happens readily once A_ub encodes
    constraints that couple multiple states together, e.g. Equalized Odds'
    parity constraints -- the solver can return a solution that splits a
    state's mass fractionally across both actions. Naively taking argmax in
    that case silently produces a *different* policy than the one that was
    actually optimized, which can be infeasible. This reconstructs the
    exact occupancy vector implied by the rounded policy and checks it
    against the original constraints, rejecting the rounding when it isn't
    actually feasible.

    Note that this only checks feasibility, not whether the rounded policy
    matches the (possibly fractional) reward the LP solve reported. For
    constraints like exact Equalized Odds parity combined with another
    objective, the true fractional optimum can require a genuinely
    randomized decision rule that no deterministic policy can replicate
    (see Hardt et al. 2016) -- an unavoidable integrality gap, not a
    rounding artifact. Callers should take the best-reward feasible
    candidate found across many solves, rather than expect any single
    candidate to match the LP's reported objective value.

    Returns
    -------
    (pi_opt, achieved_reward) if the rounded policy is feasible, else None.
    """
    if res.x is None:
        # Infeasible, unbounded, or otherwise failed solve.
        return None

    pi_opt, x_det = _round_occupancy(res.x, n_states)

    if not np.allclose(A_eq @ x_det, b_eq, atol=feasibility_atol):
        return None

    if A_ub is not None and len(A_ub) > 0:
        if np.any(A_ub @ x_det > np.asarray(b_ub) + feasibility_atol):
            return None

    achieved_reward = -1 * (np.asarray(c) @ x_det)
    return pi_opt, achieved_reward


# HiGHS' default `small_matrix_value`: constraint matrix entries whose absolute
# value is <= this are discarded by the solver before it solves.
_HIGHS_SMALL_MATRIX_VALUE = 1e-9


def _to_solver_matrix(A):
    """
    Converts a dense constraint matrix into the sparse matrix HiGHS actually
    solves with.

    In the dense constraint matrix, every transition row carries a
    `-gamma * mu0` term for every state-action, but with `gamma` ~1e-9 those
    terms are far below HiGHS' `small_matrix_value`, so HiGHS drops all of
    them anyway. Passing the dense matrix makes scipy densely stack, scan and
    convert millions of entries on every solve, which dominates the solve
    time. Dropping those same entries up front and handing HiGHS a sparse
    matrix produces the exact same model, and so the exact same solution, at
    a fraction of the cost.

    `ClassificationMDP.A_eq_` is built directly in this form, so it is
    returned unchanged (as is any sparse matrix).
    """
    if A is None or issparse(A):
        return A
    A = np.atleast_2d(np.asarray(A, dtype=float))
    rows, cols = np.nonzero(np.abs(A) > _HIGHS_SMALL_MATRIX_VALUE)
    return csc_array((A[rows, cols], (rows, cols)), shape=A.shape)


def _solve_lp(c, A_eq, b_eq, A_ub, b_ub, method="highs"):
    if A_ub is None or A_ub.shape[0] == 0:
        assert b_ub is None or len(b_ub) == 0
        return linprog(c, A_eq=A_eq, b_eq=b_eq, method=method)
    return linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, method=method)


def _find_all_solutions_lp(
    n_states,
    c,
    A_eq,
    b_eq,
    A_ub=None,
    b_ub=None,
    error_term=1e-12,
    skip_error_terms=True,
    method="highs",
    A_eq_solve=None,
    A_ub_solve=None,
):
    """
    Wrapper around scipy.optimize.linprog that finds ALL optimal solutions
    by iteratively solving the LP problem after adding/subtracting an "error"
    term to each objective component.

    Parameters
    ----------
    n_states = int
        Number of states.
    c : 1-D array
        The coefficients of the linear objective function to be minimized.
    A_eq : 2-D array
        The equality constraint matrix. Each row of ``A_eq`` specifies the
        coefficients of a linear equality constraint on ``x``.
    b_eq : 1-D array
        The equality constraint vector. Each element of ``A_eq @ x`` must equal
        the corresponding element of ``b_eq``.
    A_ub : 2-D array, optional
        The inequality constraint matrix. Each row of ``A_ub`` specifies the
        coefficients of a linear inequality constraint on ``x``.
    b_ub : 1-D array, optional
        The inequality constraint vector. Each element represents an
        upper bound on the corresponding value of ``A_ub @ x``.
    error_term : float, default 1e-12
        Allowed error from the optimal reward to still be considered optimal.
    skip_error_terms : bool, default False
        If true, doesn't try and find all solutions and instead just invokes
        the scipy solver on the input terms.
    method : str, default 'highs'
        The scipy solver method to use. Options are 'highs' (default),
        'highs-ds', 'highs-ipm'.
    A_eq_solve, A_ub_solve : scipy.sparse array, optional
        `A_eq` and `A_ub` as passed to the solver (see `_to_solver_matrix`).
        Computed from `A_eq` and `A_ub` when not given; callers solving many
        problems that share a constraint matrix can pass it in to convert it
        only once.

    Returns
    -------
    best_policies : list<numpy.array>
        List of the best-reward *feasible* policies found (see
        `_round_and_verify_policy`). Can be empty if no candidate solve
        produced a feasible deterministic policy at all (as opposed to one
        that merely falls short of the LP relaxation's bound, which is
        common and expected -- see `best_reward`).
    best_reward : float
        The best reward actually achieved by a feasible policy in
        `best_policies`. This is a lower bound on, and will often be
        strictly less than, the fractional LP relaxation's reward: exact
        parity constraints like Equalized Odds can have a real integrality
        gap, where no deterministic policy reaches what a randomized one
        could.
    """
    if A_eq_solve is None:
        A_eq_solve = _to_solver_matrix(A_eq)
    if A_ub_solve is None:
        A_ub_solve = _to_solver_matrix(A_ub)

    candidates = {}  # pi_opt (as tuple) -> achieved_reward

    def _consider(res):
        # Verified against the equality constraints the solver was given,
        # whose matrix-vector product is far cheaper than the dense `A_eq`'s.
        # It only omits entries with magnitude <= 1e-9 (see
        # `_to_solver_matrix`), which shifts `A_eq @ x` by at most ~1e-9 --
        # negligible next to the verification's 1e-6 tolerance.
        result = _round_and_verify_policy(
            res, n_states, c, A_eq_solve, b_eq, A_ub, b_ub
        )
        if result is None:
            return
        pi_opt, achieved_reward = result
        key = tuple(pi_opt.tolist())
        if key not in candidates:
            candidates[key] = achieved_reward
            logging.debug(f"Optimal Policy:\t, {pi_opt}, reward {achieved_reward} \n")

    # Always try the unperturbed problem first. Its objective value is also
    # the fractional LP relaxation's bound, used below to judge whether the
    # cheap search left room for improvement.
    res_unperturbed = _solve_lp(c, A_eq_solve, b_eq, A_ub_solve, b_ub, method=method)
    _consider(res_unperturbed)
    fractional_bound = (
        -1 * res_unperturbed.fun if res_unperturbed.x is not None else None
    )

    if not skip_error_terms:
        # Perturbing each objective coordinate in turn nudges the solver
        # towards different vertices of a degenerate optimal face, in the
        # hope of finding ones that round to valid deterministic policies.
        for i in range(len(c)):
            cpos = np.array(c)
            cpos[i] += error_term
            _consider(
                _solve_lp(cpos, A_eq_solve, b_eq, A_ub_solve, b_ub, method=method)
            )

            cneg = np.array(c)
            cneg[i] -= error_term
            _consider(
                _solve_lp(cneg, A_eq_solve, b_eq, A_ub_solve, b_ub, method=method)
            )

        best_found = max(candidates.values()) if candidates else -1 * np.inf
        # Perturbing one coordinate at a time only explores axis-aligned
        # directions off the current vertex, which is often not enough to
        # reach a different vertex of a highly degenerate optimal face
        # (e.g. Equalized Odds' parity constraints, which tie together the
        # occupancy of several states at once) -- it can consistently land
        # on the *same* feasible-but-suboptimal vertex no matter which
        # single coordinate is nudged. Only pay for the more expensive,
        # broader random search when the cheap search actually left
        # measurable room for improvement versus the fractional LP bound
        # (or found nothing at all); most well-behaved objectives won't
        # need it.
        if fractional_bound is None or not np.isclose(
            best_found, fractional_bound, atol=0.001
        ):
            # Random perturbations across all coordinates simultaneously
            # explore the degenerate face far more broadly than axis-aligned
            # ones, but hit rates against a specific tied vertex can be low
            # (empirically ~1% for a single fixed scale on some problems),
            # so this cycles through several perturbation magnitudes and
            # uses a generous trial budget. Run until the fractional bound
            # is matched (nothing more to gain) or the budget is exhausted
            # -- there may be a real integrality gap between what's
            # feasible and the fractional LP bound, so more search directly
            # improves the result when one doesn't exist.
            rng = np.random.default_rng(0)
            n_random_trials = 200
            random_scales = [1e-6, 1e-4, 1e-2]
            for trial in range(n_random_trials):
                random_scale = random_scales[trial % len(random_scales)]
                c_rand = np.asarray(c, dtype=float) + rng.normal(
                    scale=random_scale, size=len(c)
                )
                _consider(
                    _solve_lp(c_rand, A_eq_solve, b_eq, A_ub_solve, b_ub, method=method)
                )
                if candidates and fractional_bound is not None:
                    best_found = max(candidates.values())
                    if np.isclose(best_found, fractional_bound, atol=0.001):
                        break

    if not candidates:
        # No solve produced a *feasible* rounding within the search budget.
        # The codebase assumes that at least one policy will always be
        # returned, so fall back to the plain argmax rounding of the
        # unperturbed solution, unverified. This can only be reached when
        # every verified search attempt failed, so it's a rare last resort:
        # better to hand back a possibly-imperfect policy than to crash
        # every caller that indexes into an empty list.
        if res_unperturbed.x is not None:
            pi_opt, x_det = _round_occupancy(res_unperturbed.x, n_states)
            fallback_reward = -1 * (np.asarray(c) @ x_det)
            # logging.warning(
            #     "_find_all_solutions_lp found no verified-feasible "
            #     "deterministic policy; falling back to an unverified "
            #     "rounding of the LP solution (reward %s, LP bound %s).",
            #     fallback_reward,
            #     fractional_bound,
            # )
            return [pi_opt], fallback_reward

        logging.warning(
            "_find_all_solutions_lp: the LP itself is infeasible even "
            "without perturbation; no policy can be returned for this "
            "sub-problem."
        )
        return [], -1 * np.inf

    best_reward = max(candidates.values())
    best_policies = [
        np.array(pi)
        for pi, reward in candidates.items()
        if np.isclose(reward, best_reward, atol=0.001)
    ]

    logging.debug(f"\nBest Reward:\t {best_reward}")
    logging.debug("\nOptimal policies:")
    for pi in best_policies:
        logging.debug(f"\t{np.round(pi, 2)}")

    return best_policies, best_reward
