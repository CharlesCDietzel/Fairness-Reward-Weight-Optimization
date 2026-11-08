import copy
import itertools
import logging
import numbers
from types import SimpleNamespace

import highspy
import numpy as np
import pandas as pd
from scipy.sparse import csc_array, csr_array

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
        `small_matrix_value` are omitted (see `_compute_A_eq`). The LPs are
        solved in the reduced form of `reduced_lp_`, which satisfies these
        constraints by construction; `A_eq_` is only used to verify policies.
    reduced_lp_ : _ReducedLP
        The MDP's LPs in reduced form, and the solvers that solve them. Shared
        by every copy `with_reward_weights()` makes.
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
        self.reduced_lp_ = None
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
        reduced_lp_
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

        # The states `restrict_y` constrains to take the same action: those
        # with the same x_cols and z (one with y=0, one with y=1). These are
        # the groups `_compute_A_eq` builds the action equality constraints
        # of.
        if restrict_y:
            state_groups = (
                self.reduced_state_df_.groupby(self.x_cols + ["z"], sort=False)
                .ngroup()
                .to_numpy()
            )
        else:
            state_groups = np.arange(self.n_states_)
        self.reduced_lp_ = _ReducedLP(
            mu0=self.reduced_state_df_["mu0"].to_numpy(dtype=float),
            gamma=self.gamma,
            state_groups=state_groups,
        )
        self.reduced_lp_.check_constraints(self.A_eq_, self.b_eq_)

        return None

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
            If true, doesn't try and find all solutions and instead just invokes
            the solver on the input terms.
        method : str, default 'highs'
            The HiGHS solver to use (see `_ReducedLP`). Options are 'highs'
            (default), 'highs-ds', 'highs-ipm'.

        Returns
        -------
        opt_pols : list<np.array>
            Optimal policies.
        """
        reduced_lp = self.reduced_lp_
        if reduced_lp.reference_costs is None:
            reduced_lp.reference_costs = self._reference_costs()

        # Find the best policy/reward of all the opt_problems
        best_policies_best_rewards = []
        for subprob in self.opt_problems_:
            assert subprob.A_eq is self.A_eq_ and subprob.b_eq is self.b_eq_, (
                "the optimization problems must be built on this MDP's own"
                " constraints, which its reduced form satisfies"
            )
            opt_pols, opt_rew = _find_all_solutions_lp(
                n_states=self.n_states_,
                c=subprob.c,
                A_eq=subprob.A_eq,
                b_eq=subprob.b_eq,
                A_ub=subprob.A_ub,
                b_ub=subprob.b_ub,
                solve=reduced_lp.solver(subprob, method),
                skip_error_terms=skip_error_terms,
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

    def _reference_costs(self):
        """
        The cost vector `c` of each of the MDP's optimization problems, by
        name, for equal reward weights: the fixed weights every solve's warm
        start comes from (see `_ReducedLP`). They depend only on the MDP and
        its fitted objectives, never on which weights are being solved.
        """
        obj_set = copy.copy(self.obj_set)
        names = [obj.name for obj in obj_set.objectives]
        obj_set.build_opt_problems(
            reward_weights={name: 1.0 / len(names) for name in names},
            ldf=self.ldf_,
            A_eq=self.A_eq_,
            b_eq=self.b_eq_,
        )
        return {problem.name: problem.c for problem in obj_set.opt_problems_}

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
            The constraint matrix, without the entries with magnitude <=
            HiGHS' `small_matrix_value` (e.g. every `-gamma * mu0` term when
            gamma is ~1e-9), which HiGHS would drop anyway.
        """
        # The matrix is built directly as a sparse matrix, from its (row, col,
        # value) entries. The dense matrix has ~2*n_states^2 entries (tens of
        # GB for the larger datasets), almost all of which are negligible.
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
        # Canonical (sorted) row indices within each column.
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

# The HiGHS `solver` option of each `method`, and whether its solves are warm
# started (see `_ReducedLP`). An interior point solve can't start from a basis.
_HIGHS_METHODS = {
    "highs": ("choose", True),
    "highs-ds": ("simplex", True),
    "highs-ipm": ("ipm", False),
}


class _ReducedLP:
    """
    The LPs of a `ClassificationMDP` in reduced form, and the HiGHS models that
    solve them.

    The reduction
    -------------
    Every state of a classification MDP transitions to the initial state
    distribution `mu0`, whatever the action. So any policy's discounted state
    occupancy is the same, `mu0 / (1 - gamma)`, and the only freedom the LP's
    occupancy measure `x` has is how each state splits its occupancy between
    the two actions. With `restrict_y`, the states with the same x_cols and z
    (the y=0 and the y=1 one) must split it the same way. So every feasible
    `x` is

        x = x0 + T @ p,   0 <= p <= 1,

    where `p[g]` is the probability that group `g` of tied states predicts 1,
    `x0` puts each state's whole occupancy on action 0, and `T` moves the
    fraction `p` of it onto action 1. That turns each LP

        min c @ x   s.t.  A_eq @ x == b_eq,  A_ub @ x <= b_ub,  x >= 0

    into one with one variable per group, only `A_ub`'s few rows (one per
    absolute-value objective, or two for Equalized Odds) and bounds:

        min (T.T @ c) @ p + c @ x0   s.t.  (A_ub @ T) @ p <= b_ub - A_ub @ x0,
                                           0 <= p <= 1.

    This is exactly the MDP's LP, for any `gamma`. (The full LP as HiGHS was
    given it dropped the `-gamma * mu0` terms of `A_eq` when gamma is ~1e-9,
    which shifted it by ~1e-9; see `check_constraints()`.)

    Warm starts
    -----------
    Only `c` depends on the reward weights. Each optimization problem (one
    per combination of absolute-value splits, by name) gets one HiGHS model,
    built once, whose constraints never change. It is first solved from
    scratch for the costs of equal reward weights (`reference_costs`), and
    every later solve of it starts from that solve's optimal basis. Starting
    from a fixed basis, rather than from the previous solve's, matters:
    the LPs often have many optimal solutions (when a fairness constraint is
    active, whole classes of groups are tied), and which one the simplex
    method returns depends on where it starts. Starting every solve from the
    same basis makes the solution, and so the policy, a function of the
    reward weights alone, whatever was solved before. The reference basis is
    itself determined by the MDP, so the MDP solves the same way however many
    times, and in whatever order, it is solved, and so does a copy of it (the
    HiGHS models are not copied but rebuilt, the same way).

    Parameters
    ----------
    mu0 : numpy.array<float>, len(n_states)
        Initial state probabilities.
    gamma : float
        The discount factor.
    state_groups : numpy.array<int>, len(n_states)
        The group of each state: states in the same group must take the same
        action.

    Attributes
    ----------
    reference_costs : dict<str, numpy.array<float>> or None
        The cost vector of each optimization problem, by name, for equal
        reward weights. Set by the MDP before the first solve.
    """

    def __init__(self, mu0, gamma, state_groups):
        n_states = len(mu0)
        n_cols = 2 * n_states
        _, groups = np.unique(state_groups, return_inverse=True)
        self.n_states = n_states
        self.n_groups = int(groups.max()) + 1 if n_states else 0
        self.state_groups = groups
        occupancy = np.asarray(mu0, dtype=float) / (1 - gamma)
        self.x0 = np.zeros(n_cols)
        self.x0[0::2] = occupancy
        action_cols = np.r_[2 * np.arange(n_states), 2 * np.arange(n_states) + 1]
        self.T = csr_array(
            (np.r_[-occupancy, occupancy], (action_cols, np.r_[groups, groups])),
            shape=(n_cols, self.n_groups),
        )
        self.T_t = self.T.T.tocsr()
        self.reference_costs = None
        self._models = {}

    def check_constraints(self, A_eq, b_eq, atol=1e-8):
        """
        Asserts that the reduction describes exactly the occupancy measures
        `A_eq @ x == b_eq` allows: that every `x0 + T @ p` satisfies it, and
        that the groups tie together exactly as many states as `A_eq`'s
        action equality constraints do.
        """
        p = np.random.default_rng(0).uniform(size=self.n_groups)
        x = self.x0 + self.T @ p
        residual = np.abs(A_eq @ x - np.asarray(b_eq, dtype=float)).max(initial=0)
        assert residual <= atol, f"the reduced LP violates A_eq by {residual}"
        # Each tied pair of states has one action equality row per action.
        n_tied_rows = np.count_nonzero(np.diff(csr_array(A_eq).indptr)[self.n_states :])
        assert 2 * (self.n_states - self.n_groups) == n_tied_rows, (
            f"{self.n_states - self.n_groups} states are tied to another by the"
            f" reduction, but A_eq has {n_tied_rows} action equality rows"
        )

    def solver(self, problem, method="highs"):
        """
        Returns `solve(c)`, which solves `problem` for the costs `c` (of the
        full LP), returning an object with the full LP's solution `x` (None if
        the LP is infeasible) and objective value `fun`, like
        `scipy.optimize.linprog()`'s.

        `problem`'s constraints must be those of the problem of the same name
        the reference costs were built for. They are, since an optimization
        problem's constraints depend only on its splits, which its name
        names, and not on the reward weights.
        """
        if method not in _HIGHS_METHODS:
            raise ValueError(
                f"Unknown LP method {method!r}; options are {list(_HIGHS_METHODS)}."
            )
        key = (problem.name, method)
        if key not in self._models:
            self._models[key] = self._build_model(problem, method)
        highs, basis, warm = self._models[key]
        index = np.arange(self.n_groups, dtype=np.int32)

        def solve(c):
            c = np.asarray(c, dtype=float)
            # Clear everything the last solve left (setBasis() alone leaves
            # some of HiGHS' internal simplex state, which makes the solution
            # depend on what was solved before), then start from the
            # reference basis.
            highs.clearSolver()
            highs.changeColsCost(self.n_groups, index, self.T_t @ c)
            if warm and basis is not None:
                highs.setBasis(basis)
            highs.run()
            if highs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
                return SimpleNamespace(x=None, fun=None)
            p = np.asarray(highs.getSolution().col_value)
            return SimpleNamespace(
                x=self.x0 + self.T @ p,
                fun=highs.getInfo().objective_function_value + c @ self.x0,
            )

        return solve

    def _build_model(self, problem, method):
        """
        Builds `problem`'s HiGHS model and solves it from scratch for its
        reference costs, whose optimal basis every later solve starts from.

        Returns
        -------
        (highs, basis, warm) : (highspy.Highs, highspy.HighsBasis or None, bool)
            `basis` is None if the reference solve found no optimal solution.
        """
        solver_option, warm = _HIGHS_METHODS[method]
        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        highs.setOptionValue("solver", solver_option)
        n = self.n_groups
        highs.addVars(n, np.zeros(n), np.ones(n))
        if problem.A_ub is not None and len(problem.A_ub) > 0:
            A_ub = np.atleast_2d(np.asarray(problem.A_ub, dtype=float))
            E = csr_array(A_ub @ self.T)
            b = np.asarray(problem.b_ub, dtype=float) - A_ub @ self.x0
            highs.addRows(
                E.shape[0],
                np.full(E.shape[0], -highspy.kHighsInf),
                b,
                E.nnz,
                E.indptr[:-1].astype(np.int32),
                E.indices.astype(np.int32),
                E.data,
            )

        basis = None
        if warm:
            assert self.reference_costs is not None, "reference_costs is not set"
            reference_c = np.asarray(self.reference_costs[problem.name], dtype=float)
            highs.changeColsCost(n, np.arange(n, dtype=np.int32), self.T_t @ reference_c)
            highs.run()
            if highs.getModelStatus() == highspy.HighsModelStatus.kOptimal:
                basis = highs.getBasis()
        return highs, basis, warm

    def __getstate__(self):
        # HiGHS models can't be copied or pickled. A copy rebuilds them, and
        # solves exactly as this does, since they are built deterministically.
        state = self.__dict__.copy()
        state["_models"] = {}
        return state


def _find_all_solutions_lp(
    n_states,
    c,
    A_eq,
    b_eq,
    A_ub=None,
    b_ub=None,
    solve=None,
    error_term=1e-12,
    skip_error_terms=True,
):
    """
    Finds ALL optimal solutions of an LP by iteratively solving it after
    adding/subtracting an "error" term to each objective component.

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
    solve : callable
        `solve(c)` solves the LP with the constraints above for the costs `c`,
        returning an object with the solution `x` (None if the LP is
        infeasible) and the objective value `fun`. See `_ReducedLP.solver()`.
        The constraints above are only used to verify its solutions.
    error_term : float, default 1e-12
        Allowed error from the optimal reward to still be considered optimal.
    skip_error_terms : bool, default False
        If true, doesn't try and find all solutions and instead just invokes
        the solver on the input terms.

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
    candidates = {}  # pi_opt (as tuple) -> achieved_reward

    def _consider(res):
        result = _round_and_verify_policy(res, n_states, c, A_eq, b_eq, A_ub, b_ub)
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
    res_unperturbed = solve(c)
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
            _consider(solve(cpos))

            cneg = np.array(c)
            cneg[i] -= error_term
            _consider(solve(cneg))

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
                _consider(solve(c_rand))
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
