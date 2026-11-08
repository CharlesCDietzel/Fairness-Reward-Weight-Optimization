"""
Solves a `ClassificationMDP`'s linear programs in a reduced form, warm-started
from the previous solve, with ties between optimal solutions broken the same
way every time.

The reduced form
----------------
The LP's variable is the state-action occupancy `x`, with one entry per state
`s` and action `a` (column `2*s + a`). With a discount factor small enough that
HiGHS drops every off-diagonal `-gamma * mu0` term of `A_eq` (always, for the
gamma of ~1e-9 used), the equality constraints say only that:

  - each state's occupancy sums to `m_s = mu0_s / (1 - gamma * mu0_s)`, and
  - (with `restrict_y`) the y=0 and y=1 states of the same `(x, z)` take each
    action with the same probability.

So the feasible `x` are exactly `x = x0 + T @ p` for `p` in `[0, 1]^G`: one
variable per group of states that must act alike (one `(x, z)` with
`restrict_y`, otherwise one state), the probability that the group predicts 1.
Substituting that into the LP leaves `G` box-bounded variables and only the
handful of inequality rows of the absolute-value objectives (one per objective,
two for EqOdds), which is far smaller than the LP over `x` (rows: one per state
plus one per y-pair; columns: two per state).

Warm starts
-----------
Across the reward weights of an `opt_debias` search, only the objective
changes. Each sub-problem therefore keeps one persistent HiGHS model, in which
the objective is `sum_j w_j * v_j` over one auxiliary variable `v_j` per
objective (`v_j` is fixed to objective j's unweighted cost by an equality row).
A new weight vector then changes only the costs of the `v_j`, and HiGHS starts
from the optimal basis of the previous solve.

Tie-breaking
------------
The LP often has many optimal solutions: once a fairness constraint is active,
a whole class of groups is indifferent between the two actions, and which
optimal vertex simplex stops at depends on where it started. With warm starts,
that would make the policy for a weight vector depend on which weights were
solved before it. So every solve is followed by a second one over the set of
optimal solutions, `{p feasible : objective(p) <= z*}`, minimizing a fixed,
generic tie-break objective `t @ p`. Its solution is unique, so the policy
depends only on the reward weights (and the MDP), whatever the starting basis.
"""

import logging

import highspy
import numpy as np
from scipy.sparse import csc_array, csr_array, vstack

# Seeds the tie-break objective. Changing it changes which of a weight vector's
# equally optimal policies is chosen.
_TIE_BREAK_SEED = 0

# The second (tie-break) solve keeps the objective within this tolerance of the
# first solve's optimum z*: the larger of this absolute amount and this
# fraction of |z*|, in the scaled units the model is built in (see
# `_ReducedModel`), where one sample contributes ~1. The absolute part must
# exceed HiGHS' primal feasibility tolerance (1e-7): the first solve's solution
# may violate the constraints by up to that much, and when they leave only a
# sliver of feasible solutions, a tighter optimum row makes the second solve
# infeasible.
_OPTIMUM_ABS_TOL = 1e-6
_OPTIMUM_REL_TOL = 1e-9

# `method` of `ClassificationMDP.compute_optimal_policies()` (scipy linprog's
# names for the HiGHS solvers) -> HiGHS' `solver` option.
_HIGHS_SOLVERS = {"highs": "choose", "highs-ds": "simplex", "highs-ipm": "ipm"}

_INF = highspy.kHighsInf


class ReducedLPSolver:
    """
    Solves the optimization problems of one fitted `ClassificationMDP` in
    reduced form (see the module docstring).

    The persistent models are kept per sub-problem name, so the solver must
    only be given problems built from the objectives it was first given them
    for; `clear_models()` forgets them.

    Parameters
    ----------
    group_of_state : array-like<int>, len(n_states)
        The group of each state, numbered from 0. States of the same group are
        constrained to take the same action.
    occupancy : array-like<float>, len(n_states)
        The total occupancy `m_s` of each state, over both actions.
    """

    def __init__(self, group_of_state, occupancy):
        self.group_of_state = np.asarray(group_of_state, dtype=np.intp)
        self.occupancy = np.asarray(occupancy, dtype=float)
        self.n_states = len(self.occupancy)
        self.n_groups = int(self.group_of_state.max()) + 1
        assert np.array_equal(
            np.unique(self.group_of_state), np.arange(self.n_groups)
        ), "groups must be numbered 0..n_groups-1"
        assert np.all(self.occupancy > 0), "every state must have positive occupancy"

        # x = x0 + T @ p: x0 puts each state's occupancy on action 0, and p_g
        # moves group g's states' occupancy from action 0 to action 1.
        n_cols = 2 * self.n_states
        states = np.arange(self.n_states)
        self.x0 = np.zeros(n_cols)
        self.x0[2 * states] = self.occupancy
        self.T = csc_array(
            (
                np.concatenate([-self.occupancy, self.occupancy]),
                (
                    np.concatenate([2 * states, 2 * states + 1]),
                    np.concatenate([self.group_of_state, self.group_of_state]),
                ),
            ),
            shape=(n_cols, self.n_groups),
        )
        # Every model's costs and constraint rows are scaled by this, so that
        # the smallest state's occupancy is 1 rather than ~1/n_samples, which
        # keeps the costs well above HiGHS' optimality tolerances.
        self.scale = 1.0 / self.occupancy.min()
        self.tie_break = np.random.default_rng(_TIE_BREAK_SEED).standard_normal(
            self.n_groups
        )
        self._models = {}

    def __getstate__(self):
        # HiGHS models can't be copied or pickled; a copy rebuilds its own.
        state = self.__dict__.copy()
        state["_models"] = {}
        return state

    def check_parametrization(self, A_eq, b_eq, atol=1e-10):
        """
        Asserts that `x0 + T @ p` satisfies `A_eq @ x == b_eq` for every `p`,
        i.e. that this reduction is consistent with the MDP's constraints.
        Since the residual is affine in `p`, checking `p = 0` and every
        unit vector at once (via `T`'s columns) suffices.
        """
        b_eq = np.asarray(b_eq, dtype=float)
        residual_x0 = np.abs(A_eq @ self.x0 - b_eq).max()
        residual_T = abs(A_eq @ self.T).max() if self.T.nnz else 0.0
        assert residual_x0 <= atol and residual_T <= atol, (
            "the reduced LP's parametrization doesn't satisfy the MDP's equality"
            f" constraints (residuals {residual_x0:.3g}, {residual_T:.3g})"
        )

    def clear_models(self):
        """Forgets every persistent model (e.g. after refitting objectives)."""
        self._models = {}

    def solve(self, problem, method="highs"):
        """
        Solves `problem` and rounds its solution to a deterministic policy.

        Parameters
        ----------
        problem : frwo.rl.objectives.OptimizationProblem
            A problem of the MDP this solver was built for, with
            `objective_costs` and `objective_weights`.
        method : str, default 'highs'
            'highs', 'highs-ds' or 'highs-ipm', as for `scipy.optimize.linprog`.

        Returns
        -------
        (pi, reward) or None
            The policy (the action of each state) and its reward `-c @ x` for
            its occupancy `x`, or None if the problem is infeasible or the
            solver failed.
        """
        if method not in _HIGHS_SOLVERS:
            raise ValueError(
                f"Unsupported LP method {method!r}; expected one of"
                f" {sorted(_HIGHS_SOLVERS)}."
            )
        model = self._models.get(problem.name)
        if model is None:
            model = _ReducedModel(self, problem)
            self._models[problem.name] = model

        p = model.solve(problem, _HIGHS_SOLVERS[method])
        if p is None:
            return None

        # Each group predicts 1 iff that's the larger part of its occupancy,
        # ties going to 0, as `clf_mdp._round_occupancy()` rounds.
        pi = (p[self.group_of_state] > 0.5).astype(int)
        x_det = np.zeros(2 * self.n_states)
        x_det[2 * np.arange(self.n_states) + pi] = self.occupancy
        reward = -1 * (np.asarray(problem.c, dtype=float) @ x_det)
        return pi, reward


class _ReducedModel:
    """
    The persistent HiGHS model of one sub-problem.

    Columns: `p` (n_groups, in [0, 1]), then `v` (one free variable per
    objective). Rows, all scaled by `solver.scale`:
      - the sub-problem's inequality rows, `E @ p <= b`;
      - `D_j @ p - v_j == 0`, so that `v_j` is objective j's unweighted
        (scaled) cost, up to a constant;
      - the optimum row `sum_j w_j * v_j <= z*`, free except during the
        tie-break solve.
    """

    def __init__(self, solver, problem):
        self.n_groups = solver.n_groups
        self.objective_names = list(problem.objective_costs)
        n_objectives = len(self.objective_names)
        T, x0, scale = solver.T, solver.x0, solver.scale

        rows, lower, upper = [], [], []
        if problem.A_ub is not None and np.asarray(problem.A_ub).size:
            A_ub = np.atleast_2d(np.asarray(problem.A_ub, dtype=float))
            b_ub = np.atleast_1d(np.asarray(problem.b_ub, dtype=float))
            E = (T.T @ A_ub.T).T * scale
            rows.append(csr_array(np.hstack([E, np.zeros((len(E), n_objectives))])))
            lower.append(np.full(len(E), -_INF))
            upper.append((b_ub - A_ub @ x0) * scale)

        D = np.vstack(
            [
                T.T @ np.asarray(problem.objective_costs[name], dtype=float)
                for name in self.objective_names
            ]
        )
        rows.append(csr_array(np.hstack([D * scale, -np.eye(n_objectives)])))
        lower.append(np.zeros(n_objectives))
        upper.append(np.zeros(n_objectives))

        # Its coefficients (the weights) are set by every solve.
        rows.append(
            csr_array(
                np.concatenate([np.zeros(self.n_groups), np.ones(n_objectives)])[None]
            )
        )
        lower.append(np.array([-_INF]))
        upper.append(np.array([_INF]))

        matrix = vstack(rows, format="csr")
        matrix.eliminate_zeros()
        self.optimum_row = matrix.shape[0] - 1

        self.highs = highspy.Highs()
        self.highs.setOptionValue("output_flag", False)
        # Otherwise every run() starts HiGHS' thread pool, which takes ~20x as
        # long as a warm-started solve of these small models (whose simplex
        # is serial anyway).
        self.highs.setOptionValue("threads", 1)
        self.highs.addVars(self.n_groups, np.zeros(self.n_groups), np.ones(self.n_groups))
        self.highs.addVars(
            n_objectives, np.full(n_objectives, -_INF), np.full(n_objectives, _INF)
        )
        self.highs.addRows(
            matrix.shape[0],
            np.concatenate(lower),
            np.concatenate(upper),
            matrix.nnz,
            matrix.indptr[:-1].astype(np.int32),
            matrix.indices.astype(np.int32),
            matrix.data,
        )
        self.all_cols = np.arange(self.n_groups + n_objectives, dtype=np.int32)
        self.tie_break_costs = np.concatenate(
            [solver.tie_break, np.zeros(n_objectives)]
        )

    def solve(self, problem, highs_solver):
        """
        Returns the tie-broken optimal `p` of `problem`'s weights, or None if
        the problem is infeasible or the solver failed. A solve that doesn't
        succeed from the previous solve's basis is retried from scratch, so
        that whatever it returns still depends only on the weights.
        """
        assert list(problem.objective_costs) == self.objective_names
        weights = np.array(
            [problem.objective_weights[name] for name in self.objective_names],
            dtype=float,
        )
        self.highs.setOptionValue("solver", highs_solver)
        status, p = self._solve_canonical(weights)
        if status != "optimal":
            self.highs.clearSolver()
            status, p = self._solve_canonical(weights)
        if status == "infeasible":
            logging.warning(
                f"Reduced LP of sub-problem {problem.name!r} is infeasible; no"
                " policy can be returned for this sub-problem."
            )
            return None
        if status == "failed":
            logging.warning(
                f"HiGHS failed to solve the reduced LP of sub-problem"
                f" {problem.name!r} ({self.highs.getModelStatus()}); no policy"
                " can be returned for this sub-problem."
            )
            return None
        if status == "tie_break_failed":
            # The from-scratch first solve's optimal solution: still optimal
            # and dependent only on the weights, but not tie-broken by the
            # tie-break objective.
            logging.warning(
                f"HiGHS failed to break ties in the reduced LP of sub-problem"
                f" {problem.name!r} ({self.highs.getModelStatus()}); using the"
                " optimal solution it found from scratch instead."
            )
        return p

    def _run(self):
        self.highs.run()
        status = self.highs.getModelStatus()
        if status == highspy.HighsModelStatus.kOptimal:
            return "optimal"
        if status == highspy.HighsModelStatus.kInfeasible:
            return "infeasible"
        return "failed"

    def _solve_canonical(self, weights):
        n_groups, n_objectives = self.n_groups, len(weights)
        highs, row = self.highs, self.optimum_row

        # 1. The optimum z* for these weights.
        for j, weight in enumerate(weights):
            highs.changeCoeff(row, n_groups + j, weight)
        highs.changeRowBounds(row, -_INF, _INF)
        highs.changeColsCost(
            len(self.all_cols),
            self.all_cols,
            np.concatenate([np.zeros(n_groups), weights]),
        )
        status = self._run()
        if status != "optimal":
            return status, None
        optimum = highs.getInfo().objective_function_value
        p_optimal = np.asarray(highs.getSolution().col_value[:n_groups])

        # 2. Among the solutions within tolerance of z*, the one minimizing
        # the tie-break objective. The previous solve's optimal basis is
        # feasible for it, so it too is warm-started.
        tol = max(_OPTIMUM_ABS_TOL, _OPTIMUM_REL_TOL * abs(optimum))
        highs.changeRowBounds(row, -_INF, optimum + tol)
        highs.changeColsCost(len(self.all_cols), self.all_cols, self.tie_break_costs)
        status = self._run()
        if status != "optimal":
            # Infeasible here can only be numerical: step 1 found a solution.
            return "tie_break_failed", p_optimal
        return "optimal", np.asarray(highs.getSolution().col_value[:n_groups])
