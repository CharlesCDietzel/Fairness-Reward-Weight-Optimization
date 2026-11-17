"""
How the rows of a classification dataset are mapped to the states of a fitted
`ClassificationMDP`, and which action a policy of that MDP takes for a row
whose state the MDP was never fit to: an "unseen" state.

A row is looked up by its "unit": its `x_cols` and `z` values, plus its `y`
unless the MDP was fit with `restrict_y`. With `restrict_y` every state with
the same `x_cols` and `z` takes the same action whatever its `y`, so the row's
`y` (predicted, when it can't be observed) is not needed to find its action,
and including it would only make rows miss a state whose action is known.

A row whose unit the MDP has no state of takes the action of its "unseen
state fallback" (see `UNSEEN_STATE_FALLBACKS`). Both fallbacks predict 1 when
more than half of the training rows of the seen states they draw on are in
states the policy predicts 1 for (i.e. a vote of the policy's own actions,
each state weighted by its training count), and widen what they draw on when
the vote is exactly tied. A row they can't decide takes the policy's default
action.

Everything about a lookup that doesn't depend on the policy is computed once,
in `StateIndex.lookup()`, so that the actions of any number of policies of the
same MDP follow cheaply from it (see `StateLookup.actions()`).
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_array

# The unseen state fallbacks, the values of the UNSEEN_STATE_FALLBACK config
# value:
#   "none"     The policy's default action.
#   "nearest"  The vote of the seen units nearest to the row: those with the
#              same z (and y, without `restrict_y`) that differ from it in the
#              fewest `x_cols`. On a tie, the units one more column away join
#              the vote, up to `NEAREST_TIE_RINGS` times.
#   "backoff"  The vote of the seen states that match the row once its least
#              important `x_cols` are ignored: the columns are dropped one at
#              a time, least important first, until the row's remaining
#              values (always including z, and y without `restrict_y`) match
#              a seen state. A column's importance is its conditional mutual
#              information with y given z, in the MDP's training data. On a
#              tie, the next column is dropped too.
UNSEEN_STATE_FALLBACKS = ("none", "nearest", "backoff")

# How many more rings of units (each one column further from the row than the
# last) the "nearest" fallback adds to a tied vote before giving up on it.
NEAREST_TIE_RINGS = 2

# The most (rows x units) entries a "nearest" lookup compares at once, to
# bound its memory.
_NEAREST_CHUNK_ENTRIES = 4_000_000


def check_unseen_state_fallback(fallback):
    """Raises a ValueError unless `fallback` is one of UNSEEN_STATE_FALLBACKS."""
    if fallback not in UNSEEN_STATE_FALLBACKS:
        raise ValueError(
            f"Unknown unseen state fallback {fallback!r}. Options:"
            f" {list(UNSEEN_STATE_FALLBACKS)}."
        )


class StateIndex:
    """
    The units of a fitted `ClassificationMDP`'s states (see the module
    docstring), and how rows are looked up among them.

    Parameters
    ----------
    x_cols : list<str>
        The MDP's `x_cols`.
    restrict_y : bool
        Whether the MDP was fit with `restrict_y`.
    state_df : pandas.DataFrame
        The MDP's `reduced_state_df_`: one row per state, with its `x_cols`,
        `z` and `y` values.
    state_counts : numpy.array<int>, len(n_states)
        The number of training rows of each state.

    Attributes
    ----------
    key_cols : list<str>
        The columns a row is looked up by: `x_cols`, `z`, and `y` unless
        `restrict_y`.
    unit_of_state : numpy.array<int>, len(n_states)
        The unit of each state.
    unit_states : numpy.array<int>, len(n_units)
        A state of each unit (its first). Every state of a unit takes the same
        action.
    unit_counts : numpy.array<int>, len(n_units)
        The number of training rows of each unit.
    unit_codes : numpy.array<int>, shape (n_units, len(key_cols))
        Each unit's value of each key column, as an index into that column's
        values (see `encode()`).
    """

    def __init__(self, x_cols, restrict_y, state_df, state_counts):
        self.x_cols = list(x_cols)
        self.restrict_y = restrict_y
        self.match_cols = ["z"] if restrict_y else ["z", "y"]
        self.key_cols = self.x_cols + self.match_cols

        # Keys are built column-wise, like the MDP's own state keys, so that
        # they hash and compare equal to the keys of the rows looked up.
        keys = list(zip(*(state_df[col].tolist() for col in self.key_cols)))
        self._unit_lookup = {}
        self.unit_of_state = np.empty(len(keys), dtype=np.int64)
        unit_states = []
        for state, key in enumerate(keys):
            unit = self._unit_lookup.get(key)
            if unit is None:
                unit = len(unit_states)
                self._unit_lookup[key] = unit
                unit_states.append(state)
            self.unit_of_state[state] = unit
        self.unit_states = np.asarray(unit_states, dtype=np.int64)
        self.n_units = len(unit_states)
        self.unit_counts = np.bincount(
            self.unit_of_state,
            weights=np.asarray(state_counts, dtype=np.int64),
            minlength=self.n_units,
        ).astype(np.int64)

        # Each key column's values, in the order the units first take them.
        self._value_codes = []
        for col in self.key_cols:
            codes = {}
            for value in state_df[col].iloc[self.unit_states].tolist():
                codes.setdefault(value, len(codes))
            self._value_codes.append(codes)
        self.unit_codes = np.stack(
            [
                self._encode_column(j, state_df[col].iloc[self.unit_states])
                for j, col in enumerate(self.key_cols)
            ],
            axis=1,
        ).reshape(self.n_units, len(self.key_cols))

        # Kept for the "backoff" fallback's column order, computed on first
        # use.
        self._state_y = np.asarray(state_df["y"].tolist(), dtype=np.int64)
        self._state_counts = np.asarray(state_counts, dtype=np.int64)
        self._backoff_order = None

    def _encode_column(self, j, values):
        codes = self._value_codes[j]
        return np.fromiter(
            (codes.get(value, -1) for value in values.tolist()),
            dtype=np.int64,
            count=len(values),
        )

    def encode(self, df):
        """
        `df`'s value of each key column, as the index of that value among the
        column's values in the units, or -1 for a value no unit has.

        Returns
        -------
        codes : numpy.array<int>, shape (len(df), len(key_cols))
        """
        return np.stack(
            [self._encode_column(j, df[col]) for j, col in enumerate(self.key_cols)],
            axis=1,
        ).reshape(len(df), len(self.key_cols))

    def lookup(self, df, fallback):
        """
        Looks up the state of every row of `df`.

        Parameters
        ----------
        df : pandas.DataFrame
            The rows, with every column of `key_cols`, their values already
            transformed by the MDP's `state_reducer_`.
        fallback : str
            The unseen state fallback (see `UNSEEN_STATE_FALLBACKS`).

        Returns
        -------
        lookup : StateLookup
        """
        check_unseen_state_fallback(fallback)
        keys = zip(*(df[col].tolist() for col in self.key_cols))
        units = np.fromiter(
            (self._unit_lookup.get(key, -1) for key in keys),
            dtype=np.int64,
            count=len(df),
        )
        states = np.where(units >= 0, self.unit_states[np.maximum(units, 0)], -1)
        unseen = np.flatnonzero(units < 0)

        prepared = None
        if len(unseen) and fallback != "none":
            codes = self.encode(df.iloc[unseen])
            if fallback == "nearest":
                prepared = _NearestFallback(self, codes)
            else:
                prepared = _BackoffFallback(self, codes)
        return StateLookup(states=states, index=self, prepared_fallback=prepared)

    def backoff_order(self):
        """
        The order the "backoff" fallback drops the `x_cols` in: by their
        conditional mutual information with y given z in the MDP's training
        data, least first (on a tie, in `x_cols` order).

        Returns
        -------
        order : numpy.array<int>
            Indices into `x_cols`.
        """
        if self._backoff_order is None:
            n_x = len(self.x_cols)
            state_codes = self.unit_codes[self.unit_of_state]
            z = state_codes[:, n_x]
            n_z = int(z.max()) + 1 if len(z) else 1
            n_y = int(self._state_y.max()) + 1 if len(z) else 1
            total = self._state_counts.sum()
            mi = np.zeros(n_x)
            for j in range(n_x):
                x = state_codes[:, j]
                joint = np.zeros((len(self._value_codes[j]), n_y, n_z))
                np.add.at(joint, (x, self._state_y, z), self._state_counts)
                p = joint / total
                p_xz = p.sum(axis=1, keepdims=True)
                p_yz = p.sum(axis=0, keepdims=True)
                p_z = p.sum(axis=(0, 1), keepdims=True)
                # I(x; y | z) = sum p(x,y,z) log(p(z) p(x,y,z) / (p(x,z) p(y,z)))
                nz = p > 0
                numer = (p * p_z)[nz]
                denom = np.broadcast_to(p_xz * p_yz, p.shape)[nz]
                mi[j] = float(np.sum(p[nz] * np.log(numer / denom)))
            # Rounded so that columns with the same information (e.g.
            # duplicates) tie exactly and keep their column order.
            self._backoff_order = np.argsort(np.round(mi, 12), kind="stable")
        return self._backoff_order


@dataclass
class StateLookup:
    """
    The states of some rows of a classification dataset, as found by
    `StateIndex.lookup()`.

    Attributes
    ----------
    states : numpy.array<int>, len(n_rows)
        Each row's state index (a state of its unit), or -1 for a row in an
        unseen state.
    index : StateIndex
        The index the rows were looked up in.
    prepared_fallback : _NearestFallback, _BackoffFallback or None
        The unseen state fallback, prepared for the unseen rows. None for the
        "none" fallback, or when no row is unseen.
    """

    states: np.ndarray
    index: StateIndex
    prepared_fallback: object = None

    @property
    def unseen(self):
        """Whether each row is in an unseen state."""
        return self.states < 0

    def actions(self, pi, default_action):
        """
        The action of each row under the policy `pi`.

        Parameters
        ----------
        pi : array-like<int>, len(n_states)
            The action of each state of the MDP the rows were looked up in.
        default_action : int
            The action of an unseen row the fallback doesn't decide.

        Returns
        -------
        actions : numpy.array<float>, len(n_rows)
        """
        pi = np.asarray(pi)
        actions = np.zeros(len(self.states))
        unseen = self.unseen
        actions[~unseen] = pi[self.states[~unseen]]
        if unseen.any():
            if self.prepared_fallback is None:
                actions[unseen] = default_action
            else:
                actions[unseen] = self.prepared_fallback.actions(
                    pi[self.index.unit_states], default_action
                )
        return actions


def _vote(candidates, out, unresolved, s, t):
    """
    Decides the unresolved rows among `candidates` whose vote isn't tied,
    setting their `out` to 1 where the units predicting 1 have more than half
    of the vote's training rows, and 0 otherwise. `s` is the number of
    training rows predicting 1 and `t` the number of all training rows in
    each row's vote.
    """
    rows = candidates & unresolved & (t > 0) & (2 * s != t)
    out[rows] = 2 * s[rows] > t[rows]
    unresolved &= ~rows


class _NearestFallback:
    """
    The "nearest" unseen state fallback, prepared for some unseen rows: for
    each, the units of each ring (see `UNSEEN_STATE_FALLBACKS`).

    Parameters
    ----------
    index : StateIndex
    codes : numpy.array<int>, shape (n_rows, len(index.key_cols))
        The unseen rows, encoded by `index.encode()`.
    """

    def __init__(self, index, codes):
        n_rows = len(codes)
        n_x = len(index.x_cols)
        unit_codes = index.unit_codes
        n_rings = 1 + NEAREST_TIE_RINGS
        ring_rows = [[] for _ in range(n_rings)]
        ring_units = [[] for _ in range(n_rings)]

        # Only the units that match a row's z (and y) can be its neighbors, so
        # each of their combinations is compared separately.
        match_units = {}
        for unit, key in enumerate(map(tuple, unit_codes[:, n_x:])):
            match_units.setdefault(key, []).append(unit)
        match_rows = {}
        for row, key in enumerate(map(tuple, codes[:, n_x:])):
            match_rows.setdefault(key, []).append(row)

        for key, rows in match_rows.items():
            units = np.asarray(match_units.get(key, []), dtype=np.int64)
            if not len(units):
                continue
            rows = np.asarray(rows, dtype=np.int64)
            x_units = unit_codes[units, :n_x]
            chunk = max(1, _NEAREST_CHUNK_ENTRIES // len(units))
            for start in range(0, len(rows), chunk):
                block = rows[start : start + chunk]
                # How many x_cols each row differs from each unit in. A value
                # no unit has (code -1) differs from every unit's.
                dist = np.zeros((len(block), len(units)), dtype=np.int16)
                for j in range(n_x):
                    dist += codes[block, j][:, None] != x_units[:, j][None, :]
                nearest = dist.min(axis=1)
                for ring in range(n_rings):
                    r, u = np.nonzero(dist == (nearest + ring)[:, None])
                    ring_rows[ring].append(block[r])
                    ring_units[ring].append(units[u])

        self.rings = []
        for ring in range(n_rings):
            r = np.concatenate(ring_rows[ring]) if ring_rows[ring] else np.empty(0, int)
            u = np.concatenate(ring_units[ring]) if ring_units[ring] else np.empty(0, int)
            self.rings.append(
                csr_array(
                    (np.ones(len(r)), (r, u)), shape=(n_rows, index.n_units)
                )
            )
        self.unit_counts = index.unit_counts.astype(float)
        self.n_rows = n_rows

    def actions(self, unit_actions, default_action):
        """The action of each unseen row, for the action of each unit."""
        votes_1 = self.unit_counts * unit_actions
        out = np.full(self.n_rows, float(default_action))
        unresolved = np.ones(self.n_rows, dtype=bool)
        s = np.zeros(self.n_rows)
        t = np.zeros(self.n_rows)
        for ring in self.rings:
            # Each ring adds to the vote of the rings before it.
            s += ring @ votes_1
            t += ring @ self.unit_counts
            _vote(unresolved.copy(), out, unresolved, s, t)
            if not unresolved.any():
                break
        return out


class _BackoffFallback:
    """
    The "backoff" unseen state fallback, prepared for some unseen rows: at
    each level (with one more of the `x_cols` dropped, see
    `StateIndex.backoff_order()`), the cell of each unit, and the cell of each
    row (-1 if no unit is in it).

    Parameters
    ----------
    index : StateIndex
    codes : numpy.array<int>, shape (n_rows, len(index.key_cols))
        The unseen rows, encoded by `index.encode()`.
    """

    def __init__(self, index, codes):
        n_x = len(index.x_cols)
        order = index.backoff_order()
        match = list(range(n_x, len(index.key_cols)))
        n_units = index.n_units
        self.levels = []
        for level in range(1, n_x + 1):
            kept = list(order[level:]) + match
            stacked = np.vstack([index.unit_codes[:, kept], codes[:, kept]])
            _, cells = np.unique(stacked, axis=0, return_inverse=True)
            cells = cells.reshape(-1)
            unit_cells = cells[:n_units]
            row_cells = cells[n_units:]
            # A row is in a seen cell only if a unit is in it too (which a row
            # with a value no unit has, coded -1, never is).
            seen = np.zeros(int(cells.max()) + 1, dtype=bool)
            seen[unit_cells] = True
            row_cells = np.where(seen[row_cells], row_cells, -1)
            self.levels.append((unit_cells, row_cells, len(seen)))
        self.unit_counts = index.unit_counts.astype(float)
        self.n_rows = len(codes)

    def actions(self, unit_actions, default_action):
        """The action of each unseen row, for the action of each unit."""
        votes_1 = self.unit_counts * unit_actions
        out = np.full(self.n_rows, float(default_action))
        unresolved = np.ones(self.n_rows, dtype=bool)
        for unit_cells, row_cells, n_cells in self.levels:
            in_cell = row_cells >= 0
            cells = np.maximum(row_cells, 0)
            s = np.bincount(unit_cells, weights=votes_1, minlength=n_cells)[cells]
            t = np.bincount(unit_cells, weights=self.unit_counts, minlength=n_cells)[
                cells
            ]
            _vote(in_cell, out, unresolved, s, t)
            if not unresolved.any():
                break
        return out
