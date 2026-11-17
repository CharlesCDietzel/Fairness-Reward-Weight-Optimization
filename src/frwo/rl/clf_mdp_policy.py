import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from frwo.rl.state_lookup import check_unseen_state_fallback


class ClassificationMDPPolicy(BaseEstimator, ClassifierMixin):
    """
    A Scikit-Learn compatible wrapper for a Classification MDP policy.

    Parameters
    ----------
    mdp : ClassificationMDP
        The classification mdp instance.
    pi : numpy.array<int>
        The action of each state of `mdp`. The states that differ only by `y`
        take the same action, so a row's action doesn't depend on its `y`.
    default_action : int
        The action of a row whose state the MDP was never fit to (an "unseen"
        state), when `unseen_state_fallback` doesn't decide it.
    unseen_state_fallback : str, default "none"
        How the action of a row in an unseen state is chosen: "none" (the
        default action) or "nearest". See
        `frwo.rl.state_lookup.UNSEEN_STATE_FALLBACKS`.

    Attributes
    ----------
    reward_weights : list<float>
        Weights for each objective component.
    pi_df_ : pandas.DataFrame
        Policy represented as a dataframe.
    """

    def __init__(self, mdp, pi, default_action=0, unseen_state_fallback="none"):
        check_unseen_state_fallback(unseen_state_fallback)
        self.mdp = mdp
        self.pi = pi
        self.default_action = default_action
        self.unseen_state_fallback = unseen_state_fallback
        self._pi_df = None

    def fit(self, X, y):
        """
        Pass through. Doesn't do anything.

        Parameters
        ----------
        X : pandas.DataFrame
            Classification input.
        y : pandas.Series<int>
            Binary target variable.

        Returns
        -------
        None
        """
        pass

    def predict(self, X):
        """
        The policy's action for each row of `X`: the action of its state,
        found by its state columns and `z` (see `lookup_states()`).

        Parameters
        ---------
        X : pandas.DataFrame
            Input data.

        Returns
        -------
        actions : numpy.array<int>, len(len(X))
            The "predictions", actually the actions from the Clf MDP.
        """
        return self.actions_for_states(self.lookup_states(X))

    def lookup_states(self, X):
        """
        Looks up the MDP state of each row of `X` by its state columns and `z`
        (see `frwo.rl.state_lookup.StateIndex`), and prepares this policy's
        unseen state fallback for the rows in unseen states.

        The states only depend on the MDP, not on the policy `pi`, so they can
        be computed once and reused with `actions_for_states()` for any number
        of policies of the same MDP (with the same `unseen_state_fallback`).

        Parameters
        ---------
        X : pandas.DataFrame
            Input data.

        Returns
        -------
        lookup : frwo.rl.state_lookup.StateLookup
            The state of each row, -1 where the MDP has no state for it.
        """
        index = self.mdp.state_index_
        df = pd.DataFrame(X)[index.key_cols]
        lookup = index.lookup(df, self.unseen_state_fallback)
        n_unseen = int(np.count_nonzero(lookup.unseen))

        if n_unseen and logging.getLogger().isEnabledFor(logging.DEBUG):
            for i in np.flatnonzero(lookup.unseen):
                logging.debug(
                    "\tState Lookup Error: " + str(tuple(df.iloc[i].tolist()))
                )
            log_msg = f"""
                \t\tThere were {n_unseen} state lookup errors when trying
                \t\tto set the optimal action for the input dataset. There are
                \t\t{len(df)} total input rows. So {n_unseen}/{len(df)}
                \t\tuse the '{self.unseen_state_fallback}' unseen state fallback
                \t\t(default action: {self.default_action}).
                """
            logging.debug(log_msg)
        return lookup

    def actions_for_states(self, lookup):
        """
        The policy's action for each row of `lookup` (as returned by
        `lookup_states()`), using the fallback the lookup was prepared with
        for the rows in unseen states.

        Parameters
        ----------
        lookup : frwo.rl.state_lookup.StateLookup
            Rows looked up in this policy's MDP.

        Returns
        -------
        actions : numpy.array<float>, len(len(states))
        """
        assert lookup.index is self.mdp.state_index_, (
            "the rows were looked up in a different MDP's states"
        )
        return lookup.actions(self.pi, self.default_action)

    @property
    def pi_df_(self):
        """
        The policy represented as a dataframe: the self.mdp.ldf_ dataframe
        with the policy actions added as a column.

        Built on first access rather than in __init__, since constructing it
        runs a full prediction that most policies never need.
        """
        if self._pi_df is None:
            self._construct_pi_df()
        return self._pi_df

    def _construct_pi_df(self):
        """
        Adds the policy actions as a column to the self.mdp.ldf_ dataframe.

        Sets pi_df_
        ----------
        pi_df_ : pandas.DataFrame
            Policy represented as a dataframe.
        """
        pi_df = self.mdp.ldf_.iloc[:, :-2].drop_duplicates().copy()
        pi_df["a"] = self.predict(pi_df)
        self._pi_df = pi_df
        return None
