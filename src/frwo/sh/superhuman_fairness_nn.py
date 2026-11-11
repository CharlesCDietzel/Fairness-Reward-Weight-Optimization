"""
The neural network variant of the Superhuman Fairness baseline.

This is a port of the `reorg_current` branch of the reference implementation
published alongside "Superhuman Fairness" (Memarrast, Vu, Ziebart; ICML 2023):

    https://github.com/omidMemari/superhumn-fairness/tree/reorg_current

run with its `config_train_nn.json` configuration (`"model_obj": "nn"`), which
replaces the published technique's logistic regression with a PyTorch neural
network trained by gradient descent. Specifically, it ports
`LogisticRegression_pytorch` of that branch's `model.py`, and
`Super_human.base_model()`, `sample_superhuman()`, `get_sample_loss()`,
`compute_alpha()`, `eval_model()` and the `LogisticRegression_pytorch` branch
of `update_model()` in its `optimize.py`. The branch's demonstrations, its
performance/fairness measures and `compute_alpha()` are unchanged from the
published version, so they are shared with `frwo.sh.superhuman_fairness`.

The branch as committed does not run: its latest commit left debugging
`print(...)` / `exit()` calls in `sample_superhuman()` and
`run_demo_baseline()`. This ports the code as it runs without them, i.e. the
branch's `wo_count_ver.py`, which is `optimize.py` from before they were
added. The branch's `w_count_ver.py` variant, which feeds the network running
counts of each group's positive and negative decisions as four extra inputs, is
not ported.

The algorithm itself is unchanged. What differs is only what differs in the
logistic regression port, for the same reasons: the data is handed over in
memory, the project's own preprocessing builds the design matrix, the network
runs on whatever device is available rather than a hard-coded `cuda:1`, and
dedicated random generators replace the global ones. Two loops are also
vectorized without changing any value they produce: the per-row sampling of
`sample_superhuman()`, and the per-demonstration forward passes that build the
training loss, which are one forward pass over the whole pool here. The network
has no dropout or batch statistics, so every row's output, and so every
gradient, is the same either way.

Quirks of the branch, reproduced deliberately and flagged where they occur:

* The base fit applies `CrossEntropyLoss` to the network's softmax output, so
  it treats probabilities as logits.
* One Adam optimizer serves the base fit and every training iteration, so its
  moment estimates carry over from one to the other.
* The branch never uses `lr_theta`: the network's step size is Adam's learning
  rate, 1e-5, which it multiplies by 10 after the first iteration and divides
  by 10 once, the first time the gamma-superhuman sum reaches 3.6 (90% of its
  four features). Here `lr_theta` is that learning rate, defaulting to the
  branch's 1e-5.
* The training loss is `sum_j (sum_k s_jk) * sum_{i in demo j} p1(x_i) / n`,
  where `s_jk` is the subdominance tensor. Its gradient does not involve the
  sampled decisions, only the probabilities of predicting 1.
* The subdominance constant is 0 for the demonstration whose *index* is 0,
  wherever the shuffle puts it, rather than for the first one visited.
* The model returned is that of the iteration with the best gamma-superhuman
  sum, the latest one among ties: upstream pickles the model each time the sum
  matches or beats its best so far, and tests the last pickle.

The training loss has two problems, which `fix_gradient=True` (the
"Superhuman Fairness Neural Network Fixed" technique) fixes; see
`_fixed_loss()`:

1. Its gradient is dominated by a direction shared by every demonstration.
   Each demonstration covers a random half of the pool, so every
   `sum_{i in demo j} p1(x_i)` has nearly the same gradient, and the
   demonstrations' weights `sum_k s_jk` are all positive (the running-mean
   subdominance constant, which is meant to center them, averages a partly
   filled tensor and so subtracts too little). Every step therefore lowers
   every row's probability of predicting 1, and the model drifts toward
   predicting every label 0 -- the same failure as the logistic regression
   version's normalization mismatch.
2. The sampled decisions never reach the gradient: only the probabilities
   `p1(x_i)` do, so the gradient cannot tell which decisions made a sample
   beat or lose to a demonstration.

Not ported, since neither affects a single decision the model makes: the
network's `linear` layer, which `forward()` never uses (its weights are what
upstream's `get_model_theta()` reports), and the COMPAS branch of
`base_model()`, which writes a fair log-loss classifier's coefficients into
that unused layer.
"""

import logging
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch import nn

from frwo.sh.superhuman_fairness import (
    SH_BASE_MODEL_TRAIN_FRAC,
    SH_DEFAULT_ITERS,
    SH_DEFAULT_LAMDA,
    SuperhumanTrainingHistory,
    compute_alphas,
)
from frwo.utils import sklearn_clf_pipeline

# The branch's defaults: `n_nodes` (the "large" network) and `fit()`'s
# `max_iter` in its model.py, the Adam learning rate of its
# `LogisticRegression_pytorch`, and the learning-rate schedule of the network
# branch of its `update_model()`.
SH_NN_DEFAULT_HIDDEN_NODES = 512
SH_NN_DEFAULT_BASE_FIT_EPOCHS = 15000
SH_NN_DEFAULT_LR_THETA = 1e-5
SH_NN_DEFAULT_LR_BOOST_FACTOR = 10
SH_NN_DEFAULT_LR_DECAY_GAMMA_FRAC = 0.9


def _upstream_sum(values):
    """
    `sum(values)` as the reference implementation computed it.

    Upstream sums its gamma-superhuman values with the built-in `sum()` on
    Python 3.8, which adds floats left to right. Python 3.12 made `sum()` of
    floats compensated, which gives e.g. 3.7 instead of 3.6999999999999997
    for 0.9 + 1.0 + 0.9 + 0.9 -- enough to break ties between iterations, and
    to decide the threshold tests, differently than upstream does.
    """
    total = 0
    for value in values:
        total = total + float(value)
    return total


class SuperhumanNet(nn.Module):
    """
    `LogisticRegression_pytorch` of the branch's model.py: two ReLU hidden
    layers of `n_nodes` and `n_nodes / 2` units and a two-class softmax output,
    whose second column is the probability of predicting 1.
    """

    def __init__(self, n_inputs, n_nodes=SH_NN_DEFAULT_HIDDEN_NODES):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, n_nodes)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(int(n_nodes), int(n_nodes / 2))
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(int(n_nodes / 2), 2)
        self.out = nn.Softmax(dim=1)

    def logits(self, x):
        """The pre-softmax output."""
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

    def forward(self, x):
        return self.out(self.logits(x))

    def log_proba(self, x):
        """
        The log of `forward()`, computed stably: the network routinely
        saturates to probabilities that underflow to 0, whose log would be
        `-inf`.
        """
        return torch.log_softmax(self.logits(x), dim=1)


class SuperhumanFairnessNN:
    """
    The Superhuman Fairness classifier, with a neural network model.

    Has the interface of `frwo.sh.superhuman_fairness.SuperhumanFairness`
    -- `fit(X, y, demo_list)`, `predict()`, `alpha_`, `history_`,
    `demo_losses_`, `selected_iteration_` -- so the experiment runs, scores
    and reports either one the same way.

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
    lr_theta : float, default 1e-5
        The Adam learning rate of the base fit and of the first training
        iteration, which the learning-rate schedule scales from there. (The
        branch's own `lr_theta` has no effect; see the module docstring.)
    iters : int, default 30
        Maximum number of training iterations. The original's `iters`.
    lamda : float, default 0.001
        The `alpha` search tolerance. The original's `lamda`.
    rng : numpy.random.Generator, Optional
        Source of randomness for the half of the data the base model is fit
        on, the network's initial weights, decision sampling and the order
        demonstrations are visited in, as for
        `SuperhumanFairness`.
    hidden_nodes : int, default 512
        Width of the first hidden layer; the second is half as wide.
    base_fit_epochs : int, default 15000
        Full-batch Adam steps of the base fit (`fit()`'s `max_iter`).
    lr_boost_factor : float, default 10
        The learning rate is multiplied by this after the first iteration and
        divided by it once the gamma-superhuman sum first reaches
        `lr_decay_gamma_frac` of the number of features.
    lr_decay_gamma_frac : float, default 0.9
        See `lr_boost_factor`. The branch's hard-coded 3.6 is 0.9 of its four
        features.
    device : str, Optional
        The torch device to train on. Defaults to a GPU when torch can see one
        (which includes AMD GPUs under ROCm), and the CPU otherwise.
    fix_gradient : bool, default False
        Train on `_fixed_loss()` instead of upstream's loss. Everything else,
        down to the random draws, is the same.

    Attributes
    ----------
    model_ : SuperhumanNet
    preprocessor_ : sklearn.compose.ColumnTransformer
        The project's preprocessing, fit on the base model's half of the pool.
    threshold_ : float
        The decision threshold, `np.mean(Y_train)`, as in `eval_model()`.
    alpha_ : numpy.ndarray
        The per-feature `alpha` of the selected iteration.
    history_ : SuperhumanTrainingHistory
        Per-iteration diagnostics of the last `fit()`, including every
        iteration's learning rate.
    selected_iteration_ : int
        The index, into `history_`, of the iteration whose model `fit()`
        returns: the best by gamma-superhuman sum, latest among ties.
    """

    def __init__(
        self,
        feature_types,
        feature_names,
        loss_fn,
        lr_theta=SH_NN_DEFAULT_LR_THETA,
        iters=SH_DEFAULT_ITERS,
        lamda=SH_DEFAULT_LAMDA,
        rng=None,
        hidden_nodes=SH_NN_DEFAULT_HIDDEN_NODES,
        base_fit_epochs=SH_NN_DEFAULT_BASE_FIT_EPOCHS,
        lr_boost_factor=SH_NN_DEFAULT_LR_BOOST_FACTOR,
        lr_decay_gamma_frac=SH_NN_DEFAULT_LR_DECAY_GAMMA_FRAC,
        device=None,
        fix_gradient=False,
    ):
        self.feature_types = feature_types
        self.feature_names = list(feature_names)
        self.num_of_features = len(self.feature_names)
        self.loss_fn = loss_fn
        self.lr_theta = lr_theta
        self.iters = iters
        self.lamda = lamda
        self.rng = rng if rng is not None else np.random.default_rng()
        self.hidden_nodes = hidden_nodes
        self.base_fit_epochs = base_fit_epochs
        self.lr_boost_factor = lr_boost_factor
        self.lr_decay_gamma_frac = lr_decay_gamma_frac
        self.fix_gradient = fix_gradient
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model_ = None
        self.optimizer_ = None
        self.preprocessor_ = None
        self.threshold_ = None
        # `Super_human.alpha`, before the first `compute_alpha()`.
        self.alpha_ = np.array([1.0 for _ in range(self.num_of_features)])
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
        self.y_ = np.asarray(y).astype(np.int64)
        self.z_ = np.asarray(X["z"]).astype(np.int64)
        self.demo_list_ = demo_list
        self.num_of_demos_ = len(demo_list)
        self.demo_losses_ = np.array([demo.metric for demo in demo_list], dtype=float)
        # Every (demonstration, row it covers) pair, flattened, for
        # `_fixed_loss()`.
        self.pair_demo_ = np.concatenate(
            [np.full(len(demo.idx), j) for j, demo in enumerate(demo_list)]
        )
        self.pair_row_ = np.concatenate([np.asarray(demo.idx) for demo in demo_list])

        logging.info(f"\t\t SH NN training on {self.device}")
        self._fit_base_model(X, y)

        # The whole pool's design matrix, which every iteration's sampling,
        # training loss and evaluation run the network over. The original
        # re-reads its training CSV instead.
        self.design_ = self._design(X)

        self._update_model()

        return self

    def _design(self, X):
        """`X` as the network's input tensor."""
        return torch.as_tensor(
            np.asarray(self.preprocessor_.transform(X), dtype=np.float32),
            device=self.device,
        )

    def _fit_base_model(self, X, y):
        """
        Build and fit the initial network.

        `Super_human.base_model()` with `model_obj == "nn"`, and
        `LogisticRegression_pytorch.fit()`: `base_fit_epochs` full-batch Adam
        steps of cross-entropy on a stratified half of the training pool.

        Upstream applies `CrossEntropyLoss` to the network's softmax output,
        i.e. treats its probabilities as logits. That is reproduced as-is, as
        is using the same optimizer (and so the same moment estimates)
        afterwards for the training iterations.
        """
        start_time = time.time()

        X_base, _, y_base, _ = train_test_split(
            X,
            y,
            test_size=1 - SH_BASE_MODEL_TRAIN_FRAC,
            # The original hard-codes 12345; drawn from this model's generator
            # instead, so that the half differs between trials and restarts.
            random_state=int(self.rng.integers(0, 2**31 - 1)),
            stratify=y,
        )

        # The project's preprocessing, taken from a throwaway pipeline so that
        # the design matrix is the one every other model here is trained on.
        # It is fit on the same half of the pool as the network, as the
        # logistic regression port's pipeline is.
        pipeline = sklearn_clf_pipeline(
            feature_types=self.feature_types, clf_inst=LogisticRegression()
        )
        self.preprocessor_ = pipeline.named_steps["preprocessor"]
        self.preprocessor_.fit(X_base, y_base)

        X_base_t = self._design(X_base)
        y_base_t = torch.as_tensor(
            np.asarray(y_base).astype(np.int64), device=self.device
        )

        # The initial weights are drawn from torch's global generator, which
        # is seeded from this model's own generator and then restored, so that
        # the rest of the experiment's random draws are unaffected.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(self.rng.integers(0, 2**63 - 1)))
            self.model_ = SuperhumanNet(X_base_t.shape[1], self.hidden_nodes)
        self.model_.to(self.device)
        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.lr_theta)

        criterion = nn.CrossEntropyLoss()
        for _ in range(self.base_fit_epochs):
            self.optimizer_.zero_grad()
            outputs = self.model_(X_base_t)
            # Probabilities passed as logits, as upstream does.
            loss = criterion(outputs, y_base_t)
            loss.backward()
            self.optimizer_.step()

        # `eval_model()` thresholds at the mean of the training labels.
        self.threshold_ = float(np.mean(np.asarray(y)))

        logging.info(
            f"\t\t SH NN base fit: {self.base_fit_epochs} epochs,"
            f" final loss {loss.item():.5f}, {time.time() - start_time:.1f}s"
        )

    def _p1(self, design):
        """The network's probability of predicting 1, for each row."""
        return self.model_(design)[:, 1]

    def _sample_superhuman(self):
        """
        Draw one set of decisions per demonstration from the current network.

        `Super_human.sample_superhuman()` plus `sample_from_prob()`:
        `num_of_demos` independent Bernoulli(p1) draws per row of the pool, as
        in `SuperhumanFairness._sample_superhuman()`.

        Returns
        -------
        sample_matrix : numpy.ndarray, shape (num_of_demos, n_pool)
        """
        with torch.no_grad():
            p1 = self._p1(self.design_).double().cpu().numpy()
        return (self.rng.random((self.num_of_demos_, len(p1))) < p1[None, :]).astype(
            float
        )

    def _get_sample_loss(self, sample_matrix):
        """
        Score each demonstration's samples on every feature.

        `Super_human.get_samples_demo_indexed()` and `get_sample_loss()`:
        demonstration `i`'s samples, restricted to its rows, scored against
        the training pool's labels.

        Returns
        -------
        sample_loss : numpy.ndarray, shape (num_of_demos, num_of_features)
        """
        sample_loss = np.zeros((self.num_of_demos_, self.num_of_features))
        for demo_index, demo in enumerate(self.demo_list_):
            sample_loss[demo_index, :] = self.loss_fn(
                self.y_[demo.idx],
                sample_matrix[demo_index, :][demo.idx],
                self.z_[demo.idx],
            )
        return sample_loss

    def _train_step(self, sample_loss, sample_matrix):
        """
        One Adam step on the subdominance-weighted training loss.

        The network branch of `Super_human.update_model()`. Visiting the
        demonstrations in a random order, it fills the subdominance tensor
        exactly as `compute_grad_theta()` does, and adds
        `sum_{i in demo j} p1(x_i) * s_jk / num_of_demos` to the loss for each
        of its entries. The subdominance constant is the mean of the partially
        filled tensor, except that it is 0 for the demonstration whose index
        is 0 -- wherever the shuffle puts it, as upstream tests the index.

        Upstream runs the network over each demonstration's rows separately;
        here one forward pass over the pool gives each row's probability, and
        each row's weight is the total of the demonstrations covering it, which
        is the same loss.

        With `fix_gradient`, the step is taken on `_fixed_loss()` instead. The
        subdominance tensor is still filled as above, in the same shuffled
        order, so the value reported and every random draw stay the same.

        Returns
        -------
        subdom_tensor_sum : float
        """
        subdom_tensor = np.zeros((self.num_of_demos_, self.num_of_features))
        for j in self.rng.permutation(self.num_of_demos_):
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

        self.optimizer_.zero_grad()
        if self.fix_gradient:
            loss = self._fixed_loss(sample_loss, sample_matrix)
        else:
            row_weight = np.zeros(self.design_.shape[0])
            for j, demo in enumerate(self.demo_list_):
                np.add.at(
                    row_weight,
                    demo.idx,
                    subdom_tensor[j, :].sum() / self.num_of_demos_,
                )
            loss = torch.sum(
                self._p1(self.design_)
                * torch.as_tensor(row_weight, dtype=torch.float32, device=self.device)
            )
        loss.backward()
        self.optimizer_.step()

        return float(np.sum(subdom_tensor))

    def _fixed_loss(self, sample_loss, sample_matrix):
        """
        The training loss of Superhuman Fairness Neural Network Fixed.

            (1 / n) * sum_j (w_j - mean(w)) * sum_{i in demo j} log P(yhat_ji | x_i)

        where `yhat_ji` is the decision sampled for row `i` in demonstration
        `j`'s sample, and `w_j = sum_k max(alpha_k * (sample_loss_jk -
        demo_loss_jk) + 1, 0)` is that sample's subdominance.

        This fixes both problems of upstream's loss (see the module
        docstring):

        1. Centering the weights on their mean removes the component shared by
           every demonstration exactly: the shared part of the demonstrations'
           gradients is multiplied by `sum_j (w_j - mean(w)) = 0`. The mean
           replaces upstream's running-mean subdominance constant, the
           centering it was meant to provide. Subtracting the mean including
           `w_j` itself is `(n - 1) / n` times subtracting the mean of the
           other demonstrations, a baseline that leaves the gradient an
           unbiased estimate.
        2. The gradient is the score-function (REINFORCE) estimate of the
           gradient of the expected subdominance: it raises the probability of
           the decisions of samples that dominate their demonstrations better
           than average, and lowers that of the others. This is the neural
           network counterpart of the logistic regression version's
           `phi_j - E[phi]` with the normalization fixed, whose `phi_j` is
           built from the sampled decisions in the same way.
        """
        hinge = np.maximum(self.alpha_ * (sample_loss - self.demo_losses_) + 1, 0).sum(
            axis=1
        )
        weight = (hinge - hinge.mean()) / self.num_of_demos_

        sampled = sample_matrix[self.pair_demo_, self.pair_row_].astype(np.int64)
        log_proba = self.model_.log_proba(self.design_)
        log_proba_sampled = log_proba[
            torch.as_tensor(self.pair_row_, device=self.device),
            torch.as_tensor(sampled, device=self.device),
        ]
        return torch.sum(
            log_proba_sampled
            * torch.as_tensor(
                weight[self.pair_demo_], dtype=torch.float32, device=self.device
            )
        )

    def _set_learning_rate(self, lr):
        for g in self.optimizer_.param_groups:
            g["lr"] = lr

    def _get_learning_rate(self):
        return self.optimizer_.param_groups[0]["lr"]

    def _eval_model(self):
        """
        The model's loss on each feature, over the training pool.

        `Super_human.eval_model(mode="train")`.
        """
        return self.loss_fn(self.y_, self._predict_design(self.design_), self.z_)

    def _predict_design(self, design):
        with torch.no_grad():
            scores = self._p1(design)
        return (scores >= self.threshold_).long().cpu().numpy()

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

        The network branch of `Super_human.update_model()`, including its
        learning-rate schedule, its selection of the best iteration's model,
        and its two early stopping tests: every feature being 1-superhuman,
        or the gamma-superhuman sum falling for 3 consecutive iterations after
        the tenth.
        """
        gamma_superhuman_arr = []
        gamma_degrade = 0
        max_gamma = -1
        lr_boosted = False
        lr_decayed = False
        best_state = None
        best_alpha = None

        for i in range(self.iters):
            # find sample loss and store it, we will use it for computing
            # the loss and alpha
            sample_matrix = self._sample_superhuman()
            sample_loss = self._get_sample_loss(sample_matrix)

            lr = self._get_learning_rate()
            subdom_tensor_sum = self._train_step(sample_loss, sample_matrix)

            # find new alpha, from the samples of the model before the step
            new_alpha = compute_alphas(self.demo_losses_, sample_loss, self.lamda)
            self.alpha_ = new_alpha

            # eval model
            model_loss = self._eval_model()
            gamma_superhuman = self._find_gamma_superhuman(model_loss)
            gamma_superhuman_arr.append(gamma_superhuman)
            # Summed as upstream's `sum(gamma_superhuman)` is, so that the
            # comparisons below come out exactly as they do there.
            gamma_sum = _upstream_sum(gamma_superhuman)

            self.history_.subdom_sum.append(subdom_tensor_sum)
            self.history_.feature_loss.append(model_loss)
            self.history_.gamma_superhuman.append(gamma_superhuman)
            self.history_.alphas.append(new_alpha)
            self.history_.learning_rate.append(lr)
            self.history_.n_iterations = i + 1

            logging.info(
                f"\t\t SH NN iter {i + 1}/{self.iters}:"
                f" lr={lr:.2g}, subdom_sum={subdom_tensor_sum:.5f},"
                f" gamma_superhuman={np.round(gamma_superhuman, 3)},"
                f" loss={np.round(model_loss, 4)}"
            )

            if gamma_sum > max_gamma:
                max_gamma = gamma_sum

            # The learning rate is boosted after the first iteration, and
            # brought back down the first time the gamma-superhuman sum
            # reaches `lr_decay_gamma_frac` of the number of features (both
            # can happen in the same iteration).
            if not lr_boosted:
                self._set_learning_rate(
                    self._get_learning_rate() * self.lr_boost_factor
                )
                lr_boosted = True
            if (
                gamma_sum >= self.lr_decay_gamma_frac * self.num_of_features
                and not lr_decayed
            ):
                self._set_learning_rate(
                    self._get_learning_rate() / self.lr_boost_factor
                )
                lr_decayed = True

            # Keep the model of the best iteration so far, the latest among
            # ties; upstream pickles it at this point.
            if gamma_sum >= max_gamma:
                best_state = {
                    k: v.detach().clone() for k, v in self.model_.state_dict().items()
                }
                best_alpha = new_alpha.copy()
                self.selected_iteration_ = i

            # every feature is 1-superhuman --> break
            if gamma_sum >= self.num_of_features:
                break
            # look back if it has improved for the last 3 iterations, if not
            # --> break
            if len(gamma_superhuman_arr) > 10 and gamma_sum < _upstream_sum(
                gamma_superhuman_arr[-2]
            ):
                gamma_degrade += 1
            else:
                gamma_degrade = 0
            # peformance degrades for 3 cosecutive iterations.
            if gamma_degrade == 3:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
            self.alpha_ = best_alpha

    ##
    # Prediction
    ##

    def predict(self, X, y=None):
        """
        Predict labels for `X`.

        `Super_human.eval_model()`: the probability of predicting 1,
        thresholded at the mean of the training labels.

        `y` is accepted and ignored, so that this classifier is callable
        exactly like a `ClassificationMDPPolicy` from the evaluation pipeline.
        """
        return self._predict_design(self._design(X))

    def predict_proba(self, X):
        with torch.no_grad():
            return self.model_(self._design(X)).double().cpu().numpy()
