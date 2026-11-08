# Research

## Description
This repository was created on May 2, 2019, by Jack Blandin, the previous maintainer. As of summer 2025, maintenence and development has been taken over by me, Charles Dietzel. 
This repo consists of various implementations of both existing and novel machine learning, reinforcement learning, and fairness algorithms, as well as supporting experiments for various published and unpublished works. 

## Setup

The following steps show how to setup a uv environment so that all dependencies are correctly installed. If you don't have uv installed, you really should install and start using it for your own projects. It is by far the least crappy python package manager. 

```sh
# Create the uv environment from the pyproject.toml config file
uv sync

# Activate the environment
source .venv/bin/activate 
```

Once you have created and sourced the environment, you will also need to set up and start the wandb server. To do this, first install docker engine following the instructions here: https://docs.docker.com/engine/install/ubuntu/. NOTE: These instructions assume you are using Ubuntu. If not, google for how to install docker engine. If you are using Windows, idk man, figure it out. 

After docker is installed and the docker service is started, run ```wandb server start```. This will pull and automatically start the wandb server inside a docker container. 

Once the command finishes, you will see a terminal prompt asking you to provide an API key. Click the URL in the terminal (which will look something like http://localhost:8080/login) to go to a screen where you will need to make an account. Don't worry, the account will only be created on your local machine and will not exist exist on the internet. 

After you make your account, wandb will then give you your API key, which you should copy and paste into the terminal prompt. After you have done this, congratulations! You have completed the setup. 

IMPORTANT NOTE: You will need to re-run ```wandb server start``` each time you restart your computer if you want to run this code. 

# Reproducing Results

Generating the results from scratch takes three steps, all run from the repository root
with the W&B server running:

1. [Tune](#1-tuning-hyperparameters) the hyperparameters of Fairness Reward
   Weight Optimization and of each baseline with `uv run frwo-tune`.
2. [Review](#2-reviewing-and-keeping-the-tuned-hyperparameters) the best values
   found and keep them.
3. [Run](#3-running-the-final-experiments) the final experiments with
   `uv run frwo`.

In order to reproduce the results from the paper, the tuned hyperparameters are already
set up correctly, so you only need to [run](#3-running-the-final-experiments) the
experiment code. 

After the experiment is run, see [here](#figures) for how to generate the paper's plots.


## The experiment configs

The experiment settings are split across three kinds of file:

* `configs/base.yaml` holds everything the experiment configs share. Its
  comments document each setting. It has four parts:
  * `RANDOM_SEED` and `EXPERT_ALGOS` at the top: the global random seed and
    the experts.
  * `common`: the settings shared by every experiment.
  * `datasets`: one entry per dataset, holding that dataset's own settings.
* `configs/tuned_hyperparameters.yaml` holds every technique's tuned
  hyperparameters, per dataset. `frwo-tune` writes it (see
  [tuning](#1-tuning-hyperparameters)), and every experiment config runs with
  it.
* `configs/experiments/` holds one file per experiment config, such as one
  per experiment in the paper. Each holds only how it differs from the base:
  * `SELECTED_DATASETS` (required): which of the base's datasets to run.
  * `EXPERT_ALGOS`: the experts, if not the base's.
  * `common`: settings that replace the base's for every dataset, e.g.
    `DEMO_BIAS_TYPE_LIST` or `RUN_OPT_DEBIAS`.
  * `experiments`: settings that replace the base's for one dataset, under
    the dataset's name.

  `configs/experiments/main.yaml` is the main experiment config:
  hyperparameters are tuned under its settings, so keep it as close to the
  base as possible.

Each experiment's settings are the base's `common`, then the dataset's entry
under the base's `datasets`, then the dataset's tuned hyperparameters, then the
experiment config's `common`, then its entry for the dataset. Each overrides
the ones before it. The header of `configs/base.yaml` has the details.

To keep every experiment config on the same hyperparameters and data splits,
an experiment config can't set:

* a tuned hyperparameter, or
* `RANDOM_SEED`, `DATA_SPLIT_*` or `N_DATASET_SAMPLES`.

It also can only set settings the base already sets, so a misspelled key is an
error rather than being silently ignored. Every experiment config is checked
before any of them starts running.

`RUN_OPT_DEBIAS` turns Fairness Reward Weight Optimization's `opt_debias`
weight adjustment on or off for an experiment config:

```yaml
common:
  RUN_OPT_DEBIAS: true
```

With it on, Fairness Reward Weight Optimization first learns a policy for its
unadjusted reward weights, then optimizes the weights with the dataset's tuned
`opt_debias` optimizer (its `WEIGHT_ADJUST_LIST` in
`configs/tuned_hyperparameters.yaml`) and learns a policy for those, each in a
W&B run of its own. With it off, only the unadjusted run is performed.

Run every experiment config, the main one first, with either of these
commands:

```sh
uv run frwo
python3 src/frwo/FRWO_Experiments.py  # with .venv activated
```

Use `--config` to run only some of them, by name or by path. It can be given
more than once:

```sh
uv run frwo --config main
uv run frwo --config main --config demo_bias
```

Override single values for every experiment of every config with `--set`.
`--set` reads each value as JSON, so quote lists:

```sh
uv run frwo --set 'SELECTED_DATASETS=["COMPAS"]' --set SH_ITERS=10
```

`--set` only accepts keys the configs already contain. Unlike an experiment
config, it can override tuned hyperparameters, which is handy for a quick test.

`frwo` stops if a technique it would run has no tuned hyperparameters for one
of its datasets. Tune them first, or run with `--allow-untuned` to use the
base's defaults instead. It also warns if a dataset's hyperparameters were
tuned while `main.yaml` (with the base) had different settings for it than it
has now. That's a sign they should be tuned again.

## 1. Tuning hyperparameters

One command does all of the tuning:

```sh
uv run frwo-tune
```

It takes these steps on its own:

1. It creates a W&B sweep for each algorithm on each dataset, so each dataset
   gets its own hyperparameters.
2. It runs the sweeps' trials until every sweep has finished.
3. It writes the best hyperparameters each sweep found to
   `configs/tuned_hyperparameters.yaml.new`, ready for you to
   [review](#2-reviewing-and-keeping-the-tuned-hyperparameters).

It always tunes under the settings of the main experiment config,
`configs/experiments/main.yaml`. Every other experiment config then runs with
the same hyperparameters. By default, it tunes every algorithm in `main.yaml`'s
`ALGORITHMS` on every dataset in its `SELECTED_DATASETS`. Two options narrow
that down, and you can use them together:

```sh
# One algorithm, on every selected dataset
uv run frwo-tune --algorithm "Superhuman Fairness"

# Every selected algorithm, on one dataset
uv run frwo-tune --dataset COMPAS

# One algorithm on one dataset
uv run frwo-tune --algorithm "Superhuman Fairness" --dataset COMPAS
```

The named algorithm or dataset doesn't have to be selected in `main.yaml`.

Each algorithm's search is set up by its W&B sweep config in `configs/sweeps/`.
A sweep tunes only that algorithm's settings. Every other setting comes from
`main.yaml`, merged with `configs/base.yaml`. The current tuned hyperparameters
are left out, so a sweep's results never depend on what was tuned before.

| Technique | Sweep config | Tuned settings |
| --- | --- | --- |
| Fairness Reward Weight Optimization | `fairness_reward_weight_optimization.yaml` | `OPT_DEBIAS_OPTIMIZER`, `OPT_DEBIAS_LAMDA`, `FRWO_CLF_*` |
| Superhuman Fairness | `superhuman_fairness.yaml` | `SH_ITERS`, `SH_LR_THETA`, `SH_LAMDA` |
| Superhuman Fairness Fixed | `superhuman_fairness_fixed.yaml` | `SH_FIXED_ITERS`, `SH_FIXED_LR_THETA`, `SH_FIXED_LAMDA` |
| Superhuman Fairness Neural Network | `superhuman_fairness_nn.yaml` | `SH_NN_*` (all but `SH_NN_DEVICE`) |
| Superhuman Fairness Neural Network Fixed | `superhuman_fairness_nn_fixed.yaml` | `SH_NN_FIXED_*` (all but `SH_NN_FIXED_DEVICE`) |
| Fair LogLoss DP | `fair_logloss_dp.yaml` | `FAIR_LOGLOSS_DP_C`, `FAIR_LOGLOSS_DP_RANDOM_INIT` |
| Fair LogLoss EqOdds | `fair_logloss_eqodds.yaml` | `FAIR_LOGLOSS_EQODDS_C`, `FAIR_LOGLOSS_EQODDS_RANDOM_INIT` |

`Post Proc DP` and `Post Proc EqOdds` have no hyperparameters, so they are
skipped.

Every sweep is a Bayesian search of 40 runs, and every sweep minimizes the
same objective, the technique's mean validation subdominance
(`sum_abs_subdominance_val`). The test split is never used for tuning. Fairness
Reward Weight Optimization is scored on its weight-adjusted runs only, not on
the unadjusted weights it starts from. If any of a trial's runs fail, the
trial can't be the best one, and the search treats it as the worst result so
far, so hyperparameters that make a technique fail on its hard cases are never
chosen.

Tuning is repeatable: running it twice on the same configs gives the same
trials and the same best hyperparameters, however the trials happen to be timed.
To allow that, `frwo-tune` runs W&B's Bayesian search itself, from the
`sweeps` package, instead of leaving it to the W&B server, which picks its
suggestions differently every time. The sweeps still appear in the W&B UI as
usual.

### While it runs

Tuning everything is 28 sweeps of 40 runs each, and can take days. The command
prints each sweep's progress and stays running until the last sweep finishes.

Techniques are tuned one after another. The `parallel_sweeps` value in a
technique's sweep config sets how many trials it runs at once. Every sweep gets
one trial before any gets a second, so a value above the number of datasets
gives some sweeps several, and when a sweep finishes its place goes to a sweep
that still has runs left. The values in the sweep configs were measured on a
machine with 16 CPU cores and 15 GB of RAM. Each is the most trials that use no
more than 5 GB between them, which leaves about 4 GB free for other programs, so
measure again on a different machine. More trials than that run a little faster,
but can run the machine out of memory.

`parallel_sweeps` also affects the search, because trials can overlap: each
trial's hyperparameters are chosen without the results of the `parallel_sweeps -
1` trials before it, even if some of them have finished. That's what makes the
results repeatable. It also means that changing `parallel_sweeps` changes the
results.

Everything goes to the W&B project `frwo-tuning`. That keeps the sweeps out
of the main project, because the plotting notebook loads the most recent
session in its project by default, and a sweep trial's runs form a session too.
You can follow the sweeps there in the W&B UI. Each sweep trial is a W&B run of
its own, and the per-technique runs it reports record which trial launched them
in their `SWEEP_ID` and `SWEEP_RUN_ID` config. The trials' output goes to
`.tuning_state/logs/`.

A few things to keep in mind:

* Each sweep runs on a copy of `main.yaml`, merged with the base, made when
  the sweep was created, so editing the config files while tuning is running
  doesn't affect it. When tuning finishes, the best values are written to a
  copy of `configs/tuned_hyperparameters.yaml` as it is then.
* Every trial runs `N_TRIALS` trials of the unbiased dataset and of each bias
  type in `DATASET_BIAS_TYPE_LIST` and `DEMO_BIAS_TYPE_LIST`, so a sweep costs
  as much as that many final runs, times the number of sweep runs.
* Each sweep stops after a number of runs equal to its `run_cap`. To change a
  cap, edit the `run_cap` in the sweep config before tuning.

### Pausing and resuming

To pause tuning, run this from the repository root in another terminal:

```sh
uv run frwo-tune-pause
```

No new trials start, the trials that are running finish, and the command that
was tuning saves its progress and exits. That can take as long as the slowest
running trial. Nothing is lost, and you can then close the terminal or restart
the computer. If you're running more than one `frwo-tune`, this pauses all
of them. To continue, start the W&B server if it isn't running, then run:

```sh
uv run frwo-tune-resume
```

Tuning carries on with the next trial, or the next sweep if the paused one had
finished its trials, and writes `configs/tuned_hyperparameters.yaml.new` when
the last sweep finishes, as usual. It picks up every paused sweep, whichever
`frwo-tune` it came from, and runs them all from that one command.

Pausing and resuming doesn't change the results: you get the same trials and the
same best hyperparameters as a run that was never paused. Paused sweeps keep
their settings too, so changing the config files or a sweep config while
paused has no effect on them.

Some details:

* `uv run frwo-tune-pause --cancel` withdraws a pause that hasn't happened
  yet.
* `uv run frwo-tune-resume --discard` deletes the paused sweeps' progress
  instead of resuming them. Their runs stay in W&B.
* `frwo-tune` won't start a sweep that's paused or that another
  `frwo-tune` is running. Resume it or discard it first.
* A resumed sweep stays resumable until it finishes. If tuning crashes or you
  press Ctrl-C, `uv run frwo-tune-resume` carries on from where it got to.

### Stopping and continuing

Press Ctrl-C to stop tuning right away. That stops the trials that are running.
To continue, run the same command again:

* Sweeps that had finished are not run again, and their results are kept.
* Sweeps that hadn't finished carry on from where they were. The trials that
  were stopped run again with the same hyperparameters, so this doesn't change
  the results either. Their stopped runs stay in W&B, marked as failed.

The same goes if the computer restarts: start the W&B server, then run the same
command again. If you've changed a sweep config, a sweep's `parallel_sweeps`, or
that dataset's experiment settings in the meantime, that sweep starts over with
the new settings.

Closing the terminal also stops tuning, so use something like `tmux` for a long
run. What's in progress is recorded in `.tuning_state/`, and is cleared when
tuning finishes.

If a trial fails, the sweep carries on, and the failure counts as one of its 40
runs. The exception is if the W&B server can't be reached: the trial runs again
until it can. Other failures that have nothing to do with the hyperparameters,
such as the machine running out of memory, still count, and those can change the
results. If a sweep's first 20 trials all fail, for example because
of an error in the code, the command gives up on that sweep and carries on with
the others. It writes the results it has, then exits with an error. The cause will
be in that sweep's log in `.tuning_state/logs/`. Once it's fixed, run the same
command again, and the sweep starts over.

See `src/frwo/sweep.py` for the details of how a trial is run.

## 2. Reviewing and keeping the tuned hyperparameters

`configs/tuned_hyperparameters.yaml.new` is a copy of
`configs/tuned_hyperparameters.yaml` with each finished sweep's entry, under
its dataset and technique, replaced. The other entries are unchanged. Each
entry holds:

* `values`: the best values the sweep found.
* `objective`: their mean validation subdominance.
* `sweep`: the W&B sweep that found them.
* `tuned_against`: a fingerprint of `main.yaml`'s settings for the dataset when
  it was tuned, which is how `frwo` notices when they have changed since.

The sweep of Fairness Reward Weight Optimization searches over
`OPT_DEBIAS_OPTIMIZER`, which swaps the optimizer of the base's
`WEIGHT_ADJUST_LIST` entry and keeps its evaluation budget. Its best optimizer
is written as the dataset's `WEIGHT_ADJUST_LIST`, e.g.
`[[opt_debias, optuna, TPE, 200]]`. The sweep always runs the `opt_debias`
weight adjustment, whatever `main.yaml` sets `RUN_OPT_DEBIAS` to.

Compare the two files, and if you're happy with the new values, replace the old
file with the new one:

```sh
diff configs/tuned_hyperparameters.yaml configs/tuned_hyperparameters.yaml.new
mv configs/tuned_hyperparameters.yaml.new configs/tuned_hyperparameters.yaml
```

Git ignores `configs/tuned_hyperparameters.yaml.new`. It is overwritten the
next time tuning finishes, and is always built from the current
`configs/tuned_hyperparameters.yaml`. So if you tune in stages, for example one
algorithm at a time, keep each result before starting the next.

## 3. Running the final experiments

Set `N_TRIALS` in the base's `common` to the number of trials to average over.
3 is the number used for the paper results. Check that each experiment
config's `SELECTED_DATASETS` lists every dataset to report on, then run:

```sh
uv run frwo
```

That runs every experiment config in `configs/experiments/`, the main one
first. Each config starts from the same random seed, so its results are the
same whether it runs alone (`--config`) or after the others. Each bias type
also draws its random numbers from a seed of its own, so a config's unbiased
runs are identical to `main.yaml`'s whatever demo bias types it lists, and each
bias type's results don't depend on which other bias types are listed. A
dataset bias type can still change which data split is chosen, though.

Every technique whose training is random (FRWO's `opt_debias` weight search,
the Superhuman Fairness baselines and the fair log-loss baselines) is run
`N_RESTARTS` times in each of its runs (3 by default, set in the base's
`common`), and keeps the restart that did best on its own training objective.
Each trial and restart has a seed of its own, derived from `RANDOM_SEED`, so
the trials' results average over the techniques' randomness too. Within a
trial, a seed is the same for every expert, bias type and training
adjustment, so the runs being compared differ only by what is being compared.
How a run's restarts went is in its summary, under `restart/`.

The results go to the W&B project `frwo`, or to the project named by the
`WANDB_PROJECT` environment variable. The script logs a W&B session id when it
starts and again when it finishes. Every config of one run shares that
session, and each run records its config's name as `CONFIG_NAME`. To plot an
experiment config, set `WANDB_CONFIG` in the plotting notebook to its name. The
default is `main`. The notebook then loads the most recent session that ran
that config, or the session set as `WANDB_SESSION`. It saves each config's
LaTeX tables in their own folder, under `experiment_output/frwo/exp_tables/`.

### Pausing and resuming

A full run takes a long time. To pause it, run this from the repository root
in another terminal:

```sh
uv run frwo-pause
```

The experiment finishes the trial it is running, saves its progress and exits.
You can then close the terminal or restart the computer. To pick up where it
left off, run:

```sh
uv run frwo-resume
```

The experiment continues with the next trial, or the next experiment (or
experiment config) if the paused one had finished its trials.

Pausing and resuming doesn't change the results. The experiment saves its
random number generators' states when it pauses and restores them when it
resumes. A paused-and-resumed run's results are identical to an uninterrupted
run's, apart from timestamps and runtimes. The resumed runs also join the same
W&B session, so the plotting notebook sees one set of results. The experiment
uses the configs as they were when the run started, tuned hyperparameters
included, so editing the config files while it is paused has no effect until
the next new run. It also reuses the original `--set` and `--annotate` values.

Some details:

* `uv run frwo-pause --cancel` withdraws a pause request, as long as the
  current trial hasn't finished yet.
* `uv run frwo-resume --discard` deletes the saved experiment instead of
  resuming it.
* Only one experiment can be paused at a time. A run started while another
  experiment is paused prints a warning and can't be paused itself. Neither
  can a second run started while a pausable one is already running.
* A resumed experiment keeps its saved progress until it finishes. If it
  crashes, `uv run frwo-resume` starts again from the same pause point.
  The runs it reported to W&B before crashing aren't removed, so delete them
  in the W&B UI to avoid counting those trials twice.
* `frwo-pause` doesn't pause hyperparameter tuning. Tuning passes
  `--no-pause` to its experiments, because a trial has to run to completion to be
  scored. See [Pausing and resuming](#pausing-and-resuming) under tuning for its
  own commands.
* The saved state lives in `.experiment_state/` at the repository root.

## Techniques

Nine techniques can be trained and evaluated:

* **Fairness Reward Weight Optimization** -- this project's own technique.
* **Superhuman Fairness** -- the ICML 2023 technique of Memarrast et al.
  ([paper](https://proceedings.mlr.press/v202/memarrast23a/memarrast23a.pdf),
  [reference implementation](https://github.com/omidMemari/superhumn-fairness)),
  ported in `src/frwo/sh/superhuman_fairness.py`.
* **Superhuman Fairness Fixed** -- Superhuman Fairness with the normalization
  of its gradient's feature-matching term fixed. Upstream normalizes the
  expected feature vector by the whole training pool's size but each
  demonstration's by the demonstration's size, which adds a component shared by
  every demonstration that dominates the gradient and drifts the model toward
  predicting every label 0. This variant normalizes both by the
  demonstration's size, so it learns from the demonstration-specific signal
  only. It shares every `SH_*` setting except its own `SH_FIXED_ITERS`,
  `SH_FIXED_LR_THETA` and `SH_FIXED_LAMDA`, so that it can be tuned separately,
  and imitates exactly the same demonstrations as Superhuman Fairness does.
* **Superhuman Fairness Neural Network** -- the neural network version of
  Superhuman Fairness from the reference implementation's
  [`reorg_current` branch](https://github.com/omidMemari/superhumn-fairness/tree/reorg_current),
  ported in `src/frwo/sh/superhuman_fairness_nn.py`. It shares the
  `SH_*` settings except its own `SH_NN_*` ones (including `SH_NN_LR_THETA`,
  which here is the network's Adam learning rate), and imitates the same
  demonstrations as Superhuman Fairness. It trains with PyTorch, on a GPU when
  PyTorch can see one and on the CPU otherwise (see `SH_NN_DEVICE`).
* **Superhuman Fairness Neural Network Fixed** -- Superhuman Fairness Neural
  Network with its training loss fixed. Upstream's loss weights each
  demonstration's total probability of predicting 1 by its (positive)
  subdominance, so every step lowers every probability of predicting 1 and the
  model drifts toward predicting every label 0, and its gradient never sees
  the sampled decisions. This variant weights the log-probability of each
  demonstration's sampled decisions by its subdominance minus the mean over
  demonstrations: the score-function estimate of the gradient of the expected
  subdominance, with the shared direction removed. It has its own
  `SH_NN_FIXED_*` counterpart of every `SH_NN_*` setting.
* **Post Proc DP** and **Post Proc EqOdds** -- the post-processing model of
  Hardt et al. (2016), with demographic parity and with equalized odds as the
  fairness constraint.
* **Fair LogLoss DP** and **Fair LogLoss EqOdds** -- the robust fair-log-loss
  model of Rezaei et al. (2020), with the same two constraints.

The last four are the fair-classification baselines the Superhuman Fairness
paper compares itself against, ported from the same repository into
`src/frwo/sh/baselines.py` (with the Rezaei et al. classifiers vendored
verbatim in `src/frwo/sh/fair_logloss.py`).

Which techniques a run covers is the `ALGORITHMS` list in the `common` of
`configs/base.yaml`, which an experiment config can override; list any
combination:

```yaml
ALGORITHMS:
  - Fairness Reward Weight Optimization
  - Superhuman Fairness
  - Superhuman Fairness Fixed
  - Superhuman Fairness Neural Network
  - Superhuman Fairness Neural Network Fixed
  - Post Proc DP
  - Post Proc EqOdds
  - Fair LogLoss DP
  - Fair LogLoss EqOdds
```

Every technique is trained on the same dataset, the same injected label bias
and the same train/validation/test split, and is evaluated by the same code, so
their metrics are directly comparable. Each reports its own W&B run, tagged and
configured with its `ALGORITHM`; the plotting notebook selects one with its
`selected_algorithm` variable.

Each baseline draws from its own deterministically seeded random generator
rather than the global one, so enabling any of them leaves the FRWO results
bit-identical to a FRWO-only run. Like FRWO's, the seed differs between trials
and restarts but not between the experts and bias types a trial compares. A baseline that fails on some split (the
fair-log-loss optimizer can) is logged, its run marked `converged = False` --
which is what the plotting notebook filters on -- and the trial carries on with
the remaining techniques.

The techniques' own parameters live next to `ALGORITHMS` in the same config
file and are documented there: the `SH_*` entries for Superhuman Fairness (how
many demonstrations it imitates, which performance/fairness measures it
optimizes, its learning rate and iteration count) and the `FAIR_LOGLOSS_DP_*` and
`FAIR_LOGLOSS_EQODDS_*` entries for the two fair-log-loss baselines, which are
configured separately.

# Figures

Use VSCode or your IDE of choice to view and run the various python notebooks. 
All figures and plots found in the paper are generated by [this notebook](notebooks/irl/FRWO_Experiments_Plotting.ipynb).

# Publications

Currently, there are no publications that correspond to this code. Watch this space! Or don't, I'm not your dad. 
