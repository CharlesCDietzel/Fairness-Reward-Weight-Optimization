"""
Hyperparameter tuning of Fairness Reward Weight Optimization and its baselines
with W&B sweeps.

Each technique is tuned by its own sweep, configured by one of the files in
configs/sweeps/: a search over that technique's hyperparameters that
minimizes its mean validation subdominance, by a Bayesian search of 40 runs.
One command does all of it:

    uv run frwo-tune [--algorithm NAME] [--dataset NAME]

It tunes under the settings of the main experiment config,
configs/experiments/main.yaml, and every experiment config then runs with what
it finds. For every technique and dataset main.yaml selects (or just the ones
named) it creates a sweep, so that each dataset gets its own hyperparameters,
runs trials until every sweep has finished, and writes
configs/tuned_hyperparameters.yaml.new: a copy of
configs/tuned_hyperparameters.yaml with each sweep's best values set in its
dataset's and technique's entry. The techniques are tuned one after another,
each running as many trials at once as the `parallel_sweeps` of its sweep
config says, a number measured to make that technique's sweeps finish soonest
on their own. What it
has created is recorded in `.tuning_state/`, so that running the same command
again after an interruption continues the same sweeps.

`uv run frwo-tune-pause` lets the running trials finish and then exits, and
`uv run frwo-tune-resume` continues from there.

Tuning is repeatable, so pausing it cannot change its results. A W&B sweep on
its own is not: the server seeds each suggestion from a random request id, and
what a suggestion is based on depends on which trials happen to have finished.
So `frwo-tune` runs W&B's Bayesian search itself, from the `sweeps` package
the server uses, and starts the trials without a `wandb agent`:

* Each suggestion has its own seed, made from `RANDOM_SEED`, the sweep's name
  and the trial's index.
* What a suggestion is based on depends only on the trial's index. With
  `parallel_sweeps` of P, trial k gets the results of the trials before k - P + 1,
  and treats the P - 1 before it as still running, whether or not they are. So
  trial k waits for every trial before k - P + 1 to end, and trials can run in
  any order and at any time without changing what is suggested.

Each trial still is a W&B run of its sweep, so the sweep looks the same in the
W&B UI. The run comes from `python -m frwo.sweep trial`: it runs the
experiment script with the suggested hyperparameters as `--set` overrides, in a
child process, and logs the objective the search minimizes. It runs on the main
experiment config as it was resolved when the sweep was created (see
`frwo.config.load_config`), so that editing the config files cannot change a
sweep partway through. That copy leaves out the tuned hyperparameters, so that
what a sweep finds never depends on what was tuned before.

The child process is needed because the experiment script reports one W&B run
per technique, bias type and trial, while inside a sweep every `wandb.init()` is
forced onto the sweep's own run. The child is started without the sweep's
environment variables, so its runs are ordinary runs -- the same ones the script
reports outside a sweep -- each recording the sweep trial that launched it as
`SWEEP_ID`/`SWEEP_RUN_ID` in its config.
"""

import argparse
import copy
import functools
import hashlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import wandb
import yaml

from frwo.config import (
    MAIN_CONFIG_PATH,
    REPO_ROOT,
    TUNED_HYPERPARAMETERS_PATH,
    build_experiment_plan,
    format_tuned_hyperparameters,
    load_config,
    load_sweep_configs,
    load_tuned_hyperparameters,
    resolve_tuned_values,
    tuned_keys,
    tuning_fingerprint,
)
from frwo.pause import is_alive, read_json, remove_file, write_json

# This module must not import `frwo.experiment_utils`: it pulls in PyTorch
# and every model library, close to 1 GB, which a sweep trial -- several at once,
# alive for as long as its experiment runs -- does not need.

# The sweep's metric: what `trial` logs, and what the sweep configs minimize.
OBJECTIVE = "objective"

# The `run_experiment_trial()` summary key the objective is the mean of, by
# default. Validation, so that the test split stays untouched by tuning (it is
# the same in every trial, and no trial trains or validates on its rows; see
# `experiment_utils._draw_test_split()`); sum aggregated absolute
# subdominance, which is what Fairness Reward Weight Optimization's own
# `opt_debias` weight search minimizes.
DEFAULT_OBJECTIVE_KEY = "sum_abs_subdominance_val"

# Set for each trial, and read by `wandb.init()`. A child process that inherits
# them has its every run forced onto the trial's run.
SWEEP_ENV_VARS = ("WANDB_RUN_ID", "WANDB_SWEEP_ID", "WANDB_SWEEP_PARAM_PATH")

# What `frwo-tune` writes its results to, for review.
TUNED_OUTPUT_PATH = TUNED_HYPERPARAMETERS_PATH.with_name(
    TUNED_HYPERPARAMETERS_PATH.name + ".new"
)

# The W&B project of `frwo-tune`'s sweeps. It is not the project the final
# experiments report to because the plotting notebook loads the most recent
# session of its project, and every sweep trial's runs form a session.
TUNING_PROJECT = "frwo-tuning"

# What `frwo-tune` keeps while tuning: a directory per sweep it has yet to
# finish, a record per `frwo-tune` that is running, and its trials' logs.
TUNE_STATE_DIR = REPO_ROOT / ".tuning_state"
TUNE_SWEEP_DIR = TUNE_STATE_DIR / "sweeps"
TUNE_RUNNING_DIR = TUNE_STATE_DIR / "running"
TUNE_LOG_DIR = TUNE_STATE_DIR / "logs"

# How many of a sweep's trials may fail, with none succeeding, before it is
# given up on. Failing is a fair result for some hyperparameters, but not for
# this many in a row from the start.
MAX_INITIAL_FAILURES = 20

# How long `frwo-tune` waits between checks of its trials.
POLL_SECONDS = 2

# The seconds a stopped trial gets to end before it is killed.
TRIAL_STOP_SECONDS = 30

# A trial's status, in its sweep's state. A pending trial is one that has been
# suggested and has not ended: it may be running, or be waiting to run again
# because tuning was interrupted.
TRIAL_PENDING = "pending"
TRIAL_FINISHED = "finished"
TRIAL_FAILED = "failed"


def _scored_runs(runs):
    """
    The runs the objective is computed over: every run, except Fairness Reward
    Weight Optimization's runs with the unadjusted weights. Those are only its
    starting point; its results are the runs of its weight adjustments. The
    experiment script marks which is which, as each run's `scored`.
    """
    return [run for run in runs if run["scored"]]


def _summarize(runs, objective_key):
    """
    Compute what a sweep trial logs from its runs' results.

    Returns
    -------
    metrics : dict<str, float> or None
        `None` when there is nothing to score, or when any scored run failed:
        a mean over only the runs that succeeded would favor hyperparameters
        that make a technique fail on its hardest cases.
    """
    scored = _scored_runs(runs)
    if not scored or not all(run["converged"] for run in scored):
        return None

    # Every subdominance measurement the experiment reported, in its order.
    subdominance_keys = list(scored[0]["subdominance"])
    if objective_key not in subdominance_keys:
        raise ValueError(
            f"Unknown objective key {objective_key!r}; the experiment reported"
            f" {subdominance_keys}."
        )

    metrics = {
        OBJECTIVE: float(
            np.mean([run["subdominance"][objective_key] for run in scored])
        )
    }
    for key in subdominance_keys:
        metrics[f"mean/{key}"] = float(
            np.mean([run["subdominance"][key] for run in scored])
        )
    return metrics


def run_trial(args):
    """Run one sweep trial. See the module docstring."""
    run = wandb.init(job_type="sweep_trial")
    # The values the search picked, plus the sweep config's fixed ones. Keys
    # starting with "_" are wandb's own bookkeeping.
    overrides = {k: v for k, v in dict(run.config).items() if not k.startswith("_")}

    with tempfile.TemporaryDirectory() as tmp:
        results_file = os.path.join(tmp, "results.json")
        command = [
            sys.executable,
            "-m",
            "frwo.FRWO_Experiments",
            "--config",
            args.config,
            "--results-file",
            results_file,
            "--no-pause",
            "--annotate",
            f"SWEEP_ID={run.sweep_id}",
            "--annotate",
            f"SWEEP_RUN_ID={run.id}",
        ]
        for key, value in overrides.items():
            command += ["--set", f"{key}={json.dumps(value)}"]

        env = {k: v for k, v in os.environ.items() if k not in SWEEP_ENV_VARS}
        experiment = subprocess.Popen(command, env=env)
        # A trial is stopped with SIGTERM, which must not leave the experiment
        # running on its own.
        signal.signal(signal.SIGTERM, lambda *_: experiment.terminate())
        returncode = experiment.wait()

        if returncode != 0:
            run.summary["trial_error"] = f"experiment exited with code {returncode}"
            run.finish(exit_code=1)
            sys.exit(returncode)

        with open(results_file) as f:
            results = json.load(f)

    runs = results["runs"]
    metrics = _summarize(runs, args.objective_key)

    run.summary["child_session_id"] = results["session_id"]
    run.summary["objective_key"] = args.objective_key
    run.summary["n_runs"] = len(runs)
    run.summary["n_scored_runs"] = len(_scored_runs(runs))
    run.summary["n_failed_runs"] = sum(not r["converged"] for r in _scored_runs(runs))

    if metrics is None:
        # Leaving the objective unlogged keeps this trial out of the search.
        run.summary["trial_error"] = "no scored runs, or a scored run failed"
        run.finish(exit_code=1)
        sys.exit(1)

    run.log(metrics)
    run.finish()

    if args.result_file:
        write_json(Path(args.result_file), {OBJECTIVE: metrics[OBJECTIVE]})


def _dataset_sweep_config(sweep_config, dataset):
    """The config of the sweep of `sweep_config` on `dataset`."""
    config = copy.deepcopy(sweep_config)
    # How `frwo-tune` schedules the sweep, which is none of W&B's business.
    del config["parallel_sweeps"]
    config["name"] = f"{sweep_config['name']} | {dataset}"
    config["project"] = TUNING_PROJECT
    config["parameters"]["SELECTED_DATASETS"] = {"value": [dataset]}
    return config


@dataclass
class TuningSweep:
    """
    One sweep of `frwo-tune`, and how far it has got.

    Attributes
    ----------
    name : str
        The sweep's name, "<technique> | <dataset>".
    algorithm : str
        The technique it tunes.
    parallel_sweeps : int
        The `parallel_sweeps` of its sweep config.
    path : str
        The sweep's W&B path, "<entity>/<project>/<id>".
    config : dict
        The sweep's W&B config.
    command : list<str>
        The `command` of its sweep config.
    directory : pathlib.Path
        Where its state is kept, including its copy of the resolved main
        experiment config.
    position : int
        Its place in the order `frwo-tune` was asked to tune.
    exp_info : dict
        The `exp_info` of the experiment the sweep tunes.
    trials : list<dict>
        Every trial suggested so far, in order: its `params`, its `status` and,
        if it finished, its `objective`.
    processes : dict<int, subprocess.Popen>
        Its trials that are running, by index.
    paused : bool
        Whether `frwo-tune-pause` paused it.
    gave_up : bool
        Whether it kept failing before it could finish.
    """

    name: str
    algorithm: str
    parallel_sweeps: int
    path: str
    config: dict
    command: list
    directory: Path
    position: int
    exp_info: dict
    trials: list = field(default_factory=list)
    processes: dict = field(default_factory=dict)
    paused: bool = False
    gave_up: bool = False

    @property
    def tuned_keys(self):
        """The parameters the search chooses: a fixed `value` is not chosen."""
        return tuned_keys(self.config)

    @property
    def run_cap(self):
        """How many trials the sweep runs in total."""
        return self.config.get("run_cap", math.inf)

    @property
    def n_ended(self):
        return sum(trial["status"] != TRIAL_PENDING for trial in self.trials)

    @property
    def finished(self):
        return self.n_ended >= self.run_cap

    @property
    def settled(self):
        return self.finished or (self.gave_up and not self.processes)

    def next_trial(self):
        """
        The index of the trial to start next, or None if there is none yet: a
        trial can't start before everything its suggestion is based on has ended
        (see the module docstring).
        """
        if self.gave_up:
            return None
        pending = [
            index
            for index, trial in enumerate(self.trials)
            if trial["status"] == TRIAL_PENDING
        ]
        waiting = [index for index in pending if index not in self.processes]
        next_index = waiting[0] if waiting else len(self.trials)
        if next_index >= self.run_cap:
            return None
        first_pending = pending[0] if pending else next_index
        if next_index - first_pending > self.parallel_sweeps - 1:
            return None
        return next_index


class TuningPaused(Exception):
    """Raised once every running trial has ended, to stop tuning."""


def _log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _tune_targets(args, plan, sweep_configs):
    """
    What `frwo-tune` was asked to tune.

    Returns
    -------
    targets : list<tuple<str, dict>>
        Every technique to tune and the `exp_info` of the experiment to tune it
        on. Techniques without a sweep config have nothing to tune.
    """
    exp_infos = {exp_info["DATASET"]: exp_info for exp_info in plan.experiments}

    if args.algorithm and args.algorithm not in sweep_configs:
        sys.exit(
            f"Nothing to tune for algorithm {args.algorithm!r}. Algorithms with"
            f" hyperparameters: {sorted(sweep_configs)}."
        )
    if args.dataset and args.dataset not in exp_infos:
        sys.exit(f"Unknown dataset {args.dataset!r}. Datasets: {sorted(exp_infos)}.")

    targets = []
    for dataset in [args.dataset] if args.dataset else plan.selected_datasets:
        exp_info = exp_infos[dataset]
        for algorithm in [args.algorithm] if args.algorithm else exp_info["ALGORITHMS"]:
            if algorithm in sweep_configs:
                targets.append((algorithm, exp_info))
    return targets


def _sweep_directory(name):
    return TUNE_SWEEP_DIR / name.lower().replace(" | ", "__").replace(" ", "_")


def _save_sweep(sweep):
    """Record `sweep`, except for what is only true of this process."""
    write_json(
        sweep.directory / "state.json",
        {
            "name": sweep.name,
            "algorithm": sweep.algorithm,
            "parallel_sweeps": sweep.parallel_sweeps,
            "path": sweep.path,
            "config": sweep.config,
            "command": sweep.command,
            "position": sweep.position,
            "trials": sweep.trials,
            "paused": sweep.paused,
        },
    )


def _load_sweep(directory):
    """The sweep recorded in `directory`, or None if there is none."""
    state = read_json(directory / "state.json")
    if state is None:
        return None
    dataset = state["config"]["parameters"]["SELECTED_DATASETS"]["value"][0]
    plan = build_experiment_plan(load_config(directory / "experiment.yaml"))
    exp_infos = {exp_info["DATASET"]: exp_info for exp_info in plan.experiments}
    return TuningSweep(directory=directory, exp_info=exp_infos[dataset], **state)


def _load_paused_sweeps():
    if not TUNE_SWEEP_DIR.exists():
        return []
    sweeps = [_load_sweep(directory) for directory in TUNE_SWEEP_DIR.iterdir()]
    paused = [sweep for sweep in sweeps if sweep is not None and sweep.paused]
    return sorted(paused, key=lambda sweep: sweep.position)


@functools.cache
def _api():
    # Just the one: each `wandb.Api` stays open in wandb's service process, whose
    # memory would otherwise grow for as long as tuning runs.
    return wandb.Api(timeout=60)


def _fetch_sweep(path):
    sweep = _api().sweep(path)
    # `wandb.Api` keeps the sweeps it has fetched, and `flush()` does not clear
    # them, so this one may be out of date.
    sweep.load(force=True)
    return sweep


def _can_continue(saved, config, parallel_sweeps, exp_info):
    """
    Whether a sweep from an earlier `frwo-tune` is the one it would create
    now: changing its config, its experiment or what its suggestions are based
    on would mix results from different conditions.
    """
    if (
        saved is None
        or saved.config != config
        or saved.parallel_sweeps != parallel_sweeps
        or saved.exp_info != exp_info
    ):
        return False
    try:
        _fetch_sweep(saved.path)
    except wandb.errors.CommError:
        _log(f"{saved.path} is gone from W&B; starting {config['name']} again.")
        return False
    return True


def _open_sweeps(targets, sweep_configs, config):
    """
    Create the sweeps of `targets`, or find the ones an interrupted
    `frwo-tune` left, as long as nothing they depend on has changed since.
    A new sweep keeps a copy of `config`, the resolved main experiment config.
    """
    entity = os.environ.get("WANDB_ENTITY") or _api().default_entity

    sweeps = []
    for position, (algorithm, exp_info) in enumerate(targets):
        sweep_config = sweep_configs[algorithm]
        parallel_sweeps = sweep_config["parallel_sweeps"]
        config = _dataset_sweep_config(sweep_config, exp_info["DATASET"])
        directory = _sweep_directory(config["name"])
        saved = _load_sweep(directory)
        if _can_continue(saved, config, parallel_sweeps, exp_info):
            sweep = saved
            sweep.position = position
            _log(f"Continuing sweep {sweep.path}: {sweep.name}")
        else:
            sweep_id = wandb.sweep(config, entity=entity, project=TUNING_PROJECT)
            shutil.rmtree(directory, ignore_errors=True)
            directory.mkdir(parents=True)
            (directory / "experiment.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False)
            )
            sweep = TuningSweep(
                name=config["name"],
                algorithm=algorithm,
                parallel_sweeps=parallel_sweeps,
                path=f"{entity}/{TUNING_PROJECT}/{sweep_id}",
                config=config,
                command=sweep_config["command"],
                directory=directory,
                position=position,
                exp_info=exp_info,
            )
            _log(f"Created sweep {sweep.path}: {sweep.name}")
        _save_sweep(sweep)
        sweeps.append(sweep)
    return sweeps


def _trial_seed(sweep, index):
    text = f"{sweep.exp_info['RANDOM_SEED']}|{sweep.name}|{index}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def _suggest_params(sweep):
    """
    What W&B's Bayesian search suggests for the sweep's next trial, which
    depends only on the sweep's config and the results of the trials before it
    that the trial's index says it may see (see the module docstring).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Imported here rather than at the top of the module: the search pulls
        # in scikit-learn, which a sweep trial does not need.
        import sweeps as bayesian_search

        n_seen = len(sweep.trials) - (sweep.parallel_sweeps - 1)
        runs = []
        for index, trial in enumerate(sweep.trials):
            config = {key: {"value": value} for key, value in trial["params"].items()}
            if index >= n_seen:
                state = bayesian_search.RunState.running
                metrics = {}
            elif trial["status"] == TRIAL_FINISHED:
                state = bayesian_search.RunState.finished
                metrics = {OBJECTIVE: trial["objective"]}
            else:
                state = bayesian_search.RunState.failed
                metrics = {}
            # `SweepRun` silently ignores these two given by their field names.
            runs.append(
                bayesian_search.SweepRun(
                    name=str(index),
                    config=config,
                    state=state,
                    summaryMetrics=metrics,
                    sampledHistory=[metrics] if metrics else [],
                )
            )

        seed = _trial_seed(sweep, len(sweep.trials))
        np.random.seed(seed)
        random.seed(seed)
        suggestion = bayesian_search.next_runs(sweep.config, runs, validate=False)[0]

    return {
        # The search's values can be numpy scalars, which JSON has no form for.
        key: (
            param["value"].item()
            if isinstance(param["value"], np.generic)
            else param["value"]
        )
        for key, param in suggestion.config.items()
    }


def _trial_paths(sweep, index):
    """Where trial `index` is given its parameters and leaves its result."""
    directory = sweep.directory / "trials"
    return directory / f"{index}.yaml", directory / f"{index}.json"


def _start_trial(sweep, index):
    """Start trial `index` of `sweep`, suggesting it if it is a new one."""
    if index == len(sweep.trials):
        sweep.trials.append({"params": _suggest_params(sweep), "status": TRIAL_PENDING})
        _save_sweep(sweep)

    params_path, result_path = _trial_paths(sweep, index)
    params_path.parent.mkdir(exist_ok=True)
    result_path.unlink(missing_ok=True)
    params_path.write_text(
        yaml.safe_dump(
            {
                key: {"value": value}
                for key, value in sweep.trials[index]["params"].items()
            }
        )
    )

    TUNE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = TUNE_LOG_DIR / f"{sweep.directory.name}.log"

    entity, project, sweep_id = sweep.path.split("/")
    env = dict(os.environ)
    # Using wandb here starts a service process and records it in WANDB_SERVICE.
    # A trial that inherits that depends on this process staying alive.
    env.pop("WANDB_SERVICE", None)
    # What makes the trial's run one of the sweep's. The project and entity also
    # send its child's runs to the same place.
    env |= {
        "WANDB_ENTITY": entity,
        "WANDB_PROJECT": project,
        "WANDB_RUN_ID": uuid.uuid4().hex[:8],
        "WANDB_SWEEP_ID": sweep_id,
        "WANDB_SWEEP_PARAM_PATH": str(params_path),
    }

    command = [
        sys.executable if part == "${interpreter}" else part
        for part in sweep.command
        if part != "${env}"
    ]
    with open(log_path, "a") as log_file:
        sweep.processes[index] = subprocess.Popen(
            command
            + [
                "--config",
                str(sweep.directory / "experiment.yaml"),
                "--result-file",
                str(result_path),
            ],
            # The datasets are loaded relative to the repository.
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            # Stopping trials is left to `_stop_trials()`, as a signal from the
            # terminal could reach them before this process had stopped
            # starting new ones.
            start_new_session=True,
        )


def _start_trials(sweeps, parallel_sweeps):
    """
    Start trials until `parallel_sweeps` are running. Every sweep gets one
    before any gets a second, so that many sweeps run at once if that many are
    left. With fewer left, the spare trials go to the ones that are, rather than
    leaving the slowest sweep to finish alone. How the trials are shared out
    has no effect on the results.
    """
    n_running = sum(len(sweep.processes) for sweep in sweeps)
    while n_running < parallel_sweeps:
        candidates = [sweep for sweep in sweeps if sweep.next_trial() is not None]
        if not candidates:
            return
        sweep = min(candidates, key=lambda candidate: len(candidate.processes))
        _start_trial(sweep, sweep.next_trial())
        n_running += 1


def _server_is_reachable(sweep):
    try:
        _fetch_sweep(sweep.path)
    except wandb.errors.Error:
        return False
    return True


def _end_trials(sweep):
    """Record the trials of `sweep` that have ended."""
    for index, process in list(sweep.processes.items()):
        if process.poll() is None:
            continue
        del sweep.processes[index]

        trial = sweep.trials[index]
        result = read_json(_trial_paths(sweep, index)[1])
        if process.returncode == 0 and result is not None:
            trial["status"] = TRIAL_FINISHED
            trial["objective"] = result[OBJECTIVE]
        elif _server_is_reachable(sweep):
            trial["status"] = TRIAL_FAILED
        else:
            # The hyperparameters can't be blamed for this. Run it again, or
            # the server being away would change the results, and fail every
            # trial left in the sweep.
            _log(f"{sweep.name}: W&B is unreachable, so trial {index} will run again.")
            continue
        _save_sweep(sweep)
        _log(f"{sweep.name}: {sweep.n_ended}/{sweep.run_cap} runs done")

        if sweep.finished:
            _log(f"{sweep.name}: finished")

    n_failed = sum(trial["status"] == TRIAL_FAILED for trial in sweep.trials)
    n_finished = sum(trial["status"] == TRIAL_FINISHED for trial in sweep.trials)
    if not sweep.gave_up and n_failed >= MAX_INITIAL_FAILURES and not n_finished:
        sweep.gave_up = True
        _log(f"{sweep.name}: giving up, its trials keep failing. See {TUNE_LOG_DIR}.")


def _run_trials(sweeps, parallel_sweeps, registration):
    """
    Run trials until every sweep has either finished or been given up on.

    Raises
    ------
    TuningPaused
        If a pause was requested, once every trial that was running has ended.
    """
    while not all(sweep.settled for sweep in sweeps):
        for sweep in sweeps:
            _end_trials(sweep)
        if not registration.pause_requested():
            _start_trials(sweeps, parallel_sweeps)
        elif not any(sweep.processes for sweep in sweeps):
            raise TuningPaused()
        if not all(sweep.settled for sweep in sweeps):
            time.sleep(POLL_SECONDS)


def _stop_trials(sweeps):
    """
    Stop every running trial. Each one leads a process group, which the
    experiment it started belongs to as well.
    """
    processes = [process for sweep in sweeps for process in sweep.processes.values()]

    def signal_trial(process, signum):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    for process in processes:
        signal_trial(process, signal.SIGTERM)
    deadline = time.monotonic() + TRIAL_STOP_SECONDS
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            signal_trial(process, signal.SIGKILL)


def _best_trial(sweep):
    """
    The finished trial of `sweep` with the lowest objective, or None. Of several
    with the same objective, it is the earliest.
    """
    trials = [trial for trial in sweep.trials if trial["status"] == TRIAL_FINISHED]
    return min(trials, key=lambda trial: trial["objective"], default=None)


def _write_tuned_hyperparameters(sweeps):
    """
    Write `TUNED_OUTPUT_PATH`: the tuned hyperparameters file as it is now,
    with the best values of `sweeps` set in the entries of their datasets and
    techniques.

    Returns
    -------
    n_written : int
        How many of `sweeps` had a best run to write.
    """
    tuned = load_tuned_hyperparameters()
    n_written = 0
    for sweep in sweeps:
        best = _best_trial(sweep)
        if best is None:
            _log(f"{sweep.name}: no run was scored, so it has nothing to write.")
            continue
        values = {key: best["params"][key] for key in sweep.tuned_keys}
        tuned.setdefault(sweep.exp_info["DATASET"], {})[sweep.algorithm] = {
            "values": resolve_tuned_values(sweep.exp_info, values),
            "objective": best[OBJECTIVE],
            "sweep": sweep.path,
            "tuned_against": tuning_fingerprint(sweep.exp_info),
        }
        n_written += 1
        _log(f"{sweep.name}: best objective {best[OBJECTIVE]:.6g}, {values}")

    if n_written:
        TUNED_OUTPUT_PATH.write_text(format_tuned_hyperparameters(tuned))
    return n_written


def _end_wandb_sweep(sweep):
    """
    Mark `sweep` as finished in the W&B UI, which won't notice on its own without
    an agent. Only the UI goes by this, so failing to is no reason to stop.
    """
    returncode = subprocess.run(
        [sys.executable, "-m", "wandb", "sweep", "--stop", sweep.path],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode
    if returncode:
        _log(f"{sweep.name}: could not mark {sweep.path} as finished in W&B.")


def _running_tuners():
    """
    The record of every `frwo-tune` that is running. One left behind by a
    process that died without cleaning up is removed, and doesn't count.
    """
    if not TUNE_RUNNING_DIR.exists():
        return []
    records = []
    for path in TUNE_RUNNING_DIR.glob("*.json"):
        record = read_json(path)
        if is_alive(record["pid"]):
            records.append(record)
        else:
            path.unlink(missing_ok=True)
            remove_file(_pause_request_path(record["pid"]))
    return records


def _pause_request_path(pid):
    return TUNE_RUNNING_DIR / f"{pid}.pause"


class _TuningRegistration:
    """
    Records this process as tuning `names` for as long as the `with` block lasts,
    which is how `frwo-tune-pause` finds it, and how two of them are kept
    from running the same sweep.
    """

    def __init__(self, names):
        self.names = names
        self.record_path = TUNE_RUNNING_DIR / f"{os.getpid()}.json"
        self.request_path = _pause_request_path(os.getpid())

    def __enter__(self):
        for other in _running_tuners():
            taken = sorted(set(other["sweeps"]) & set(self.names))
            if taken:
                sys.exit(
                    f"Another frwo-tune (pid {other['pid']}) is already running"
                    f" these sweeps: {taken}."
                )
        write_json(self.record_path, {"pid": os.getpid(), "sweeps": self.names})
        return self

    def __exit__(self, *exc):
        remove_file(self.request_path)
        remove_file(self.record_path)
        return False

    def pause_requested(self):
        return self.request_path.exists()


def _tune(sweeps, registration):
    """Run `sweeps` to the end, unless tuning is paused or interrupted."""
    # However this process is told to stop, its trials have to stop with it.
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _interrupt)
    try:
        # One technique at a time, because each one's `parallel_sweeps` is the
        # number that suits it with the machine to itself.
        for algorithm in dict.fromkeys(sweep.algorithm for sweep in sweeps):
            algorithm_sweeps = [s for s in sweeps if s.algorithm == algorithm]
            parallel_sweeps = algorithm_sweeps[0].parallel_sweeps
            _log(f"Tuning {algorithm}, up to {parallel_sweeps} trials at once")
            _run_trials(algorithm_sweeps, parallel_sweeps, registration)
    except TuningPaused:
        for sweep in sweeps:
            sweep.paused = True
            _save_sweep(sweep)
        _log("Tuning is paused. Resume it with `uv run frwo-tune-resume`.")
        return
    except KeyboardInterrupt:
        _log("Stopping the trials. Run the same command again to continue.")
        _stop_trials(sweeps)
        sys.exit(130)

    finished = [sweep for sweep in sweeps if sweep.finished]
    n_written = _write_tuned_hyperparameters(finished)
    for sweep in finished:
        _end_wandb_sweep(sweep)
        shutil.rmtree(sweep.directory)

    # A sweep given up on starts over when tuning is next run, with the same
    # suggestions: none of its trials got anywhere, and its results can't go on
    # to depend on failures that fixing their cause would have prevented.
    for sweep in sweeps:
        if sweep.gave_up and not sweep.finished:
            sweep.trials = []
            _save_sweep(sweep)

    if n_written:
        _log(f"Wrote the best values of {n_written} sweeps to {TUNED_OUTPUT_PATH}")
    n_failed = len(sweeps) - n_written
    if n_failed:
        sys.exit(f"{n_failed} of {len(sweeps)} sweeps did not produce a result.")


def parse_tune_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="frwo-tune",
        description=(
            "Tune hyperparameters with W&B sweeps, under the settings of"
            f" {MAIN_CONFIG_PATH.name}: create the sweeps, run their trials, and"
            f" write the best values found to {TUNED_OUTPUT_PATH.name}. Running"
            " it again after an interruption continues the same sweeps."
        ),
    )
    parser.add_argument(
        "--algorithm",
        help=(
            "Only tune this algorithm. Default: every one selected in"
            f" {MAIN_CONFIG_PATH.name}."
        ),
    )
    parser.add_argument(
        "--dataset",
        help=(
            "Only tune on this dataset. Default: every one selected in"
            f" {MAIN_CONFIG_PATH.name}."
        ),
    )
    return parser.parse_args(argv)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def tune_main(argv=None):
    """Tune hyperparameters from start to finish. See the module docstring."""
    args = parse_tune_args(argv)

    config = load_config(MAIN_CONFIG_PATH, tuned=False)
    plan = build_experiment_plan(config)
    sweep_configs = load_sweep_configs()
    targets = _tune_targets(args, plan, sweep_configs)
    if not targets:
        sys.exit("Nothing to tune: no selected algorithm has hyperparameters.")

    names = [
        f"{sweep_configs[algorithm]['name']} | {exp_info['DATASET']}"
        for algorithm, exp_info in targets
    ]
    paused = sorted({sweep.name for sweep in _load_paused_sweeps()} & set(names))
    if paused:
        sys.exit(
            f"These sweeps are paused: {paused}. Resume them with `uv run"
            " frwo-tune-resume`, or discard them with `uv run"
            " frwo-tune-resume --discard`."
        )

    with _TuningRegistration(names) as registration:
        _tune(_open_sweeps(targets, sweep_configs, config), registration)


def tune_pause_main(argv=None):
    """The `frwo-tune-pause` command."""
    parser = argparse.ArgumentParser(
        prog="frwo-tune-pause",
        description=(
            "Pause all running hyperparameter tuning once its current trials"
            " finish. Resume it with `uv run frwo-tune-resume`."
        ),
    )
    parser.add_argument(
        "--cancel",
        action="store_true",
        help="Withdraw a pause that has not happened yet.",
    )
    args = parser.parse_args(argv)

    running = _running_tuners()
    if not running:
        print("No hyperparameter tuning is running.", file=sys.stderr)
        sys.exit(1)

    for record in running:
        if args.cancel:
            remove_file(_pause_request_path(record["pid"]))
        else:
            write_json(_pause_request_path(record["pid"]), {"pid": record["pid"]})
    n_sweeps = sum(len(record["sweeps"]) for record in running)
    if args.cancel:
        print(f"Tuning of {n_sweeps} sweeps will carry on.")
    else:
        print(
            f"Tuning of {n_sweeps} sweeps will pause once its current trials"
            " finish. Resume it with `uv run frwo-tune-resume`."
        )


def tune_resume_main(argv=None):
    """The `frwo-tune-resume` command."""
    parser = argparse.ArgumentParser(
        prog="frwo-tune-resume",
        description=(
            "Resume the hyperparameter tuning paused with `uv run"
            " frwo-tune-pause`, and write the best values found when it"
            " finishes, as `frwo-tune` does."
        ),
    )
    parser.add_argument(
        "--discard",
        action="store_true",
        help="Delete the paused sweeps' progress instead of resuming them.",
    )
    args = parser.parse_args(argv)

    if any(_pause_request_path(record["pid"]).exists() for record in _running_tuners()):
        sys.exit("Tuning is still finishing its trials. Try again when it has paused.")
    sweeps = _load_paused_sweeps()
    if not sweeps:
        sys.exit("No hyperparameter tuning is paused.")

    if args.discard:
        for sweep in sweeps:
            shutil.rmtree(sweep.directory)
        print(f"Discarded the progress of {len(sweeps)} paused sweeps.")
        return

    with _TuningRegistration([sweep.name for sweep in sweeps]) as registration:
        _log(f"Resuming {len(sweeps)} sweeps")
        _tune(sweeps, registration)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    trial = subparsers.add_parser(
        "trial",
        help="Run one sweep trial. Started by `frwo-tune`, not by hand.",
    )
    trial.add_argument("--config", default=str(MAIN_CONFIG_PATH))
    trial.add_argument(
        "--result-file",
        help="Also write the objective to this JSON file, if the trial succeeds.",
    )
    trial.add_argument(
        "--objective-key",
        default=DEFAULT_OBJECTIVE_KEY,
        help=(
            "The subdominance measurement whose mean over the trial's runs is"
            " the objective. Default: %(default)s"
        ),
    )
    trial.set_defaults(func=run_trial)

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
