import argparse
import contextlib
import cProfile
import json
import logging
import pstats
import random
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from IPython.display import HTML, display

from frwo.config import (
    DEFAULT_CONFIG_PATH,
    build_experiment_plan,
    load_config,
    parse_assignment,
)
from frwo.datasets import *
from frwo.experiment_utils import *
from frwo.irl.frwo import *
from frwo.pause import (
    ExperimentPaused,
    PauseListener,
    capture_rng_state,
    discard_checkpoint,
    load_checkpoint,
    restore_rng_state,
    running_experiment,
    save_checkpoint,
)
from frwo.utils import *


def play_notification():
    subprocess.run(
        ["powershell.exe", "-Command", "[System.Media.SystemSounds]::Hand.Play()"]
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate Fairness Reward Weight Optimization and its"
            " baselines, as configured by an experiment config file, and"
            " report the results to W&B."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="The experiment config file. Default: %(default)s",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override one config value for every experiment, e.g. `--set"
            " SH_ITERS=10` or `--set 'SELECTED_DATASETS=[\"COMPAS\"]'`. VALUE"
            " is read as JSON, falling back to a plain string. Repeatable."
        ),
    )
    parser.add_argument(
        "--annotate",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Record an extra value in every W&B run's config, without"
            " affecting the experiments. Repeatable."
        ),
    )
    parser.add_argument(
        "--results-file",
        help=(
            "Also write every run's outcome and subdominance results to this"
            " JSON file. Used by W&B sweep trials (see frwo.sweep)."
        ),
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help=(
            "Don't let `frwo-pause` pause this experiment. Used by W&B"
            " sweep trials, which must run to completion."
        ),
    )
    return parser.parse_args(argv)


def _run_result_recorder(records):
    """
    A `RUN_RESULT_LISTENERS` entry that collects each run's outcome into
    `records`, keyed by run id so that a run first finalized and then marked as
    failed ends up failed.
    """

    def record(run, summary):
        records[run.id] = {
            "run_id": run.id,
            "ALGORITHM": run.config["ALGORITHM"],
            "EXPERIMENT_NAME": run.config["EXPERIMENT_NAME"],
            "DATASET": run.config["DATASET"],
            "BIAS_TYPE": run.config["BIAS_TYPE"],
            "DATASET_BIAS_TYPE": run.config["DATASET_BIAS_TYPE"],
            "DEMO_BIAS_TYPE": run.config["DEMO_BIAS_TYPE"],
            "WEIGHT_ADJUST": run.config["WEIGHT_ADJUST"],
            "TRIAL": run.config["TRIAL"],
            # Whether a sweep trial's objective counts this run: every run but
            # Fairness Reward Weight Optimization's with the unadjusted
            # weights, which are only its starting point.
            "scored": not (
                run.config["ALGORITHM"] == ALGORITHM_FRWO
                and not run.config["WEIGHT_ADJUST"]
            ),
            "converged": summary is not None,
            "subdominance": (
                {key: float(summary[key]) for key in SUBDOMINANCE_KEYS}
                if summary is not None
                else None
            ),
        }

    return record


def _setup_output():
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")

    display(HTML("<style>.container { width:2000px !important; }</style>"))
    pd.set_option("display.max_columns", None)
    # pd.set_option('display.max_colwidth', None)

    # Prevent long logging lines from wrapping
    # display(HTML("<style>div.output_area pre {white-space: pre;}</style>"))
    np.set_printoptions(linewidth=np.inf)


def main(argv=None):
    args = parse_args(argv)
    _setup_output()

    # Every experiment setting lives in the config file; see the header of
    # configs/experiments/experiment.yaml for how it is laid out.
    logging.info(f"Loading experiment config: {args.config}")
    run_experiments(
        config=load_config(args.config),
        config_path=args.config,
        overrides=args.overrides,
        annotations=args.annotate,
        results_file=args.results_file,
        pausable=not args.no_pause,
    )


def resume_main(argv=None):
    """The `frwo-resume` command. See `frwo.pause`."""
    parser = argparse.ArgumentParser(
        description=(
            "Resume the experiment paused with `uv run frwo-pause`,"
            " starting with its next trial."
        )
    )
    parser.add_argument(
        "--discard",
        action="store_true",
        help="Delete the paused experiment instead of resuming it.",
    )
    args = parser.parse_args(argv)

    checkpoint = load_checkpoint()
    if checkpoint is None:
        sys.exit("No experiment is paused.")
    if args.discard:
        discard_checkpoint()
        print(
            f"Discarded the paused experiment (W&B session {checkpoint['session_id']})."
        )
        return
    running = running_experiment()
    if running is not None and running["session_id"] == checkpoint["session_id"]:
        sys.exit(
            f"The paused experiment has already been resumed (pid {running['pid']})."
        )

    _setup_output()
    logging.info(
        f"Resuming the experiment paused after {checkpoint['paused_after']},"
        f" with the config {checkpoint['config_path']} as it was when the"
        " experiment started."
    )
    run_experiments(
        config=checkpoint["config"],
        config_path=checkpoint["config_path"],
        overrides=checkpoint["overrides"],
        annotations=checkpoint["annotations"],
        results_file=checkpoint["results_file"],
        pausable=True,
        resume=checkpoint,
    )


def run_experiments(
    config,
    config_path,
    overrides,
    annotations,
    results_file,
    pausable,
    resume=None,
):
    """
    Run every experiment the config selects, or, given the checkpoint of a
    paused experiment as `resume`, the ones it had not finished yet.

    Parameters
    ----------
    config : dict
        The contents of the config file.
    config_path : str
        Where the config file is. Only recorded in a checkpoint, for the log.
    overrides, annotations : list<str>
        The `--set` and `--annotate` values, as `KEY=VALUE` text.
    results_file : str or None
        See `--results-file`.
    pausable : bool
        Whether `frwo-pause` can pause the experiment.
    resume : dict, Optional
        The checkpoint to resume from (see `frwo.pause`).
    """
    plan = build_experiment_plan(
        config,
        overrides=dict(parse_assignment(text) for text in overrides),
        annotations=dict(parse_assignment(text) for text in annotations),
    )

    np.random.seed(plan.random_seed)
    random.seed(plan.random_seed)

    run_records = dict(resume["run_records"]) if resume else {}
    if results_file:
        RUN_RESULT_LISTENERS.append(_run_result_recorder(run_records))

    # Run experiments. Results are reported to Weights & Biases (project
    # `WANDB_PROJECT`, default "frwo"); the server address and credentials
    # come from the usual wandb configuration, i.e. the WANDB_BASE_URL
    # environment variable or ~/.config/wandb/settings.
    #
    # Every run of every experiment below shares this one session id, which is
    # what lets the plotting notebook pick out this execution's results rather
    # than mixing them with those of earlier executions. A resumed experiment
    # keeps its session id, so it is still one session.
    session_id = resume["session_id"] if resume else new_session_id()
    logging.info(f"Logging results to W&B project: {WANDB_PROJECT}")
    logging.info(f"W&B session: {session_id}")

    # Every experiment to run, in order: each selected one, once per expert.
    # `exp_i` and `expert_i` locate it in the plan, which is how a checkpoint
    # records where the experiment paused.
    jobs = []
    for exp_i, base_exp_info in enumerate(plan.experiments):
        if base_exp_info["DATASET"] not in plan.selected_datasets:
            logging.info(
                f"Skipping experiment {base_exp_info['EXPERIMENT_NAME']} since it is not in the selected datasets list."
            )
            continue
        for expert_i, expert_algo in enumerate(plan.expert_algos):
            exp_info = dict(base_exp_info)
            exp_info["EXPERT_ALGO"] = expert_algo
            jobs.append((exp_i, expert_i, exp_info))

    with (
        PauseListener(session_id) if pausable else contextlib.nullcontext()
    ) as listener:
        try:
            for job_i, (exp_i, expert_i, exp_info) in enumerate(jobs):
                logging.info(
                    f"EXPERIMENT {expert_i+1}/{len(plan.expert_algos)}"
                    f" of {exp_info['EXPERIMENT_NAME']}"
                )

                # Until a resumed experiment reaches the first trial it has not
                # run, skip what it ran before the pause.
                first_trial = 0
                trial_rng_state = None
                if resume is not None:
                    last_exp_i, last_expert_i, last_trial = resume["last_completed"]
                    last_job = (last_exp_i, last_expert_i)
                    if (exp_i, expert_i) < last_job or (
                        (exp_i, expert_i) == last_job
                        and last_trial + 1 >= exp_info["N_TRIALS"]
                    ):
                        logging.info("Skipping it, since it finished before the pause.")
                        continue
                    if (exp_i, expert_i) == last_job:
                        # Paused partway through this experiment: regenerate
                        # its dataset from the same random state as before,
                        # then pick up the random state the last trial left.
                        first_trial = last_trial + 1
                        restore_rng_state(resume["experiment_rng_state"])
                        trial_rng_state = resume["rng_state"]
                    else:
                        restore_rng_state(resume["rng_state"])
                    logging.info(f"Resuming it at trial {first_trial}.")
                    resume = None

                experiment_rng_state = capture_rng_state()

                source_X, source_y, source_feature_types = generate_dataset(
                    exp_info["DATASET"],
                    n_samples=exp_info["N_DATASET_SAMPLES"],
                )

                for f in source_feature_types["categoric"]:
                    # .map(str) (rather than .astype(str)) matches Pandas < 3
                    # behavior: it naively stringifies every value, including
                    # missing ones (NaN -> "nan"). Pandas >= 3's .astype(str) is
                    # NA-aware and leaves missing values as missing instead.
                    # .astype(object) then keeps the legacy object dtype instead
                    # of the new strict "str" dtype, which rejects assigning
                    # non-string sentinel values used elsewhere in the pipeline
                    # (e.g. state reduction).
                    source_X[f] = source_X[f].map(str).astype(object)

                source_X_cols = input_columns(source_feature_types)

                if exp_info["USE_HIDDEN_FEATURES_SOURCE"]:
                    source_X_cols += source_feature_types["hidden"]
                _source_X = source_X[source_X_cols]

                if trial_rng_state is not None:
                    restore_rng_state(trial_rng_state)

                logging.info(
                    f"For dataset: {exp_info['DATASET']} and expert algo: {exp_info['EXPERT_ALGO']}:"
                )

                def pause_if_requested(trial_i):
                    is_last_trial = (
                        job_i == len(jobs) - 1 and trial_i == exp_info["N_TRIALS"] - 1
                    )
                    if (
                        listener is None
                        or is_last_trial
                        or not listener.pause_requested()
                    ):
                        return
                    save_checkpoint(
                        {
                            "session_id": session_id,
                            "config_path": str(config_path),
                            "config": config,
                            "overrides": list(overrides),
                            "annotations": list(annotations),
                            "results_file": results_file,
                            "last_completed": [exp_i, expert_i, trial_i],
                            "paused_after": (
                                f"trial {trial_i} of experiment"
                                f" {exp_info['EXPERIMENT_NAME']} with expert"
                                f" {exp_info['EXPERT_ALGO']}"
                            ),
                            "experiment_rng_state": experiment_rng_state,
                            "rng_state": capture_rng_state(),
                            "run_records": run_records,
                        }
                    )
                    raise ExperimentPaused()

                run_bias_experiment(
                    exp_info,
                    source_X=_source_X,
                    source_y=source_y,
                    source_feature_types=source_feature_types,
                    session_id=session_id,
                    first_trial=first_trial,
                    after_trial=pause_if_requested,
                )
        except ExperimentPaused:
            logging.info(
                f"EXPERIMENT PAUSED. W&B session: {session_id}. Resume it with"
                " `uv run frwo-resume`."
            )
            return

    if results_file:
        with open(results_file, "w") as f:
            json.dump(
                {"session_id": session_id, "runs": list(run_records.values())},
                f,
                indent=2,
            )

    # A resumed experiment keeps its checkpoint until it finishes, so that if
    # it crashes it can still be resumed from where it was paused.
    paused = load_checkpoint()
    if paused is not None and paused["session_id"] == session_id:
        discard_checkpoint()

    logging.info(f"TRAINING FINISHED SUCESSFULLY! W&B session: {session_id}")

    # play_notification()


if __name__ == "__main__":
    main()
