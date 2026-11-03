"""
Loading of the experiment configuration files.

Three kinds of file make up an experiment's config:

* configs/base.yaml: everything every experiment config shares -- the
  `exp_info` values of every experiment (`common`), each dataset's own values
  (`datasets`), the random seed and the experts.
* configs/tuned_hyperparameters.yaml: the hyperparameters `frwo-tune` found,
  per dataset and technique, including each dataset's `opt_debias` weight
  adjustment (WEIGHT_ADJUST_LIST). Every experiment config runs with them, and
  turns the weight adjustment on or off with RUN_OPT_DEBIAS.
* configs/experiments/*.yaml: one file per experiment config, holding only
  where it differs from the base. `frwo-tune` tunes the hyperparameters under
  the main one, configs/experiments/main.yaml.

See the header of configs/base.yaml for how they are merged.

`load_config()` merges them into a resolved config: one dict holding every
setting of every dataset's experiment, which no longer depends on any file.
That is also what a checkpoint and a sweep keep, so that editing the files
cannot change a paused experiment or a sweep partway through.
`build_experiment_plan()` turns a resolved config into exactly the `exp_info`
dicts the experiment code has always received, and applies `--set KEY=VALUE`
overrides on top, which is also how a W&B sweep sets the hyperparameters it
tunes.
"""

import copy
import functools
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
BASE_CONFIG_PATH = CONFIG_DIR / "base.yaml"
EXPERIMENT_CONFIG_DIR = CONFIG_DIR / "experiments"
SWEEP_CONFIG_DIR = CONFIG_DIR / "sweeps"
TUNED_HYPERPARAMETERS_PATH = CONFIG_DIR / "tuned_hyperparameters.yaml"

# The experiment config `frwo-tune` tunes the hyperparameters under.
MAIN_CONFIG_NAME = "main"
MAIN_CONFIG_PATH = EXPERIMENT_CONFIG_DIR / f"{MAIN_CONFIG_NAME}.yaml"

# The top-level config values: they configure the run as a whole rather than
# one experiment, but are overridden with `--set` like any `exp_info` value.
TOP_LEVEL_KEYS = ("RANDOM_SEED", "SELECTED_DATASETS", "EXPERT_ALGOS")

# What each kind of file may hold at its top level.
BASE_FILE_KEYS = ("RANDOM_SEED", "EXPERT_ALGOS", "common", "datasets")
EXPERIMENT_FILE_KEYS = ("SELECTED_DATASETS", "EXPERT_ALGOS", "common", "experiments")

# Every experiment's EXPERIMENT_NAME and DATASET are its key under the base's
# `datasets`, and no file sets them.
NAME_KEYS = ("EXPERIMENT_NAME", "DATASET")

# Settings an experiment config may not change from the base: those that pick
# each trial's data split and seed its random draws (see
# `experiment_utils._data_split_seed()` and `_baseline_seed()`). They are what
# the hyperparameters were tuned on, so every experiment config shares them.
LOCKED_KEYS = (
    "RANDOM_SEED",
    *NAME_KEYS,
    "DATA_SPLIT_FRACTIONS",
    "DATA_SPLIT_STRATIFY",
    "DATA_SPLIT_MAX_TRIES",
    "DATA_SPLIT_MAX_IMBALANCE",
    "N_DATASET_SAMPLES",
)

# `exp_info` keys whose values the experiment code has always received as
# tuples (of tuples). YAML has no tuples, so these are converted back on load,
# keeping `exp_info` identical to the dicts that used to be written in Python.
# The bias types and weight adjustments in particular are compared with `()`.
TUPLE_KEYS = (
    "DATA_SPLIT_FRACTIONS",
    "DATASET_BIAS_TYPE_LIST",
    "DEMO_BIAS_TYPE_LIST",
    "WEIGHT_ADJUST_LIST",
    "SUBDOMINANCE_PERF_METRICS_LIST",
    "SUBDOMINANCE_FAIR_METRICS_LIST",
)

# List-valued keys that may be left empty. With every entry commented out, YAML
# reads such a key as null rather than as an empty list, so null is read as
# empty.
EMPTYABLE_LIST_KEYS = (
    "DATASET_BIAS_TYPE_LIST",
    "DEMO_BIAS_TYPE_LIST",
)

# An override that is not itself an `exp_info` key: `"<library>/<optimizer>"`,
# e.g. `"optuna/TPE"`, swaps the library and optimizer of every `opt_debias`
# entry of WEIGHT_ADJUST_LIST while keeping the rest of the entry (its
# evaluation budget). This is how a sweep tunes Fairness Reward Weight
# Optimization's weight-debiasing optimizer, which is otherwise buried inside a
# list. `frwo-tune` writes the best one as the WEIGHT_ADJUST_LIST it stands
# for (see `resolve_tuned_values()`).
OPT_DEBIAS_OPTIMIZER = "OPT_DEBIAS_OPTIMIZER"

# Whether Fairness Reward Weight Optimization runs its `opt_debias` weight
# adjustment after its unadjusted run. Each experiment config turns it on or
# off; it never affects what the hyperparameters are tuned on.
RUN_OPT_DEBIAS = "RUN_OPT_DEBIAS"

# What every `exp_info` records as the experiment config it came from.
CONFIG_NAME = "CONFIG_NAME"


@dataclass
class ExperimentPlan:
    """
    Everything one experiment config runs.

    Attributes
    ----------
    config_name : str
        The experiment config's name.
    random_seed : int
        Seeds the global numpy and `random` generators.
    expert_algos : list<str>
        Every selected experiment is run once per expert.
    selected_datasets : list<str>
        The DATASETs of the experiments to run; the rest are skipped.
    experiments : list<dict>
        The `exp_info` of every dataset's experiment, in the base's order,
        lacking only the `EXPERT_ALGO` each run sets.
    """

    config_name: str
    random_seed: int
    expert_algos: list
    selected_datasets: list
    experiments: list


def _read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def discover_configs():
    """
    Every experiment config in `EXPERIMENT_CONFIG_DIR`: the main one first,
    then the rest by name.
    """
    if not MAIN_CONFIG_PATH.exists():
        raise FileNotFoundError(f"The main experiment config {MAIN_CONFIG_PATH} is missing.")
    others = sorted(
        path for path in EXPERIMENT_CONFIG_DIR.glob("*.yaml") if path != MAIN_CONFIG_PATH
    )
    return [MAIN_CONFIG_PATH, *others]


def resolve_config_arg(text):
    """
    The experiment config a `--config` argument names: a path, or the name of
    a file in `EXPERIMENT_CONFIG_DIR` (e.g. `main`).
    """
    path = Path(text)
    if not path.exists() and path.suffix not in (".yaml", ".yml"):
        path = EXPERIMENT_CONFIG_DIR / f"{text}.yaml"
    if not path.exists():
        available = [p.stem for p in discover_configs()]
        raise FileNotFoundError(
            f"No experiment config {text!r}. Available: {available}, or give a path."
        )
    return path


def load_sweep_configs():
    """The contents of every sweep config in `SWEEP_CONFIG_DIR`, by the technique it tunes."""
    sweep_configs = {}
    for path in sorted(SWEEP_CONFIG_DIR.glob("*.yaml")):
        sweep_config = _read_yaml(path)
        algorithm = sweep_config["parameters"]["ALGORITHMS"]["value"][0]
        sweep_configs[algorithm] = sweep_config
    return sweep_configs


def tuned_keys(sweep_config):
    """The parameters a sweep's search chooses: a fixed `value` is not chosen."""
    return [
        key for key, spec in sweep_config["parameters"].items() if "value" not in spec
    ]


def tunable_keys(sweep_configs=None):
    """
    The config keys some sweep tunes, by the technique that tunes them: its
    tuned parameters, with `OPT_DEBIAS_OPTIMIZER` standing for the
    WEIGHT_ADJUST_LIST it is written as.
    """
    sweep_configs = load_sweep_configs() if sweep_configs is None else sweep_configs
    return {
        algorithm: [
            "WEIGHT_ADJUST_LIST" if key == OPT_DEBIAS_OPTIMIZER else key
            for key in tuned_keys(sweep_config)
        ]
        for algorithm, sweep_config in sweep_configs.items()
    }


def load_tuned_hyperparameters(path=TUNED_HYPERPARAMETERS_PATH):
    """
    The contents of the tuned hyperparameters file: by dataset, then by
    technique, the tuned `values` and where they came from. Empty if nothing
    has been tuned yet.
    """
    path = Path(path)
    return _read_yaml(path) if path.exists() else {}


class _TunedHyperparametersDumper(yaml.SafeDumper):
    """Writes lists on one line, e.g. `[[opt_debias, optuna, TPE, 200]]`."""


_TunedHyperparametersDumper.add_representer(
    list,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True
    ),
)


def format_tuned_hyperparameters(tuned):
    """The text of a tuned hyperparameters file holding `tuned`."""
    header = (
        "# The hyperparameters `uv run frwo-tune` found, under the settings of\n"
        "# configs/experiments/main.yaml. Every experiment config runs with them.\n"
        "#\n"
        "# Written by `frwo-tune` (as tuned_hyperparameters.yaml.new, for review),\n"
        "# by dataset and then technique:\n"
        "#   values:        the best values its sweep found. Fairness Reward Weight\n"
        "#                  Optimization's best optimizer is written as the\n"
        "#                  dataset's WEIGHT_ADJUST_LIST, which runs when an\n"
        "#                  experiment config sets RUN_OPT_DEBIAS.\n"
        "#   objective:     their mean validation subdominance.\n"
        "#   sweep:         the W&B sweep that found them.\n"
        "#   tuned_against: a fingerprint of main.yaml's settings for the dataset\n"
        "#                  when they were tuned (see `tuning_fingerprint()` in\n"
        "#                  src/frwo/config.py). `frwo` warns when it no longer\n"
        "#                  matches. null for values tuned before it was recorded.\n"
        "\n"
    )
    body = yaml.dump(
        tuned,
        Dumper=_TunedHyperparametersDumper,
        sort_keys=False,
        default_flow_style=False,
        width=1000,
    )
    return header + body


def tuning_fingerprint(exp_info, tunable=None):
    """
    A fingerprint of the settings of an experiment that hyperparameters tuned
    on it depend on: all of them, except which techniques it runs, which config
    it came from, whether it runs the `opt_debias` weight adjustment (a sweep
    of Fairness Reward Weight Optimization always does) and the tunable
    hyperparameters themselves.
    """
    if tunable is None:
        tunable = tunable_keys()
    excluded = {"ALGORITHMS", CONFIG_NAME, OPT_DEBIAS_OPTIMIZER, RUN_OPT_DEBIAS}
    excluded |= {key for keys in tunable.values() for key in keys}
    settings = {key: value for key, value in exp_info.items() if key not in excluded}
    text = json.dumps(settings, sort_keys=True, default=list)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _check_file_keys(path, data, allowed):
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ValueError(f"{path}: unknown top-level keys {unknown}. Allowed: {list(allowed)}.")


def _tuned_values(tuned, datasets, tunable):
    """
    Check the tuned hyperparameters file, and merge each dataset's values
    across techniques.

    Returns
    -------
    values : dict<str, dict>
        By dataset, every tuned value.
    applied : dict<str, dict<str, str or None>>
        By dataset and technique, the `tuned_against` of each entry.
    """
    values, applied = {}, {}
    for dataset, by_algorithm in tuned.items():
        if dataset not in datasets:
            raise ValueError(
                f"{TUNED_HYPERPARAMETERS_PATH}: {dataset!r} is not a dataset of"
                f" {BASE_CONFIG_PATH}."
            )
        values[dataset], applied[dataset] = {}, {}
        for algorithm, entry in (by_algorithm or {}).items():
            if algorithm not in tunable:
                raise ValueError(
                    f"{TUNED_HYPERPARAMETERS_PATH}: {dataset} has values for"
                    f" {algorithm!r}, which no sweep config tunes."
                )
            entry_values = entry.get("values") or {}
            unknown = sorted(set(entry_values) - set(tunable[algorithm]))
            if unknown:
                raise ValueError(
                    f"{TUNED_HYPERPARAMETERS_PATH}: {dataset} / {algorithm} sets"
                    f" {unknown}, which its sweep config does not tune."
                )
            shared = sorted(set(entry_values) & set(values[dataset]))
            if shared:
                raise ValueError(
                    f"{TUNED_HYPERPARAMETERS_PATH}: {dataset} sets {shared} for"
                    " more than one technique."
                )
            values[dataset] |= copy.deepcopy(entry_values)
            applied[dataset][algorithm] = entry.get("tuned_against")
    return values, applied


def load_config(path=MAIN_CONFIG_PATH, tuned=True):
    """
    Read an experiment config, merged with the base and (if `tuned`) the tuned
    hyperparameters into a resolved config.

    A file that is already a resolved config (one a sweep keeps, see
    `frwo.sweep`) is returned as it is.

    Returns
    -------
    config : dict
        `CONFIG_NAME`, the `TOP_LEVEL_KEYS`, every dataset's settings as
        `experiments`, and as `tuned_hyperparameters` the `tuned_against` of
        every tuned entry it holds (or None, if `tuned` is False).
    """
    path = Path(path)
    data = _read_yaml(path)
    if data.get("resolved"):
        return data

    base = _read_yaml(BASE_CONFIG_PATH)
    _check_file_keys(BASE_CONFIG_PATH, base, BASE_FILE_KEYS)
    _check_file_keys(path, data, EXPERIMENT_FILE_KEYS)
    if "SELECTED_DATASETS" not in data:
        raise ValueError(f"{path}: SELECTED_DATASETS is missing.")

    tunable = tunable_keys()
    tunable_set = {key for keys in tunable.values() for key in keys}
    base_common = base.get("common") or {}
    datasets = base.get("datasets") or {}

    for dataset, own in datasets.items():
        named = sorted(set(own or {}) & set(NAME_KEYS))
        if named:
            raise ValueError(
                f"{BASE_CONFIG_PATH}: {dataset} sets {named}; each dataset's is its key."
            )

    # What an experiment config may set: any setting of the base, except the
    # locked and tunable ones.
    known = set(base_common) | {key for own in datasets.values() for key in (own or {})}
    sections = [("common", data.get("common") or {})]
    for dataset, own in (data.get("experiments") or {}).items():
        if dataset not in datasets:
            raise ValueError(
                f"{path}: experiments has an entry for {dataset!r}, which is not a"
                f" dataset of {BASE_CONFIG_PATH}. Datasets: {list(datasets)}."
            )
        sections.append((f"experiments.{dataset}", own or {}))
    for section, settings in sections:
        for key in settings:
            if key in LOCKED_KEYS:
                raise ValueError(
                    f"{path}: {section} sets {key}, which picks the data splits and"
                    f" seeds and so must stay as {BASE_CONFIG_PATH} sets it."
                )
            if key in tunable_set:
                raise ValueError(
                    f"{path}: {section} sets {key}, a tuned hyperparameter. Every"
                    f" experiment config runs with the values in"
                    f" {TUNED_HYPERPARAMETERS_PATH}."
                )
            if key not in known:
                raise ValueError(
                    f"{path}: {section} sets {key}, which {BASE_CONFIG_PATH} does"
                    " not. Add it to the base's `common` first."
                )

    tuned_values, applied = ({}, {})
    if tuned:
        tuned_values, applied = _tuned_values(
            load_tuned_hyperparameters(), datasets, tunable
        )

    experiments = {}
    for dataset, own in datasets.items():
        settings = {"EXPERIMENT_NAME": dataset, "DATASET": dataset}
        settings |= copy.deepcopy(base_common)
        settings |= copy.deepcopy(own or {})
        settings |= tuned_values.get(dataset, {})
        settings |= copy.deepcopy(data.get("common") or {})
        settings |= copy.deepcopy((data.get("experiments") or {}).get(dataset) or {})
        experiments[dataset] = settings

    return {
        "resolved": True,
        CONFIG_NAME: path.stem,
        "RANDOM_SEED": base.get("RANDOM_SEED"),
        "SELECTED_DATASETS": data["SELECTED_DATASETS"],
        "EXPERT_ALGOS": data.get("EXPERT_ALGOS", base.get("EXPERT_ALGOS")),
        "experiments": experiments,
        "tuned_hyperparameters": applied if tuned else None,
    }


def _to_tuple(value):
    """Turn a list -- and every list inside it -- into a tuple."""
    if isinstance(value, list):
        return tuple(_to_tuple(v) for v in value)
    return value


def parse_assignment(text):
    """
    Parse one `KEY=VALUE` command line assignment.

    The value is read as JSON, so `10`, `1e-5`, `true`, `null` and
    `["COMPAS"]` get their JSON types; anything that is not valid JSON, such as
    `highs-ds`, is kept as a string.
    """
    key, sep, raw = text.partition("=")
    if not sep or not key:
        raise ValueError(f"Expected KEY=VALUE, got {text!r}.")

    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key.strip(), value


def _known_keys(config):
    keys = set(TOP_LEVEL_KEYS) | {OPT_DEBIAS_OPTIMIZER}
    for experiment in config["experiments"].values():
        keys |= set(experiment)
    return keys


def _to_list(value):
    """Turn a tuple -- and every tuple inside it -- into a list."""
    if isinstance(value, (list, tuple)):
        return [_to_list(v) for v in value]
    return value


def _apply_opt_debias_optimizer(weight_adjust_list, optimizer):
    """Swap `optimizer` into every `opt_debias` entry of `weight_adjust_list`."""
    library, sep, name = str(optimizer).partition("/")
    if not sep:
        raise ValueError(
            f"{OPT_DEBIAS_OPTIMIZER} must look like '<library>/<optimizer>',"
            f" e.g. 'optuna/TPE'; got {optimizer!r}."
        )
    if not any(entry[0] == "opt_debias" for entry in weight_adjust_list):
        raise ValueError(
            f"{OPT_DEBIAS_OPTIMIZER} is set, but WEIGHT_ADJUST_LIST has no"
            " 'opt_debias' entry for it to apply to."
        )
    return tuple(
        ("opt_debias", library, name, *entry[3:])
        if entry[0] == "opt_debias"
        else entry
        for entry in weight_adjust_list
    )


def build_experiment_plan(config, overrides=None, annotations=None):
    """
    Resolve a resolved config into the experiments to run.

    Parameters
    ----------
    config : dict
        A resolved config, from `load_config()`.
    overrides : dict<str, object>, optional
        Values that replace the config's, applied last. A key must be one the
        config sets somewhere (or `OPT_DEBIAS_OPTIMIZER`), which catches typos
        -- a misspelled override would otherwise be silently ignored. Applied
        to every experiment.
    annotations : dict<str, object>, optional
        Extra values added to every experiment's `exp_info` -- and so to every
        W&B run's config -- without that check, e.g. which sweep trial launched
        the run.

    Returns
    -------
    plan : ExperimentPlan
    """
    overrides = dict(overrides or {})

    unknown = sorted(set(overrides) - _known_keys(config))
    if unknown:
        raise ValueError(
            f"Unknown config keys in overrides: {unknown}. Overrides must name"
            f" a key the config already sets; add new keys to the `common`"
            f" section of {BASE_CONFIG_PATH} first."
        )

    top_level = {key: config.get(key) for key in TOP_LEVEL_KEYS}
    for key in TOP_LEVEL_KEYS:
        if key in overrides:
            top_level[key] = overrides.pop(key)
    optimizer_override = overrides.pop(OPT_DEBIAS_OPTIMIZER, None)

    experiments = []
    for settings in config["experiments"].values():
        exp_info = copy.deepcopy(settings)
        exp_info[CONFIG_NAME] = config[CONFIG_NAME]
        exp_info["RANDOM_SEED"] = top_level["RANDOM_SEED"]
        exp_info |= copy.deepcopy(overrides)

        for key in EMPTYABLE_LIST_KEYS:
            if key in exp_info and exp_info[key] is None:
                exp_info[key] = []
        for key in TUPLE_KEYS:
            if key in exp_info:
                exp_info[key] = _to_tuple(exp_info[key])
        if optimizer_override is not None:
            exp_info["WEIGHT_ADJUST_LIST"] = _apply_opt_debias_optimizer(
                exp_info["WEIGHT_ADJUST_LIST"], optimizer_override
            )
        exp_info |= annotations or {}

        experiments.append(exp_info)

    datasets = {exp_info["DATASET"] for exp_info in experiments}
    missing = [d for d in top_level["SELECTED_DATASETS"] if d not in datasets]
    if missing:
        raise ValueError(
            f"{config[CONFIG_NAME]}: SELECTED_DATASETS lists {missing}, which is"
            f" not a dataset of {BASE_CONFIG_PATH}. Available: {sorted(datasets)}."
        )

    return ExperimentPlan(
        config_name=config[CONFIG_NAME],
        random_seed=top_level["RANDOM_SEED"],
        expert_algos=list(top_level["EXPERT_ALGOS"]),
        selected_datasets=list(top_level["SELECTED_DATASETS"]),
        experiments=experiments,
    )


def resolve_tuned_values(exp_info, values):
    """
    Translate the values a sweep tuned for one experiment into the values to
    write in the tuned hyperparameters file.

    Every value is written as is, except `OPT_DEBIAS_OPTIMIZER`, which only
    exists as an override: it is written as the experiment's
    `WEIGHT_ADJUST_LIST` with that optimizer swapped in.

    Parameters
    ----------
    exp_info : dict
        The experiment's `exp_info`, from `build_experiment_plan()`.
    values : dict<str, object>
        The tuned values, by sweep parameter name.

    Returns
    -------
    values : dict<str, object>
        The tuned values, by config key.
    """
    return {
        "WEIGHT_ADJUST_LIST" if key == OPT_DEBIAS_OPTIMIZER else key: (
            _to_list(
                _apply_opt_debias_optimizer(exp_info["WEIGHT_ADJUST_LIST"], value)
            )
            if key == OPT_DEBIAS_OPTIMIZER
            else value
        )
        for key, value in values.items()
    }


def check_tuning(config, plan, allow_untuned=False):
    """
    Check that every technique `plan` runs with hyperparameters to tune has
    tuned values for each of its datasets, and warn about values tuned on
    settings of the main experiment config that have changed since.

    A config loaded without the tuned hyperparameters, such as a sweep's, is
    not checked.

    Raises
    ------
    ValueError
        If a technique has no tuned values for a dataset, unless
        `allow_untuned`, which only warns.
    """
    applied = config.get("tuned_hyperparameters")
    if applied is None:
        return

    tunable = tunable_keys()
    selected = [e for e in plan.experiments if e["DATASET"] in plan.selected_datasets]
    untuned = [
        f"{exp_info['DATASET']} / {algorithm}"
        for exp_info in selected
        for algorithm in exp_info["ALGORITHMS"]
        if algorithm in tunable and algorithm not in applied.get(exp_info["DATASET"], {})
    ]
    if untuned:
        message = (
            f"{plan.config_name}: no tuned hyperparameters for {untuned} in"
            f" {TUNED_HYPERPARAMETERS_PATH}"
        )
        if not allow_untuned:
            raise ValueError(
                message + ". Tune them with `uv run frwo-tune`, or run with"
                " `--allow-untuned` to use the defaults of the base config."
            )
        logging.warning(message + "; using the defaults of the base config.")

    fingerprints = _main_fingerprints()
    stale = [
        f"{exp_info['DATASET']} / {algorithm}"
        for exp_info in selected
        for algorithm, tuned_against in applied.get(exp_info["DATASET"], {}).items()
        if algorithm in exp_info["ALGORITHMS"]
        and tuned_against is not None
        and tuned_against != fingerprints.get(exp_info["DATASET"])
    ]
    if stale:
        logging.warning(
            f"{plan.config_name}: the hyperparameters of {stale} were tuned on"
            f" settings of {MAIN_CONFIG_PATH.name} that have changed since. Tune"
            " them again with `uv run frwo-tune` to keep them current."
        )


@functools.cache
def _main_fingerprints():
    """By dataset, the `tuning_fingerprint()` of the main config as it is now."""
    plan = build_experiment_plan(load_config(MAIN_CONFIG_PATH, tuned=False))
    tunable = tunable_keys()
    return {e["DATASET"]: tuning_fingerprint(e, tunable) for e in plan.experiments}
