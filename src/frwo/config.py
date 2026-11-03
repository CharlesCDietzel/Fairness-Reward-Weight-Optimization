"""
Loading of the experiment configuration file,
configs/experiments/experiment.yaml.

The file holds everything the experiment script used to set up in Python: the
`exp_info` values shared by every experiment (`common`), each experiment's own
values (`experiments`), and which experiments and experts a run covers. See the
header of configs/experiments/experiment.yaml for how they are merged.

`build_experiment_plan()` turns the file into exactly the `exp_info` dicts the
experiment code has always received, and applies `--set KEY=VALUE` overrides on
top, which is also how a W&B sweep sets the hyperparameters it tunes.

`set_experiment_values()` goes the other way: it writes tuned values into the
text of the file, leaving its comments and everything else as they are.
"""

import copy
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "experiments"
    / "experiment.yaml"
)

# The top-level config values: they configure the run as a whole rather than
# one experiment, but are overridden with `--set` like any `exp_info` value.
TOP_LEVEL_KEYS = ("RANDOM_SEED", "SELECTED_DATASETS", "EXPERT_ALGOS")

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
    "WEIGHT_ADJUST_LIST",
)

# An override that is not itself an `exp_info` key: `"<library>/<optimizer>"`,
# e.g. `"optuna/TPE"`, swaps the library and optimizer of every `opt_debias`
# entry of WEIGHT_ADJUST_LIST while keeping the rest of the entry (its
# evaluation budget). This is how a sweep tunes Fairness Reward Weight
# Optimization's weight-debiasing optimizer, which is otherwise buried inside a
# list.
OPT_DEBIAS_OPTIMIZER = "OPT_DEBIAS_OPTIMIZER"


@dataclass
class ExperimentPlan:
    """
    Everything one execution of the experiment script runs.

    Attributes
    ----------
    random_seed : int
        Seeds the global numpy and `random` generators.
    expert_algos : list<str>
        Every selected experiment is run once per expert.
    selected_datasets : list<str>
        The DATASETs of the experiments to run; the rest are skipped.
    experiments : list<dict>
        The `exp_info` of every experiment in the config file, in file order,
        lacking only the `EXPERT_ALGO` each run sets.
    """

    random_seed: int
    expert_algos: list
    selected_datasets: list
    experiments: list


def load_config(path=None):
    """Read the experiment configuration file (default: `DEFAULT_CONFIG_PATH`)."""
    with open(path or DEFAULT_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _to_tuple(value):
    """Turn a list -- and every list inside it -- into a tuple."""
    if isinstance(value, list):
        return tuple(_to_tuple(v) for v in value)
    return value


def _to_list(value):
    """Turn a tuple -- and every tuple inside it -- into a list."""
    if isinstance(value, (list, tuple)):
        return [_to_list(v) for v in value]
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
    keys |= set(config.get("common") or {})
    for experiment in (config.get("experiments") or {}).values():
        keys |= set(experiment or {})
    return keys


def _apply_opt_debias_optimizer(weight_adjust_list, optimizer):
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
    Resolve a loaded config file into the experiments to run.

    Parameters
    ----------
    config : dict
        The contents of the config file, from `load_config()`.
    overrides : dict<str, object>, optional
        Values that replace the file's, applied last. A key must be one the
        file sets somewhere (or `OPT_DEBIAS_OPTIMIZER`), which catches typos
        -- a misspelled override would otherwise be silently ignored. Applied
        to every experiment, including over its own values.
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
            " a key the config file already sets; add new keys to its"
            " `common` section first."
        )

    top_level = {key: config.get(key) for key in TOP_LEVEL_KEYS}
    for key in TOP_LEVEL_KEYS:
        if key in overrides:
            top_level[key] = overrides.pop(key)
    optimizer = overrides.pop(OPT_DEBIAS_OPTIMIZER, None)

    experiments = []
    for name, own in (config.get("experiments") or {}).items():
        own = copy.deepcopy(own or {})
        exp_info = {"EXPERIMENT_NAME": name}
        exp_info |= copy.deepcopy(config.get("common") or {})
        exp_info |= own
        exp_info.setdefault("DATASET", name)
        exp_info["RANDOM_SEED"] = top_level["RANDOM_SEED"]
        exp_info |= copy.deepcopy(overrides)

        for key in EMPTYABLE_LIST_KEYS:
            if key in exp_info and exp_info[key] is None:
                exp_info[key] = []
        for key in TUPLE_KEYS:
            if key in exp_info:
                exp_info[key] = _to_tuple(exp_info[key])
        if optimizer is not None:
            exp_info["WEIGHT_ADJUST_LIST"] = _apply_opt_debias_optimizer(
                exp_info["WEIGHT_ADJUST_LIST"], optimizer
            )
        exp_info |= annotations or {}

        experiments.append(exp_info)

    datasets = {exp_info["DATASET"] for exp_info in experiments}
    missing = [d for d in top_level["SELECTED_DATASETS"] if d not in datasets]
    if missing:
        raise ValueError(
            f"SELECTED_DATASETS lists {missing}, which no experiment in the"
            f" config file runs. Available: {sorted(datasets)}."
        )

    return ExperimentPlan(
        random_seed=top_level["RANDOM_SEED"],
        expert_algos=list(top_level["EXPERT_ALGOS"]),
        selected_datasets=list(top_level["SELECTED_DATASETS"]),
        experiments=experiments,
    )


def resolve_tuned_values(exp_info, values):
    """
    Translate the values a sweep tuned for one experiment into the values to
    write in the config file.

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
        The tuned values, by config file key.
    """
    values = dict(values)
    optimizer = values.pop(OPT_DEBIAS_OPTIMIZER, None)
    if optimizer is not None:
        values["WEIGHT_ADJUST_LIST"] = _to_list(
            _apply_opt_debias_optimizer(exp_info["WEIGHT_ADJUST_LIST"], optimizer)
        )
    return values


def _format_yaml_value(value):
    """`value` as the single line of YAML that reads back as exactly `value`."""
    text = yaml.safe_dump(value, default_flow_style=True, width=sys.maxsize)
    # A lone scalar is dumped as a document, closed with an end marker.
    return text.removesuffix("...\n").strip()


def set_experiment_values(text, experiment, values):
    """
    Set `values` in the entry of `experiment` under `experiments`, in the text
    of a config file. Keys the entry already sets are changed in place, and any
    others are added at the end of the entry. Nothing else in the text changes.

    Parameters
    ----------
    text : str
        The contents of the config file.
    experiment : str
        The EXPERIMENT_NAME of the entry to change.
    values : dict<str, object>
        The config file keys and values to set.

    Returns
    -------
    text : str
        The new contents of the config file.

    Raises
    ------
    ValueError
        If the entry can't be found, or the result would differ from `text` in
        anything but `values`.
    """
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    entry_pattern = re.compile(rf"  {re.escape(experiment)}:[ \t]*(#.*)?\n")
    try:
        start = lines.index("experiments:\n")
        entry = next(
            i for i in range(start, len(lines)) if entry_pattern.fullmatch(lines[i])
        )
    except (ValueError, StopIteration):
        raise ValueError(f"No entry for {experiment!r} under `experiments`.") from None

    # The entry's lines are the ones indented past its key, and it ends at its
    # last such line: blank lines and comments after that belong to the next
    # entry.
    end = entry + 1
    for i in range(entry + 1, len(lines)):
        if lines[i].startswith("    "):
            end = i + 1
        elif lines[i].strip():
            break

    for key, value in values.items():
        formatted = _format_yaml_value(value)
        key_pattern = re.compile(rf"(    {re.escape(key)}:[ \t]*)(.*?)([ \t]+#.*)?\n")
        for i in range(entry + 1, end):
            match = key_pattern.fullmatch(lines[i])
            if match:
                # A value in block style continues on more indented lines.
                last = i + 1
                while last < end and lines[last].startswith("     "):
                    last += 1
                comment = match.group(3) or ""
                lines[i:last] = [f"    {key}: {formatted}{comment}\n"]
                end -= last - i - 1
                break
        else:
            lines.insert(end, f"    {key}: {formatted}\n")
            end += 1

    new_text = "".join(lines)

    expected = yaml.safe_load(text)
    expected["experiments"][experiment] = (
        expected["experiments"][experiment] or {}
    ) | values
    if yaml.safe_load(new_text) != expected:
        raise ValueError(
            f"Could not write {sorted(values)} into the entry of {experiment!r}"
            " without changing anything else in the config file."
        )
    return new_text
