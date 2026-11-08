"""
Pausing and resuming the experiment script between trials.

`uv run frwo-pause` asks the running experiment (`uv run frwo`) to
pause. The experiment checks for the request each time it finishes a trial.
When it finds one, it writes a checkpoint and exits. `uv run frwo-resume`
(see `frwo.FRWO_Experiments.resume_main`) then continues
from that checkpoint with the next trial.

Resuming gives exactly the results an uninterrupted run would have. The
experiment's only randomness that carries over from one trial to the next is
the global numpy and `random` generators (every other generator is seeded
afresh for each trial, from values that depend only on the config). So the
checkpoint holds those generators' states, along with everything else needed to
rebuild the run:

* every experiment config it runs, resolved (see `frwo.config.load_config`),
  and the `--set`/`--annotate` values, so that editing the config files while
  the experiment is paused changes nothing,
* the W&B session id, so that the resumed runs join the same session,
* which trial was the last to finish, and of which config,
* the generators' states when that trial's experiment started, which its
  dataset is regenerated from,
* the generators' states when that trial finished,
* the run results collected so far for `--results-file`.

Everything lives in `STATE_DIR`:

* `running.json`: which process is running a pausable experiment.
* `pause_request.json`: a pending pause request, naming the process it is for.
* `checkpoint.json`: the paused experiment.

Only one experiment can be paused at a time. This module imports nothing heavy,
so `frwo-pause` starts instantly.
"""

import argparse
import datetime
import json
import logging
import os
import random
import sys
from pathlib import Path

STATE_DIR = Path(
    os.environ.get(
        "FRWO_STATE_DIR",
        Path(__file__).resolve().parents[2] / ".experiment_state",
    )
)
RUNNING_FILE = STATE_DIR / "running.json"
PAUSE_REQUEST_FILE = STATE_DIR / "pause_request.json"
CHECKPOINT_FILE = STATE_DIR / "checkpoint.json"

CHECKPOINT_VERSION = 2


def read_json(path):
    """The contents of the JSON file `path`, or None if there is none."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def write_json(path, data):
    """Write `data` to `path` atomically, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def remove_file(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def is_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running_experiment():
    """
    The `running.json` record of the pausable experiment that is running, or
    None if there is none. A record left behind by a process that died without
    cleaning up counts as none.
    """
    record = read_json(RUNNING_FILE)
    if record is None or not is_alive(record["pid"]):
        return None
    return record


def capture_rng_state():
    """The global numpy and `random` generators' states, as JSON-safe data."""
    import numpy as np

    name, keys, pos, has_gauss, cached_gaussian = np.random.get_state()
    version, internal, gauss_next = random.getstate()
    return {
        "numpy": [name, keys.tolist(), pos, has_gauss, cached_gaussian],
        "random": [version, list(internal), gauss_next],
    }


def restore_rng_state(state):
    """Set the global numpy and `random` generators to a captured state."""
    import numpy as np

    name, keys, pos, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state(
        (name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached_gaussian)
    )
    version, internal, gauss_next = state["random"]
    random.setstate((version, tuple(internal), gauss_next))


def load_checkpoint():
    """The paused experiment's checkpoint, or None if none is paused."""
    checkpoint = read_json(CHECKPOINT_FILE)
    if checkpoint is not None and checkpoint["version"] != CHECKPOINT_VERSION:
        raise RuntimeError(
            f"{CHECKPOINT_FILE} was written by an incompatible version of this"
            " code and cannot be resumed."
        )
    return checkpoint


def save_checkpoint(checkpoint):
    write_json(CHECKPOINT_FILE, {"version": CHECKPOINT_VERSION, **checkpoint})


def discard_checkpoint():
    remove_file(CHECKPOINT_FILE)


class ExperimentPaused(Exception):
    """Raised once the checkpoint is written, to stop the experiment."""


class PauseListener:
    """
    Registers this process as the running pausable experiment for as long as
    the `with` block lasts, and tells it whether a pause was requested.

    Parameters
    ----------
    session_id : str
        The W&B session the experiment reports to.
    """

    def __init__(self, session_id):
        self.session_id = session_id
        self.enabled = False

    def __enter__(self):
        other = running_experiment()
        if other is not None:
            logging.warning(
                f"Another experiment (pid {other['pid']}) is already running"
                " and can be paused, so this one cannot be."
            )
            return self
        paused = load_checkpoint()
        if paused is not None and paused["session_id"] != self.session_id:
            logging.warning(
                f"An experiment (W&B session {paused['session_id']}) is"
                " already paused, so this one cannot be. Resume it with `uv"
                " run frwo-resume`, or discard it with `uv run"
                " frwo-resume --discard`."
            )
            return self
        # A request left over from a process that died is not for this one.
        remove_file(PAUSE_REQUEST_FILE)
        write_json(
            RUNNING_FILE,
            {
                "pid": os.getpid(),
                "session_id": self.session_id,
                "started": datetime.datetime.now().isoformat(timespec="seconds"),
            },
        )
        self.enabled = True
        return self

    def __exit__(self, *exc):
        if self.enabled:
            remove_file(PAUSE_REQUEST_FILE)
            remove_file(RUNNING_FILE)
        return False

    def pause_requested(self):
        if not self.enabled:
            return False
        request = read_json(PAUSE_REQUEST_FILE)
        return request is not None and request["pid"] == os.getpid()


def pause_main(argv=None):
    """The `frwo-pause` command."""
    parser = argparse.ArgumentParser(
        description=(
            "Pause the running experiment once its current trial finishes."
            " Resume it with `uv run frwo-resume`."
        )
    )
    parser.add_argument(
        "--cancel",
        action="store_true",
        help="Withdraw a pause request the experiment has not acted on yet.",
    )
    args = parser.parse_args(argv)

    running = running_experiment()
    if running is None:
        message = "No pausable experiment is running."
        paused = load_checkpoint()
        if paused is not None:
            message += (
                f" An experiment (W&B session {paused['session_id']}) is already"
                " paused, and one started while it is paused cannot be paused."
            )
        print(message, file=sys.stderr)
        sys.exit(1)
    description = (
        f"experiment (pid {running['pid']}, W&B session {running['session_id']})"
    )

    if args.cancel:
        if read_json(PAUSE_REQUEST_FILE) is None:
            print(f"No pause of the {description} is pending.")
        else:
            remove_file(PAUSE_REQUEST_FILE)
            print(f"Withdrew the pause request of the {description}.")
        return

    write_json(
        PAUSE_REQUEST_FILE,
        {
            "pid": running["pid"],
            "requested": datetime.datetime.now().isoformat(timespec="seconds"),
        },
    )
    print(
        f"The {description} will pause once its current trial finishes."
        " Resume it with `uv run frwo-resume`."
    )


if __name__ == "__main__":
    pause_main()
