__all__ = ["main"]


def __getattr__(name):
    # Imported on first use rather than here: the experiment script pulls in
    # PyTorch and every model library, close to 1 GB, which lightweight
    # submodules such as `frwo.config` and `frwo.sweep` would otherwise
    # load too.
    if name == "main":
        from frwo.FRWO_Experiments import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
