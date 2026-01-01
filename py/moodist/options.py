# Copyright (c) Meta Platforms, Inc. and affiliates.

"""MoodistOptions: Per-ProcessGroup options with attribute access and context manager support."""

from typing import Optional


class MoodistOptionsContext:
    """Context manager for temporarily overriding ProcessGroup options.

    Restores all overridden options to their original values on exit.
    Returns the ProcessGroup when used with 'with' statement.
    """

    def __init__(self, pg, overrides: dict):
        self._pg = pg
        self._overrides = overrides
        self._saved = {}

    def __enter__(self):
        for key, value in self._overrides.items():
            self._saved[key] = self._pg._get_option(key)
            self._pg._set_option(key, value)
        return self._pg  # Return the PG so 'with ... as pg' works

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self._saved.items():
            self._pg._set_option(key, value)
        return False


class MoodistOptions:
    """Per-ProcessGroup options with attribute access and context manager support.

    Usage:
        pg = moodist.MoodistProcessGroup(store, rank, size)

        # Direct attribute access:
        pg.options().prefer_kernel_less = True
        pg.options().num_chunks = 4

        # Read options:
        if pg.options().prefer_kernel_less:
            ...

        # Context manager for temporary overrides:
        with pg.options(prefer_kernel_less=True, num_chunks=2):
            pg.allgather(...)
        # Original values restored after the with block
    """

    # Map Python names to C++ option names and their types
    # bool options: 0/1 in C++
    # int options: -1 = auto, >=0 = value
    _OPTION_INFO = {
        "prefer_kernel_less": ("prefer_kernel_less", "bool"),
        "force_kernel_less": ("force_kernel_less", "bool"),
        "num_chunks": ("num_chunks", "int"),
        "chunk_size": ("chunk_size", "int"),
        "method": ("method", "int"),
    }

    def __init__(self, pg):
        # Use object.__setattr__ to avoid triggering our custom __setattr__
        object.__setattr__(self, "_pg", pg)

    def __call__(self, **kwargs) -> MoodistOptionsContext:
        """Return a context manager for temporarily overriding options.

        Example:
            with pg.options(prefer_kernel_less=True, num_chunks=4):
                pg.allgather(...)
        """
        # Convert kwargs to C++ format
        converted = {}
        for key, value in kwargs.items():
            if key not in self._OPTION_INFO:
                raise ValueError(f"Unknown option: {key}")
            cpp_name, opt_type = self._OPTION_INFO[key]
            if opt_type == "bool":
                converted[cpp_name] = 1 if value else 0
            elif value is None:
                converted[cpp_name] = -1  # auto
            else:
                converted[cpp_name] = int(value)
        return MoodistOptionsContext(self._pg, converted)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        if name not in self._OPTION_INFO:
            raise AttributeError(f"Unknown option: {name}")
        cpp_name, opt_type = self._OPTION_INFO[name]
        value = self._pg._get_option(cpp_name)
        if opt_type == "bool":
            return value != 0
        elif value == -1:
            return None  # auto
        return value

    def __setattr__(self, name: str, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name not in self._OPTION_INFO:
            raise AttributeError(f"Unknown option: {name}")
        cpp_name, opt_type = self._OPTION_INFO[name]
        if opt_type == "bool":
            self._pg._set_option(cpp_name, 1 if value else 0)
        elif value is None:
            self._pg._set_option(cpp_name, -1)  # auto
        else:
            self._pg._set_option(cpp_name, int(value))
