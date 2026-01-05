# Copyright (c) Meta Platforms, Inc. and affiliates.

"""C extension loading."""

import importlib

import torch

from .version import __version__, cversions

_torchversion = torch.__version__

_C = None
for _k, _v in cversions.items():
    if _torchversion.startswith(_k + ".") or _torchversion == _k:
        if _C is not None:
            raise RuntimeError(
                "Moodist matched multiple pytorch versions: %s matches %s"
                % (_torchversion, list(cversions.keys()))
            )
        _C = importlib.import_module(_v, "moodist")

if _C is None:
    raise RuntimeError(
        "Moodist was not built for the currently installed pytorch version."
        " Found pytorch %s. Moodist was built for: %s"
        % (_torchversion, list(cversions.keys()))
    )

# Export C extension symbols - use getattr with None default for optional symbols
MoodistProcessGroup = getattr(_C, "MoodistProcessGroup", None)
MoodistBackend = getattr(_C, "MoodistBackend", None)
enable_profiling = getattr(_C, "enable_profiling", None)
enable_cuda_allocator = getattr(_C, "enable_cuda_allocator", None)
enable_cpu_allocator = getattr(_C, "enable_cpu_allocator", None)
cpu_allocator_debug = getattr(_C, "cpu_allocator_debug", None)
cuda_copy = getattr(_C, "cuda_copy", None)
set_prefer_kernel_less = getattr(_C, "set_prefer_kernel_less", None)
TcpStore = getattr(_C, "TcpStore", None)

# Serialize functions - exposed through _C which loads libserialize at runtime
serialize = getattr(_C, "serialize", None)
deserialize = getattr(_C, "deserialize", None)
