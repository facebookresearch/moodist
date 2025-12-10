# Copyright (c) Meta Platforms, Inc. and affiliates.

"""C extension loading."""

import importlib

import torch

from .version import __version__, cversions

_clist = [
    "MoodistProcessGroup",
    "MoodistBackend",
    "enable_profiling",
    "enable_cuda_allocator",
    "enable_cpu_allocator",
    "cpu_allocator_debug",
    "cuda_copy",
    "set_prefer_kernel_less",
    "TcpStore",
    "serialize",
    "deserialize",
]

_torchversion = torch.__version__

_found = False
_C = None
for _k, _v in cversions.items():
    if _torchversion.startswith(_k):
        assert not _found, "Moodist matched multiple pytorch versions? %s %s" % (
            _torchversion,
            list(cversions.keys()),
        )
        _C = importlib.import_module(_v, "moodist")
        _found = True

if not _found:
    raise RuntimeError(
        "Moodist was not built for the currently installed pytorch version."
        " Found pytorch %s. Moodist was built for: %s"
        % (_torchversion, list(cversions.keys()))
    )

# Export C extension symbols
MoodistProcessGroup = _C.MoodistProcessGroup
MoodistBackend = _C.MoodistBackend
enable_profiling = _C.enable_profiling
enable_cuda_allocator = _C.enable_cuda_allocator
enable_cpu_allocator = _C.enable_cpu_allocator
cpu_allocator_debug = _C.cpu_allocator_debug
cuda_copy = _C.cuda_copy
set_prefer_kernel_less = _C.set_prefer_kernel_less
TcpStore = _C.TcpStore
serialize = _C.serialize
deserialize = _C.deserialize
