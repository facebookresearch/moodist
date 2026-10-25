"""
Tests for moodist.get_core_api_ptr, the CoreApi handoff to separately built C++ extensions.

Extensions pass the moodist API version they were compiled against (moodist::kMoodistApiVersion in
api/moodist_version.h); moodist refuses to hand out its CoreApi if the layout version differs.
"""

import moodist
from framework import TestContext, test

# Must equal moodist::kMoodistApiVersion in api/moodist_version.h.
CORE_API_VERSION = 3


@test
def test_core_api_ptr_unchecked(ctx: TestContext):
    """Without a version, the pointer is returned unchecked."""
    assert moodist.get_core_api_ptr() != 0


@test
def test_core_api_ptr_matching_version(ctx: TestContext):
    """The matching version returns the same pointer."""
    assert moodist.get_core_api_ptr(CORE_API_VERSION) == moodist.get_core_api_ptr()
    assert moodist.get_core_api_ptr(None) == moodist.get_core_api_ptr()


@test
def test_core_api_ptr_version_mismatch(ctx: TestContext):
    """Any other version raises instead of returning a pointer to a mismatched struct."""
    for version in (0, CORE_API_VERSION - 1, CORE_API_VERSION + 1):
        try:
            moodist.get_core_api_ptr(version)
        except RuntimeError as e:
            assert "version mismatch" in str(e), str(e)
        else:
            raise AssertionError(f"get_core_api_ptr({version}) did not raise")


@test
def test_core_api_ptr_bad_argument(ctx: TestContext):
    """A non-integer version is a TypeError."""
    try:
        moodist.get_core_api_ptr("3")
    except TypeError:
        pass
    else:
        raise AssertionError("get_core_api_ptr('3') did not raise TypeError")
