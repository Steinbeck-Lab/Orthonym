"""a phase B5 general lever: isolated_naming_session gives a nested naming a
fresh depth-0 recursion budget, then restores the enclosing session.

Root cause it fixes: the recovery-lane producer (name_t4_complete) is a fresh
whole-molecule naming, but the lane invokes it mid-name at session_depth >= 1,
so its recursion hits MAX_NAMING_DEPTH prematurely and degrades to an abstention
(measured: it names 8 in-scope suppressed rows from a clean depth-0 session but
abstains at depth >= 1).
"""
import pytest

from orthonym.assembly import fragment_naming as fn

pytestmark = pytest.mark.unit


def test_isolated_session_zeroes_depth_inside_and_restores_after():
    # giant-molecule hang fix updated this contract: isolation still zeroes and
    # restores the DEPTH budget (session_depth + visited), but the fragment memo
    # CACHE now PERSISTS across the boundary (it is a context-free pure function and
    # is owned by the whole-molecule name scope). Resetting/restoring the cache was
    # the bug that made a giant re-explore the same fragment thousands of times.
    fn._fragment_guard.name_call_depth = 0
    fn._fragment_guard.session_depth = 3
    fn._fragment_guard.cache = {"parent": "value"}
    fn._fragment_guard.visited = {"SMILES"}
    try:
        with fn.isolated_naming_session():
            assert fn._session_depth() == 0
            # the live cache is still visible inside (not zeroed)...
            assert fn._fragment_guard.cache == {"parent": "value"}
            #...and a name discovered inside is added to that same live cache.
            fn._fragment_guard.session_depth = 2
            fn._fragment_guard.cache["nested"] = "x"
        # depth budget restored exactly...
        assert fn._fragment_guard.session_depth == 3
        assert fn._fragment_guard.visited == {"SMILES"}
        #...but the cache PERSISTS with both entries (memo survives the molecule).
        assert fn._fragment_guard.cache == {"parent": "value", "nested": "x"}
    finally:
        fn._fragment_guard.session_depth = 0
        fn._fragment_guard.cache = None
        fn._fragment_guard.visited = set()
        fn._fragment_guard.name_call_depth = 0


def test_isolated_session_restores_even_on_exception():
    fn._fragment_guard.session_depth = 5
    try:
        with pytest.raises(ValueError):
            with fn.isolated_naming_session():
                assert fn._session_depth() == 0
                raise ValueError("boom")
        assert fn._fragment_guard.session_depth == 5
    finally:
        fn._fragment_guard.session_depth = 0
        fn._fragment_guard.cache = None
        fn._fragment_guard.visited = set()
