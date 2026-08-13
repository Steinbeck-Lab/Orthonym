"""Phase 1 B5 general lever: isolated_naming_session gives a nested naming a
fresh depth-0 recursion budget, then restores the enclosing session.

Root cause it fixes: the recovery-lane T4 producer (name_t4_complete) is a fresh
whole-molecule naming, but the lane invokes it mid-name() at session_depth >= 1,
so its recursion hits MAX_NAMING_DEPTH prematurely and degrades to an abstention
(measured: it names 8 in-scope suppressed rows from a clean depth-0 session but
abstains at depth >= 1).
"""
import pytest

from orthonym.assembly import fragment_naming as fn

pytestmark = pytest.mark.unit


def test_isolated_session_zeroes_depth_inside_and_restores_after():
    fn._fragment_guard.session_depth = 3
    fn._fragment_guard.cache = {"parent": "value"}
    fn._fragment_guard.visited = {"SMILES"}
    try:
        with fn.isolated_naming_session():
            assert fn._session_depth() == 0
            # a fresh nested session may build its own state...
            fn._fragment_guard.session_depth = 2
            fn._fragment_guard.cache = {"nested": "x"}
        # ...but the enclosing session is restored exactly on exit.
        assert fn._fragment_guard.session_depth == 3
        assert fn._fragment_guard.cache == {"parent": "value"}
        assert fn._fragment_guard.visited == {"SMILES"}
    finally:
        fn._fragment_guard.session_depth = 0
        fn._fragment_guard.cache = None
        fn._fragment_guard.visited = set()


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
