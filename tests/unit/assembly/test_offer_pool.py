# tests/unit/assembly/test_offer_pool.py
""" a phase Task L2.1/L4-core: whole-molecule `Offer` + rank/select.

Pure data structure -- no OPSIN, no rdkit, no namer import. `rank_offers`
sorts ASCENDING by (0 if complete else 1, 0 if is_pin else 1,
_TIER_RANK[tier], source, name) so a complete + PIN + lower-tier + earlier
name/source always sorts first; `select_offer` is the head of that order
(or None on an empty pool). `select_rt_passing` (L4-core) additionally
requires an INJECTED `rt_ok(offer)` predicate to return True -- the RT-GATE-
OVER-OFFERS safety net: an added offer can never win without passing it.
"""
import pytest

from orthonym.assembly.offer_pool import (
    Offer, rank_offers, select_offer, select_rt_passing)

pytestmark = pytest.mark.unit


def _o(name, is_pin, tier, complete, source="x"):
    return Offer(name=name, result_obj=None, is_pin=is_pin, tier=tier,
                 source=source, complete=complete)


def test_complete_pin_beats_incomplete_and_nonpin():
    a = _o("ethanol", True, "pin_verified", True)
    b = _o("hydroxyethane", False, "systematic_verified", True)
    c = _o("partial", True, "pin_verified", False)
    assert select_offer([c, b, a]).name == "ethanol"     # complete + PIN wins
    assert rank_offers([c, b, a])[-1].name == "partial"  # incomplete sorts last


def test_deterministic_and_empty():
    assert select_offer([]) is None
    x = _o("aaa", True, "pin_verified", True)
    y = _o("bbb", True, "pin_verified", True)
    assert [o.name for o in rank_offers([y, x])] == ["aaa", "bbb"]  # name tiebreak, stable


class TestSelectRtPassing:
    """L4-core: `select_rt_passing(offers, rt_ok)` -- the FIRST offer, in
    `rank_offers` order, that is both `.complete` and `rt_ok(offer)`."""

    def test_top_ranked_fails_rt_ok_second_complete_wins(self):
        top = _o("wrong-molecule-name", True, "pin_verified", True)     # ranks first
        second = _o("hydroxyethane", False, "systematic_verified", True)        # ranks second
        rt_ok = lambda o: o.name != "wrong-molecule-name"
        assert rank_offers([top, second])[0] is top             # sanity: rank unchanged
        assert select_rt_passing([top, second], rt_ok) is second

    def test_single_offer_rt_ok_true_returns_it(self):
        only = _o("ethanol", True, "pin_verified", True)
        assert select_rt_passing([only], lambda o: True) is only

    def test_single_offer_rt_ok_false_returns_none(self):
        only = _o("ethanol", True, "pin_verified", True)
        assert select_rt_passing([only], lambda o: False) is None

    def test_all_fail_rt_ok_returns_none(self):
        a = _o("aaa", True, "pin_verified", True)
        b = _o("bbb", False, "systematic_verified", True)
        assert select_rt_passing([a, b], lambda o: False) is None

    def test_incomplete_offer_never_wins_even_if_rt_ok_true(self):
        incomplete = _o("partial", True, "pin_verified", False)
        complete = _o("hydroxyethane", False, "systematic_verified", True)
        assert select_rt_passing(
            [incomplete, complete], lambda o: True) is complete

    def test_empty_pool_returns_none(self):
        assert select_rt_passing([], lambda o: True) is None

    def test_deterministic_order_first_passing_in_rank_wins(self):
        # Two offers tie on complete/is_pin/tier -- source/name break the tie
        # (rank_offers is TOTAL), and select_rt_passing must walk that exact
        # order, not pool-insertion order.
        x = _o("aaa", True, "pin_verified", True)
        y = _o("bbb", True, "pin_verified", True)
        rt_ok = lambda o: o.name == "bbb"
        assert select_rt_passing([y, x], rt_ok) is y   # x (ranked first) fails, y wins
