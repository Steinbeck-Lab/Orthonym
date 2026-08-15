# tests/unit/assembly/test_offer_pool.py
"""v33 Phase 0 Task L2.1: whole-molecule `Offer` + `rank_offers`/`select_offer`.

Pure data structure -- no OPSIN, no rdkit, no namer import. `rank_offers`
sorts ASCENDING by (0 if complete else 1, 0 if is_pin else 1,
_TIER_RANK[tier], source, name) so a complete + PIN + lower-tier + earlier
name/source always sorts first; `select_offer` is the head of that order
(or None on an empty pool).
"""
import pytest

from orthonym.assembly.offer_pool import Offer, rank_offers, select_offer

pytestmark = pytest.mark.unit


def _o(name, is_pin, tier, complete, source="x"):
    return Offer(name=name, result_obj=None, is_pin=is_pin, tier=tier,
                 source=source, complete=complete)


def test_complete_pin_beats_incomplete_and_nonpin():
    a = _o("ethanol", True, "T1", True)
    b = _o("hydroxyethane", False, "T3", True)
    c = _o("partial", True, "T1", False)
    assert select_offer([c, b, a]).name == "ethanol"     # complete + PIN wins
    assert rank_offers([c, b, a])[-1].name == "partial"  # incomplete sorts last


def test_deterministic_and_empty():
    assert select_offer([]) is None
    x = _o("aaa", True, "T1", True)
    y = _o("bbb", True, "T1", True)
    assert [o.name for o in rank_offers([y, x])] == ["aaa", "bbb"]  # name tiebreak, stable
