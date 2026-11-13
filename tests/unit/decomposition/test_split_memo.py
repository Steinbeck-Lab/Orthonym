"""The split verdict memo (``decomposition/split_memo.py``) serves a repeat of one split with what the first call left.

A hit charges no budget unit (it does no work), logs again the cycle hits the first call logged (the fragment named
around the split reads them to keep its name in context), and an unproven split is not served from the memo: its
repeat takes the engine's own path, which fixes the exploration floor of the budget armed now.
"""
from rdkit import Chem

from orthonym.assembly import fragment_naming as fn
from orthonym.assembly import memo
from orthonym.decomposition.engine import _unproven_split_key
from orthonym.decomposition.split_memo import memoised_split

_MOL = Chem.MolFromSmiles("CCOC(C)=O")
_BOND = {"type": "ester", "bond_idx": 2}


def _in_scope(body):
    token = memo.push_scope()
    saved = (getattr(fn._fragment_guard, "cycle_hits", None), getattr(fn._fragment_guard, "visited", None))
    fn._fragment_guard.cycle_hits, fn._fragment_guard.visited = [], set()
    try:
        return body()
    finally:
        fn._fragment_guard.cycle_hits, fn._fragment_guard.visited = saved
        memo.pop_scope(token)


def test_a_hit_replays_the_name_and_the_cycle_hits_without_a_second_call():
    calls = []

    def fresh():
        calls.append(1)
        fn._note_cycle_hit("CCO")
        return "ethyl acetate"

    def body():
        first = memoised_split(_MOL, _BOND, "pin", fresh)
        fn._fragment_guard.cycle_hits = []
        second = memoised_split(_MOL, _BOND, "pin", fresh)
        return first, second, list(fn._fragment_guard.cycle_hits)

    first, second, hits = _in_scope(body)
    if memo._MODE != "on":
        return
    assert (first, second, len(calls)) == ("ethyl acetate", "ethyl acetate", 1)
    assert hits == ["CCO"]


def test_an_unproven_split_is_asked_again_through_the_engine_path():
    calls = []

    def fresh():
        calls.append(1)
        return None

    def body():
        memoised_split(_MOL, _BOND, "pin", fresh)
        memo.side_put("unproven_split", _unproven_split_key(_MOL, _BOND, "pin"), True)
        memoised_split(_MOL, _BOND, "pin", fresh)

    _in_scope(body)
    if memo._MODE == "on":
        assert len(calls) == 2
