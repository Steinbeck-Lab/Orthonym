"""``substituent_naming.name_substituent_fragment``: the record calls a memo hit repeats are the distinct fragments.

A hit makes the ``record_non_pin_fragment`` calls the stored call made, into every enclosing record-call log.
The log kept every call, repeats included, so a fragment named twice by each level of a nested branch
(the chain tie-break names every substituent of each tied chain, then the assembly names them again)
doubled the log with each level: 16,777,215 replays for a 24-level branched peptide, 83 of 100 s. The
record only adds a fragment that is not in it yet, so the replay of each distinct fragment once, in order
of first call, leaves the record the replay of every call left.
"""
import pytest
from rdkit import Chem

from orthonym.assembly import memo, substituent_naming as sn
from orthonym.metrics import provenance as pv

pytestmark = pytest.mark.unit

_DEPTH = 22


def _chain_mol():
    return Chem.MolFromSmiles("C" * (_DEPTH + 2))


def _run(monkeypatch, depth=_DEPTH):
    mol = _chain_mol()
    spy = []
    real_record = pv.record_non_pin_fragment

    def counting_record(fragment):
        spy.append(fragment)
        real_record(fragment)
    monkeypatch.setattr(pv, "record_non_pin_fragment", counting_record)

    def fake(mol_, sub_atoms, attach_idx, parent_chain):
        level = attach_idx
        pv.record_non_pin_fragment("frag-%d" % level)
        if level < depth:
            nxt = list(range(level + 1, depth + 2))
            sn.name_substituent_fragment(mol_, nxt, level + 1, [])      # named for the sort key...
            sn.name_substituent_fragment(mol_, nxt, level + 1, [])      #... and again for the assembly
        pv.record_non_pin_fragment("frag-%d" % level)                   # a repeat inside one call
        return "name-%d" % level
    monkeypatch.setattr(sn, "_name_substituent_fragment_uncached", fake)
    pv.clear_provenance()
    scope = memo.push_scope()
    try:
        value = sn.name_substituent_fragment(mol, list(range(0, depth + 2)), 0, [])
        stored = [entry for entry in memo._cache_var.get().values() if isinstance(entry, tuple) and len(entry) == 3]
    finally:
        memo.pop_scope(scope)
    return value, spy, stored, pv.get_provenance()["non_pin_fragments"]


def test_replays_do_not_double_with_each_level(monkeypatch):
    value, calls, stored, fragments = _run(monkeypatch)
    assert value == "name-0"
    assert len(calls) < 50 * _DEPTH ** 2          # 2**22 = 4.2 million before; a few thousand now


def test_each_stored_record_holds_a_fragment_once_in_order_of_first_call(monkeypatch):
    _value, _calls, stored, _fragments = _run(monkeypatch)
    assert stored
    for _mol, _name, records in stored:
        assert len(records) == len(set(records))
    top = max(stored, key=lambda entry: len(entry[2]))[2]
    assert top[0] == "frag-0"


def test_the_record_a_hit_leaves_is_the_record_of_the_first_call(monkeypatch):
    _value, _calls, _stored, fragments = _run(monkeypatch)
    assert fragments == tuple("frag-%d" % level for level in range(_DEPTH + 1))
