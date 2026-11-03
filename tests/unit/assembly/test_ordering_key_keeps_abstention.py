"""The speculative substituent naming that only builds a sort key leaves the caller's abstention slot as it was.

``substituent_enumerator._name_substituent_for_ordering_uncached`` names a fragment to order prefixes; a
decline inside it used to overwrite the abstention the caller had recorded, so a census read the sort key's reason.
"""
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import _name_substituent_for_ordering_uncached
from orthonym.metrics.abstention import peek_abstention, restore_abstention


def test_the_slot_is_restored():
    sentinel = object()
    restore_abstention(sentinel)
    mol = Chem.MolFromSmiles("[Xe]Cc1ccccc1")       # a fragment the namer declines, recording an abstention
    _name_substituent_for_ordering_uncached(mol, [0, 1], 1)
    assert peek_abstention() is sentinel
    restore_abstention(None)
