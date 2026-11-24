"""The ``name_substituent`` memo key holds the CIP label of the bonds that join the fragment to the rest.

The double bond of an '-ylidene' prefix is outside the fragment; the prefix cites its E/Z (or leaves it to the
parent) by the label it finds there, so a prefix named before the stereo was assigned must not be served after it
(``ORTHONYM_MEMO=verify``, 2026-11-24: '(2Z)-...-2-ylidene' stored before the assignment, the prefix without '(2Z)'
named after it).
"""
import pytest
from rdkit import Chem

from orthonym.assembly.fragment_key import external_bond_cips
from orthonym.assembly.substituent_enumerator import _substituent_memo_key

pytestmark = pytest.mark.unit


def test_the_joining_double_bond_label_is_in_the_key():
    mol = Chem.MolFromSmiles("C/C=C1/CCCC1")          # ethylidenecyclopentane: the C=C joins the two parts
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    chain = frozenset(set(range(mol.GetNumAtoms())) - ring)
    attach = next(a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() in chain and any(n.GetIdx() in ring for n in a.GetNeighbors()))
    joining = mol.GetBondBetweenAtoms(attach, next(n.GetIdx() for n in mol.GetAtomWithIdx(attach).GetNeighbors()
                                                   if n.GetIdx() in ring))
    before = _substituent_memo_key(mol, chain, attach, False, True)
    assert external_bond_cips(mol, chain) == ((joining.GetIdx(), None),)
    joining.SetProp("_CIPCode", "E")
    after = _substituent_memo_key(mol, chain, attach, False, True)
    assert external_bond_cips(mol, chain) == ((joining.GetIdx(), "E"),)
    assert before is not None and after is not None and before != after
