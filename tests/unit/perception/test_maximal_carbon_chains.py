"""find_principal_chain enumerates only the carbon paths closed at both ends.

TRIAGE 'Suite fix -- j3-long-alkanes' (g1 C2): the full enumeration scored n^2
paths in O(n) each for an unbranched C_n chain (C145: 25 s), which made
tests/integration/test_behavioral_freeze.py run for hours. A path that can be
extended at an end always scores lower than its extension (chain_score terms 0-1
cannot drop on a superset, term 2 grows), so only closed paths can win; the
chosen chain is unchanged (A/B over the gate, breadth and chain corpora recorded
in TRIAGE).
"""
import pytest
from rdkit import Chem

from orthonym.perception import chains as ch
from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.rules.seniority import get_principal_group

pytestmark = pytest.mark.unit

_SMILES = [
    "C", "CC", "CCCCCC", "CCC(C)CC", "CCC(CC)CC", "CC(C)(C)C(C)(C)C",
    "OCC(CO)(CO)CO", "C=CC(C=C)C=C", "C#CC(C#C)C#C", "CCC(CC)C(CC)CC",
    "OC(=O)CC(CC(=O)O)CC(=O)O", "CCCCCCCCc1ccccc1", "CCCC(CCCC)C1CCCCC1",
    "C1CCCCC1", "C1CC2CCC1C2", "CC1CCC(C)C1", "CCOCC", "CCN(CC)CC",
    "CC(C)CC(CC(C)C)CC(C)C", "CC(=O)OCC(COC(C)=O)OC(C)=O",
]


def _closed_subsequence(mol, exclude):
    """The reference: the full enumeration, filtered to paths closed at both ends."""
    excl = exclude or set()

    def eligible(i):
        return mol.GetAtomWithIdx(i).GetSymbol() == "C" and i not in excl

    def closed(path):
        on = set(path)
        return not any(
            eligible(nb.GetIdx()) and nb.GetIdx() not in on
            for end in (path[0], path[-1])
            for nb in mol.GetAtomWithIdx(end).GetNeighbors()
        )

    return [p for p in ch.find_all_carbon_chains(mol, 1, exclude) if closed(p)]


@pytest.mark.parametrize("smi", _SMILES)
def test_closed_paths_are_the_full_enumeration_filtered_in_order(smi):
    mol = Chem.MolFromSmiles(smi)
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    for exclude in (None, ring_atoms or None):
        assert ch.find_maximal_carbon_chains(mol, exclude) == _closed_subsequence(mol, exclude)


def test_unbranched_chain_has_exactly_its_two_orientations():
    mol = Chem.MolFromSmiles("C" * 300)
    paths = ch.find_maximal_carbon_chains(mol)
    assert paths == [list(range(300)), list(range(299, -1, -1))]


@pytest.mark.parametrize("smi", _SMILES)
def test_principal_chain_equals_the_full_enumeration_choice(smi, monkeypatch):
    mol = Chem.MolFromSmiles(smi)
    fgs = detect_functional_groups(mol)
    pg, _ = get_principal_group(mol, fgs)
    got = ch.find_principal_chain(mol, fgs, pg)
    monkeypatch.setattr(
        ch, "find_maximal_carbon_chains",
        lambda m, exclude_atoms=None: ch.find_all_carbon_chains(m, 1, exclude_atoms))
    assert got == ch.find_principal_chain(mol, fgs, pg)


def test_principal_chain_does_not_enumerate_every_path(monkeypatch):
    def _refuse(*a, **k):
        raise AssertionError("find_principal_chain enumerated every carbon path")
    monkeypatch.setattr(ch, "find_all_carbon_chains", _refuse)
    mol = Chem.MolFromSmiles("C" * 400)
    assert ch.find_principal_chain(mol, {}, None) in (list(range(400)), list(range(399, -1, -1)))
