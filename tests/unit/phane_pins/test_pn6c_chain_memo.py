"""The ring systems and the amplificant hydrides of the last molecule are kept (``rules.phane_pin.amplificants``,
``validation.spelling.linear_phane``).

A molecule whose linear phane has many tied chains (the polyphenylene dendrimer of GitHub issue 121, C222H150, 37
benzene rings: 720 chains of seven rings) asked for the ring systems and for the hydride of each ring of each chain
again, 17 s of the 46 s of its phane name. The answers are pure functions of the molecule, so a second call for the same
molecule object is a lookup, and another molecule object starts afresh.
"""
import pytest
from rdkit import Chem

from orthonym.rules.phane_pin import amplificants
from orthonym.validation.spelling import linear_phane

pytestmark = pytest.mark.unit

TERPHENYL = Chem.MolFromSmiles("c1ccc(cc1)-c1ccc(cc1)-c1ccccc1")


def test_the_ring_systems_of_a_molecule_are_perceived_once(monkeypatch):
    real, calls = linear_phane._ring_systems, []
    monkeypatch.setattr(linear_phane, "_ring_systems", lambda mol: calls.append(mol) or real(mol))
    mol = Chem.Mol(TERPHENYL)
    first = [linear_phane.ring_system_atoms(mol, k) for k in range(3)]
    again = [linear_phane.ring_system_atoms(mol, k) for k in range(3)]
    assert first == again and len(calls) == 1
    linear_phane.ring_system_atoms(Chem.Mol(TERPHENYL), 0)         # another molecule object: perceived afresh
    assert len(calls) == 2


def test_the_hydride_of_a_ring_system_is_built_once_for_a_molecule(monkeypatch):
    calls = []
    monkeypatch.setattr(amplificants, "_amplificant_hydride", lambda mol, atoms: calls.append(atoms) or ("hydride", atoms))
    mol = Chem.Mol(TERPHENYL)
    ring = frozenset(range(6))
    assert amplificants.amplificant_hydride(mol, ring) == ("hydride", ring)
    assert amplificants.amplificant_hydride(mol, ring) == ("hydride", ring)
    assert len(calls) == 1
    amplificants.amplificant_hydride(mol, frozenset(range(6, 12)))   # another ring system of the molecule
    assert len(calls) == 2
    amplificants.amplificant_hydride(Chem.Mol(TERPHENYL), ring)      # another molecule object: built afresh
    assert len(calls) == 3
