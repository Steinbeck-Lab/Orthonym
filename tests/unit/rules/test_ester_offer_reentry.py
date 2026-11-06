"""The offer of an acyclic ester over a lactone (``rules.seniority._acyclic_ester_offered``) is not made again for a copy
of the molecule whose offer is being built.

 class 9 "Esters" (the Blue Book): "functional class names are given to noncyclic esters; lactones and
other cyclic esters are named as heterocycles". The offer builds the functional-class name of the ester
(``rules.esters.name_ester``) to learn that it can be built. Naming the alcohol component asks for the principal group of
the same molecule again, and answers it as if the offer were not made. The guard that says "being offered" held the ``id``
of the Mol; the polycyclic analysis of the alcohol component asks on a copy of the molecule (another object, the same atoms
in the same order), which the guard did not know, so the offers nested 33 deep for a lactam with an acetate (the best-effort
rows of dev2000 took 5 s instead of 1 s) and each of them named the ester once more. The guard now holds the SMILES in atom
order that keys the memo of the offer.
"""
import pytest
from rdkit import Chem

from orthonym.rules import esters, seniority

pytestmark = pytest.mark.unit

ESTER = "CC(=O)OCCO"


def _mol_and_matches():
    mol = Chem.MolFromSmiles(ESTER)
    matches = [tuple(m) for m in mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2][#6]"))]
    assert matches
    return mol, matches


def test_a_copy_asked_while_the_offer_is_built_is_not_offered_again(monkeypatch):
    mol, matches = _mol_and_matches()
    built, inner = [], []

    def name_ester(m, match, *args, **kwargs):
        built.append(1)
        assert len(built) < 5, "the offer recursed"
        inner.append(seniority._acyclic_ester_offered(Chem.Mol(m), matches))       # the analysis works on a copy
        return "2-hydroxyethyl acetate"
    monkeypatch.setattr(esters, "name_ester", name_ester)
    assert seniority._acyclic_ester_offered(mol, matches) is True
    assert built == [1]
    assert inner == [False]


def test_the_object_asked_again_is_not_offered_again(monkeypatch):
    mol, matches = _mol_and_matches()
    inner = []

    def name_ester(m, match, *args, **kwargs):
        inner.append(seniority._acyclic_ester_offered(m, matches))
        return "2-hydroxyethyl acetate"
    monkeypatch.setattr(esters, "name_ester", name_ester)
    assert seniority._acyclic_ester_offered(mol, matches) is True
    assert inner == [False]


def test_the_guard_is_released_when_the_offer_ends(monkeypatch):
    mol, matches = _mol_and_matches()
    monkeypatch.setattr(esters, "name_ester", lambda m, match, *a, **k: "2-hydroxyethyl acetate")
    assert seniority._acyclic_ester_offered(mol, matches) is True
    assert not seniority._ESTER_OFFER_ACTIVE.get()
    monkeypatch.setattr(esters, "name_ester", lambda m, match, *a, **k: None)
    assert seniority._acyclic_ester_offered(Chem.Mol(mol), matches) is False        # a name that cannot be built: no offer
    assert not seniority._ESTER_OFFER_ACTIVE.get()


def test_a_name_that_raises_is_no_offer_and_releases_the_guard(monkeypatch):
    mol, matches = _mol_and_matches()

    def name_ester(m, match, *args, **kwargs):
        raise ValueError("cannot build")
    monkeypatch.setattr(esters, "name_ester", name_ester)
    assert seniority._acyclic_ester_offered(mol, matches) is False
    assert not seniority._ESTER_OFFER_ACTIVE.get()


def test_another_molecule_is_offered_while_one_is_being_built(monkeypatch):
    mol, matches = _mol_and_matches()
    other = Chem.MolFromSmiles("CC(=O)OCCCO")
    other_matches = [tuple(m) for m in other.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2][#6]"))]
    inner = []

    def name_ester(m, match, *args, **kwargs):
        if m is mol:
            inner.append(seniority._acyclic_ester_offered(other, other_matches))
        return "an ester name"
    monkeypatch.setattr(esters, "name_ester", name_ester)
    assert seniority._acyclic_ester_offered(mol, matches) is True
    assert inner == [True]
