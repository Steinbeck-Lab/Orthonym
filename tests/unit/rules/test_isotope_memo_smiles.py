"""The memo key of the isotope placement search computes the SMILES of its reference molecule once per molecule object
(``rules/isotope_memos._smiles_of``): one search reads one molecule up to 75,000 times, and its SMILES cost 0.5 ms each
time on a large labelled molecule (time-out audit 2026-11-12, an ISO row: 14 of 49 s)."""
from rdkit import Chem

from orthonym.rules import isotope_memos


def test_the_smiles_is_computed_once_per_molecule_object(monkeypatch):
    calls = []
    real = Chem.MolToSmiles
    monkeypatch.setattr(isotope_memos.Chem, "MolToSmiles", lambda m, *a, **k: calls.append(1) or real(m, *a, **k))
    monkeypatch.setattr(isotope_memos, "_SMILES_OF", {})
    mol = Chem.MolFromSmiles("[2H]c1ccccc1")
    assert isotope_memos._smiles_of(mol) == real(mol)
    assert isotope_memos._smiles_of(mol) == real(mol)
    assert len(calls) == 1


def test_another_object_is_not_served_the_cached_smiles(monkeypatch):
    monkeypatch.setattr(isotope_memos, "_SMILES_OF", {})
    a, b = Chem.MolFromSmiles("[2H]c1ccccc1"), Chem.MolFromSmiles("[13CH3]O")
    assert isotope_memos._smiles_of(a) != isotope_memos._smiles_of(b)
    assert isotope_memos._smiles_of(b) == Chem.MolToSmiles(b)


def test_the_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(isotope_memos, "_SMILES_OF", {})
    keep = [Chem.MolFromSmiles("C" * (i + 1)) for i in range(isotope_memos._SMILES_SLOTS + 10)]
    for m in keep:
        isotope_memos._smiles_of(m)
    assert len(isotope_memos._SMILES_OF) <= isotope_memos._SMILES_SLOTS
