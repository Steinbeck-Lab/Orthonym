"""``lipids._isolated_acid_smiles`` is memoised per molecule object for the naming call: the
same site's acid is asked for by the classifier, the acyl namers and the acyloxy leaf. The
memo returns what the uncached cut returns, holds only inside a scope, and an in-place edit
of the molecule is a miss."""
import pytest
from rdkit import Chem

from orthonym.assembly import memo
from orthonym.rules import lipids

pytestmark = pytest.mark.unit

# glycerol tri(hydrogen succinate): three half-ester sites
_SMILES = "O=C(O)CCC(=O)OCC(OC(=O)CCC(=O)O)COC(=O)CCC(=O)O"


def _sites(mol):
    patt = Chem.MolFromSmarts("[CX3](=O)[OX2][CH2,CH]")
    return [(m[0], m[2]) for m in mol.GetSubstructMatches(patt)]


def test_memo_equals_the_uncached_cut_for_every_site_and_both_charges():
    mol = Chem.MolFromSmiles(_SMILES)
    sites = _sites(mol)
    assert len(sites) == 3
    tok = memo.push_scope()
    try:
        for c, o in sites:
            for anionic in (False, True):
                got = lipids._isolated_acid_smiles(mol, c, o, anionic)
                assert got == lipids._isolated_acid_smiles_uncached(mol, c, o, anionic)
    finally:
        memo.pop_scope(tok)
    c, o = sites[0]
    assert lipids._isolated_acid_smiles(mol, c, o) == "O=C(O)CCC(=O)O"
    assert lipids._isolated_acid_smiles(mol, c, o, True) == "O=C([O-])CCC(=O)O"


def test_a_repeat_inside_a_scope_is_served_from_the_memo(monkeypatch):
    mol = Chem.MolFromSmiles(_SMILES)
    c, o = _sites(mol)[0]
    calls = []
    real = lipids._isolated_acid_smiles_uncached
    monkeypatch.setattr(lipids, "_isolated_acid_smiles_uncached",
                        lambda *a, **k: calls.append(a) or real(*a, **k))
    tok = memo.push_scope()
    try:
        first = lipids._isolated_acid_smiles(mol, c, o)
        again = lipids._isolated_acid_smiles(mol, c, o)
        other = lipids._isolated_acid_smiles(mol, c, o, anionic=True)
    finally:
        memo.pop_scope(tok)
    assert first == again and other != first
    assert len(calls) == 2          # neutral once, anionic once


def test_no_scope_means_no_memo(monkeypatch):
    mol = Chem.MolFromSmiles(_SMILES)
    c, o = _sites(mol)[0]
    calls = []
    real = lipids._isolated_acid_smiles_uncached
    monkeypatch.setattr(lipids, "_isolated_acid_smiles_uncached",
                        lambda *a, **k: calls.append(a) or real(*a, **k))
    assert memo._cache_var.get() is None
    lipids._isolated_acid_smiles(mol, c, o)
    lipids._isolated_acid_smiles(mol, c, o)
    assert len(calls) == 2


def test_an_in_place_edit_of_the_molecule_is_a_miss():
    mol = Chem.MolFromSmiles(_SMILES)
    c, o = _sites(mol)[0]
    tok = memo.push_scope()
    try:
        before = lipids._isolated_acid_smiles(mol, c, o)
        # the free carboxy group at the far end of this site's acid becomes the carboxylate
        tail = next(a.GetIdx() for a in mol.GetAtoms()
                    if a.GetAtomicNum() == 8 and a.GetTotalNumHs() == 1
                    and a.GetNeighbors()[0].GetIdx() < c)
        mol.GetAtomWithIdx(tail).SetFormalCharge(-1)
        mol.GetAtomWithIdx(tail).SetNumExplicitHs(0)
        Chem.SanitizeMol(mol)
        after = lipids._isolated_acid_smiles(mol, c, o)
    finally:
        memo.pop_scope(tok)
    assert before == "O=C(O)CCC(=O)O"
    assert after == "O=C([O-])CCC(=O)O"


@pytest.mark.parametrize("edit", [
    lambda a: a.SetAtomMapNum(7),
    lambda a: a.SetNoImplicit(True),
    lambda a: a.SetNumRadicalElectrons(1),
])
def test_an_in_place_edit_the_inchi_fingerprint_does_not_see_is_a_miss(edit):
    # map number, the no-implicit flag and radicals are written into the cut fragment's SMILES
    # but not read by the standard InChI: the key must carry them too
    mol = Chem.MolFromSmiles("O=C(O)CCC(=O)OCC")
    c, o = _sites(mol)[0]
    tok = memo.push_scope()
    try:
        before = lipids._isolated_acid_smiles(mol, c, o)
        edit(mol.GetAtomWithIdx(3))
        after = lipids._isolated_acid_smiles(mol, c, o)
        assert after == lipids._isolated_acid_smiles_uncached(mol, c, o)
    finally:
        memo.pop_scope(tok)
    assert before == "O=C(O)CCC(=O)O" and after != before


def test_no_key_is_built_outside_a_scope_or_for_an_rwmol(monkeypatch):
    calls = []
    monkeypatch.setattr(lipids, "_acid_cut_sig", lambda m: calls.append(m) or ())
    mol = Chem.MolFromSmiles(_SMILES)
    c, o = _sites(mol)[0]
    lipids._isolated_acid_smiles(mol, c, o)
    tok = memo.push_scope()
    try:
        lipids._isolated_acid_smiles(Chem.RWMol(mol), c, o)
    finally:
        memo.pop_scope(tok)
    assert calls == []
