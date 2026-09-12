from rdkit import Chem
from orthonym.perception.smarts_cache import compiled


def test_compiled_matches_like_fresh():
    mol = Chem.MolFromSmiles("CC(=O)N[C@@H](C)C(=O)O")
    for p in ("[CX3](=O)[NX3;H0,H1][CX4]", "[NX3;H2,H1,H0][CX4][CX3](=O)[OX2H1]", "c1ccccc1"):
        assert mol.GetSubstructMatches(compiled(p), uniquify=True) == mol.GetSubstructMatches(Chem.MolFromSmarts(p), uniquify=True)
    assert compiled("[CX3](=O)[OX2H1]") is compiled("[CX3](=O)[OX2H1]")
    assert compiled("not a smarts (((") is None
