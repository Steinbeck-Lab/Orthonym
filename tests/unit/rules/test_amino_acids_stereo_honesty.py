from rdkit import Chem
from orthonym.rules.amino_acids import name_amino_acid

def _c(smi):
    m = Chem.MolFromSmiles(smi)
    return m, Chem.MolToSmiles(m, canonical=True)

def test_flat_s_methylcysteine_declines_retained():
    m, cs = _c("CSCC(N)C(=O)O")
    out = name_amino_acid(m, cs)
    # must NOT be the config-implying retained name; systematic is fine
    assert out != "S-methylcysteine"
    assert out is None or "cysteine" not in out

def test_defined_l_alanine_keeps_descriptor_name():
    m, cs = _c("N[C@@H](C)C(=O)O")
    out = name_amino_acid(m, cs)
    assert out == "L-alanine"

def test_achiral_glycine_keeps_retained():
    m, cs = _c("NCC(=O)O")
    out = name_amino_acid(m, cs)
    assert out == "glycine"
