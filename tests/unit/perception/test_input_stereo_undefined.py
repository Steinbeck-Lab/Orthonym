from rdkit import Chem
from orthonym.perception.stereo import input_stereo_undefined

def _m(smi): return Chem.MolFromSmiles(smi)

def test_flat_stereocentre_is_undefined():
    # flat alanine: real alpha stereocentre, no wedge -> undefined
    assert input_stereo_undefined(_m("CC(N)C(=O)O")) is True

def test_defined_stereocentre_is_not_undefined():
    # L-alanine: alpha-carbon defined -> not undefined
    assert input_stereo_undefined(_m("N[C@@H](C)C(=O)O")) is False

def test_achiral_molecule_is_not_undefined():
    # glycine: no stereocentre at all
    assert input_stereo_undefined(_m("NCC(=O)O")) is False
    assert input_stereo_undefined(_m("c1ccccc1")) is False

def test_flat_steroid_is_undefined():
    # flat androstane-diol shape: multiple undefined ring stereocentres
    assert input_stereo_undefined(_m("CC12CCC3C(CCC4CC(O)CCC43C)C1CCC2O")) is True

def test_undefined_double_bond_is_undefined():
    # a stereogenic C=C with no direction (2-butene, undirected)
    assert input_stereo_undefined(_m("CC=CC")) is True

def test_defined_double_bond_is_not_undefined():
    assert input_stereo_undefined(_m("C/C=C/C")) is False

def test_atom_indices_scopes_tetrahedral_check():
    # flat threonine has TWO undefined centres; restrict to a non-stereo atom -> False
    m = _m("CC(O)C(N)C(=O)O")
    achiral_atom = 0  # a methyl carbon, never a stereocentre
    assert input_stereo_undefined(m, atom_indices=[achiral_atom]) is False
    assert input_stereo_undefined(m) is True

def test_fail_closed_on_none():
    assert input_stereo_undefined(None) is True
