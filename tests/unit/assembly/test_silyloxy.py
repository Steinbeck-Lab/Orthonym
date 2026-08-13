"""Phase 1 B5: silyloxy substituent form.

`-O-[Si]<` is `{silyl}oxy` ('(trimethylsilyl)oxy'), NOT the cascade's mis-rooted
`hydroxy{silyl}` (a DIFFERENT molecule, -Si-OH). Silicon is a substitutive
parent hydride (silane -> silyl), and the direct -Si case already names
correctly; this routes the O-attached case through the same recursive namer +
an `oxy` morpheme (TMS/TBS/TIPS uniformly).
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import name_substituent

pytestmark = pytest.mark.unit


def _oxy_frag(smi):
    """`-O-[Si]...` fragment of a `C-O-Si...` model + the O attach atom."""
    mol = Chem.MolFromSmiles(smi)  # atom 0 = the parent CH3
    frag = set(range(mol.GetNumAtoms())) - {0}
    attach = next(n.GetIdx() for n in mol.GetAtomWithIdx(0).GetNeighbors())
    return mol, sorted(frag), attach


@pytest.mark.parametrize("smi,expected", [
    ("CO[Si](C)(C)C", "(trimethylsilyl)oxy"),
    ("CO[Si](C)(C)C(C)(C)C", "(tert-butyldimethylsilyl)oxy"),
])
def test_silyloxy_prefix(smi, expected):
    mol, frag, attach = _oxy_frag(smi)
    assert name_substituent(mol, frag, attach) == expected


def test_direct_silyl_unchanged():
    """The DIRECT -Si case was already correct and must stay so."""
    mol = Chem.MolFromSmiles("C[Si](C)(C)C")
    frag = {1, 2, 3, 4}  # the Si and its three methyls
    assert name_substituent(mol, sorted(frag), 1) == "trimethylsilyl"
