"""a phase B5: silyloxy substituent form.

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
    # (the Blue Book): the second cited group is enclosed, its multiplier outside the marks
    # ('tert-butyldi(methyl)(oxiranylmethoxy)silane (PIN)',:18929); a compound group takes its own marks.
    # ASSUMED-book-inconsistency (coordinator ruling 2026-10-24): '(tert-butylperoxy)dimethylsilyl propanoate (PIN)'
    # (:23389), 'aminodichlorosilyl (preselected prefix)' (:36385) and the alternative 'acetyldimethylsilyl' (:3553)
    # cite a multiplied later group of a silyl bare; the rule text is followed.
    ("CO[Si](C)(C)C(C)(C)C", "[tert-butyldi(methyl)silyl]oxy"),
])
def test_silyloxy_prefix(smi, expected):
    mol, frag, attach = _oxy_frag(smi)
    assert name_substituent(mol, frag, attach) == expected


def test_direct_silyl_unchanged():
    """The DIRECT -Si case was already correct and must stay so."""
    mol = Chem.MolFromSmiles("C[Si](C)(C)C")
    frag = {1, 2, 3, 4}  # the Si and its three methyls
    assert name_substituent(mol, sorted(frag), 1) == "trimethylsilyl"
