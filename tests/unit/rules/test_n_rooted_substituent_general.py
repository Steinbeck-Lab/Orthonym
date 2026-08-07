"""v30 breadth — the general-engine substituent namer must name an N-rooted substituent
`-NH-R` / `-N(R)R'` / `-NH-C(=O)R` as the amino/amido PREFIX, NOT the cascade's carbon-rooted
misroot.

Root cause (hetsweep, same mechanism as the alkoxy fix ffffdab4): `name_substituent` normalizes
the attach atom to the N, `carbon_free_valence_prefix` declines the non-carbon root, and the
cascade re-roots on CARBON -> `-NH-CH3` became `aminomethyl` (that is `-CH2-NH2`, a DIFFERENT
molecule), `-N(CH3)2` -> unparseable `amino-N-methylmethyl`, `-NH-C(=O)CH3` -> `carbamoylmethyl`
(`-CH2-C(=O)NH2`). Fix: route the N-rooted case to the existing `_name_amino_branch` (the same
builder the PIN path uses). Hydrazinyl/diazenyl/N-O/sulfonamido return None from it and fall
through unchanged (they already work or are a separate follow-up).
"""
from rdkit import Chem
from orthonym.assembly.substituent_enumerator import name_substituent


def _sub(smi, frag, root):
    m = Chem.MolFromSmiles(smi)
    return name_substituent(m, set(frag), root, allow_mancude=True)


def test_methylamino():
    # OC(=O)c1ccc(NC)cc1 ... but name_substituent takes the fragment; build -NH-CH3 on benzene
    m = Chem.MolFromSmiles("CNc1ccccc1")   # C0-N1-c2(ring); frag {C0,N1}, root N1
    assert name_substituent(m, {0, 1}, 1, allow_mancude=True) == "methylamino"


def test_ethylamino():
    m = Chem.MolFromSmiles("CCNc1ccccc1")  # C0-C1-N2-c3; frag {C0,C1,N2}, root N2
    assert name_substituent(m, {0, 1, 2}, 2, allow_mancude=True) == "ethylamino"


def test_dimethylamino():
    m = Chem.MolFromSmiles("CN(C)c1ccccc1")  # C0-N1(-C2)-c3; frag {C0,N1,C2}, root N1
    assert name_substituent(m, {0, 1, 2}, 1, allow_mancude=True) == "dimethylamino"


def test_acetamido():
    m = Chem.MolFromSmiles("CC(=O)Nc1ccccc1")  # C0-C1(=O2)-N3-c4; frag {C0,C1,O2,N3}, root N3
    assert name_substituent(m, {0, 1, 2, 3}, 3, allow_mancude=True) == "acetamido"


def test_not_aminomethyl_for_methylamino():
    # the wrong-molecule token must never come back for an N-rooted -NH-CH3.
    m = Chem.MolFromSmiles("CNc1ccccc1")
    assert name_substituent(m, {0, 1}, 1, allow_mancude=True) != "aminomethyl"


def test_real_aminomethyl_unchanged():
    # a genuine -CH2-NH2 (carbon-rooted, N is a pendant primary amine) stays aminomethyl.
    m = Chem.MolFromSmiles("NCc1ccccc1")   # N0-C1-c2; frag {N0,C1}, root C1 (carbon)
    assert name_substituent(m, {0, 1}, 1, allow_mancude=True) == "aminomethyl"
