""" - N-substituted carbamimidate perception (W2F p6 Task 4).

BB (the Blue Book): dedicated carbamimidate SMARTS
[CX3](=[NX2])([NX3])[OX2][#6] (imino-N substitution allowed + a required amino
N). The restricted iminoester [NX2H1] pattern is left untouched so the
AUDIT-FRN 2.4 =NH guard on plain iminoesters holds.

NOTE (re-anchor at HEAD 9a33eb63): the plan helper imports
``find_functional_groups``; the real function is ``detect_functional_groups``.
"""
from rdkit import Chem
from orthonym.perception.functional_groups import detect_functional_groups


def test_carbamimidate_smarts_matches_n_substituted():
    mol = Chem.MolFromSmiles("CCOC(=NC)N(c1ccccc1)c1ccccc1")
    fgs = detect_functional_groups(mol)
    assert fgs.get("carbamimidate")           # N-substituted -> matches the NEW pattern


def test_plain_iminoester_pattern_unchanged():
    # regression: a plain =NH iminoester still matches iminoester, not carbamimidate
    mol = Chem.MolFromSmiles("CCOC(C)=N")
    fgs = detect_functional_groups(mol)
    assert fgs.get("iminoester")
    assert not fgs.get("carbamimidate")       # no second N -> not a carbamimidate


def test_n_substituted_iminoester_still_not_detected():
    # AUDIT-FRN 2.4 guard intact: N-sub iminoester (carbon on the C) stays undetected
    mol = Chem.MolFromSmiles("CCC(=NC)OC")
    fgs = detect_functional_groups(mol)
    assert "iminoester" not in fgs
    assert "carbamimidate" not in fgs
