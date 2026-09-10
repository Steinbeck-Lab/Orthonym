"""Thiohydroximate/oxime O-sulfate ANION producer (a phase, glucosinolate
aglycone class).

``R2C=N-O-S(=O)(=O)-[O-]`` -> ``[({R2C}ylidene)amino] sulfate``: the sulfate
acid-ester anion whose ester owner is anchored at an oxime NITROGEN (a plain,
unsubstituted oxime N: the ester O + one C=N double bond) rather than the
usual ester-oxygen-to-carbon owner ``name_sulfate_ester_anion`` handles.
Reuses Slice A's sulfate word/coverage-audit shape and the SAME ``-ylidene``
primitive (``composer._double_bonded_carbon_prefix``) the semicarbazone/
hydrazone namers already use.

A FULL glucosinolate (the oxime N-O-sulfate PLUS an S-glycosidic thioglucoside
on the OTHER side of the ylidene carbon) is out of scope: the substituent
namer cannot yet name a ring-containing ylidene fragment, so it fails closed
(``None``) rather than emit a wrong or partial name -- verified via a real
ChEBI glucosinolate SMILES.
"""
import pytest
from rdkit import Chem
from orthonym import Orthonym
from orthonym.rules.acid_ester_anion import (
    name_oxime_o_sulfate_anion,
    name_acid_ester_anion,
    name_sulfate_ester_anion,
)


def _s_idx(smi):
    m = Chem.MolFromSmiles(smi)
    return next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'S'
                and not a.IsInRing())


def test_oxime_o_sulfate_methyl():
    # 1-sulfanylethylidene owner (the coordinator's RT-verified target).
    smi = "CC(=NOS(=O)(=O)[O-])S"
    assert (name_oxime_o_sulfate_anion(Chem.MolFromSmiles(smi), _s_idx(smi))
            == "[(1-sulfanylethylidene)amino] sulfate")


def test_oxime_o_sulfate_allyl():
    # sinigrin-type (prop-2-enyl) side chain, sugar stripped to a bare -SH.
    smi = "C=CCC(=NOS(=O)(=O)[O-])S"
    assert (name_oxime_o_sulfate_anion(Chem.MolFromSmiles(smi), _s_idx(smi))
            == "[(1-sulfanylbut-3-en-1-ylidene)amino] sulfate")


def test_oxime_o_sulfate_isopropyl():
    smi = "CC(C)C(=NOS(=O)(=O)[O-])S"
    assert (name_oxime_o_sulfate_anion(Chem.MolFromSmiles(smi), _s_idx(smi))
            == "[(2-methyl-1-sulfanylpropylidene)amino] sulfate")


def test_plain_sulfate_ester_still_owns_c_anchored_owner():
    # name_sulfate_ester_anion (Slice A) must still win for the ordinary
    # C-anchored owner; the new N-anchored sibling must not intercept it.
    smi = "CCCCCCCCCCCCOS(=O)(=O)[O-]"
    assert name_sulfate_ester_anion(Chem.MolFromSmiles(smi), _s_idx(smi)) == "dodecyl sulfate"
    assert name_oxime_o_sulfate_anion(Chem.MolFromSmiles(smi), _s_idx(smi)) is None


def test_n_substituted_oxime_ether_failclosed():
    # An N-substituted oxime-ether-like N (a THIRD N-substituent beyond the
    # ester O + the C=N) is a different, unbuilt class -> decline.
    smi = "CC(=[N+](C)OS(=O)(=O)[O-])S"
    m = Chem.MolFromSmiles(smi)
    assert m is not None
    assert name_oxime_o_sulfate_anion(m, _s_idx(smi)) is None


def test_full_glucosinolate_sugar_failclosed():
    # Real ChEBI glucosinolate (glucoraphasatin-family): the oxime-O-sulfate
    # PLUS an intact S-thioglucoside on the other ylidene arm. The ring in the
    # ylidene fragment is beyond the substituent namer's ylidene primitive ->
    # honest decline, never a partial/wrong name.
    smi = ("CS(=O)(=O)CCCC(=NOS(=O)(=O)[O-])"
           "S[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O")
    m = Chem.MolFromSmiles(smi)
    assert m is not None
    s_idx = next(a.GetIdx() for a in m.GetAtoms()
                 if a.GetSymbol() == 'S' and not a.IsInRing()
                 and any(nb.GetSymbol() == 'O' and not nb.IsInRing()
                         for nb in a.GetNeighbors()))
    assert name_oxime_o_sulfate_anion(m, s_idx) is None
    assert name_acid_ester_anion(m) is None


def test_dispatcher_routes_oxime_o_sulfate():
    assert (name_acid_ester_anion(Chem.MolFromSmiles("CC(=NOS(=O)(=O)[O-])S"))
            == "[(1-sulfanylethylidene)amino] sulfate")


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("CC(=NOS(=O)(=O)[O-])S", "[(1-sulfanylethylidene)amino] sulfate"),
    ("C=CCC(=NOS(=O)(=O)[O-])S", "[(1-sulfanylbut-3-en-1-ylidene)amino] sulfate"),
    ("CC(C)C(=NOS(=O)(=O)[O-])S", "[(2-methyl-1-sulfanylpropylidene)amino] sulfate"),
])
def test_integration_oxime_o_sulfate_names(namer, smi, expected):
    assert namer.name(smi) == expected


@pytest.mark.opsin_gate
def test_integration_full_glucosinolate_failclosed(namer):
    smi = ("CS(=O)(=O)CCCC(=NOS(=O)(=O)[O-])"
           "S[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O")
    assert namer.name(smi) == "unknown organic compound"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # Slice A regressions plain C-anchored ester owner) must
    # stay byte-identical -- the new N-anchored sibling must never intercept
    # or change these.
    ("CCCCCCCCCCCCOS(=O)(=O)[O-]", "dodecyl sulfate"),
    ("CCCCCCCCCCCCOP(=O)([O-])[O-]", "dodecyl phosphate"),
    ("CCCCCCCCCCCCOP(=O)([O-])O", "dodecyl hydrogen phosphate"),
    ("CCOP(=O)([O-])OCC", "diethyl phosphate"),
    ("C[N+](C)(C)CCOS(=O)(=O)[O-]", "2-(trimethylazaniumyl)ethyl sulfate"),
])
def test_integration_slice_a_regressions_unchanged(namer, smi, expected):
    assert namer.name(smi) == expected
