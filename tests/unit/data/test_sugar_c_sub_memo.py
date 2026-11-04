"""Lever M (a performance pass): name_c_substituted_sugar runs its expensive body once per structure within a scope."""
from rdkit import Chem
from orthonym.assembly import memo
from orthonym.data import sugar_names as sn


def test_memo_within_scope(monkeypatch):
    calls = {"n": 0}
    orig = sn._name_c_substituted_sugar_impl
    def spy(mol, csmi):
        calls["n"] += 1
        return orig(mol, csmi)
    monkeypatch.setattr(sn, "_name_c_substituted_sugar_impl", spy)
    smi = "C[C@@]1(O)[C@H](O)[C@@H](O)[C@H](O)[C@@H](CO)O1"     # a C-methyl hexopyranose (input to the C-substituted path)
    a = Chem.MolFromSmiles(smi); b = Chem.MolFromSmiles(smi)
    tok = memo.push_scope()
    try:
        ra = sn.name_c_substituted_sugar(a, smi); rb = sn.name_c_substituted_sugar(b, smi)
    finally:
        memo.pop_scope(tok)
    assert ra == rb
    assert calls["n"] == 1
    assert sn.name_c_substituted_sugar(Chem.MolFromSmiles(smi), smi) == ra     # no scope: computed directly, same answer
    assert calls["n"] == 2


# --- necessary-condition guard before the stereoisomer enumeration (speed lever 2b) ---------------------------
# "Substitution at a nonterminal carbon atom" (the Blue Book): "The compound is named as a
# C-substituted monosaccharide", so a reconstructed parent that no sugar entry or sugar skeleton describes cannot
# give a name.

import pytest
import rdkit.Chem.EnumerateStereoisomers as _enum_mod


def _count_enumerations(monkeypatch):
    calls = {"n": 0}
    orig = _enum_mod.EnumerateStereoisomers

    def spy(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)
    monkeypatch.setattr(_enum_mod, "EnumerateStereoisomers", spy)
    return calls


def _spy_guard(monkeypatch):
    verdicts = []
    orig = sn._c_sugar_parent_can_be_a_sugar

    def spy(parent):
        verdicts.append(orig(parent))
        return verdicts[-1]
    monkeypatch.setattr(sn, "_c_sugar_parent_can_be_a_sugar", spy)
    return verdicts


@pytest.mark.parametrize("smi", [
    "CC(=O)OC[C@H]1OC(O)[C@H](F)[C@@H](O)[C@H]1O",       # an acetyl: a double bond, so no clean sugar ring
    "NCCO[C@H]1[C@H](O)[C@@H](CO)O[C@@H](O)[C@@H]1F",    # a nitrogen anywhere in the molecule
    "CC1(C)OC[C@H]2OC(O)[C@H](F)[C@@H](O)[C@H]2O1",      # a second ring
])
def test_non_sugar_parent_skips_the_stereoisomer_enumeration(monkeypatch, smi):
    """The guard is reached for these C-substituted rings, says no, and the enumeration never runs."""
    verdicts = _spy_guard(monkeypatch)
    calls = _count_enumerations(monkeypatch)
    mol = Chem.MolFromSmiles(smi)
    assert sn._name_c_substituted_sugar_impl(mol, Chem.MolToSmiles(mol)) is None
    assert verdicts == [False]
    assert calls["n"] == 0


@pytest.mark.parametrize("smi, expected", [
    ("OC1CCCCO1", True),                                     # the constitution gates read no stereo: a lone hydroxy-oxane passes
    ("CC(=O)OC1OC(CO)C(O)C(O)C1O", False),                  # a double bond
    ("NC1OC(CO)C(O)C(O)C1O", False),                         # a nitrogen
    ("OC1CCCCCO1", False),                                   # a seven-membered ring
    ("OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O", True),   # a catalogued pyranose
    ("OCC1OC(O)C(O)C(O)C1O", True),                          # stereo-free hexopyranose: the skeleton gates pass
    ("OCC1OC(O)C(O)C1O", True),                              # a pentofuranose with a hydroxymethyl tail
])
def test_parent_can_be_a_sugar(smi, expected):
    assert sn._c_sugar_parent_can_be_a_sugar(Chem.MolFromSmiles(smi)) is expected


def test_a_real_c_substituted_sugar_still_enumerates_and_keeps_its_name(monkeypatch):
    calls = _count_enumerations(monkeypatch)
    smi = "OC[C@H]1O[C@@H](O)[C@H](F)[C@@H](O)[C@@H]1O"
    mol = Chem.MolFromSmiles(smi)
    assert sn._name_c_substituted_sugar_impl(mol, Chem.MolToSmiles(mol)) == "2-deoxy-2-fluoro-β-D-glucopyranose"
    assert calls["n"] == 1


@pytest.mark.parametrize("smi, expected", [
    ("C[C@@]1(O)[C@H](O)[C@@H](O)[C@H](O)[C@@H](CO)O1", "(2S,3R,4S,5S,6R)-6-(hydroxymethyl)-2-methyloxane-2,3,4,5-tetrol"),
    ("OC[C@H]1OC(O)[C@H](F)[C@@H](O)[C@H]1O", "2-deoxy-2-fluoro-D-galactopyranose"),
    ("OC[C@H]1O[C@@H](O)[C@H](F)[C@@H](O)[C@@H]1O", "2-deoxy-2-fluoro-β-D-glucopyranose"),
])
def test_names_before_the_guard_are_kept(smi, expected):
    """The names a run before the guard gave, through the whole default cascade."""
    from orthonym import name_compound
    assert name_compound(smi) == expected
