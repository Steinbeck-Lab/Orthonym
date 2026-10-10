"""Leads item N8e: a SMILES that writes every atom in brackets is named as the plain SMILES is.

A bracket atom carries its hydrogen count and no implicit hydrogens, so a bond a producer cuts
leaves a radical where the plain spelling leaves an H (rules/esters.py acid-of-ester,
rules/radicals.py identity check, rules/multiplicative.py bridge split). The ethyl pyruvate
'[CH3][CH2][O][C](=[O])[C](=[O])[CH3]' abstained at the default tier while 'CCOC(=O)C(C)=O' is
'ethyl 2-oxopropanoate' (pin_verified): one molecule, two names (the contributor guide core principle 4,
determinism; no Blue Book rule is involved).

The input scope of a public call (``namer._lone_pair_input_enter``) now names RDKit's own
non-canonical SMILES of its reading (brackets only where a charge, an isotope, a radical, a
stereo tag or a valence needs them), only when it reads with fewer NoImplicit atoms than the
caller's string and as the same molecule, and not for a string with a lone-pair stereocentre.
Every name is read back by OPSIN's own StdInChIKey against RDKit's key of the input.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym, namer
from orthonym.cli import _emit_tier_flags
from tests.unit.namer.leads_l1_support import opsin_key, rdkit_key

pytestmark = pytest.mark.opsin_gate


def _all_brackets(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), allHsExplicit=True, canonical=False)


def _engine(tier):
    return (Orthonym(style="pin") if tier == "pin"
            else Orthonym(style="pin", **_emit_tier_flags(tier)))


#: Rows of dev2000 that named at the PIN tier from the plain spelling and abstained from the
#: bracketed one (leads phase 1, brk_run.py), with the plain name.
DEV_ROWS = [
    ("CCOC(=O)C(C)=O", "ethyl 2-oxopropanoate"),
    ("CCOC(=O)C(C)O", "ethyl 2-hydroxypropanoate"),
    ("CCCCOC(=O)c1ccccc1C(=O)OCC(CC)CCCC", "butyl 2-ethylhexyl benzene-1,2-dicarboxylate"),
    ("CCC(O)c1ccc2c(c1)OCO2", "1-(2H-1,3-benzodioxol-5-yl)propan-1-ol"),
    ("COC(=O)c1ccc(CCCCCCCCCC[C@@H](C)O)cn1",
     "methyl 5-[(11R)-11-hydroxydodecyl]pyridine-2-carboxylate"),
    ("CCCCCC[C@@H](O)C/C=C\\CCCCCCCC(=O)OC", "methyl (9Z,12R)-12-hydroxyoctadec-9-enoate"),
    ("CCOC(=O)C(=CNc1cc(C(F)(F)F)cc(C(F)(F)F)c1)[N+](=O)[O-]",
     "ethyl 3-[3,5-bis(trifluoromethyl)anilino]-2-nitroprop-2-enoate"),
]
BRACKETED_BENZOATE = ("[CH3][O][C](=[O])[c]1[cH][cH][c]([Cl])[c]([NH][C](=[O])[CH2][CH2][NH]"
                      "[CH]2[CH2][CH2]2)[cH]1")
PLAIN_BENZOATE = "COC(=O)c1ccc(Cl)c(NC(=O)CCNC2CC2)c1"
BENZOATE_NAME = "methyl 4-chloro-3-[3-(cyclopropylamino)propanamido]benzoate"


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles, expected", DEV_ROWS)
def test_all_brackets_spelling_gets_the_plain_name(smiles, expected, tier):
    bracketed = _all_brackets(smiles)
    assert bracketed != smiles and "[C" in bracketed
    row = _engine(tier).name_tiered(bracketed)
    assert row["name"] == expected
    assert row["tier"] == "pin_verified" and row["is_pin"] is True
    assert opsin_key(row["name"]) == rdkit_key(smiles)


def test_the_benzoate_of_the_leads_report():
    assert _engine("pin").name_tiered(PLAIN_BENZOATE)["name"] == BENZOATE_NAME
    for tier in ("pin", "best-effort"):
        row = _engine(tier).name_tiered(BRACKETED_BENZOATE)
        assert row["name"] == BENZOATE_NAME and row["tier"] == "pin_verified"
    assert opsin_key(BENZOATE_NAME) == rdkit_key(PLAIN_BENZOATE)


def test_every_public_entry_names_the_bracketed_input():
    o = Orthonym(style="pin")
    bracketed = "[CH3][CH2][O][C](=[O])[C](=[O])[CH3]"
    assert o.name(bracketed) == "ethyl 2-oxopropanoate"
    assert o.name_with_confidence(bracketed)["name"] == "ethyl 2-oxopropanoate"
    assert o.name_with_tree(bracketed).name == "ethyl 2-oxopropanoate"
    from orthonym import name_compound
    assert name_compound(bracketed) == "ethyl 2-oxopropanoate"


def test_a_radical_keeps_its_brackets():
    # '[CH2]C' is the ethyl radical: not rewritten, and '[CH2][CH3]' is the same radical
    assert namer._redundant_bracket_spelling("[CH2]C") is None
    named, _ = namer._redundant_bracket_spelling("[CH2][CH3]")
    assert named == "[CH2]C"
    o = Orthonym(style="pin")
    assert o.name("[CH2][CH3]") == o.name("[CH2]C") == "ethyl"


@pytest.mark.parametrize("smiles", [
    "CCO", "C[C@H](O)CC", "[Na+].[Cl-]", "[13CH3]C", "c1cc[nH]c1", "[NH4+]", "C[N+](C)(C)C",
    "[CH2]C", "[O-][N+](=O)c1ccccc1",
    # a lone-pair stereocentre: the standard reading depends on where it is written
    "[CH3][S@](=[O])[CH2][CH3]",
])
def test_nothing_to_drop_or_not_to_touch(smiles):
    assert namer._redundant_bracket_spelling(smiles) is None


@pytest.mark.parametrize("smiles", [
    "[CH3][CH2][O][C](=[O])[C](=[O])[CH3]",
    "[CH3][C@H]([OH])[CH2][CH3]",
    "[O-][N+](=[O])[c]1[cH][cH][cH][cH][cH]1",
    "[CH3][CH2][NH3+]",
])
def test_the_same_molecule_and_the_atom_order(smiles):
    named, order = namer._redundant_bracket_spelling(smiles)
    given, out = Chem.MolFromSmiles(smiles), Chem.MolFromSmiles(named)
    assert Chem.MolToSmiles(given) == Chem.MolToSmiles(out)
    assert sum(a.GetNoImplicit() for a in out.GetAtoms()) < sum(
        a.GetNoImplicit() for a in given.GetAtoms())
    assert sorted(order) == list(range(given.GetNumAtoms()))
    for k, idx in enumerate(order):
        assert out.GetAtomWithIdx(k).GetSymbol() == given.GetAtomWithIdx(idx).GetSymbol()


def test_the_scope_rekeys_atom_hints_to_the_callers_atoms():
    caller = "[CH3][CH2][O][C](=[O])[C](=[O])[CH3]"
    named, opened = namer._lone_pair_input_enter(caller)
    try:
        assert opened and "[" not in named and named != caller
        given, out = Chem.MolFromSmiles(caller), Chem.MolFromSmiles(named)
        rekeyed = namer._lone_pair_caller_atoms({k: k for k in range(out.GetNumAtoms())})
        assert sorted(rekeyed) == list(range(given.GetNumAtoms()))
        for caller_idx, named_idx in rekeyed.items():
            assert (given.GetAtomWithIdx(caller_idx).GetSymbol()
                    == out.GetAtomWithIdx(named_idx).GetSymbol())
        # the exit checks read the caller's string
        assert namer._lone_pair_as_written(named) == caller
        assert namer._LP_INPUT.rewrite[3] is False
    finally:
        namer._lone_pair_input_exit(opened)


def test_composes_with_a_stereo_group():
    # the group is read first; the brackets are dropped from what is left
    smiles = "[CH3][C@H]([OH])[CH2][CH3] |&1:1|"
    named, opened = namer._lone_pair_input_enter(smiles)
    try:
        assert "@" not in named and "[" not in named and "|" not in named
        assert namer._LP_INPUT.rewrite[3] is True
    finally:
        namer._lone_pair_input_exit(opened)
    row = _engine("best-effort").name_tiered(smiles)
    assert row["name"] == "butan-2-ol" and row["tier"] == "systematic_verified"
    default = _engine("pin").name_tiered(smiles)
    assert default["tier"] == "abstain" and default["limit_code"] == "NO_VERIFIED_PIN"
