"""Leads item N8d: an AND or OR stereo group of a CXSMILES string is not read as a configuration.

``C[C@H](O)CC |&1:1|`` is a racemate and ``|o1:1|`` is one enantiomer that is not said; the str
path wrote plain SMILES at its first step, which keeps no stereo group, and named both
'(2S)-butan-2-ol' (pin_verified), the name of one enantiomer. The input scope of a public call
(``namer._lone_pair_input_enter``) now reads the groups first and names the structure with the
configuration of the group members cleared, which holds for a racemate and for an unknown
enantiomer; the members of an absolute (``a``) group and the ungrouped centres keep theirs.

Blue Book 'Racemates' (the Blue Book),:45916: 'Racemates may be denoted by using
the prefix rac cited at the front of the name of one enantiomer... This prefix rac with plain
stereodescriptors is preferred in preferred IUPAC names'; the example at:45940 is
'rac-(2R)-2-bromobutane (PIN)'. 'Relative configuration' (:45901) for 'rel'. That PIN
spelling is NOT built (the InChIKey round trip cannot see a stereo group, so it needs its own
check); the name without the group's configuration is recorded as a non-PIN name, the default
tier declines it with the rule, and the wider tiers keep it.

Every emitted name is read back by OPSIN's own StdInChIKey and compared with RDKit's InChIKey of
the structure with the group members cleared; it must NOT be the key of either enantiomer.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym, name_compound, namer
from orthonym.cli import _emit_tier_flags
from tests.support.default_tier import assert_default_tier_declines
from tests.unit.namer.leads_l1_support import mol_key_without_stereo_of, opsin_key, rdkit_key

pytestmark = pytest.mark.opsin_gate

# (input, atom indices of the AND/OR group members, name at the wider tiers)
GROUPED = [
    ("C[C@H](O)CC |&1:1|", [1], "butan-2-ol"),
    ("C[C@H](O)CC |o1:1|", [1], "butan-2-ol"),
    ("N[C@@H](C)C(=O)O |&1:1|", [1], "2-aminopropanoic acid"),
    ("C[C@H](O)[C@@H](C)CC |&1:1,3|", [1, 3], "3-methylpentan-2-ol"),
]
WIDER = ["valid", "complete", "best-effort"]


def _engine(tier):
    return (Orthonym(style="pin") if tier == "pin"
            else Orthonym(style="pin", **_emit_tier_flags(tier)))


@pytest.mark.parametrize("smiles, members, expected", GROUPED)
def test_default_tier_declines_with_the_rule(smiles, members, expected):
    row = assert_default_tier_declines(smiles)
    assert [f["rule"] for f in row["spelling_failures"]] == ["P-93.1.3"]
    # the label of the plain call is the decline, on every public entry
    o = Orthonym(style="pin")
    assert o.name(smiles) == row["name"]
    assert name_compound(smiles) == row["name"]
    assert o.name_with_confidence(smiles)["name"] == row["name"]
    assert o.name_with_tree(smiles).name == row["name"]


@pytest.mark.parametrize("tier", WIDER)
@pytest.mark.parametrize("smiles, members, expected", GROUPED)
def test_wider_tiers_name_the_structure_without_the_group_configuration(
        tier, smiles, members, expected):
    row = _engine(tier).name_tiered(smiles)
    assert row["name"] == expected
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False
    # OPSIN reads the name as the structure with the members' configuration cleared...
    assert opsin_key(row["name"]) == mol_key_without_stereo_of(smiles, members)
    #... and not as the enantiomer the SMILES draws (the name of the group is not that one)
    assert opsin_key(row["name"]) != rdkit_key(smiles.split(" |")[0])
    # every public entry gives the same name
    o = _engine(tier)
    assert o.name(smiles) == expected
    assert o.name_with_confidence(smiles)["name"] == expected
    assert o.name_with_tree(smiles).name == expected


def test_the_wrong_name_of_the_enantiomer_is_what_the_old_path_gave():
    # the key the old name '(2S)-butan-2-ol' denotes differs from the group's structure
    assert opsin_key("(2S)-butan-2-ol") != mol_key_without_stereo_of("C[C@H](O)CC", [1])
    assert opsin_key("(2S)-butan-2-ol") == rdkit_key("C[C@H](O)CC")


def test_an_absolute_group_is_not_changed():
    # '|a:1|' is one enantiomer, named as before; so is the string without an extension
    for tier in ["pin", *WIDER]:
        row = _engine(tier).name_tiered("C[C@H](O)CC |a:1|")
        assert row["name"] == "(2S)-butan-2-ol" and row["tier"] == "pin_verified"
        assert opsin_key(row["name"]) == rdkit_key("C[C@H](O)CC")


@pytest.mark.parametrize("tier", ["pin", *WIDER])
def test_a_mixed_string_keeps_the_absolute_members(tier):
    # atom 1 absolute, atom 4 AND: the name keeps atom 1's configuration (2S) and omits atom 4's
    smiles = "C[C@H](O)C[C@@H](C)CC |a:1,&1:4|"
    row = _engine(tier).name_tiered(smiles)
    if tier == "pin":
        assert row["tier"] == "abstain" and row["limit_code"] == "NO_VERIFIED_PIN"
        return
    assert row["name"] == "(2S)-4-methylhexan-2-ol"
    assert opsin_key(row["name"]) == mol_key_without_stereo_of(smiles, [4])


def test_a_group_with_no_stereo_element_changes_nothing():
    # a group over an atom that carries no configuration leaves nothing out
    assert namer._enhanced_stereo_cleared_spelling("CCO |&1:0|") is None
    assert namer._enhanced_stereo_cleared_spelling("CCO") is None
    assert namer._enhanced_stereo_cleared_spelling("C[C@H](O)CC") is None
    assert namer._enhanced_stereo_cleared_spelling("C[C@H](O)CC |a:1|") is None
    assert namer._enhanced_stereo_cleared_spelling("not a smiles |&1:0|") is None


def test_the_cleared_string_and_its_atom_order():
    named, order = namer._enhanced_stereo_cleared_spelling("C[C@H](O)[C@@H](C)CC |&1:1,3|")
    assert "@" not in named and "|" not in named
    reading = Chem.MolFromSmiles("C[C@H](O)[C@@H](C)CC")
    out = Chem.MolFromSmiles(named)
    assert sorted(order) == list(range(reading.GetNumAtoms()))
    # atom k of the string named is atom order[k] of the caller's reading
    for k, idx in enumerate(order):
        assert out.GetAtomWithIdx(k).GetSymbol() == reading.GetAtomWithIdx(idx).GetSymbol()
    # the double-bond configuration of a grouped alkene is cleared the same way
    group = Chem.MolFromSmiles("C/C=C/C |&1:1,2|")
    if group is not None and any(g.GetBonds() for g in group.GetStereoGroups()):
        named, _ = namer._enhanced_stereo_cleared_spelling("C/C=C/C |&1:1,2|")
        assert "/" not in named


def test_a_lone_pair_stereocentre_of_a_grouped_molecule_is_cleared_not_misread():
    # the sulfoxide's configuration is omitted with the group's, never changed
    smiles = "C[C@H](O)C[S@](=O)c1ccccc1 |&1:1|"
    named, _ = namer._enhanced_stereo_cleared_spelling(smiles)
    assert "@" not in named
    row = _engine("best-effort").name_tiered(smiles)
    flat = "CC(O)CS(=O)c1ccccc1"
    assert row["name"] and opsin_key(row["name"]) == rdkit_key(flat)


def test_the_scope_reports_the_cleared_string_as_written():
    smiles = "C[C@H](O)CC |&1:1|"
    named, opened = namer._lone_pair_input_enter(smiles)
    try:
        assert opened and "|" not in named and "@" not in named
        assert namer._lone_pair_as_written(named) == named
        assert namer._LP_INPUT.rewrite[3] is True
        # a nested public call names the string it is given
        assert namer._lone_pair_input_enter(smiles) == (smiles, False)
    finally:
        namer._lone_pair_input_exit(opened)
    assert getattr(namer._LP_INPUT, "rewrite", None) is None


# ---- a meso compound is not a group of unknown configuration --------------------------------

#: 'C[C@H](O)[C@H](O)C' is the meso diol (2R,3S)-butane-2,3-diol: inverting both centres gives the
#: same compound, so an AND or OR group over both centres (a racemate, or one enantiomer that is
#: not said, of a structure that has no enantiomer) draws the one compound, and its configuration
#: is known. 'Racemates' (the Blue Book) applies to a racemic compound; a meso
#: compound has no enantiomer to be racemic with, so nothing the group could leave out is unknown.
MESO = ["C[C@H](O)[C@H](O)C |o1:1,3|", "C[C@H](O)[C@H](O)C |&1:1,3|",
        "C[C@H](O)[C@H](O)C |o1:1,3,&2:0|"]


@pytest.mark.parametrize("tier", ["pin", *WIDER])
@pytest.mark.parametrize("smiles", MESO)
def test_a_meso_compound_in_one_group_keeps_its_configuration_and_its_pin(smiles, tier):
    row = _engine(tier).name_tiered(smiles)
    assert (row["name"], row["tier"], row["is_pin"]) == (
        "(2R,3S)-butane-2,3-diol", "pin_verified", True), (tier, row)
    # OPSIN reads the name as the compound the string draws (the full key, configuration included)
    assert opsin_key(row["name"]) == rdkit_key(smiles.split(" |")[0])
    o = _engine(tier)
    assert o.name(smiles) == row["name"]
    assert o.name_with_confidence(smiles)["name"] == row["name"]
    assert o.name_with_tree(smiles).name == row["name"]
    if tier == "pin":
        assert name_compound(smiles) == row["name"]


def test_the_group_of_a_meso_compound_clears_nothing():
    for smiles in MESO:
        assert namer._enhanced_stereo_cleared_spelling(smiles) is None, smiles
        mol = Chem.MolFromSmiles(smiles)
        assert all(namer._stereo_group_inverts_to_itself(mol, g)
                   for g in mol.GetStereoGroups() if g.GetAtoms() and any(
                       a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in g.GetAtoms()))


@pytest.mark.parametrize("smiles, expected", [
    # the (R,R)/(S,S) diol is chiral: a group over both centres is a racemate or an unknown enantiomer
    ("C[C@H](O)[C@@H](O)C |&1:1,3|", "butane-2,3-diol"),
    ("C[C@H](O)[C@@H](O)C |o1:1,3|", "butane-2,3-diol"),
    # one centre of two: inverting it gives the other diastereomer, not the same compound
    ("C[C@H](O)[C@H](O)C |&1:1|", "(2R)-butane-2,3-diol"),
])
def test_a_group_that_does_not_invert_to_itself_is_still_cleared(smiles, expected):
    assert namer._enhanced_stereo_cleared_spelling(smiles) is not None
    row = assert_default_tier_declines(smiles)
    assert [f["rule"] for f in row["spelling_failures"]] == ["P-93.1.3"]
    for tier in WIDER:
        got = _engine(tier).name_tiered(smiles)
        assert (got["name"], got["tier"]) == (expected, "systematic_verified"), (tier, got)
        # the (R,R)/(S,S) pair is never the meso compound
        assert opsin_key(got["name"]) != rdkit_key("C[C@H](O)[C@H](O)C")


def test_a_stereo_double_bond_in_a_group_is_not_shown_to_invert_to_itself():
    # fails closed: the group is cleared as before
    mol = Chem.MolFromSmiles("C[C@H](O)[C@H](O)C |o1:1,3|")
    group = mol.GetStereoGroups()[0]

    class _Bond:
        def GetStereo(self):
            return Chem.BondStereo.STEREOE

        def GetBondDir(self):
            return Chem.BondDir.NONE

    class _Group:
        def GetBonds(self):
            return [_Bond()]

        def GetAtoms(self):
            return group.GetAtoms()

    assert namer._stereo_group_inverts_to_itself(mol, group) is True
    assert namer._stereo_group_inverts_to_itself(mol, _Group()) is False
