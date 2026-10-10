"""Leads program item 7, part (a) (lane L5): the assembly tier cites the ring C=O of a fusion parent
as the '-one' suffix when no group senior to ketones is present.

Venadaparib (and its azetidine analogue without the phane) is named at the valid, complete and
best-effort tiers by the general engine's assembly tier (``_name_terminal_ring_assembly``), which used to
cite every group as a prefix: '4-[...]-1-oxo-1,2-dihydrophthalazine'. (the Blue Book)
names a ketone of a mancude parent by 'added indicated hydrogen' ('naphthalen-1(2H)-one (PIN)',:28418),
 (PSEUDOKETONES,:29370, sentence: "Acyclic pseudoketones, including those in which the carbonyl
group is linked to a heteroatom of a heterocycle (hidden amides, for instance), are named substitutively
by using the suffix 'one' to indicate the principal function.") counts the ring lactams and the hidden
amides in that class, and (:24862) allows 'oxo' only for a group that cannot be the suffix.
The hidden amide is the acyl prefix of the ring that holds its nitrogen ('azetidine-1-carbonyl').

The PIN tier is untouched: venadaparib's phane PIN stays declined (the label is the general engine's
'systematic_verified'). Part (b), moving the hidden amide into the ketone class in perception, is a
separate item (it gives von Baeyer names without a fusion-before-von-Baeyer guarantee).
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from tests.support.rt_assert import assert_full_rt

pytestmark = pytest.mark.opsin_gate

VENADAPARIB = "C1(CC1)NCC1CN(C1)C(=O)C=1C=C(C=CC1F)CC1=NNC(C2=CC=CC=C12)=O"
ANALOGUE = "O=C1NN=C(Cc2cccc(C(=O)N3CCC3)c2)c2ccccc12"

ROWS = [
    (VENADAPARIB,
     "4-[(3-{3-[(cyclopropylamino)methyl]azetidine-1-carbonyl}-4-fluorophenyl)methyl]phthalazin-1(2H)-one"),
    (ANALOGUE, "4-{[3-(azetidine-1-carbonyl)phenyl]methyl}phthalazin-1(2H)-one"),
]


def _row(smiles, tier):
    if tier == "pin":
        return Orthonym(style="pin").name_tiered(smiles)
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.parametrize("tier", ["valid", "complete", "best-effort"])
@pytest.mark.parametrize("smiles,expected", ROWS)
def test_the_ring_one_is_the_suffix_with_added_hydrogen(smiles, expected, tier):
    row = _row(smiles, tier)
    assert row["name"] == expected and row["tier"] == "systematic_verified", row
    assert row["source"] == "general_engine", row
    assert_full_rt(expected, smiles)


@pytest.mark.parametrize("smiles", [VENADAPARIB, ANALOGUE])
def test_the_pin_tier_still_declines(smiles):
    # the PIN is a pseudoketone/phane name this writer does not build; nothing wrong ships
    assert _row(smiles, "pin")["tier"] == "abstain"


@pytest.mark.parametrize("smiles,expected", [
    # no hidden amide: the PIN path names the ring itself
    ("O=C1NN=C(Cc2ccccc2)c2ccccc12", "4-benzylphthalazin-1(2H)-one"),
])
def test_control_the_pin_path_is_unchanged(smiles, expected):
    row = _row(smiles, "pin")
    assert row["name"] == expected and row["tier"] == "pin_verified", row


def test_a_group_senior_to_ketones_keeps_the_prefix_spelling():
    # a carboxylic acid outranks ketones, the Blue Book): the ring C=O stays 'oxo'
    from rdkit import Chem

    from orthonym.assembly.general_engine import _ketone_class_ring_suffix_atoms
    from orthonym.perception.functional_groups import detect_functional_groups

    class _F:  # the part of the features the helper reads
        pass

    for smiles, expect_suffix in (
            ("O=C1NN=C(Cc2cccc(C(=O)N3CCC3)c2)c2ccccc12", True),    # hidden amide only
            ("O=C1NN=C(Cc2cccc(C(=O)O)c2)c2ccccc12", False),         # carboxylic acid
            ("O=C1NN=C(Cc2cccc(C#N)c2)c2ccccc12", False),            # nitrile
            ("O=C1NN=C(Cc2cccc(C=O)c2)c2ccccc12", False),            # aldehyde
            ("O=C1NN=C(Cc2cccc(O)c2)c2ccccc12", True),               # a phenol is junior
            # perceived classes outside SENIORITY_ORDER that rank above ketones: an acyclic urea
            #, the Blue Book), guanidine,:34266), carbamate
            # (an ester, class 9 of Table 4.1,:18182)
            ("O=C1NN=C(Cc2cccc(C(=O)N3CCC3)c2NC(N)=O)c2ccccc12", False),
            ("O=C1NN=C(Cc2cccc(C(=O)N3CCC3)c2NC(=O)OC)c2ccccc12", False),
            ("O=C1NN=C(Cc2cccc(C(=O)N3CCC3)c2NC(N)=N)c2ccccc12", False),
            ("O=C1NN=C(Cc2cccc(C(=O)N3CCC3)c2NC#N)c2ccccc12", False),
            # a prefix-only group (isocyanato) ranks nowhere: the ring '-one' stays the suffix
            ("O=C1NN=C(Cc2cccc(C(=O)N3CCC3)c2N=C=O)c2ccccc12", True),
    ):
        mol = Chem.MolFromSmiles(smiles)
        feats = _F()
        feats.functional_groups = detect_functional_groups(mol)
        ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
        got = _ketone_class_ring_suffix_atoms(mol, feats, ring)
        assert bool(got) == expect_suffix, (smiles, got)


@pytest.mark.parametrize("smiles", [
    "O=C1NN=C(Cc2cccc(C(=O)N3CCC3)c2NC(N)=O)c2ccccc12",         # urea
    "O=C1NN=C(Cc2cccc(C(=O)N3CCC3)c2NC(=O)OC)c2ccccc12",        # carbamate
    "O=C1NN=C(Cc2cccc(C(=O)N3CCC3)c2NC(N)=N)c2ccccc12",         # guanidine
    "O=C1NN=C(Cc2cccc(C(=O)N3CCC3)c2NC#N)c2ccccc12",            # cyanamide
])
def test_a_group_senior_to_ketones_outside_the_seniority_table_keeps_the_ring_one_a_prefix(smiles):
    # the ring '-one' is cited as the suffix only when no group senior to ketones is present: the
    # name keeps 'oxo' on the ring, the Blue Book) and reads back exactly
    for tier in ("valid", "best-effort"):
        row = _row(smiles, tier)
        assert row["tier"] == "systematic_verified", row
        assert "phthalazin-1(2H)-one" not in row["name"], row
        assert "1-oxo-1,2-dihydrophthalazine" in row["name"], row
        assert_full_rt(row["name"], smiles)
