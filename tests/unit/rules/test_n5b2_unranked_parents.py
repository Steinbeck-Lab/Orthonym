"""A molecule with a parent of its own outside the ring system is not named by the fusion builder.

urea, thiourea, guanidine, carbamic acid and its esters, cyanamide and carbamimidic esters are
parent hydrides of their own that ``rules/seniority.SENIORITY_ORDER`` does not rank, so
``get_principal_group`` answers None for a molecule that carries one and the builder read "no
principal group, the ring is the parent". (the Blue Book): "The preferred
IUPAC name for the 'amidine' related to carbonic acid, H2N-C(=NH)-NH2, is the retained name
'guanidine'"; (:34266): "In the presence of a characteristic group having seniority
over guanidine (see item 11 in, the following prefixes are used": 'carbamimidoylamino'
(preferred prefix). The substituent namer spells 'guanidinyl', which the Blue Book does not print,
and the cyanamide group as 'carbamoyl' (another molecule), so the builder declines when no ranked
group is present (the guanidine, urea or carbamate is then the parent) and for a cyanamide. When a
senior group makes the ring the parent, the builder keeps building and a name that carries the
'guanidinyl' prefix is recorded as a non-PIN fragment (demoted, not deleted): the default tier does
not ship it and best-effort keeps the engine's own 'carbamimidoylamino' form."""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import build_fused
from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort


@pytest.mark.opsin_gate
def test_a_cyclic_carbamate_is_a_pseudoketone_not_an_unranked_parent():
    """The guanidine/urea/carbamate guard looks at connecting atoms outside the ring system, so a
    cyclic carbamate or urea of the ring system is still named:29314)."""
    s = "O=C1Nc2ccc(Cl)cc2[C@@](C#CC2CC2)(C(F)(F)F)O1"
    got = build_fused(Chem.MolFromSmiles(s))
    assert got is not None and got[0].endswith("-1,4-dihydro-2H-3,1-benzoxazin-2-one"), got
    assert_full_rt(got[0], s)


@pytest.mark.parametrize("smiles", [
    "Cc1cc2nc(N=C(N)N)nc(C)c2cc1C", "NC(=N)NC1CCC2CCCC2C1", "N#CNc1ccc2OCOc2c1",
])
def test_a_parent_of_its_own_outside_the_ring_system_declines(smiles):
    """guanidine, cyanamide (and urea, carbamate): (:34266) spells the prefix
    'carbamimidoylamino', never the 'guanidinyl' the substituent namer gives; whether the ring or
    that parent is the parent of the molecule is the engine's choice."""
    assert build_fused(Chem.MolFromSmiles(smiles)) is None


@pytest.mark.parametrize("smiles", [
    "Cc1cc2nc(N=C(N)N)nc(C)c2cc1C",             # guanidine is a parent of its own
])
@pytest.mark.opsin_gate
def test_the_engine_never_spells_guanidinyl(smiles):
    row = default_tier_row(smiles)
    assert row["tier"] != "pin_verified" or "guanidin" not in (row["name"] or ""), row
    be = name_best_effort(smiles)
    if be["name"] and not be["name"].startswith("unknown"):
        assert_full_rt(be["name"], smiles)


# a senior group (the acid) makes the ring system the parent and the group a prefix:
# (the Blue Book, "In the presence of a characteristic group having seniority over guanidine
# (see item 11 in, the following prefixes are used": 'carbamimidoylamino'); urea, 'carbamoylamino'
# (seniority.py prefix table,:33320)
SENIOR_GROUP_PRESENT = [
    ("OC(=O)c1ccc2c(c1)CCC2NC(N)=N", "1-(carbamimidoylamino)-2,3-dihydro-1H-indene-5-carboxylic acid"),
    ("OC(=O)C1CC2CCCCC2C1NC(N)=O", "1-(carbamoylamino)octahydro-1H-indene-2-carboxylic acid"),
]


@pytest.mark.parametrize("smiles,expected", SENIOR_GROUP_PRESENT)
@pytest.mark.opsin_gate
def test_a_senior_group_keeps_the_ring_as_the_parent_at_best_effort(smiles, expected):
    row = default_tier_row(smiles)
    assert "guanidinyl" not in (row["name"] or "") or row["tier"] != "pin_verified", row
    be = name_best_effort(smiles)
    assert be["name"] == expected, be
    assert_full_rt(expected, smiles)


@pytest.mark.opsin_gate
def test_the_guanidinyl_prefix_is_never_a_pin_beside_a_senior_group():
    from orthonym.metrics import provenance
    smiles = "OC(=O)c1ccc2c(c1)CCC2NC(N)=N"
    got = build_fused(Chem.MolFromSmiles(smiles))
    assert got is not None and got[0] == "1-guanidinyl-2,3-dihydro-1H-indene-5-carboxylic acid", got
    assert "guanidinyl" in provenance._NON_PIN_FRAGMENTS.get()
    assert default_tier_row(smiles)["tier"] != "pin_verified"


def test_a_cyanamide_beside_a_senior_group_declines():
    """the substituent namer spells -NH-C#N as 'carbamoyl', another molecule"""
    assert build_fused(Chem.MolFromSmiles("OC(=O)c1ccc2c(c1)CCC2NC#N")) is None
