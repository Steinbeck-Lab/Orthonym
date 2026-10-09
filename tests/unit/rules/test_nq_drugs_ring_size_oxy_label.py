"""The label of a benzene name that carries a ring-size oxy group is decided by the whole name
the benzene composer built, not by a 14-character fragment of it.

Where a ring assembly, the Blue Book) may be the parent, the benzene parent
built around '(oxan-2-yl)oxy' / '(oxolan-2-yl)oxy' is not the PIN: (:19461), "The
senior ring or ring system has the greater number of rings"; the assembly acid is
'...[1,1'-biphenyl]-2-carboxylic acid (PIN)' (:49805). The group's writer says so
(``'parent_may_not_be_pin'``); ``name_substituted_benzene`` records the whole name. A
fragment record lowered the label of every name that contains the fragment, the senior
assembly name of the same molecule included (read back with OPSIN 2.9.0: the two assembly
names below give the InChIKeys of the two molecules).
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.metrics import provenance as P

pytestmark = pytest.mark.opsin_gate

# (SMILES, the benzene name the producer builds, the assembly name that is senior)
ROWS = [
    ("OC(=O)c1ccc(OC2CCCCO2)cc1-c1ccccc1",
     "4-[(oxan-2-yl)oxy]-2-phenylbenzoic acid",
     "5-[(oxan-2-yl)oxy][1,1'-biphenyl]-2-carboxylic acid"),
    ("OC(=O)c1cc(OC2CCCCO2)ccc1-c1ccccc1",
     "5-[(oxan-2-yl)oxy]-2-phenylbenzoic acid",
     "4-[(oxan-2-yl)oxy][1,1'-biphenyl]-2-carboxylic acid"),
]


def _scope(smiles, tier):
    namer = (Orthonym(style="pin") if tier == "pin"
             else Orthonym(style="pin", **_emit_tier_flags(tier)))
    P.clear_provenance()
    row = namer.name_tiered(smiles)
    return row, P.get_provenance()


@pytest.mark.parametrize("smiles,built,assembly", ROWS)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_record_is_the_whole_name_not_the_fragment(smiles, built, assembly, tier):
    row, prov = _scope(smiles, tier)
    assert P.name_carries_non_pin_part(prov, built), prov
    # the name-scoped record is the whole name the composer built, not the fragment; the
    # whole-call flag (fail closed: no text of the benzene name survives the ester and salt
    # composers) also lowers a name of the assembly parent, which no producer builds today
    assert built in prov["non_pin_labels"], prov
    assert "(oxan-2-yl)oxy" not in prov["non_pin_labels"], prov
    assert prov["general_ring_prefix"] is True, prov


@pytest.mark.parametrize("smiles,built,assembly", ROWS)
def test_the_names_and_tiers_of_the_class_are_unchanged(smiles, built, assembly):
    assert _scope(smiles, "pin")[0]["tier"] == "abstain"
    row = _scope(smiles, "best-effort")[0]
    assert (row["name"], row["tier"]) == (built, "systematic_verified"), row


def test_the_oxolanyl_row_keeps_its_name_and_label():
    row = _scope("OC(=O)c1ccc(OC2CCCO2)cc1-c1ccccc1", "best-effort")[0]
    assert (row["name"], row["tier"]) == (
        "4-[(oxolan-2-yl)oxy]-2-phenylbenzoic acid", "systematic_verified"), row


@pytest.mark.parametrize("smiles,name", [
    ("OC(=O)c1ccc(OC2CCCCO2)cc1", "4-[(oxan-2-yl)oxy]benzoic acid"),
    ("CC(COC1CCCCO1)C(=O)[O-]", "2-methyl-3-[(oxan-2-yl)oxy]propanoate"),
])
def test_a_name_without_the_class_keeps_its_pin(smiles, name):
    row, prov = _scope(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert prov["non_pin_labels"] == (), prov


# the esters, salts and anions of the class: the later composers rewrite '...benzoic acid' to
# '...benzoate', so the text of the benzene composer's name does not survive to the label
# step; the structural fact (a benzene parent where an assembly may be the parent,
# the Blue Book; the assembly acid:49803) lowers them at every tier
REWRITTEN_ROWS = [
    ("COC(=O)c1ccc(OC2CCCCO2)cc1-c1ccccc1", "methyl 4-[(oxan-2-yl)oxy]-2-phenylbenzoate"),
    ("[Na+].[O-]C(=O)c1ccc(OC2CCCCO2)cc1-c1ccccc1",
     "sodium 4-[(oxan-2-yl)oxy]-2-phenylbenzoate"),
    ("CCOC(=O)c1cc(OC2CCCO2)ccc1-c1ccccc1", "ethyl 5-[(oxolan-2-yl)oxy]-2-phenylbenzoate"),
]


@pytest.mark.parametrize("tier", ["pin", "valid", "complete", "best-effort"])
@pytest.mark.parametrize("smiles,name", REWRITTEN_ROWS)
def test_a_rewritten_name_of_the_class_is_never_pin_verified(tier, smiles, name):
    from tests.support.rt_assert import assert_full_rt
    row, _ = _scope(smiles, tier)
    assert row["tier"] != "pin_verified", row
    if tier != "pin":
        assert row["name"] == name, row
        assert_full_rt(row["name"], smiles)


def test_the_ester_without_the_flag_keeps_its_pin():
    row, _ = _scope("COC(=O)c1ccc(OC2CCCCO2)cc1", "pin")
    assert (row["name"], row["tier"]) == (
        "methyl 4-[(oxan-2-yl)oxy]benzoate", "pin_verified"), row
