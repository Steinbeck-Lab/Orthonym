""" stereoparent names: identical prefixes are multiplied, and a retained parent keeps its spelling.

 (the Blue Book-2786): multiplicative prefixes "denote multiplicity of identical features";
 (b) (:7067): 'di', 'tri',... multiply simple substituent prefixes -- a C-methyl and an
N-methyl are one prefix group ('7,17-dimethyl', not '7-methyl-17-methyl'). (:51007):
"morphinan and ibogamine, are exceptions and treated as retained names"; the Blue Book writes
'4,5alpha-epoxymorphinan' (:52332) and '...-17-methyl-7,8-didehydromorphinan-3,6alpha-diol'
(:2680), never 'morphinane'. Slice S4: a morphinan without the 4,5-epoxy bridge has a bridged
fused PIN that needs the 'azanoethano' bridge prefix,:14155 "-NH-CH2-CH2-
(azanoethano) (preferred prefix)"); OPSIN 2.9.0 reads the general-nomenclature 'epiminoethano'
but not the preferred 'azanoethano', so no PIN is built; its
name is labelled below the PIN,:50943; the user's decision on natural-product names,
"the default tier declines a natural-product name only where no systematic PIN is built"): the
best-effort tier keeps it (systematic_verified), the default tier declines it. The tropane name
(no bridged fused reading) keeps the pin_verified label.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import name_is_rt_exact

#: a 4,5-epoxymorphinan: the strict path builds its bridged fused PIN (slice S4;
#: the Blue Book), which replaces the name
#: '(5R,6S,7R,9R,13S,14R)-4,5-epoxy-3,6-dimethoxy-7,17-dimethylmorphinan'
PIN_ROWS = [
    ("COc1ccc2c3c1O[C@H]1[C@@H](OC)[C@H](C)C[C@H]4[C@@H](C2)N(C)CC[C@@]341",
     "(4R,4aR,6R,7S,7aR,12bS)-7,9-dimethoxy-3,6-dimethyl-2,3,4,4a,5,6,7,7a-octahydro-1H-4,12-methano[1]benzofuro[3,2-e]isoquinoline"),
]
ROWS = [
    ("Cc1ccc2c(c1)[C@@]13CCCC[C@@H]3[C@@H](C2)N(C)CC1",
     "(9R,13S,14S)-3,17-dimethylmorphinan"),
    ("Cc1ccc2c(c1)[C@@]13CC[C@H](C)C[C@@H]3[C@@H](C2)N(C)CC1",
     "(7S,9R,13S,14S)-3,7,17-trimethylmorphinan"),
    ("Oc1ccc2c(c1)[C@@]13CC[C@H](O)C[C@@H]3[C@@H](C2)N(C)CC1",
     "(7S,9R,13S,14S)-17-methylmorphinan-3,7-diol"),
]
CONTROL_ROWS = [  # different prefixes stay separate; the unchanged forms stay
    ("COc1ccc2c(c1)[C@@]13CCCC[C@@H]3[C@@H](C2)N(C)CC1",
     "(9R,13S,14S)-3-methoxy-17-methylmorphinan"),
    ("Cc1ccc2c(c1)[C@@]13CCCC[C@@H]3[C@@H](C2)N(CC)CC1",
     "(9R,13S,14S)-17-ethyl-3-methylmorphinan"),
    ("Oc1ccc2c(c1)[C@@]13CCCC[C@@H]3[C@@H](C2)N(C)CC1",
     "(9R,13S,14S)-17-methylmorphinan-3-ol"),
    ("CN1[C@@H]2CC[C@H]1C[C@H](O)[C@@H]2O",
     "(1R,2R,3S,5S)-tropane-2,3-diol"),
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s0-p101"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


#: the morphinans without the 4,5-epoxy bridge (no PIN is built): below the PIN
BELOW_PIN = ROWS + CONTROL_ROWS[:3]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", ROWS + CONTROL_ROWS + PIN_ROWS)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_p101_stereoparent_spelling(smiles, name, tier):
    row = _row(smiles, tier)
    below = (smiles, name) in BELOW_PIN
    if below and tier == "pin":
        assert (row["tier"], row.get("limit_code")) == ("abstain", "NO_VERIFIED_PIN"), row
        return
    assert row.get("name") == name, row.get("name")
    assert row["tier"] == ("systematic_verified" if below else "pin_verified"), row["tier"]
    assert name_is_rt_exact(name, smiles)


def test_saturated_parent_ending_follows_the_retained_name():
    from orthonym.rules.natural_products import _saturated_parent_e

    assert _saturated_parent_e("morphin", "morphinan", "diol") == ""
    assert _saturated_parent_e("morphin", "morphinan", "") == ""
    assert _saturated_parent_e("androst", "androstane", "diol") == "e"
    assert _saturated_parent_e("androst", "androstane", "ol") == ""
    assert _saturated_parent_e("trop", "tropane", "") == "e"


BARE_PARENT_ROWS = [  # (a) spellings (the Blue Book,:51392,:51396)
    ("C=C[C@H]1C[N@@]2CC[C@H]1C[C@@H]2Cc1ccnc2ccccc12", "cinchonan"),
    ("c1ccc2c(c1)CC1c3ccccc3CCN1C2", "berbine"),
    ("CC[C@@H]1CN2[C@H]3C[C@]45C[C@H]3[C@H]1C[C@H]2[C@@H]4N(C)c1ccccc15", "ajmalan"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", BARE_PARENT_ROWS)
def test_bare_alkaloid_parent_is_spelled_as_the_parent_table(smiles, name):
    row = _row(smiles, "pin")
    assert row.get("name") == name and row["tier"] == "pin_verified", (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
def test_name_exact_hetisan_is_spelled_as_the_parent_table():
    """'hetisan' (:51392,:57087) is a name-exact list parent: OPSIN reads neither spelling, the
    name is shipped on the exact canonical SMILES."""
    row = _row("C=C1C[C@]23C[C@H]4[C@@H]5[C@@]6(C)CCC[C@]57C(C2C[C@H]1C[C@H]37)N4C6", "pin")
    assert row.get("name") == "hetisan" and row["tier"] == "pin_verified", (row.get("name"), row["tier"])
