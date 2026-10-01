""" stereoparent names: identical prefixes are multiplied, and a retained parent keeps its spelling.

 (the Blue Book-2786): multiplicative prefixes "denote multiplicity of identical features";
 (b) (:7067): 'di', 'tri',... multiply simple substituent prefixes -- a C-methyl and an
N-methyl are one prefix group ('7,17-dimethyl', not '7-methyl-17-methyl'). (:51007):
"morphinan and ibogamine, are exceptions and treated as retained names"; the Blue Book writes
'4,5alpha-epoxymorphinan' (:52332) and '...-17-methyl-7,8-didehydromorphinan-3,6alpha-diol'
(:2680), never 'morphinane'. The names keep the pin_verified label (user decision).
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import name_is_rt_exact

ROWS = [
    ("COc1ccc2c3c1O[C@H]1[C@@H](OC)[C@H](C)C[C@H]4[C@@H](C2)N(C)CC[C@@]341",
     "(5R,6S,7R,9R,13S,14R)-4,5-epoxy-3,6-dimethoxy-7,17-dimethylmorphinan"),
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


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", ROWS + CONTROL_ROWS)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_p101_stereoparent_spelling(smiles, name, tier):
    row = _row(smiles, tier)
    assert row.get("name") == name, row.get("name")
    assert row["tier"] == "pin_verified", row["tier"]
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
