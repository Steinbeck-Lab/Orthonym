"""Slice S4 protection: what the morphine-class build must leave alone, at every task.

- Natural products without a bridged fused reading keep their names and labels at both
  tiers: 'cholesterol' (the exact-match table), 'cholesta-4,6-dien-3beta-ol' (a
  steroid name; a dev split and milestone1500), 'germacrane' (the exact-match list of
  parents, a monocycle). 'hetisan', an exact-match list parent the bridged fused builder does
  not name, keeps its name at the default tier (best-effort ships a verified von Baeyer
  name, as before).
- The best-effort tier keeps the name of every class member the builder does not complete
  (an ester, a morphinan without the 4,5-epoxy bridge; the ring ketones are named since slice
  S3, ``test_bf_s3_morphinan.py``), each read back to the
  input's full InChIKey; only its label may change (Task S4.4)."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import name_is_rt_exact

KEEP_BOTH_TIERS = [
    ("C[C@H](CCCC(C)C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C", "cholesterol"),
    ("CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3C=CC4=C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
     "cholesta-4,6-dien-3β-ol"),
    ("CC(C)[C@@H]1CC[C@H](C)CCC[C@H](C)CC1", "germacrane"),
]
HETISAN = "C=C1C[C@]23C[C@H]4[C@@H]5[C@@]6(C)CCC[C@]57C(C2C[C@H]1C[C@H]37)N4C6"

KEEP_BEST_EFFORT_NAME = [
    ("CN1CC[C@]23c4c5ccc(OC(C)=O)c4O[C@H]2[C@@H](OC(C)=O)C=C[C@H]3[C@H]1C5", "diamorphine"),
    ("COc1ccc2c(c1)[C@@]13CCCC[C@@H]3[C@@H](C2)N(C)CC1", "(9R,13S,14S)-3-methoxy-17-methylmorphinan"),
    ("Oc1ccc2c(c1)[C@@]13CCCC[C@@H]3[C@@H](C2)N(C)CC1", "(9R,13S,14S)-17-methylmorphinan-3-ol"),
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s4"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", KEEP_BOTH_TIERS)
def test_natural_products_outside_the_class_keep_name_and_label(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])


@pytest.mark.opsin_gate
def test_an_exact_match_list_parent_the_builder_does_not_name_keeps_its_name():
    row = _row(HETISAN, "pin")
    assert (row.get("name"), row["tier"]) == ("hetisan", "pin_verified"), (row.get("name"), row["tier"])


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", KEEP_BEST_EFFORT_NAME)
def test_best_effort_keeps_the_name_of_members_the_builder_does_not_complete(smiles, name):
    row = _row(smiles, "best-effort")
    assert row.get("name") == name, (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)
