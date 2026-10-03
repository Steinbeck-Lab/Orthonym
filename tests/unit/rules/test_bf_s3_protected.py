"""Slice S3 protection: names and declines the added-hydrogen rule must leave alone.

- A monovalent suffix on a =CH- atom of the mancude parent needs no hydrogen, so the lowest
  indicated hydrogen stays (b), the Blue Book, '1H-phenalen-4-ol (PIN)':3250):
  '2,3,4,7,8,8a-hexahydro-1H-3a,7-methanoazulen-4-ol'.
- A ketone on a saturated bridge atom needs none either (slice S1.8b,:28386):
  '1,2,3,4-tetrahydro-1,4-methanonaphthalen-9-one'.
- Not pseudoketones of the ring system, so the builder still declines: an N-acyl group on a
  ring nitrogen (an acyclic pseudoketone,:29370), a lactone beside an acyclic ester
   :32112 "A lactone, as a pseudoketone, ranks lower in the seniority of classes
  than an acid or an ester") and a lactone beside a nitrile:29628 "There is no
  seniority order difference between ketones and pseudoketones"; nitriles are senior to
  ketones,."""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.bridged_fused_pin import build

KEEP = [
    ("OC1C=CC2CC3CCCC13C2", "2,3,4,7,8,8a-hexahydro-1H-3a,7-methanoazulen-4-ol"),
    ("O=C1C2CCC1c1ccccc12", "1,2,3,4-tetrahydro-1,4-methanonaphthalen-9-one"),
]
DECLINE = [
    "CC(=O)N1CC2C3CCC(C3)C2C1",          # N-acetyl on a ring nitrogen
    "COC(=O)C1CC2CC1C1C(=O)OCC21",       # lactone + methyl ester: the ester is senior
    "CC(=O)OC1CC2CC1C1C(=O)OCC21",       # lactone + acetate ester: the ester is senior
    "N#CC1CC2CC1C1C(=O)OCC21",           # lactone + nitrile: the nitrile is senior
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s3"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", KEEP)
def test_names_the_added_hydrogen_rule_must_keep(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])


@pytest.mark.parametrize("smiles", DECLINE)
def test_groups_that_are_not_pseudoketones_of_the_ring_system_decline(smiles):
    assert build(Chem.MolFromSmiles(smiles)) is None
