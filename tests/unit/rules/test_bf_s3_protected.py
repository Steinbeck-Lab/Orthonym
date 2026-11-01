"""Slice S3 protection: names and declines the added-hydrogen rule must leave alone.

- A monovalent suffix on a =CH- atom of the mancude parent needs no hydrogen, so the lowest
  indicated hydrogen stays (b), the Blue Book, '1H-phenalen-4-ol (PIN)':3250):
  '2,3,4,7,8,8a-hexahydro-1H-3a,7-methanoazulen-4-ol'.
- A ketone on a saturated bridge atom needs none either (slice S1.8b,:28386):
  '1,2,3,4-tetrahydro-1,4-methanonaphthalen-9-one'.
- Not pseudoketones of the ring system, so the builder still declines: an N-acyl group on a
  ring nitrogen (an acyclic pseudoketone,:29370) and a lactone beside an acyclic ester
   :32112 "A lactone, as a pseudoketone, ranks lower in the seniority of classes
  than an acid or an ester").
- A lactone beside a nitrile (change-asserted-value, lane DT5 2026-11-01): the row used to decline here
  as "the nitrile is senior". It is: (the Blue Book) puts nitriles in class 14 and
  "ketones..., pseudoketones..." in class 16:29628 "There is no seniority order difference
  between ketones and pseudoketones"), so the nitrile takes the suffix and the lactone is the 'oxo' prefix,
  as (:32120) prints for a lactone beside an acid: '8-oxo-7-oxabicyclo[4.2.0]octane-4,5-
  dicarboxylic acid (PIN)'. Independent check: OPSIN 2.9.0 and main read the name below back to the
  input's full InChIKey (a fresh parse outside the engine, ``PSEUDOKETONE_BESIDE_NITRILE``); mutation:
  ``bridged_fused_pin._pseudoketone_principal`` returning None for a senior group fails the test."""
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
]
#: a lactone beside a nitrile: the nitrile (class 14) takes the suffix, the lactone (class 16) is 'oxo'
PSEUDOKETONE_BESIDE_NITRILE = [
    ("N#CC1CC2CC1C1C(=O)OCC21", "3-oxooctahydro-4,7-methano-2-benzofuran-5-carbonitrile"),
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


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", PSEUDOKETONE_BESIDE_NITRILE)
def test_a_lactone_beside_a_nitrile_is_the_oxo_prefix(smiles, name, tier):
    from tests.support.rt_assert import name_is_rt_exact
    assert name_is_rt_exact(name, smiles)
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
