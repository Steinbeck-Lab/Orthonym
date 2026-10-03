""" (the Blue Book-17028): hydro prefixes express the hydrogenation of the MANCUDE
parent ring system (quinoline, indole, 2-benzofuran, 1,8-naphthyridine,...), never a fusion name
built from the saturated ring ('cyclohexa[b]pyridine'). (:12000): the six-membered
carbocyclic attached component is 'benzo'; 'cyclohexa' is not a fusion prefix, so a name that still
carries it is never labelled a PIN."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import name_is_rt_exact

CLASS_ROWS = [
    ("C1CCc2ncccc2C1", "5,6,7,8-tetrahydroquinoline"),
    ("C1CCc2cnccc2C1", "5,6,7,8-tetrahydroisoquinoline"),
    ("C1CCc2nccnc2C1", "5,6,7,8-tetrahydroquinoxaline"),
    ("C1CCc2[nH]ccc2C1", "4,5,6,7-tetrahydro-1H-indole"),
    ("C1CCc2cocc2C1", "4,5,6,7-tetrahydro-2-benzofuran"),
    ("CC1CCc2ncccc2C1", "6-methyl-5,6,7,8-tetrahydroquinoline"),
    ("CC1CCc2cocc2C1", "5-methyl-4,5,6,7-tetrahydro-2-benzofuran"),
    ("Cn1ccc2CCCCc21", "1-methyl-4,5,6,7-tetrahydro-1H-indole"),
    ("C1CC=Cc2ncccc21", "5,6-dihydroquinoline"),
    ("C1CNc2ncccc2C1", "1,2,3,4-tetrahydro-1,8-naphthyridine"),
    # the catalogue holds 2-benzothiophene and 1,2-benzothiazole now,
    # the Blue Book; 'hexahydro-2-benzothiophene-1,3-dione (PIN)',:32546)
    ("C1CCc2cscc2C1", "4,5,6,7-tetrahydro-2-benzothiophene"),
    ("C1CCc2sncc2C1", "4,5,6,7-tetrahydro-1,2-benzothiazole"),
]
CONTROL_ROWS = [
    ("C1CNc2ccccc2C1", "1,2,3,4-tetrahydroquinoline"),
    ("C1Cc2ccccc2CN1", "1,2,3,4-tetrahydroisoquinoline"),
]
# The name comes from the indicated-hydrogen path (5,8-dihydro, a suffix on the saturated ring):
# the 'cyclohexa' name is kept, labelled below PIN. (The 2-benzothiophene and 1,2-benzothiazole
# twins moved to CLASS_ROWS when the catalogue got those parents.)
CYCLOHEXA_NON_PIN_ROWS = [
    ("C1=CCc2ncccc2C1", "5H,8H-cyclohexa[b]pyridine"),
    ("OC(=O)C1CCc2ncccc2C1", "5H,7H,8H-cyclohexa[b]pyridine-6-carboxylic acid"),
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s0-hydro"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", CLASS_ROWS + CONTROL_ROWS)
def test_hydro_prefixes_on_the_mancude_parent(smiles, name):
    row = _row(smiles, "pin")
    assert row.get("name") == name, row.get("name")
    assert row["tier"] == "pin_verified", row["tier"]
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", CLASS_ROWS)
def test_no_cyclohexa_fusion_prefix_at_best_effort(smiles, name):
    be = _row(smiles, "best-effort")
    assert be.get("name") == name, be.get("name")
    assert "cyclohexa[" not in (be.get("name") or ""), be.get("name")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", CYCLOHEXA_NON_PIN_ROWS)
def test_a_cyclohexa_fusion_name_is_not_a_pin(smiles, name):
    pin = _row(smiles, "pin")
    assert pin["tier"] == "abstain", (pin.get("name"), pin["tier"])
    be = _row(smiles, "best-effort")
    assert be.get("name") == name, be.get("name")
    assert be["tier"] == "systematic_verified" and be["is_pin"] is False, be["tier"]
    assert name_is_rt_exact(name, smiles)
