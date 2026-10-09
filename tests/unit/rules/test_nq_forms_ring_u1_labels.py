"""Item 12a (labels): a pin_verified label at a wider tier needs the default tier to emit
the same string. The strict twin of ``Orthonym._strict_pin_twin_name`` runs with the default
tier's emission rule: a string the strict path builds but the default tier declines
(NO_VERIFIED_PIN) comes back as the label, never equal to the name, so the label is demoted.
Hard rule of the project: a PIN label only for a certified PIN."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

pytestmark = pytest.mark.opsin_gate

BE = lambda: Orthonym(style="pin", **_emit_tier_flags("best-effort"))  # noqa: E731

#: rows the default tier declines (NO_VERIFIED_PIN) whose best-effort name was labelled
#: pin_verified once the 'a' ring (recorded as non-PIN) gave way to the book ring
DECLINED = [
    "COC(=O)CCc1cc(O)c(C)c(=O)o1",
    "COC(=O)CCCc1c(OC)cc(=O)oc1C",
    "COC(=O)/C=C/c1oc(=O)c(CO)c(OC)c1C=O",
]


@pytest.mark.parametrize("smiles", DECLINED)
def test_a_name_the_default_tier_declines_is_not_pin_verified_at_best_effort(smiles):
    assert Orthonym(style="pin").name_tiered(smiles)["tier"] == "abstain"
    row = BE().name_tiered(smiles)
    assert row["name"] and row["tier"] != "pin_verified" and row["is_pin"] is False


@pytest.mark.parametrize("smiles", ["COC(=O)CCc1ccccc1", "CC(=O)Nc1ccc(O)cc1"])
def test_a_name_the_default_tier_emits_keeps_its_label_at_best_effort(smiles):
    d = Orthonym(style="pin").name_tiered(smiles)
    b = BE().name_tiered(smiles)
    assert d["tier"] == "pin_verified" and b["tier"] == "pin_verified" and d["name"] == b["name"]
