"""Benzo rings ortho-fused to ovalene or coronene, named with the lettered peripheral sides of the parent.

 (the Blue Book): the peripheral sides are lettered beginning with 'a' for the side 1,2, the letters
as low as possible; (b): the parent component has the most rings (ovalene, 10; coronene, 7), and a system
that holds another component with as many rings keeps the name below the PIN."""
import pytest
from rdkit import Chem

from orthonym.cli import _tiered_namer
from orthonym.validation.opsin_roundtrip import opsin_parse
from tests.support.rt_assert import name_best_effort

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _smiles(name):
    return Chem.MolToSmiles(Chem.MolFromSmiles(opsin_parse(name)))


@pytest.mark.parametrize("name", [
    "hexabenzo[a,d,g,l,o,r]ovalene",
    "benzo[a]ovalene",
    "benzo[a]coronene",
    "tribenzo[a,d,g]coronene",
    "dibenzo[a,j]coronene",
])
def test_benzo_parents_are_pins(name):
    row = _tiered_namer("pin", "pin").name_tiered(_smiles(name))
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


def test_a_second_seven_ring_component_keeps_the_name_below_the_pin():
    """trinaphthylene (7 rings) lies in this system as coronene does: (b) does not decide the base."""
    smiles = _smiles("hexabenzo[a,d,g,j,m,p]coronene")
    assert _tiered_namer("pin", "pin").name_tiered(smiles)["tier"] == "abstain"
    row = name_best_effort(smiles)
    assert row["name"] == "hexabenzo[a,d,g,j,m,p]coronene" and not row["is_pin"], row
