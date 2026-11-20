"""Best-effort keeps the PIN tier's fusion name of a fused ring substituent (lane/c4c, 2026-11-20).

 "Five-membered ring requirement" (the Blue Book): "Fusion nomenclature gives preferred IUPAC names
only to compounds having at least two rings of at least five or more members... When fusion names are not allowed,
unsaturated von Baeyer ring system names are preferred IUPAC names". The best-effort tier's universal cage namer
(``rules.ring_substituents._polycyclic_substituent_name``, ``allow_mancude``) names such a fused substituent with its
von Baeyer name, and that name shipped at best-effort although the PIN tier names the molecule with the fusion name.
The namer now asks the PIN tier (``assembly.pin_first``), whose verified PIN is taken."""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def test_best_effort_takes_the_pin_tiers_fusion_name():
    smiles = "CN1CCC(CC1)OC1c2ccccc2CCc2ccccc12"
    name = "4-[(10,11-dihydro-5H-dibenzo[a,d][7]annulen-5-yl)oxy]-1-methylpiperidine"
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert (row["name"], row["tier"]) == (name, "pin_verified"), (tier, row)


@pytest.mark.parametrize("smiles,expected", [
    ("C1CC2CCC1C2", False),                       # norbornane: no aromatic ring
    ("c1ccc2c(c1)CCc1ccccc1C2", True),            # dibenzo[a,d][7]annulene, two benzo rings fused to the 7-ring
    ("c1ccccc1C1CC1", False),                     # no fused pair of rings of five or more
])
def test_which_ring_systems_take_a_fusion_name(smiles, expected):
    from rdkit import Chem

    from orthonym.rules.fusion_name_ask import takes_a_fusion_name as _takes_a_fusion_name
    assert _takes_a_fusion_name(Chem.MolFromSmiles(smiles)) is expected
