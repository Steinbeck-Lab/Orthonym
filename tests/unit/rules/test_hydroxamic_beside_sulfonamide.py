"""An N-hydroxy amide beside a sulfonamide, sulfinamide, thioamide or hydrazide is the principal group.

 'Hydroxamic acids' (the Blue Book): "Hydroxamic acids have the generic structure R-CO-NH-OH and are
named as *N*-hydroxy amides" ('N-hydroxyacetamide (PIN)',:30148). Amides rank "in the order of the corresponding
acids" class 11,:18184; carboxylic before carbothioic and sulfonic acids), and hydrazides follow as
class 12, so the carboxamide is the parent and the sulfonamide is cited as a prefix. The perception order put the
hydroxamic acid below these classes, which gave '3-(hydroxyamino)-3-oxopropane-1-sulfonamide' a PIN label.
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("O=C(NO)CCS(N)(=O)=O", "N-hydroxy-3-sulfamoylpropanamide"),
    ("O=C(NO)c1ccc(S(N)(=O)=O)cc1", "N-hydroxy-4-sulfamoylbenzamide"),
    # belinostat
    ("O=C(/C=C/c1cccc(S(=O)(=O)Nc2ccccc2)c1)NO", "(2E)-N-hydroxy-3-[3-(phenylsulfamoyl)phenyl]prop-2-enamide"),
    ("CC(=O)NO", "N-hydroxyacetamide"),
])
def test_n_hydroxy_amide_is_the_parent(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


@pytest.mark.parametrize("smiles", [
    "O=C(NO)CCS(N)(=O)=O",
    "C1CC2=C(CC1NS(=O)(=O)C3=CC=CC(=C3)C(F)(F)F)C=C(C=C2)C(=O)NO",
    "O=C(NO)CCC(=O)NN",
    "O=C(NO)CCC(N)=S",
])
def test_no_pin_on_the_sulfonamide_or_hydrazide_parent(smiles):
    row = tier_row(smiles, "best-effort")
    name = row["name"].rstrip(")")
    assert not (row["tier"] == "pin_verified" and name.endswith(("sulfonamide", "hydrazide", "thioamide"))), row
