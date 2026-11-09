"""-NH-CO-S-R on a benzene ring is the carbamate prefix '[(alkylsulfanyl)carbonyl]amino' (program 14 lead, 2026-11-09).

 "Esters cited as prefixes" (the Blue Book): the ester of a carbamic or carbamothioic acid beside a
senior acid is cited as '(alkoxycarbonyl)amino', and its S-ester as '[(alkylsulfanyl)carbonyl]amino', as the chain
parents already give ('3-{[(methylsulfanyl)carbonyl]amino}propanoic acid'). The benzene writer read the S-ester as an
amido prefix ('2-[(methylsulfanyl)methanamido]benzoic acid', labelled below the PIN; the PIN tier declined).
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("CSC(=O)Nc1ccccc1C(=O)O", "2-{[(methylsulfanyl)carbonyl]amino}benzoic acid"),
    ("CCSC(=O)Nc1ccc(C(=O)O)cc1", "4-{[(ethylsulfanyl)carbonyl]amino}benzoic acid"),
    # unchanged: the O-ester and a plain amide
    ("COC(=O)Nc1ccccc1C(=O)O", "2-[(methoxycarbonyl)amino]benzoic acid"),
    ("CC(=O)Nc1ccccc1C(=O)O", "2-acetamidobenzoic acid"),
])
def test_the_s_ester_of_a_ring_nitrogen_is_a_carbamate_prefix(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")
