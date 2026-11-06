"""An ester name whose alcohol part holds the ester of a second acid of its own skeleton is not labelled the PIN.

'1-(tert-butoxycarbonyl)piperidin-4-yl benzoate' names the right molecule, but the molecule is the ester of two acids and
the senior acid is the parent of the PIN: ('phenyl 3-(benzoyloxy)benzoate (PIN) [not
3-(phenoxycarbonyl)phenyl benzoate...]'), the ring-nitrogen acid a carboxylic acid (the Blue Book, "the carboxylic
acid is senior to the carbonic acid derivative") and piperidine senior to benzene, so the PIN is
'tert-butyl 4-(benzoyloxy)piperidine-1-carboxylate'. The decomposition ester assembler does not make that choice, so its
name is labelled below the PIN when a ring-nitrogen ester outranks the ester the name is built on
(``decomposition.ester_in_alcohol``, ranked by ``rules.ester_seniority`` with the ring-nitrogen acid ranked by its ring).
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles", [
    "CC(C)(C)OC(=O)N1CCC(CC1)OC(=O)c1ccccc1",
    "CC(C)(C)OC(=O)N1CCC(CC1)OC(=O)C1CCCCC1",
    "CC(C)(C)OC(=O)N1CCC(COC(=O)c2ccccc2)CC1",
    "CC(C)(C)OC(=O)N1CCC(CC1)OC(=O)c1ccc2ccccc2c1",   # piperidine is a heterocycle: senior to naphthalene
])
def test_not_a_pin(smiles):
    pin = tier_row(smiles, "pin")
    assert pin["tier"] != "pin_verified", pin
    best = tier_row(smiles, "best-effort")
    assert best["tier"] in ("systematic_verified", "pin_unverified", "best_effort"), best


@pytest.mark.parametrize("smiles,expected", [
    ("COC(=O)c1ccccc1OC(C)=O", "methyl 2-(acetyloxy)benzoate"),
    # pyridine is senior to piperidine, the lower hydrogenation state): the pyridine ester is the parent
    ("CC(C)(C)OC(=O)N1CCC(CC1)OC(=O)c1ccncc1", "1-(tert-butoxycarbonyl)piperidin-4-yl pyridine-4-carboxylate"),
])
def test_acyloxy_on_the_acid_keeps_the_pin(smiles, expected):
    assert (tier_row(smiles, "pin")["name"], tier_row(smiles, "pin")["tier"]) == (expected, "pin_verified")
