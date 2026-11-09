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


# change-asserted-value (2026-11-09): the carbamate handler now makes the ring-nitrogen ester the parent when it
# outranks the other ester (handlers.carbamate._senior_ring_n_ester), so the PIN tier names the PIN this file's
# docstring derives instead of declining; independent check = OPSIN read-back (pin_verified); mutation = dropping
# _senior_ring_n_ester restores the decline and fails these rows.
@pytest.mark.parametrize("smiles, expected", [
    ("CC(C)(C)OC(=O)N1CCC(CC1)OC(=O)c1ccccc1", "tert-butyl 4-(benzoyloxy)piperidine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCC(CC1)OC(=O)C1CCCCC1", "tert-butyl 4-[(cyclohexanecarbonyl)oxy]piperidine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCC(COC(=O)c2ccccc2)CC1", "tert-butyl 4-[(benzoyloxy)methyl]piperidine-1-carboxylate"),
    # piperidine is a heterocycle: senior to naphthalene
    ("CC(C)(C)OC(=O)N1CCC(CC1)OC(=O)c1ccc2ccccc2c1", "tert-butyl 4-[(naphthalene-2-carbonyl)oxy]piperidine-1-carboxylate"),
])
def test_the_ring_nitrogen_ester_is_the_pin_parent(smiles, expected):
    pin = tier_row(smiles, "pin")
    assert (pin["name"], pin["tier"]) == (expected, "pin_verified"), pin


@pytest.mark.parametrize("smiles,expected", [
    ("COC(=O)c1ccccc1OC(C)=O", "methyl 2-(acetyloxy)benzoate"),
    # pyridine is senior to piperidine, the lower hydrogenation state): the pyridine ester is the parent
    ("CC(C)(C)OC(=O)N1CCC(CC1)OC(=O)c1ccncc1", "1-(tert-butoxycarbonyl)piperidin-4-yl pyridine-4-carboxylate"),
])
def test_acyloxy_on_the_acid_keeps_the_pin(smiles, expected):
    assert (tier_row(smiles, "pin")["name"], tier_row(smiles, "pin")["tier"]) == (expected, "pin_verified")
