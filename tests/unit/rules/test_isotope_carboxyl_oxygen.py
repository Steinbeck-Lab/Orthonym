"""A heavy oxygen on the =O of a carboxy group is cited without a locant, a hydrogen nuclide on its hydroxy oxygen by
the italic letter: (the Blue Book) 'CH3-C18O-O2H (O-2H,18O)acetic acid (PIN)'. OPSIN reads the
locant-free 18O as ambiguous; the book prints it, so the warning is licensed for this class (one carboxy group, or
several the structure cannot tell apart)."""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("[2H]OC(C)=[18O]", "(O-2H,18O)acetic acid"),
    ("CCC(=[18O])O[2H]", "(O-2H,18O)propanoic acid"),
    ("[2H]OC(=[18O])c1ccccc1", "(O-2H,18O)benzoic acid"),
    ("CC(=[18O])O", "(18O)acetic acid"),
    ("CC(=O)[18OH]", "(O-18O)acetic acid"),
])
def test_carboxyl_oxygen_nuclides(smiles, name):
    assert_pin_at_both_tiers(smiles, name)


def test_two_different_carboxy_groups_are_not_licensed():
    from rdkit import Chem
    from orthonym.rules.isotope_letters import heavy_oxygen_on_carboxyl_carbonyl
    assert not heavy_oxygen_on_carboxyl_carbonyl(Chem.MolFromSmiles("CC(CC(=O)O)C(=[18O])O"))
    assert heavy_oxygen_on_carboxyl_carbonyl(Chem.MolFromSmiles("[18O]=C(O)CCC(=O)O"))
