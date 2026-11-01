"""A phosphanyl or arsanyl group with substituents on it is a substituted group: 'bis', not 'di'.

 (c) (the Blue Book): a multiplied prefix that is itself substituted takes 'bis', 'tris'; the silyl group of
the same family already did ('1,4-bis[ethyldi(methyl)silyl]benzene'), because its writer records the derivation
(``assembly.prefix_marks.mononuclear_group_name``), and the phosphanyl and arsanyl writers returned a bare string that
the multiplier predicate reads as simple: '3,4-di[ethyl(methyl)phosphanyl]benzoic acid'. (:7272) gives the
group itself, '4-[ethyl(methyl)phosphanyl]-1H-imidazole (PIN)' (:39228), and the parent, 'ethyldi(methyl)phosphane
(PIN)' (:7290).
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,pin", [
    ("CCP(C)c1ccc(C(=O)O)cc1P(C)CC", "3,4-bis[ethyl(methyl)phosphanyl]benzoic acid"),
    ("OC(=O)c1cc(P(C)CC)cc(P(C)CC)c1", "3,5-bis[ethyl(methyl)phosphanyl]benzoic acid"),
    # unchanged: one group; the group without a second kind of substituent
    ("OC(=O)c1ccc(P(C)CC)cc1", "4-[ethyl(methyl)phosphanyl]benzoic acid"),
    ("OC(=O)c1cc(P(C)C)cc(P(C)C)c1", "3,5-bis(dimethylphosphanyl)benzoic acid"),
])
def test_the_multiplied_phosphanyl_group_takes_bis(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
