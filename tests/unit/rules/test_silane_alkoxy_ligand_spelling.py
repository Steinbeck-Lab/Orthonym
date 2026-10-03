"""Alkoxy groups on a silicon parent hydride take the prefix forms.

 (the Blue Book): the contracted prefixes methoxy, ethoxy, propoxy, butoxy, phenoxy
and tert-butoxy are retained; every other R-O- group keeps its organyl name -- '(propan-2-yl)oxy
(preferred prefix)' (:27683), '(benzyloxy)carbonyl' (:18116), '(cyclohexyloxy)benzene (PIN)'
(:27768). The silane ligand namer cut 'yl' and added 'oxy', shipping
'cyclohexyldi(methyl)(propan-2-oxy)silane' and '(benzoxy)(tert-butyl)di(methyl)silane' at
pin_verified, and the ligands were sorted with '(' and digits ahead of letters
,:3477). Every name below reads back to the input's full InChIKey with OPSIN 2.9.0.
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,pin", [
    ("CC(C)O[Si](C)(C)C1CCCCC1", "cyclohexyldi(methyl)[(propan-2-yl)oxy]silane"),
    ("CC(C)(C)[Si](C)(C)OCc1ccccc1", "(benzyloxy)(tert-butyl)di(methyl)silane"),
    ("CCO[Si](C)(C)C", "ethoxytri(methyl)silane"),
    ("CCCCCO[Si](C)(C)C", "trimethyl(pentyloxy)silane"),          # was '(pentoxy)'
    # (:3477): '2-methylpropoxy' (:27689) is alphabetized at 'm', after 'methyl'
    ("CC(C)CO[Si](C)(C)C", "trimethyl(2-methylpropoxy)silane"),
])
def test_alkoxy_ligand_prefix_forms(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
