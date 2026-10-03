"""A chain carrying a disulfanyl group is numbered from its free valence, and a phosphane with
such an organyl group is named on the phosphane.

 (the Blue Book): disulfides are named with the prefix '(R)disulfanyl',
'(methyldisulfanyl)methane (PIN)' (:27874), '1-(methyldiselanyl)-2-(methyldisulfanyl)ethane
(PIN)' (:27880). (:39151): "Alkyl, aryl, etc. groups and groups derived from parent
hydrides containing O, S, Se, and Te atoms are always denoted by prefixes" (:39153), so the
phosphane is the parent: 'dimethyl[2-(methyldisulfanyl)ethyl]phosphane', not the substitutive
'1-(dimethylphosphanyl)-2-(methyldisulfanyl)ethane' that shipped pin_verified. The chain namer
used to decline every chalcogen-chalcogen link, and the string converter turned the capped
'(methyldisulfanyl)ethane' into '(methyldisulfanyl)ethyl' without the 2-locant.
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,pin", [
    ("CP(C)CCSSC", "dimethyl[2-(methyldisulfanyl)ethyl]phosphane"),
    ("CSSCCO", "2-(methyldisulfanyl)ethan-1-ol"),
    ("CSSCCOC(C)=O", "2-(methyldisulfanyl)ethyl acetate"),
    ("CP(C)CCOC", "(2-methoxyethyl)di(methyl)phosphane"),
])
def test_disulfanyl_organyl(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
