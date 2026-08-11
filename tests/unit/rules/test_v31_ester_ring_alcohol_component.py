"""v31 breadth (coverage-by-construction, class 1): an ester whose ALCOHOL
component is a ring must name the cyclic component correctly, not linearise it.

Root cause: `esters.get_alkyl_fragment_name` derives the alcohol-component word
from a CARBON COUNT when the centralized `name_substituent_fragment` primitive
declines it (its DROP-24 guard declines ring-bearing fragments). The count path
then linearises the ring and drops the ring's own substituents --
`CC(=O)O[C@H]1CCCCC[C@@H]1O` came out `(1S,2S)-heptyl acetate` (cycloheptane ->
7 chain carbons, -OH dropped): a WRONG molecule that SELF-01 suppressed into a
silent abstention. Fix: route a declined ring alcohol component through the
recursive substituent namer (`name_substituent`, the C4 keystone), which names a
decorated ring at PIN. Fail-closed: only runs after the old primitive declined,
so it converts a currently-abstaining ester or stays declined -- never regresses.
"""
import pytest

from orthonym import name_compound


@pytest.mark.parametrize("smiles,expected", [
    # ring alcohol component with a substituent (was linearised + OH dropped)
    ("CC(=O)O[C@H]1CCCCC[C@@H]1O", "(1S,2S)-2-hydroxycycloheptyl acetate"),
    ("CC(=O)OC1CCCC1O", "2-hydroxycyclopentyl acetate"),
    ("CC(=O)OC1CCC(O)CC1", "4-hydroxycyclohexyl acetate"),
    # plain cyclic alcohol component (was linearised: cyclohexyl -> 'hexyl')
    ("CC(=O)OC1CCCCC1", "cyclohexyl acetate"),
])
def test_ester_ring_alcohol_component_named(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # non-ring ester forms must be unchanged (the fix only fires for declined rings)
    ("CC(=O)OC", "methyl acetate"),
    ("CC(=O)OC(C)C", "propan-2-yl acetate"),
    ("CC(=O)OCC(C)C", "2-methylpropyl acetate"),
    ("CCOC(=O)c1ccccc1", "ethyl benzoate"),
    ("CC(=O)OCc1ccccc1", "benzyl acetate"),
    ("CC(=O)Oc1ccccc1", "phenyl acetate"),
])
def test_plain_esters_unchanged(smiles, expected):
    assert name_compound(smiles) == expected
