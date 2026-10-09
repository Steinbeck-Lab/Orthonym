"""A benzene ring that carries a phenyl group and a ring-bearing oxy group is the ring assembly
'[1,1'-biphenyl]-n-yl', never '4-...-2-phenylphenyl' (lane W, item 21).

 "Substituent prefixes derived from ring assemblies" (the Blue Book): a substituent
group derived from a ring assembly is '[1,1'-biphenyl]-4-yl (preferred prefix)'; the appendix table
"DETACHABLE PREFIXES USED FOR SUBSTITUTIVE NOMENCLATURE" lists '[1,1'-biphenyl]-4-yl (not
4-phenylphenyl)' (:55597). "Ring
assemblies with a single bond junction" (:15560) names two identical rings joined by a single
bond as an assembly. ``_decorated_biphenylyl_substituent_name`` asked for exactly two rings in the
fragment, so a decoration with its own ring (an oxanyloxy, a benzyloxy) sent the fragment to the
single-ring producer, which cited the second ring as 'phenyl' and the name was labelled pin_verified."""
import re

import pytest

from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

ROWS = [
    # the item: main '3-({4-[(oxan-2-yl)oxy]-2-phenylphenyl}methyl)pyridine-4-carboxylic acid'
    ("OC(=O)c1ccncc1Cc1ccc(OC2CCCCO2)cc1-c1ccccc1",
     "3-({5-[(oxan-2-yl)oxy][1,1'-biphenyl]-2-yl}methyl)pyridine-4-carboxylic acid"),
    # a benzyloxy decoration: main '3-{[4-(benzyloxy)-2-phenylphenyl]methyl}pyridine-4-carboxylic acid'
    ("OC(=O)c1ccncc1Cc1ccc(OCc2ccccc2)cc1-c1ccccc1",
     "3-{[5-(benzyloxy)[1,1'-biphenyl]-2-yl]methyl}pyridine-4-carboxylic acid"),
    # the control that was already right
    ("OC(=O)c1ccncc1Cc1ccc(OC)cc1-c1ccccc1",
     "3-[(5-methoxy[1,1'-biphenyl]-2-yl)methyl]pyridine-4-carboxylic acid"),
]


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_biphenylyl_prefix_at_both_tiers(smiles, expected):
    row = default_tier_row(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert name_best_effort(smiles)["name"] == expected
    assert_full_rt(expected, smiles)


# a third ring joined to the assembly is not a biphenyl: the producer declines and the name is
# never certified with the phenylphenyl spelling
DECLINES = [
    "OC(=O)c1ccncc1Cc1ccc(-c2ccccc2)cc1-c1ccccc1",       # terphenyl
    "OC(=O)c1ccncc1Cc1ccc(OC2CCCCC2)cc1-c1ccccc1",
    "OC(=O)c1ccncc1Cc1ccc(N2CCOCC2)cc1-c1ccccc1",
]


@pytest.mark.parametrize("smiles", DECLINES)
def test_other_shapes_never_ship_the_phenylphenyl_spelling_as_a_pin(smiles):
    row = default_tier_row(smiles)
    assert not (row["tier"] == "pin_verified" and re.search(r"phenyl\w*phenyl", row["name"] or "")), row
    be = name_best_effort(smiles)
    if be["tier"] == "pin_verified":
        assert not re.search(r"phenylphenyl", be["name"]), be
