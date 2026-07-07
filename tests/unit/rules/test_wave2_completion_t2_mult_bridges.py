"""Wave-2 completion Tier 2: multiplicative single-atom bridge extensions.

Carbonyl (P-15.3.1.2.1.1) and substituted-methylene (P-15.3.1.2.1.2) bridges.
All OPSIN-RT probed at build time.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound

pytestmark = pytest.mark.unit


def _name(smiles):
    return name_compound(Chem.CanonSmiles(smiles))


@pytest.mark.parametrize("smiles,expected", [
    # carbonyl bridge (=O), benzoic-acid units
    ("OC(=O)c1ccc(C(=O)c2ccc(C(=O)O)cc2)cc1", "4,4'-carbonyldibenzoic acid"),
    # substituted-methylene bridge (CHCl), phenol units, enclosed in parens
    ("Oc1ccc(C(Cl)c2ccc(O)cc2)cc1", "4,4'-(chloromethylene)diphenol"),
])
def test_new_bridges(smiles, expected):
    assert _name(smiles) == expected


def test_plain_bridges_unchanged():
    # Protect: the pre-existing methylene / oxy / sulfanediyl bridges are
    # byte-identical (the relaxed ring-neighbour guard must not disturb them).
    assert _name("OC(=O)c1ccc(Cc2ccc(C(=O)O)cc2)cc1") == "4,4'-methylenedibenzoic acid"
    assert _name("OC(=O)c1ccc(Oc2ccc(C(=O)O)cc2)cc1") == "4,4'-oxydibenzoic acid"
