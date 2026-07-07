"""Wave-2 completion Tier 2: data/catalog additive wins.

thiourea + pentazolidine retained parents; catenated Group-14/bridge hydrides
(P-21.2.3/P-52.1.3). All OPSIN-RT probed at build time.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.catenated_hydrides import name_catenated_hydride

pytestmark = pytest.mark.unit


def _name(smiles):
    return name_compound(Chem.CanonSmiles(smiles))


@pytest.mark.parametrize("smiles,expected", [
    ("NC(=S)N", "thiourea"),         # P-66.1.6.1.3 retained parent
    ("N1NNNN1", "pentazolidine"),    # P-52.1.5.1 all-N HW-preselected ring
])
def test_retained_parents(smiles, expected):
    assert _name(smiles) == expected


def test_urea_guanidine_unchanged():
    assert _name("NC(=O)N") == "urea"
    assert _name("N=C(N)N") == "guanidine"


CATENATED = [
    ("[SiH3]O[SiH3]", "disiloxane"),
    ("[SiH3]O[SiH2]O[SiH3]", "trisiloxane"),
    ("[SiH3]O[SiH2]O[SiH2]O[SiH3]", "tetrasiloxane"),
    ("[SiH3]N[SiH3]", "disilazane"),
    ("[SiH3]S[SiH3]", "disilathiane"),
    ("[SnH3]O[SnH3]", "distannoxane"),
    ("[GeH3]O[GeH3]", "digermoxane"),
]


@pytest.mark.parametrize("smiles,expected", CATENATED)
def test_catenated_hydride_unit(smiles, expected):
    assert name_catenated_hydride(Chem.MolFromSmiles(smiles)) == expected


@pytest.mark.parametrize("smiles,expected", CATENATED[:4])
def test_catenated_hydride_e2e(smiles, expected):
    assert _name(smiles) == expected


def test_catenated_hydride_fail_closed():
    # carbon present -> not this class (methoxysilane) -> declines
    assert name_catenated_hydride(Chem.MolFromSmiles("CO[SiH3]")) is None
    # bare silane (single hub) -> not catenated
    assert name_catenated_hydride(Chem.MolFromSmiles("[SiH4]")) is None
    # mixed Group-14 kinds -> declines
    assert name_catenated_hydride(Chem.MolFromSmiles("[SiH3]O[GeH3]")) is None
    # ring -> declines
    assert name_catenated_hydride(Chem.MolFromSmiles("[SiH2]1O[SiH2]O[SiH2]O1")) is None


def test_bare_silane_still_names_e2e():
    assert _name("[SiH4]") == "silane"
