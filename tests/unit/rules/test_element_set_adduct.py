"""A drawing of two or more different bare atoms (an alloy or an element set, ``[Fe].[Ni]``) is named by its element
names joined by em dashes, with the proportions: 'iron—nickel (1/1)' (``rules.adducts._element_set_name``).

The Blue Book does not name inorganic adducts: "In this subsection, only adducts formed from organic compounds are
considered. Inorganic adducts are discussed in ref. 12,.5." "Organic adducts", the Blue Book). The
components are cited by "the names of individual compounds" "Mixed organic – inorganic adducts",:4667), so
a neutral atom takes its element name ('chlorine'), not the radical name of its parent hydride ('chloranyl', which
OPSIN 2.9.0 does not read). A compound without carbon has a preselected name at most "PRESELECTED NAMES",
:2062): the name is made at the best-effort tier only and never carries the PIN label.
"""
import pytest

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

ROWS = [
    ("[Fe].[Ni]", "iron—nickel (1/1)"),                                 # metals only (the decline at adducts.py:976)
    ("[B].[Ni].[Ni]", "boron—nickel (1/2)"),                            # boron: element name, not 'boranylidyne'
    ("[Cs].[Ga].[Cl]", "caesium—chlorine—gallium (1/1/1)"),             # a non-metal atom (the decline at:1010)
    ("[La].[Eu]", "europium—lanthanum (1/1)"),                          # alphanumerical order of the names
    ("[57Fe].[Ni]", "(57Fe)iron—nickel (1/1)"),                         # nuclide symbol before the name
    ("[Si].[Ti].[Zr]", "silicon—titanium—zirconium (1/1/1)"),
]


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_best_effort_name(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected and res["tier"] == "systematic_verified" and not res["is_pin"], res
    assert name_is_rt_exact(res["name"], smiles), res


@pytest.mark.parametrize("smiles,_expected", ROWS)
def test_pin_tier_declines(smiles, _expected):
    assert _tiered_namer("pin", "pin").name_tiered(smiles)["tier"] == "abstain"


@pytest.mark.parametrize("smiles,expected", [
    ("[Fe][Ni]", None),                         # bonded atoms: not an element set
    ("[AlH3].[Fe]", None),                      # an atom with hydrogen is a parent hydride, not an element
    ("[Ag].[Ag].[Ag]", None),                   # one element only: no adduct
    ("[Zn+2].[Cl-].[Cl-]", "zinc dichloride"),  # ions: the salt producers' name, unchanged
])
def test_out_of_scope(smiles, expected):
    assert name_best_effort(smiles)["name"] == expected
