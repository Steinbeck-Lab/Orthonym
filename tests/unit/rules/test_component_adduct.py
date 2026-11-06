"""The last name of a mixture at the best-effort tier: every component named alone, cited in the order with the
proportions, shipped only when OPSIN reads it back to exactly the drawn structure (``rules.adducts._component_adduct_name``).

"Names are formed by citing the names of individual compounds" "Mixed organic – inorganic adducts",
the Blue Book), and "preferred IUPAC names cannot be assigned to mixed adducts": the name is never a PIN.
"""
import pytest

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("C.C.F.F.F", "methane—hydrogen fluoride (2/3)"),
    ("[SiH4].[AlH3]", "alumane—silane (1/1)"),       # hydrides only: no element set, named by its components
    ("N.N.O", "ammonia—water (2/1)"),
    ("[Zn-].[Zn-].[Na+]", "sodium(1+)—zinc(1-) (1/2)"),
])
def test_named_by_its_components(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected and res["tier"] == "systematic_verified" and not res["is_pin"], res
    assert name_is_rt_exact(res["name"], smiles), res


@pytest.mark.parametrize("smiles", [
    "[Th+4].[H-].[H-].[H-].[H-]",   # thorium is not admitted to CARBANION_SALTS: no hydride beside it
    "[Th].N",                       # nor to AMMINE: no ammonia beside it
    "[B+3].[H-].[H-].[H-]",         # a one-atom p-block ion is never cited by its free-ion name ('boranetriylium')
    "CCO.[O]",                      # OPSIN reads no adduct with a radical component
    # OPSIN reads 'trihydroxy(oxo)-λ5-chlorane' beside an amine as the salt: the drawn structure is not reproduced
    "C1CCN(CC1)C2=CC3CCC2C3.OCl(=O)(O)O",
])
def test_no_name_unless_the_drawing_reads_back(smiles):
    assert name_best_effort(smiles)["name"] is None


@pytest.mark.parametrize("smiles,expected", [
    ("CCO.O", "ethanol—water (1/1)"),
    ("O.[Zn]", "zinc—water (1/1)"),
    ("CC(=O)[O-].[Na+]", "sodium acetate"),
])
def test_names_made_before_are_unchanged(smiles, expected):
    assert name_best_effort(smiles)["name"] == expected


def test_pin_tier_declines():
    assert _tiered_namer("pin", "pin").name_tiered("C.C.F.F.F")["tier"] == "abstain"
