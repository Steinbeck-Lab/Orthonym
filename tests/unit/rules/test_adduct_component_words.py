"""A one-atom closed-shell anion is a component of a mixture like any other (``rules.adducts._one_atom_component_word``).

"Names are formed by citing the names of individual compounds" "Mixed organic – inorganic adducts",
the Blue Book). The lone-atom namer leaves 'chloride' to the anion classes (``metal_atoms.lone_atom_name_and_basis``),
so the component producer names such an anion through the single-component pipeline. The name ships only when OPSIN
reads it back to exactly the drawn structure, and "preferred IUPAC names cannot be assigned to mixed adducts".
"""
import pytest
from rdkit import Chem

from orthonym.cli import _tiered_namer
from orthonym.rules.adducts import _one_atom_component_word
from tests.support.opsin_marks import release_opsin_only
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

_NEW = [
    ("O.[F-]", "fluoride—water (1/1)"),
    ("O.[Cl-]", "chloride—water (1/1)"),
    ("[NH4+].[Cl-].[Cl-].[I-]", "chloride—iodide—azanium (2/1/1)"),
    ("[Mg+2].[SiH4].[Cl-].[Cl-]", "chloride—magnesium(2+)—silane (2/1/1)"),
]


@pytest.mark.parametrize("smiles,expected", _NEW)
def test_a_one_atom_anion_is_named_as_a_component(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected and res["tier"] == "systematic_verified" and not res["is_pin"], res
    assert name_is_rt_exact(res["name"], smiles), res


@pytest.mark.parametrize("smiles", [s for s, _ in _NEW])
def test_pin_tier_declines(smiles):
    assert _tiered_namer("pin", "pin").name_tiered(smiles)["tier"] == "abstain"


def test_the_word_of_a_halide_ion():
    assert _one_atom_component_word("[Br-]", Chem.MolFromSmiles("[Br-]"), {}) == "bromide"


@pytest.mark.parametrize("smiles,expected", [
    ("CCO.[Ar]", "ethanol—argon (1/1)"),
    ("CCO.[Cl-]", "ethanol—chloride (1/1)"),
    ("O.[Na+].[Cl-]", "sodium chloride monohydrate"),
    ("[K+].[BH4-]", "potassium boranuide"),
])
def test_names_made_before_are_unchanged(smiles, expected):
    assert name_best_effort(smiles)["name"] == expected


@pytest.mark.parametrize("smiles", [
    "[Th].N",             # thorium is not admitted to AMMINE: no ammonia beside it (data/metal_scope.py)
])
def test_refusals_on_purpose_stay(smiles):
    assert name_best_effort(smiles)["name"] is None


@release_opsin_only
def test_a_charge_reading_that_moves_stays_unnamed_on_the_release():
    # OPSIN 2.9.0 reads no 'oxidanediide' beside 'azanium' as drawn
    assert name_best_effort("[NH4+].[O-2]")["name"] is None
