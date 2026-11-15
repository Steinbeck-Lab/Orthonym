"""Broken drawings of chlorine and bromine named through the iodine drawing, beyond the first build
(``assembly/halogen_drawing.py``; lane X of the all-organic program).

* The read-back compares the drawing atom by atom on the halogens (``validation/depiction_match.py``): the standard
  InChIKey does not see a hydrogen on an over-valent halogen (``C1CC[Cl]CC1`` and ``C1CC[ClH]CC1`` share one).
* An input with iodine of its own: each choice of iodine stems to respell is read back; the first that OPSIN reads to
  the input's drawing ships.
* A stem that begins an enclosed prefix ships when the spelling parser shows the prefixes keep their alphanumerical
  order "Simple prefixes... are arranged alphabetically", the Blue Book), so (g) numbers them
  as in the iodine name.
* A neutral chlorine with two bonds is the λ3 radical "Nonstandard bonding numbers"); a chloranuide with
  three bonds has an odd number of electrons and no name.
Every name is systematic_verified, never a PIN (another drawing never certifies one).
"""
import pytest

from orthonym import opsin_capabilities as caps
from orthonym.assembly.halogen_drawing import _reads_back, halogen_drawing, prefix_order_kept
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import name_best_effort

pytestmark = pytest.mark.opsin_gate


def _reads_uide() -> bool:
    try:
        return bool(caps.opsin_capabilities().get("halogen_uide"))
    except Exception:  # noqa: BLE001
        return False


#: OPSIN 2.9.0 reads no '-anuide' of a halogen ('diphenyliodanuide'): the name cannot read back there
needs_uide = pytest.mark.skipif(not _reads_uide(), reason="the OPSIN in use reads no halogen -uide name")


@pytest.mark.parametrize("smiles, name", [
    ("C=Clc1ccc(I)cc1", "1-iodo-4-(methylidene-λ3-chloranyl)benzene"),
    pytest.param("C1=CC(=CC=C1[Cl-]S(=O)(=O)O)I", "(4-iodophenyl)(sulfo)chloranuide", marks=needs_uide),
    pytest.param("CC1=CC(=CC(=C1[Br-]C)C)I", "(4-iodo-2,6-dimethylphenyl)(methyl)bromanuide", marks=needs_uide),
])
def test_an_input_with_iodine_of_its_own(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified" and not r["is_pin"], r


@pytest.mark.parametrize("smiles, name", [
    ("CCC1=C(C(=CC=C1)CC)N2CC(CC2(C)C)(C3CCCCC3)Cl(C)C",
     "4-cyclohexyl-1-(2,6-diethylphenyl)-4-(dimethyl-λ3-chloranyl)-2,2-dimethylpyrrolidine"),
    ("CC1=NC=C(C=N1)Br2OC(C(O2)(C)C)(C)C",
     "2-methyl-5-(3,3,4,4-tetramethyl-1λ3-broma-2,5-dioxacyclopentan-1-yl)pyrimidine"),
])
def test_an_enclosed_prefix_that_keeps_its_order(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified" and not r["is_pin"], r


@pytest.mark.parametrize("iodine_name, name, kept", [
    ("1-fluoro-4-(methylidene-λ3-iodanyl)benzene", "1-fluoro-4-(methylidene-λ3-chloranyl)benzene", True),
    ("1-ethyl-4-(methylidene-λ3-iodanyl)benzene", "1-ethyl-4-(methylidene-λ3-chloranyl)benzene", True),
    # 'difluoro-λ3-chloranyl' sorts before 'difluoroethyl', 'difluoro-λ3-iodanyl' after it
    ("1-(2,2-difluoroethyl)-3-(difluoro-λ3-iodanyl)benzene", "1-(2,2-difluoroethyl)-3-(difluoro-λ3-chloranyl)benzene",
     False),
])
def test_prefix_order_kept(iodine_name, name, kept):
    assert prefix_order_kept(iodine_name, name) is kept


def test_a_neutral_two_bond_chlorine_is_drawn_as_a_radical():
    drawing = halogen_drawing("C1CC[Cl]CC1")
    assert drawing.radical and drawing.iodine_smiles == "C1CC[I]CC1"
    assert halogen_drawing("C[Cl-]1CCCCC1") is None        # odd-electron and charged


def test_a_name_with_one_hydrogen_more_on_the_halogen_is_refused():
    drawing = halogen_drawing("C1CC[Cl]CC1")
    with jvm_slots(1, purpose="test-halogen-drawing"):
        assert not _reads_back("1λ3-chlorinane", drawing)       # [ClH]1CCCCC1: the same key, not the drawing
