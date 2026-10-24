"""A ring system with a fusion name is never a von Baeyer spiro component inside a PIN (Task 3b).

 "Five-membered ring requirement" (the Blue Book,:23710): "Fusion nomenclature gives
preferred IUPAC names only to compounds having at least two rings of at least five or more members."
 (:10260) names the components of a spiro compound by their own names, so a spiro component
that has a fusion name is cited by it, and a von Baeyer spelling of that component cannot be a PIN.

At main the von Baeyer analysis of the component records its descriptor as a non-PIN fragment
(``polycyclic._record_if_fusion_nameable``) and the default tier declines the name it built with
NO_VERIFIED_PIN, the reason code that says a wider tier returns a name (a refusal inside the spiro
producer would report UNNAMEABLE for molecules the best-effort tier names). These tests pin that
behaviour for the fused components of the class: the strict path still builds the name, the default
tier declines it, the best-effort tier gives the same name read back to the full InChIKey.
"""
import pytest

from tests.support.default_tier import assert_default_tier_declines, strict_path_row
from tests.support.large_polycycles import be, ladder, pin
from tests.support.rt_assert import assert_full_rt

pytestmark = pytest.mark.opsin_gate
L = ladder()

#: spiro compounds whose von Baeyer component has a fusion name (W01: ring sizes 4, 4, 6, 6, a
#: one-carbon peri bridge on naphthalene; the other two are saturated ortho-fused tricycles)
FUSED_COMPONENT_ROWS = [
    ("W01_spiro_cyclobutanaphthalene", L["W01_spiro_cyclobutanaphthalene_F7"],
     "spiro[cyclobutane-1,2'-tricyclo[5.3.1.0^3,11]undeca-1(10),3(11),4,6,8-pentaene]"),
    ("F10_tricyclo_8_4_0", "C1CCC2(C1)CCC1CC3CCCCC3CC1C2",
     "spiro[cyclopentane-1,5'-tricyclo[8.4.0.0^3,8]tetradecane]"),
    ("F10_tricyclo_6_4_0", "C1CCC2(C1)CCC1C3CCCC3CC1C2",
     "spiro[cyclopentane-1,10'-tricyclo[6.4.0.0^2,6]dodecane]"),
]


@pytest.mark.unit
@pytest.mark.parametrize("key,smiles,name", FUSED_COMPONENT_ROWS, ids=[r[0] for r in FUSED_COMPONENT_ROWS])
def test_fused_spiro_component_is_never_a_von_baeyer_pin(key, smiles, name):
    assert_default_tier_declines(smiles)
    r = pin(smiles)
    assert not (r["is_pin"] and "cyclo[" in (r["name"] or "")), r
    row = strict_path_row(smiles)
    assert row["name"] == name and not row["is_pin"], row
    assert_full_rt(name, smiles)


@pytest.mark.unit
@pytest.mark.parametrize("key,smiles,name", FUSED_COMPONENT_ROWS, ids=[r[0] for r in FUSED_COMPONENT_ROWS])
def test_best_effort_still_names_it_verified_and_non_pin(key, smiles, name):
    r = be(smiles)
    assert r["name"] == name, r
    assert r["opsin"] in ("verified", "verified_reconstructor") and not r["is_pin"], r
    assert r["tier"] != "pin_verified", r


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("C1CCC2(CC1)c1ccccc1-c1ccccc12", "spiro[cyclohexane-1,9'-fluorene]"),
    ("C1CCC2(C1)C1=CC=CC=C1c1ccccc21", "spiro[cyclopentane-1,9'-fluorene]"),
    ("C1CCC2(C1)CCC1CCCCC1C2", "3',4',4'a,5',6',7',8',8'a-octahydro-1'H-spiro[cyclopentane-1,2'-naphthalene]"),
    ("C1CCC2(C1)CC1CCC2C1", "spiro[bicyclo[2.2.1]heptane-2,1'-cyclopentane]"),   # no ortho-fused pair: von Baeyer
])
def test_spiro_controls_keep_their_pin(smiles, expected):
    r = pin(smiles)
    assert (r["name"], r["tier"]) == (expected, "pin_verified"), r


@pytest.mark.unit
def test_spiro_adamantane_is_a_name_of_a_retained_parent():
    # adamantane is retained, the Blue Book, "RETAINED NAMES FOR VON BAEYER PARENT
    # HYDRIDES"): the von Baeyer spelling of the component is declined at the default tier (Task 12)
    smiles = L["W03_spiro_adamantane"]
    assert_default_tier_declines(smiles)
    assert strict_path_row(smiles)["name"] == "spiro[cyclobutane-1,2'-tricyclo[3.3.1.1^3,7]decane]"


@pytest.mark.unit
def test_chromone_prefix_never_ships_as_a_von_baeyer_pin():
    # PROTECT only: a von Baeyer prefix of a mancude fused ring is never labelled a PIN
    r = pin(L["W02_chromone_ester_F7"])
    assert not (r["is_pin"] and "oxabicyclo[" in (r["name"] or "")), r
