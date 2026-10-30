"""Printed PINs of the acyl halides and pseudohalides that the PIN tier declined.

 "Acyl pseudohalides from suffix acids" (heading:31458, text:31460): the acyl group of the acid, then the
class word "preceded by a multiplicative prefix, as needed"; 'SCN-CO-CO-NCS oxalyl diisothiocyanate (PIN)' (:31468).
Table 6.3 (:31416) lists the isothiocyanate class beside the isocyanate, and its order of seniority is
N3 > CN > NC > NCO > NCS (:31460). "Acyl halides from suffix acids" (:31412): 'Br-O2S-CH2-CH2-SO2-Br
ethane-1,2-disulfonyl dibromide (PIN)' (:31452).
"""
import pytest

from orthonym.cli import _tiered_namer
from orthonym.opsin_capabilities import opsin_capabilities
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
from tests.support.rt_assert import assert_full_rt


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("O=C(N=C=S)C(=O)N=C=S", "oxalyl diisothiocyanate"),                 #:31468 verbatim
    ("BrS(=O)(=O)CCS(=O)(=O)Br", "ethane-1,2-disulfonyl dibromide"),     #:31452 verbatim
    ("O=C(N=C=S)CCC(=O)N=C=S", "butanedioyl diisothiocyanate"),
    ("O=C(N=C=S)CC", "propanoyl isothiocyanate"),
    ("O=C(N=C=S)c1ccccc1", "benzoyl isothiocyanate"),
    ("O=C(N=C=S)CCl", "chloroacetyl isothiocyanate"),
    ("ClS(=O)(=O)CCS(=O)(=O)Cl", "ethane-1,2-disulfonyl dichloride"),
    ("O=S(=O)(Cl)c1ccc(cc1)S(=O)(=O)Cl", "benzene-1,4-disulfonyl dichloride"),
    ("ClS(=O)CCS(=O)Cl", "ethane-1,2-disulfinyl dichloride"),
    ("ClS(=O)(=O)CCCS(=O)(=O)Cl", "propane-1,3-disulfonyl dichloride"),
])
def test_the_printed_pins_are_built(smiles, expected):
    row = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified"
    assert opsin_roundtrip_check(smiles, row["name"])["passed"], row["name"]


@pytest.mark.unit
@pytest.mark.opsin_gate
def test_two_different_halides_on_one_acid_are_not_cited_as_one():
    """'bromide chloride' of a sulfonic acid is not built; the PIN tier declines, the wider tiers name it."""
    smiles = "ClS(=O)(=O)CCS(=O)(=O)Br"
    assert _tiered_namer("pin", "pin").name_tiered(smiles)["tier"] == "abstain"
    wide = _tiered_namer("pin", "best-effort").name_tiered(smiles)
    assert wide["name"] and wide["tier"] == "systematic_verified"


# --- labels that were checked against the book and stay -----------------------------------------------------------


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected,why", [
    ("N#CB(O)O", "cyanoboronic acid",
     "P-68.1.4.1 (:37313): 'boronic acid' is the preselected name 'when the hydrogen atom(s) attached to the boron "
     "atom is (are) substituted'; P-68.1.5 'Substitutive nomenclature' (:37377, text :37381): 'acids having retained "
     "names have seniority over other suffixes', so the nitrile is a prefix; 'cyanatosilicic acid (PIN)' (:35698)"),
    ("N#CN=[N+]=[N-]", "carbononitridic azide",
     "P-65.5.3.3 (:31501): 'NC-N3 carbononitridic azide (PIN)', method (1) generates the PINs"),
])
def test_a_name_checked_against_the_book_keeps_its_label(smiles, expected, why):
    row = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", why


@pytest.mark.unit
@pytest.mark.opsin_gate
def test_dimethyliodanuide_is_the_printed_spelling_read_through_its_twin():
    """ "Anions formed by addition of hydride ions", method (1) leads to PINs: 'diphenyliodanuide (PIN)'
    (:41110). OPSIN 2.9.0 does not read 'uide'; the name is read through the twin 'dimethyl-lambda3-iodanide'
    (``validation.uide_twin``), which is its label's ``opsin_twin``. An OPSIN that reads the name itself (OPSIN's
    main branch) reads it to the input, and the label is ``opsin``."""
    row = _tiered_namer("pin", "pin").name_tiered("C[I-]C")
    assert row["name"] == "dimethyliodanuide" and row["tier"] == "pin_verified"
    if opsin_capabilities()["halogen_uide"]:
        assert row["verified"] == "opsin"
        assert_full_rt(row["name"], "C[I-]C")
    else:
        assert row["verified"] == "opsin_twin"


# --- the retained acyl groups of and the carbonic esters --------------------------------------------------


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("NC(=O)C(=O)N=C=S", "oxamoyl isothiocyanate"),
    ("NC(=O)C(=O)N=C=O", "oxamoyl isocyanate"),
    ("NC(=O)C(=O)N=[N+]=[N-]", "oxamoyl azide"),
    ("NC(=O)C(=O)C#N", "oxamoyl cyanide"),
    ("NC(=O)C(=O)Br", "oxamoyl bromide"),                 #:31456 verbatim
    ("NC(=O)N=C=S", "carbamoyl isothiocyanate"),          #:31495 'carbamoyl isocyanate (PIN)'
    ("NC(=O)N=C=O", "carbamoyl isocyanate"),
])
def test_the_retained_acyl_group_names_the_pseudohalide_too(smiles, expected):
    """ "Acyl pseudohalides from suffix acids" (:31460-:31462): "The names formyl, acetyl, benzoyl, oxalyl,
    and oxamoyl are retained to generate preferred IUPAC names", so 'oxamoyl isothiocyanate' and not
    'amino(oxo)acetyl isothiocyanate'."""
    row = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified"
    assert opsin_roundtrip_check(smiles, row["name"])["passed"], row["name"]


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("CCOC(=O)N=C=S", "ethyl isothiocyanatomethanoate"),
    ("CCOC(=O)N=C=O", "ethyl isocyanatomethanoate"),
    ("OC(=O)N=C=S", "isothiocyanatomethanoic acid"),
])
def test_the_carbon_of_an_ester_or_an_acid_is_no_acyl_carbon_of_a_pseudohalide(smiles, expected):
    """The acyl carbon of an acyl isocyanate or isothiocyanate bears no -O- or -S-, as that of an acyl cyanide: the
    ester is named as an ester, and the pattern that took its carbon dropped the isothiocyanate and left a name
    of the 3-thia-1-azapropa-1,2-dienyl kind."""
    row = _tiered_namer("pin", "best-effort").name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "systematic_verified"
    assert opsin_roundtrip_check(smiles, row["name"])["passed"], row["name"]
