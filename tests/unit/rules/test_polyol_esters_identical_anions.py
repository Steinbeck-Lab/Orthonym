"""A polyol of three to six identical esters is a functional-class multiplicative name.

 "Polyesters formed from a single 'alcoholic' component" (``the Blue Book Blue Book``),
 (:31819): "When anions are identical functional class multiplicative nomenclature is used. Names are
formed by citing the multivalent group, the multiplicative prefix, and the multiplied anionic component name.
Multiplicative prefixes 'di', 'tri', etc. are used when anions are unsubstituted; when substituted, prefixes 'bis',
'tris', etc. are used." Printed: 'ethane-1,2-diyl diacetate (PIN)' (:31823), 'propane-1,3-diyl bis(chloroacetate)
(PIN)' (:31825), 'propane-1,2,3-triyl triacetate (PIN)' (:31827). The book prints no example above three; the rule has
no limit, and the names above three are the rule applied (rule-derived, not a printed PIN). (:31698) cites
an ester as a prefix only when the ester methods cannot describe the compound, so 'butane-1,2,3,4-tetrayl tetraacetate'
and not '1,2,3,4-tetrakis(acetyloxy)butane'. The gold row -01-PROTECT asserted the prefix form as the PIN of the
tetraacetate; it now asserts the functional-class name (the lipid path still does not claim it: producer
``inner:ester_family``).
"""
import pytest

from orthonym.cli import _tiered_namer
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)OCC(OC(C)=O)C(OC(C)=O)COC(C)=O", "butane-1,2,3,4-tetrayl tetraacetate"),
    ("CC(=O)OCC(OC(C)=O)C(OC(C)=O)C(OC(C)=O)COC(C)=O", "pentane-1,2,3,4,5-pentayl pentaacetate"),
    ("CC(=O)OCC(OC(C)=O)C(OC(C)=O)C(OC(C)=O)C(OC(C)=O)COC(C)=O", "hexane-1,2,3,4,5,6-hexayl hexaacetate"),
    ("CC(=O)OCCC(OC(C)=O)COC(C)=O", "butane-1,2,4-triyl triacetate"),
    ("C(OC(=O)c1ccccc1)C(OC(=O)c1ccccc1)C(OC(=O)c1ccccc1)COC(=O)c1ccccc1", "butane-1,2,3,4-tetrayl tetrabenzoate"),
    # a substituted anion takes 'bis', 'tris', 'tetrakis' (:31819)
    ("ClCC(=O)OCC(OC(=O)CCl)C(OC(=O)CCl)COC(=O)CCl", "butane-1,2,3,4-tetrayl tetrakis(chloroacetate)"),
])
def test_identical_anions_of_a_polyol_are_cited_once_with_the_multiplicative_prefix(smiles, expected):
    row = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified"
    assert opsin_roundtrip_check(smiles, row["name"])["passed"], row["name"]


@pytest.mark.unit
@pytest.mark.opsin_gate
def test_the_lipid_path_keeps_glycerol_and_does_not_claim_the_tetraol():
    glycerol = _tiered_namer("pin", "pin").name_tiered("CC(=O)OCC(OC(C)=O)COC(C)=O")
    assert glycerol["name"] == "propane-1,2,3-triyl triacetate" and glycerol["producer"] == "outer:lipid"
    tetraol = _tiered_namer("pin", "pin").name_tiered("CC(=O)OCC(OC(C)=O)C(OC(C)=O)COC(C)=O")
    assert tetraol["producer"] == "inner:ester_family"


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    "CC(=O)OCC(COC(C)=O)(COC(C)=O)COC(C)=O",             # a branched polyol: not an unbranched chain
    "CC(=O)OCC(OC(C)=O)C(OC(C)=O)COC(=O)CC",             # two different anions: method (1) is not built
])
def test_the_polyols_the_producer_does_not_build_stay_below_the_pin(smiles):
    assert _tiered_namer("pin", "pin").name_tiered(smiles)["tier"] == "abstain"
