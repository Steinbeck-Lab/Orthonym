"""A carboxamide outranks a sulfonamide as the suffix of a benzene name.

 "SENIORITY ORDER FOR CLASSES", Table 4.1 (the Blue Book): class 11 "Amides [in the order of the
corresponding acids...]" (:18184); (:33559): "within the amide class, amides rank in the same order as the
corresponding acid"; carboxylic acids rank above sulfonic acids (class 7a,:18172). The benzene producer ranked the
sulfonamide above the carboxamide and shipped 'N-methyl-4-(methylcarbamoyl)benzene-1-sulfonamide' at pin_verified.
Hydrazides (class 12,:18185) stay below every amide.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("CNC(=O)c1ccc(S(=O)(=O)NC)cc1", "N-methyl-4-(methylsulfamoyl)benzamide"),
    ("NC(=O)c1ccc(S(N)(=O)=O)cc1", "4-sulfamoylbenzamide"),
    ("CNC(=O)c1ccc(S(N)(=O)=O)cc1", "N-methyl-4-sulfamoylbenzamide"),
    ("NC(=O)c1ccc(S(=O)(=O)NC)cc1", "4-(methylsulfamoyl)benzamide"),
    ("NNC(=O)c1ccc(S(N)(=O)=O)cc1", "4-(hydrazinecarbonyl)benzene-1-sulfonamide"),   # a hydrazide below the amide
    ("NC(=O)c1ccc(C(N)=S)cc1", "4-carbamothioylbenzamide"),
])
def test_benzene_amide_seniority(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)
