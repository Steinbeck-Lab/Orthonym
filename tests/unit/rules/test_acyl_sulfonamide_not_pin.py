"""An N-carboacyl sulfonamide named with the 'sulfonamide' suffix is not the PIN.

 'Secondary and tertiary amides' (the Blue Book): amides "(R-CO)2NH, (R-SO2)2NH, etc.... are named as
N-acyl derivatives of the senior primary amide" ('N-acetylbenzamide (PIN)',:33111); amides rank "in the order of the
corresponding acids",:18184), carboxylic before sulfonic (:18172), so the carboxamide is the parent.
Rule-derived: the book prints no acyl sulfonamide. Sulfonylureas and sulfonyl carbamates (carbonic acid, class 7b)
keep their names (rules.pin_vocabulary.acyl_sulfonamide_named_as_sulfonamide).
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles", [
    "CC(=O)NS(=O)(=O)c1ccccc1",
    "CC(=O)NS(=O)(=O)c1ccc(N)cc1",
    "CCC(=O)NS(=O)(=O)c1ccccc1",
    "CS(=O)(=O)NC(C)=O",
    "O=CNS(=O)(=O)c1ccccc1",                  # formic acid is a carboxylic acid too
    "CC(=O)NS(=O)(=O)c1ccc(cc1)Cc1ccc(cc1)S(=O)(=O)NC(C)=O",
])
def test_acyl_sulfonamide_is_not_labelled_pin(smiles):
    row = tier_row(smiles, "best-effort")
    assert row["name"].rstrip(")").endswith("sulfonamide") and row["tier"] != "pin_verified", row


@pytest.mark.parametrize("smiles,name", [
    ("CCCCNC(=O)NS(=O)(=O)c1ccc(C)cc1", "N-(butylcarbamoyl)-4-methylbenzene-1-sulfonamide"),
    ("COC(=O)NS(=O)(=O)c1ccc(N)cc1", "methyl (4-aminobenzene-1-sulfonyl)carbamate"),
    ("CNS(=O)(=O)c1ccccc1", "N-methylbenzenesulfonamide"),
    ("CC(=O)Nc1ccc(S(N)(=O)=O)cc1", "N-(4-sulfamoylphenyl)acetamide"),
    # the acyl sulfonamide as a prefix under a senior group keeps its preferred prefix,:33019)
    ("CS(=O)(=O)NC(=O)CC(=O)OC", "methyl 3-(methanesulfonamido)-3-oxopropanoate"),
])
def test_other_sulfonamides_keep_the_pin(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
