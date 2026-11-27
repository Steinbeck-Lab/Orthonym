"""Thiourea ranks in class 11 (amides of carbonic acid), senior to hydrazides through imines.

IUPAC 2013 P-41 (Table 4.1) places amides at class 11, senior to hydrazides (12),
imides (13), nitriles (14), aldehydes (15), ketones (16), alcohols and thiols (17),
amines (19), and imines (20). Under P-66.1.6.1.3, thiourea is named as a chalcogen
analogue of urea with N,N' letter locants.

Retained aldehyde names (benzaldehyde, acetaldehyde, formaldehyde) are also
recognized by ``principal_class_check`` under P-41 so senior classes cannot pass
as prefixes on a retained aldehyde parent.
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.checks_parent import principal_class_check
from tests.support.pin_tiers import assert_not_pin_labelled, assert_pin_at_both_tiers

PIN_ROWS = [
    ("O=Cc1ccc(NC(=S)N)cc1", "N-(4-formylphenyl)thiourea"),
    ("CNC(=S)NCCS", "N-methyl-N'-(2-sulfanylethyl)thiourea"),
    ("OCCNC(=S)N", "N-(2-hydroxyethyl)thiourea"),
    ("NCCNC(=S)N", "N-(2-aminoethyl)thiourea"),
    ("N#CCCNC(=S)N", "N-(2-cyanoethyl)thiourea"),
    ("CC(=O)CCNC(=S)N", "N-(3-oxobutyl)thiourea"),
    ("O=CCCNC(=S)N", "N-(3-oxopropyl)thiourea"),
    ("Oc1ccc(NC(=S)N)cc1", "N-(4-hydroxyphenyl)thiourea"),
    ("Nc1ccc(NC(=S)N)cc1", "N-(4-aminophenyl)thiourea"),
    ("N#Cc1ccc(NC(=S)N)cc1", "N-(4-cyanophenyl)thiourea"),
    ("CC(=O)c1ccc(NC(=S)N)cc1", "N-(4-acetylphenyl)thiourea"),
]

NOT_PIN_ROWS = [
    ("O=Cc1ccc(NC(=S)N)cc1", "4-(carbamothioylamino)benzaldehyde"),
    ("CNC(=S)NCCS", "2-[(methylcarbamothioyl)amino]ethane-1-thiol"),
    ("OCCNC(=S)N", "2-(carbamothioylamino)ethan-1-ol"),
    ("N#CCCNC(=S)N", "3-(carbamothioylamino)propanenitrile"),
]

CONTROL_ROWS = [
    ("OC(=O)CCNC(=S)N", "3-(carbamothioylamino)propanoic acid"),
    ("NC(=O)CCNC(=S)N", "3-(carbamothioylamino)propanamide"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", PIN_ROWS + CONTROL_ROWS)
def test_thiourea_class11_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,non_pin", NOT_PIN_ROWS)
def test_thiourea_prefix_name_is_not_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


@pytest.mark.unit
@pytest.mark.parametrize("smiles,name", [
    ("O=Cc1ccc(NC(=S)N)cc1", "4-(carbamothioylamino)benzaldehyde"),
    ("O=Cc1ccc(C#N)cc1", "4-cyanobenzaldehyde"),
    ("O=Cc1ccc(C(=O)O)cc1", "4-carboxybenzaldehyde"),
    ("O=CC(=S)N", "2-amino-2-sulfanylideneacetaldehyde"),
])
def test_retained_aldehyde_parent_flagged_under_p41(smiles, name):
    failure = principal_class_check(Chem.MolFromSmiles(smiles), name)
    assert failure is not None and failure.rule == "P-41", name
