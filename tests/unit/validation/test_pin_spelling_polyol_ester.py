"""DT3 step B: esters of a single polyol are named by functional class multiplicative nomenclature.

 (the Blue Book): "When anions are identical functional class multiplicative nomenclature is
used" ('propane-1,3-diyl bis(chloroacetate) (PIN)',:31825). "When anions are different, two
methods are used:" (:31831): "Method (1) generates preferred IUPAC names but names formed by using method (2) are
acceptable in general nomenclature" ('(1) propane-1,2,3-triyl 2-acetate 1-hexadecanoate 3-[(9Z)-octadec-9-enoate]
(PIN) (2) 2-(acetyloxy)-3-(hexadecanoyloxy)propyl (9Z)-octadec-9-enoate',:31846; '(1) 1,4-phenylene acetate
dichloroacetate (PIN) (2) 4-(acetyloxy)phenyl dichloroacetate'). A polyester of several alcoholic components, or
of a polyol that is no multivalent group of one parent, is named by (:31860 ff.) with acyloxy
prefixes: '2-(acetyloxy)ethyl methyl butanedioate (PIN)', '2-[2-(acetyloxy)ethyl]phenyl acetate (PIN)'.
The Blue Book's own PINs must not be flagged (the whole book is read in ``tests_local``).
"""
import pytest
from rdkit import Chem

from orthonym.validation.pin_spelling import ABSTAIN, PASS, check_pin_spelling
from orthonym.validation.spelling.checks_polyol_ester import (
    polyol_esters_by_acyloxy_prefixes, single_polyol_ester)

pytestmark = pytest.mark.unit

GLYCERIDE = "CCCCCCCCCCCCCCCC(=O)OC[C@H](OC(C)=O)COC(=O)CCCCCCC/C=C\\CCCCCCCC"


def _verdict(smiles, name):
    return polyol_esters_by_acyloxy_prefixes(Chem.MolFromSmiles(smiles), name)


@pytest.mark.parametrize("smiles, name", [
    (GLYCERIDE, "2-(acetyloxy)-3-(hexadecanoyloxy)propyl (9Z)-octadec-9-enoate"),          #:31846 (2)
    ("CC(=O)Oc1ccc(OC(=O)C(Cl)Cl)cc1", "4-(acetyloxy)phenyl dichloroacetate"),               #:31837 (2)
    ("CC(=O)OCOC=O", "(formyloxy)methyl acetate"),                                           #:31837 (2)
    ("CC(=O)OCC(OC(C)=O)COC(=O)CC", "2,3-bis(acetyloxy)propyl propanoate"),                  #:31837 (2)
    ("ClCC(=O)OCCCOC(=O)CCl", "3-[(chloroacetyl)oxy]propyl chloroacetate"),                  # identical anions::31825
])
def test_a_polyol_ester_named_by_acyloxy_prefixes_is_flagged(smiles, name):
    failures = check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)
    assert [f.rule for f in failures if f.rule.startswith("P-65.6.3.3.3")] == ["P-65.6.3.3.3"], name


@pytest.mark.parametrize("smiles, name", [
    (GLYCERIDE, "propane-1,2,3-triyl 2-acetate 1-hexadecanoate 3-[(9Z)-octadec-9-enoate]"),    #:31846 (1) PIN
    ("ClCC(=O)OCCCOC(=O)CCl", "propane-1,3-diyl bis(chloroacetate)"),                           #:31825 PIN
    ("CC(=O)OCC(OC(C)=O)COC(C)=O", "propane-1,2,3-triyl triacetate"),                           #:31825 PIN
    ("CC(=O)Oc1ccc(OC(=O)C(Cl)Cl)cc1", "1,4-phenylene acetate dichloroacetate"),                #:31837 (1) PIN
    ("CC(=O)OCOC=O", "methylene acetate formate"),                                              #:31837 (1) PIN
])
def test_the_functional_class_multiplicative_name_passes(smiles, name):
    assert _verdict(smiles, name) == PASS


@pytest.mark.parametrize("smiles, name", [
    ("COC(=O)CCC(=O)OCCOC(C)=O", "2-(acetyloxy)ethyl methyl butanedioate"),                     #:31860 PIN
    ("CC(=O)OCCOC(=O)CC(=O)OCCOC(C)=O", "bis[2-(acetyloxy)ethyl] propanedioate"),               #:31864 PIN
    ("CCC(=O)OCCOC(=O)CCC(=O)OCCOC(C)=O", "2-(acetyloxy)ethyl 2-(propanoyloxy)ethyl butanedioate"),   #:31866
    ("CCOC(=O)CC(=O)Oc1ccc(OC(=O)CCC(=O)OC)cc1", "4-[(3-ethoxy-3-oxopropanoyl)oxy]phenyl methyl butanedioate"),
    ("CC(=O)Oc1ccc(-c2ccc(OC(C)=O)cn2)cc1", "6-[4-(acetyloxy)phenyl]pyridin-3-yl acetate"),      #:31895 PIN
    ("CC(=O)OCCc1ccccc1OC(C)=O", "2-[2-(acetyloxy)ethyl]phenyl acetate"),                       #:31897 PIN
    ("CCOC(=O)C(C)OC(C)=O", "ethyl 2-(acetyloxy)propanoate"),                                   # second ester on the acid
])
def test_the_polyesters_of_p_65_6_3_3_4_are_not_flagged(smiles, name):
    """The book's PINs for a polyester of several alcoholic components, or of a polyol of more than one parent,
    cite acyloxy prefixes: the check does not decide them."""
    assert _verdict(smiles, name) in (PASS, ABSTAIN)
    assert not [f for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)
                if f.rule.startswith("P-65.6.3.3.3")]


def test_where_the_esters_sit():
    assert single_polyol_ester(Chem.MolFromSmiles(GLYCERIDE)) == ("single", 3)
    assert single_polyol_ester(Chem.MolFromSmiles("CCOC(=O)C(C)OC(C)=O"))[0] == "none"          # one per alcohol
    assert single_polyol_ester(Chem.MolFromSmiles("COC(=O)CCC(=O)OCCOC(C)=O"))[0] == "other"
    assert single_polyol_ester(Chem.MolFromSmiles("CCO"))[0] == "none"
    assert single_polyol_ester(Chem.MolFromSmiles("O=C1CCCCO1"))[0] == "ring"                  # a lactone: not read
    assert single_polyol_ester(Chem.MolFromSmiles("O=C1OCCOC(=O)CCCC1"))[0] == "ring"


def test_a_molecule_with_a_group_senior_to_the_esters_is_not_decided():
    """ (:18162): an acid is senior to the esters; the esters are then prefixes,:31696)."""
    smiles = "CC(=O)OCC(OC(C)=O)C(=O)O"
    assert single_polyol_ester(Chem.MolFromSmiles(smiles)) == ("single", 2)
    assert _verdict(smiles, "2,3-bis(acetyloxy)propyl formate") == ABSTAIN          # a name an acid would not have


def test_a_name_of_the_p_10_classes_and_a_name_without_an_ester_word_abstain():
    assert _verdict(GLYCERIDE, "1,2-di-O-acetyl-3-O-hexadecanoyl-sn-glycerol glycerate") == ABSTAIN
    assert _verdict("CC(=O)OCC(OC(C)=O)COC(C)=O", "glycerol triacetate trimer") == ABSTAIN
    assert _verdict("CC(=O)OCC(OC(C)=O)COC(C)=O", "2,3-bis(acetyloxy)propyl") == ABSTAIN
