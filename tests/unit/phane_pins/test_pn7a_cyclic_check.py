import pytest

from orthonym import opsin_capabilities as caps
from orthonym.validation.pin_spelling import check_pin_spelling
from orthonym.validation.spelling.cyclophane import is_phane_name
from tests.support.default_tier import assert_default_tier_declines
from tests.support.rt_assert import name_best_effort

from ._helpers import gold, mol

RULE = "P-52.2.5.1"
#: the Blue Book phane PIN, its structure (PN0), and the name both jars give without a phane producer
FALSE_PINS = [
    ("29005", "1,4,7,10(2,5)-tetrafuranacyclododecaphan-11-en-2-one",
     "C1(CC=2OC(=CC2)CCC=2OC(=CC2)CCC=2OC(=CC2)C=CC2=CC=C1O2)=O",
     "25,26,27,28-tetraoxapentacyclo[20.2.1.1^4,7.1^10,13.1^16,19]octacosa-1(24),4,6,10,12,16,18,20,22-nonaen-2-one"),
    ("19659", "4,12(1,3)-dibenzenaspiro[7.7]pentadecaphane",
     "C1CCC2=CC(=CC=C2)CCCC12CCCC1=CC(=CC=C1)CCC2",
     "5,5'-spirobi[bicyclo[7.3.1]trideca-1(12),9(13),10-triene]"),
]


def _rules(smiles, name):
    return [f.rule for f in check_pin_spelling(mol(smiles), name, strict=True) if f.rule == RULE]


@pytest.fixture
def release_capabilities(monkeypatch):
    monkeypatch.setattr(caps, "opsin_capabilities", lambda: {"phane": False, "halogen_uide": False})


@pytest.mark.parametrize("name,expected", [
    ("1,4(1,4)-dibenzenacyclohexaphane", True),
    ("1,4(1,4)-dibenzenacyclohexaphan-2-ene", True),
    ("4,12(1,3)-dibenzenaspiro[7.7]pentadecaphane", True),
    ("2,4,6-trioxa-1,7(1),3,5(1,4)-tetrabenzenaheptaphane", True),
    ("2,4,6-trioxa-1,7(1),3,5(1,4)-tetrabenzenaheptaphan-1^4-yl", True),
    ("triphenylphosphane", False),
    ("methylphosphane", False),
    ("tricyclo[9.3.1.1^4,8]hexadeca-1(15),11,13-triene", False),
    ("tryptophan", False),
    (None, False),
])
def test_is_phane_name(name, expected):
    assert is_phane_name(name) is expected


@pytest.mark.parametrize("line,book,smiles,engine_name", FALSE_PINS, ids=lambda v: str(v)[:8])
def test_a_non_phane_name_of_a_cyclophane_fails(line, book, smiles, engine_name):
    # (1) with: the von Baeyer or spiro name the engine builds is not the PIN
    assert _rules(smiles, engine_name) == [RULE]
    assert _rules(smiles, book) == []                  # the book's own PIN passes


@pytest.mark.parametrize("line,name,smiles", gold("cycle"), ids=lambda v: str(v)[:24])
def test_every_book_cyclophane_pin_passes(line, name, smiles):
    assert _rules(smiles, name) == []


@pytest.mark.parametrize("smiles,name", [
    ("c1cc2cc(c1)CCCCCCCCCCCCCCCC2", "bicyclo[16.3.1]docosa-1(22),18,20-triene"),                       #:23847
    ("c1cc2cc(c1)CCc1cccc(c1)CC2", "tricyclo[9.3.1.1^4,8]hexadeca-1(15),4(16),5,7,11,13-hexaene"),      #:15021
    ("c1cc2cc(c1)CCC1CCCC(CC2)C1", "tricyclo[9.3.1.1^4,8]hexadeca-1(15),11,13-triene"),                 #:23857
    ("c1cc2c3cc(ccc3c1)CSCc1cccc3ccc(cc13)CSC2",
     "5,7,14,16-tetrahydro-1,17:8,10-diethenodibenzo[c,j][1,8]dithiacyclotetradecine"),                 #:23895
])
def test_the_books_other_names_of_a_cyclophane_fail(smiles, name):
    # the von Baeyer and bridged fused names the book prints beside its cyclophane PINs (, rows)
    assert _rules(smiles, name) == [RULE]


@pytest.mark.parametrize("smiles,name", [
    ("C1CCCCCc2ccccc2CCCCC1", "5,6,7,8,9,10,11,12,13,14,15,16-dodecahydrobenzo[14]annulene"),     #:23839, a fusion PIN
    ("C1CC2CCCC(C1)CCC1CCCC(C1)CC2", "tricyclo[9.3.1.1^4,8]hexadecane"),                          #:23867, a von Baeyer PIN
    ("c1ccc(-c2ccccc2)cc1", "1,1'-biphenyl"),
])
def test_names_of_structures_whose_pin_is_no_phane_pass(smiles, name):
    assert _rules(smiles, name) == []


@pytest.mark.parametrize("smiles,name", [
    ("C1CCCCc2cccc1c2", "bicyclo[5.3.1]undeca-1(11),7,9-triene"),                  # [5]metacyclophane: a ring of 8 atoms
    ("C1CCCCCc2cccc1c2", "bicyclo[6.3.1]dodeca-1(12),8,10-triene"),                # [6]metacyclophane: a ring of 9 atoms
    ("C1CCCCCCCc2cc3CCCCc3c1c2", "an engine name"),                                 # a tetralin: not mancude as a system
])
def test_a_structure_the_test_does_not_judge_is_not_decided(smiles, name):
    # the book prints phane PINs for macrocycles of ten atoms and more (:15021): the check abstains on the others
    assert _rules(smiles, name) == []


@pytest.mark.opsin_gate
@pytest.mark.parametrize("line,book,smiles,engine_name", FALSE_PINS, ids=lambda v: str(v)[:8])
def test_the_default_tier_declines_the_false_pins_with_the_release_jar(release_capabilities, line, book, smiles,
                                                                         engine_name):
    row = assert_default_tier_declines(smiles)
    assert RULE in [f["rule"] for f in row["spelling_failures"]], row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("line,book,smiles,engine_name", FALSE_PINS, ids=lambda v: str(v)[:8])
def test_best_effort_keeps_the_name_with_an_honest_label(release_capabilities, line, book, smiles, engine_name):
    row = name_best_effort(smiles)
    assert (row["name"], row["tier"], row["is_pin"]) == (engine_name, "systematic_verified", False), row
    assert RULE in [f["rule"] for f in row["spelling_failures"]], row
