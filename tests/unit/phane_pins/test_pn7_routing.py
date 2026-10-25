import pytest

from orthonym.rules import phane_pin

from ._helpers import default_tier_row, gold, mol, name_best_effort, needs_phane_opsin, substituted

pytestmark = [pytest.mark.opsin_gate, needs_phane_opsin]

CYCLIC_NOT_LEGACY = "c1cc2cc(c1)CCC1CCCC(CC2)C1"            #:23857, cyclohexana + benzena


@pytest.mark.parametrize("line,name,smiles", gold("cycle") + gold("path"), ids=lambda v: str(v)[:30])
def test_gold_structures_get_their_book_pin(line, name, smiles):
    #,: the default tier ships the book's PIN, labelled pin_verified
    row = default_tier_row(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified" and row["spelling_failures"] == [], row


@pytest.mark.parametrize("line,name,smiles",
                         [r for r in substituted("suffix") + substituted("skeleton") if r[0] in ("29005", "19659")],
                         ids=lambda v: str(v)[:30])
def test_the_false_pins_get_the_phane_pin(line, name, smiles):
    #:29005 (PN4a) and:19659 (PN5): with both jars a von Baeyer or spiro name was labelled pin_verified (PN7a)
    row = default_tier_row(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified" and row["spelling_failures"] == [], row


@pytest.mark.parametrize("line,name,smiles", substituted("skeleton"), ids=lambda v: str(v)[:30])
def test_the_polycyclic_skeleton_rows_get_their_pin(line, name, smiles):
    #,: a bicyclo[x.y.0] skeleton word is no ortho-fused von Baeyer name of the vocabulary of
    # non-PIN forms (:15065 is a book PIN), so the label stays pin_verified
    row = default_tier_row(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified" and row["spelling_failures"] == [], row


def test_the_spelling_checks_pass_a_phane_pin():
    from orthonym.validation.pin_spelling import check_pin_spelling
    book = [r for r in substituted("suffix") + substituted("skeleton") if r[0] != "-"]
    for line, name, smiles in gold("cycle") + gold("path") + book:
        assert check_pin_spelling(mol(smiles), name, strict=True) == [], name


def test_a_producer_that_cannot_name_keeps_todays_name(monkeypatch):
    monkeypatch.setattr(phane_pin, "general_phane_name", lambda m: None)
    row = name_best_effort(CYCLIC_NOT_LEGACY)
    assert row["name"] == "tricyclo[9.3.1.1^4,8]hexadeca-1(15),11,13-triene" and row["tier"] == "systematic_verified"
    assert default_tier_row(CYCLIC_NOT_LEGACY)["tier"] == "abstain"


def test_a_phane_name_that_round_trips_to_another_molecule_is_voided(monkeypatch):
    wrong = "1,4(1,4)-dibenzenacyclohexaphane"           # a name for the para isomer
    monkeypatch.setattr(phane_pin, "general_phane_name", lambda m: wrong)
    row = name_best_effort(CYCLIC_NOT_LEGACY)
    assert row["name"] == "tricyclo[9.3.1.1^4,8]hexadeca-1(15),11,13-triene", row
    assert default_tier_row(CYCLIC_NOT_LEGACY)["tier"] == "abstain"


def test_a_senior_atom_parent_stays_the_parent():
    #: the book's silane (:19325, 'trimethyl[1^2H-1(6)-pyrana-3,5(1,4),7(1)-tribenzenaheptaphan-7^4-yl]silane
    # (PIN)') is not renamed by a phane: the silane is the parent and cites the phane as its prefix
    silane = "C[Si](C)(C)c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1"
    name = "trimethyl[2,4,6-trioxa-1,7(1),3,5(1,4)-tetrabenzenaheptaphan-1^4-yl]silane"
    for row in (name_best_effort(silane), default_tier_row(silane)):
        assert row["name"] == name and row["tier"] == "pin_verified" and row["spelling_failures"] == [], row


@pytest.mark.parametrize("line,name,smiles", substituted("stereo"), ids=lambda v: str(v)[:30])
def test_the_stereo_rows_get_their_pin_at_the_default_tier(line, name, smiles):
    #: the descriptors are at the front of the name; the read-back compares the isomeric SMILES as well
    row = default_tier_row(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified" and row["spelling_failures"] == [], row
