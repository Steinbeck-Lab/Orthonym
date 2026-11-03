"""PIN-label audit of the Blue Book rows whose name differed from the book while labelled the PIN (2026-11-03).

Each case cites the rule that decides it; the book rows that disagree with their own rule are named beside it."""
import pytest

from orthonym.cli import _tiered_namer

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _pin(smiles):
    return _tiered_namer("pin", "pin").name_tiered(smiles)


@pytest.mark.parametrize("smiles,name", [
    # (the Blue Book,:7140) 'bis' to avoid ambiguity; '5,7-bis(sulfanylidene)-...-dione (PIN)' (:32555)
    ("O=c1oc(=O)c2cc3c(=S)sc(=S)c3cc12",
     "5,7-bis(sulfanylidene)-5,7-dihydro-1H,3H-thieno[3,4-f][2]benzofuran-1,3-dione"),
])
def test_a_multiplied_sulfanylidene_takes_bis(smiles, name):
    row = _pin(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


@pytest.mark.parametrize("smiles,name", [
    # Systematic names (:29860) "locants are used when hydrocarbon chains are modified by skeletal
    # replacement, as shown in "; '3,6,10,13-tetrathiapentadecane-1,15-dioic acid (PIN)' (:29874)
    ("O=C(O)CSCCSCCCSCCSCC(=O)O", "3,6,10,13-tetrathiapentadecane-1,15-dioic acid"),
    ("ClC(=O)COCCOCCOCCOCC(Cl)=O", "3,6,9,12-tetraoxatetradecane-1,14-dioyl dichloride"),
])
def test_a_heterochain_diacid_cites_its_suffix_locants(smiles, name):
    row = _pin(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


@pytest.mark.parametrize("smiles,name", [
    # (the Blue Book) the acyl group, then the class words; 'butanedioyl isocyanide isothiocyanate
    # (PIN)' (:31470), in alphabetical order as 'butanedioyl bromide chloride (PIN)' (:31454)
    ("[C-]#[N+]C(=O)CCC(=O)N=C=S", "butanedioyl isocyanide isothiocyanate"),
    ("N#CC(=O)CCC(=O)N=C=O", "butanedioyl cyanide isocyanate"),
])
def test_two_different_acyl_pseudohalides_are_two_class_words(smiles, name):
    row = _pin(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


def test_an_azonous_acid_gets_no_hydroxylamine_pin():
    # (the Blue Book): with an -OH left the replaced oxoacid is an acid; 'CH3-N(OH)(SH)
    # methylazonothious acid (PIN)' (:35604). The class is not built: no hydroxylamine PIN, a name at best-effort.
    row = _pin("CN(O)S")
    assert row["tier"] == "abstain", row
    wide = _tiered_namer("pin", "best-effort").name_tiered("CN(O)S")
    assert wide["name"] and wide["tier"] != "pin_verified", wide
    assert _pin("CNO")["name"] == "N-hydroxymethanamine"


@pytest.mark.parametrize("smiles,name", [
    # (the Blue Book); '2-[(4-carboxycyclohexyl)methyl]cyclohexane-1-carboxylic acid (PIN)
    # [not 4-[(2-carboxycyclohexyl)methyl]...]' (:23328-23330)
    ("O=C(O)C1CCC(CC2CCCCC2C(=O)O)CC1", "2-[(4-carboxycyclohexyl)methyl]cyclohexane-1-carboxylic acid"),
    ("O=C(O)C1CCCC(CC2CCCCC2C(=O)O)C1", "2-[(3-carboxycyclohexyl)methyl]cyclohexane-1-carboxylic acid"),
    ("CC1CCC(CC2CCCC(C)C2)CC1", "1-methyl-3-[(4-methylcyclohexyl)methyl]cyclohexane"),
])
def test_of_two_identical_cycloalkanes_the_parent_has_the_lower_prefix_locants(smiles, name):
    row = _pin(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


def test_a_drawn_thio_dicarbonic_acid_without_letter_locants_is_no_pin():
    # (the Blue Book-31085): '1,3-dithiodicarbonic acid (PIN; the location of the sulfur atoms is
    # unknown)' against '1,3-dithiodicarbonic O1,O3-acid (PIN)' for the drawn HO-CS-O-CS-OH
    assert _pin("OC(=S)OC(O)=S")["tier"] == "abstain"
    wide = _tiered_namer("pin", "best-effort").name_tiered("OC(=S)OC(O)=S")
    assert (wide["name"], wide["tier"]) == ("1,3-dithiodicarbonic acid", "systematic_verified"), wide
    assert _pin("SC(=S)OC(S)=S")["name"] == "1,1,3,3-tetrathiodicarbonic acid"
