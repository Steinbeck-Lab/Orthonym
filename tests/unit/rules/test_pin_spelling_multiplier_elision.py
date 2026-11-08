"""Spelling check (c) and the arene writer's suffix multiplier (the Blue Book, 'benzenehexol (not
benzenehexaol)':7619): 'naphthalene-1,4,5,8-tetraol' was pin_verified."""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.checks_multiplier_elision import multiplier_elision_check
from tests.support.pin_tiers import tier_row


@pytest.mark.parametrize("name", ["naphthalene-1,4,5,8-tetraol", "benzene-1,2,4,5-tetraamine", "cyclohexane-1,2,4,5-tetraone",
                                  "benzenehexaol", "dodecane-1,12-dodecaol"])
def test_not_the_pin(name):
    assert multiplier_elision_check(Chem.MolFromSmiles("C"), name) is not None


@pytest.mark.parametrize("name", ["naphthalene-1,4,5,8-tetrol", "benzene-1,2,4,5-tetramine", "hexaamminecobalt(3+)",
                                  "2,4,6,8-tetraoxa-nonane", "naphthalene-1,4,5,8-tetracarboxylic acid", "tetraethyl",
                                  # a functional class word keeps its 'a': 'diphosphoric tetraamide (PIN)' (the Blue Book
                                  "N1-methyl-1,2,3-triimidodiphosphoric tetraamide", "tetraallyl"])
def test_pin_spellings_pass(name):
    assert multiplier_elision_check(Chem.MolFromSmiles("C"), name) is None


@pytest.mark.opsin_gate
def test_the_arene_writer_elides():
    row = tier_row("Oc1ccc(O)c2c(O)ccc(O)c12", "pin")
    assert (row["name"], row["tier"]) == ("naphthalene-1,4,5,8-tetrol", "pin_verified"), row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    # the partly hydrogenated arene writer elides too (was the false PIN '...-tetraol')
    ("OC1C(O)c2ccccc2C(O)C1O", "1,2,3,4-tetrahydronaphthalene-1,2,3,4-tetrol"),
])
def test_the_hydro_arene_writer_elides(smiles, expected):
    assert tier_row(smiles, "best-effort")["name"] == expected
