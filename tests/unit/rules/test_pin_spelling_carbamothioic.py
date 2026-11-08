"""Spelling check (``validation.spelling.checks_carbamothioic``).

 (the Blue Book): 'carbamothioic O-acid (PIN)' (:30774), 'carbamothioic S-acid (PIN)' (:30826),
'carbonothioic S-acid (PIN) (not sulfanylformic acid)' (:30778); "Substitution of formic acid is not recommended".
OPSIN 2.9.0 reads no O-/S-locanted acid word: the PIN tier ships the locanted word verified through its twin
('carbamothioic S-acid'); every name without the right locant is below the PIN (all were pin_verified).
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.checks_carbamothioic import carbamothioic_acid_check
from tests.support.pin_tiers import tier_row


@pytest.mark.parametrize("smiles,name", [
    ("NC(=S)O", "carbamothioic acid"),
    ("NC(=O)S", "carbamothioic acid"),
    ("OC(=S)O", "carbonothioic acid"),
    ("CNC(=S)O", "(methylamino)methanethioic O-acid"),
    ("CCN(CC)C(=S)S", "(diethylamino)methane(dithioic acid)"),
    ("SC(=O)S", "carbonodithioic acid"),
    # mutation: the wrong locant
    ("NC(=S)O", "carbamothioic S-acid"),
    ("NC(=O)S", "carbamothioic O-acid"),
])
def test_not_the_pin(smiles, name):
    assert carbamothioic_acid_check(Chem.MolFromSmiles(smiles), name) is not None


@pytest.mark.parametrize("smiles,name", [
    ("NC(=S)S", "carbamodithioic acid"),
    ("NC(=O)O", "carbamic acid"),
    ("CCN(CC)C(=O)O", "diethylcarbamic acid"),
    ("CC(=S)O", "ethanethioic O-acid"),
    ("CN(C)C(=S)OC", "O-methyl dimethylcarbamothioate"),
    ("NC(=S)O", "carbamothioic O-acid"),
    ("NC(=O)S", "carbamothioic S-acid"),
    ("SC(=O)S", "carbonodithioic S,S-acid"),
    ("OC(=S)O", "carbonothioic O,O-acid"),
    ("CCN(CC)C(=S)S", "diethylcarbamodithioic acid"),
    ("CNC(=S)O", "methylcarbamothioic O-acid"),
])
def test_the_pin_passes(smiles, name):
    assert carbamothioic_acid_check(Chem.MolFromSmiles(smiles), name) is None


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin,readable", [
    ("NC(=S)O", "carbamothioic O-acid", "carbamothioic acid"),
    ("NC(=O)S", "carbamothioic S-acid", "carbamothioic acid"),
    ("NC(=O)[SeH]", "carbamoselenoic Se-acid", None),
    ("SC(=O)S", "carbonodithioic S,S-acid", "carbonodithioic acid"),
    ("OC(=S)O", "carbonothioic O,O-acid", "carbonothioic acid"),
])
def test_shipped_names(smiles, pin, readable):
    # the PIN tier ships the book's locanted word; best-effort ships the word OPSIN reads, below the PIN
    # (user decision 2026-10-22, "Best-effort ships readable")
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row
    if readable:
        row = tier_row(smiles, "best-effort")
        assert (row["name"], row["tier"]) == (readable, "systematic_verified"), row
