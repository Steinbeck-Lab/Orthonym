"""Unit tests for the shared λ-convention module (v23 Phase 6).

P-31.1.4.2 / Table 2.8 standard bonding numbers + the fail-closed
"is this valence non-standard?" decision, promoted from spiro.py so spiro,
acyclic skeletal-replacement and the mononuclear-hydride namers share one
implementation.
"""
import pytest
from rdkit import Chem

from orthonym.rules.lambda_convention import (
    STANDARD_BONDING_NUMBER,
    format_lambda_token,
    nonstandard_bonding_number,
)


def _idx_of(mol, symbol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == symbol:
            return atom.GetIdx()
    raise AssertionError(f"no {symbol} in mol")


class TestStandardBondingNumberTable:
    def test_chalcogens_divalent(self):
        for el in ("O", "S", "Se", "Te"):
            assert STANDARD_BONDING_NUMBER[el] == 2

    def test_pnictogens_trivalent(self):
        for el in ("N", "P", "As", "Sb", "Bi"):
            assert STANDARD_BONDING_NUMBER[el] == 3

    def test_group14_tetravalent(self):
        for el in ("Si", "Ge", "Sn", "Pb"):
            assert STANDARD_BONDING_NUMBER[el] == 4

    def test_halogens_monovalent(self):
        for el in ("F", "Cl", "Br", "I"):
            assert STANDARD_BONDING_NUMBER[el] == 1


class TestNonstandardBondingNumber:
    def test_tetravalent_sulfur_flagged(self):
        # SF4 -> tetravalent S (standard 2) -> lambda4
        mol = Chem.MolFromSmiles("FS(F)(F)F")
        assert nonstandard_bonding_number(mol, _idx_of(mol, "S")) == 4

    def test_hexavalent_sulfur_flagged(self):
        mol = Chem.MolFromSmiles("FS(F)(F)(F)(F)F")  # SF6
        assert nonstandard_bonding_number(mol, _idx_of(mol, "S")) == 6

    def test_pentavalent_phosphorus_flagged(self):
        mol = Chem.MolFromSmiles("FP(F)(F)(F)F")  # PF5
        assert nonstandard_bonding_number(mol, _idx_of(mol, "P")) == 5

    def test_pentavalent_iodine_flagged(self):
        mol = Chem.MolFromSmiles("FI(F)(F)(F)F")  # IF5
        assert nonstandard_bonding_number(mol, _idx_of(mol, "I")) == 5

    def test_standard_divalent_oxygen_none(self):
        mol = Chem.MolFromSmiles("COC")  # ether O, valence 2
        assert nonstandard_bonding_number(mol, _idx_of(mol, "O")) is None

    def test_standard_trivalent_phosphorus_none(self):
        mol = Chem.MolFromSmiles("ClP(Cl)Cl")  # PCl3, valence 3 = standard
        assert nonstandard_bonding_number(mol, _idx_of(mol, "P")) is None

    def test_charged_atom_never_flagged(self):
        # Fail-closed: a charged atom returns None even if valence looks odd.
        mol = Chem.MolFromSmiles("C[S+](C)C")  # sulfonium
        assert nonstandard_bonding_number(mol, _idx_of(mol, "S")) is None

    def test_element_absent_from_table_none(self):
        mol = Chem.MolFromSmiles("[Fe]")
        assert nonstandard_bonding_number(mol, _idx_of(mol, "Fe")) is None


class TestFormatLambdaToken:
    def test_lambda_present(self):
        assert format_lambda_token(4, 4) == "4λ4"
        assert format_lambda_token(6, 6) == "6λ6"

    def test_lambda_absent(self):
        assert format_lambda_token(2, None) == "2"
        assert format_lambda_token(11, None) == "11"
