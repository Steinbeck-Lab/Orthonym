"""
Unit tests for generic alcohol detection (PERC-05).

Ensures the generic alcohol catch-all SMARTS [OX2H][CX4] detects hydroxyl
groups on sp3 carbons regardless of non-carbon neighbors (halogens, nitrogen,
sulfur, etc.).  Also verifies that specific sub-type patterns (primary_alcohol,
secondary_alcohol, tertiary_alcohol) still work, and that phenol is NOT caught
by the generic alcohol pattern.
"""

import pytest
from rdkit import Chem
from orthonym.perception.functional_groups import detect_functional_groups


class TestGenericAlcoholDetection:
    """PERC-05: Generic alcohol catch-all [OX2H][CX4] detects OH on non-standard carbons."""

    def test_halogenated_alcohol(self):
        """OC(F)Cl: OH on carbon with halogen neighbors."""
        mol = Chem.MolFromSmiles("OC(F)Cl")
        fgs = detect_functional_groups(mol)
        assert "alcohol" in fgs and len(fgs["alcohol"]) > 0

    def test_geminal_diol(self):
        """CC(O)(O)CC: Both OH groups detected."""
        mol = Chem.MolFromSmiles("CC(O)(O)CC")
        fgs = detect_functional_groups(mol)
        assert "alcohol" in fgs and len(fgs["alcohol"]) >= 2

    def test_amino_alcohol(self):
        """CC(N)(O)C(=O)O: OH on carbon with nitrogen neighbor."""
        mol = Chem.MolFromSmiles("CC(N)(O)C(=O)O")
        fgs = detect_functional_groups(mol)
        assert "alcohol" in fgs and len(fgs["alcohol"]) > 0

    def test_simple_methanol_also_matches(self):
        """CO (methanol): Generic pattern also catches simple alcohols."""
        mol = Chem.MolFromSmiles("CO")
        fgs = detect_functional_groups(mol)
        assert "alcohol" in fgs and len(fgs["alcohol"]) > 0

    def test_phenol_not_caught_by_generic(self):
        """c1ccccc1O: Phenol is [OX2H][cX3], NOT [OX2H][CX4] -- generic should not match."""
        mol = Chem.MolFromSmiles("c1ccccc1O")
        fgs = detect_functional_groups(mol)
        # Generic alcohol pattern requires CX4 (sp3 carbon), phenol carbon is cX3
        assert "phenol" in fgs and len(fgs["phenol"]) > 0
        # Generic "alcohol" key should NOT match phenol's aromatic carbon
        alcohol_matches = fgs.get("alcohol", [])
        assert len(alcohol_matches) == 0, (
            f"Generic alcohol should not match phenol's aromatic carbon, got {alcohol_matches}"
        )

    def test_specific_subtypes_still_work(self):
        """CCO (ethanol): primary_alcohol specific pattern still matches."""
        mol = Chem.MolFromSmiles("CCO")
        fgs = detect_functional_groups(mol)
        assert "primary_alcohol" in fgs and len(fgs["primary_alcohol"]) > 0
        assert "alcohol" in fgs and len(fgs["alcohol"]) > 0  # Generic also matches
