"""
Unit tests for vinyl amine detection .

Ensures secondary and tertiary amine SMARTS patterns detect amines bonded
to sp2 (vinyl) carbons, not just sp3 and aromatic carbons. Also verifies
that amide and guanidine exclusions are preserved (no false positives).
"""

import pytest
from rdkit import Chem
from orthonym.perception.functional_groups import detect_functional_groups


class TestVinylAmineDetection:
    """: Secondary/tertiary amine SMARTS detect vinyl amines."""

    def test_vinyl_secondary_amine(self):
        """C=CNHC: secondary amine on sp2 carbon."""
        mol = Chem.MolFromSmiles("C=CNC")
        fgs = detect_functional_groups(mol)
        assert "secondary_amine" in fgs and len(fgs["secondary_amine"]) > 0

    def test_vinyl_tertiary_amine(self):
        """C=CN(C)C: tertiary amine on sp2 carbon."""
        mol = Chem.MolFromSmiles("C=CN(C)C")
        fgs = detect_functional_groups(mol)
        assert "tertiary_amine" in fgs and len(fgs["tertiary_amine"]) > 0

    def test_vinyl_primary_amine_already_works(self):
        """C=CNH2: primary amine uses [#6], already matches sp2 C."""
        mol = Chem.MolFromSmiles("C=CN")
        fgs = detect_functional_groups(mol)
        assert "primary_amine" in fgs and len(fgs["primary_amine"]) > 0

    def test_amide_not_false_positive(self):
        """CC(=O)NC: amide N should NOT match as secondary_amine."""
        mol = Chem.MolFromSmiles("CC(=O)NC")
        fgs = detect_functional_groups(mol)
        secondary_amine_matches = fgs.get("secondary_amine", [])
        # The amide exclusion !$([NX3][CX3]=O) must still work
        assert len(secondary_amine_matches) == 0

    def test_guanidine_not_false_positive(self):
        """NC(=N)NC: guanidine N should NOT match as secondary_amine."""
        mol = Chem.MolFromSmiles("NC(=N)NC")
        fgs = detect_functional_groups(mol)
        # Guanidine exclusion !$([NX3][CX3]=[NX2]) must still work
        secondary_amine_matches = fgs.get("secondary_amine", [])
        assert len(secondary_amine_matches) == 0
