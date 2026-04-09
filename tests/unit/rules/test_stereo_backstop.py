"""Tests for the universal stereo backstop function in namer.py.

Phase 140 Plan 01: STER-16 -- ensure every naming path includes stereodescriptors.
The _final_stereo_check function is a safety net that catches gaps in
handler-specific stereo injection.
"""

import pytest
import re
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from orthonym.namer import _final_stereo_check


@pytest.mark.unit
class TestFinalStereoCheck:
    """Tests for the universal stereo backstop in namer.py."""

    def test_name_unchanged_when_stereo_already_present(self):
        """Name with existing stereo prefix should not be modified."""
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        result = _final_stereo_check(mol, "(2R)-butan-2-ol")
        assert result == "(2R)-butan-2-ol"

    def test_name_unchanged_when_no_stereocenters(self):
        """Molecule without stereo should return name unchanged."""
        mol = Chem.MolFromSmiles("CCC")
        rdCIPLabeler.AssignCIPLabels(mol)
        result = _final_stereo_check(mol, "propane")
        assert result == "propane"

    def test_stereo_added_when_missing_from_name(self):
        """Name missing stereo for a chiral molecule should get stereo prefix."""
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        result = _final_stereo_check(mol, "butan-2-ol")
        assert re.match(r'\(\d*[RS]\)-', result), f"Expected stereo prefix, got: {result}"
        assert "butan-2-ol" in result

    def test_name_unchanged_for_empty_string(self):
        """Empty string and 'unknown' should be returned unchanged."""
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        assert _final_stereo_check(mol, "") == ""
        assert _final_stereo_check(mol, "unknown") == "unknown"

    def test_ez_descriptor_added_when_missing(self):
        """E/Z molecule with name missing stereo should get E/Z prefix."""
        mol = Chem.MolFromSmiles("C/C=C/C")
        rdCIPLabeler.AssignCIPLabels(mol)
        result = _final_stereo_check(mol, "but-2-ene")
        assert re.match(r'\(\d*[EZ]\)-', result), f"Expected E/Z prefix, got: {result}"

    def test_complex_stereo_prefix_not_duplicated(self):
        """Name that already has a multi-descriptor prefix should not be modified."""
        mol = Chem.MolFromSmiles("C[C@H](O)[C@@H](O)C")
        rdCIPLabeler.AssignCIPLabels(mol)
        result = _final_stereo_check(mol, "(2R,3S)-pentane-2,3-diol")
        assert result == "(2R,3S)-pentane-2,3-diol"

    def test_none_mol_returns_name_unchanged(self):
        """If mol is None, name should be returned unchanged."""
        result = _final_stereo_check(None, "propane")
        assert result == "propane"
