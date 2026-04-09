"""Tests for the universal stereo backstop function in namer.py.

Phase 140 Plan 01: STER-16 -- ensure every naming path includes stereodescriptors.
The _final_stereo_check function is a safety net that DETECTS gaps in
handler-specific stereo injection and logs them for targeted fixes.

Design choice: the backstop does NOT inject stereo with raw atom-index locants
because they don't correspond to IUPAC numbering.  It logs a WARNING to
identify handler gaps.  Handler-specific _inject_stereo_if_missing() remains
the primary stereo injection mechanism.
"""

import logging
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

    def test_gap_detected_when_stereo_missing_from_name(self, caplog):
        """Backstop detects stereo gap and logs warning but does not modify name.

        The backstop returns name unchanged because injecting stereo with raw
        atom-index locants would produce incorrect IUPAC names.
        """
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        with caplog.at_level(logging.WARNING):
            result = _final_stereo_check(mol, "butan-2-ol", handler="test_handler")
        # Name unchanged -- backstop is detection-only
        assert result == "butan-2-ol"
        # But WARNING was logged identifying the gap with handler attribution
        warning_msgs = [rec.message for rec in caplog.records if "Stereo backstop" in rec.message]
        assert len(warning_msgs) >= 1, "Expected WARNING about stereo gap"
        assert "test_handler" in warning_msgs[0], "WARNING should include handler name"

    def test_name_unchanged_for_empty_string(self):
        """Empty string and 'unknown' should be returned unchanged."""
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        assert _final_stereo_check(mol, "") == ""
        assert _final_stereo_check(mol, "unknown") == "unknown"

    def test_ez_gap_detected_when_missing(self, caplog):
        """Backstop detects E/Z gap and logs warning but does not modify name."""
        mol = Chem.MolFromSmiles("C/C=C/C")
        rdCIPLabeler.AssignCIPLabels(mol)
        with caplog.at_level(logging.WARNING):
            result = _final_stereo_check(mol, "but-2-ene")
        # Name unchanged
        assert result == "but-2-ene"
        # WARNING logged
        assert any("Stereo backstop" in rec.message for rec in caplog.records)

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

    def test_no_warning_when_stereo_already_present(self, caplog):
        """No warning should be logged when stereo is already in name."""
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        with caplog.at_level(logging.WARNING):
            _final_stereo_check(mol, "(2R)-butan-2-ol")
        stereo_warnings = [r for r in caplog.records if "Stereo backstop" in r.message]
        assert len(stereo_warnings) == 0, "Should not warn when stereo already present"

    def test_no_warning_when_no_stereo_in_molecule(self, caplog):
        """No warning should be logged for achiral molecules."""
        mol = Chem.MolFromSmiles("CCC")
        rdCIPLabeler.AssignCIPLabels(mol)
        with caplog.at_level(logging.WARNING):
            _final_stereo_check(mol, "propane")
        stereo_warnings = [r for r in caplog.records if "Stereo backstop" in r.message]
        assert len(stereo_warnings) == 0, "Should not warn for achiral molecule"

    def test_stereo_embedded_in_name_body_not_flagged(self):
        """Names with stereo descriptors inside (not just prefix) are not flagged."""
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        # Stereo embedded in a parenthetical group within the name
        result = _final_stereo_check(mol, "something-(2R)-else")
        # The embedded (2R) should be detected and name returned unchanged
        assert result == "something-(2R)-else"

    def test_carbohydrate_stereo_not_flagged(self):
        """Names with alpha/beta-D/L carbohydrate notation are not flagged."""
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        result = _final_stereo_check(mol, "alpha-D-glucopyranose")
        assert result == "alpha-D-glucopyranose"
        result2 = _final_stereo_check(mol, "beta-L-mannose")
        assert result2 == "beta-L-mannose"

    def test_warning_includes_handler_name(self, caplog):
        """WARNING log should include handler attribution."""
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        with caplog.at_level(logging.WARNING):
            result = _final_stereo_check(mol, "butan-2-ol", handler="polyfunctional_chain")
        assert result == "butan-2-ol"
        warning_msgs = [r.message for r in caplog.records if "Stereo backstop" in r.message]
        assert len(warning_msgs) == 1
        assert "polyfunctional_chain" in warning_msgs[0]

    def test_handler_defaults_to_unknown(self, caplog):
        """Calling without handler parameter should still work."""
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        with caplog.at_level(logging.WARNING):
            result = _final_stereo_check(mol, "butan-2-ol")
        assert result == "butan-2-ol"
        warning_msgs = [r.message for r in caplog.records if "Stereo backstop" in r.message]
        assert len(warning_msgs) == 1
        assert "unknown" in warning_msgs[0]
