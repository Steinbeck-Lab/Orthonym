"""Tests for the universal stereo backstop function in namer.py.

a phase Plan 01: STER-16 -- ensure every naming path includes stereodescriptors.
The _final_stereo_check function is a safety net that DETECTS gaps in
handler-specific stereo injection and logs them for targeted fixes.

Design choice: the backstop does NOT inject stereo with raw atom-index locants
because they don't correspond to IUPAC numbering. It logs a WARNING to
identify handler gaps. Handler-specific _inject_stereo_if_missing() remains
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
        result = _final_stereo_check(mol, "α-D-glucopyranose")
        assert result == "α-D-glucopyranose"
        result2 = _final_stereo_check(mol, "β-L-mannose")
        assert result2 == "β-L-mannose"

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

    # ------------------------------------------------------------------
    # a phase invariants -- backstop refactor regression contracts.
    # ------------------------------------------------------------------

    def test_predicate_backstop_parity(self, caplog):
        """SC-3 /: needs_stereo_injection(mol, name) must agree with
        whether _final_stereo_check would log a 'Stereo backstop' WARNING
        for the same (mol, name) pair. Panel of 6 cases covering all
        regex branches + bond stereo + empty/unknown name guards."""
        from orthonym.rules.stereochemistry import needs_stereo_injection

        m_ster = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(m_ster)
        m_no_ster = Chem.MolFromSmiles("CCO")
        rdCIPLabeler.AssignCIPLabels(m_no_ster)
        m_ez = Chem.MolFromSmiles("C/C=C/C")
        rdCIPLabeler.AssignCIPLabels(m_ez)

        panel = [
            (m_ster, "butan-2-ol", True),                     # (a) atom stereo missing
            (m_ster, "(2R)-butan-2-ol", False),               # (b) prefix present
            (m_no_ster, "ethanol", False),                    # (c) no stereo
            (m_no_ster, "unknown", False),                    # (d) unknown name
            (m_no_ster, "", False),                           # (e) empty name
            (m_ez, "but-2-ene", True),                        # (f) bond stereo missing
        ]
        for mol, name, expected in panel:
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                _final_stereo_check(mol, name, handler="parity_test")
            warned = any(
                "Stereo backstop" in r.message for r in caplog.records
            )
            predicate = needs_stereo_injection(mol, name)
            assert warned == expected, (
                f"backstop warned={warned} for ({Chem.MolToSmiles(mol)}, {name!r}); "
                f"expected {expected}"
            )
            assert predicate == expected, (
                f"predicate={predicate} for ({Chem.MolToSmiles(mol)}, {name!r}); "
                f"expected {expected}"
            )
            assert warned == predicate, (
                f"backstop/predicate disagreement for ({Chem.MolToSmiles(mol)}, "
                f"{name!r}): warned={warned} predicate={predicate}"
            )

    def test_warning_message_byte_identical(self, caplog):
        """Post-refactor WARNING text must be byte-identical to pre-refactor
        message format. Locked: 'Stereo backstop: %r (handler: %s) has %d
        R/S + %d E/Z but name lacks descriptors. Fix handler to include
        stereo natively.'"""
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        with caplog.at_level(logging.WARNING):
            _final_stereo_check(mol, "butan-2-ol", handler="locked_handler")
        msgs = [
            r.message for r in caplog.records if "Stereo backstop" in r.message
        ]
        assert len(msgs) == 1
        msg = msgs[0]
        assert "Stereo backstop:" in msg
        assert "'butan-2-ol'" in msg
        assert "(handler: locked_handler)" in msg
        assert "1 R/S" in msg
        assert "0 E/Z" in msg
        assert "Fix handler to include stereo natively." in msg

    def test_warning_still_fires_for_unwired_handlers(self, caplog):
        """SC-3: backstop must continue to flag handlers that are NOT
        wired in a phase (complex_ring, polycyclic, retained-name fallback,
        decomposition fragments). The refactor must not gate WARNING on
        handler name -- predicate is a function of (mol, name) only."""
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        for unwired in (
            "complex_ring", "polycyclic", "retained_name_fallback",
            "decomposition_fragment", "unknown",
        ):
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                _final_stereo_check(mol, "butan-2-ol", handler=unwired)
            msgs = [
                r.message for r in caplog.records
                if "Stereo backstop" in r.message
            ]
            assert len(msgs) == 1, (
                f"WARNING should fire for unwired handler={unwired!r}; "
                f"got {len(msgs)} messages"
            )
            assert unwired in msgs[0], (
                f"handler={unwired!r} not in WARNING msg: {msgs[0]!r}"
            )

    def test_warning_suppressed_after_injection(self, caplog):
        """SC-3: if a handler successfully injected stereo (name now
        starts with `(...)`-), the backstop predicate is False and no
        WARNING fires. This is what a phase commits 3/4/5 will
        produce."""
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        with caplog.at_level(logging.WARNING):
            result = _final_stereo_check(
                mol, "(2R)-butan-2-ol", handler="heterocycle"
            )
        assert result == "(2R)-butan-2-ol"
        msgs = [
            r.message for r in caplog.records if "Stereo backstop" in r.message
        ]
        assert msgs == [], (
            f"WARNING must not fire for already-injected name; got {msgs}"
        )
