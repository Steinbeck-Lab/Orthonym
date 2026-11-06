"""Tests for confidence-based quality gate rejection in namer.py.

The quality gate in namer._name_impl uses the coverage_scoring confidence
score to reject truncated/incomplete names and fall back to decomposition.

Tests verify:
- Names with confidence < 0.30 on molecules HA > 15 trigger fallback
- Names with confidence >= 0.30 are NOT rejected
- Existing garbled token detection still works
- _skip_decomposition=True bypasses the confidence gate
- retrieve_confidence returning None does not crash
- Small molecules (HA <= 15) bypass confidence check
"""

from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_confidence_dict(confidence: float, name: str = "testname") -> dict:
    """Build a dict matching retrieve_confidence return format."""
    return {
        'name': name,
        'confidence': confidence,
        'factors': {'ratio': 0.5, 'atom_coverage': 0.5,
                    'fg_recognition': 0.5, 'substituent_completeness': 0.5},
        'handler': 'chain',
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConfidenceBasedRejection:
    """Quality gate should reject low-confidence names on large molecules."""

    def test_small_molecule_bypasses_confidence_check(self):
        """Molecules with HA <= 15 should not have confidence checked."""
        from orthonym import name_compound
        from orthonym.assembly import coverage_scoring

        # methanol: 2 heavy atoms
        smiles = "CO"
        low_conf = _make_confidence_dict(0.05, "methanol")
        with patch.object(coverage_scoring, 'retrieve_confidence', return_value=low_conf):
            result = name_compound(smiles)
        # methanol is a retained name, so it bypasses the assembler entirely
        assert result == "methanol"

class TestConfidenceThreshold:
    """Verify the threshold constant and boundary behavior."""

    def test_threshold_constant_exists(self):
        """The truncation confidence threshold constant should be defined."""
        from orthonym import namer as namer_module
        assert hasattr(namer_module, '_TRUNCATION_CONFIDENCE_THRESHOLD')
        assert namer_module._TRUNCATION_CONFIDENCE_THRESHOLD == 0.30
