"""
Unit tests for the OPSIN format validator (FMT-01).

Tests that validate_name_format() correctly detects known
OPSIN-incompatible patterns and passes valid names.
"""

import pytest

from orthonym.validation.format_validator import validate_name_format


class TestValidateNameFormat:
    """Core format validation tests (FMT-01)."""

    def test_simple_valid_name(self):
        """Simple single-word name should pass."""
        ok, msg = validate_name_format("ethanol")
        assert ok is True
        assert msg == "ok"

    def test_valid_name_with_locants(self):
        """Name with locants should pass."""
        ok, msg = validate_name_format("2-methylpropan-1-ol")
        assert ok is True
        assert msg == "ok"

    def test_valid_ester_multiword(self):
        """Valid ester multi-word name should pass."""
        ok, msg = validate_name_format("methyl propanoate")
        assert ok is True
        assert msg == "ok"

    def test_valid_stereodescriptor(self):
        """Name with stereodescriptor in parens should pass."""
        ok, msg = validate_name_format("(2R)-butan-2-ol")
        assert ok is True
        assert msg == "ok"

    def test_valid_complex_name(self):
        """Complex valid name with nested brackets should pass."""
        ok, msg = validate_name_format(
            "(2S)-2-amino-3-(1H-indol-3-yl)propanoic acid"
        )
        assert ok is True
        assert msg == "ok"

    def test_unbalanced_parentheses(self):
        """Unbalanced parentheses should fail."""
        ok, msg = validate_name_format("bad(name")
        assert ok is False
        assert "unbalanced" in msg

    def test_unbalanced_closing_first(self):
        """Closing before opening should fail."""
        ok, msg = validate_name_format(")badname(")
        assert ok is False
        assert "unbalanced" in msg

    def test_unknown_multiword(self):
        """Unknown multi-word pattern should fail."""
        ok, msg = validate_name_format("methyl weird multiword name")
        assert ok is False
        assert "unknown_multi_word" in msg

    def test_empty_name(self):
        """Empty name should fail."""
        ok, msg = validate_name_format("")
        assert ok is False
        assert "empty" in msg

    def test_whitespace_only(self):
        """Whitespace-only name should fail."""
        ok, msg = validate_name_format("   ")
        assert ok is False
        assert "empty" in msg

    def test_empty_parentheses(self):
        """Name with empty () should fail."""
        ok, msg = validate_name_format("methyl()ethane")
        assert ok is False
        assert "empty_parentheses" in msg

    def test_bare_oxy(self):
        """Bare 'oxy' without qualifier should fail."""
        ok, msg = validate_name_format("2-oxy-propane")
        assert ok is False
        assert "bare_oxy" in msg

    def test_oxy_in_word_is_ok(self):
        """'oxy' as part of a larger word (methoxy, ethoxy) should pass."""
        ok, msg = validate_name_format("2-methoxyethanol")
        assert ok is True
        assert msg == "ok"

    def test_double_hyphen(self):
        """Double hyphens should fail."""
        ok, msg = validate_name_format("2--methylpropane")
        assert ok is False
        assert "double_hyphen" in msg

    def test_valid_vb_descriptor(self):
        """VB descriptors with dots in brackets should pass."""
        ok, msg = validate_name_format("bicyclo[2.2.1]heptane")
        assert ok is True
        assert msg == "ok"

    def test_valid_salt(self):
        """Salt name should pass multi-word check."""
        ok, msg = validate_name_format("sodium chloride")
        assert ok is True
        assert msg == "ok"

    def test_valid_anhydride(self):
        """Anhydride name should pass."""
        ok, msg = validate_name_format("acetic anhydride")
        assert ok is True
        assert msg == "ok"

    def test_valid_oxide(self):
        """Oxide name should pass."""
        ok, msg = validate_name_format("pyridine oxide")
        assert ok is True
        assert msg == "ok"

    def test_valid_acid_halide(self):
        """Acid halide name should pass."""
        ok, msg = validate_name_format("acetic acid chloride")
        assert ok is True
        assert msg == "ok"


class TestMultiWordValidation:
    """Tests for multi-word name validation (FMT-05)."""

    def test_ester_pattern(self):
        """Ester multi-word: 'methyl propanoate' should pass."""
        ok, msg = validate_name_format("methyl propanoate")
        assert ok is True

    def test_multi_ester_pattern(self):
        """Multi-ester: 'dimethyl butanedioate' should pass."""
        ok, msg = validate_name_format("dimethyl butanedioate")
        assert ok is True

    def test_anhydride_pattern(self):
        """Anhydride: 'acetic anhydride' should pass."""
        ok, msg = validate_name_format("acetic anhydride")
        assert ok is True

    def test_oxide_pattern(self):
        """Oxide: 'pyridine oxide' should pass."""
        ok, msg = validate_name_format("pyridine oxide")
        assert ok is True

    def test_sulfide_pattern(self):
        """Sulfide: 'dimethyl sulfide' should pass."""
        ok, msg = validate_name_format("dimethyl sulfide")
        assert ok is True

    def test_salt_pattern_sodium(self):
        """Salt: 'sodium acetate' should pass."""
        ok, msg = validate_name_format("sodium acetate")
        assert ok is True

    def test_salt_pattern_potassium(self):
        """Salt: 'potassium formate' should pass."""
        ok, msg = validate_name_format("potassium formate")
        assert ok is True

    def test_acid_chloride(self):
        """Acid chloride: 'propanoic acid chloride' should pass."""
        ok, msg = validate_name_format("propanoic acid chloride")
        assert ok is True

    def test_carbonyl_derivative(self):
        """Carbonyl derivative: 'acetaldehyde oxime' should pass."""
        ok, msg = validate_name_format("acetaldehyde oxime")
        assert ok is True

    def test_unknown_pattern_rejected(self):
        """Unknown multi-word pattern should be rejected."""
        ok, msg = validate_name_format("this is not a valid pattern")
        assert ok is False
        assert "unknown_multi_word" in msg

    def test_two_word_unknown(self):
        """Two-word name that doesn't match patterns should fail."""
        ok, msg = validate_name_format("methyl hexafluorogiggle")
        assert ok is False
        assert "unknown_multi_word" in msg


class TestBracketNesting:
    """Tests for bracket nesting hierarchy (FMT-04)."""

    def test_simple_parens(self):
        """Simple parentheses should pass."""
        ok, msg = validate_name_format("(2R)-butan-2-ol")
        assert ok is True

    def test_nested_parens_in_brackets(self):
        """Parentheses inside square brackets should pass."""
        ok, msg = validate_name_format("bicyclo[2.2.1]heptane")
        assert ok is True

    def test_complex_nesting(self):
        """Complex nested brackets should pass if balanced."""
        ok, msg = validate_name_format(
            "2-[3-(4-methylphenyl)propyl]naphthalene"
        )
        assert ok is True

    def test_unbalanced_square_bracket(self):
        """Unbalanced square bracket should fail."""
        ok, msg = validate_name_format("bicyclo[2.2.1heptane")
        assert ok is False
        assert "unbalanced" in msg
