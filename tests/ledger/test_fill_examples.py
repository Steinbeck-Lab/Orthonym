"""
tests/ledger/test_fill_examples.py — Unit tests for fill_examples.py.

Tests (RED first, then GREEN after implementation):

1. extract_smiles recovers SMILES from a backtick-in-prose cell.
2. extract_smiles resolves a back-reference.
3. extract_smiles returns None for a skeleton-name cell.
4. extract_smiles returns None for a dash cell.
5. extract_expected returns None for non-name prose.
6. extract_expected returns a name from a backticked expected.
"""

from __future__ import annotations

import pytest

# Import functions under test — these will fail (RED) until fill_examples.py exists.
from scripts.ledger.fill_examples import extract_smiles, extract_expected


class TestExtractSmiles:
    def test_backtick_in_prose(self):
        """Backtick-wrapped SMILES surrounded by annotation text is recovered."""
        cell = "`OC[C@@H](O)[C@@H](O)[C@H](O)[C@H](O)C=O` (acyclic D-glucose)"
        result = extract_smiles(cell, last_valid_smiles=None)
        assert result == "OC[C@@H](O)[C@@H](O)[C@H](O)[C@H](O)C=O"

    def test_back_reference_as_above(self):
        """'(as above)' resolves to last_valid_smiles when no SMILES is in the cell."""
        result = extract_smiles("(as above)", last_valid_smiles="CCO")
        assert result == "CCO"

    def test_back_reference_same_as(self):
        """'same as P-24.6' resolves to last_valid_smiles."""
        result = extract_smiles("same as P-24.6", last_valid_smiles="C1CCCC1")
        assert result == "C1CCCC1"

    def test_back_reference_above(self):
        """'(above)' in text (e.g. 'quinovose (above)') resolves to last_valid_smiles."""
        result = extract_smiles("quinovose (above)", last_valid_smiles="CC1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")
        assert result == "CC1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"

    def test_skeleton_name_returns_none(self):
        """A backtick-wrapped skeleton name that is not valid SMILES returns None."""
        # '21H-biline' is a valid IUPAC name but NOT a valid SMILES string
        result = extract_smiles("`21H-biline` skeleton", last_valid_smiles=None)
        assert result is None

    def test_abietane_skeleton_returns_smiles(self):
        """A cell with both text AND a backticked SMILES returns the SMILES."""
        cell = "`CC(C)C1CCC2(C)C(CCC3C2CCC3)C1` (abietane skeleton)"
        result = extract_smiles(cell, last_valid_smiles=None)
        assert result is not None
        # Should be the backticked SMILES
        assert result.startswith("CC(C)C1CCC2")

    def test_dash_returns_none(self):
        """A dash-only cell returns None."""
        assert extract_smiles("—", last_valid_smiles="CCO") is None
        assert extract_smiles("—", last_valid_smiles=None) is None

    def test_empty_returns_none(self):
        """An empty cell returns None."""
        assert extract_smiles("", last_valid_smiles="CCO") is None

    def test_back_reference_with_no_last_returns_none(self):
        """A back-reference with no prior SMILES returns None."""
        assert extract_smiles("(as above)", last_valid_smiles=None) is None

    def test_plain_smiles_no_backtick(self):
        """A bare valid SMILES string (no backticks) is parsed directly."""
        result = extract_smiles("CCO", last_valid_smiles=None)
        assert result == "CCO"


class TestExtractExpected:
    def test_prose_returns_none(self):
        """Non-name prose like 'substitutive form acceptable' returns None."""
        result = extract_expected("substitutive form acceptable")
        assert result is None

    def test_acceptable_keyword_returns_none(self):
        """Cell containing 'acceptable' returns None."""
        result = extract_expected("all = acceptable")
        assert result is None

    def test_backticked_name_returned(self):
        """Backtick-wrapped name is returned (stripped of backticks)."""
        result = extract_expected("`2-methyloxathiolane`")
        assert result == "2-methyloxathiolane"

    def test_bare_name_returned(self):
        """A bare name (no backticks, no prose markers) is returned."""
        result = extract_expected("ethanol")
        assert result == "ethanol"

    def test_dash_returns_none(self):
        """A dash-only cell returns None."""
        assert extract_expected("—") is None
        assert extract_expected("") is None

    def test_truncates_at_semicolon(self):
        """Cell truncated at first '; ' to extract only the first name."""
        result = extract_expected("2-methylbutane ; 3-methylbutane")
        assert result == "2-methylbutane"

    def test_correct_keyword_returns_none(self):
        """Cell containing 'correct' (definitional) returns None."""
        result = extract_expected("correct (PIN)")
        assert result is None

    def test_pin_paren_keyword_returns_none(self):
        """Bare (no backtick) cell containing 'PIN)' annotation returns None."""
        result = extract_expected("methanol (PIN)")
        # No backtick → prose marker "pin)" triggers None
        assert result is None

    def test_backtick_name_with_pin_annotation(self):
        """Backtick-wrapped name followed by (PIN) annotation → returns the name."""
        result = extract_expected("`methyl propanoate` (PIN)")
        # Backtick token is the name; surrounding "(PIN)" is just annotation
        assert result == "methyl propanoate"

    def test_backtick_name_with_pin_annotation_complex(self):
        """Backtick SMILES-in-expected-like name with (PIN) annotation."""
        result = extract_expected("`4,4'-sulfanediyldibenzoic acid` (PIN)")
        assert result == "4,4'-sulfanediyldibenzoic acid"
