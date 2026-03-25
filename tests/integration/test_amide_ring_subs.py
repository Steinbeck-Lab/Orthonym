"""
Tests for amide handler ring substituent naming (Phase 121, Plan 02).

Verifies that N-substituent fragments containing rings get sub-substituent
prefixes with correct IUPAC locants via the universal discovery pipeline.

Covers:
  ASML-08: Ring enrichment for N-substituent naming in amide handler
  - Ring-containing N-substituents get sub-substituent prefixes with locants
  - Ring-only N-substituents are returned unchanged
  - Simple non-ring N-substituents are unaffected
"""

import re
import pytest
from orthonym import name_compound


class TestAmideRingSubstituentEnrichment:
    """Verify ring-containing N-substituent fragments include sub-substituent prefixes."""

    @pytest.mark.integration
    def test_n_4_methylcyclohexyl_acetamide_has_locant(self):
        """N-(4-methylcyclohexyl)acetamide: methyl must have a ring locant."""
        result = name_compound("CC(=O)NC1CCC(C)CC1")
        assert result is not None
        assert re.search(r'\d-methylcyclohexyl', result), (
            f"Expected locanted methylcyclohexyl in {result!r}"
        )

    @pytest.mark.integration
    def test_n_cyclohexylacetamide_unchanged(self):
        """N-cyclohexylacetamide: regression guard for bare ring N-substituent."""
        result = name_compound("CC(=O)NC1CCCCC1")
        assert result == "N-cyclohexylacetamide", f"Got {result!r}"

    @pytest.mark.integration
    def test_n_methylacetamide_unchanged(self):
        """N-methylacetamide: regression guard for simple non-ring N-substituent."""
        result = name_compound("CC(=O)NC")
        assert result == "N-methylacetamide", f"Got {result!r}"

    @pytest.mark.integration
    def test_n_phenylacetamide_unchanged(self):
        """N-phenylacetamide: regression guard for aromatic ring N-substituent."""
        result = name_compound("CC(=O)Nc1ccccc1")
        assert result == "N-phenylacetamide", f"Got {result!r}"

    @pytest.mark.integration
    def test_n_dimethylcyclohexyl_acetamide_has_locants(self):
        """N-(3,5-dimethylcyclohexyl)acetamide: both methyls must have locants."""
        result = name_compound("CC(=O)NC1CC(C)CC(C)C1")
        assert result is not None
        assert re.search(r'\d,\d-dimethylcyclohexyl', result), (
            f"Expected locanted dimethylcyclohexyl in {result!r}"
        )

    @pytest.mark.integration
    def test_n_4_methylcyclohexyl_propanamide(self):
        """N-(4-methylcyclohexyl)propanamide: ring N-sub with longer acyl chain."""
        result = name_compound("CCC(=O)NC1CCC(C)CC1")
        assert result is not None
        assert re.search(r'\d-methylcyclohexyl', result), (
            f"Expected locanted methylcyclohexyl in {result!r}"
        )

    @pytest.mark.integration
    def test_n_tert_butyl_not_bracketed(self):
        """N-tert-butylpropanamide: retained name must NOT be parenthesized."""
        result = name_compound("CCC(=O)NC(C)(C)C")
        assert result == "N-tert-butylpropanamide", f"Got {result!r}"

    @pytest.mark.integration
    def test_complex_n_sub_parenthesized(self):
        """Complex N-substituent names with locants get parenthesized."""
        result = name_compound("CC(=O)NC1CCC(C)CC1")
        assert result is not None
        assert "(" in result and ")" in result, (
            f"Expected parenthesized N-substituent in {result!r}"
        )


class TestAmideRingEnrichmentRegressionGuard:
    """Verify existing amide naming behavior is preserved."""

    @pytest.mark.integration
    def test_existing_amide_tests_pass(self):
        """Canary: key amide compounds from existing tests still work."""
        result = name_compound("CN(C)C=O")
        assert "dimethyl" in result, f"Expected 'dimethyl' in {result!r}"

        result = name_compound("CC(=O)NCC")
        assert "N-ethyl" in result, f"Expected 'N-ethyl' in {result!r}"

        result = name_compound("CC(=O)NCc1ccccc1")
        assert "benzyl" in result, f"Expected 'benzyl' in {result!r}"
