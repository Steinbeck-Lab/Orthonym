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
        """N-(4-methylcyclohexyl)acetamide: methyl must have a ring locant.

        The methyl group on the cyclohexyl ring must appear with an integer
        locant (e.g., "4-methylcyclohexyl"), not as bare "methylcyclohexyl".
        The current bug: name_substituent returns "methylcyclohexyl" without
        a locant, producing "N-methylcyclohexylacetamide" which is ambiguous.
        """
        result = name_compound("CC(=O)NC1CCC(C)CC1")
        assert result is not None
        # Must contain a digit-locanted methylcyclohexyl (e.g., "4-methylcyclohexyl")
        assert re.search(r'\d-methylcyclohexyl', result), (
            f"Expected locanted methylcyclohexyl (e.g., '4-methylcyclohexyl') in {result!r}"
        )

    @pytest.mark.integration
    def test_n_cyclohexylacetamide_unchanged(self):
        """N-cyclohexylacetamide: ring without sub-substituents stays correct.

        Regression guard: bare cyclohexyl N-substituent must not be affected.
        """
        result = name_compound("CC(=O)NC1CCCCC1")
        assert result == "N-cyclohexylacetamide", f"Expected 'N-cyclohexylacetamide', got {result!r}"

    @pytest.mark.integration
    def test_n_methylacetamide_unchanged(self):
        """N-methylacetamide: simple non-ring N-substituent stays correct.

        Regression guard: non-ring N-substituents must not be affected.
        """
        result = name_compound("CC(=O)NC")
        assert result == "N-methylacetamide", f"Expected 'N-methylacetamide', got {result!r}"

    @pytest.mark.integration
    def test_n_phenylacetamide_unchanged(self):
        """N-phenylacetamide: aromatic ring N-substituent stays correct.

        Regression guard: aromatic ring without sub-substituents must not change.
        """
        result = name_compound("CC(=O)Nc1ccccc1")
        assert result == "N-phenylacetamide", f"Expected 'N-phenylacetamide', got {result!r}"

    @pytest.mark.integration
    def test_n_dimethylcyclohexyl_acetamide_has_locants(self):
        """N-(3,5-dimethylcyclohexyl)acetamide: both methyls must have locants.

        Both methyl groups must appear with correct IUPAC locants and the
        'di' multiplier, e.g., "3,5-dimethylcyclohexyl".
        """
        result = name_compound("CC(=O)NC1CC(C)CC(C)C1")
        assert result is not None
        # Must contain locanted dimethylcyclohexyl (e.g., "3,5-dimethylcyclohexyl")
        assert re.search(r'\d,\d-dimethylcyclohexyl', result), (
            f"Expected locanted dimethylcyclohexyl (e.g., '3,5-dimethylcyclohexyl') in {result!r}"
        )

    @pytest.mark.integration
    def test_n_4_hydroxycyclohexyl_formamide(self):
        """N-(4-hydroxycyclohexyl)formamide: ring with hydroxy sub-substituent.

        The hydroxy group on the cyclohexyl ring must appear with a ring locant.
        """
        result = name_compound("O=CNC1CCC(O)CC1")
        assert result is not None
        # Must contain locanted hydroxycyclohexyl
        assert re.search(r'\d-hydroxycyclohexyl', result), (
            f"Expected locanted hydroxycyclohexyl in {result!r}"
        )


class TestAmideRingEnrichmentRegressionGuard:
    """Verify existing amide naming behavior is preserved."""

    @pytest.mark.integration
    def test_existing_amide_tests_pass(self):
        """Canary: key amide compounds from existing tests still work."""
        # N,N-dimethylformamide
        result = name_compound("CN(C)C=O")
        assert "dimethyl" in result, f"Expected 'dimethyl' in {result!r}"

        # N-ethylacetamide
        result = name_compound("CC(=O)NCC")
        assert "N-ethyl" in result, f"Expected 'N-ethyl' in {result!r}"

        # N-benzylacetamide
        result = name_compound("CC(=O)NCc1ccccc1")
        assert "benzyl" in result, f"Expected 'benzyl' in {result!r}"
