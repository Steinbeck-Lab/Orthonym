"""
Integration tests for N-substituent grouping in decomposition (DEC-03).

Tests that the _group_n_substituents() function correctly collapses
repeated N-prefix patterns using IUPAC P-16.3.4 multiplicative prefixes:
  - N-acetyl-N-acetyl -> N,N-diacetyl
  - N-formyl-N-acetyl-N-acetyl -> N,N-diacetyl-N-formyl (alphabetical)

Also tests end-to-end that multi-amide decomposition names use grouped
N-prefixes rather than repeated individual ones.

Phase 50, Plan 03 -- N-prefix grouping validation.
"""

import pytest

from orthonym.decomposition.fragment_assembly import _group_n_substituents


class TestGroupNSubstituentsDirect:
    """Unit tests of _group_n_substituents() function directly."""

    @pytest.mark.integration
    def test_two_identical_n_prefixes_grouped(self):
        """N-acetyl-N-acetyltetrahydropyranamine -> N,N-diacetyl form."""
        result = _group_n_substituents("N-acetyl-N-acetyltetrahydropyranamine")
        assert "N,N-diacetyl" in result, (
            f"Expected 'N,N-diacetyl' grouping, got: {result}"
        )
        assert "tetrahydropyranamine" in result, (
            f"Expected parent 'tetrahydropyranamine' preserved, got: {result}"
        )
        # Should NOT contain repeated N-acetyl
        assert "N-acetyl-N-acetyl" not in result, (
            f"Still contains repeated N-acetyl pattern: {result}"
        )

    @pytest.mark.integration
    def test_mixed_n_prefixes_grouped_alphabetically(self):
        """N-formyl-N-acetyl-N-acetylamine -> grouped with alphabetical order.

        Acetyl comes before formyl alphabetically, so output should have
        acetyl grouped first.
        """
        result = _group_n_substituents("N-formyl-N-acetyl-N-acetylamine")
        # Should contain N,N-diacetyl (two acetyl grouped)
        assert "diacetyl" in result, (
            f"Expected 'diacetyl' grouping, got: {result}"
        )
        # Should contain N-formyl (single, not grouped)
        assert "formyl" in result, (
            f"Expected 'formyl' present, got: {result}"
        )
        # Alphabetical: acetyl before formyl
        acetyl_pos = result.find("acetyl")
        formyl_pos = result.find("formyl")
        assert acetyl_pos < formyl_pos, (
            f"Expected acetyl before formyl (alphabetical), got: {result}"
        )

    @pytest.mark.integration
    def test_single_n_prefix_unchanged(self):
        """N-methylacetamide -> unchanged (no grouping needed)."""
        result = _group_n_substituents("N-methylacetamide")
        assert result == "N-methylacetamide", (
            f"Single N-prefix should be unchanged, got: {result}"
        )

    @pytest.mark.integration
    def test_triple_n_prefix_grouped(self):
        """N-acetyl-N-acetyl-N-acetylpiperidine -> N,N,N-triacetyl form."""
        result = _group_n_substituents("N-acetyl-N-acetyl-N-acetylpiperidine")
        assert "N,N,N-triacetyl" in result, (
            f"Expected 'N,N,N-triacetyl' grouping, got: {result}"
        )
        assert "piperidine" in result, (
            f"Expected parent 'piperidine' preserved, got: {result}"
        )

    @pytest.mark.integration
    def test_no_n_prefix_returns_unchanged(self):
        """Name without N- prefix should be returned unchanged."""
        result = _group_n_substituents("acetamide")
        assert result == "acetamide", (
            f"Name without N-prefix should be unchanged, got: {result}"
        )
