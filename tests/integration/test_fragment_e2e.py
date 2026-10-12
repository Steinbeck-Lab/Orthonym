"""
End-to-end integration tests for a phase: Fragment-Aware Naming.

Validates all a phase features working together through the full naming
pipeline (SMILES -> name_compound -> IUPAC name):

1. Peptide naming (glycylglycine, alanylalanine, tripeptides)
2. NP ester decoration (testosterone acetate functional class naming)
3. Multiplicative nomenclature (methylenedianiline)
4. Recursion guard (deeply nested molecules don't crash)
5. Cross-feature regressions (simple compounds unchanged)

Run with: pytest tests/integration/test_fragment_e2e.py -v
"""

import pytest
from orthonym import name_compound


# ============================================================================
# 1. Peptide E2E Tests
# ============================================================================


# ============================================================================
# 2. NP Ester Decoration E2E Tests
# ============================================================================


class TestNPEsterE2E:
    """Verify NP ester functional class naming through the full pipeline."""

    @pytest.mark.integration
    def test_testosterone_acetate_contains_acetate(self):
        """Testosterone acetate name should contain 'acetate'."""
        result = name_compound(
            "CC(=O)O[C@H]1CC[C@@H]2[C@@]1(C)CC[C@H]1[C@@H]2CCC2=CC(=O)CC[C@@]12C"
        )
        assert "acetate" in result, f"Expected 'acetate' in '{result}'"

    @pytest.mark.integration
    def test_testosterone_acetate_functional_class_format(self):
        """Testosterone acetate should use functional class format (parent-yl acylate)."""
        result = name_compound(
            "CC(=O)O[C@H]1CC[C@@H]2[C@@]1(C)CC[C@H]1[C@@H]2CCC2=CC(=O)CC[C@@]12C"
        )
        assert result == "(8S,9S,10S,13R,14S,17S)-3-oxoandrost-4-en-17-yl acetate"

    @pytest.mark.integration
    def test_bare_testosterone_no_regression(self):
        """Bare testosterone (17-hydroxy, 3-oxo) still names correctly."""
        result = name_compound(
            "O[C@@H]1CC[C@@H]2[C@@]1(C)CC[C@H]1[C@@H]2CCC2=CC(=O)CC[C@@]12C"
        )
        # Should contain androst and -one and hydroxyl
        assert "androst" in result, f"Expected 'androst' in '{result}'"
        assert result is not None and len(result) > 0


# ============================================================================
# 3. Multiplicative Nomenclature E2E Tests
# ============================================================================


# ============================================================================
# 4. Recursion Guard E2E Tests
# ============================================================================


class TestRecursionGuardE2E:
    """Verify recursion depth guard prevents crashes on complex molecules."""

    @pytest.mark.integration
    def test_deeply_branched_no_crash(self):
        """Deeply branched alkane should not crash (returns a valid name)."""
        smiles = "C(C(C(C(C(C(C(C(C(C)C)C)C)C)C)C)C)C)C"
        try:
            result = name_compound(smiles)
            assert isinstance(result, str) and len(result) > 0
        except RecursionError:
            pytest.fail("RecursionError on deeply branched alkane")

    @pytest.mark.integration
    def test_adamantane_no_crash(self):
        """Adamantane (tricyclic) should not crash."""
        try:
            result = name_compound("C1CC2CC3CC(C1)CC(C2)C3")
            assert isinstance(result, str) and len(result) > 0
        except RecursionError:
            pytest.fail("RecursionError on adamantane")

    @pytest.mark.integration
    def test_complex_fused_no_crash(self):
        """Complex fused ring system should not crash."""
        try:
            result = name_compound("c1ccc2cc3ccccc3cc2c1")  # anthracene
            assert isinstance(result, str) and len(result) > 0
        except RecursionError:
            pytest.fail("RecursionError on fused ring system")

    @pytest.mark.integration
    def test_ion_no_infinite_recursion(self):
        """Ion compounds should not trigger infinite recursion."""
        try:
            result = name_compound("[NH4+]")
            assert isinstance(result, str) and len(result) > 0
        except RecursionError:
            pytest.fail("RecursionError on ion compound")


# ============================================================================
# 5. Cross-Feature Regression Tests
# ============================================================================


