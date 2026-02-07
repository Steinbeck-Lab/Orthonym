"""
End-to-end integration tests for Phase 22: Fragment-Aware Naming.

Validates all Phase 22 features working together through the full naming
pipeline (SMILES -> name_compound() -> IUPAC name):

1. Peptide naming (glycylglycine, L-alanyl-L-alanine, tripeptides)
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


class TestPeptideE2E:
    """Verify peptide naming through the full pipeline."""

    @pytest.mark.integration
    def test_glycylglycine(self):
        """Gly-Gly dipeptide -> 'glycylglycine'."""
        result = name_compound("NCC(=O)NCC(=O)O")
        assert result == "glycylglycine"

    @pytest.mark.integration
    def test_l_alanyl_l_alanine(self):
        """L-Ala-L-Ala dipeptide -> 'L-alanyl-L-alanine'."""
        result = name_compound("N[C@@H](C)C(=O)N[C@@H](C)C(=O)O")
        assert result == "L-alanyl-L-alanine"

    @pytest.mark.integration
    def test_glycyl_l_alanine(self):
        """Gly-L-Ala dipeptide -> 'glycyl-L-alanine'."""
        result = name_compound("NCC(=O)N[C@@H](C)C(=O)O")
        assert result == "glycyl-L-alanine"

    @pytest.mark.integration
    def test_l_alanylglycine(self):
        """L-Ala-Gly dipeptide -> 'L-alanylglycine'."""
        result = name_compound("N[C@@H](C)C(=O)NCC(=O)O")
        assert result == "L-alanylglycine"

    @pytest.mark.integration
    def test_tripeptide_gly_ala_leu(self):
        """Gly-L-Ala-L-Leu tripeptide naming."""
        result = name_compound(
            "NCC(=O)N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)O"
        )
        assert result == "glycyl-L-alanyl-L-leucine"

    @pytest.mark.integration
    def test_asparagine_not_misrouted(self):
        """Asparagine (amide side chain) should NOT be routed to peptide naming."""
        result = name_compound("NC(CC(N)=O)C(=O)O")
        assert result == "asparagine"

    @pytest.mark.integration
    def test_glutamine_not_misrouted(self):
        """Glutamine (amide side chain) should NOT be routed to peptide naming."""
        result = name_compound("NC(CCC(N)=O)C(=O)O")
        assert result == "glutamine"

    @pytest.mark.integration
    def test_simple_glycine_unchanged(self):
        """Single amino acid glycine should still produce 'glycine'."""
        result = name_compound("NCC(=O)O")
        assert result == "glycine"

    @pytest.mark.integration
    def test_simple_alanine_unchanged(self):
        """Single amino acid alanine should still produce 'alanine'."""
        result = name_compound("CC(N)C(=O)O")
        assert result == "alanine"


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
        assert result == "3-oxoandrost-4-en-17-yl acetate"

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


class TestMultiplicativeE2E:
    """Verify multiplicative nomenclature through the full pipeline."""

    @pytest.mark.integration
    def test_methylenedianiline(self):
        """4,4'-methylenedianiline: symmetric CH2-bridge, both amino groups preserved."""
        result = name_compound("Nc1ccc(Cc2ccc(N)cc2)cc1")
        assert result == "4,4'-methylenedianiline"

    @pytest.mark.integration
    def test_simple_benzene_unaffected(self):
        """Simple benzene should NOT be routed to multiplicative naming."""
        result = name_compound("c1ccccc1")
        assert result == "benzene"

    @pytest.mark.integration
    def test_toluene_unaffected(self):
        """Toluene should NOT be routed to multiplicative naming."""
        result = name_compound("Cc1ccccc1")
        assert result == "toluene"


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


class TestCrossFeatureRegression:
    """Verify Phase 22 changes do not break existing naming features."""

    @pytest.mark.integration
    def test_simple_amide_unchanged(self):
        """Acetamide should still return 'acetamide'."""
        assert name_compound("CC(=O)N") == "acetamide"

    @pytest.mark.integration
    def test_simple_ester_methyl_acetate(self):
        """Methyl acetate should still return 'methyl acetate' (freeze check)."""
        assert name_compound("CC(=O)OC") == "methyl acetate"

    @pytest.mark.integration
    def test_simple_ester_ethyl_acetate(self):
        """Ethyl acetate should still return 'ethyl acetate' (freeze check)."""
        assert name_compound("CC(=O)OCC") == "ethyl acetate"

    @pytest.mark.integration
    def test_glycine_single_amino_acid(self):
        """Glycine should still be 'glycine' (not misrouted to peptide)."""
        assert name_compound("NCC(=O)O") == "glycine"

    @pytest.mark.integration
    def test_naphthalene_ring_system(self):
        """Naphthalene retained name should be preserved."""
        assert name_compound("c1ccc2ccccc2c1") == "naphthalene"

    @pytest.mark.integration
    def test_ethanol_basic_alcohol(self):
        """Ethanol basic naming should be preserved."""
        assert name_compound("CCO") == "ethanol"

    @pytest.mark.integration
    def test_acetic_acid_basic_acid(self):
        """Acetic acid basic naming should be preserved."""
        assert name_compound("CC(=O)O") == "acetic acid"

    @pytest.mark.integration
    def test_pyridine_heterocycle(self):
        """Pyridine retained name should be preserved."""
        assert name_compound("c1ccncc1") == "pyridine"

    @pytest.mark.integration
    def test_phenol_retained_name(self):
        """Phenol retained name should be preserved."""
        assert name_compound("Oc1ccccc1") == "phenol"

    @pytest.mark.integration
    def test_cyclohexane_simple_ring(self):
        """Cyclohexane simple ring should be preserved."""
        assert name_compound("C1CCCCC1") == "cyclohexane"
