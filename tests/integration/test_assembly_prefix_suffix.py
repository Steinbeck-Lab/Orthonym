"""Integration tests for assembly prefix/suffix correctness (Phase 131)."""
import pytest
from orthonym.namer import name_compound


@pytest.mark.integration
class TestASML12DihydroxyMerge:
    """ASML-12: Identical substituent prefixes merged with multiplicative prefix."""

    def test_dihydroxypentanoic_acid(self):
        result = name_compound("OCC(O)CCC(=O)O")
        assert result is not None
        assert "dihydroxy" in result, f"Expected dihydroxy, got: {result}"
        assert "pentanoic acid" in result, f"Expected pentanoic acid, got: {result}"

    def test_no_duplicate_hydroxy_stacking(self):
        """Should not produce '5-hydroxy-4-hydroxy...' pattern."""
        result = name_compound("OCC(O)CCC(=O)O")
        # Count occurrences of 'hydroxy' -- should be exactly 1 (as part of 'dihydroxy')
        count = result.count("hydroxy")
        assert count == 1, f"Expected 1 'hydroxy' occurrence (in dihydroxy), got {count} in: {result}"

    def test_bare_bare_merge_regression(self):
        """ASML-12 regression: two bare 'hydroxy' (no locants) still merge to 'dihydroxy'.

        This tests the existing bare-merge behavior is NOT broken by the
        locant-aware merge addition. Bare prefixes are those without locants,
        e.g., two 'hydroxy' from different FG subtypes on an unlocanted chain.
        """
        from orthonym.rules.polyfunctional import _merge_bare_duplicate_prefixes
        # Direct unit test of bare merge
        result = _merge_bare_duplicate_prefixes(["hydroxy", "hydroxy"])
        assert len(result) == 1, f"Expected 1 merged prefix, got: {result}"
        assert "dihydroxy" in result[0], f"Expected dihydroxy, got: {result}"


@pytest.mark.integration
class TestASML13PhenolRouting:
    """ASML-13: Ring hydroxyl uses suffix form when OH is principal group."""

    def test_chlorophenol(self):
        result = name_compound("c1cc(O)c(Cl)cc1")
        assert result is not None
        assert "phenol" in result.lower(), f"Expected phenol, got: {result}"
        assert "chloro" in result.lower(), f"Expected chloro prefix, got: {result}"

    def test_plain_phenol(self):
        result = name_compound("Oc1ccccc1")
        assert result is not None
        assert result.lower() == "phenol", f"Expected phenol, got: {result}"

    def test_hydroxybenzoic_acid_unchanged(self):
        """OH stays as prefix when acid is the principal group."""
        result = name_compound("OC1=CC=C(C(O)=O)C=C1")
        assert result is not None
        assert "hydroxy" in result.lower(), f"Expected hydroxy prefix, got: {result}"
        assert "benzoic acid" in result.lower(), f"Expected benzoic acid, got: {result}"

    def test_methylphenol(self):
        """Cresol: should produce methylphenol not methylhydroxybenzene."""
        result = name_compound("Cc1ccc(O)cc1")
        assert result is not None
        assert "phenol" in result.lower(), f"Expected phenol, got: {result}"
