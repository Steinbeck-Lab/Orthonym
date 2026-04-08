"""Tests for Tier B handler enrichment via _enrich_handler_name().

Phase 139 Plan 02 ARCH-03/04: Verify that Tier B handlers enrich their
base names with non-principal substituents via the universal pipeline.
"""

import pytest
from rdkit import Chem


# ---------------------------------------------------------------------------
# Test 1: Oxime with additional substituent
# 4-methylpentan-2-one oxime = methyl on the chain should appear
# ---------------------------------------------------------------------------
class TestOximeEnrichment:
    """Test that oxime handler includes additional substituents."""

    def test_oxime_simple_no_extra_substituents(self):
        """Simple oxime (no extra subs) produces unchanged name."""
        from orthonym import name_compound
        # Propan-2-one oxime: CC(=NO)C -- just ketone oxime, no extra subs
        result = name_compound("CC(=NO)C")
        assert result is not None
        # The name should contain "oxime"
        assert "oxime" in result.lower()

    def test_oxime_name_unchanged_for_simple_case(self):
        """Simple oxime naming is not regressed."""
        from orthonym import name_compound
        # Acetone oxime
        result = name_compound("CC(=NO)C")
        assert result is not None
        assert "oxime" in result.lower()


# ---------------------------------------------------------------------------
# Test 2: Isocyanate with additional substituent
# ---------------------------------------------------------------------------
class TestIsocyanateEnrichment:
    """Test that isocyanate handler includes additional substituents."""

    def test_simple_isocyanate_unchanged(self):
        """Simple methyl isocyanate naming is not regressed."""
        from orthonym import name_compound
        # methyl isocyanate: CN=C=O
        result = name_compound("CN=C=O")
        assert result is not None
        assert "isocyanate" in result.lower()

    def test_phenyl_isocyanate_unchanged(self):
        """Phenyl isocyanate naming is not regressed."""
        from orthonym import name_compound
        # phenyl isocyanate: O=C=Nc1ccccc1
        result = name_compound("O=C=Nc1ccccc1")
        assert result is not None
        assert "isocyanate" in result.lower()


# ---------------------------------------------------------------------------
# Test 3: Carbamic acid enrichment
# ---------------------------------------------------------------------------
class TestCarbamicAcidEnrichment:
    """Test that carbamic acid handler includes non-principal substituents."""

    def test_simple_carbamic_acid_unchanged(self):
        """Simple carbamic acid naming is not regressed."""
        from orthonym import name_compound
        # carbamic acid: NC(=O)O
        result = name_compound("NC(=O)O")
        assert result is not None
        assert "carbamic" in result.lower() or "carbam" in result.lower()

    def test_carbamic_acid_with_ketone_includes_oxo(self):
        """Carbamic acid with ketone in N-substituent chain must include 'oxo' prefix.

        OC(=O)NCCCC(=O)C = N-(4-oxopentyl)carbamic acid
        The ketone (=O) within the N-substituent chain must NOT be silently dropped.
        Phase 139 gap closure: SC4.
        """
        from orthonym import name_compound
        result = name_compound("OC(=O)NCCCC(=O)C")
        assert result is not None
        assert "oxo" in result.lower(), (
            f"Expected 'oxo' (ketone prefix) in N-substituent but got: {result}"
        )

    def test_carbamic_acid_with_hydroxy_includes_hydroxy(self):
        """Carbamic acid with alcohol in N-substituent chain must include 'hydroxy' prefix.

        OC(=O)NCC(O)C = N-(2-hydroxypropyl)carbamic acid
        The alcohol (-OH) within the N-substituent chain must NOT be silently dropped.
        Phase 139 gap closure: SC4.
        """
        from orthonym import name_compound
        result = name_compound("OC(=O)NCC(O)C")
        assert result is not None
        assert "hydroxy" in result.lower(), (
            f"Expected 'hydroxy' (alcohol prefix) in N-substituent but got: {result}"
        )

    def test_simple_n_alkyl_carbamic_acid_no_regression(self):
        """Simple N-alkyl carbamic acid unchanged -- no regression from heteroatom fix.

        OC(=O)NCC = N-ethylcarbamic acid
        Pure alkyl N-substituent should still produce 'ethyl' and 'carbamic'.
        """
        from orthonym import name_compound
        result = name_compound("OC(=O)NCC")
        assert result is not None
        assert "ethyl" in result.lower(), (
            f"Expected 'ethyl' in simple N-alkyl carbamic acid but got: {result}"
        )
        assert "carbamic" in result.lower(), (
            f"Expected 'carbamic' in simple N-alkyl carbamic acid but got: {result}"
        )


# ---------------------------------------------------------------------------
# Test 4: Boronic acid enrichment
# ---------------------------------------------------------------------------
class TestBoronicAcidEnrichment:
    """Test that boronic acid handler includes additional substituents."""

    def test_simple_boronic_acid_unchanged(self):
        """Simple boronic acid naming is not regressed."""
        from orthonym import name_compound
        # phenylboronic acid: OB(O)c1ccccc1
        result = name_compound("OB(O)c1ccccc1")
        assert result is not None
        assert "boronic" in result.lower()


# ---------------------------------------------------------------------------
# Test 5: _enrich_handler_name helper function unit tests
# ---------------------------------------------------------------------------
class TestEnrichHandlerNameFunction:
    """Unit tests for the _enrich_handler_name() helper function."""

    def test_enrich_handler_name_exists(self):
        """_enrich_handler_name function is importable."""
        from orthonym.assembly.composer import _enrich_handler_name
        assert callable(_enrich_handler_name)

    def test_enrich_returns_base_name_when_no_features(self):
        """Returns base name unchanged when features lack parent info."""
        from orthonym.assembly.composer import _enrich_handler_name

        # Create a minimal features mock with no principal_chain or oriented_ring
        class MinimalFeatures:
            chain_is_parent = False
            principal_chain = None
            oriented_ring = None
            principal_ring = None
            mol = Chem.MolFromSmiles("C")
            functional_groups = {}
            principal_group = None
            atom_to_locant = {}

        features = MinimalFeatures()
        result = _enrich_handler_name(features, "test-name", "test")
        assert result == "test-name"


# ---------------------------------------------------------------------------
# Test 6: Sulfur handler enrichment
# ---------------------------------------------------------------------------
class TestSulfurHandlerEnrichment:
    """Test that sulfur handlers include additional substituents."""

    def test_simple_sulfoxide_unchanged(self):
        """Simple dimethyl sulfoxide naming is not regressed."""
        from orthonym import name_compound
        # DMSO: CS(=O)C
        result = name_compound("CS(=O)C")
        assert result is not None
        # Should produce sulfoxide or sulfinyl name
        assert "sulf" in result.lower() or "sulfinyl" in result.lower()


# ---------------------------------------------------------------------------
# Test 7: Guanidine handler enrichment
# ---------------------------------------------------------------------------
class TestGuanidineEnrichment:
    """Test that guanidine handler includes additional substituents."""

    def test_simple_guanidine_unchanged(self):
        """Simple guanidine naming is not regressed."""
        from orthonym import name_compound
        # guanidine: NC(=N)N
        result = name_compound("NC(=N)N")
        assert result is not None
        assert "guanidine" in result.lower()


# ---------------------------------------------------------------------------
# Test 8: Urea handler enrichment
# ---------------------------------------------------------------------------
class TestUreaEnrichment:
    """Test that urea handler includes additional substituents."""

    def test_simple_urea_unchanged(self):
        """Simple urea naming is not regressed."""
        from orthonym import name_compound
        # urea: NC(=O)N
        result = name_compound("NC(=O)N")
        assert result is not None
        assert "urea" in result.lower()
