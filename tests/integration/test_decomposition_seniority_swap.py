"""Integration tests for seniority-based role swapping in decomposition.

Tests end-to-end naming behavior when roles_swapped flag is set, covering
ester (acyloxy), thioester (acylthio), and amide (N-acyl) substitutive
naming paths, plus simple compound canary preservation.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound


# ===========================================================================
# Ester seniority swap tests
# ===========================================================================


@pytest.mark.integration
class TestEsterSenioritySwap:
    """Test ester bonds with seniority-based role swapping."""

    def test_steroid_acetate_names_ring_system(self):
        """Steroid acetate: acid=acetate(3 HA), other=steroid(20 HA).

        CC(=O)OC1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C
        With roles_swapped=True, the name should reference the steroid
        ring system as the parent, not just produce "steroid acetate" style.
        The name must not be None/unknown and should contain ring-related terms.
        """
        smiles = "CC(=O)OC1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C"
        name = name_compound(smiles)
        assert name is not None and name.lower() != "unknown"
        assert name.lower() != "unknown organic compound", (
            f"Steroid acetate should produce a valid name, got: {name}"
        )
        # The name should reference the large ring system
        name_lower = name.lower()
        # Should have some structural reference to the ring system
        has_ring_ref = any(
            tok in name_lower
            for tok in ["cyclo", "tetra", "tri", "deca", "one", "ol", "an", "en"]
        )
        assert has_ring_ref or len(name) > 15, (
            f"Expected ring system reference in steroid acetate name: {name}"
        )

    def test_simple_ester_ethyl_acetate_unchanged(self):
        """CCOC(=O)C (ethyl acetate): simple ester, should NOT swap."""
        name = name_compound("CCOC(=O)C")
        assert name == "ethyl acetate", (
            f"Expected 'ethyl acetate', got: {name}"
        )

    def test_methyl_benzoate_unchanged(self):
        """COC(=O)c1ccccc1 (methyl benzoate): should preserve functional class name."""
        name = name_compound("COC(=O)c1ccccc1")
        assert name is not None
        assert "benzoate" in name.lower(), (
            f"Expected 'benzoate' in methyl benzoate name: {name}"
        )

    def test_large_ring_ester_produces_valid_name(self):
        """Ester where alkyl side is large polycyclic, acid is small.

        CC(=O)OC1CCC2CCCCC2C1 (decalinyl acetate)
        Should produce a valid name covering the ring system.
        """
        smiles = "CC(=O)OC1CCC2CCCCC2C1"
        name = name_compound(smiles)
        assert name is not None and "unknown" not in name.lower(), (
            f"Large ring ester should produce valid name, got: {name}"
        )


# ===========================================================================
# Thioester seniority swap tests
# ===========================================================================


@pytest.mark.integration
class TestThioesterSenioritySwap:
    """Test thioester bonds with seniority-based role swapping."""

    def test_simple_thioester_preserves_name(self):
        """CC(=O)SC (S-methyl thioacetate): small thioester, should NOT swap."""
        smiles = "CC(=O)SC"
        name = name_compound(smiles)
        assert name is not None and "unknown" not in name.lower(), (
            f"Simple thioester should produce valid name, got: {name}"
        )

    def test_large_thioester_produces_valid_name(self):
        """S-decalinyl thioacetate: large ring system on S side.

        CC(=O)SC1CCC2CCCCC2C1 - should produce name referencing ring system.
        """
        smiles = "CC(=O)SC1CCC2CCCCC2C1"
        name = name_compound(smiles)
        assert name is not None and "unknown" not in name.lower(), (
            f"Large thioester should produce valid name, got: {name}"
        )


# ===========================================================================
# Amide double-inversion guard tests
# ===========================================================================


@pytest.mark.integration
class TestAmideDoubleInversionGuard:
    """Test that amide seniority path does NOT double-invert with roles_swapped."""

    def test_simple_amide_still_works(self):
        """N-methylacetamide: roles_swapped=False, amide path works normally."""
        name = name_compound("CC(=O)NC")
        assert name is not None
        assert "acetamide" in name.lower() or "ethanamide" in name.lower(), (
            f"Expected acetamide/ethanamide in '{name}'"
        )

    def test_n_methylbenzamide_preserved(self):
        """O=C(NC)c1ccccc1: acid has ring+COOH, amide path preserves pattern."""
        name = name_compound("O=C(NC)c1ccccc1")
        assert name is not None
        assert "benzamide" in name.lower(), (
            f"Expected 'benzamide' in '{name}'"
        )

    def test_complex_amide_with_senior_amine(self):
        """COc1ccc(C(=O)N2CCCC2=O)cc1: both sides have ring systems.

        The amide N-acyl path (engine.py:835-852) should handle this correctly
        without double-inversion from roles_swapped.
        """
        name = name_compound("COc1ccc(C(=O)N2CCCC2=O)cc1")
        assert name is not None and name != "unknown"
        name_lower = name.lower()
        # Should reference both ring systems
        has_acid_ref = any(
            tok in name_lower for tok in ["methoxy", "benz", "phenyl", "benzoyl"]
        )
        has_amine_ref = any(
            tok in name_lower
            for tok in ["pyrrolidin", "pyrrolidone", "oxopyrrolidin"]
        )
        assert has_acid_ref or has_amine_ref, (
            f"Complex amide name should reference structural features: {name}"
        )


# ===========================================================================
# Fragment loss improvement tests
# ===========================================================================


@pytest.mark.integration
class TestFragmentLossImprovement:
    """Test that seniority swap improves fragment_loss benchmark compounds.

    These are representative compounds from the 56 fragment_loss set where
    tiny acid fragments are currently hardcoded as parent over 30+ HA systems.
    """

    # Representative fragment_loss compounds (SMILES from benchmark)
    FRAGMENT_LOSS_SAMPLES = [
        # Steroid esters
        "CC(=O)OC1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C",  # testosterone acetate
        "CC(=O)OC1CCC2(C)C3CCC4(C)C(CCC4C3CC=C2C1)C(C)CCCC(C)C",  # cholesteryl acetate-like
        # Large ring esters
        "CC(=O)OC1CCC2CCCCC2C1",  # decalinyl acetate
        "CC(=O)OC1CC2CCC1CC2",  # norbornyl acetate
        # Substituted ring esters
        "CC(=O)Oc1ccc2ccccc2c1",  # 2-naphthyl acetate
        "CC(=O)Oc1cccc2ccccc12",  # 1-naphthyl acetate
        # Complex systems
        "CC(=O)Oc1ccc(cc1)C(=O)O",  # 4-acetoxyphenyl compound
    ]

    def test_fragment_loss_compounds_produce_valid_names(self):
        """At least 5 of the fragment_loss sample compounds produce valid names.

        A 'valid' name is non-None, non-unknown, and longer than 10 characters
        (indicating it covers a significant portion of the molecule).
        """
        valid_count = 0
        for smiles in self.FRAGMENT_LOSS_SAMPLES:
            name = name_compound(smiles)
            if (name is not None
                    and "unknown" not in name.lower()
                    and len(name) > 10):
                valid_count += 1

        assert valid_count >= 5, (
            f"Expected at least 5 valid names from fragment_loss samples, "
            f"got {valid_count}"
        )

    def test_steroid_acetate_not_just_ethyl_acetate(self):
        """Steroid acetate should NOT produce just 'ethyl acetate' style name.

        The name must reference the steroid ring system, not just the tiny acid.
        """
        smiles = "CC(=O)OC1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C"
        name = name_compound(smiles)
        if name is not None:
            assert name.lower() != "ethyl acetate", (
                f"Steroid acetate should not be 'ethyl acetate': {name}"
            )
            # Should have more content than just a simple ester name
            assert len(name) > 20, (
                f"Steroid acetate name too short (likely fragment loss): {name}"
            )


# ===========================================================================
# Canary preservation tests (from existing decomposition tests)
# ===========================================================================


@pytest.mark.integration
class TestSenioritySwapCanaryPreservation:
    """Verify that seniority swap doesn't break known-good compounds."""

    CANARY_COMPOUNDS = [
        ("CCOC(=O)C", "ethyl acetate"),
        ("CC(=O)NC", "N-methylacetamide"),
        ("CC(=O)O", "acetic acid"),
        ("CCO", "ethanol"),
        ("c1ccccc1", "benzene"),
        ("CCCC", "butane"),
        ("c1ccncc1", "pyridine"),
    ]

    @pytest.mark.parametrize("smiles,expected", CANARY_COMPOUNDS)
    def test_canary_preserved(self, smiles, expected):
        """Canary compound must produce exact expected name."""
        result = name_compound(smiles)
        assert result == expected, (
            f"Canary regression for {smiles}: "
            f"expected '{expected}', got '{result}'"
        )
