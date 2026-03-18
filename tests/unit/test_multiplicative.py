"""Tests for multiplicative nomenclature (IUPAC P-51.3).

Multiplicative naming applies to molecules with two or more identical parent
structures connected by a polyvalent linking group (bridge).  For example,
4,4'-methylenedianiline has two identical aniline units linked by a CH2 bridge.

RED phase: these tests define the expected behavior before implementation.
"""

import pytest
from orthonym import name_compound


# ---------------------------------------------------------------------------
# Core multiplicative tests -- these should FAIL until implementation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMultiplicativeMethyleneBridge:
    """CH2 (methylene) bridge linking two identical ring systems."""

    def test_methylenedianiline_preserves_both_halves(self):
        """4,4'-Methylenedianiline: Nc1ccc(Cc2ccc(N)cc2)cc1
        Both amino groups must be represented in the name.
        Must NOT produce 'aminobenzene' (current wrong output that drops half).
        """
        name = name_compound("Nc1ccc(Cc2ccc(N)cc2)cc1")
        # The name should reference both amine groups
        assert name != "aminobenzene", (
            "Name must not drop half the molecule"
        )
        # Should contain 'di' multiplier indicating two identical units
        has_di_prefix = ("dianiline" in name or "diamine" in name
                         or "diamino" in name or "di" in name.lower())
        assert has_di_prefix, (
            f"Expected multiplicative name with 'di' prefix, got: {name}"
        )

    def test_methylenedianiline_contains_methylene(self):
        """The bridge name 'methylene' should appear."""
        name = name_compound("Nc1ccc(Cc2ccc(N)cc2)cc1")
        assert "methylene" in name.lower(), (
            f"Expected 'methylene' bridge name, got: {name}"
        )


@pytest.mark.unit
class TestMultiplicativeOxyBridge:
    """O (oxy) bridge linking two identical ring systems."""

    def test_oxydibenzoic_acid_preserves_both_halves(self):
        """4,4'-Oxydibenzoic acid: OC(=O)c1ccc(Oc2ccc(C(=O)O)cc2)cc1
        Both carboxylic acid groups must be preserved.
        """
        name = name_compound("OC(=O)c1ccc(Oc2ccc(C(=O)O)cc2)cc1")
        # Name should reference both acid groups
        has_both = ("dibenzoic" in name or "dicarboxylic" in name
                    or "diacid" in name
                    or name.count("acid") >= 1 and "di" in name.lower())
        assert has_both, (
            f"Expected name preserving both acid groups, got: {name}"
        )

    def test_oxydibenzoic_acid_contains_oxy_bridge(self):
        """The bridge name 'oxy' should appear."""
        name = name_compound("OC(=O)c1ccc(Oc2ccc(C(=O)O)cc2)cc1")
        assert "oxy" in name.lower(), (
            f"Expected 'oxy' bridge name, got: {name}"
        )


@pytest.mark.unit
class TestMultiplicativeEthyleneBridge:
    """CH2CH2 (ethylene) bridge linking two identical ring systems."""

    def test_ethylenediphenol_preserves_both_halves(self):
        """4,4'-Ethylenediphenol: Oc1ccc(CCc2ccc(O)cc2)cc1
        Both hydroxyl groups must be preserved.
        """
        name = name_compound("Oc1ccc(CCc2ccc(O)cc2)cc1")
        # Should reference both OH groups with a multiplicative prefix
        has_both = ("diphenol" in name or "ethylenedi" in name.lower()
                    or name.count("hydroxy") >= 2)
        assert has_both, (
            f"Expected name preserving both hydroxyl groups, got: {name}"
        )

    def test_ethylenediphenol_contains_ethylene(self):
        """The bridge name 'ethylene' or 'ethane-1,2-diyl' should appear."""
        name = name_compound("Oc1ccc(CCc2ccc(O)cc2)cc1")
        has_bridge = ("ethylene" in name.lower()
                      or "ethane" in name.lower())
        assert has_bridge, (
            f"Expected 'ethylene' bridge name, got: {name}"
        )


@pytest.mark.unit
class TestMultiplicativeIminoBridge:
    """NH (imino) bridge linking two identical ring systems."""

    def test_iminodianiline_preserves_both_halves(self):
        """Nc1ccc(Nc2ccc(N)cc2)cc1 -- NH-linked dianiline.
        Both terminal amino groups must be preserved.
        """
        name = name_compound("Nc1ccc(Nc2ccc(N)cc2)cc1")
        assert name != "aminobenzene", (
            "Name must not drop half the molecule"
        )
        has_multi = ("dianiline" in name or "diamine" in name
                     or "diamino" in name or "di" in name.lower())
        assert has_multi, (
            f"Expected multiplicative name, got: {name}"
        )


# ---------------------------------------------------------------------------
# Negative tests -- these should PASS even before implementation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMultiplicativeNegative:
    """Molecules that must NOT trigger multiplicative naming."""

    def test_simple_benzene_not_affected(self):
        """Benzene should still be 'benzene'."""
        assert name_compound("c1ccccc1") == "benzene"

    def test_toluene_not_affected(self):
        """Toluene should still be 'toluene'."""
        assert name_compound("Cc1ccccc1") == "toluene"

    def test_biphenyl_not_multiplicative(self):
        """Biphenyl has a direct ring-ring bond, no bridge atom.
        It should NOT be treated as multiplicative.
        """
        name = name_compound("c1ccc(-c2ccccc2)cc1")
        # biphenyl may name in various ways, but multiplicative should not fire
        # (no single bridge atom between the rings)
        assert "methylene" not in name.lower()
        assert "oxy" not in name.lower()

    def test_simple_aniline_not_multiplicative(self):
        """Single aniline should be 'aniline' or '4-aminobenzene', not multiplicative."""
        name = name_compound("Nc1ccccc1")
        # Should not contain 'di' multiplier
        assert "dianiline" not in name.lower()
        assert "diamine" not in name.lower()


# ---------------------------------------------------------------------------
# 3+ unit multiplicative tests -- star-topology bridges
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMultiplicativeTriUnit:
    """3-unit multiplicative naming with star-topology bridges."""

    def test_nitrilotriphenol(self):
        """N bridge connecting 3 phenol rings -> 4,4',4''-nitrilotriphenol"""
        name = name_compound("Oc1ccc(N(c2ccc(O)cc2)c2ccc(O)cc2)cc1")
        assert name is not None
        assert "nitrilo" in name.lower()
        assert "triphenol" in name.lower()

    def test_nitrilotrianiline(self):
        """N bridge connecting 3 aniline rings -> 4,4',4''-nitrilotrianiline"""
        name = name_compound("Nc1ccc(N(c2ccc(N)cc2)c2ccc(N)cc2)cc1")
        assert name is not None
        assert "nitrilo" in name.lower()
        assert "trianiline" in name.lower()

    def test_nitrilotribenzoic_acid(self):
        """N bridge connecting 3 benzoic acid rings -> 4,4',4''-nitrilotribenzoic acid"""
        name = name_compound("OC(=O)c1ccc(N(c2ccc(C(=O)O)cc2)c2ccc(C(=O)O)cc2)cc1")
        assert name is not None
        assert "nitrilo" in name.lower()
        assert "tribenzoic" in name.lower()

    def test_methylidynetriphenol(self):
        """CH bridge connecting 3 phenol rings -> 4,4',4''-methylidynetriphenol"""
        name = name_compound("Oc1ccc(C(c2ccc(O)cc2)c2ccc(O)cc2)cc1")
        assert name is not None
        assert "methylidyne" in name.lower()
        assert "triphenol" in name.lower()

    def test_methylidynetribenzoic_acid(self):
        """CH bridge connecting 3 benzoic acid rings"""
        name = name_compound("OC(=O)c1ccc(C(c2ccc(C(=O)O)cc2)c2ccc(C(=O)O)cc2)cc1")
        assert name is not None
        assert "methylidyne" in name.lower()
        assert "tribenzoic" in name.lower()


@pytest.mark.unit
class TestMultiplicativeTetraUnit:
    """4-unit multiplicative naming."""

    def test_methanetetrayltetraphenol(self):
        """C bridge connecting 4 phenol rings -> 4,4',4'',4'''-methanetetrayltetraphenol"""
        name = name_compound("Oc1ccc(C(c2ccc(O)cc2)(c2ccc(O)cc2)c2ccc(O)cc2)cc1")
        assert name is not None
        assert "methanetetrayl" in name.lower()
        assert "tetraphenol" in name.lower()

    def test_methanetetrayltetrabenzoic_acid(self):
        """C bridge connecting 4 benzoic acid rings"""
        name = name_compound("OC(=O)c1ccc(C(c2ccc(C(=O)O)cc2)(c2ccc(C(=O)O)cc2)c2ccc(C(=O)O)cc2)cc1")
        assert name is not None
        assert "methanetetrayl" in name.lower()
        assert "tetrabenzoic" in name.lower()


@pytest.mark.unit
class TestMultiplicativePrimedLocants:
    """Test primed locant string formatting."""

    def test_tri_unit_primed_locants(self):
        """3-unit compound should have 4,4',4'' locant pattern."""
        name = name_compound("Oc1ccc(N(c2ccc(O)cc2)c2ccc(O)cc2)cc1")
        assert name is not None
        # Check for primed locant pattern
        assert "4'" in name or "4," in name

    def test_tetra_unit_primed_locants(self):
        """4-unit compound should have 4,4',4'',4''' locant pattern."""
        name = name_compound("Oc1ccc(C(c2ccc(O)cc2)(c2ccc(O)cc2)c2ccc(O)cc2)cc1")
        assert name is not None
        assert "'''" in name  # Triple prime for 4th unit


@pytest.mark.unit
class TestMultiplicativeMultiNegative:
    """Cases that must NOT trigger multi-bridge naming."""

    def test_non_identical_fragments_rejected(self):
        """Mixed substituents on ring fragments should not match."""
        # N connecting phenol + aniline + benzoic acid = non-identical fragments
        name = name_compound("Oc1ccc(N(c2ccc(N)cc2)c2ccc(C(=O)O)cc2)cc1")
        assert name is not None
        assert "nitrilo" not in name.lower()

    def test_two_unit_still_uses_di(self):
        """2-unit compounds must still use 'di' prefix (existing path)."""
        name = name_compound("Nc1ccc(Cc2ccc(N)cc2)cc1")
        assert "di" in name.lower()
        assert "tri" not in name.lower()
