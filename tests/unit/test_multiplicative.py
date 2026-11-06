"""Tests for multiplicative nomenclature (IUPAC.

Multiplicative naming applies to molecules with two or more identical parent
structures connected by a polyvalent linking group (bridge). For example,
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
# 3+ unit multiplicative tests -- star-topology bridges
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMultiplicativeTriUnit:
    """3-unit multiplicative naming with star-topology bridges."""

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
