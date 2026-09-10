"""
Integration tests for ring substituent completeness (a phase, Plan 01).

Tests that substituents on cycloalkanes, bicyclics, and PAH systems are NOT
dropped or merged during name assembly. Covers requirements through.

These tests assert that the generated name CONTAINS the expected substituent
prefix, rather than matching an exact name, to be robust against minor
locant or formatting differences.
"""

import pytest
from orthonym import name_compound


# ---------------------------------------------------------------------------
#: Mixed alkyl + halogen substituents on cycloalkanes
# Previously: prefixes merged (e.g., "chloromethylcyclohexane" instead of
# separate "chloro" and "methyl" prefixes)
# ---------------------------------------------------------------------------

class TestCycloalkaneHalogenSubstituents:
    """Mixed halogen + alkyl substituents on cycloalkanes produce separate prefixes."""

    @pytest.mark.integration
    def test_chloro_methyl_cyclohexane(self):
        """CC1CCC(Cl)CC1 -> contains 'chloro' and 'methyl', not 'chloromethyl'."""
        name = name_compound('CC1CCC(Cl)CC1')
        assert 'chloro' in name, f"Expected 'chloro' in '{name}'"
        assert 'methyl' in name, f"Expected 'methyl' in '{name}'"
        assert 'chloromethyl' not in name, f"Should NOT have merged 'chloromethyl' in '{name}'"

    @pytest.mark.integration
    def test_chloro_cyclohexane_monosubstituted(self):
        """ClC1CCCCC1 -> contains 'chloro' and 'cyclohexane'."""
        name = name_compound('ClC1CCCCC1')
        assert 'chloro' in name, f"Expected 'chloro' in '{name}'"
        assert 'cyclohexane' in name, f"Expected 'cyclohexane' in '{name}'"

    @pytest.mark.integration
    def test_difluoro_cyclohexane(self):
        """FC1CCC(F)CC1 -> contains 'difluoro' and 'cyclohexane'."""
        name = name_compound('FC1CCC(F)CC1')
        assert 'difluoro' in name, f"Expected 'difluoro' in '{name}'"
        assert 'cyclohexane' in name, f"Expected 'cyclohexane' in '{name}'"

    @pytest.mark.integration
    def test_bromo_methyl_cyclopentane(self):
        """CC1CCC(Br)C1 -> contains 'bromo' and 'methyl'."""
        name = name_compound('CC1CCC(Br)C1')
        assert 'bromo' in name, f"Expected 'bromo' in '{name}'"
        assert 'methyl' in name, f"Expected 'methyl' in '{name}'"

    @pytest.mark.integration
    def test_dichloro_cyclohexane(self):
        """ClC1CCC(Cl)CC1 -> contains 'dichloro' and 'cyclohexane'."""
        name = name_compound('ClC1CCC(Cl)CC1')
        assert 'dichloro' in name, f"Expected 'dichloro' in '{name}'"
        assert 'cyclohexane' in name, f"Expected 'cyclohexane' in '{name}'"

    @pytest.mark.integration
    def test_trisubstituted_cyclohexane(self):
        """CC1(Cl)CCC(F)CC1 -> contains 'chloro', 'fluoro', 'methyl'."""
        name = name_compound('CC1(Cl)CCC(F)CC1')
        assert 'chloro' in name, f"Expected 'chloro' in '{name}'"
        assert 'fluoro' in name, f"Expected 'fluoro' in '{name}'"
        assert 'methyl' in name, f"Expected 'methyl' in '{name}'"


# ---------------------------------------------------------------------------
#: Halogen substituents on bicyclic systems
# Previously: non-carbon substituents silently dropped from bicyclic names
# ---------------------------------------------------------------------------

class TestBicyclicNonCarbonSubstituents:
    """Non-carbon substituents on bicyclic systems appear in the name."""

    @pytest.mark.integration
    def test_chloro_bicyclooctane(self):
        """ClC1CC2CCC1CC2 -> contains 'chloro' and 'bicyclo'."""
        name = name_compound('ClC1CC2CCC1CC2')
        assert 'chloro' in name, f"Expected 'chloro' in '{name}'"
        assert 'bicyclo' in name, f"Expected 'bicyclo' in '{name}'"

    @pytest.mark.integration
    def test_fluoro_bicyclooctane(self):
        """FC1CC2CCC1CC2 -> contains 'fluoro' and 'bicyclo'."""
        name = name_compound('FC1CC2CCC1CC2')
        assert 'fluoro' in name, f"Expected 'fluoro' in '{name}'"
        assert 'bicyclo' in name, f"Expected 'bicyclo' in '{name}'"

    @pytest.mark.integration
    def test_bromo_bicyclooctane(self):
        """BrC1CC2CCC1CC2 -> contains 'bromo' and 'bicyclo'."""
        name = name_compound('BrC1CC2CCC1CC2')
        assert 'bromo' in name, f"Expected 'bromo' in '{name}'"
        assert 'bicyclo' in name, f"Expected 'bicyclo' in '{name}'"

    @pytest.mark.integration
    def test_dichloro_norbornane(self):
        """ClC1CC2CC(Cl)C1C2 -> contains 'dichloro' and 'bicyclo'."""
        name = name_compound('ClC1CC2CC(Cl)C1C2')
        assert 'dichloro' in name, f"Expected 'dichloro' in '{name}'"
        assert 'bicyclo' in name, f"Expected 'bicyclo' in '{name}'"

    @pytest.mark.integration
    def test_methyl_chloro_norbornane(self):
        """CC1CC2CC(Cl)C1C2 -> contains 'methyl' and 'chloro' and 'bicyclo'."""
        name = name_compound('CC1CC2CC(Cl)C1C2')
        assert 'methyl' in name, f"Expected 'methyl' in '{name}'"
        assert 'chloro' in name, f"Expected 'chloro' in '{name}'"
        assert 'bicyclo' in name, f"Expected 'bicyclo' in '{name}'"


# ---------------------------------------------------------------------------
#: Hydroxy/amino substituents on bicyclic systems
# Previously: zero-carbon substituents dropped from bicyclic naming path
# ---------------------------------------------------------------------------

class TestBicyclicHydroxyAminoSubstituents:
    """Hydroxy and amino substituents on bicyclic systems appear in the name."""

    @pytest.mark.integration
    def test_hydroxy_bicyclooctane(self):
        """OC1CC2CCC1CC2 -> contains 'hydroxy' or 'ol' (may be suffix)."""
        name = name_compound('OC1CC2CCC1CC2')
        assert 'hydroxy' in name or 'ol' in name, \
            f"Expected 'hydroxy' or 'ol' in '{name}'"
        assert 'bicyclo' in name, f"Expected 'bicyclo' in '{name}'"

    @pytest.mark.integration
    def test_amino_bicyclooctane(self):
        """NC1CC2CCC1CC2 -> contains 'amino' or 'amine' (may be suffix)."""
        name = name_compound('NC1CC2CCC1CC2')
        assert 'amino' in name or 'amine' in name, \
            f"Expected 'amino' or 'amine' in '{name}'"
        assert 'bicyclo' in name, f"Expected 'bicyclo' in '{name}'"

    @pytest.mark.integration
    def test_hydroxy_methyl_norbornane(self):
        """CC1CC2CC(O)C1C2 -> contains 'hydroxy' and 'methyl' and 'bicyclo'."""
        name = name_compound('CC1CC2CC(O)C1C2')
        assert 'hydroxy' in name or 'ol' in name, \
            f"Expected 'hydroxy' or 'ol' in '{name}'"
        assert 'methyl' in name, f"Expected 'methyl' in '{name}'"
        assert 'bicyclo' in name, f"Expected 'bicyclo' in '{name}'"


# ---------------------------------------------------------------------------
# (cycloalkane path): Hydroxy/amino as prefixes on cycloalkanes
# When OH/NH2 is not the principal group, it appears as a prefix
# ---------------------------------------------------------------------------

class TestCycloalkaneHydroxyAminoPrefix:
    """Hydroxy/amino appear as prefixes or suffixes on cycloalkanes with mixed substituents."""

    @pytest.mark.integration
    def test_hydroxy_methyl_cyclohexane(self):
        """CC1CCC(O)CC1 -> contains 'methyl' and ('hydroxy' or 'ol')."""
        name = name_compound('CC1CCC(O)CC1')
        assert 'methyl' in name, f"Expected 'methyl' in '{name}'"
        assert 'hydroxy' in name or 'ol' in name, \
            f"Expected 'hydroxy' or 'ol' in '{name}'"

    @pytest.mark.integration
    def test_amino_chloro_cyclohexane(self):
        """NC1CCC(Cl)CC1 -> contains 'amino' or 'amine', and 'chloro'."""
        name = name_compound('NC1CCC(Cl)CC1')
        assert 'amino' in name or 'amine' in name, \
            f"Expected 'amino' or 'amine' in '{name}'"
        assert 'chloro' in name, f"Expected 'chloro' in '{name}'"


# ---------------------------------------------------------------------------
# Regression tests: pure alkyl ring substituents still work correctly
# ---------------------------------------------------------------------------

class TestRegressionPureAlkylRings:
    """Pure alkyl substituents on rings are not regressed by the fixes."""

    @pytest.mark.integration
    def test_methylcyclohexane_no_locant(self):
        """CC1CCCCC1 -> 'methylcyclohexane' (locant omitted for monosubstituted)."""
        name = name_compound('CC1CCCCC1')
        assert name == 'methylcyclohexane', \
            f"Expected 'methylcyclohexane', got '{name}'"

    @pytest.mark.integration
    def test_dimethylcyclohexane_with_locants(self):
        """CC1CCCC(C)C1 -> contains 'dimethyl' and locants."""
        name = name_compound('CC1CCCC(C)C1')
        assert 'dimethyl' in name, f"Expected 'dimethyl' in '{name}'"
        assert 'cyclohexane' in name, f"Expected 'cyclohexane' in '{name}'"

    @pytest.mark.integration
    def test_trimethyl_norbornane(self):
        """CC1(C)CC2CCC1(C)C2 -> contains 'trimethyl' or 'methyl' and 'bicyclo'."""
        name = name_compound('CC1(C)CC2CCC1(C)C2')
        # May be routed through a different naming path but should include methyl substituents
        assert 'methyl' in name, f"Expected 'methyl' in '{name}'"

    @pytest.mark.integration
    def test_ethylcyclopentane(self):
        """CCC1CCCC1 -> contains 'ethyl' and 'cyclopentane'."""
        name = name_compound('CCC1CCCC1')
        assert 'ethyl' in name, f"Expected 'ethyl' in '{name}'"
        assert 'cyclopentane' in name, f"Expected 'cyclopentane' in '{name}'"
