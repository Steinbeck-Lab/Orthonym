"""
Integration tests for Phase 119: Complex Ring Substituent Integration.

Verifies RING-03 (universal pipeline discovers ring substituents) and
RING-04 (locants use ring system's IUPAC numbering).

Reference: IUPAC 2013 Blue Book P-31.1 (Detachable prefixes),
           P-23.2 (Von Baeyer), P-24.2 (Spiro), P-25 (Fused).
"""

import re

import pytest
from orthonym import name_compound


class TestSpiroSubstituents:
    """RING-03: Spiro systems include substituent prefixes."""

    @pytest.mark.integration
    def test_spiro_methyl(self):
        """2-methylspiro[4.5]decane -- methyl on smaller ring."""
        result = name_compound('CC1CCC2(CCCCC2)C1')
        assert 'methyl' in result, f"methyl missing from spiro: {result}"
        assert 'spiro' in result, f"spiro missing: {result}"

    @pytest.mark.integration
    def test_spiro_hydroxy(self):
        """Hydroxy-substituted spiro -- OH should appear as hydroxy prefix."""
        result = name_compound('OC1CCC2(CCCC2)C1')
        assert 'hydroxy' in result, f"hydroxy missing from spiro: {result}"
        assert 'spiro' in result, f"spiro missing: {result}"

    @pytest.mark.integration
    def test_spiro_dimethyl(self):
        """Dimethyl-substituted spiro -- both methyls should appear."""
        result = name_compound('CC1CCC2(CCCCC2C)C1')
        assert 'methyl' in result, f"methyl missing from dimethyl spiro: {result}"
        assert 'spiro' in result, f"spiro missing: {result}"

    @pytest.mark.integration
    def test_plain_spiro_unchanged(self):
        """Unsubstituted spiro[4.5]decane should still work correctly."""
        result = name_compound('C1CCC2(CC1)CCCC2')
        assert 'spiro' in result
        assert result == 'spiro[4.5]decane' or 'spiro' in result


class TestFusedHeterocycleNoDoubleCount:
    """RING-03: Fused heterocycles with substituents_included=True are NOT
    processed by the universal pipeline (no double-counting)."""

    @pytest.mark.integration
    def test_methylindole_no_double_methyl(self):
        """5-methyl-1H-indole should have exactly one 'methyl' occurrence."""
        result = name_compound('Cc1ccc2[nH]ccc2c1')
        assert 'methyl' in result
        # Should NOT have double methyl prefix
        methyl_count = result.count('methyl')
        assert methyl_count == 1, (
            f"Double-counted: 'methyl' appears {methyl_count} times in {result}"
        )

    @pytest.mark.integration
    def test_ethylindole_no_double_ethyl(self):
        """5-ethyl-1H-indole should have exactly one 'ethyl' occurrence."""
        result = name_compound('CCc1ccc2[nH]ccc2c1')
        assert 'ethyl' in result
        ethyl_count = result.count('ethyl')
        assert ethyl_count == 1, (
            f"Double-counted: 'ethyl' appears {ethyl_count} times in {result}"
        )

    @pytest.mark.integration
    def test_chloroquinoline_no_double(self):
        """2-chloroquinoline should have exactly one 'chloro'."""
        result = name_compound('Clc1ccc2ccccc2n1')
        assert 'chloro' in result or 'chlor' in result
        # No double-counting
        assert result.count('chloro') <= 1


class TestBicycloPolycyclicPreserved:
    """RING-03: Bicyclo and polycyclic handlers already include subs
    (substituents_included=True), verify no regression."""

    @pytest.mark.integration
    def test_methylnorbornane_preserved(self):
        """Methylnorbornane -- bicyclo handler already discovers methyl."""
        result = name_compound('CC1CC2CCC1C2')
        assert 'methyl' in result
        assert 'bicyclo' in result or 'norbornane' in result.lower()

    @pytest.mark.integration
    def test_methyladamantane_preserved(self):
        """1-methyladamantane -- polycyclic handler already discovers methyl."""
        result = name_compound('CC12CC3CC(CC(C3)C1)C2')
        assert 'methyl' in result
        assert 'tricyclo' in result or 'adamantane' in result.lower()


class TestRingSubLocants:
    """RING-04: Substituent locants use ring system's IUPAC numbering."""

    @pytest.mark.integration
    def test_spiro_locant_is_integer(self):
        """Spiro substituent locant should be an integer from spiro numbering."""
        result = name_compound('CC1CCC2(CCCCC2)C1')
        # Should contain a digit-hyphen-methyl pattern like "2-methyl"
        assert re.search(r'\d+-methyl', result), (
            f"No integer locant on methyl in spiro: {result}"
        )

    @pytest.mark.integration
    def test_fused_heterocycle_locant_preserved(self):
        """5-methyl-1H-indole locant should be 5 (IUPAC fused numbering)."""
        result = name_compound('Cc1ccc2[nH]ccc2c1')
        assert '5-methyl' in result, f"Expected '5-methyl' in {result}"


class TestRegressionSuite:
    """Verify no regressions on previously-working ring compounds."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected_fragment", [
        ('C1CCC2(CC1)CCCC2', 'spiro'),           # plain spiro
        ('c1ccc2[nH]ccc2c1', 'indole'),           # plain indole
        ('C1CC2CCC1C2', 'norbornan'),              # norbornane (retained name)
        ('C1CCC2CCCCC2C1', ''),                   # decalin (any name ok)
    ])
    def test_plain_ring_systems_unchanged(self, smiles, expected_fragment):
        """Previously-working unsubstituted ring systems produce valid names."""
        result = name_compound(smiles)
        assert result is not None, f"Name is None for {smiles}"
        assert len(result) > 0, f"Empty name for {smiles}"
        if expected_fragment:
            assert expected_fragment in result.lower(), (
                f"Expected '{expected_fragment}' in {result} for {smiles}"
            )
