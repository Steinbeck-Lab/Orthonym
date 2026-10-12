"""
a phase (Ring Foundation) End-to-End Integration Tests.

Tests validate all a phase success criteria and RING requirements:

Success Criteria:
1. SC1: C1CCCCC1 -> "cyclohexane"
2. SC2: CC1CCCCC1 -> "methylcyclohexane"
3. SC3: c1ccccc1 -> "benzene"
4. SC4: Cc1ccccc1 -> "toluene"
5. SC5: Ring selected as parent when appropriate

RING Requirements:
-: Cycloalkane detection
-: Substituted cycloalkanes
-: Cycloalkene detection
-: Ring vs chain parent selection
-: Benzene retained names
-: Substituted benzene naming
-: Polycyclic aromatic naming
"""

import pytest
from orthonym import name_compound


@pytest.mark.integration
class TestPhase2SuccessCriteria:
    """Tests for a phase success criteria."""

    def test_sc5_ring_over_chain_cyclohexylmethane(self):
        """SC5: When ring is parent, chain is substituent."""
        # methylcyclohexane: cyclohexane ring with methyl substituent
        result = name_compound('CC1CCCCC1')
        assert 'cyclohexane' in result
        assert 'methyl' in result


@pytest.mark.integration
class TestRING03Cycloalkenes:
    """Tests for: Cycloalkene detection and naming."""

    def test_cyclohexa_1_3_diene(self):
        """Cyclohexa-1,3-diene (6-membered ring with 2 double bonds)."""
        result = name_compound('C1=CC=CCC1')
        assert 'cyclohexa' in result
        assert 'diene' in result


@pytest.mark.integration
class TestRING04RingVsChainParentSelection:
    """Tests for: Ring vs chain parent selection."""

    def test_methylcyclohexane_ring_parent(self):
        """Methylcyclohexane: ring is parent, methyl is substituent."""
        result = name_compound('CC1CCCCC1')
        assert 'cyclohexane' in result
        # NOT hexane with a cyclo substituent

    def test_ethylcyclopentane_ring_parent(self):
        """Ethylcyclopentane: ring is parent, ethyl is substituent."""
        result = name_compound('CCC1CCCC1')
        assert 'cyclopentane' in result


@pytest.mark.integration
class TestEdgeCases:
    """Tests for edge cases in ring foundation."""

    def test_cyclopropane_not_propane(self):
        """Cyclopropane should not be named as propane."""
        result = name_compound('C1CC1')
        assert 'cyclo' in result
        assert result != 'propane'

    def test_benzene_not_hexatriene(self):
        """Benzene should be aromatic, not named as hexatriene."""
        result = name_compound('c1ccccc1')
        assert result == 'benzene'
        assert 'hexatriene' not in result

    def test_complex_substitution_dimethyl_ethyl_cyclohexane(self):
        """Multiple different substituents on cyclohexane."""
        # 1-ethyl-3,3-dimethylcyclohexane or similar
        result = name_compound('CCC1CCCC(C)(C)C1')
        assert 'cyclohexane' in result
        assert 'ethyl' in result
        assert 'methyl' in result


