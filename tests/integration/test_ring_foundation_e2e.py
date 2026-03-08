"""
Phase 2 (Ring Foundation) End-to-End Integration Tests.

Tests validate all Phase 2 success criteria and RING requirements:

Success Criteria:
1. SC1: C1CCCCC1 -> "cyclohexane"
2. SC2: CC1CCCCC1 -> "methylcyclohexane"
3. SC3: c1ccccc1 -> "benzene"
4. SC4: Cc1ccccc1 -> "toluene"
5. SC5: Ring selected as parent when appropriate

RING Requirements:
- RING-01: Cycloalkane detection
- RING-02: Substituted cycloalkanes
- RING-03: Cycloalkene detection
- RING-04: Ring vs chain parent selection
- RING-05: Benzene retained names
- RING-06: Substituted benzene naming
- RING-07: Polycyclic aromatic naming
"""

import pytest
from orthonym import name_compound


@pytest.mark.integration
class TestPhase2SuccessCriteria:
    """Tests for Phase 2 success criteria."""

    def test_sc1_cyclohexane(self):
        """SC1: C1CCCCC1 -> cyclohexane"""
        assert name_compound('C1CCCCC1') == 'cyclohexane'

    def test_sc2_methylcyclohexane(self):
        """SC2: CC1CCCCC1 -> methylcyclohexane"""
        assert name_compound('CC1CCCCC1') == 'methylcyclohexane'

    def test_sc3_benzene(self):
        """SC3: c1ccccc1 -> benzene"""
        assert name_compound('c1ccccc1') == 'benzene'

    def test_sc4_toluene(self):
        """SC4: Cc1ccccc1 -> toluene (retained name)"""
        result = name_compound('Cc1ccccc1')
        # Toluene is a retained name for methylbenzene
        assert result == 'toluene'

    def test_sc5_ring_over_chain_cyclohexylmethane(self):
        """SC5: When ring is parent, chain is substituent."""
        # methylcyclohexane: cyclohexane ring with methyl substituent
        result = name_compound('CC1CCCCC1')
        assert 'cyclohexane' in result
        assert 'methyl' in result


@pytest.mark.integration
class TestRING01Cycloalkanes:
    """Tests for RING-01: Cycloalkane detection and naming."""

    def test_cyclopropane(self):
        """Cyclopropane (3-membered ring)."""
        assert name_compound('C1CC1') == 'cyclopropane'

    def test_cyclobutane(self):
        """Cyclobutane (4-membered ring)."""
        assert name_compound('C1CCC1') == 'cyclobutane'

    def test_cyclopentane(self):
        """Cyclopentane (5-membered ring)."""
        assert name_compound('C1CCCC1') == 'cyclopentane'

    def test_cyclohexane(self):
        """Cyclohexane (6-membered ring)."""
        assert name_compound('C1CCCCC1') == 'cyclohexane'

    def test_cycloheptane(self):
        """Cycloheptane (7-membered ring)."""
        assert name_compound('C1CCCCCC1') == 'cycloheptane'

    def test_cyclooctane(self):
        """Cyclooctane (8-membered ring)."""
        assert name_compound('C1CCCCCCC1') == 'cyclooctane'


@pytest.mark.integration
class TestRING02SubstitutedCycloalkanes:
    """Tests for RING-02: Substituted cycloalkane naming."""

    def test_methylcyclopentane(self):
        """Methylcyclopentane: monosubstituted (no locant)."""
        result = name_compound('CC1CCCC1')
        assert result == 'methylcyclopentane'

    def test_methylcyclohexane(self):
        """Methylcyclohexane: monosubstituted (no locant)."""
        result = name_compound('CC1CCCCC1')
        assert result == 'methylcyclohexane'

    def test_ethylcyclohexane(self):
        """Ethylcyclohexane: monosubstituted (no locant)."""
        result = name_compound('CCC1CCCCC1')
        assert result == 'ethylcyclohexane'

    def test_1_2_dimethylcyclohexane(self):
        """1,2-dimethylcyclohexane: disubstituted (locants required)."""
        result = name_compound('CC1CCCCC1C')
        assert '1,2-dimethylcyclohexane' == result

    def test_1_3_dimethylcyclohexane(self):
        """1,3-dimethylcyclohexane: disubstituted."""
        result = name_compound('CC1CCCC(C)C1')
        assert '1,3-dimethylcyclohexane' == result

    def test_1_4_dimethylcyclohexane(self):
        """1,4-dimethylcyclohexane: disubstituted."""
        result = name_compound('CC1CCC(C)CC1')
        assert '1,4-dimethylcyclohexane' == result


@pytest.mark.integration
class TestRING03Cycloalkenes:
    """Tests for RING-03: Cycloalkene detection and naming."""

    def test_cyclopropene(self):
        """Cyclopropene (3-membered ring with double bond)."""
        result = name_compound('C1=CC1')
        assert result == 'cyclopropene'

    def test_cyclobutene(self):
        """Cyclobutene (4-membered ring with double bond)."""
        result = name_compound('C1=CCC1')
        assert result == 'cyclobutene'

    def test_cyclopentene(self):
        """Cyclopentene (5-membered ring with double bond)."""
        result = name_compound('C1=CCCC1')
        assert result == 'cyclopentene'

    def test_cyclohexene(self):
        """Cyclohexene (6-membered ring with double bond)."""
        result = name_compound('C1=CCCCC1')
        assert result == 'cyclohexene'

    def test_cycloheptene(self):
        """Cycloheptene (7-membered ring with double bond)."""
        result = name_compound('C1=CCCCCC1')
        assert result == 'cycloheptene'

    def test_cyclohexa_1_3_diene(self):
        """Cyclohexa-1,3-diene (6-membered ring with 2 double bonds)."""
        result = name_compound('C1=CC=CCC1')
        assert 'cyclohexa' in result
        assert 'diene' in result


@pytest.mark.integration
class TestRING04RingVsChainParentSelection:
    """Tests for RING-04: Ring vs chain parent selection."""

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
class TestRING05BenzeneRetainedNames:
    """Tests for RING-05: Benzene derivatives with retained names."""

    def test_benzene(self):
        """Benzene retained name."""
        assert name_compound('c1ccccc1') == 'benzene'

    def test_toluene(self):
        """Toluene (methylbenzene) retained name."""
        assert name_compound('Cc1ccccc1') == 'toluene'

    def test_phenol(self):
        """Phenol (hydroxybenzene) retained name."""
        assert name_compound('Oc1ccccc1') == 'phenol'

    def test_aniline(self):
        """Aniline (aminobenzene) retained name."""
        assert name_compound('Nc1ccccc1') == 'aniline'

    def test_styrene(self):
        """Styrene (ethenylbenzene) retained name."""
        assert name_compound('C=Cc1ccccc1') == 'styrene'

    def test_cumene(self):
        """Cumene (isopropylbenzene) retained name."""
        assert name_compound('CC(C)c1ccccc1') == 'cumene'


@pytest.mark.integration
class TestRING06SubstitutedBenzenes:
    """Tests for RING-06: Substituted benzene naming with locants."""

    def test_chlorobenzene(self):
        """Chlorobenzene: monosubstituted (no locant)."""
        result = name_compound('Clc1ccccc1')
        assert result == 'chlorobenzene'

    def test_1_2_dimethylbenzene(self):
        """1,2-dimethylbenzene (o-xylene) systematic name."""
        result = name_compound('Cc1ccccc1C')
        assert result == '1,2-dimethylbenzene'

    def test_1_3_dimethylbenzene(self):
        """1,3-dimethylbenzene (m-xylene) systematic name."""
        result = name_compound('Cc1cccc(C)c1')
        assert result == '1,3-dimethylbenzene'

    def test_1_4_dimethylbenzene(self):
        """1,4-dimethylbenzene (p-xylene) systematic name."""
        result = name_compound('Cc1ccc(C)cc1')
        assert result == '1,4-dimethylbenzene'

    def test_1_chloro_4_methylbenzene(self):
        """1-chloro-4-methylbenzene: alphabetical ordering."""
        result = name_compound('Cc1ccc(Cl)cc1')
        assert result == '1-chloro-4-methylbenzene'


@pytest.mark.integration
class TestRING07PolycyclicAromatics:
    """Tests for RING-07: Polycyclic aromatic naming."""

    def test_naphthalene(self):
        """Naphthalene retained name."""
        assert name_compound('c1ccc2ccccc2c1') == 'naphthalene'

    def test_anthracene(self):
        """Anthracene retained name."""
        assert name_compound('c1ccc2cc3ccccc3cc2c1') == 'anthracene'

    def test_phenanthrene(self):
        """Phenanthrene retained name."""
        assert name_compound('c1ccc2c(c1)ccc1ccccc12') == 'phenanthrene'

    def test_1_methylnaphthalene(self):
        """1-methylnaphthalene: alpha position."""
        result = name_compound('Cc1cccc2ccccc12')
        assert result == '1-methylnaphthalene'

    def test_2_methylnaphthalene(self):
        """2-methylnaphthalene: beta position."""
        result = name_compound('Cc1ccc2ccccc2c1')
        assert result == '2-methylnaphthalene'

    def test_1_chloronaphthalene(self):
        """1-chloronaphthalene: alpha position."""
        result = name_compound('Clc1cccc2ccccc12')
        assert result == '1-chloronaphthalene'

    def test_2_chloronaphthalene(self):
        """2-chloronaphthalene: beta position."""
        result = name_compound('Clc1ccc2ccccc2c1')
        assert result == '2-chloronaphthalene'


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

    def test_cyclodecane(self):
        """Cyclodecane (10-membered ring)."""
        result = name_compound('C1CCCCCCCCC1')
        assert 'cyclodecane' == result

    def test_complex_substitution_dimethyl_ethyl_cyclohexane(self):
        """Multiple different substituents on cyclohexane."""
        # 1-ethyl-3,3-dimethylcyclohexane or similar
        result = name_compound('CCC1CCCC(C)(C)C1')
        assert 'cyclohexane' in result
        assert 'ethyl' in result
        assert 'methyl' in result


@pytest.mark.integration
class TestPhase1Regression:
    """Regression tests ensuring Phase 1 functionality still works."""

    def test_simple_alkanes_still_work(self):
        """Simple alkanes should still be named correctly."""
        assert name_compound('CC') == 'ethane'
        assert name_compound('CCC') == 'propane'
        assert name_compound('CCCC') == 'butane'

    def test_branched_alkanes_still_work(self):
        """Branched alkanes should still be named correctly."""
        result = name_compound('CC(C)C')
        assert '2-methylpropane' == result

    def test_functional_groups_still_work(self):
        """Functional group compounds should still be named correctly."""
        assert name_compound('CO') == 'methanol'
        assert name_compound('CCO') == 'ethanol'
        assert name_compound('CC(=O)O') == 'acetic acid'

    def test_unsaturated_still_work(self):
        """Unsaturated compounds should still be named correctly."""
        assert name_compound('C=C') == 'ethene'
        assert name_compound('C#C') == 'acetylene'  # retained name (P-31.1.2.1 PIN)
        # IUPAC PIN style uses locants: prop-1-ene not propene
        assert name_compound('CC=C') == 'prop-1-ene'
