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
class TestEdgeCases:
    """Tests for edge cases in ring foundation."""

    def test_cyclopropane_not_propane(self):
        """Cyclopropane should not be named as propane."""
        result = name_compound('C1CC1')
        assert 'cyclo' in result
        assert result != 'propane'

    def test_complex_substitution_dimethyl_ethyl_cyclohexane(self):
        """Multiple different substituents on cyclohexane."""
        # 1-ethyl-3,3-dimethylcyclohexane or similar
        result = name_compound('CCC1CCCC(C)(C)C1')
        assert 'cyclohexane' in result
        assert 'ethyl' in result
        assert 'methyl' in result
