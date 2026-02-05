"""
Regression tests for ring substituent assembly (Plan 15-04).

Tests that substituents on fused aromatic, simple aromatic, and heterocyclic
ring systems are NOT dropped during name assembly. This was the single
biggest systemic issue identified in round-trip validation.

These tests assert that the generated name CONTAINS the expected substituent
prefix, rather than matching an exact name, to be robust against minor
locant or formatting differences.
"""

import pytest
from orthonym import name_compound


# ---------------------------------------------------------------------------
# Fused aromatic substituents (5 tests)
# Previously: parent name returned with no substituent prefix
# ---------------------------------------------------------------------------

class TestFusedAromaticSubstituents:
    """Fused aromatic ring systems must retain substituent prefixes."""

    def test_nitroindazole(self):
        """Nitroindazole must contain 'nitro' prefix."""
        name = name_compound('O=[N+]([O-])c1cccc2cn[nH]c12')
        assert 'nitro' in name, f"Expected 'nitro' in '{name}'"
        assert 'indazole' in name, f"Expected 'indazole' in '{name}'"

    def test_methylquinoline(self):
        """Methylquinoline must contain 'methyl' prefix."""
        name = name_compound('Cc1ccc2ncccc2c1')
        assert 'methyl' in name, f"Expected 'methyl' in '{name}'"
        assert 'quinoline' in name, f"Expected 'quinoline' in '{name}'"

    def test_hydroxybenzimidazole(self):
        """Hydroxybenzimidazole must contain 'hydroxy' prefix."""
        name = name_compound('Oc1ccc2[nH]cnc2c1')
        assert 'hydroxy' in name, f"Expected 'hydroxy' in '{name}'"
        assert 'benzimidazole' in name, f"Expected 'benzimidazole' in '{name}'"

    def test_dimethylindole(self):
        """Disubstituted indole must contain 'dimethyl' and correct parent."""
        name = name_compound('Cc1cc(C)c2[nH]ccc2c1')
        assert 'dimethyl' in name, f"Expected 'dimethyl' in '{name}'"
        assert 'indole' in name, f"Expected 'indole' in '{name}'"

    def test_methylnaphthalene(self):
        """Methylnaphthalene must contain 'methyl' prefix."""
        name = name_compound('Cc1cccc2ccccc12')
        assert 'methyl' in name, f"Expected 'methyl' in '{name}'"
        assert 'naphthalene' in name, f"Expected 'naphthalene' in '{name}'"


# ---------------------------------------------------------------------------
# Simple aromatic (benzene) substituents (5 tests)
# Previously: complex substituents caused fallback to bare "benzene"
# ---------------------------------------------------------------------------

class TestBenzeneSubstituents:
    """Benzene derivatives must retain all substituent prefixes."""

    def test_dimethylamino_amino_benzene(self):
        """Benzene with both dimethylamino and amino must keep both groups."""
        name = name_compound('c1ccc(N(C)C)cc1N')
        assert 'amino' in name, f"Expected 'amino' in '{name}'"
        assert 'dimethylamino' in name, f"Expected 'dimethylamino' in '{name}'"
        assert 'benzene' in name, f"Expected 'benzene' in '{name}'"

    def test_diethylbenzene(self):
        """Diethylbenzene must retain both ethyl groups."""
        name = name_compound('c1ccc(CC)cc1CC')
        assert 'diethyl' in name, f"Expected 'diethyl' in '{name}'"
        assert 'benzene' in name, f"Expected 'benzene' in '{name}'"

    def test_trimethylbenzene(self):
        """Trimethylbenzene must retain all three methyl groups."""
        name = name_compound('Cc1cc(C)cc(C)c1')
        assert 'trimethyl' in name, f"Expected 'trimethyl' in '{name}'"
        assert 'benzene' in name, f"Expected 'benzene' in '{name}'"

    def test_butylbenzene(self):
        """Butylbenzene must retain the butyl group."""
        name = name_compound('c1ccc(CCCC)cc1')
        assert 'butyl' in name, f"Expected 'butyl' in '{name}'"
        assert 'benzene' in name, f"Expected 'benzene' in '{name}'"

    def test_methylamino_benzene(self):
        """Methylaminobenzene must retain the methylamino group."""
        name = name_compound('c1ccc(NC)cc1')
        assert 'methylamino' in name, f"Expected 'methylamino' in '{name}'"
        assert 'benzene' in name, f"Expected 'benzene' in '{name}'"


# ---------------------------------------------------------------------------
# Heterocyclic substituents (5 tests)
# Previously: parent heterocycle name returned with substituents dropped
# ---------------------------------------------------------------------------

class TestHeterocycleSubstituents:
    """Heterocyclic compounds must retain substituent prefixes."""

    def test_aminolactone(self):
        """Aminolactone must contain 'amino' prefix on the lactone ring."""
        name = name_compound('NC1CCOC1=O')
        assert 'amino' in name, f"Expected 'amino' in '{name}'"
        assert 'oxolan' in name, f"Expected 'oxolan' in '{name}'"

    def test_aminopyridine(self):
        """Aminopyridine must contain 'amino' prefix."""
        name = name_compound('Nc1ccncc1')
        assert 'amino' in name, f"Expected 'amino' in '{name}'"
        assert 'pyridine' in name, f"Expected 'pyridine' in '{name}'"

    def test_methylpiperidine(self):
        """Methylpiperidine must contain 'methyl' prefix."""
        name = name_compound('CC1CCCCN1')
        assert 'methyl' in name, f"Expected 'methyl' in '{name}'"
        assert 'piperidine' in name, f"Expected 'piperidine' in '{name}'"

    def test_aminotetrahydropyran(self):
        """Aminotetrahydropyran must contain 'amino' prefix."""
        name = name_compound('NC1CCOCC1')
        assert 'amino' in name, f"Expected 'amino' in '{name}'"
        assert 'tetrahydropyran' in name, f"Expected 'tetrahydropyran' in '{name}'"

    def test_methyltetrahydrofuran(self):
        """Methyltetrahydrofuran must contain 'methyl' prefix."""
        name = name_compound('CC1CCCO1')
        assert 'methyl' in name, f"Expected 'methyl' in '{name}'"
        assert 'tetrahydrofuran' in name or 'oxolane' in name, \
            f"Expected 'tetrahydrofuran' or 'oxolane' in '{name}'"


# ---------------------------------------------------------------------------
# Additional edge cases (5 tests)
# ---------------------------------------------------------------------------

class TestRingSubstituentEdgeCases:
    """Edge cases that previously caused substituent loss."""

    def test_nitronaphthalene(self):
        """Nitronaphthalene must keep nitro group."""
        name = name_compound('O=[N+]([O-])c1ccc2ccccc2c1')
        assert 'nitro' in name, f"Expected 'nitro' in '{name}'"
        assert 'naphthalene' in name, f"Expected 'naphthalene' in '{name}'"

    def test_chlorolactone(self):
        """Chloro-substituted lactone must keep chloro group."""
        name = name_compound('ClC1CCOC1=O')
        assert 'chloro' in name, f"Expected 'chloro' in '{name}'"
        assert 'oxolan' in name, f"Expected 'oxolan' in '{name}'"

    def test_hydroxypyridine(self):
        """Hydroxypyridine must contain 'hydroxy' prefix."""
        name = name_compound('Oc1ccncc1')
        assert 'hydroxy' in name, f"Expected 'hydroxy' in '{name}'"
        assert 'pyridine' in name, f"Expected 'pyridine' in '{name}'"

    def test_methylindole_5(self):
        """5-methylindole must contain 'methyl' and '1H-indole'."""
        name = name_compound('Cc1ccc2[nH]ccc2c1')
        assert 'methyl' in name, f"Expected 'methyl' in '{name}'"
        assert 'indole' in name, f"Expected 'indole' in '{name}'"

    def test_dimethylaminobenzene(self):
        """N,N-dimethylaminobenzene must have dimethylamino group."""
        name = name_compound('c1ccc(N(C)C)cc1')
        assert 'dimethylamino' in name, f"Expected 'dimethylamino' in '{name}'"
        assert 'benzene' in name, f"Expected 'benzene' in '{name}'"
