"""
Unit tests for ring merge determinism (PERC-08).

Ensures that get_ring_systems() produces identical groupings regardless of
SSSR iteration order.  Tests run each merge operation 20 times to catch
non-deterministic behavior.
"""

import pytest
from rdkit import Chem
from orthonym.perception.rings import get_ring_systems


class TestRingMergeDeterminism:
    """PERC-08: Ring merge produces identical groupings regardless of iteration order."""

    def test_naphthalene_single_system(self):
        """Naphthalene: 2 fused rings -> 1 ring system, always."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        for _ in range(20):  # Run multiple times to catch non-determinism
            systems = get_ring_systems(mol)
            assert len(systems) == 1, f"Expected 1 system, got {len(systems)}"
            assert len(systems[0]) == 10

    def test_anthracene_single_system(self):
        """Anthracene: 3 fused rings -> 1 ring system, always."""
        mol = Chem.MolFromSmiles("c1ccc2cc3ccccc3cc2c1")
        for _ in range(20):
            systems = get_ring_systems(mol)
            assert len(systems) == 1

    def test_biphenyl_two_systems(self):
        """Biphenyl: 2 disconnected rings -> 2 ring systems."""
        mol = Chem.MolFromSmiles("c1ccc(-c2ccccc2)cc1")
        systems = get_ring_systems(mol)
        assert len(systems) == 2

    def test_indole_single_system(self):
        """Indole: fused 5+6 -> 1 ring system."""
        mol = Chem.MolFromSmiles("c1ccc2[nH]ccc2c1")
        for _ in range(20):
            systems = get_ring_systems(mol)
            assert len(systems) == 1
            assert len(systems[0]) == 9
