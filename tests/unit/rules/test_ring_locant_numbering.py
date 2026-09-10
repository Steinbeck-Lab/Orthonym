"""
Tests for: Ring locant comparison uses actual IUPAC ring numbering.

The three parent selection comparison functions (_compare_pg_locants,
_compare_multiple_bond_locants, _compare_substituent_locants) must use
IUPAC ring numbering (from iupac_locants data) for heterocycles, instead
of sorted atom index positional proxy which produces incorrect locants
when IUPAC numbering depends on heteroatom orientation.
"""

import pytest
from rdkit import Chem

from orthonym.rules.parent_selection import _build_ring_pos


class TestBuildRingPos:
    """: Ring locant comparison uses actual IUPAC ring numbering."""

    def test_with_iupac_locants(self):
        """When iupac_locants provided, use them instead of sorted indices."""
        ring_set = {5, 3, 7, 1, 9}  # Arbitrary atom indices
        iupac_locants = {5: 1, 3: 2, 7: 3, 1: 4, 9: 5}  # IUPAC numbering
        ring_info = {"iupac_locants": iupac_locants}
        ring_pos = _build_ring_pos(ring_set, ring_info=ring_info)
        assert ring_pos[5] == 1  # Atom 5 is IUPAC position 1
        assert ring_pos[3] == 2  # Atom 3 is IUPAC position 2
        assert ring_pos[1] == 4  # Atom 1 is IUPAC position 4

    def test_without_ring_info_fallback(self):
        """Without ring_info, fall back to sorted-index approach."""
        ring_set = {5, 3, 7, 1, 9}
        ring_pos = _build_ring_pos(ring_set)
        # Sorted: [1, 3, 5, 7, 9] -> {1:1, 3:2, 5:3, 7:4, 9:5}
        assert ring_pos[1] == 1
        assert ring_pos[3] == 2
        assert ring_pos[5] == 3

    def test_with_empty_ring_info(self):
        """Empty ring_info dict falls back to sorted indices."""
        ring_set = {5, 3, 7}
        ring_pos = _build_ring_pos(ring_set, ring_info={})
        sorted_indices = sorted(ring_set)
        for i, idx in enumerate(sorted_indices):
            assert ring_pos[idx] == i + 1

    def test_iupac_locants_only_for_ring_atoms(self):
        """iupac_locants may contain atoms outside ring_set; only ring atoms used."""
        ring_set = {2, 4, 6}
        iupac_locants = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
        ring_info = {"iupac_locants": iupac_locants}
        ring_pos = _build_ring_pos(ring_set, ring_info=ring_info)
        assert set(ring_pos.keys()) == {2, 4, 6}
        assert ring_pos[2] == 2
        assert ring_pos[4] == 4
        assert ring_pos[6] == 6

    def test_string_locants_trigger_fallback(self):
        """String locants (e.g., '3a', '7a' fusion locants) cause fallback.

        IUPAC fused heterocycles have string locants for fusion atoms.
        Since compare_locant_sets works with int locants only, _build_ring_pos
        must fall back to sorted indices when ring_set contains atoms mapped
        to string locants (partial int coverage).
        """
        ring_set = {0, 1, 2, 3, 4}
        # Atom 3 has string locant '3a' (fusion atom) -- only 4/5 are int
        iupac_locants = {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1}
        ring_info = {"iupac_locants": iupac_locants}
        ring_pos = _build_ring_pos(ring_set, ring_info=ring_info)
        # Should fall back because len(ring_pos with ints) != len(ring_set)
        sorted_indices = sorted(ring_set)
        for i, idx in enumerate(sorted_indices):
            assert ring_pos[idx] == i + 1

    def test_with_none_ring_info(self):
        """Explicitly passing None falls back to sorted indices."""
        ring_set = {10, 20, 30}
        ring_pos = _build_ring_pos(ring_set, ring_info=None)
        assert ring_pos[10] == 1
        assert ring_pos[20] == 2
        assert ring_pos[30] == 3

    def test_complete_int_coverage_uses_iupac(self):
        """When all ring atoms have integer iupac_locants, use them."""
        ring_set = {0, 1, 2, 3, 4, 5}
        # All 6 atoms have integer locants (no fusion atoms in ring_set)
        iupac_locants = {0: 4, 1: 5, 2: 6, 3: 1, 4: 2, 5: 3}
        ring_info = {"iupac_locants": iupac_locants}
        ring_pos = _build_ring_pos(ring_set, ring_info=ring_info)
        assert ring_pos[0] == 4
        assert ring_pos[1] == 5
        assert ring_pos[3] == 1
        assert ring_pos[4] == 2
