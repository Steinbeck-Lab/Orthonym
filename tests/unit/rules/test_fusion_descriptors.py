"""
Unit tests for fusion descriptor generation.

Tests cover:
- get_fusion_edge for finding shared edges in parent rings
- get_fusion_letter for edge-to-letter conversion
- get_child_locants for child ring atom positions
- generate_fusion_descriptor for [num,num-letter] format
- get_fusion_prefix for ring name to prefix conversion
- build_systematic_fusion_name for complete name assembly
- identify_parent_and_child for ring role determination
"""

import pytest
from rdkit import Chem

from orthonym.rules.fusion_descriptors import (
    EDGE_LETTERS,
    get_fusion_edge,
    get_fusion_letter,
    get_child_locants,
    generate_fusion_descriptor,
    get_fusion_prefix,
    build_systematic_fusion_name,
    identify_parent_and_child,
    generate_systematic_name_for_fused_pair,
)


# ============================================================================
# Test EDGE_LETTERS constant
# ============================================================================

class TestEdgeLetters:
    """Tests for EDGE_LETTERS constant."""

    @pytest.mark.unit
    def test_edge_letters_starts_with_a(self):
        """EDGE_LETTERS should start with 'a'."""
        assert EDGE_LETTERS[0] == 'a'

    @pytest.mark.unit
    def test_edge_letters_is_alphabet(self):
        """EDGE_LETTERS should be the lowercase alphabet."""
        assert EDGE_LETTERS == 'abcdefghijklmnopqrstuvwxyz'

    @pytest.mark.unit
    def test_edge_letters_length(self):
        """EDGE_LETTERS should have 26 characters."""
        assert len(EDGE_LETTERS) == 26


# ============================================================================
# Test get_fusion_edge
# ============================================================================

class TestGetFusionEdge:
    """Tests for get_fusion_edge function."""

    @pytest.mark.unit
    def test_first_edge_6_membered(self):
        """Edge 0-1 in 6-membered ring should return 0."""
        ring = [0, 1, 2, 3, 4, 5]
        assert get_fusion_edge(ring, 0, 1) == 0

    @pytest.mark.unit
    def test_second_edge_6_membered(self):
        """Edge 1-2 in 6-membered ring should return 1."""
        ring = [0, 1, 2, 3, 4, 5]
        assert get_fusion_edge(ring, 1, 2) == 1

    @pytest.mark.unit
    def test_third_edge_6_membered(self):
        """Edge 2-3 in 6-membered ring should return 2."""
        ring = [0, 1, 2, 3, 4, 5]
        assert get_fusion_edge(ring, 2, 3) == 2

    @pytest.mark.unit
    def test_fourth_edge_6_membered(self):
        """Edge 3-4 in 6-membered ring should return 3."""
        ring = [0, 1, 2, 3, 4, 5]
        assert get_fusion_edge(ring, 3, 4) == 3

    @pytest.mark.unit
    def test_fifth_edge_6_membered(self):
        """Edge 4-5 in 6-membered ring should return 4."""
        ring = [0, 1, 2, 3, 4, 5]
        assert get_fusion_edge(ring, 4, 5) == 4

    @pytest.mark.unit
    def test_wraparound_edge_6_membered(self):
        """Edge 5-0 (wraparound) in 6-membered ring should return 5."""
        ring = [0, 1, 2, 3, 4, 5]
        assert get_fusion_edge(ring, 5, 0) == 5

    @pytest.mark.unit
    def test_order_independent(self):
        """Edge detection should work regardless of atom order."""
        ring = [0, 1, 2, 3, 4, 5]
        # Same edge, different order
        assert get_fusion_edge(ring, 1, 0) == 0
        assert get_fusion_edge(ring, 2, 1) == 1

    @pytest.mark.unit
    def test_5_membered_ring(self):
        """Should work for 5-membered rings."""
        ring = [0, 1, 2, 3, 4]
        assert get_fusion_edge(ring, 0, 1) == 0
        assert get_fusion_edge(ring, 4, 0) == 4

    @pytest.mark.unit
    def test_non_adjacent_atoms_returns_negative(self):
        """Non-adjacent atoms should return -1."""
        ring = [0, 1, 2, 3, 4, 5]
        assert get_fusion_edge(ring, 0, 2) == -1
        assert get_fusion_edge(ring, 1, 4) == -1

    @pytest.mark.unit
    def test_atom_not_in_ring_returns_negative(self):
        """Atoms not in ring should return -1."""
        ring = [0, 1, 2, 3, 4, 5]
        assert get_fusion_edge(ring, 0, 10) == -1
        assert get_fusion_edge(ring, 10, 11) == -1


# ============================================================================
# Test get_fusion_letter
# ============================================================================

class TestGetFusionLetter:
    """Tests for get_fusion_letter function."""

    @pytest.mark.unit
    def test_first_edge_is_a(self):
        """First edge should return 'a'."""
        ring = [0, 1, 2, 3, 4, 5]
        assert get_fusion_letter(ring, (0, 1)) == 'a'

    @pytest.mark.unit
    def test_second_edge_is_b(self):
        """Second edge should return 'b'."""
        ring = [0, 1, 2, 3, 4, 5]
        assert get_fusion_letter(ring, (1, 2)) == 'b'

    @pytest.mark.unit
    def test_third_edge_is_c(self):
        """Third edge should return 'c'."""
        ring = [0, 1, 2, 3, 4, 5]
        assert get_fusion_letter(ring, (2, 3)) == 'c'

    @pytest.mark.unit
    def test_fourth_edge_is_d(self):
        """Fourth edge should return 'd'."""
        ring = [0, 1, 2, 3, 4, 5]
        assert get_fusion_letter(ring, (3, 4)) == 'd'

    @pytest.mark.unit
    def test_fifth_edge_is_e(self):
        """Fifth edge should return 'e'."""
        ring = [0, 1, 2, 3, 4, 5]
        assert get_fusion_letter(ring, (4, 5)) == 'e'

    @pytest.mark.unit
    def test_last_edge_depends_on_size(self):
        """Last edge letter depends on ring size."""
        ring_5 = [0, 1, 2, 3, 4]
        ring_6 = [0, 1, 2, 3, 4, 5]
        ring_7 = [0, 1, 2, 3, 4, 5, 6]

        # Last edge of 5-membered ring (edge 4-0) = 'e'
        assert get_fusion_letter(ring_5, (4, 0)) == 'e'
        # Last edge of 6-membered ring (edge 5-0) = 'f'
        assert get_fusion_letter(ring_6, (5, 0)) == 'f'
        # Last edge of 7-membered ring (edge 6-0) = 'g'
        assert get_fusion_letter(ring_7, (6, 0)) == 'g'

    @pytest.mark.unit
    def test_non_adjacent_atoms_empty_string(self):
        """Non-adjacent atoms should return empty string."""
        ring = [0, 1, 2, 3, 4, 5]
        assert get_fusion_letter(ring, (0, 3)) == ''

    @pytest.mark.unit
    def test_atom_not_in_ring_empty_string(self):
        """Atoms not in ring should return empty string."""
        ring = [0, 1, 2, 3, 4, 5]
        assert get_fusion_letter(ring, (0, 10)) == ''


# ============================================================================
# Test get_child_locants
# ============================================================================

class TestGetChildLocants:
    """Tests for get_child_locants function."""

    @pytest.mark.unit
    def test_first_two_positions(self):
        """First two positions should return (1, 2)."""
        ring = [6, 7, 8, 9, 10]
        assert get_child_locants(ring, (6, 7)) == (1, 2)

    @pytest.mark.unit
    def test_middle_positions(self):
        """Middle positions should return correct locants."""
        ring = [0, 1, 2, 3, 4]
        assert get_child_locants(ring, (2, 3)) == (3, 4)

    @pytest.mark.unit
    def test_last_two_positions(self):
        """Last two positions should return correct locants."""
        ring = [0, 1, 2, 3, 4]
        assert get_child_locants(ring, (3, 4)) == (4, 5)

    @pytest.mark.unit
    def test_sorted_order_returned(self):
        """Locants should be returned in sorted (lower, higher) order."""
        ring = [0, 1, 2, 3, 4]
        # Even if we give them in reverse order
        assert get_child_locants(ring, (4, 3)) == (4, 5)
        assert get_child_locants(ring, (2, 1)) == (2, 3)

    @pytest.mark.unit
    def test_non_sequential_atom_indices(self):
        """Should work with non-sequential atom indices."""
        ring = [10, 15, 20, 25, 30]
        assert get_child_locants(ring, (15, 20)) == (2, 3)

    @pytest.mark.unit
    def test_atom_not_in_ring_returns_zero(self):
        """Atoms not in ring should return (0, 0)."""
        ring = [0, 1, 2, 3, 4]
        assert get_child_locants(ring, (10, 11)) == (0, 0)


# ============================================================================
# Test generate_fusion_descriptor
# ============================================================================

class TestGenerateFusionDescriptor:
    """Tests for generate_fusion_descriptor function."""

    @pytest.mark.unit
    def test_simple_fusion_descriptor(self):
        """Simple fusion should generate correct descriptor."""
        parent = [0, 1, 2, 3, 4, 5]  # 6-membered
        child = [4, 5, 6, 7, 8]      # 5-membered, shares 4,5 with parent
        shared = {4, 5}

        desc = generate_fusion_descriptor(parent, child, shared)
        assert desc == '[1,2-e]'

    @pytest.mark.unit
    def test_fusion_at_edge_a(self):
        """Fusion at edge 'a' should generate [x,y-a]."""
        parent = [0, 1, 2, 3, 4, 5]
        child = [0, 1, 10, 11, 12]
        shared = {0, 1}

        desc = generate_fusion_descriptor(parent, child, shared)
        assert desc == '[1,2-a]'

    @pytest.mark.unit
    def test_fusion_at_edge_d(self):
        """Fusion at edge 'd' should generate [x,y-d]."""
        parent = [0, 1, 2, 3, 4, 5]
        child = [3, 4, 10, 11, 12]
        shared = {3, 4}

        desc = generate_fusion_descriptor(parent, child, shared)
        assert desc == '[1,2-d]'

    @pytest.mark.unit
    def test_not_two_shared_atoms_returns_empty(self):
        """Less than 2 or more than 2 shared atoms should return empty."""
        parent = [0, 1, 2, 3, 4, 5]
        child = [4, 5, 6, 7, 8]

        # Only 1 shared atom
        desc = generate_fusion_descriptor(parent, child, {4})
        assert desc == ''

        # 3 shared atoms
        desc = generate_fusion_descriptor(parent, child, {3, 4, 5})
        assert desc == ''

    @pytest.mark.unit
    def test_empty_shared_atoms_returns_empty(self):
        """Empty shared atoms should return empty string."""
        parent = [0, 1, 2, 3, 4, 5]
        child = [6, 7, 8, 9, 10]

        desc = generate_fusion_descriptor(parent, child, set())
        assert desc == ''


# ============================================================================
# Test get_fusion_prefix
# ============================================================================

class TestGetFusionPrefix:
    """Tests for get_fusion_prefix function."""

    @pytest.mark.unit
    def test_benzene_to_benzo(self):
        """benzene should convert to benzo."""
        assert get_fusion_prefix('benzene') == 'benzo'

    @pytest.mark.unit
    def test_naphthalene_to_naphtho(self):
        """naphthalene should convert to naphtho."""
        assert get_fusion_prefix('naphthalene') == 'naphtho'

    @pytest.mark.unit
    def test_furan_to_furo(self):
        """furan should convert to furo."""
        assert get_fusion_prefix('furan') == 'furo'

    @pytest.mark.unit
    def test_pyrrole_to_pyrrolo(self):
        """pyrrole should convert to pyrrolo."""
        assert get_fusion_prefix('pyrrole') == 'pyrrolo'

    @pytest.mark.unit
    def test_thiophene_to_thieno(self):
        """thiophene should convert to thieno."""
        assert get_fusion_prefix('thiophene') == 'thieno'

    @pytest.mark.unit
    def test_imidazole_to_imidazo(self):
        """imidazole should convert to imidazo."""
        assert get_fusion_prefix('imidazole') == 'imidazo'

    @pytest.mark.unit
    def test_pyridine_to_pyrido(self):
        """pyridine should convert to pyrido."""
        assert get_fusion_prefix('pyridine') == 'pyrido'

    @pytest.mark.unit
    def test_oxazole_to_oxazolo(self):
        """oxazole should convert to oxazolo."""
        assert get_fusion_prefix('oxazole') == 'oxazolo'

    @pytest.mark.unit
    def test_thiazole_to_thiazolo(self):
        """thiazole should convert to thiazolo."""
        assert get_fusion_prefix('thiazole') == 'thiazolo'

    @pytest.mark.unit
    def test_anthracene_to_anthra(self):
        """anthracene should convert to anthra."""
        assert get_fusion_prefix('anthracene') == 'anthra'

    @pytest.mark.unit
    def test_phenanthrene_to_phenanthro(self):
        """phenanthrene should convert to phenanthro."""
        assert get_fusion_prefix('phenanthrene') == 'phenanthro'

    @pytest.mark.unit
    def test_pyrimidine_to_pyrimido(self):
        """pyrimidine should convert to pyrimido."""
        assert get_fusion_prefix('pyrimidine') == 'pyrimido'

    @pytest.mark.unit
    def test_indole_to_indolo(self):
        """indole (and 1H-indole) should convert to indolo."""
        assert get_fusion_prefix('indole') == 'indolo'
        assert get_fusion_prefix('1H-indole') == 'indolo'

    @pytest.mark.unit
    def test_quinoline_to_quinolino(self):
        """quinoline should convert to quinolino."""
        assert get_fusion_prefix('quinoline') == 'quinolino'

    @pytest.mark.unit
    def test_fallback_ene_rule(self):
        """Unknown -ene endings should use fallback rule."""
        # Fallback: -ene -> -o (use name not in FUSION_PREFIXES dict)
        assert get_fusion_prefix('cyclononene') == 'cyclonono'

    @pytest.mark.unit
    def test_cyclohexene_prefix(self):
        """cyclohexene should map to cyclohexa (IUPAC standard)."""
        assert get_fusion_prefix('cyclohexene') == 'cyclohexa'

    @pytest.mark.unit
    def test_cyclopentadiene_prefix(self):
        """cyclopentadiene should map to cyclopenta."""
        assert get_fusion_prefix('cyclopentadiene') == 'cyclopenta'

    @pytest.mark.unit
    def test_fallback_ole_rule(self):
        """Unknown -ole endings should use fallback rule."""
        # Fallback: -ole -> -olo
        assert get_fusion_prefix('someole') == 'someolo'

    @pytest.mark.unit
    def test_fallback_ine_rule(self):
        """Unknown -ine endings should use fallback rule."""
        # Fallback: -ine -> -ino
        assert get_fusion_prefix('somerine') == 'somerino'


# ============================================================================
# Test build_systematic_fusion_name
# ============================================================================

class TestBuildSystematicFusionName:
    """Tests for build_systematic_fusion_name function."""

    @pytest.mark.unit
    def test_benzo_a_anthracene(self):
        """benzo[a]anthracene should be built correctly."""
        name = build_systematic_fusion_name('anthracene', 'benzene', '[a]')
        assert name == 'benzo[a]anthracene'

    @pytest.mark.unit
    def test_naphtho_2_1_b_furan(self):
        """naphtho[2,1-b]furan should be built correctly."""
        name = build_systematic_fusion_name('furan', 'naphthalene', '[2,1-b]')
        assert name == 'naphtho[2,1-b]furan'

    @pytest.mark.unit
    def test_furo_3_2_b_pyridine(self):
        """furo[3,2-b]pyridine should be built correctly."""
        name = build_systematic_fusion_name('pyridine', 'furan', '[3,2-b]')
        assert name == 'furo[3,2-b]pyridine'

    @pytest.mark.unit
    def test_pyrido_2_3_b_pyrazine(self):
        """pyrido[2,3-b]pyrazine should be built correctly."""
        name = build_systematic_fusion_name('pyrazine', 'pyridine', '[2,3-b]')
        assert name == 'pyrido[2,3-b]pyrazine'

    @pytest.mark.unit
    def test_no_elision_before_vowel(self):
        """IUPAC 2013: 'o' should NOT be elided before vowels."""
        # benzo before anthracene - 'o' before 'a' is NOT elided
        name = build_systematic_fusion_name('anthracene', 'benzene', '[a]')
        assert name == 'benzo[a]anthracene'
        assert 'benz[a]anthracene' not in name  # Not elided

    @pytest.mark.unit
    def test_thieno_fusion(self):
        """thieno fusion should work correctly."""
        name = build_systematic_fusion_name('pyridine', 'thiophene', '[2,3-b]')
        assert name == 'thieno[2,3-b]pyridine'


# ============================================================================
# Test identify_parent_and_child
# ============================================================================

class TestIdentifyParentAndChild:
    """Tests for identify_parent_and_child function."""

    @pytest.mark.unit
    def test_heterocyclic_is_parent_over_carbocyclic(self):
        """IUPAC seniority: heterocyclic ring is parent over carbocyclic."""
        mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')  # indole
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()

        # Find the two rings
        ring_6 = None
        ring_5 = None
        for ring in rings:
            if len(ring) == 6:
                ring_6 = set(ring)
            elif len(ring) == 5:
                ring_5 = set(ring)

        if ring_6 and ring_5:
            parent_name, child_name, parent_ring, child_ring = identify_parent_and_child(
                mol, ring_6, ring_5
            )
            # Pyrrole (N-heterocycle) is more senior than benzene (carbocycle)
            assert parent_name == 'pyrrole', f"Expected pyrrole as parent, got {parent_name}"
            assert child_name == 'benzene', f"Expected benzene as child, got {child_name}"
            assert len(parent_ring) == 5
            assert len(child_ring) == 6

    @pytest.mark.unit
    def test_benzene_identified(self):
        """Benzene ring should be identified."""
        mol = Chem.MolFromSmiles('c1ccccc1')  # benzene
        ring_atoms = list(mol.GetRingInfo().AtomRings()[0])

        from orthonym.rules.fusion_descriptors import _identify_ring_name
        name = _identify_ring_name(mol, ring_atoms)
        assert name == 'benzene'

    @pytest.mark.unit
    def test_furan_identified(self):
        """Furan ring should be identified."""
        mol = Chem.MolFromSmiles('c1ccoc1')  # furan
        ring_atoms = list(mol.GetRingInfo().AtomRings()[0])

        from orthonym.rules.fusion_descriptors import _identify_ring_name
        name = _identify_ring_name(mol, ring_atoms)
        assert name == 'furan'

    @pytest.mark.unit
    def test_pyridine_identified(self):
        """Pyridine ring should be identified."""
        mol = Chem.MolFromSmiles('c1ccncc1')  # pyridine
        ring_atoms = list(mol.GetRingInfo().AtomRings()[0])

        from orthonym.rules.fusion_descriptors import _identify_ring_name
        name = _identify_ring_name(mol, ring_atoms)
        assert name == 'pyridine'

    @pytest.mark.unit
    def test_thiophene_identified(self):
        """Thiophene ring should be identified."""
        mol = Chem.MolFromSmiles('c1ccsc1')  # thiophene
        ring_atoms = list(mol.GetRingInfo().AtomRings()[0])

        from orthonym.rules.fusion_descriptors import _identify_ring_name
        name = _identify_ring_name(mol, ring_atoms)
        assert name == 'thiophene'

    @pytest.mark.unit
    def test_pyrrole_identified(self):
        """Pyrrole ring should be identified."""
        mol = Chem.MolFromSmiles('c1cc[nH]c1')  # pyrrole
        ring_atoms = list(mol.GetRingInfo().AtomRings()[0])

        from orthonym.rules.fusion_descriptors import _identify_ring_name
        name = _identify_ring_name(mol, ring_atoms)
        assert name == 'pyrrole'


# ============================================================================
# Test generate_systematic_name_for_fused_pair
# ============================================================================

class TestGenerateSystematicNameForFusedPair:
    """Tests for generate_systematic_name_for_fused_pair function."""

    @pytest.mark.unit
    def test_returns_none_for_wrong_shared_count(self):
        """Should return None if shared atoms != 2."""
        mol = Chem.MolFromSmiles('c1ccccc1')  # benzene
        ring = list(mol.GetRingInfo().AtomRings()[0])

        # Only 1 shared atom
        result = generate_systematic_name_for_fused_pair(
            mol, ring, ring, {0}
        )
        assert result is None

        # No shared atoms
        result = generate_systematic_name_for_fused_pair(
            mol, ring, ring, set()
        )
        assert result is None

    @pytest.mark.unit
    def test_with_simple_fused_system(self):
        """Should generate a name for simple fused system."""
        # Create a simple test case with overlapping ring lists
        mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')  # indole
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()

        if len(rings) >= 2:
            ring1 = list(rings[0])
            ring2 = list(rings[1])
            shared = set(ring1) & set(ring2)

            # This should return a name (may not match retained name exactly)
            result = generate_systematic_name_for_fused_pair(
                mol, ring1, ring2, shared
            )
            # Should return some result for a valid fused system
            assert result is not None or len(shared) != 2


# ============================================================================
# Edge cases and boundary conditions
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_empty_ring(self):
        """Empty ring should be handled gracefully."""
        assert get_fusion_edge([], 0, 1) == -1
        assert get_fusion_letter([], (0, 1)) == ''
        assert get_child_locants([], (0, 1)) == (0, 0)

    @pytest.mark.unit
    def test_single_atom_ring(self):
        """Single atom ring should be handled gracefully."""
        assert get_fusion_edge([0], 0, 1) == -1
        assert get_child_locants([0], (0, 0)) == (1, 1)

    @pytest.mark.unit
    def test_large_ring(self):
        """Large rings (> 26 edges) should handle edge letters gracefully."""
        # 30-membered ring would have 30 edges, but only 26 letters
        ring = list(range(30))
        # Edge 25 (last valid)
        assert get_fusion_letter(ring, (25, 26)) == 'z'
        # Edge 26+ would be out of range
        edge_idx = get_fusion_edge(ring, 26, 27)
        if edge_idx >= 26:
            letter = get_fusion_letter(ring, (26, 27))
            assert letter == ''  # Out of range

    @pytest.mark.unit
    def test_descriptor_format_consistency(self):
        """Fusion descriptor format should be consistent."""
        parent = [0, 1, 2, 3, 4, 5]
        child = [0, 1, 10, 11, 12]
        shared = {0, 1}

        desc = generate_fusion_descriptor(parent, child, shared)
        # Should have format [num,num-letter]
        assert desc.startswith('[')
        assert desc.endswith(']')
        assert '-' in desc
        assert ',' in desc


# ============================================================================
# Test IUPAC-ordered descriptor generation (Plan 112-02)
# ============================================================================

class TestIUPACOrderedDescriptors:
    """Tests for fusion descriptors generated from IUPAC-ordered rings."""

    @pytest.mark.unit
    def test_furo_pyrrole_descriptor_iupac(self):
        """furo[2,3-b]pyrrole: furan fused to pyrrole at edge b."""
        mol = Chem.MolFromSmiles('c1cc2ccoc2[nH]1')
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()
        shared = set(rings[0]) & set(rings[1])

        result = generate_systematic_name_for_fused_pair(
            mol, list(rings[0]), list(rings[1]), shared
        )
        assert result == 'furo[2,3-b]pyrrole'

    @pytest.mark.unit
    def test_thieno_thiophene_descriptor(self):
        """thieno[2,3-b]thiophene: correct edge letter and child locants."""
        mol = Chem.MolFromSmiles('c1cc2ccsc2s1')
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()
        shared = set(rings[0]) & set(rings[1])

        result = generate_systematic_name_for_fused_pair(
            mol, list(rings[0]), list(rings[1]), shared
        )
        assert result == 'thieno[2,3-b]thiophene'

    @pytest.mark.unit
    def test_thieno_pyridine_descriptor_3_2_b(self):
        """thieno[3,2-b]pyridine: descending child locants with edge b."""
        mol = Chem.MolFromSmiles('c1cnc2ccsc2c1')
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()
        shared = set(rings[0]) & set(rings[1])

        result = generate_systematic_name_for_fused_pair(
            mol, list(rings[0]), list(rings[1]), shared
        )
        assert result == 'thieno[3,2-b]pyridine'

    @pytest.mark.unit
    def test_furo_pyridine_descriptor_3_2_b(self):
        """furo[3,2-b]pyridine: descending child locants with edge b."""
        mol = Chem.MolFromSmiles('c1cnc2ccoc2c1')
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()
        shared = set(rings[0]) & set(rings[1])

        result = generate_systematic_name_for_fused_pair(
            mol, list(rings[0]), list(rings[1]), shared
        )
        assert result == 'furo[3,2-b]pyridine'

    @pytest.mark.unit
    def test_furo_pyrimidine_descriptor_2_3_d(self):
        """furo[2,3-d]pyrimidine: edge d on pyrimidine parent."""
        mol = Chem.MolFromSmiles('c1ncc2ccoc2n1')
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()
        shared = set(rings[0]) & set(rings[1])

        result = generate_systematic_name_for_fused_pair(
            mol, list(rings[0]), list(rings[1]), shared
        )
        assert result == 'furo[2,3-d]pyrimidine'

    @pytest.mark.unit
    def test_thieno_pyrimidine_descriptor_2_3_d(self):
        """thieno[2,3-d]pyrimidine: edge d on pyrimidine parent."""
        mol = Chem.MolFromSmiles('c1ncc2ccsc2n1')
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()
        shared = set(rings[0]) & set(rings[1])

        result = generate_systematic_name_for_fused_pair(
            mol, list(rings[0]), list(rings[1]), shared
        )
        assert result == 'thieno[2,3-d]pyrimidine'

    @pytest.mark.unit
    def test_benzo_c_thiophene_descriptor(self):
        """benzo[c]thiophene: benzene child locants omitted."""
        mol = Chem.MolFromSmiles('c1ccc2cscc2c1')
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()
        shared = set(rings[0]) & set(rings[1])

        result = generate_systematic_name_for_fused_pair(
            mol, list(rings[0]), list(rings[1]), shared
        )
        assert result == 'benzo[c]thiophene'

    @pytest.mark.unit
    def test_benzo_d_thiazole_descriptor(self):
        """benzo[d]thiazole: benzene child locants omitted."""
        mol = Chem.MolFromSmiles('c1ccc2scnc2c1')
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()
        shared = set(rings[0]) & set(rings[1])

        result = generate_systematic_name_for_fused_pair(
            mol, list(rings[0]), list(rings[1]), shared
        )
        assert result == 'benzo[d]thiazole'

    @pytest.mark.unit
    def test_benzo_d_imidazole_descriptor(self):
        """benzo[d]imidazole: benzene child locants omitted."""
        mol = Chem.MolFromSmiles('c1ccc2[nH]cnc2c1')
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()
        shared = set(rings[0]) & set(rings[1])

        result = generate_systematic_name_for_fused_pair(
            mol, list(rings[0]), list(rings[1]), shared
        )
        assert result == 'benzo[d]imidazole'

    @pytest.mark.unit
    def test_pyrido_pyrazine_descriptor(self):
        """pyrido[2,3-b]pyrazine: pyridine-pyrazine fusion."""
        mol = Chem.MolFromSmiles('c1cnc2nccnc2c1')
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()
        shared = set(rings[0]) & set(rings[1])

        result = generate_systematic_name_for_fused_pair(
            mol, list(rings[0]), list(rings[1]), shared
        )
        assert result == 'pyrido[2,3-b]pyrazine'

    @pytest.mark.unit
    def test_furo_c_pyridine_descriptor(self):
        """furo[3,2-c]pyridine: edge c on pyridine parent."""
        mol = Chem.MolFromSmiles('c1cc2occc2cn1')
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()
        shared = set(rings[0]) & set(rings[1])

        result = generate_systematic_name_for_fused_pair(
            mol, list(rings[0]), list(rings[1]), shared
        )
        assert result == 'furo[3,2-c]pyridine'

    @pytest.mark.unit
    def test_thieno_c_pyridine_descriptor(self):
        """thieno[2,3-c]pyridine: edge c on pyridine parent."""
        mol = Chem.MolFromSmiles('c1cc2ccsc2cn1')
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()
        shared = set(rings[0]) & set(rings[1])

        result = generate_systematic_name_for_fused_pair(
            mol, list(rings[0]), list(rings[1]), shared
        )
        assert result == 'thieno[2,3-c]pyridine'

    @pytest.mark.unit
    def test_pyrrolo_pyridine_descriptor(self):
        """pyrrolo[2,3-b]pyridine: correct descriptor for pyrrole-pyridine."""
        mol = Chem.MolFromSmiles('c1cnc2[nH]ccc2c1')
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()
        shared = set(rings[0]) & set(rings[1])

        result = generate_systematic_name_for_fused_pair(
            mol, list(rings[0]), list(rings[1]), shared
        )
        assert result == 'pyrrolo[2,3-b]pyridine'

    @pytest.mark.unit
    def test_descriptor_format_valid_regex(self):
        """All generated descriptors match valid IUPAC format."""
        import re
        # Pattern: [num,num-letter] or [letter]
        pattern = re.compile(r'^\[\d+,\d+-[a-z]\]$|^\[[a-z]\]$')

        test_smiles = [
            'c1cc2ccoc2[nH]1',   # furo[2,3-b]pyrrole
            'c1cnc2ccsc2c1',     # thieno[3,2-b]pyridine
            'c1ccc2cscc2c1',     # benzo[c]thiophene
            'c1ncc2ccoc2n1',     # furo[2,3-d]pyrimidine
        ]

        for smiles in test_smiles:
            mol = Chem.MolFromSmiles(smiles)
            ri = mol.GetRingInfo()
            rings = ri.AtomRings()
            shared = set(rings[0]) & set(rings[1])

            result = generate_systematic_name_for_fused_pair(
                mol, list(rings[0]), list(rings[1]), shared
            )
            assert result is not None, f"No name for {smiles}"

            # Extract descriptor from name
            desc_match = re.search(r'\[.*?\]', result)
            assert desc_match is not None, f"No descriptor in {result}"
            desc = desc_match.group()
            assert pattern.match(desc), f"Invalid descriptor format: {desc} in {result}"
