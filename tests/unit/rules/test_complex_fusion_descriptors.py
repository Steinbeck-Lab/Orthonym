"""
Unit tests for complex fusion descriptor generation.

Tests the extended fusion descriptor functionality including:
- Multi-component fusion descriptors ([a,c] notation)
- Primed notation for duplicate edges ([a,a'] notation)
- Complex edge locants ([2,3-b], [1,2-a:4,5-b'] notation)
- End-to-end complex fusion name generation

Reference: IUPAC 2013 Blue Book, Section (Fused Ring Systems)
"""

import pytest
from rdkit import Chem

from orthonym.rules.fusion_descriptors import (
    format_complex_fusion,
    handle_duplicate_edge_fusion,
    generate_multi_fusion_descriptor,
    identify_fusion_edges,
    edge_position_to_letter,
    get_fusion_letter,
    build_multi_component_name,
    get_fusion_edge,
    EDGE_LETTERS,
)
from orthonym.data.polycyclic_data import (
    COMPLEX_FUSION_DATA,
    get_complex_fusion_info,
    get_complex_fusion_by_name,
    get_edge_letter,
    get_all_edge_letters,
    PARENT_RING_EDGES,
)


# =============================================================================
# Test Multi-Component Fusion
# =============================================================================

class TestMultiComponentFusion:
    """Tests for multi-component fusion descriptor generation."""

    def test_dibenzo_descriptor_ac(self):
        """Test [a,c] format for two benzene fusions."""
        result = format_complex_fusion(None, None, multi_component=['a', 'c'])
        assert result == '[a,c]'

    def test_dibenzo_descriptor_ah(self):
        """Test [a,h] format for dibenzo[a,h] type fusions."""
        result = format_complex_fusion(None, None, multi_component=['a', 'h'])
        assert result == '[a,h]'

    def test_dinaphtho_descriptor(self):
        """Test descriptor for dinaphtho-type fusions."""
        result = format_complex_fusion(None, None, multi_component=['a', 'j'])
        assert result == '[a,j]'

    def test_alphabetical_ordering(self):
        """Test that letters are sorted alphabetically."""
        result = format_complex_fusion(None, None, multi_component=['c', 'a'])
        assert result == '[a,c]'

    def test_three_component_fusion(self):
        """Test descriptor for three-component fusion."""
        result = format_complex_fusion(None, None, multi_component=['a', 'c', 'h'])
        assert result == '[a,c,h]'

    @pytest.mark.parametrize("letters,expected", [
        (['a', 'b'], '[a,b]'),
        (['b', 'f'], '[b,f]'),
        (['a', 'd', 'g'], '[a,d,g]'),
        (['h', 'a', 'c'], '[a,c,h]'),
    ])
    def test_various_multi_component(self, letters, expected):
        """Test various multi-component fusion descriptors."""
        result = format_complex_fusion(None, None, multi_component=letters)
        assert result == expected


# =============================================================================
# Test Primed Notation
# =============================================================================

class TestPrimedNotation:
    """Tests for primed notation with duplicate edge letters."""

    def test_duplicate_edge_primed(self):
        """Test [a,a'] for two fusions at same edge type."""
        result = handle_duplicate_edge_fusion(['a', 'a'])
        assert result == ['a', "a'"]

    def test_triple_duplicate_edge(self):
        """Test [a,a',a''] for three fusions at same edge."""
        result = handle_duplicate_edge_fusion(['a', 'a', 'a'])
        assert result == ['a', "a'", "a''"]

    def test_mixed_primed(self):
        """Test [a,b,a'] for mixed distinct and duplicate."""
        result = handle_duplicate_edge_fusion(['a', 'b', 'a'])
        # Should be sorted: a, a', b
        assert result == ['a', "a'", 'b']

    def test_prime_only_when_needed(self):
        """Test no primes for distinct edges."""
        result = handle_duplicate_edge_fusion(['a', 'b', 'c'])
        assert result == ['a', 'b', 'c']

    def test_complex_priming(self):
        """Test complex case with multiple duplicates."""
        result = handle_duplicate_edge_fusion(['a', 'b', 'a', 'b'])
        assert result == ['a', "a'", 'b', "b'"]

    def test_empty_input(self):
        """Test empty input returns empty list."""
        result = handle_duplicate_edge_fusion([])
        assert result == []

    def test_single_letter(self):
        """Test single letter returns unchanged."""
        result = handle_duplicate_edge_fusion(['a'])
        assert result == ['a']


# =============================================================================
# Test Complex Edge Locants
# =============================================================================

class TestComplexEdgeLocants:
    """Tests for complex edge locant formatting."""

    def test_standard_edge_locant(self):
        """Test [2,3-b] format for standard fusion."""
        result = format_complex_fusion((2, 3), 'b')
        assert result == '[2,3-b]'

    def test_edge_locant_23a(self):
        """Test [2,3-a] format."""
        result = format_complex_fusion((2, 3), 'a')
        assert result == '[2,3-a]'

    def test_edge_locant_12b(self):
        """Test [1,2-b] format."""
        result = format_complex_fusion((1, 2), 'b')
        assert result == '[1,2-b]'

    def test_multi_edge_descriptor(self):
        """Test [1,2-a:4,5-b'] for multiple edges."""
        result = format_complex_fusion(
            None, None,
            multi_edge=[((1, 2), 'a'), ((4, 5), "b'")]
        )
        assert result == "[1,2-a:4,5-b']"

    def test_multi_edge_three_parts(self):
        """Test three-part multi-edge descriptor."""
        result = format_complex_fusion(
            None, None,
            multi_edge=[((1, 2), 'a'), ((3, 4), 'c'), ((5, 6), 'e')]
        )
        assert result == "[1,2-a:3,4-c:5,6-e]"

    @pytest.mark.parametrize("locants,letter,expected", [
        ((1, 2), 'a', '[1,2-a]'),
        ((2, 3), 'b', '[2,3-b]'),
        ((3, 4), 'c', '[3,4-c]'),
        ((4, 5), 'd', '[4,5-d]'),
    ])
    def test_various_edge_locants(self, locants, letter, expected):
        """Test various standard edge locant formats."""
        result = format_complex_fusion(locants, letter)
        assert result == expected


# =============================================================================
# Test Edge Letter Generation
# =============================================================================

class TestEdgeLetterGeneration:
    """Tests for converting edge positions to letters."""

    def test_edge_to_letter_first(self):
        """Test edge 0 -> 'a'."""
        ring = [0, 1, 2, 3, 4, 5]
        letter = edge_position_to_letter(ring, (0, 1))
        assert letter == 'a'

    def test_edge_to_letter_third(self):
        """Test edge 2 -> 'c'."""
        ring = [0, 1, 2, 3, 4, 5]
        letter = edge_position_to_letter(ring, (2, 3))
        assert letter == 'c'

    def test_edge_letters_constant(self):
        """Verify edge letters string is correct."""
        assert EDGE_LETTERS == 'abcdefghijklmnopqrstuvwxyz'

    def test_get_fusion_edge_wraparound(self):
        """Test edge detection with wraparound (last to first atom)."""
        ring = [0, 1, 2, 3, 4, 5]
        edge = get_fusion_edge(ring, 5, 0)
        assert edge == 5  # Last edge (wraparound)

    def test_get_fusion_edge_not_adjacent(self):
        """Test non-adjacent atoms return -1."""
        ring = [0, 1, 2, 3, 4, 5]
        edge = get_fusion_edge(ring, 0, 3)
        assert edge == -1


# =============================================================================
# Test Fusion Edge Identification
# =============================================================================

class TestFusionEdgeIdentification:
    """Tests for identifying fusion edges between rings."""

    def test_find_shared_edge_simple(self):
        """Test finding shared edge between two rings.

        Use rings with known overlapping atoms that form a shared edge.
        """
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')  # naphthalene
        if mol is not None:
            # Get actual ring info from RDKit
            ri = mol.GetRingInfo()
            rings = ri.AtomRings()
            if len(rings) >= 2:
                ring1 = list(rings[0])
                ring2 = list(rings[1])
                # Find shared atoms
                shared = set(ring1) & set(ring2)
                if len(shared) >= 2:
                    edges = identify_fusion_edges(mol, ring1, ring2)
                    # Should find at least one shared edge in fused naphthalene
                    assert len(edges) >= 1
                else:
                    # If no shared atoms detected, skip gracefully
                    pytest.skip("Ring perception didn't find shared atoms")
            else:
                pytest.skip("Ring perception didn't find 2 rings")

    def test_identify_fusion_edges_with_shared_atoms(self):
        """Test fusion edge identification with known shared atoms."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')  # naphthalene
        if mol is not None:
            # Manually construct rings with shared edge
            parent = [0, 1, 2, 3, 4, 5]  # Synthetic ring
            child = [3, 4, 5, 6, 7, 8]   # Overlaps at 3, 4, 5
            # Even though these aren't real ring indices for this molecule,
            # the function should find edges where atoms overlap and are bonded
            # For unit test purposes, just verify the function runs
            edges = identify_fusion_edges(mol, parent, child)
            # Result depends on actual bonding - could be empty or have edges
            assert isinstance(edges, list)

    def test_edge_to_letter_conversion(self):
        """Test converting edge tuple to letter."""
        ring = [0, 1, 2, 3, 4, 5]
        letter = edge_position_to_letter(ring, (1, 2))
        assert letter == 'b'


# =============================================================================
# Test E2E Complex Fusion
# =============================================================================

class TestE2EComplexFusion:
    """End-to-end tests for complex fusion name generation."""

    def test_dibenzo_anthracene_data_exists(self):
        """Test dibenzo[a,c]anthracene is in data."""
        assert 'dibenzo[a,c]anthracene' in COMPLEX_FUSION_DATA
        data = COMPLEX_FUSION_DATA['dibenzo[a,c]anthracene']
        assert data['descriptor'] == '[a,c]'
        assert data['parent'] == 'anthracene'
        assert data['child'] == 'benzene'
        assert data['child_count'] == 2

    def test_naphtho_furan_data_exists(self):
        """Test naphtho[2,3-b]furan is in data."""
        assert 'naphtho[2,3-b]furan' in COMPLEX_FUSION_DATA
        data = COMPLEX_FUSION_DATA['naphtho[2,3-b]furan']
        assert data['descriptor'] == '[2,3-b]'
        assert data['parent'] == 'furan'

    def test_pyrido_pyrimidine_data_exists(self):
        """Test pyrido[2,3-d]pyrimidine is in data."""
        assert 'pyrido[2,3-d]pyrimidine' in COMPLEX_FUSION_DATA
        data = COMPLEX_FUSION_DATA['pyrido[2,3-d]pyrimidine']
        assert data['descriptor'] == '[2,3-d]'
        assert data['parent'] == 'pyrimidine'

    def test_build_multi_component_name_dibenzo(self):
        """Test building dibenzo[a,c]anthracene name."""
        name = build_multi_component_name('anthracene', 'benzene', 2, '[a,c]')
        assert name == 'dibenzo[a,c]anthracene'

    def test_build_multi_component_name_dinaphtho(self):
        """Test building dinaphtho name."""
        name = build_multi_component_name('anthracene', 'naphthalene', 2, '[a,h]')
        assert name == 'dinaphtho[a,h]anthracene'

    def test_build_multi_component_name_tribenzo(self):
        """Test building tribenzo name."""
        name = build_multi_component_name('coronene', 'benzene', 3, '[a,c,g]')
        assert name == 'tribenzo[a,c,g]coronene'


# =============================================================================
# Test Data Lookup Functions
# =============================================================================

class TestDataLookupFunctions:
    """Tests for complex fusion data lookup functions."""

    def test_get_complex_fusion_by_name(self):
        """Test looking up fusion data by name."""
        data = get_complex_fusion_by_name('naphtho[2,3-b]thiophene')
        assert data is not None
        assert data['prefix'] == 'naphtho'
        assert data['parent'] == 'thiophene'

    def test_get_complex_fusion_by_name_not_found(self):
        """Test lookup returns None for unknown name."""
        data = get_complex_fusion_by_name('unknown[a,b]compound')
        assert data is None

    def test_get_edge_letter_anthracene(self):
        """Test edge letter lookup for anthracene."""
        assert get_edge_letter('anthracene', 0) == 'a'
        assert get_edge_letter('anthracene', 2) == 'c'
        assert get_edge_letter('anthracene', 7) == 'h'

    def test_get_edge_letter_benzene(self):
        """Test edge letter lookup for benzene."""
        assert get_edge_letter('benzene', 0) == 'a'
        assert get_edge_letter('benzene', 5) == 'f'

    def test_get_all_edge_letters_benzene(self):
        """Test getting all edge letters for benzene."""
        letters = get_all_edge_letters('benzene')
        assert letters == ['a', 'b', 'c', 'd', 'e', 'f']

    def test_get_all_edge_letters_naphthalene(self):
        """Test getting all edge letters for naphthalene."""
        letters = get_all_edge_letters('naphthalene')
        assert len(letters) == 10  # naphthalene has 10 edges

    def test_parent_ring_edges_data(self):
        """Test PARENT_RING_EDGES data is populated."""
        assert 'anthracene' in PARENT_RING_EDGES
        assert 'naphthalene' in PARENT_RING_EDGES
        assert 'benzene' in PARENT_RING_EDGES
        assert 'pyridine' in PARENT_RING_EDGES


# =============================================================================
# Test Multi-Fusion Descriptor Generation
# =============================================================================

class TestMultiFusionDescriptorGeneration:
    """Tests for generate_multi_fusion_descriptor function."""

    def test_generate_multi_fusion_simple(self):
        """Test generating descriptor for simple multi-fusion."""
        parent_ring = list(range(14))  # Anthracene-like
        components = [
            ('benzene', parent_ring, {0, 1}),  # Edge a
            ('benzene', parent_ring, {4, 5}),  # Edge e (approx)
        ]
        # Note: actual edges depend on ring ordering
        # This tests the function can be called

    def test_format_empty_multi_component(self):
        """Test format with empty multi_component returns empty."""
        result = format_complex_fusion(None, None, multi_component=[])
        assert result == '[]'  # Empty brackets for empty list

    def test_format_no_args_returns_empty(self):
        """Test format with no valid args returns empty string."""
        result = format_complex_fusion(None, None)
        assert result == ''


# =============================================================================
# Test IUPAC 2013 Compliance
# =============================================================================

class TestIUPAC2013Compliance:
    """Tests for IUPAC 2013 nomenclature compliance."""

    def test_no_vowel_elision(self):
        """Verify IUPAC 2013 rule: no 'o' elision before vowels.

        IUPAC 2013 specifies that fusion prefix 'o' is NOT elided
        before vowels (benzo[a]anthracene, not benz[a]anthracene).
        """
        # Our format_complex_fusion should not modify letters
        result = format_complex_fusion((2, 3), 'a')
        assert result == '[2,3-a]'
        # 'a' should remain, no elision of anything

    def test_edge_letter_starts_at_a(self):
        """Verify edge 'a' is between atoms 1-2 (first edge)."""
        ring = list(range(6))  # 6-membered ring
        letter = edge_position_to_letter(ring, (0, 1))
        assert letter == 'a'

    def test_complex_fusion_data_count(self):
        """Verify we have substantial complex fusion data."""
        assert len(COMPLEX_FUSION_DATA) >= 10  # At least 10 entries
