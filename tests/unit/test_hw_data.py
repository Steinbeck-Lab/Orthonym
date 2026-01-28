"""
Unit tests for Hantzsch-Widman nomenclature data modules.

Tests HW heteroatom prefix lookups, HW stem suffix lookups,
priority ordering, and complete HW name building.
"""

import pytest
from src.orthonym.data.hw_heteroatoms import (
    HW_PREFIXES,
    HETEROATOM_PRIORITY,
    get_hw_prefix,
    get_heteroatom_priority,
    compare_heteroatom_priority,
    sort_heteroatoms_by_priority,
)
from src.orthonym.data.hw_stems import (
    HW_STEMS,
    HETEROATOMS_USE_ANE,
    HETEROATOMS_USE_INANE,
    get_hw_stem,
    get_unsaturated_stem,
    get_saturated_stem,
    is_supported_ring_size,
    get_supported_ring_sizes,
)


# =============================================================================
# HW Heteroatom Prefix Tests
# =============================================================================

class TestHWPrefixes:
    """Test HW_PREFIXES dict and get_hw_prefix function."""

    @pytest.mark.unit
    @pytest.mark.parametrize("element,expected_prefix", [
        ('O', 'oxa'),
        ('S', 'thia'),
        ('Se', 'selena'),
        ('Te', 'tellura'),
        ('N', 'aza'),
        ('P', 'phospha'),
        ('As', 'arsa'),
        ('Sb', 'stiba'),
        ('Bi', 'bisma'),
        ('Si', 'sila'),
        ('Ge', 'germa'),
        ('Sn', 'stanna'),
        ('Pb', 'plumba'),
        ('B', 'bora'),
        ('Hg', 'mercura'),
    ])
    def test_hw_prefix_lookup(self, element, expected_prefix):
        """Test HW prefix for each supported heteroatom."""
        assert get_hw_prefix(element) == expected_prefix
        assert HW_PREFIXES[element] == expected_prefix

    @pytest.mark.unit
    def test_hw_prefix_unknown_element(self):
        """Unknown elements should return None."""
        assert get_hw_prefix('C') is None  # Carbon has no HW prefix
        assert get_hw_prefix('H') is None  # Hydrogen has no HW prefix
        assert get_hw_prefix('X') is None  # Invalid element
        assert get_hw_prefix('') is None   # Empty string

    @pytest.mark.unit
    def test_hw_prefixes_dict_completeness(self):
        """HW_PREFIXES should contain all common heteroatoms."""
        required_elements = {'O', 'S', 'Se', 'Te', 'N', 'P', 'As', 'Si', 'Ge', 'B'}
        for elem in required_elements:
            assert elem in HW_PREFIXES, f"Missing element: {elem}"


class TestHeteroatomPriority:
    """Test HETEROATOM_PRIORITY dict and priority functions."""

    @pytest.mark.unit
    @pytest.mark.parametrize("element,expected_priority", [
        ('O', 1),
        ('S', 2),
        ('Se', 3),
        ('Te', 4),
        ('N', 5),
        ('P', 6),
        ('As', 7),
        ('Sb', 8),
        ('Bi', 9),
        ('Si', 10),
        ('Ge', 11),
        ('Sn', 12),
        ('Pb', 13),
        ('B', 14),
        ('Hg', 15),
    ])
    def test_heteroatom_priority_lookup(self, element, expected_priority):
        """Test priority value for each heteroatom."""
        assert get_heteroatom_priority(element) == expected_priority
        assert HETEROATOM_PRIORITY[element] == expected_priority

    @pytest.mark.unit
    def test_heteroatom_priority_unknown(self):
        """Unknown elements should return 999 (lowest priority)."""
        assert get_heteroatom_priority('C') == 999
        assert get_heteroatom_priority('H') == 999
        assert get_heteroatom_priority('X') == 999
        assert get_heteroatom_priority('') == 999

    @pytest.mark.unit
    def test_priority_ordering(self):
        """Verify IUPAC priority ordering: O > S > N > P > Si > B."""
        # O has highest priority (lowest number)
        assert get_heteroatom_priority('O') < get_heteroatom_priority('S')
        assert get_heteroatom_priority('S') < get_heteroatom_priority('N')
        assert get_heteroatom_priority('N') < get_heteroatom_priority('P')
        assert get_heteroatom_priority('P') < get_heteroatom_priority('Si')
        assert get_heteroatom_priority('Si') < get_heteroatom_priority('B')

    @pytest.mark.unit
    def test_compare_heteroatom_priority(self):
        """Test compare_heteroatom_priority function."""
        # O has higher priority than N (returns -1)
        assert compare_heteroatom_priority('O', 'N') == -1
        # N has lower priority than O (returns 1)
        assert compare_heteroatom_priority('N', 'O') == 1
        # Same element has same priority (returns 0)
        assert compare_heteroatom_priority('O', 'O') == 0
        assert compare_heteroatom_priority('N', 'N') == 0

    @pytest.mark.unit
    def test_sort_heteroatoms_by_priority(self):
        """Test sorting heteroatoms by priority."""
        unsorted = ['N', 'O', 'S']
        sorted_list = sort_heteroatoms_by_priority(unsorted)
        assert sorted_list == ['O', 'S', 'N']

        unsorted2 = ['B', 'Si', 'P', 'N', 'O']
        sorted_list2 = sort_heteroatoms_by_priority(unsorted2)
        assert sorted_list2 == ['O', 'N', 'P', 'Si', 'B']


# =============================================================================
# HW Stem Suffix Tests
# =============================================================================

class TestHWStems:
    """Test HW_STEMS dict and stem functions."""

    @pytest.mark.unit
    @pytest.mark.parametrize("ring_size,expected_stem", [
        (3, 'irene'),
        (4, 'ete'),
        (5, 'ole'),
        (6, 'ine'),
        (7, 'epine'),
        (8, 'ocine'),
        (9, 'onine'),
        (10, 'ecine'),
    ])
    def test_unsaturated_stems(self, ring_size, expected_stem):
        """Test unsaturated HW stems for ring sizes 3-10."""
        assert get_hw_stem(ring_size, is_saturated=False) == expected_stem
        assert get_unsaturated_stem(ring_size) == expected_stem

    @pytest.mark.unit
    @pytest.mark.parametrize("ring_size,expected_stem", [
        (3, 'irane'),
        (4, 'etane'),
        (5, 'olane'),
        (7, 'epane'),
        (8, 'ocane'),
        (9, 'onane'),
        (10, 'ecane'),
    ])
    def test_saturated_stems_standard(self, ring_size, expected_stem):
        """Test saturated HW stems (non-6-membered, default heteroatom)."""
        assert get_hw_stem(ring_size, is_saturated=True) == expected_stem

    @pytest.mark.unit
    def test_saturated_6_membered_with_oxygen(self):
        """6-membered saturated rings with O/S/Se/Te use 'ane'."""
        for heteroatom in ['O', 'S', 'Se', 'Te', 'Bi', 'Hg']:
            assert get_hw_stem(6, is_saturated=True, heteroatom=heteroatom) == 'ane', \
                f"Failed for {heteroatom}"
            assert get_saturated_stem(6, heteroatom=heteroatom) == 'ane'

    @pytest.mark.unit
    def test_saturated_6_membered_with_nitrogen(self):
        """6-membered saturated rings with N/P/Si/etc use 'inane'."""
        for heteroatom in ['N', 'Si', 'Ge', 'Sn', 'Pb', 'B', 'P', 'As', 'Sb']:
            assert get_hw_stem(6, is_saturated=True, heteroatom=heteroatom) == 'inane', \
                f"Failed for {heteroatom}"
            assert get_saturated_stem(6, heteroatom=heteroatom) == 'inane'

    @pytest.mark.unit
    def test_n_saturated_variants(self):
        """Test N-specific saturated variants for 3, 4, 5-membered rings."""
        # 3-membered: aziridine uses 'iridine'
        assert get_hw_stem(3, is_saturated=True, heteroatom='N') == 'iridine'
        # 4-membered: azetidine uses 'etidine'
        assert get_hw_stem(4, is_saturated=True, heteroatom='N') == 'etidine'
        # 5-membered: pyrrolidine uses 'olidine'
        assert get_hw_stem(5, is_saturated=True, heteroatom='N') == 'olidine'

    @pytest.mark.unit
    def test_unsupported_ring_size(self):
        """Unsupported ring sizes should return None."""
        assert get_hw_stem(2, is_saturated=False) is None
        assert get_hw_stem(11, is_saturated=True) is None
        assert get_hw_stem(0, is_saturated=False) is None
        assert get_unsaturated_stem(11) is None
        assert get_saturated_stem(2) is None


class TestHWStemsHelpers:
    """Test HW stem helper functions."""

    @pytest.mark.unit
    def test_is_supported_ring_size(self):
        """Test ring size support detection."""
        for size in [3, 4, 5, 6, 7, 8, 9, 10]:
            assert is_supported_ring_size(size) is True
        for size in [0, 1, 2, 11, 12, 100]:
            assert is_supported_ring_size(size) is False

    @pytest.mark.unit
    def test_get_supported_ring_sizes(self):
        """Test getting list of supported ring sizes."""
        sizes = get_supported_ring_sizes()
        assert sizes == [3, 4, 5, 6, 7, 8, 9, 10]
        assert len(sizes) == 8

    @pytest.mark.unit
    def test_heteroatoms_use_ane_set(self):
        """Test HETEROATOMS_USE_ANE set contents."""
        expected = {'O', 'S', 'Se', 'Te', 'Bi', 'Hg'}
        assert HETEROATOMS_USE_ANE == expected

    @pytest.mark.unit
    def test_heteroatoms_use_inane_set(self):
        """Test HETEROATOMS_USE_INANE set contents."""
        expected = {'N', 'Si', 'Ge', 'Sn', 'Pb', 'B', 'P', 'As', 'Sb'}
        assert HETEROATOMS_USE_INANE == expected


# =============================================================================
# Combined HW Name Building Tests
# =============================================================================

def elide_terminal_a(prefix: str, stem: str) -> str:
    """
    Apply HW 'a' elision: drop terminal 'a' from prefix if stem starts with vowel.

    IUPAC rule: In Hantzsch-Widman names, the terminal 'a' of heteroatom prefixes
    is elided before stems beginning with 'i', 'o', 'e', or 'a'.

    Examples:
        oxa + irane = oxirane (not oxairane)
        oxa + ole = oxole (not oxaole)
        aza + ine = azine (not azaine)
    """
    vowels = {'a', 'e', 'i', 'o', 'u'}
    if prefix.endswith('a') and stem and stem[0] in vowels:
        return prefix[:-1] + stem
    return prefix + stem


class TestHWNameBuilding:
    """Test building complete HW names from prefix + stem with 'a' elision."""

    @pytest.mark.unit
    @pytest.mark.parametrize("heteroatom,ring_size,saturated,expected_name", [
        # 3-membered rings
        ('O', 3, True, 'oxirane'),       # Epoxide
        ('N', 3, True, 'aziridine'),     # Aziridine
        ('S', 3, True, 'thiirane'),      # Thiirane
        # 4-membered rings
        ('O', 4, True, 'oxetane'),       # Oxetane
        ('N', 4, True, 'azetidine'),     # Azetidine
        ('S', 4, True, 'thietane'),      # Thietane
        # 5-membered unsaturated
        ('O', 5, False, 'oxole'),        # Furan systematic (retained: furan)
        ('N', 5, False, 'azole'),        # Pyrrole systematic
        ('S', 5, False, 'thiole'),       # Thiophene systematic
        # 5-membered saturated
        ('O', 5, True, 'oxolane'),       # THF systematic
        ('S', 5, True, 'thiolane'),      # Tetrahydrothiophene systematic
        ('N', 5, True, 'azolidine'),     # Pyrrolidine systematic (note: uses olidine)
        # 6-membered unsaturated
        ('O', 6, False, 'oxine'),        # Pyran systematic
        ('N', 6, False, 'azine'),        # Pyridine systematic
        ('S', 6, False, 'thiine'),       # Thiopyran systematic
        # 6-membered saturated
        ('O', 6, True, 'oxane'),         # Tetrahydropyran systematic
        ('S', 6, True, 'thiane'),        # Thiane
        ('N', 6, True, 'azinane'),       # Piperidine systematic (retained: piperidine)
    ])
    def test_build_hw_name_with_elision(self, heteroatom, ring_size, saturated, expected_name):
        """Test building complete HW names with 'a' elision applied."""
        prefix = get_hw_prefix(heteroatom)
        stem = get_hw_stem(ring_size, is_saturated=saturated, heteroatom=heteroatom)

        # Build name with 'a' elision (HW rule)
        name = elide_terminal_a(prefix, stem)
        assert name == expected_name, f"Expected {expected_name}, got {name}"

    @pytest.mark.unit
    def test_raw_concatenation_without_elision(self):
        """Verify raw concatenation (before elision) produces 'a' + vowel.

        The data module provides raw components. Elision is an assembly rule
        applied when building the final name.
        """
        # Raw concatenation shows the 'a' is present
        assert get_hw_prefix('O') + get_hw_stem(5, False) == 'oxaole'
        assert get_hw_prefix('N') + get_hw_stem(5, False) == 'azaole'
        assert get_hw_prefix('S') + get_hw_stem(5, False) == 'thiaole'

        # With elision, we get the correct HW name
        assert elide_terminal_a('oxa', 'ole') == 'oxole'
        assert elide_terminal_a('aza', 'ole') == 'azole'
        assert elide_terminal_a('thia', 'ole') == 'thiole'

    @pytest.mark.unit
    def test_elision_rule_details(self):
        """Test 'a' elision behavior in detail."""
        # 'a' is elided before vowels
        assert elide_terminal_a('oxa', 'irane') == 'oxirane'  # i
        assert elide_terminal_a('oxa', 'ole') == 'oxole'      # o
        assert elide_terminal_a('oxa', 'etane') == 'oxetane'  # e
        assert elide_terminal_a('oxa', 'ane') == 'oxane'      # a

        # 'a' is NOT elided before consonants (though this is rare in HW)
        assert elide_terminal_a('oxa', 'pane') == 'oxapane'   # hypothetical

        # Prefixes not ending in 'a' are unchanged
        # (Currently all HW prefixes end in 'a', but test the logic)
        assert elide_terminal_a('test', 'ole') == 'testole'

    @pytest.mark.unit
    def test_selenium_tellurium_prefixes(self):
        """Test less common Group 16 heteroatoms."""
        assert get_hw_prefix('Se') == 'selena'
        assert get_hw_prefix('Te') == 'tellura'

        # Build selenole (Se 5-ring unsaturated)
        prefix = get_hw_prefix('Se')
        stem = get_hw_stem(5, False)
        assert prefix + stem == 'selenaole'  # Raw concatenation

    @pytest.mark.unit
    def test_phosphorus_arsenic_prefixes(self):
        """Test Group 15 heteroatoms beyond nitrogen."""
        assert get_hw_prefix('P') == 'phospha'
        assert get_hw_prefix('As') == 'arsa'

        # Phosphole (P 5-ring unsaturated)
        prefix = get_hw_prefix('P')
        stem = get_hw_stem(5, False)
        assert prefix + stem == 'phosphaole'

    @pytest.mark.unit
    def test_silicon_boron_prefixes(self):
        """Test Group 14 and 13 heteroatoms."""
        assert get_hw_prefix('Si') == 'sila'
        assert get_hw_prefix('B') == 'bora'

        # Silane ring (Si 5-ring saturated)
        prefix = get_hw_prefix('Si')
        stem = get_hw_stem(5, True, heteroatom='Si')  # Not N, so uses 'olane'
        assert prefix + stem == 'silaolane'

    @pytest.mark.unit
    def test_larger_ring_names(self):
        """Test HW names for 7-10 membered rings."""
        # Oxepane (O 7-ring saturated)
        assert get_hw_prefix('O') + get_hw_stem(7, True) == 'oxaepane'

        # Azocine (N 8-ring unsaturated)
        assert get_hw_prefix('N') + get_hw_stem(8, False) == 'azaocine'

        # Thionane (S 9-ring saturated)
        assert get_hw_prefix('S') + get_hw_stem(9, True) == 'thiaonane'

        # Oxecane (O 10-ring saturated)
        assert get_hw_prefix('O') + get_hw_stem(10, True) == 'oxaecane'


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.unit
    def test_case_sensitivity(self):
        """Element symbols are case-sensitive."""
        assert get_hw_prefix('O') == 'oxa'
        assert get_hw_prefix('o') is None  # Lowercase not valid
        assert get_hw_prefix('N') == 'aza'
        assert get_hw_prefix('n') is None

    @pytest.mark.unit
    def test_empty_and_none_handling(self):
        """Test handling of empty/None inputs."""
        assert get_hw_prefix('') is None
        assert get_heteroatom_priority('') == 999
        assert get_hw_stem(0, True) is None
        assert get_hw_stem(-1, False) is None

    @pytest.mark.unit
    def test_priority_determinism(self):
        """Priority should be deterministic for multiple lookups."""
        for _ in range(10):
            assert get_heteroatom_priority('O') == 1
            assert get_heteroatom_priority('N') == 5

    @pytest.mark.unit
    def test_stem_determinism(self):
        """Stem lookup should be deterministic for multiple lookups."""
        for _ in range(10):
            assert get_hw_stem(5, False) == 'ole'
            assert get_hw_stem(6, True, 'O') == 'ane'
            assert get_hw_stem(6, True, 'N') == 'inane'
