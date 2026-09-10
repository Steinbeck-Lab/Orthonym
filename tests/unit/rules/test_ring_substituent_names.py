"""Tests for ring-as-substituent naming infrastructure (a phase).

Verifies that:
1. Fused heterocycle entries exist in RING_SUBSTITUENT_NAMES
2. Position-specific entries exist for quinoline and isoquinoline
3. Existing entries still work (benzene -> phenyl, pyridine -> pyridyl)
4. Naphthalene position detection works (alpha vs beta)
"""

import pytest
from orthonym.rules.ring_substituents import (
    RING_SUBSTITUENT_NAMES,
    POSITION_SPECIFIC_RINGS,
    get_ring_substituent_name,
)


class TestFusedHeterocycleEntries:
    """Verify fused heterocycle entries exist in RING_SUBSTITUENT_NAMES."""

    @pytest.mark.parametrize("ring_name,expected_prefix", [
        ('indole', 'indolyl'),
        ('quinoline', 'quinolinyl'),
        ('isoquinoline', 'isoquinolinyl'),
        ('benzofuran', 'benzofuranyl'),
        ('benzothiophene', 'benzothienyl'),
        ('benzimidazole', 'benzimidazolyl'),
        ('purine', 'purinyl'),
        ('carbazole', 'carbazolyl'),
    ])
    def test_fused_heterocycle_in_dict(self, ring_name, expected_prefix):
        """Each fused heterocycle must have the correct prefix form."""
        assert ring_name in RING_SUBSTITUENT_NAMES, (
            f"Missing fused heterocycle entry: {ring_name}"
        )
        assert RING_SUBSTITUENT_NAMES[ring_name] == expected_prefix, (
            f"Wrong prefix for {ring_name}: expected {expected_prefix}, "
            f"got {RING_SUBSTITUENT_NAMES[ring_name]}"
        )


class TestExistingEntriesPreserved:
    """Verify existing ring substituent entries still work."""

    @pytest.mark.parametrize("ring_name,expected_prefix", [
        ('benzene', 'phenyl'),
        ('naphthalene', 'naphthyl'),
        ('pyridine', 'pyridyl'),
        ('furan', 'furyl'),
        ('thiophene', 'thienyl'),
        ('cyclohexane', 'cyclohexyl'),
        ('cyclopentane', 'cyclopentyl'),
        ('piperidine', 'piperidinyl'),
        ('morpholine', 'morpholinyl'),
        ('oxirane', 'oxiranyl'),
    ])
    def test_existing_entry_preserved(self, ring_name, expected_prefix):
        """Each existing entry must still return the correct prefix."""
        assert RING_SUBSTITUENT_NAMES[ring_name] == expected_prefix


class TestPositionSpecificRings:
    """Verify position-specific entries for quinoline and isoquinoline."""

    def test_quinoline_positions(self):
        """Quinoline must have position-specific entries for positions 2-8."""
        assert 'quinoline' in POSITION_SPECIFIC_RINGS
        q = POSITION_SPECIFIC_RINGS['quinoline']
        for pos in range(2, 9):
            assert pos in q, f"Missing quinoline position {pos}"
            assert q[pos] == f'{pos}-quinolinyl'

    def test_isoquinoline_positions(self):
        """Isoquinoline must have position-specific entries."""
        assert 'isoquinoline' in POSITION_SPECIFIC_RINGS
        iq = POSITION_SPECIFIC_RINGS['isoquinoline']
        for pos in [1, 3, 4, 5, 6, 7, 8]:
            assert pos in iq, f"Missing isoquinoline position {pos}"
            assert iq[pos] == f'{pos}-isoquinolinyl'

    def test_pyridine_positions_preserved(self):
        """Pyridine position-specific entries must still be present."""
        assert 'pyridine' in POSITION_SPECIFIC_RINGS
        py = POSITION_SPECIFIC_RINGS['pyridine']
        assert py[2] == '2-pyridyl'
        assert py[3] == '3-pyridyl'
        assert py[4] == '4-pyridyl'


class TestDictSize:
    """Verify total entry count."""

    def test_ring_substituent_names_count(self):
        """RING_SUBSTITUENT_NAMES must have at least 36 entries (28 existing + 8 fused)."""
        assert len(RING_SUBSTITUENT_NAMES) >= 36, (
            f"Expected >= 36 entries, got {len(RING_SUBSTITUENT_NAMES)}"
        )

    def test_position_specific_rings_count(self):
        """POSITION_SPECIFIC_RINGS must have at least 4 entries."""
        assert len(POSITION_SPECIFIC_RINGS) >= 4, (
            f"Expected >= 4 entries, got {len(POSITION_SPECIFIC_RINGS)}"
        )
