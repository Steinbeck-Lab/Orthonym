"""
Unit tests for locant validation module.

Tests cover four validation domains:
1. Suffix locant capacity (locants within parent structure bounds)
2. Suffix-prefix collision detection (overlapping atom positions)
3. Stereo locant validation (filtering invalid stereodescriptor locants)
4. Multiplier-locant count consistency

Reference: IUPAC 2013 Blue Book,,
"""

import pytest

from orthonym.rules.locant_validation import (
    detect_locant_collisions,
    reconcile_multiplier_count,
    validate_stereo_locants,
    validate_suffix_locants,
)


# ---------------------------------------------------------------------------
# TestSuffixLocantCapacity
# ---------------------------------------------------------------------------


class TestSuffixLocantCapacity:
    """Validate suffix locants against parent structure size."""

    def test_locants_within_ring_capacity_unchanged(self):
        """Suffix locants within a 6-membered ring pass through unchanged."""
        locants, count = validate_suffix_locants([1, 3], parent_size=6, count=2)
        assert locants == [1, 3]
        assert count == 2

    def test_locants_exceeding_ring_size_filtered(self):
        """Locants exceeding the ring size are removed."""
        locants, count = validate_suffix_locants([1, 3, 8], parent_size=6, count=3)
        assert locants == [1, 3]
        assert count == 2

    def test_locant_zero_always_filtered(self):
        """Locant 0 is never a valid IUPAC locant (1-indexed)."""
        locants, count = validate_suffix_locants([0, 2, 4], parent_size=6, count=3)
        assert locants == [2, 4]
        assert count == 2

    def test_empty_locants_remain_empty(self):
        """Empty input produces empty output."""
        locants, count = validate_suffix_locants([], parent_size=6, count=0)
        assert locants == []
        assert count == 0

    def test_multiplier_count_adjusted_to_filtered(self):
        """Count adjusts to match remaining locants after filtering."""
        locants, count = validate_suffix_locants([1, 3, 8], parent_size=6, count=3)
        assert count == 2  # was 3, now matches [1, 3]

    def test_single_locant_within_range(self):
        """A single valid locant passes."""
        locants, count = validate_suffix_locants([2], parent_size=3, count=1)
        assert locants == [2]
        assert count == 1

    def test_all_locants_exceed_parent(self):
        """All locants above parent size -> empty list."""
        locants, count = validate_suffix_locants([7, 8, 9], parent_size=6, count=3)
        assert locants == []
        assert count == 0

    def test_chain_parent_validation(self):
        """Chain length used as parent_size; valid locants kept."""
        locants, count = validate_suffix_locants([1, 3, 5], parent_size=5, count=3)
        assert locants == [1, 3, 5]
        assert count == 3

    def test_chain_parent_out_of_range(self):
        """Chain length 4 filters locant 5."""
        locants, count = validate_suffix_locants([1, 3, 5], parent_size=4, count=3)
        assert locants == [1, 3]
        assert count == 2

    def test_parent_size_zero_edge_case(self):
        """parent_size=0 means no valid locants possible."""
        locants, count = validate_suffix_locants([1, 2], parent_size=0, count=2)
        assert locants == []
        assert count == 0


# ---------------------------------------------------------------------------
# TestSuffixPrefixCollision
# ---------------------------------------------------------------------------


class TestSuffixPrefixCollision:
    """Detect collisions between suffix and prefix locant positions on rings."""

    def test_no_collision_disjoint(self):
        """Suffix on 1, prefix on 3 -- no collision."""
        collisions = detect_locant_collisions(
            suffix_locants=[1],
            prefix_locant_groups=[[3]],
            parent_type="ring",
            parent_size=6,
        )
        assert collisions == []

    def test_direct_collision_on_ring(self):
        """Suffix and prefix both on position 3 -> collision detected."""
        collisions = detect_locant_collisions(
            suffix_locants=[3],
            prefix_locant_groups=[[3]],
            parent_type="ring",
            parent_size=6,
        )
        assert len(collisions) == 1
        assert collisions[0] == (0, 3)  # (prefix_group_index, colliding_locant)

    def test_multiple_suffix_single_prefix_collision(self):
        """Suffix [1,4], prefix [[4]] -> collision at 4."""
        collisions = detect_locant_collisions(
            suffix_locants=[1, 4],
            prefix_locant_groups=[[4]],
            parent_type="ring",
            parent_size=6,
        )
        assert len(collisions) == 1
        assert collisions[0] == (0, 4)

    def test_chain_skips_collision_check(self):
        """Chain parent type returns empty list (no ring collision semantics)."""
        collisions = detect_locant_collisions(
            suffix_locants=[1],
            prefix_locant_groups=[[3]],
            parent_type="chain",
            parent_size=5,
        )
        assert collisions == []

    def test_multiple_prefixes_one_colliding(self):
        """Suffix [2], prefixes [[2],[5]] -> collision on group 0."""
        collisions = detect_locant_collisions(
            suffix_locants=[2],
            prefix_locant_groups=[[2], [5]],
            parent_type="ring",
            parent_size=6,
        )
        assert len(collisions) == 1
        assert collisions[0] == (0, 2)

    def test_collision_returns_all_colliding_pairs(self):
        """Collision detection returns a list of collision tuples."""
        collisions = detect_locant_collisions(
            suffix_locants=[3],
            prefix_locant_groups=[[3]],
            parent_type="ring",
            parent_size=6,
        )
        # Each element is (prefix_group_index, colliding_locant)
        assert all(isinstance(c, tuple) and len(c) == 2 for c in collisions)

    def test_no_collision_named_ring_disjoint(self):
        """Named ring with disjoint suffix/prefix positions has no collision."""
        collisions = detect_locant_collisions(
            suffix_locants=[1],
            prefix_locant_groups=[[4]],
            parent_type="ring",
            parent_size=6,
        )
        assert collisions == []

    def test_unnamed_ring_collision_detected(self):
        """Unnamed ring with matching locants detects collision."""
        collisions = detect_locant_collisions(
            suffix_locants=[1],
            prefix_locant_groups=[[1]],
            parent_type="ring",
            parent_size=6,
        )
        assert len(collisions) == 1
        assert collisions[0] == (0, 1)

    def test_disjoint_sets_no_collision(self):
        """Completely disjoint locant sets produce no collision."""
        collisions = detect_locant_collisions(
            suffix_locants=[1, 3, 5],
            prefix_locant_groups=[[2], [4], [6]],
            parent_type="ring",
            parent_size=6,
        )
        assert collisions == []

    def test_empty_suffix_locants_no_collision(self):
        """Empty suffix -> no collision possible."""
        collisions = detect_locant_collisions(
            suffix_locants=[],
            prefix_locant_groups=[[3]],
            parent_type="ring",
            parent_size=6,
        )
        assert collisions == []

    def test_empty_prefix_locants_no_collision(self):
        """Empty prefix groups -> no collision possible."""
        collisions = detect_locant_collisions(
            suffix_locants=[1],
            prefix_locant_groups=[],
            parent_type="ring",
            parent_size=6,
        )
        assert collisions == []

    def test_multiple_collisions_both_detected(self):
        """Suffix [1,3], prefixes [[1],[3]] -> both collide."""
        collisions = detect_locant_collisions(
            suffix_locants=[1, 3],
            prefix_locant_groups=[[1], [3]],
            parent_type="ring",
            parent_size=6,
        )
        assert len(collisions) == 2
        collision_set = {(g, loc) for g, loc in collisions}
        assert (0, 1) in collision_set
        assert (1, 3) in collision_set


# ---------------------------------------------------------------------------
# TestStereoLocantValidation
# ---------------------------------------------------------------------------


class TestStereoLocantValidation:
    """Validate stereo descriptor locants against parent size."""

    def test_valid_stereo_locants_unchanged(self):
        """Valid R/S stereo locants within ring pass through."""
        result = validate_stereo_locants(
            [(2, "R"), (4, "S")], parent_size=6
        )
        assert result == [(2, "R"), (4, "S")]

    def test_locant_zero_filtered(self):
        """Locant 0 is invalid for stereo descriptors."""
        result = validate_stereo_locants(
            [(0, "R"), (2, "S")], parent_size=6
        )
        assert result == [(2, "S")]

    def test_locant_exceeding_parent_filtered(self):
        """Locant 7 exceeds parent_size=5."""
        result = validate_stereo_locants(
            [(2, "R"), (7, "S")], parent_size=5
        )
        assert result == [(2, "R")]

    def test_negative_locant_filtered(self):
        """Negative locant is never valid."""
        result = validate_stereo_locants(
            [(-1, "R")], parent_size=6
        )
        assert result == []

    def test_string_locant_fused_system(self):
        """String locant '4a' is valid for fused ring systems."""
        result = validate_stereo_locants(
            [("4a", "R")], parent_size=10
        )
        assert result == [("4a", "R")]

    def test_ez_descriptors_valid_locant(self):
        """E/Z double bond descriptors with valid locant pass."""
        result = validate_stereo_locants(
            [(2, "E")], parent_size=6
        )
        assert result == [(2, "E")]

    def test_all_stereo_locants_invalid(self):
        """All locants invalid -> empty list."""
        result = validate_stereo_locants(
            [(0, "R"), (-1, "S"), (99, "R")], parent_size=6
        )
        assert result == []

    def test_mixed_valid_invalid(self):
        """Only valid locants survive filtering."""
        result = validate_stereo_locants(
            [(0, "R"), (3, "S"), (99, "R")], parent_size=6
        )
        assert result == [(3, "S")]


# ---------------------------------------------------------------------------
# TestMultiplierLocantConsistency
# ---------------------------------------------------------------------------


class TestMultiplierLocantConsistency:
    """Ensure multiplier prefix count matches locant list length."""

    def test_count_matches_locants(self):
        """Count already matches -> no change."""
        result = reconcile_multiplier_count(count=2, locants=[1, 3])
        assert result == 2

    def test_count_exceeds_locants(self):
        """Count 3 but only 2 locants -> adjust to 2."""
        result = reconcile_multiplier_count(count=3, locants=[1, 3])
        assert result == 2

    def test_count_less_than_locants(self):
        """Count 1 but 2 locants -> adjust to 2."""
        result = reconcile_multiplier_count(count=1, locants=[1, 3])
        assert result == 2

    def test_zero_count_with_nonempty_locants(self):
        """Zero count with locants present -> adjust to len(locants)."""
        result = reconcile_multiplier_count(count=0, locants=[1, 3])
        assert result == 2

    def test_empty_locants_preserves_count(self):
        """Empty locants with count > 0 -> preserve original count."""
        result = reconcile_multiplier_count(count=3, locants=[])
        assert result == 3
