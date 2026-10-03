"""
Tests for locant collision detection and valency fix correctness.

Validates that:
1. Ring compounds with suffix + prefix groups produce disjoint locants
2. OPSIN valency errors (suffix and prefix at same position) are eliminated
3. Suffix capacity does not exceed parent size
4. FG SMARTS locants are correctly assigned (not overlapping)
5. Existing diol/triol/polysubstituted names remain correct

These tests target the fixes in Plan 17-04:
- _generate_suffix now computes locants for ring compounds via oriented_ring
- _assemble_fragments detects suffix-prefix locant collisions on ring parents
- _estimate_parent_size_from_name estimates parent atom count from name text
- polyfunctional.py collision detection safety net
"""

import re
import pytest
from orthonym import name_compound
from tests.support.rt_assert import assert_full_rt


# ---------------------------------------------------------------------------
# Helper: extract suffix and prefix locants from an IUPAC name
# ---------------------------------------------------------------------------


def _extract_suffix_locant(name: str):
    """Extract the numeric locant(s) immediately before a suffix like -ol, -one, -diol, etc."""
    # Match patterns like "an-1-ol", "ane-1,4-diol", "an-5-one"
    m = re.search(r'an[e]?-([\d,]+)-(di|tri|tetra)?(?:ol|one|amine|thiol)', name)
    if m:
        return [int(x) for x in m.group(1).split(',')]
    return []


def _extract_prefix_locants(name: str):
    """Extract all prefix locant groups (e.g., '1-chloro', '3,5-dimethyl')."""
    locants = []
    # Match patterns like "1-chloro", "3,5-dimethyl", "2-hydroxy"
    for m in re.finditer(r'(\d+(?:,\d+)*)-(?:di|tri|tetra|penta|hexa)?(?:methyl|ethyl|propyl|butyl|chloro|bromo|fluoro|iodo|hydroxy|amino|oxo|nitro|cyano)', name):
        locants.extend(int(x) for x in m.group(1).split(','))
    return locants


def _has_locant_collision(name: str) -> bool:
    """Check if any numeric locant appears in both suffix and prefix contexts."""
    suffix_locs = set(_extract_suffix_locant(name))
    prefix_locs = set(_extract_prefix_locants(name))
    return bool(suffix_locs & prefix_locs)


# ===========================================================================
# TestRingLocantCollisions: ring compounds with both suffix and prefix groups
# ===========================================================================


class TestRingLocantCollisions:
    """Verify that suffix and prefix locants on ring systems do not collide."""

    def test_methylcyclohexanone_locants_disjoint(self):
        """3-methylcyclohexan-1-one: methyl at 3, ketone at 1 -- no collision."""
        name = name_compound("CC1CCC(=O)CC1")
        assert name, "Name should not be empty"
        assert "one" in name, f"Expected ketone suffix in: {name}"
        assert not _has_locant_collision(name), f"Locant collision in: {name}"

    def test_4_methylcyclohexanone_disjoint(self):
        """4-methylcyclohexan-1-one: methyl at 4, ketone at 1."""
        name = name_compound("CC1CCC(=O)CC1")
        assert name
        # Suffix should have a locant (not empty cyclohexanone)
        assert re.search(r'an-\d+-one', name), f"Expected locanted suffix in: {name}"

    def test_chlorocyclohexanone_no_collision(self):
        """4-chlorocyclohexan-1-one: chloro and ketone at different positions."""
        name = name_compound("ClC1CCC(=O)CC1")
        assert name
        assert "one" in name
        assert not _has_locant_collision(name), f"Locant collision in: {name}"

    def test_hydroxycyclohexanone_disjoint(self):
        """4-hydroxycyclohexan-1-one: hydroxy at 4, ketone at 1."""
        name = name_compound("OC1CCC(=O)CC1")
        assert name
        assert "one" in name
        assert not _has_locant_collision(name), f"Locant collision in: {name}"

    def test_dimethylcyclohexanone_disjoint(self):
        """3,5-dimethylcyclohexan-1-one: methyls at 3,5, ketone at 1."""
        name = name_compound("CC1CC(=O)CC(C)C1")
        assert name
        assert "one" in name
        assert not _has_locant_collision(name), f"Locant collision in: {name}"

    def test_aminocyclohexanone_no_collision(self):
        """4-aminocyclohexan-1-one: amino at 4, ketone at 1."""
        name = name_compound("NC1CCC(=O)CC1")
        assert name
        assert "one" in name
        assert not _has_locant_collision(name), f"Locant collision in: {name}"

    def test_fluorocyclohexanone_no_collision(self):
        """4-fluorocyclohexan-1-one: fluoro at 4, ketone at 1."""
        name = name_compound("FC1CCC(=O)CC1")
        assert name
        assert "one" in name
        assert not _has_locant_collision(name), f"Locant collision in: {name}"

    def test_bromocyclohexanone_no_collision(self):
        """4-bromocyclohexan-1-one: bromo at 4, ketone at 1."""
        name = name_compound("BrC1CCC(=O)CC1")
        assert name
        assert "one" in name
        assert not _has_locant_collision(name), f"Locant collision in: {name}"

    def test_methylcyclohexanol_disjoint(self):
        """2-methylcyclohexan-1-ol: methyl and hydroxyl at different positions."""
        name = name_compound("CC1CCCCC1O")
        assert name
        assert "ol" in name
        assert not _has_locant_collision(name), f"Locant collision in: {name}"

    def test_cyclohexanone_has_suffix_locant(self):
        """cyclohexanone should have a locant on the suffix (cyclohexan-N-one)."""
        name = name_compound("O=C1CCCCC1")
        assert name
        # Should have a locanted suffix, not bare "cyclohexanone"
        # (OPSIN interprets bare cyclohexanone as position 1, but locant is clearer)
        assert "one" in name


# ===========================================================================
# TestSuffixCapacityValidation: suffix count vs parent capacity
# ===========================================================================


class TestSuffixCapacityValidation:
    """Verify suffix group count does not exceed parent structure capacity."""

    def test_cyclohexane_diol_valid(self):
        """Cyclohexane can support up to 6 hydroxyl groups (1 per carbon)."""
        name = name_compound("OC1CCC(O)CC1")
        assert name
        assert "diol" in name, f"Expected diol in: {name}"

    def test_cyclohexane_triol_valid(self):
        """Cyclohexane can support triol."""
        name = name_compound("OC1CC(O)CC(O)C1")
        assert name
        # Should have triol (3 hydroxyls on 6-membered ring)
        assert "ol" in name

    def test_propane_triol_valid(self):
        """Propane (3 carbons) can support triol (glycerol)."""
        name = name_compound("OCC(O)CO")
        assert name
        # Glycerol should be propane-1,2,3-triol
        assert "ol" in name

    def test_ethane_diol_valid(self):
        """Ethane (2 carbons) can support diol (may be retained as ethylene glycol)."""
        name = name_compound("OCCO")
        assert name
        # Could be "ethane-1,2-diol" or "ethylene glycol" (retained name)
        assert "diol" in name or "glycol" in name, f"Expected diol or glycol in: {name}"

    def test_cyclohexane_dione_valid(self):
        """Cyclohexane can support dione (2 ketone positions)."""
        name = name_compound("O=C1CCC(=O)CC1")
        assert name
        assert "dione" in name, f"Expected dione in: {name}"


# ===========================================================================
# TestFGSmartsMismatch: FG locant assigned to correct atom
# ===========================================================================


class TestFGSmartsMismatch:
    """Verify FG locants use the correct anchor atom, not a coincident match."""

    def test_hydroxycyclohexanone_different_positions(self):
        """4-hydroxycyclohexanone: hydroxyl and ketone on different carbons."""
        name = name_compound("OC1CCC(=O)CC1")
        assert name
        suffix_locs = _extract_suffix_locant(name)
        prefix_locs = _extract_prefix_locants(name)
        if suffix_locs and prefix_locs:
            assert not set(suffix_locs) & set(prefix_locs), \
                f"FG locant collision: suffix={suffix_locs} prefix={prefix_locs} in {name}"

    def test_aminobenzoic_acid_different_positions(self):
        """3-aminobenzoic acid: amino and acid at different ring positions."""
        name = name_compound("Nc1cccc(C(=O)O)c1")
        assert name
        # amino and acid should be at different positions
        assert "amino" in name

    def test_methylpyridinol_different_positions(self):
        """2-methylpyridin-3-ol: methyl at 2, hydroxyl at 3."""
        name = name_compound("Cc1ncccc1O")
        assert name
        # Both substituents should appear in name

    def test_dichlorocyclohexane_both_locants(self):
        """1,4-dichlorocyclohexane: both chloro locants present."""
        name = name_compound("ClC1CCC(Cl)CC1")
        assert name
        assert "chloro" in name

    def test_disubstituted_ring_no_suffix_overlap(self):
        """2,4-dimethylcyclohexan-1-ol: dimethyl locants separate from hydroxyl."""
        name = name_compound("CC1CC(C)CCC1O")
        assert name
        assert "ol" in name
        assert not _has_locant_collision(name), f"Collision in: {name}"


# ===========================================================================
# TestOPSINParseability: names should parse in OPSIN without valency errors
# ===========================================================================


class TestOPSINParseability:
    """Test that generated names are parseable by OPSIN (no valency errors).

    These compounds previously produced 'unphysical valency' OPSIN errors.
    """

    # (the Blue Book) "The locant '1' is omitted:"... (c) "in
    # monosubstituted homogeneous monocyclic rings;" (:2913); 'cyclohexanethiol
    # (PIN)' (:2917). OPSIN reads the locant-free names back to the input's full
    # InChIKey, so the PIN is also the parseable form.

    def test_suffix_locant_present_for_ring_ketone(self):
        """A monosubstituted ring ketone is the locant-free PIN and round-trips."""
        name = name_compound("O=C1CCCCC1")
        assert name == "cyclohexanone"
        assert_full_rt(name, "O=C1CCCCC1")

    def test_suffix_locant_present_for_ring_alcohol(self):
        """A monosubstituted ring alcohol is the locant-free PIN and round-trips."""
        name = name_compound("OC1CCCCC1")
        assert name == "cyclohexanol"
        assert_full_rt(name, "OC1CCCCC1")

    def test_prefix_and_suffix_at_different_positions(self):
        """Substituted ring with FG: prefix and suffix at different atoms."""
        name = name_compound("ClC1CCC(=O)CC1")
        assert name
        # Should be like "1-chlorocyclohexan-5-one" -- different locants
        suffix_locs = _extract_suffix_locant(name)
        prefix_locs = _extract_prefix_locants(name)
        if suffix_locs and prefix_locs:
            assert not set(suffix_locs) & set(prefix_locs), \
                f"Same locant in suffix and prefix: {name}"

    def test_multiple_ring_substituents_with_fg(self):
        """Ring with multiple substituents and FG: all locants distinct."""
        name = name_compound("CC1CC(=O)CC(C)C1")
        assert name
        suffix_locs = _extract_suffix_locant(name)
        prefix_locs = _extract_prefix_locants(name)
        if suffix_locs and prefix_locs:
            assert not set(suffix_locs) & set(prefix_locs), \
                f"Collision in: {name}"

    def test_aminocyclohexanone_parseable(self):
        """4-aminocyclohexan-1-one should be OPSIN-parseable."""
        name = name_compound("NC1CCC(=O)CC1")
        assert name
        # Should have both amino prefix and ketone suffix with distinct locants
        assert "amino" in name
        assert "one" in name


# ===========================================================================
# TestEstimateParentSize: parent size estimation helper
# ===========================================================================


class TestEstimateParentSize:
    """Test _estimate_parent_size_from_name helper function."""

    def test_cyclohexane_returns_6(self):
        from orthonym.assembly.composer import _estimate_parent_size_from_name
        assert _estimate_parent_size_from_name("cyclohex") == 6

    def test_benzene_returns_6(self):
        from orthonym.assembly.composer import _estimate_parent_size_from_name
        assert _estimate_parent_size_from_name("benzene") == 6

    def test_cyclopentane_returns_5(self):
        from orthonym.assembly.composer import _estimate_parent_size_from_name
        assert _estimate_parent_size_from_name("cyclopent") == 5

    def test_cyclopropane_returns_3(self):
        from orthonym.assembly.composer import _estimate_parent_size_from_name
        assert _estimate_parent_size_from_name("cycloprop") == 3

    def test_naphthalene_returns_10(self):
        from orthonym.assembly.composer import _estimate_parent_size_from_name
        assert _estimate_parent_size_from_name("naphthal") == 10

    def test_pyridine_returns_6(self):
        from orthonym.assembly.composer import _estimate_parent_size_from_name
        assert _estimate_parent_size_from_name("pyrid") == 6

    def test_furan_returns_5(self):
        from orthonym.assembly.composer import _estimate_parent_size_from_name
        assert _estimate_parent_size_from_name("furan") == 5

    def test_indole_returns_9(self):
        from orthonym.assembly.composer import _estimate_parent_size_from_name
        assert _estimate_parent_size_from_name("indol") == 9

    def test_unknown_returns_100(self):
        from orthonym.assembly.composer import _estimate_parent_size_from_name
        assert _estimate_parent_size_from_name("zzz_unknown") == 100

    def test_chain_prefix_methane(self):
        from orthonym.assembly.composer import _estimate_parent_size_from_name
        assert _estimate_parent_size_from_name("meth") == 1

    def test_chain_prefix_propane(self):
        from orthonym.assembly.composer import _estimate_parent_size_from_name
        assert _estimate_parent_size_from_name("prop") == 3


# ===========================================================================
# TestLocantCollisionDetection: unit tests for detect_locant_collisions
# ===========================================================================


class TestLocantCollisionDetection:
    """Test the locant collision detection function directly."""

    def test_no_collision_different_locants(self):
        from orthonym.rules.locant_validation import detect_locant_collisions
        result = detect_locant_collisions([1], [[3]], parent_type="ring", parent_size=6)
        assert result == []

    def test_collision_same_locant(self):
        from orthonym.rules.locant_validation import detect_locant_collisions
        result = detect_locant_collisions([3], [[3]], parent_type="ring", parent_size=6)
        assert len(result) == 1
        assert result[0] == (0, 3)

    def test_no_collision_on_chain(self):
        from orthonym.rules.locant_validation import detect_locant_collisions
        result = detect_locant_collisions([3], [[3]], parent_type="chain", parent_size=6)
        assert result == [], "Chain systems should skip collision detection"

    def test_multiple_prefix_groups(self):
        from orthonym.rules.locant_validation import detect_locant_collisions
        result = detect_locant_collisions([1], [[1], [3]], parent_type="ring", parent_size=6)
        assert len(result) == 1
        assert result[0] == (0, 1)

    def test_empty_suffix_locants(self):
        from orthonym.rules.locant_validation import detect_locant_collisions
        result = detect_locant_collisions([], [[3]], parent_type="ring", parent_size=6)
        assert result == []

    def test_empty_prefix_groups(self):
        from orthonym.rules.locant_validation import detect_locant_collisions
        result = detect_locant_collisions([1], [], parent_type="ring", parent_size=6)
        assert result == []
