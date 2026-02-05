"""
Tests for stereo locant filtering and multiplier/locant consistency.

Plan 17-05: Validates that:
- Stereo descriptors with locant 0 or exceeding parent size are filtered
- Multiplier prefix always matches the number of locants
- Letter-suffix locants (4a, 8a) are validated correctly for fused systems
"""

import pytest
from orthonym import name_compound
from orthonym.rules.locant_validation import (
    validate_stereo_locants,
    reconcile_multiplier_count,
)


# ============================================================================
# TestStereoLocantFiltering - Unit tests for validate_stereo_locants
# ============================================================================

class TestStereoLocantFiltering:
    """Test the validate_stereo_locants function directly."""

    def test_valid_locants_preserved(self):
        """Valid stereo locants within parent size should be kept."""
        descriptors = [(2, 'R'), (4, 'S')]
        result = validate_stereo_locants(descriptors, parent_size=6)
        assert result == [(2, 'R'), (4, 'S')]

    def test_locant_zero_filtered(self):
        """Locant 0 is not valid IUPAC (1-indexed) and must be removed."""
        descriptors = [(0, 'R'), (3, 'S')]
        result = validate_stereo_locants(descriptors, parent_size=6)
        assert result == [(3, 'S')]

    def test_locant_exceeding_parent_filtered(self):
        """Locants larger than parent size must be removed."""
        descriptors = [(2, 'R'), (99, 'S')]
        result = validate_stereo_locants(descriptors, parent_size=6)
        assert result == [(2, 'R')]

    def test_all_invalid_returns_empty(self):
        """When all locants are invalid, return empty list."""
        descriptors = [(0, 'R'), (15, 'S')]
        result = validate_stereo_locants(descriptors, parent_size=6)
        assert result == []

    def test_empty_descriptors_returns_empty(self):
        """Empty input returns empty output."""
        result = validate_stereo_locants([], parent_size=6)
        assert result == []

    def test_fused_ring_letter_suffix_locant_valid(self):
        """Fused ring locants like '4a' should validate if base number <= parent_size."""
        descriptors = [('4a', 'S'), ('8a', 'R')]
        result = validate_stereo_locants(descriptors, parent_size=10)
        assert result == [('4a', 'S'), ('8a', 'R')]

    def test_fused_ring_letter_suffix_locant_invalid(self):
        """Fused ring locant '12a' should be removed if parent_size is 6."""
        descriptors = [('4a', 'S'), ('12a', 'R')]
        result = validate_stereo_locants(descriptors, parent_size=6)
        assert result == [('4a', 'S')]

    def test_mixed_valid_invalid(self):
        """Multiple stereocenters, some valid and some out of range."""
        descriptors = [(1, 'R'), (3, 'S'), (7, 'R'), (10, 'S')]
        result = validate_stereo_locants(descriptors, parent_size=6)
        assert result == [(1, 'R'), (3, 'S')]

    def test_negative_locant_filtered(self):
        """Negative locants should be filtered out."""
        descriptors = [(-1, 'R'), (2, 'S')]
        result = validate_stereo_locants(descriptors, parent_size=6)
        assert result == [(2, 'S')]

    def test_ez_bond_locant_validated(self):
        """E/Z descriptors also have their locants validated."""
        descriptors = [(2, 'E'), (0, 'Z'), (5, 'R')]
        result = validate_stereo_locants(descriptors, parent_size=6)
        assert result == [(2, 'E'), (5, 'R')]

    def test_large_ring_all_valid(self):
        """Cyclodecane: locants up to 10 are valid."""
        descriptors = [(1, 'R'), (5, 'S'), (10, 'R')]
        result = validate_stereo_locants(descriptors, parent_size=10)
        assert result == [(1, 'R'), (5, 'S'), (10, 'R')]


# ============================================================================
# TestMultiplierLocantMatch - Unit tests for reconcile_multiplier_count
# ============================================================================

class TestMultiplierLocantMatch:
    """Test the reconcile_multiplier_count function directly."""

    def test_count_matches_locants(self):
        """When count matches locant count, no change."""
        result = reconcile_multiplier_count(count=2, locants=[1, 3])
        assert result == 2

    def test_count_higher_than_locants(self):
        """When count exceeds locant count, reduce to locant count."""
        result = reconcile_multiplier_count(count=3, locants=[1, 3])
        assert result == 2

    def test_count_lower_than_locants(self):
        """When count is less than locants, increase to locant count."""
        result = reconcile_multiplier_count(count=1, locants=[1, 3, 5])
        assert result == 3

    def test_single_locant_no_multiplier(self):
        """Single locant yields count of 1."""
        result = reconcile_multiplier_count(count=1, locants=[2])
        assert result == 1

    def test_empty_locants_preserves_count(self):
        """With empty locants, original count is preserved."""
        result = reconcile_multiplier_count(count=2, locants=[])
        assert result == 2

    def test_zero_count_with_locants(self):
        """Even if count was 0, locants override."""
        result = reconcile_multiplier_count(count=0, locants=[1, 2])
        assert result == 2

    def test_triple_locants(self):
        """Three locants yields count 3 (tri)."""
        result = reconcile_multiplier_count(count=2, locants=[1, 3, 5])
        assert result == 3

    def test_four_locants(self):
        """Four locants yields count 4 (tetra)."""
        result = reconcile_multiplier_count(count=3, locants=[1, 2, 3, 4])
        assert result == 4


# ============================================================================
# TestStereoLocantIntegration - End-to-end tests via name_compound
# ============================================================================

class TestStereoLocantIntegration:
    """Test that stereo locant filtering works end-to-end in naming."""

    def test_butan2ol_stereo_valid(self):
        """(2R)-butan-2-ol: stereo locant 2 is within parent size 4."""
        result = name_compound('C[C@@H](O)CC')
        assert result  # should produce a name
        # Stereo prefix should contain locant 2
        assert '2' in result
        # Should NOT contain (0R) or (0S)
        assert '(0R)' not in result
        assert '(0S)' not in result

    def test_no_stereo_locant_zero(self):
        """No name should ever contain (0R), (0S), (0E), or (0Z)."""
        # Test a variety of chiral molecules
        test_smiles = [
            'C[C@@H](O)CC',      # butan-2-ol
            'O[C@@H](F)Cl',      # halogenated methanol
            'C[C@@H](N)CC(=O)O', # amino acid
        ]
        for smi in test_smiles:
            result = name_compound(smi)
            if result:
                assert '(0R)' not in result, f"SMILES {smi} produced (0R) in: {result}"
                assert '(0S)' not in result, f"SMILES {smi} produced (0S) in: {result}"
                assert '(0E)' not in result, f"SMILES {smi} produced (0E) in: {result}"
                assert '(0Z)' not in result, f"SMILES {smi} produced (0Z) in: {result}"

    def test_cyclohexane_stereo_within_ring_size(self):
        """Stereo locants should be within ring size for cyclic compounds."""
        # cis-1,3-dimethylcyclohexane
        result = name_compound('C[C@H]1CCC[C@@H](C)C1')
        if result:
            # If stereo descriptors present, locants should be <= 6
            # There should be no locant > 6 in stereo prefix
            import re
            stereo_match = re.match(r'\(([^)]+)\)-', result)
            if stereo_match:
                stereo_parts = stereo_match.group(1).split(',')
                for part in stereo_parts:
                    # Extract numeric locant
                    num_str = ''.join(c for c in part if c.isdigit())
                    if num_str:
                        locant = int(num_str)
                        assert 1 <= locant <= 6, f"Stereo locant {locant} outside ring size 6"

    def test_propane_diol_multiplier_matches_locants(self):
        """propane-1,2-diol: 'di' should match 2 locants."""
        result = name_compound('OCC(O)C')
        if result and 'diol' in result:
            # Count locants before 'diol'
            import re
            match = re.search(r'([\d,]+)-diol', result)
            if match:
                locants = match.group(1).split(',')
                # 'di' means 2 locants
                assert len(locants) == 2, f"Expected 2 locants for diol, got {len(locants)}: {result}"

    def test_ez_geometry_valid_locant(self):
        """E/Z descriptors should have valid locants."""
        # (E)-but-2-ene
        result = name_compound('C/C=C/C')
        if result:
            assert '(0E)' not in result
            assert '(0Z)' not in result

    def test_single_hydroxyl_no_multiplier(self):
        """Single hydroxyl: no multiplier prefix needed."""
        result = name_compound('CCCO')
        if result:
            # Should NOT have "diol" or "triol"
            assert 'diol' not in result
            assert 'triol' not in result

    def test_trisubstituted_ring_multiplier_match(self):
        """Trisubstituted ring: 'tri' matches 3 locants."""
        # 1,3,5-trimethylcyclohexane
        result = name_compound('CC1CC(C)CC(C)C1')
        if result and 'trimethyl' in result:
            import re
            match = re.search(r'([\d,]+)-trimethyl', result)
            if match:
                locants = match.group(1).split(',')
                assert len(locants) == 3, f"Expected 3 locants for trimethyl, got {len(locants)}: {result}"

    def test_dimethyl_prefix_two_locants(self):
        """2,4-dimethyl should have exactly 2 locants."""
        # 2,4-dimethylpentane
        result = name_compound('CC(C)CC(C)C')
        if result and 'dimethyl' in result:
            import re
            match = re.search(r'([\d,]+)-dimethyl', result)
            if match:
                locants = match.group(1).split(',')
                assert len(locants) == 2, f"Expected 2 locants for dimethyl, got {len(locants)}: {result}"
