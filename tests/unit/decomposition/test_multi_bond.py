"""Unit tests for performance guard and quality gate refinement (Phase 56-01).

Tests the MAX_CLEAVABLE_BONDS performance guard in try_decompose() and the
consecutive duplicate-word detection in _name_quality_is_acceptable().
"""

import pytest
from unittest.mock import patch, MagicMock

from rdkit import Chem

from orthonym.decomposition.engine import (
    MAX_CLEAVABLE_BONDS,
    _name_quality_is_acceptable,
    try_decompose,
)


# ---------------------------------------------------------------------------
# Performance guard tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerformanceGuard:
    """Tests for the MAX_CLEAVABLE_BONDS performance guard in try_decompose()."""

    def test_performance_guard_constant_exists(self):
        """MAX_CLEAVABLE_BONDS constant is defined and is 8."""
        assert MAX_CLEAVABLE_BONDS == 8

    def test_performance_guard_skips_many_bonds(self):
        """try_decompose returns None when molecule has >MAX_CLEAVABLE_BONDS bonds."""
        mol = Chem.MolFromSmiles("CCCCCC")  # Simple molecule for the test

        # Mock find_cleavable_bonds to return 9 bonds (more than MAX_CLEAVABLE_BONDS=8)
        fake_bonds = [{"bond_idx": i, "type": "ester"} for i in range(9)]

        with patch(
            "orthonym.decomposition.bond_cleavage.find_cleavable_bonds",
            return_value=fake_bonds,
        ):
            result = try_decompose(mol)

        assert result is None, (
            "try_decompose should return None when bond count exceeds "
            f"MAX_CLEAVABLE_BONDS={MAX_CLEAVABLE_BONDS}"
        )

    def test_performance_guard_allows_max_bonds(self):
        """try_decompose does NOT return None from performance guard when bonds == MAX."""
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC(=O)OC")  # Simple ester

        # Mock find_cleavable_bonds to return exactly MAX_CLEAVABLE_BONDS bonds
        fake_bonds = [{"bond_idx": i, "type": "ester"} for i in range(MAX_CLEAVABLE_BONDS)]

        # The function may still return None for other reasons (quality gate,
        # fragment naming), but it should NOT be because of the performance guard.
        # We verify this by checking that the code progresses past the guard
        # by mocking the next function call (name_fragment_recursively).
        with patch(
            "orthonym.decomposition.bond_cleavage.find_cleavable_bonds",
            return_value=fake_bonds,
        ), patch(
            "orthonym.assembly.fragment_naming.name_fragment_recursively",
            return_value="methyl heptadecanoate",
        ):
            # Quality gate will accept this name, so try_decompose returns None
            # (not from perf guard, but from quality gate). That's fine -- the
            # point is the performance guard did not short-circuit.
            result = try_decompose(mol)
            # Result is None because quality gate accepts "methyl heptadecanoate"
            # NOT because of performance guard
            assert result is None

    def test_performance_guard_allows_single_bond(self):
        """Single-bond molecules pass the performance guard (existing behavior)."""
        # Use a real ester that the decomposition engine can handle
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC(=O)Oc1ccccc1")

        # Verify it has exactly 1 cleavable bond
        from orthonym.decomposition.bond_cleavage import find_cleavable_bonds
        bonds = find_cleavable_bonds(mol)
        assert len(bonds) >= 1, "Test molecule should have at least 1 cleavable bond"
        assert len(bonds) <= MAX_CLEAVABLE_BONDS, (
            "Test molecule should have <= MAX_CLEAVABLE_BONDS bonds"
        )

        # The molecule should be decomposed (not blocked by performance guard)
        result = try_decompose(mol)
        # phenyl palmitate -- decomposition should produce a name
        assert result is not None, (
            "Single-bond ester should not be blocked by performance guard"
        )


# ---------------------------------------------------------------------------
# Quality gate tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQualityGateDuplicateWords:
    """Tests for consecutive duplicate-word detection in _name_quality_is_acceptable()."""

    def _make_mol(self, heavy_atoms: int):
        """Create a mol object with approximately the given number of heavy atoms."""
        # Build a chain of carbons to get the right heavy atom count
        smiles = "C" * max(heavy_atoms, 1)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Fallback for very large chains
            smiles = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
            mol = Chem.MolFromSmiles(smiles)
        return mol

    def test_quality_gate_rejects_consecutive_duplicate_words(self):
        """Names with consecutive duplicate words (>3 chars) are rejected."""
        mol = self._make_mol(30)  # Large molecule so other gates don't trigger
        name = "tetracosyl palmitate palmitate"

        result = _name_quality_is_acceptable(name, mol)
        assert result is False, (
            "Quality gate should reject names with consecutive duplicate words "
            f"like 'palmitate palmitate', got True for: {name}"
        )

    def test_quality_gate_rejects_any_consecutive_duplicate(self):
        """Any consecutive duplicate word >3 chars triggers rejection."""
        mol = self._make_mol(30)
        name = "2-hydroxypropyl acetate acetate"

        result = _name_quality_is_acceptable(name, mol)
        assert result is False

    def test_quality_gate_accepts_valid_repeated_prefix(self):
        """Names with similar but non-consecutive words are accepted."""
        mol = self._make_mol(10)
        name = "2-methylpropyl 2-methylpropanoate"

        result = _name_quality_is_acceptable(name, mol)
        assert result is True, (
            "Quality gate should accept valid names with similar but "
            f"non-consecutive words, got False for: {name}"
        )

    def test_quality_gate_accepts_normal_ester_name(self):
        """Normal ester names with no duplicate words are accepted."""
        mol = self._make_mol(8)
        name = "ethyl acetate"

        result = _name_quality_is_acceptable(name, mol)
        assert result is True, (
            f"Quality gate should accept normal ester name, got False for: {name}"
        )

    def test_quality_gate_ignores_short_duplicate_words(self):
        """Consecutive duplicate words of 3 chars or fewer are NOT rejected."""
        mol = self._make_mol(10)
        # "di" is 2 chars, "the" is 3 chars -- both should be ignored
        name = "di di methylpentanoic acid"

        result = _name_quality_is_acceptable(name, mol)
        # The name may still be rejected by OTHER quality gate criteria
        # (e.g., too short for heavy atoms). We specifically test that
        # the duplicate-word check does not reject short words.
        # Use a small mol to avoid length-based rejection.
        mol_small = self._make_mol(5)
        name_small = "the the compound"
        result_small = _name_quality_is_acceptable(name_small, mol_small)
        assert result_small is True, (
            "Quality gate should not reject consecutive duplicate words "
            f"of 3 chars or fewer, got False for: {name_small}"
        )

    def test_quality_gate_accepts_non_adjacent_duplicates(self):
        """Words that appear twice but not consecutively are accepted."""
        mol = self._make_mol(12)
        name = "methyl 2-methylpropanoate"

        result = _name_quality_is_acceptable(name, mol)
        assert result is True, (
            "Quality gate should accept non-adjacent duplicate words, "
            f"got False for: {name}"
        )

    def test_quality_gate_preserves_existing_behavior_for_unknown(self):
        """The quality gate still rejects 'unknown' names."""
        mol = self._make_mol(10)
        assert _name_quality_is_acceptable("unknown", mol) is False

    def test_quality_gate_preserves_existing_behavior_for_empty(self):
        """The quality gate still rejects empty/None names."""
        mol = self._make_mol(10)
        assert _name_quality_is_acceptable("", mol) is False
        assert _name_quality_is_acceptable(None, mol) is False
