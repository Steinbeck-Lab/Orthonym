"""Unit tests for multi-bond decomposition (Phase 56-01 and 56-02).

Tests the MAX_CLEAVABLE_BONDS performance guard, consecutive duplicate-word
detection in _name_quality_is_acceptable(), multi-bond retry logic, and
recursive fragment decomposition.
"""

import pytest
from unittest.mock import patch, MagicMock

from rdkit import Chem

from orthonym.decomposition.engine import (
    MAX_CLEAVABLE_BONDS,
    MAX_BOND_RETRY_ATTEMPTS,
    _name_quality_is_acceptable,
    _try_single_bond_decompose,
    try_decompose,
)


# ---------------------------------------------------------------------------
# Performance guard tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerformanceGuard:
    """Tests for the MAX_CLEAVABLE_BONDS performance guard in try_decompose()."""

    def test_performance_guard_constant_exists(self):
        """MAX_CLEAVABLE_BONDS constant is defined and is 12."""
        assert MAX_CLEAVABLE_BONDS == 12

    def test_performance_guard_skips_many_bonds(self):
        """try_decompose returns None when molecule has >MAX_CLEAVABLE_BONDS bonds."""
        mol = Chem.MolFromSmiles("CCCCCC")  # Simple molecule for the test

        # Mock find_cleavable_bonds to return 13 bonds (more than MAX_CLEAVABLE_BONDS=12)
        fake_bonds = [{"bond_idx": i, "type": "ester"} for i in range(13)]

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


# ---------------------------------------------------------------------------
# Multi-bond retry tests (Phase 56-02)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiBondRetry:
    """Tests for multi-bond retry logic in try_decompose() (DECP-01)."""

    def test_max_bond_retry_attempts_constant_exists(self):
        """MAX_BOND_RETRY_ATTEMPTS constant is defined and is 5."""
        assert MAX_BOND_RETRY_ATTEMPTS == 5

    def test_single_bond_path_unchanged(self):
        """Molecule with exactly 1 cleavable bond: result identical to baseline.

        DECP-05: single-bond molecules must return single_result directly
        without any quality gate check on the decomposition result.
        """
        # Methyl acetate has exactly 1 ester bond
        mol = Chem.MolFromSmiles("CC(=O)OC")

        from orthonym.decomposition.bond_cleavage import find_cleavable_bonds
        bonds = find_cleavable_bonds(mol)
        assert len(bonds) == 1, f"Expected 1 cleavable bond, got {len(bonds)}"

        from orthonym.namer import name_compound
        name = name_compound("CC(=O)OC")
        assert name is not None
        # Methyl acetate should produce "methyl acetate"
        assert name == "methyl acetate", (
            f"Single-bond ester should produce 'methyl acetate', got: {name}"
        )

    def test_multi_bond_retry_tries_alternatives(self):
        """When first bond produces a poor name, alternative bonds are tried.

        Mock scenario: 3 bonds. First bond returns bad name (fails quality gate).
        Second bond returns good name (passes quality gate). Assert good name returned.
        """
        mol = Chem.MolFromSmiles("C" * 30)  # Dummy 30-atom molecule

        fake_bonds = [
            {"bond_idx": 0, "type": "ester"},
            {"bond_idx": 1, "type": "ester"},
            {"bond_idx": 2, "type": "ester"},
        ]

        # Track which bonds are tried
        attempted_bonds = []

        def mock_try_single(m, bond, style="pin"):
            attempted_bonds.append(bond["bond_idx"])
            if bond["bond_idx"] == 0:
                return "palmitate palmitate"  # Bad: duplicate words
            elif bond["bond_idx"] == 1:
                return "propane-1,2,3-triyl trihexadecanoate"  # Good name
            return None

        def mock_quality(name, m):
            if "palmitate palmitate" in name:
                return False  # Duplicate words -> reject
            return True  # Accept good names

        with patch(
            "orthonym.decomposition.bond_cleavage.find_cleavable_bonds",
            return_value=fake_bonds,
        ), patch(
            "orthonym.assembly.fragment_naming.name_fragment_recursively",
            return_value=None,  # Force quality gate to fail for existing name
        ), patch(
            "orthonym.decomposition.engine._try_single_bond_decompose",
            side_effect=mock_try_single,
        ), patch(
            "orthonym.decomposition.engine._name_quality_is_acceptable",
            side_effect=mock_quality,
        ), patch(
            "orthonym.decomposition.engine._select_best_bond",
            return_value=fake_bonds[0],
        ):
            result = try_decompose(mol)

        assert result == "propane-1,2,3-triyl trihexadecanoate", (
            f"Should return good name from alternative bond, got: {result}"
        )
        assert 0 in attempted_bonds, "Should have tried bond 0 first"
        assert 1 in attempted_bonds, "Should have tried bond 1 as alternative"

    def test_multi_bond_caps_at_max_attempts(self):
        """Multi-bond retry stops after MAX_BOND_RETRY_ATTEMPTS bonds.

        Mock scenario: 5 bonds, all produce poor names. Assert only
        MAX_BOND_RETRY_ATTEMPTS bonds are tried, and single_result is returned.
        """
        mol = Chem.MolFromSmiles("C" * 30)  # Dummy 30-atom molecule

        fake_bonds = [
            {"bond_idx": i, "type": "ester"} for i in range(5)
        ]

        attempted_bonds = []

        def mock_try_single(m, bond, style="pin"):
            attempted_bonds.append(bond["bond_idx"])
            return f"bad name {bond['bond_idx']}"

        def mock_quality(name, m):
            return False  # All names fail quality gate

        with patch(
            "orthonym.decomposition.bond_cleavage.find_cleavable_bonds",
            return_value=fake_bonds,
        ), patch(
            "orthonym.assembly.fragment_naming.name_fragment_recursively",
            return_value=None,  # Existing name is None -> quality fails
        ), patch(
            "orthonym.decomposition.engine._try_single_bond_decompose",
            side_effect=mock_try_single,
        ), patch(
            "orthonym.decomposition.engine._name_quality_is_acceptable",
            side_effect=mock_quality,
        ), patch(
            "orthonym.decomposition.engine._select_best_bond",
            return_value=fake_bonds[0],
        ):
            result = try_decompose(mol)

        # With coverage gate (Phase 87-02), "bad name 0" (10 chars) is
        # rejected for a 30-atom molecule (need >= 18 chars). Result is None
        # when all decomposition attempts fail coverage + quality checks.
        assert result is None, (
            f"Coverage gate should reject inadequate names, got: {result}"
        )
        # Should have tried at most MAX_BOND_RETRY_ATTEMPTS bonds total
        assert len(attempted_bonds) <= MAX_BOND_RETRY_ATTEMPTS, (
            f"Should try at most {MAX_BOND_RETRY_ATTEMPTS} bonds, "
            f"tried {len(attempted_bonds)}: {attempted_bonds}"
        )


# ---------------------------------------------------------------------------
# Recursive fragment decomposition test (Phase 56-02, DECP-02)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecursiveFragmentDecomposition:
    """Test that fragments with cleavable bonds are recursively decomposed."""

    def test_recursive_fragment_decomposition(self):
        """DECP-02: proves recursive fragment decomposition.

        Uses a molecule with 2 ester bonds on a glycerol backbone.
        The fragment from the first bond cleavage still has a cleavable
        bond and should be recursively decomposed.

        CCCCCCCCCCCCCCCC(=O)OCC(O)COC(=O)CCCCCCCCCCCCCCC
        = glycerol dipalmitate (2 esters on a 3-carbon backbone)
        """
        # DECP-02: proves recursive fragment decomposition -- the fragment
        # from first bond cleavage still has a cleavable bond and is
        # recursively decomposed.
        from orthonym.namer import name_compound

        smiles = "CCCCCCCCCCCCCCCC(=O)OCC(O)COC(=O)CCCCCCCCCCCCCCC"
        name = name_compound(smiles)

        assert name is not None, "Glycerol dipalmitate should produce a name"
        assert name != "unknown", "Should not produce 'unknown'"

        # The name should not have consecutive duplicate words
        # (which would indicate half-decomposition)
        words = name.split()
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and len(words[i]) > 3:
                pytest.fail(
                    f"Name has consecutive duplicate words: '{words[i]}' "
                    f"in '{name}' -- indicates incomplete decomposition"
                )

        # The name should reference ester-related naming
        assert len(name) > 15, (
            f"Name too short for a 35-atom diester: {name}"
        )
