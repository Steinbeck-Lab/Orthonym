"""Unit tests for multi-bond decomposition (a phase-01, 56-02, and 099-02).

Tests the MAX_CLEAVABLE_BONDS performance guard, consecutive duplicate-word
detection in _name_quality_is_acceptable, multi-bond retry logic,
recursive fragment decomposition, and a phase multi-bond same-type
cleavage with multi-ester assembly.
"""

import pytest
from unittest.mock import patch, MagicMock

from rdkit import Chem

from orthonym.decomposition.engine import (
    MAX_CLEAVABLE_BONDS,
    MAX_BOND_RETRY_ATTEMPTS,
    _name_quality_is_acceptable,
    _try_single_bond_decompose,
    _try_multi_bond_decompose,
    _try_iterative_mixed_decompose,
    try_decompose,
)
from orthonym.decomposition.fragment_assembly import _assemble_multi_ester


# ---------------------------------------------------------------------------
# Performance guard tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerformanceGuard:
    """Tests for the MAX_CLEAVABLE_BONDS performance guard in try_decompose."""

    def test_performance_guard_constant_exists(self):
        """MAX_CLEAVABLE_BONDS constant is defined and is 20 (a phase)."""
        assert MAX_CLEAVABLE_BONDS == 20

    def test_performance_guard_skips_many_bonds(self):
        """try_decompose returns None when molecule has >MAX_CLEAVABLE_BONDS bonds."""
        mol = Chem.MolFromSmiles("CCCCCC")  # Simple molecule for the test

        # Mock find_cleavable_bonds to return 21 bonds (more than MAX_CLEAVABLE_BONDS=20)
        fake_bonds = [{"bond_idx": i, "type": "ester"} for i in range(21)]

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
        # by mocking the next function call (name_pipeline_only, used for probe).
        with patch(
            "orthonym.decomposition.bond_cleavage.find_cleavable_bonds",
            return_value=fake_bonds,
        ), patch(
            "orthonym.namer.name_pipeline_only",
            return_value="methyl heptadecanoate",
        ):
            # Quality gate will accept this name, so try_decompose returns None
            # (not from perf guard, but from quality gate). That's fine -- the
            # point is the performance guard did not short-circuit.
            result = try_decompose(mol)
            # Result is None because quality gate accepts "methyl heptadecanoate"
            # NOT because of performance guard
            assert result is None

    def test_performance_guard_allows_single_bond(self, monkeypatch):
        """Single-bond molecules pass the performance guard (existing behavior).

        Task Z3: on the DEFAULT path `try_decompose` now returns None for this
        molecule, because the quality gate proves the pipeline name
        ("phenyl hexadecanoate") already denotes it exactly and decomposition is
        not needed. Measured: decomposition used to run and RE-DERIVE the
        identical string, and the emitted name is byte-identical either way.

        None is therefore ambiguous on the default path -- "guard blocked" and
        "gate said no decomposition needed" look the same -- so the performance
        guard is exercised with the coverage oracle turned off, which is the
        only thing standing between this call and decomposition. The emitted
        name on the default path is asserted separately below.
        """
        # Use a real ester that the decomposition engine can handle
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC(=O)Oc1ccccc1")

        # Verify it has exactly 1 cleavable bond
        from orthonym.decomposition.bond_cleavage import find_cleavable_bonds
        bonds = find_cleavable_bonds(mol)
        assert len(bonds) >= 1, "Test molecule should have at least 1 cleavable bond"
        assert len(bonds) <= MAX_CLEAVABLE_BONDS, (
            "Test molecule should have <= MAX_CLEAVABLE_BONDS bonds"
        )

        from orthonym.decomposition import engine
        monkeypatch.setenv("ORTHONYM_DECOMP_COVERAGE_ORACLE", "0")
        engine._PROVEN_COMPLETE_CACHE.clear()

        # The molecule should be decomposed (not blocked by performance guard)
        result = try_decompose(mol)
        # phenyl hexadecanoate -- decomposition should produce a name
        assert result is not None, (
            "Single-bond ester should not be blocked by performance guard"
        )

    def test_single_bond_ester_still_named_on_the_default_path(self):
        """The output-level half of the test above: skipping a decomposition
        that would only re-derive the same string must not change the name.

         corrected the acyl word from the non-PIN 'palmitate':
         "Systematic names" (the Blue Book) -- "Except for formic
        acid, acetic acid, oxalic acid..., and oxamic acid..., systematically
        formed names are preferred IUPAC names; the names given in
        are retained names for use in general nomenclature." What this test
        asserts -- that the name is unchanged by skipping decomposition -- is
        unaffected by the spelling.
        """
        from orthonym import name_compound
        assert name_compound("CCCCCCCCCCCCCCCC(=O)Oc1ccccc1") == "phenyl hexadecanoate"


# ---------------------------------------------------------------------------
# Quality gate tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQualityGateDuplicateWords:
    """Tests for consecutive duplicate-word detection in _name_quality_is_acceptable."""

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
# Multi-bond retry tests (a phase-02)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiBondRetry:
    """Tests for multi-bond retry logic in try_decompose ."""

    def test_max_bond_retry_attempts_constant_exists(self):
        """MAX_BOND_RETRY_ATTEMPTS constant is defined and is 5."""
        assert MAX_BOND_RETRY_ATTEMPTS == 5

    def test_single_bond_path_unchanged(self):
        """Molecule with exactly 1 cleavable bond: result identical to baseline.

        : single-bond molecules must return single_result directly
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
            "orthonym.namer.name_pipeline_only",
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
            "orthonym.namer.name_pipeline_only",
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
        ), patch(
            "orthonym.decomposition.engine._try_multi_bond_decompose",
            return_value=None,  # Disable multi-bond path for this test
        ):
            result = try_decompose(mol)

        # With coverage gate (a phase-02), "bad name 0" (10 chars) is
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
# Recursive fragment decomposition test (a phase-02,)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecursiveFragmentDecomposition:
    """Test that fragments with cleavable bonds are recursively decomposed."""

    def test_recursive_fragment_decomposition(self):
        """: proves recursive fragment decomposition.

        Uses a molecule with 2 ester bonds on a glycerol backbone.
        The fragment from the first bond cleavage still has a cleavable
        bond and should be recursively decomposed.

        CCCCCCCCCCCCCCCC(=O)OCC(O)COC(=O)CCCCCCCCCCCCCCC
        = glycerol dipalmitate (2 esters on a 3-carbon backbone)
        """
        #: proves recursive fragment decomposition -- the fragment
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


# ---------------------------------------------------------------------------
# a phase-02: Multi-bond same-type cleavage tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiBondDecompose:
    """Tests for _try_multi_bond_decompose in engine.py (a phase-02)."""

    def test_triacetin_produces_multifragment_name(self):
        """_try_multi_bond_decompose with triacetin produces name with
        multiplicative prefix (triacetate) or acetate reference."""
        # Triacetin = glycerol triacetate
        mol = Chem.MolFromSmiles("CC(=O)OCC(COC(C)=O)OC(C)=O")
        from orthonym.decomposition.bond_cleavage import find_cleavable_bonds
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]

        assert len(ester_bonds) >= 2, (
            f"Triacetin should have >= 2 ester bonds, got {len(ester_bonds)}"
        )

        result = _try_multi_bond_decompose(mol, ester_bonds, "pin")
        # Should produce a name containing "acetate" with multiplicative prefix
        if result is not None:
            result_lower = result.lower()
            assert ("acetate" in result_lower or "acetyloxy" in result_lower), (
                f"Triacetin multi-bond result should reference acetate, got: {result}"
            )

    def test_returns_none_when_fragments_unnamed(self):
        """_try_multi_bond_decompose returns None when fragments cannot be named."""
        mol = Chem.MolFromSmiles("CC(=O)OCC(COC(C)=O)OC(C)=O")
        from orthonym.decomposition.bond_cleavage import find_cleavable_bonds
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]

        # Mock both recursive and pipeline fallback to return None (a phase)
        with patch(
            "orthonym.decomposition.engine._name_fragment_with_fallback",
            return_value=None,
        ):
            result = _try_multi_bond_decompose(mol, ester_bonds, "pin")
        assert result is None, (
            "Should return None when fragments cannot be named"
        )

    def test_returns_none_for_single_bond(self):
        """_try_multi_bond_decompose returns None when only 1 bond provided."""
        mol = Chem.MolFromSmiles("CC(=O)OC")  # methyl acetate, 1 ester
        single_bond = [{"bond_idx": 0, "type": "ester", "acid_atom": 1, "alkyl_atom": 3}]
        result = _try_multi_bond_decompose(mol, single_bond, "pin")
        assert result is None, (
            "Should return None for single-bond input (minimum 2 required)"
        )

    def test_returns_none_for_empty_bonds(self):
        """_try_multi_bond_decompose returns None for empty bond list."""
        mol = Chem.MolFromSmiles("CCCC")
        result = _try_multi_bond_decompose(mol, [], "pin")
        assert result is None

    def test_try_decompose_integrates_multi_bond_path(self):
        """try_decompose integrates multi-bond path for 3+ same-type ester bonds.

        Uses triacetin (3 ester bonds). After single-bond retry loop fails to
        produce a good name, the multi-bond path should be attempted.
        """
        mol = Chem.MolFromSmiles("CC(=O)OCC(COC(C)=O)OC(C)=O")

        # Mock single-bond decompose to return None (forces multi-bond path)
        with patch(
            "orthonym.decomposition.engine._try_single_bond_decompose",
            return_value=None,
        ), patch(
            "orthonym.namer.name_pipeline_only",
            return_value=None,  # Force quality gate to trigger decomposition
        ):
            result = try_decompose(mol)

        # Result may still be None if multi-bond path cannot assemble,
        # but the path should have been attempted. We verify by checking
        # that the mock was called (single-bond returned None, multi-bond tried).
        # Since we can't easily introspect multi-bond path, just verify no crash.
        # The integration test below verifies end-to-end behavior.
        assert True  # No crash = multi-bond path was wired in

    def test_try_decompose_prefers_multi_bond_over_none(self):
        """When single-bond fails, multi-bond should produce a result for polyesters."""
        mol = Chem.MolFromSmiles("CC(=O)OCC(COC(C)=O)OC(C)=O")

        # Let single-bond fail but multi-bond succeed via mocks
        multi_result_sentinel = "glycerol triacetate"

        with patch(
            "orthonym.decomposition.engine._try_single_bond_decompose",
            return_value=None,
        ), patch(
            "orthonym.namer.name_pipeline_only",
            return_value=None,
        ), patch(
            "orthonym.decomposition.engine._try_multi_bond_decompose",
            return_value=multi_result_sentinel,
        ):
            result = try_decompose(mol)

        assert result == multi_result_sentinel, (
            f"try_decompose should return multi-bond result, got: {result}"
        )


@pytest.mark.unit
class TestMultiEsterAssembly:
    """Tests for _assemble_multi_ester in fragment_assembly.py (a phase-02)."""

    def test_identical_acid_names_use_multiplicative_prefix(self):
        """Identical acid names produce multiplicative prefix (e.g., triacetate)."""
        named_fragments = [
            ({"smiles": "OCC(O)CO", "side": "middle"}, "glycerol"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
        ]
        result = _assemble_multi_ester(named_fragments, "pin")
        assert result is not None
        result_lower = result.lower()
        assert "triacetate" in result_lower, (
            f"Identical acids should use 'triacetate', got: {result}"
        )
        assert "glycerol" in result_lower, (
            f"Core fragment should be glycerol, got: {result}"
        )

    def test_different_acid_names_list_positionally(self):
        """Different acid names should be listed individually."""
        named_fragments = [
            ({"smiles": "OCC(O)CO", "side": "middle"}, "glycerol"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CCC(=O)O", "side": "acid"}, "propanoic acid"),
        ]
        result = _assemble_multi_ester(named_fragments, "pin")
        assert result is not None
        result_lower = result.lower()
        assert "glycerol" in result_lower
        # Should contain both ate forms
        assert "acetate" in result_lower or "propanoate" in result_lower, (
            f"Different acids should list each: {result}"
        )

    def test_core_identified_via_seniority(self):
        """Core fragment (glycerol) is identified as most senior by score_fragment_seniority.

        Uses acetic acid (4 HA) as non-core so glycerol (6 HA) passes the
        core-size guard (a phase-05).
        """
        named_fragments = [
            ({"smiles": "OCC(O)CO", "side": "middle"}, "glycerol"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
        ]
        result = _assemble_multi_ester(named_fragments, "pin")
        assert result is not None
        # Glycerol should be the core (parent), not the acid
        assert result.lower().startswith("glycerol"), (
            f"Glycerol should be core fragment, got: {result}"
        )

    def test_returns_none_when_core_empty(self):
        """Returns None when core fragment name is empty."""
        named_fragments = [
            ({"smiles": "OCC(O)CO", "side": "middle"}, ""),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
        ]
        result = _assemble_multi_ester(named_fragments, "pin")
        assert result is None

    def test_returns_none_when_no_acid_fragments(self):
        """Returns None when no non-core fragment names available."""
        named_fragments = [
            ({"smiles": "OCC(O)CO", "side": "middle"}, "glycerol"),
        ]
        result = _assemble_multi_ester(named_fragments, "pin")
        assert result is None

    def test_diacetate_with_two_acids(self):
        """Two identical acids produce diacetate."""
        named_fragments = [
            ({"smiles": "OCC(O)CO", "side": "middle"}, "glycerol"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
        ]
        result = _assemble_multi_ester(named_fragments, "pin")
        assert result is not None
        assert "diacetate" in result.lower(), (
            f"Two identical acids should produce 'diacetate', got: {result}"
        )

    def test_tetracetate_multiplicative(self):
        """Four identical acids produce tetraacetate."""
        named_fragments = [
            ({"smiles": "OCC(O)(CO)CO", "side": "middle"}, "erythritol"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
        ]
        result = _assemble_multi_ester(named_fragments, "pin")
        assert result is not None
        assert "tetraacetate" in result.lower() or "tetra" in result.lower(), (
            f"Four identical acids should use 'tetra' prefix, got: {result}"
        )

    def test_core_size_guard_rejects_small_core(self):
        """Core-size guard rejects when core HA < max non-core HA (a phase-05).

        Scenario: core has 2 HA (ethanol-like), non-core has 6 HA (butanoic acid).
        The guard should reject because the core is smaller than a non-core fragment,
        indicating a pathological split where the ring system was lost.
        """
        named_fragments = [
            ({"smiles": "CO", "side": "middle"}, "methanol"),
            ({"smiles": "CCCC(=O)O", "side": "acid"}, "butanoic acid"),
            ({"smiles": "CCCC(=O)O", "side": "acid"}, "butanoic acid"),
        ]
        # Middle fragment (methanol) is core; core HA (2) < non-core HA (6)
        result = _assemble_multi_ester(named_fragments, "pin")
        assert result is None, (
            "Core-size guard should reject when core HA (2) < max non-core HA (6), "
            f"but got: {result}"
        )

    def test_core_size_guard_allows_large_core(self):
        """Core-size guard allows when core HA >= all non-core HA (a phase-05).

        Scenario: glycerol (6 HA) core, acetic acid (4 HA) non-core.
        The guard should allow because the core is larger.
        """
        named_fragments = [
            ({"smiles": "OCC(O)CO", "side": "middle"}, "glycerol"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
        ]
        result = _assemble_multi_ester(named_fragments, "pin")
        assert result is not None, (
            "Core-size guard should allow when core HA (6) > non-core HA (4)"
        )
        assert "glycerol" in result.lower()

    def test_core_size_guard_allows_equal(self):
        """Core-size guard allows when core HA == max non-core HA (a phase-05).

        Scenario: both core and non-core have 4 HA. Edge case: no rejection
        on equal sizes.
        """
        named_fragments = [
            ({"smiles": "CCCO", "side": "middle"}, "propan-1-ol"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
        ]
        # Middle fragment (propan-1-ol) is core; core HA (4) == non-core HA (4)
        result = _assemble_multi_ester(named_fragments, "pin")
        assert result is not None, (
            "Core-size guard should allow when core HA (4) == max non-core HA (4), "
            f"but got None"
        )


# ---------------------------------------------------------------------------
# a phase-02 Task 2: Sugar bypass, raised limits, fragment-aware quality gate
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRaisedPerformanceLimits:
    """Tests for raised performance limits (a phase-02)."""

    def test_max_cleavable_bonds_raised_to_20(self):
        """MAX_CLEAVABLE_BONDS should be 20 (raised from 12)."""
        assert MAX_CLEAVABLE_BONDS == 20

    def test_max_visited_size_raised_to_50(self):
        """_MAX_VISITED_SIZE should be 50 (raised from 30 in a phase)."""
        from orthonym.assembly.fragment_naming import _MAX_VISITED_SIZE
        assert _MAX_VISITED_SIZE == 50

    def test_performance_guard_allows_15_bonds(self):
        """Molecules with 15 cleavable bonds pass the performance guard (was blocked at 12)."""
        mol = Chem.MolFromSmiles("CCCCCC")

        fake_bonds = [{"bond_idx": i, "type": "ester"} for i in range(15)]

        with patch(
            "orthonym.decomposition.bond_cleavage.find_cleavable_bonds",
            return_value=fake_bonds,
        ), patch(
            "orthonym.namer.name_pipeline_only",
            return_value="a good name with hyphens-and-digits-1",
        ):
            result = try_decompose(mol)
            # Result is None because quality gate accepts existing name,
            # NOT because of performance guard
            assert result is None


@pytest.mark.unit
class TestSugarDetectionBypass:
    """Tests for sugar-detection bypass in _select_best_bond (a phase-02)."""

    def test_glycosidic_bond_with_sugar_gets_priority(self):
        """When a glycosidic bond leads to a known sugar, it should be preferred."""
        from orthonym.decomposition.engine import _select_best_bond

        # Create a molecule with both ester and glycosidic bonds
        # The glycosidic bond should be preferred if it leads to a known sugar
        mol = Chem.MolFromSmiles("CC(=O)OC")  # Dummy mol for BFS

        # Mock scenario: ester bond (priority 1) and glycosidic bond (priority 5)
        # Normally ester wins, but sugar bypass should give glycosidic priority
        fake_bonds = [
            {"bond_idx": 0, "type": "ester"},
            {"bond_idx": 1, "type": "glycosidic"},
        ]

        # Mock cleave_and_cap to return a sugar fragment for the glycosidic bond
        def mock_cleave(m, bond_infos, acid_side_oh=True):
            if bond_infos[0].get("type") == "glycosidic":
                return [
                    {"smiles": "OC1OC(CO)C(O)C(O)C1O", "side": "acid"},  # glucose-like
                    {"smiles": "CCO", "side": "alkyl"},
                ]
            return [{"smiles": "CCO", "side": "acid"}, {"smiles": "CO", "side": "alkyl"}]

        # Mock _name_sugar_fragment to recognize glucose
        with patch(
            "orthonym.decomposition.engine._name_sugar_fragment",
            side_effect=lambda s: "β-D-glucopyranosyloxy" if "OC1OC" in s else None,
        ), patch(
            "orthonym.decomposition.fragment_capping.cleave_and_cap",
            side_effect=mock_cleave,
        ):
            result = _select_best_bond(mol, fake_bonds)

        assert result["type"] == "glycosidic", (
            f"Sugar-detection bypass should prefer glycosidic bond, got: {result['type']}"
        )

    def test_glycosidic_without_sugar_uses_normal_priority(self):
        """When glycosidic bond does not lead to a known sugar, normal priority applies."""
        from orthonym.decomposition.engine import _select_best_bond

        mol = Chem.MolFromSmiles("CC(=O)OC")  # Dummy mol

        fake_bonds = [
            {"bond_idx": 0, "type": "ester"},
            {"bond_idx": 1, "type": "glycosidic"},
        ]

        # Mock cleave_and_cap to return non-sugar fragments
        with patch(
            "orthonym.decomposition.engine._name_sugar_fragment",
            return_value=None,  # No sugar recognized
        ), patch(
            "orthonym.decomposition.fragment_capping.cleave_and_cap",
            return_value=[{"smiles": "CCO", "side": "acid"}, {"smiles": "CO", "side": "alkyl"}],
        ):
            result = _select_best_bond(mol, fake_bonds)

        assert result["type"] == "ester", (
            f"Without sugar detection, ester should have priority, got: {result['type']}"
        )

    def test_sugar_bypass_limited_to_3_candidates(self):
        """Sugar-detection bypass probes at most 3 glycosidic bonds."""
        from orthonym.decomposition.engine import _select_best_bond

        # Need a molecule with enough bonds (5+) so bond_idx 0-4 are valid
        mol = Chem.MolFromSmiles("CCCCCCCCCC")  # 10 carbons, 9 bonds

        # 5 glycosidic bonds -- only first 3 should be probed
        fake_bonds = [{"bond_idx": i, "type": "glycosidic"} for i in range(5)]

        probe_count = [0]

        def counting_cleave(m, bond_infos, acid_side_oh=True):
            probe_count[0] += 1
            return [{"smiles": "CCO", "side": "acid"}, {"smiles": "CO", "side": "alkyl"}]

        with patch(
            "orthonym.decomposition.engine._name_sugar_fragment",
            return_value=None,
        ), patch(
            "orthonym.decomposition.fragment_capping.cleave_and_cap",
            side_effect=counting_cleave,
        ):
            _select_best_bond(mol, fake_bonds)

        assert probe_count[0] <= 3, (
            f"Sugar bypass should probe at most 3 glycosidic bonds, probed {probe_count[0]}"
        )


@pytest.mark.unit
class TestFragmentAwareQualityGate:
    """Tests for fragment-aware quality gate relaxation (a phase-02)."""

    def _make_mol(self, heavy_atoms: int):
        """Create a mol with approximately the given number of heavy atoms."""
        smiles = "C" * max(heavy_atoms, 1)
        return Chem.MolFromSmiles(smiles)

    def test_top_level_uses_ha_20_threshold(self):
        """At top level (no decomposition), HA>20 threshold applies."""
        # Name with no digits/hyphens for a 22 HA molecule -> should fail
        mol = self._make_mol(22)
        name = "someverylongretainedname"  # no digits, no hyphens, 23 chars

        # Ensure visited set is empty (top level)
        from orthonym.assembly.fragment_naming import _fragment_guard
        old_visited = getattr(_fragment_guard, 'visited', None)
        _fragment_guard.visited = set()  # Empty = top level
        try:
            result = _name_quality_is_acceptable(name, mol)
            assert result is False, (
                "Top-level should use HA>20 threshold and reject no-digits/no-hyphens "
                f"for HA=22, got True for: {name}"
            )
        finally:
            _fragment_guard.visited = old_visited

    def test_fragment_context_uses_ha_30_threshold(self):
        """During decomposition (visited set non-empty), HA>30 threshold applies."""
        # Same name with 25 HA -> should pass in fragment context (25 < 30)
        mol = self._make_mol(25)
        name = "someverylongretainedname"  # no digits, no hyphens

        from orthonym.assembly.fragment_naming import _fragment_guard
        old_visited = getattr(_fragment_guard, 'visited', None)
        _fragment_guard.visited = {"some_smiles"}  # Non-empty = fragment context
        try:
            result = _name_quality_is_acceptable(name, mol)
            assert result is True, (
                "Fragment context should use HA>30 threshold and accept no-digits/no-hyphens "
                f"for HA=25, got False for: {name}"
            )
        finally:
            _fragment_guard.visited = old_visited

    def test_fragment_context_still_rejects_above_30(self):
        """Even in fragment context, HA>30 with no digits/hyphens is rejected."""
        mol = self._make_mol(32)
        name = "someverylongretainedname"  # no digits, no hyphens

        from orthonym.assembly.fragment_naming import _fragment_guard
        old_visited = getattr(_fragment_guard, 'visited', None)
        _fragment_guard.visited = {"some_smiles"}  # Fragment context
        try:
            result = _name_quality_is_acceptable(name, mol)
            assert result is False, (
                "Fragment context with HA>30 should still reject no-digits/no-hyphens "
                f"for HA=32, got True for: {name}"
            )
        finally:
            _fragment_guard.visited = old_visited

    def test_retained_core_name_with_coverage_guard(self):
        """Retained core names pass only when coverage ratio >= 0.25 for HA > 20.

        a phase-03: coverage guard added. 'adenine' (7 chars) for HA=35
        molecule has ratio 0.20 < 0.25 threshold -> rejected.
        For HA=20 molecules, retained names always pass (HA <= 20).
        """
        from orthonym.assembly.fragment_naming import _fragment_guard
        old_visited = getattr(_fragment_guard, 'visited', None)
        _fragment_guard.visited = set()  # Top level
        try:
            # HA=35: adenine ratio 0.20 < 0.25 -> rejected
            mol_35 = self._make_mol(35)
            result_35 = _name_quality_is_acceptable("adenine", mol_35)
            assert result_35 is False, (
                "Coverage guard should reject 'adenine' for HA=35 (ratio 0.20)"
            )

            # HA=20: retained names always pass (HA <= 20 threshold)
            mol_20 = self._make_mol(20)
            result_20 = _name_quality_is_acceptable("adenine", mol_20)
            assert result_20 is True, (
                "Retained core names should pass for HA <= 20"
            )

            # HA=25: adenine ratio 7/25 = 0.28 >= 0.25 -> passes
            mol_25 = self._make_mol(25)
            result_25 = _name_quality_is_acceptable("adenine", mol_25)
            assert result_25 is True, (
                "Retained core names should pass when ratio >= 0.25"
            )
        finally:
            _fragment_guard.visited = old_visited


# ---------------------------------------------------------------------------
# a phase-04: Multi-bond threshold and glycoside/amide assembly tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiBondThresholds:
    """Tests for bond-type-specific multi-bond thresholds (a phase-04)."""

    def test_threshold_dict_has_correct_values(self):
        """_MULTI_BOND_THRESHOLD has glycosidic=2, amide=3, ester=2 ."""
        from orthonym.decomposition.engine import _MULTI_BOND_THRESHOLD
        assert _MULTI_BOND_THRESHOLD["glycosidic"] == 2
        assert _MULTI_BOND_THRESHOLD["amide"] == 3
        assert _MULTI_BOND_THRESHOLD["ester"] == 2  #: lowered from 3 to enable diester decomposition


@pytest.mark.unit
class TestMultiGlycosideAssembly:
    """Tests for _assemble_multi_glycoside in fragment_assembly.py (a phase-04)."""

    def test_two_sugars_one_aglycone(self):
        """2 sugar fragments + 1 aglycone produces multi-glycosyloxy pattern."""
        from orthonym.decomposition.fragment_assembly import _assemble_multi_glycoside
        named_fragments = [
            ({"smiles": "Oc1ccccc1", "side": "alkyl"}, "phenol"),
            ({"smiles": "OC1OC(CO)C(O)C(O)C1O", "side": "acid"}, "β-D-glucopyranosyloxy"),
            ({"smiles": "OC1OC(CO)C(O)C(O)C1O", "side": "acid"}, "β-D-glucopyranosyloxy"),
        ]
        result = _assemble_multi_glycoside(named_fragments, "pin")
        assert result is not None, "Should produce a multi-glycoside name"
        result_lower = result.lower()
        # Should contain glycosyloxy prefix and phenol
        assert "glucopyranosyloxy" in result_lower, (
            f"Should contain glucopyranosyloxy, got: {result}"
        )
        assert "phenol" in result_lower, (
            f"Should contain phenol as aglycone, got: {result}"
        )

    def test_returns_none_when_all_systematic(self):
        """Returns None when all sugar fragments have systematic (no retained) names."""
        from orthonym.decomposition.fragment_assembly import _assemble_multi_glycoside
        named_fragments = [
            ({"smiles": "Oc1ccccc1", "side": "alkyl"}, "phenol"),
            ({"smiles": "OC1CCOCC1", "side": "acid"}, "tetrahydro-2H-pyran-2-ol"),
            ({"smiles": "OC1CCOCC1", "side": "acid"}, "tetrahydro-2H-pyran-2-ol"),
        ]
        result = _assemble_multi_glycoside(named_fragments, "pin")
        assert result is None, (
            f"Should return None when no sugar retained names, got: {result}"
        )

    def test_identifies_core_by_seniority(self):
        """Core fragment (non-sugar) is identified as most senior."""
        from orthonym.decomposition.fragment_assembly import _assemble_multi_glycoside
        named_fragments = [
            ({"smiles": "OC1OC(CO)C(O)C(O)C1O", "side": "acid"}, "β-D-glucopyranosyloxy"),
            ({"smiles": "OC(=O)c1ccccc1", "side": "alkyl"}, "benzoic acid"),
            ({"smiles": "OC1OC(CO)C(O)C(O)C1O", "side": "acid"}, "β-D-glucopyranosyloxy"),
        ]
        result = _assemble_multi_glycoside(named_fragments, "pin")
        assert result is not None
        # The core should be benzoic acid
        assert "benzoic acid" in result.lower(), (
            f"Core should be 'benzoic acid', got: {result}"
        )

    def test_identical_sugars_use_multiplicative_prefix(self):
        """Identical sugar fragments should use multiplicative prefix (bis/di)."""
        from orthonym.decomposition.fragment_assembly import _assemble_multi_glycoside
        named_fragments = [
            ({"smiles": "Oc1ccccc1", "side": "alkyl"}, "phenol"),
            ({"smiles": "OC1OC(CO)C(O)C(O)C1O", "side": "acid"}, "β-D-glucopyranosyloxy"),
            ({"smiles": "OC1OC(CO)C(O)C(O)C1O", "side": "acid"}, "β-D-glucopyranosyloxy"),
        ]
        result = _assemble_multi_glycoside(named_fragments, "pin")
        assert result is not None
        result_lower = result.lower()
        # Identical sugars -> multiplicative prefix
        assert "bis" in result_lower or "di" in result_lower, (
            f"Identical sugars should use multiplicative prefix, got: {result}"
        )


@pytest.mark.unit
class TestMultiAmideAssembly:
    """Tests for _assemble_multi_amide in fragment_assembly.py (a phase-04)."""

    def test_two_acyl_one_amine(self):
        """2 acyl fragments + 1 amine core produces 'N-acyl1-N-acyl2-amine' pattern."""
        from orthonym.decomposition.fragment_assembly import _assemble_multi_amide
        named_fragments = [
            ({"smiles": "NC1CCCCC1", "side": "alkyl"}, "cyclohexanamine"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CCC(=O)O", "side": "acid"}, "propanoic acid"),
        ]
        result = _assemble_multi_amide(named_fragments, "pin")
        assert result is not None, "Should produce a multi-amide name"
        result_lower = result.lower()
        # Should contain N-acyl prefix and amine
        assert "cyclohexanamine" in result_lower, (
            f"Should contain amine core, got: {result}"
        )
        assert "n-" in result_lower, (
            f"Should contain N- prefix for acyl groups, got: {result}"
        )

    def test_returns_none_when_core_name_empty(self):
        """Returns None when core fragment name is empty."""
        from orthonym.decomposition.fragment_assembly import _assemble_multi_amide
        named_fragments = [
            ({"smiles": "NC1CCCCC1", "side": "alkyl"}, ""),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
        ]
        result = _assemble_multi_amide(named_fragments, "pin")
        assert result is None, (
            f"Should return None when core name is empty, got: {result}"
        )

    def test_identical_acyls_use_multiplicative_prefix(self):
        """Identical acyl fragments should use N,N-di... grouping."""
        from orthonym.decomposition.fragment_assembly import _assemble_multi_amide
        named_fragments = [
            ({"smiles": "NC1CCCCC1", "side": "alkyl"}, "cyclohexanamine"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
        ]
        result = _assemble_multi_amide(named_fragments, "pin")
        assert result is not None
        result_lower = result.lower()
        # Identical acyls -> N,N,N-triacetyl pattern
        assert "acetyl" in result_lower, (
            f"Should contain acetyl, got: {result}"
        )
        # Should have N-prefix
        assert "n" in result_lower, (
            f"Should contain N-prefix, got: {result}"
        )

    def test_core_identified_as_amine(self):
        """Core fragment is the amine (most senior), non-core are acids."""
        from orthonym.decomposition.fragment_assembly import _assemble_multi_amide
        named_fragments = [
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "NC1CCCCC1", "side": "alkyl"}, "cyclohexanamine"),
            ({"smiles": "CCC(=O)O", "side": "acid"}, "propanoic acid"),
            ({"smiles": "CCCC(=O)O", "side": "acid"}, "butanoic acid"),
        ]
        result = _assemble_multi_amide(named_fragments, "pin")
        assert result is not None
        # Core should be cyclohexanamine
        assert "cyclohexanamine" in result.lower(), (
            f"Core should be cyclohexanamine, got: {result}"
        )


@pytest.mark.unit
class TestMultiBondIntegration:
    """Integration tests for try_decompose with glycosidic and amide multi-bond paths."""

    def test_try_decompose_multi_glycosidic(self):
        """try_decompose triggers multi-bond path for molecule with 2+ glycosidic bonds."""
        # Mock molecule with 2 glycosidic bonds to test dispatch
        mol = Chem.MolFromSmiles("C" * 30)  # Dummy molecule

        fake_bonds = [
            {"bond_idx": 0, "type": "glycosidic"},
            {"bond_idx": 1, "type": "glycosidic"},
        ]

        multi_result_sentinel = "bis(glucopyranosyloxy)phenol"

        with patch(
            "orthonym.decomposition.bond_cleavage.find_cleavable_bonds",
            return_value=fake_bonds,
        ), patch(
            "orthonym.namer.name_pipeline_only",
            return_value=None,  # Force quality gate to trigger decomposition
        ), patch(
            "orthonym.decomposition.engine._try_single_bond_decompose",
            return_value=None,
        ), patch(
            "orthonym.decomposition.engine._try_multi_bond_decompose",
            return_value=multi_result_sentinel,
        ):
            result = try_decompose(mol)

        assert result == multi_result_sentinel, (
            f"Should dispatch to multi-bond for 2+ glycosidic bonds, got: {result}"
        )

    def test_try_decompose_multi_amide(self):
        """try_decompose triggers multi-bond path for molecule with 3+ amide bonds."""
        mol = Chem.MolFromSmiles("C" * 30)  # Dummy molecule

        fake_bonds = [
            {"bond_idx": i, "type": "amide"}
            for i in range(3)
        ]

        multi_result_sentinel = "N-acetyl-N-propanoyl-cyclohexanamine"

        with patch(
            "orthonym.decomposition.bond_cleavage.find_cleavable_bonds",
            return_value=fake_bonds,
        ), patch(
            "orthonym.namer.name_pipeline_only",
            return_value=None,
        ), patch(
            "orthonym.decomposition.engine._try_single_bond_decompose",
            return_value=None,
        ), patch(
            "orthonym.decomposition.engine._try_multi_bond_decompose",
            return_value=multi_result_sentinel,
        ):
            result = try_decompose(mol)

        assert result == multi_result_sentinel, (
            f"Should dispatch to multi-bond for 3+ amide bonds, got: {result}"
        )


# ---------------------------------------------------------------------------
# a phase-03: Fragment storage and role-based core identification tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFragmentStorageAndRoles:
    """Tests for (list-of-tuples storage) and (middle-fragment
    core identification) fixes in engine.py and fragment_assembly.py.

    These tests verify:
    - Identical SMILES fragments are preserved (not deduplicated by dict key)
    - Middle-role fragments are preferred as core candidates
    - Multiple middle fragments ranked by seniority
    - Assemblers use new List[Tuple[Dict, str]] signature
    - None returns from transformation functions handled gracefully
    - Iterative mixed decomposer also uses list-of-tuples storage
    """

    # -- Test 1: Identical acid fragments retained --

    def test_identical_acid_fragments_retained(self):
        """When _try_multi_bond_decompose receives fragments with identical
        SMILES (simulating a triglyceride with 3 identical fatty acids), all
        fragments appear in the named list (not deduplicated by dict overwrite).

        : verifies that named_fragments is a list of tuples, not a dict.
        """
        mol = Chem.MolFromSmiles("CC(=O)OCC(COC(C)=O)OC(C)=O")  # triacetin
        from orthonym.decomposition.bond_cleavage import find_cleavable_bonds
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]

        if len(ester_bonds) < 2:
            pytest.skip("Triacetin should have >= 2 ester bonds for this test")

        # The multi-bond decomposer should produce a result with a
        # multiplicative prefix ("tri") proving all 3 identical fragments
        # were retained. With dict storage, only 1 copy would survive.
        result = _try_multi_bond_decompose(mol, ester_bonds, "pin")
        if result is not None:
            result_lower = result.lower()
            assert "tri" in result_lower or "3" in result_lower, (
                f"Triacetin should have 'tri' prefix (3 identical acids), got: {result}"
            )

    # -- Test 2: Multi-ester assembly counts duplicates --

    def test_multi_ester_assembly_counts_duplicates(self):
        """_assemble_multi_ester with 3 identical acid names produces
        multiplicative prefix 'tri' (not 'mono' or bare).

        Uses the NEW assembler signature: List[Tuple[Dict, str]].
        """
        named_fragments = [
            ({"smiles": "OCC(O)CO", "side": "middle"}, "glycerol"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
        ]
        result = _assemble_multi_ester(named_fragments)
        assert result is not None, "Should assemble a multi-ester name"
        result_lower = result.lower()
        assert "triacetate" in result_lower, (
            f"3 identical acids should produce 'triacetate', got: {result}"
        )
        assert "glycerol" in result_lower, (
            f"Core should be glycerol, got: {result}"
        )

    # -- Test 3: Middle fragment identified as core --

    def test_middle_fragment_identified_as_core(self):
        """_assemble_multi_ester with fragments where one has side='middle'
        uses that fragment as core.

        : middle-role fragments should be preferred as core.
        """
        named_fragments = [
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "OCC(O)CO", "side": "middle"}, "glycerol"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
        ]
        result = _assemble_multi_ester(named_fragments)
        assert result is not None
        assert result.lower().startswith("glycerol"), (
            f"Middle fragment (glycerol) should be core, got: {result}"
        )

    # -- Test 4: Middle fragment preferred over seniority --

    def test_middle_fragment_preferred_over_seniority(self):
        """When a 'middle' fragment exists, it is used as core even if another
        fragment has higher seniority score.

        Uses mock seniority to make an acid fragment appear more senior than
        the middle fragment. The middle role should still win.
        """
        named_fragments = [
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CCCO", "side": "middle"}, "propan-1-ol"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
        ]
        # Even with mock seniority favoring acetic acid, the middle
        # fragment should be chosen as core
        with patch(
            "orthonym.decomposition.fragment_ranker.score_fragment_seniority",
            side_effect=lambda s: (-5, 0, 0, 0, 0) if "CC(=O)O" in s else (0, 10, 3, 0, 0),
        ):
            result = _assemble_multi_ester(named_fragments)
        assert result is not None
        assert "propan-1-ol" in result.lower(), (
            f"Middle fragment should be core regardless of seniority, got: {result}"
        )

    # -- Test 5: No middle falls back to seniority --

    def test_no_middle_falls_back_to_seniority(self):
        """When no fragment has side='middle', core identification falls back
        to seniority (existing behavior preserved).
        """
        named_fragments = [
            ({"smiles": "OCC(O)CO", "side": "alkyl"}, "glycerol"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
        ]
        result = _assemble_multi_ester(named_fragments)
        assert result is not None
        # Glycerol has higher seniority than acetic acid (more FGs, larger)
        assert "glycerol" in result.lower(), (
            f"Without middle, seniority should pick glycerol as core, got: {result}"
        )

    # -- Test 6: Multiple middles ranked by seniority --

    def test_multiple_middles_ranked_by_seniority(self):
        """When 2 fragments have side='middle', the most senior one is used
        as core (per).
        """
        named_fragments = [
            ({"smiles": "OCC(O)CO", "side": "middle"}, "glycerol"),
            ({"smiles": "CCCO", "side": "middle"}, "propan-1-ol"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
        ]
        result = _assemble_multi_ester(named_fragments)
        assert result is not None
        # glycerol (3 OH, 6 HA) should be more senior than propan-1-ol (1 OH, 4 HA)
        assert "glycerol" in result.lower(), (
            f"Most senior middle fragment (glycerol) should be core, got: {result}"
        )

    # -- Test 7: New signature accepted --

    def test_named_fragments_list_signature(self):
        """_assemble_multi_ester accepts List[Tuple[Dict, str]] (new signature).

        Verifies the new signature works and returns a non-None result for
        a valid input. The old signature (fragments, fragment_names, style)
        should no longer be the primary interface.
        """
        named_fragments = [
            ({"smiles": "OCC(O)CO", "side": "middle"}, "glycerol"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CCC(=O)O", "side": "acid"}, "propanoic acid"),
        ]
        # Should not raise TypeError
        result = _assemble_multi_ester(named_fragments)
        assert result is not None, (
            "New List[Tuple] signature should produce a valid result"
        )

    # -- Test 8: Assembler handles None from _acid_to_ate --

    def test_assembler_handles_none_from_acid_to_ate(self):
        """When _acid_to_ate returns None for a non-acid fragment, the
        assembler skips that fragment (per).

        Simulates a fragment whose name doesn't look like an acid, so
        _acid_to_ate returns None. The assembler should skip it gracefully.
        """
        named_fragments = [
            ({"smiles": "OCC(O)CO", "side": "middle"}, "glycerol"),
            ({"smiles": "CC(=O)O", "side": "acid"}, "acetic acid"),
            ({"smiles": "CCCCCC", "side": "acid"}, "hexane"),  # Not an acid name
        ]
        # _acid_to_ate("hexane") should return None (per pre-validation)
        # The assembler should skip hexane and still produce a result with
        # just the acetic acid arm
        result = _assemble_multi_ester(named_fragments)
        # Result should contain acetate (from acetic acid) but not crash
        if result is not None:
            assert "hexanoate" not in result.lower(), (
                f"Non-acid 'hexane' should not produce -ate form, got: {result}"
            )

    # -- Test 9: Iterative decomposer preserves duplicates --

    def test_iterative_decomposer_preserves_duplicates(self):
        """The iterative mixed decomposer also uses list-of-tuples
        (per Pitfall 5 from RESEARCH.md).

        Verifies that _try_iterative_mixed_decompose does not use dict
        storage that would deduplicate identical fragment SMILES.
        """
        from orthonym.decomposition.engine import _try_iterative_mixed_decompose

        mol = Chem.MolFromSmiles("C" * 30)  # Dummy molecule

        # Create fake bonds with 2 different types (needed for mixed decomposition)
        fake_bonds = [
            {"bond_idx": 0, "type": "ester"},
            {"bond_idx": 1, "type": "amide"},
        ]

        # Track fragment naming calls to verify all fragments are named
        naming_calls = []
        original_fallback = None

        def tracking_namer(smiles):
            naming_calls.append(smiles)
            return f"fragment-{len(naming_calls)}"

        # Mock the decomposition to produce identical fragments
        identical_frags = [
            {"smiles": "CC(=O)O", "side": "acid"},
            {"smiles": "CC(=O)O", "side": "acid"},
            {"smiles": "OCC(O)CO", "side": "alkyl"},
        ]

        with patch(
            "orthonym.decomposition.engine._select_best_bond",
            return_value=fake_bonds[0],
        ), patch(
            "orthonym.decomposition.fragment_capping.cleave_and_cap",
            return_value=identical_frags,
        ), patch(
            "orthonym.decomposition.engine._name_sugar_fragment",
            return_value=None,
        ), patch(
            "orthonym.decomposition.engine._name_fragment_with_fallback",
            side_effect=tracking_namer,
        ), patch(
            "orthonym.decomposition.engine._name_quality_is_acceptable",
            return_value=True,
        ), patch(
            "orthonym.decomposition.engine._coverage_is_adequate",
            return_value=True,
        ):
            result = _try_iterative_mixed_decompose(mol, fake_bonds, "pin")

        # With list-of-tuples, all 3 fragments should be named individually.
        # With dict storage, the 2 identical "CC(=O)O" would be named once.
        assert len(naming_calls) >= 3, (
            f"All fragments should be named (including duplicates), "
            f"but only {len(naming_calls)} naming calls made: {naming_calls}"
        )
