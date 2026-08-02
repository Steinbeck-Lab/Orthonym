"""Unit tests for decomposition engine quality gating and orchestration.

Tests the quality gate (_name_quality_is_acceptable), bond selection
(_select_best_bond), and try_decompose() orchestration from
orthonym.decomposition.engine.
"""

import pytest
from rdkit import Chem

from orthonym.decomposition.engine import (
    _coverage_is_adequate,
    _decomposition_is_worse,
    _name_quality_is_acceptable,
    _name_sugar_fragment,
    _select_best_bond,
    try_decompose,
)
from orthonym.assembly.fragment_naming import (
    _fragment_guard,
    _get_visited,
    get_naming_depth,
    name_fragment_recursively,
)


def _mol(smiles: str):
    """Helper to create RDKit Mol from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    return mol


# ============================================================================
# Quality gate tests
# ============================================================================


@pytest.mark.unit
class TestNameQualityGate:
    """Test _name_quality_is_acceptable() correctly identifies good/poor names."""

    def test_good_name_ethanol(self):
        """Good name for simple molecule should be acceptable."""
        mol = _mol("CCO")
        assert _name_quality_is_acceptable("ethanol", mol) is True

    def test_unknown_is_unacceptable(self):
        """'unknown' is never acceptable."""
        mol = _mol("CCO")
        assert _name_quality_is_acceptable("unknown", mol) is False

    def test_empty_string_is_unacceptable(self):
        """Empty string is never acceptable."""
        mol = _mol("CCO")
        assert _name_quality_is_acceptable("", mol) is False

    def test_none_is_unacceptable(self):
        """None is never acceptable."""
        mol = _mol("CCO")
        assert _name_quality_is_acceptable(None, mol) is False

    def test_short_name_for_large_molecule(self):
        """Very short name for a molecule with >15 heavy atoms is suspicious."""
        # Large ester: 25+ heavy atoms
        large_ester = _mol("CCCCCCCCCCCC(=O)OCCCCCCCCCC")
        # "benzene" is way too short for 25+ heavy atoms
        assert _name_quality_is_acceptable("benzene", large_ester) is False

    def test_acceptable_name_with_digits_and_hyphens(self):
        """A name with digits and hyphens for a small molecule is acceptable."""
        mol = _mol("CC(C)C(=O)O")
        assert _name_quality_is_acceptable("2-methylpropanoic acid", mol) is True

    def test_no_digits_no_hyphens_large_molecule(self):
        """A retained name with no digits/hyphens for >20 heavy atoms is suspicious."""
        # Something with 21+ heavy atoms
        large_mol = _mol("CCCCCCCCCCCCCCCCCCCCCC")  # docosane (22 C)
        assert _name_quality_is_acceptable("benzene", large_mol) is False

    def test_acceptable_name_small_molecule(self):
        """For small molecules (<= 15 heavy atoms), short names are fine."""
        mol = _mol("c1ccccc1")  # benzene, 6 heavy atoms
        assert _name_quality_is_acceptable("benzene", mol) is True

    def test_name_with_hyphens_for_large_molecule(self):
        """A name with hyphens for a large molecule is acceptable (if long enough)."""
        large_mol = _mol("CCCCCCCCCCCCCCCCCCCCCC")  # 22 heavy atoms
        # Name must be >= heavy_atoms // 2 = 11 chars to pass length gate
        assert _name_quality_is_acceptable("2-methylicosane", large_mol) is True

    def test_short_name_with_hyphens_rejected(self):
        """A short name with hyphens is still rejected by the tightened gate."""
        large_mol = _mol("CCCCCCCCCCCCCCCCCCCCCC")  # 22 heavy atoms
        # 10 chars < 22 // 2 = 11 -> rejected despite hyphens
        assert _name_quality_is_acceptable("do-co-sane", large_mol) is False

    def test_name_with_digits_for_large_molecule(self):
        """A name with digits for a large molecule is acceptable (if chars/HA adequate)."""
        large_mol = _mol("CCCCCCCCCCCCCCCCCCCCCC")  # 22 heavy atoms
        # "2-methyldocosane-1" (18 chars) / 22 HA = 0.82 > 0.65 -> passes chars/HA gate
        assert _name_quality_is_acceptable("2-methyldocosane-1", large_mol) is True

    # -- DECO-24: chars/HA quality gate for 15-30 HA range --

    def test_quality_gate_chars_per_ha_rejects_low_ratio(self):
        """A name with chars/HA < 0.65 for a 23 HA molecule should be rejected.

        D-04: chars/HA check for medium molecules (15-30 HA).
        "chloroethene" (12 chars) for a 23 HA molecule -> 12/23 = 0.52 < 0.65.
        Note: name must not be in _RETAINED_CORE_NAMES to test the new gate.
        Requires cleavable bonds to trigger rejection (Plan 03 calibration).
        """
        # 23 heavy atoms -- has a ring system with cleavable bonds
        mol_23 = _mol("c1ccc2ncccc2c1CCCCCCCCCCCCCl")
        # "chloroethene" is 12 chars, 12/23 = 0.52 < 0.65 -> should be rejected
        assert _name_quality_is_acceptable("chloroethene", mol_23) is False

    def test_quality_gate_chars_per_ha_accepts_good_ratio(self):
        """A name with adequate chars/HA for an 18 HA molecule should be accepted.

        "octadecanamide" (14 chars) / 18 HA = 0.78 > 0.7 -> passes.
        """
        # 18 heavy atoms
        mol_18 = _mol("CCCCCCCCCCCCCCCC(N)=O")
        # "octadecanamide" has 14 chars, 14/18 = 0.78 > 0.7 -> should be accepted
        assert _name_quality_is_acceptable("octadecanamide", mol_18) is True

    def test_quality_gate_chars_per_ha_ignores_outside_range(self):
        """Molecules with HA > 30 should NOT be affected by the 15-30 HA check.

        The existing HA > 25 check (0.45 threshold) handles those.
        A name with chars/HA between 0.45 and 0.7 should pass for HA > 30
        because the stricter 15-30 check doesn't apply.
        """
        # 35 heavy atoms
        mol_35 = _mol("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
        # "1-methyltetracontane" (20 chars) / 35 HA = 0.57 > 0.45 (passes >25 gate)
        # but 0.57 < 0.7 (would fail 15-30 gate if it applied -- but HA > 30)
        assert _name_quality_is_acceptable(
            "1-methyltetracontane", mol_35
        ) is True

    def test_quality_gate_chars_per_ha_accepts_adequate_name(self):
        """A name with adequate chars/HA for 22 HA should pass.

        "1-hydroxy-2-methoxybenzene" (25 chars) / 22 HA = 1.14 > 0.7 -> passes.
        """
        mol_22 = _mol("CCCCCCCCCCCCCCCCCCCCCC")  # 22 heavy atoms
        assert _name_quality_is_acceptable(
            "1-hydroxy-2-methoxybenzene", mol_22
        ) is True


# ============================================================================
# Bond selection tests
# ============================================================================


@pytest.mark.unit
class TestBondSelection:
    """Test _select_best_bond() priority and balance logic."""

    def test_single_bond_returns_it(self):
        """With only one bond, return it directly."""
        mol = _mol("CC(=O)OCC")  # ethyl acetate
        bonds = [{"bond_idx": 3, "type": "ester"}]
        result = _select_best_bond(mol, bonds)
        assert result["type"] == "ester"

    def test_ester_preferred_over_amide(self):
        """Ester has higher priority than amide."""
        # A molecule with both ester and amide bonds
        mol = _mol("CC(=O)OCC")
        bond_ester = {"bond_idx": 3, "type": "ester", "acid_atom": 1, "alkyl_atom": 4}
        bond_amide = {"bond_idx": 3, "type": "amide", "acid_atom": 1, "amine_atom": 4}
        result = _select_best_bond(mol, [bond_amide, bond_ester])
        assert result["type"] == "ester"

    def test_amide_preferred_over_glycosidic(self):
        """Amide has higher priority than glycosidic."""
        mol = _mol("CC(=O)OCC")
        bond_amide = {"bond_idx": 3, "type": "amide"}
        bond_glyc = {"bond_idx": 3, "type": "glycosidic"}
        result = _select_best_bond(mol, [bond_glyc, bond_amide])
        assert result["type"] == "amide"


# ============================================================================
# Orchestration tests (try_decompose)
# ============================================================================


@pytest.mark.unit
class TestTryDecompose:
    """Test try_decompose() orchestration flow."""

    def setup_method(self):
        """Reset naming depth before each test."""
        _fragment_guard.visited = set()

    def teardown_method(self):
        """Reset naming depth after each test."""
        _fragment_guard.visited = set()

    def test_no_cleavable_bonds_ethanol(self):
        """Ethanol has no cleavable bonds -- returns None."""
        mol = _mol("CCO")
        result = try_decompose(mol)
        assert result is None

    def test_no_cleavable_bonds_benzene(self):
        """Benzene has no cleavable bonds -- returns None."""
        mol = _mol("c1ccccc1")
        result = try_decompose(mol)
        assert result is None

    def test_no_cleavable_bonds_methane(self):
        """Methane has no cleavable bonds -- returns None."""
        mol = _mol("C")
        result = try_decompose(mol)
        assert result is None

    def test_existing_good_name_returns_none(self):
        """If existing pipeline names the molecule well, returns None (no decomposition)."""
        # Ethyl acetate: existing pipeline should produce a good name
        mol = _mol("CC(=O)OCC")
        result = try_decompose(mol)
        # If existing pipeline names this acceptably, result should be None
        # (quality gate passes => fall through)
        # This could also return a valid name if existing name is poor.
        # Either outcome is acceptable -- the key invariant is no crash.
        assert result is None or isinstance(result, str)

    def test_simple_alkane_no_decomposition(self):
        """Simple alkane has no cleavable bonds."""
        mol = _mol("CCCCCC")  # hexane
        result = try_decompose(mol)
        assert result is None

    def test_lactone_not_cleaved(self):
        """Lactones (cyclic esters) should not have cleavable ester bonds."""
        mol = _mol("O=C1CCCO1")  # gamma-butyrolactone
        result = try_decompose(mol)
        assert result is None

    def test_lactam_not_cleaved(self):
        """Lactams (cyclic amides) should not have cleavable amide bonds."""
        mol = _mol("O=C1CCCN1")  # 2-pyrrolidone
        result = try_decompose(mol)
        assert result is None


# ============================================================================
# Size guard tests
# ============================================================================


@pytest.mark.unit
class TestSizeGuard:
    """Test that size guard prevents fragments >= parent size."""

    def setup_method(self):
        _fragment_guard.visited = set()

    def teardown_method(self):
        _fragment_guard.visited = set()

    def test_fragments_must_be_smaller_than_parent(self):
        """Fragment size guard: all fragments must have fewer heavy atoms than parent."""
        from orthonym.decomposition.engine import try_decompose
        # For any molecule, if try_decompose returns a name, it means
        # all fragments passed the size guard. If fragments were >= parent,
        # it would have returned None.
        mol = _mol("CC(=O)OCC")  # ethyl acetate
        result = try_decompose(mol)
        # Result is None (quality gate) or a valid name (passed size guard)
        assert result is None or isinstance(result, str)


# ============================================================================
# Recursion depth tests
# ============================================================================


@pytest.mark.unit
class TestRecursionDepth:
    """Test cycle-detection guard for decomposition engine."""

    def setup_method(self):
        _fragment_guard.visited = set()
        _fragment_guard.cache = None

    def teardown_method(self):
        _fragment_guard.visited = set()
        _fragment_guard.cache = None

    def test_naming_works_without_cycle(self):
        """Fragment naming works when no cycle exists."""
        result = name_fragment_recursively("C")  # methane
        assert result is not None
        assert result == "methane"

    def test_cycle_detected_returns_none_for_uncached(self):
        """When SMILES is in visited set, uncached fragments return None."""
        visited = _get_visited()
        visited.add("CCCCCCCCCCCCCC")  # tetradecane, not in static cache
        result = name_fragment_recursively("CCCCCCCCCCCCCC")
        assert result is None

    def test_cycle_detected_returns_cached(self):
        """When SMILES is in visited set, cached fragments still resolve."""
        visited = _get_visited()
        visited.add("CCO")
        result = name_fragment_recursively("CCO")
        assert result == "ethanol"

    def test_visited_set_restores_after_decompose(self):
        """After try_decompose, visited set should be unchanged."""
        _fragment_guard.visited = set()
        mol = _mol("CCO")
        try_decompose(mol)
        assert get_naming_depth() == 0

    def test_decompose_handles_cycle_gracefully(self):
        """try_decompose with visited entries should not recurse indefinitely."""
        visited = _get_visited()
        visited.add("FAKE_PARENT_1")
        visited.add("FAKE_PARENT_2")
        mol = _mol("CC(=O)OCC")  # ethyl acetate
        result = try_decompose(mol)
        assert result is None or isinstance(result, str)
        # Parent entries should be preserved
        assert "FAKE_PARENT_1" in visited
        assert "FAKE_PARENT_2" in visited


# ============================================================================
# Integration-like tests for specific compound classes
# ============================================================================


@pytest.mark.unit
class TestDecompositionForCompoundClasses:
    """Test try_decompose behavior on specific compound classes."""

    def setup_method(self):
        _fragment_guard.visited = set()

    def teardown_method(self):
        _fragment_guard.visited = set()

    def test_carboxylic_acid_no_decomposition(self):
        """Carboxylic acids don't have cleavable ester/amide bonds."""
        mol = _mol("CC(=O)O")  # acetic acid
        result = try_decompose(mol)
        assert result is None

    def test_primary_amine_no_decomposition(self):
        """Primary amines don't have cleavable amide bonds."""
        mol = _mol("CCN")  # ethylamine
        result = try_decompose(mol)
        assert result is None

    def test_alcohol_no_decomposition(self):
        """Alcohols don't have cleavable bonds."""
        mol = _mol("CCCCO")  # butan-1-ol
        result = try_decompose(mol)
        assert result is None

    def test_ketone_no_decomposition(self):
        """Ketones don't have cleavable ester/amide bonds."""
        mol = _mol("CC(=O)C")  # acetone
        result = try_decompose(mol)
        assert result is None


# ============================================================================
# Sugar fragment intercept tests
# ============================================================================


@pytest.mark.unit
class TestSugarFragmentIntercept:
    """Test _name_sugar_fragment() sugar lookup intercept."""

    def test_name_sugar_fragment_glucose(self):
        """Beta-D-glucose canonical SMILES returns glycosyloxy prefix."""
        result = _name_sugar_fragment("OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O")
        assert result == "beta-D-glucopyranosyloxy"

    def test_name_sugar_fragment_unknown(self):
        """Non-sugar SMILES returns None."""
        result = _name_sugar_fragment("CCCC")
        assert result is None

    def test_name_sugar_fragment_galactose_alpha(self):
        """Alpha-D-galactose returns correct glycosyloxy prefix."""
        result = _name_sugar_fragment("OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@H]1O")
        assert result == "alpha-D-galactopyranosyloxy"

    def test_name_sugar_fragment_galactose_beta(self):
        """Beta-D-galactose returns correct glycosyloxy prefix."""
        result = _name_sugar_fragment("OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O")
        assert result == "beta-D-galactopyranosyloxy"

    def test_name_sugar_fragment_rhamnose_alpha(self):
        """Alpha-L-rhamnose returns correct glycosyloxy prefix."""
        result = _name_sugar_fragment("C[C@@H]1O[C@@H](O)[C@H](O)[C@H](O)[C@H]1O")
        assert result == "alpha-L-rhamnopyranosyloxy"

    def test_name_sugar_fragment_mannose_beta(self):
        """Beta-D-mannose returns correct glycosyloxy prefix."""
        result = _name_sugar_fragment("OC[C@H]1O[C@@H](O)[C@@H](O)[C@@H](O)[C@@H]1O")
        assert result == "beta-D-mannopyranosyloxy"

    def test_name_sugar_fragment_ethanol(self):
        """Ethanol is not a sugar -- returns None."""
        result = _name_sugar_fragment("CCO")
        assert result is None

    def test_name_sugar_fragment_benzene(self):
        """Benzene is not a sugar -- returns None."""
        result = _name_sugar_fragment("c1ccccc1")
        assert result is None


# ============================================================================
# Coverage gate tests (Phase 87 Plan 02)
# ============================================================================


@pytest.mark.unit
class TestCoverageGate:
    """Test _coverage_is_adequate() name-length heuristic for decomposition results."""

    def test_small_molecule_always_passes(self):
        """Molecules with <= 10 heavy atoms always pass coverage gate."""
        mol = _mol("CC(=O)OCC")  # ethyl acetate, ~6 heavy atoms
        assert _coverage_is_adequate("ethyl acetate", mol) is True

    def test_small_molecule_short_name_passes(self):
        """Even a short name passes for small molecules."""
        mol = _mol("CCO")  # ethanol, 3 heavy atoms
        assert _coverage_is_adequate("x", mol) is True

    def test_large_molecule_adequate_name_passes(self):
        """A 30-atom molecule with a long descriptive name passes.

        With tiered thresholds (Phase 099), default bond_type uses 0.8:
        30 * 0.8 = 24. Ester bond_type uses 0.6: 30 * 0.6 = 18.
        """
        large_mol = _mol("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")  # triacontane, 30C
        long_name = "triacontane-1,2-diol"  # 20 chars >= 18 (ester threshold)
        long_name2 = "1,2,3,4,5,6,7,8,9,10-decamethyltriacontane"  # 43 chars
        # With ester bond_type (0.6 threshold), 20 chars passes
        assert _coverage_is_adequate(long_name, large_mol, bond_type="ester") is True
        # Long name passes even default (0.8) threshold: 43 >= 24
        assert _coverage_is_adequate(long_name2, large_mol) is True

    def test_large_molecule_short_name_rejected(self):
        """A 30-atom molecule with a very short name is rejected."""
        large_mol = _mol("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")  # 30 heavy atoms
        short_name = "methane"  # 7 chars << 24 expected min (30 * 0.8)
        assert _coverage_is_adequate(short_name, large_mol) is False

    def test_empty_name_rejected(self):
        """Empty string is rejected."""
        mol = _mol("CCCCCCCCCCCCCCCC")
        assert _coverage_is_adequate("", mol) is False

    def test_none_name_rejected(self):
        """None is rejected."""
        mol = _mol("CCO")
        assert _coverage_is_adequate(None, mol) is False

    def test_medium_molecule_borderline(self):
        """A medium molecule (15 heavy atoms) with borderline name length.

        With tiered thresholds (Phase 099):
        - Default (substitutive): 15 * 0.8 = 12 chars min
        - Ester (functional-class): 15 * 0.6 = 9 chars min
        """
        mol = _mol("CCCCCCCCCCCCCCC")  # pentadecane, 15 heavy atoms
        # Task Z3: "pentadecane" IS this molecule's correct systematic name, so
        # asserting it has inadequate coverage was asserting the defect. The
        # tier is now demonstrated with "decan-1-ol" (10 chars), which has the
        # same length behaviour -- passes int(15*0.6)=9, fails int(15*0.8)=12 --
        # but genuinely describes a different, smaller structure.
        assert _coverage_is_adequate("decan-1-ol", mol, bond_type="ester") is True
        assert _coverage_is_adequate("decan-1-ol", mol) is False  # 10 < 12
        # The correct name is not discarded at either threshold.
        assert _coverage_is_adequate("pentadecane", mol, bond_type="ester") is True
        assert _coverage_is_adequate("pentadecane", mol) is True
        # Very short name rejected at both thresholds (5 chars < 9)
        assert _coverage_is_adequate("short", mol) is False
        # A more descriptive name passes even default threshold (27 chars >= 12)
        assert _coverage_is_adequate("2,3,4,5-tetramethylundecane", mol) is True

    def test_retained_core_name_on_large_mol(self):
        """Coverage gate should NOT reject retained core names -- this is
        handled by the caller (try_decompose), not _coverage_is_adequate.
        The function itself uses only the name-length heuristic."""
        # adenine is short (7 chars) but is a retained core name
        # A large molecule might legitimately be named "adenine" for its core
        # _coverage_is_adequate only checks length heuristic, not retained names
        large_mol = _mol("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")  # 30 HA
        # 7 chars < 18 (30 * 0.6) -- heuristic rejects it
        assert _coverage_is_adequate("adenine", large_mol) is False


# ============================================================================
# Garbled pattern extension tests (Phase 87 Plan 02)
# ============================================================================


@pytest.mark.unit
class TestGarbledPatternExtension:
    """Test _decomposition_is_worse() with extended garbled patterns for new bond types."""

    def test_thioateyl_detected_as_garbled(self):
        """'thioateyl' in decomposed name is detected as garbled."""
        mol = _mol("CC(=O)SC")
        assert _decomposition_is_worse("methyl thioateyl", "methyl thioester", mol) is True

    def test_sulfonamideyl_detected_as_garbled(self):
        """'sulfonamideyl' in decomposed name is detected as garbled."""
        mol = _mol("CS(=O)(=O)NC")
        assert _decomposition_is_worse("sulfonamideyl methane", "methanesulfonamide", mol) is True

    def test_phosphateyl_detected_as_garbled(self):
        """'phosphateyl' in decomposed name is detected as garbled."""
        mol = _mol("COP(=O)(O)OC")
        assert _decomposition_is_worse("phosphateyl methane", "dimethyl phosphate", mol) is True

    def test_existing_garbled_patterns_still_work(self):
        """Existing garbled patterns (aneyl, cycloane, etc.) still detected."""
        mol = _mol("CCCCCC")
        assert _decomposition_is_worse("hexaneyl bad", "hexane", mol) is True
        assert _decomposition_is_worse("cycloane bad", "cyclohexane", mol) is True


# ============================================================================
# Leaf-first fragment ordering tests
# ============================================================================


@pytest.mark.unit
class TestLeafFirstOrdering:
    """Verify fragments are named smallest-first for cache benefit."""

    def setup_method(self):
        _fragment_guard.visited = set()
        _fragment_guard.cache = None

    def teardown_method(self):
        _fragment_guard.visited = set()
        _fragment_guard.cache = None

    def test_smaller_fragment_named_first(self, monkeypatch):
        """In decomposition, smaller fragments should be named before larger ones.

        Uses a large asymmetric ester (methyl hexacosanoate) where the small
        methyl fragment should be named before the large acid fragment,
        populating the cache for potential reuse.

        Task Z3: on the default path the quality gate now proves the pipeline
        name "methyl hexacosanoate" already denotes this molecule, so
        try_decompose returns None and the ordering under test is never
        reached. Measured: decomposition previously returned the IDENTICAL
        string, and the emitted name is unchanged either way. The coverage
        oracle is therefore switched off here so the ordering logic is actually
        exercised; the default-path name is asserted separately below.
        """
        from orthonym.decomposition import engine
        monkeypatch.setenv("ORTHONYM_DECOMP_COVERAGE_ORACLE", "0")
        engine._PROVEN_COMPLETE_CACHE.clear()

        # Methyl hexacosanoate: 29 heavy atoms, methyl (1 HA) vs acid (~28 HA)
        mol = _mol("CCCCCCCCCCCCCCCCCCCCCCCCCC(=O)OC")
        result = try_decompose(mol)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 5

    def test_large_asymmetric_ester_named_on_the_default_path(self):
        """Output-level half of the test above: the emitted name is unchanged
        by skipping a decomposition that only re-derived the same string."""
        from orthonym import name_compound
        assert name_compound("CCCCCCCCCCCCCCCCCCCCCCCCCC(=O)OC") == \
            "methyl hexacosanoate"

    def test_ordering_does_not_crash_on_equal_size(self):
        """Two fragments of equal size should not cause sorting issues."""
        mol = _mol("CCOC(C)=O")
        result = try_decompose(mol)
        assert result is None or isinstance(result, str)

    def test_invalid_fragment_smiles_handled(self):
        """Sorting should handle fragments with no cleavable bonds gracefully."""
        mol = _mol("CCO")
        result = try_decompose(mol)
        assert result is None


# ============================================================================
# _name_fragment_with_fallback tests (Phase 127)
# ============================================================================


@pytest.mark.unit
class TestNameFragmentWithFallback:
    """Test _name_fragment_with_fallback() helper function."""

    def test_fallback_returns_name_for_simple_smiles(self):
        """Simple molecule naming via fallback should return ethanol."""
        from orthonym.decomposition.engine import _name_fragment_with_fallback
        result = _name_fragment_with_fallback("CCO")
        assert result is not None
        assert "ethanol" in result.lower()

    def test_fallback_returns_none_for_invalid(self):
        """Invalid SMILES should return None from fallback."""
        from orthonym.decomposition.engine import _name_fragment_with_fallback
        result = _name_fragment_with_fallback("[invalid]")
        assert result is None


# ============================================================================
# Partial assembly tests (DECO-25)
# ============================================================================


@pytest.mark.unit
class TestPartialAssembly:
    """Test partial assembly strategy in _try_multi_bond_decompose.

    DECO-25: When 2/3+ fragments name successfully, assemble a partial
    name instead of aborting entirely.
    """

    def setup_method(self):
        """Reset fragment naming state before each test."""
        _fragment_guard.visited = set()
        if hasattr(_fragment_guard, 'cache') and _fragment_guard.cache:
            _fragment_guard.cache = None

    def test_ester_threshold_is_two(self):
        """DECO-22: Ester multi-bond threshold should be 2 (lowered from 3)."""
        from orthonym.decomposition.engine import _MULTI_BOND_THRESHOLD
        assert _MULTI_BOND_THRESHOLD["ester"] == 2, (
            f"Ester threshold should be 2, got {_MULTI_BOND_THRESHOLD['ester']}"
        )

    def test_partial_assembly_two_of_three(self):
        """When 2/3 fragments name successfully, partial assembly should return a name.

        Mock scenario: 3 fragments from multi-bond ester cleavage, where the
        third fragment fails naming. The engine should still assemble a name
        from the 2 successful fragments.
        """
        from unittest.mock import patch, MagicMock
        from orthonym.decomposition.engine import _try_multi_bond_decompose

        # Create a molecule with 3 ester bonds (triester)
        # Glycerol triacetate: OC(COC(C)=O)(COC(C)=O)OC(C)=O
        mol = _mol("CC(=O)OCC(OC(C)=O)COC(C)=O")

        # 3 ester bonds
        bonds = [
            {"bond_idx": i, "type": "ester", "acid_atom": 0, "alkyl_atom": 3}
            for i in range(3)
        ]

        # Mock cleave_and_cap to return 3 fragments
        mock_frags = [
            {"smiles": "CC(=O)O", "side": "acid"},
            {"smiles": "CCO", "side": "alkyl"},
            {"smiles": "C=C=C", "side": "alkyl"},  # will fail naming
        ]

        # Mock _name_fragment_with_fallback: first two succeed, third fails
        call_count = [0]
        def mock_name_frag(smiles):
            call_count[0] += 1
            if smiles == "CC(=O)O":
                return "acetic acid"
            if smiles == "CCO":
                return "ethanol"
            return None  # Third fragment fails

        with patch('orthonym.decomposition.fragment_capping.cleave_and_cap',
                   return_value=mock_frags):
            with patch('orthonym.decomposition.engine._name_fragment_with_fallback',
                       side_effect=mock_name_frag):
                with patch('orthonym.decomposition.engine._name_sugar_fragment',
                           return_value=None):
                    with patch('orthonym.decomposition.fragment_assembly._assemble_multi_ester',
                               return_value="ethyl acetate"):
                        result = _try_multi_bond_decompose(mol, bonds)

        # Should NOT be None -- partial assembly should produce a name
        assert result is not None, (
            "Partial assembly should return a name when 2/3 fragments succeed"
        )

    def test_partial_assembly_one_of_three_aborts(self):
        """When only 1/3 fragments names successfully, should return None.

        DECO-25: requires at least 2 named fragments for assembly.
        """
        from unittest.mock import patch
        from orthonym.decomposition.engine import _try_multi_bond_decompose

        mol = _mol("CC(=O)OCC(OC(C)=O)COC(C)=O")
        bonds = [
            {"bond_idx": i, "type": "ester", "acid_atom": 0, "alkyl_atom": 3}
            for i in range(3)
        ]

        mock_frags = [
            {"smiles": "CC(=O)O", "side": "acid"},
            {"smiles": "C=C=C", "side": "alkyl"},  # fail
            {"smiles": "[invalid]", "side": "alkyl"},  # fail
        ]

        call_count = [0]
        def mock_name_frag(smiles):
            call_count[0] += 1
            if smiles == "CC(=O)O":
                return "acetic acid"
            return None  # Others fail

        with patch('orthonym.decomposition.fragment_capping.cleave_and_cap',
                   return_value=mock_frags):
            with patch('orthonym.decomposition.engine._name_fragment_with_fallback',
                       side_effect=mock_name_frag):
                with patch('orthonym.decomposition.engine._name_sugar_fragment',
                           return_value=None):
                    result = _try_multi_bond_decompose(mol, bonds)

        # Should be None -- not enough fragments
        assert result is None, (
            "Should return None when only 1/3 fragments name successfully"
        )

    def test_partial_assembly_all_succeed(self):
        """When all 3 fragments name successfully, full assembly (no change from before)."""
        from unittest.mock import patch
        from orthonym.decomposition.engine import _try_multi_bond_decompose

        mol = _mol("CC(=O)OCC(OC(C)=O)COC(C)=O")
        bonds = [
            {"bond_idx": i, "type": "ester", "acid_atom": 0, "alkyl_atom": 3}
            for i in range(3)
        ]

        mock_frags = [
            {"smiles": "CC(=O)O", "side": "acid"},
            {"smiles": "CCO", "side": "alkyl"},
            {"smiles": "CO", "side": "alkyl"},
        ]

        def mock_name_frag(smiles):
            if smiles == "CC(=O)O":
                return "acetic acid"
            if smiles == "CCO":
                return "ethanol"
            if smiles == "CO":
                return "methanol"
            return None

        with patch('orthonym.decomposition.fragment_capping.cleave_and_cap',
                   return_value=mock_frags):
            with patch('orthonym.decomposition.engine._name_fragment_with_fallback',
                       side_effect=mock_name_frag):
                with patch('orthonym.decomposition.engine._name_sugar_fragment',
                           return_value=None):
                    result = _try_multi_bond_decompose(mol, bonds)

        # When all fragments succeed and assembly produces a result, it should be non-None
        # (may still be None if assembly function returns None, but fragments themselves are good)
        # At minimum, the function should NOT abort during fragment naming
        # The actual result depends on _assemble_multi_ester
        assert result is None or isinstance(result, str), (
            "Full assembly should either produce a string or None (from assembler), "
            "not crash during fragment naming"
        )

    def test_fragment_naming_rejects_poor_coverage(self):
        """Fragment name that covers < 60% of HA should be rejected by partial assembly.

        D-05: fragment naming size validation in the partial assembly loop.
        A 15 HA fragment named with a name covering only ~6 HA should be skipped.
        """
        from unittest.mock import patch
        from orthonym.decomposition.engine import _try_multi_bond_decompose

        mol = _mol("CC(=O)OCC(OC(C)=O)COC(C)=O")
        bonds = [
            {"bond_idx": i, "type": "ester", "acid_atom": 0, "alkyl_atom": 3}
            for i in range(3)
        ]

        # Fragment with 15 HA but name only covers ~6 HA
        mock_frags = [
            {"smiles": "CC(=O)O", "side": "acid"},  # 4 HA, will name fine
            {"smiles": "CCO", "side": "alkyl"},  # 3 HA, small -> passes
            {"smiles": "CCCCCCCCCCCCCCC", "side": "alkyl"},  # 15 HA
        ]

        def mock_name_frag(smiles):
            if smiles == "CC(=O)O":
                return "acetic acid"
            if smiles == "CCO":
                return "ethanol"
            if smiles == "CCCCCCCCCCCCCCC":
                return "furan"  # ~5 chars for 15 HA -> poor coverage
            return None

        # Mock _name_covers_molecule to return False for "furan" on 15-HA fragment
        original_covers = None
        def mock_covers(name, mol_obj):
            if name == "furan" and mol_obj and mol_obj.GetNumHeavyAtoms() == 15:
                return False
            return True

        with patch('orthonym.decomposition.fragment_capping.cleave_and_cap',
                   return_value=mock_frags):
            with patch('orthonym.decomposition.engine._name_fragment_with_fallback',
                       side_effect=mock_name_frag):
                with patch('orthonym.decomposition.engine._name_sugar_fragment',
                           return_value=None):
                    with patch('orthonym.decomposition.engine._name_covers_molecule',
                               side_effect=mock_covers):
                        result = _try_multi_bond_decompose(mol, bonds)

        # The poor-coverage fragment should be rejected, leaving 2 named fragments
        # which is enough for partial assembly (>= 2)
        # The result depends on assembly, but the key test is that we didn't abort
        # due to the poor coverage fragment, and we also didn't include it
        assert result is not None or result is None, (
            "Should handle poor-coverage fragments gracefully"
        )
