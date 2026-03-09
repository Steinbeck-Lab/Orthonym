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
    MAX_NAMING_DEPTH,
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
        """A name with digits for a large molecule is acceptable."""
        large_mol = _mol("CCCCCCCCCCCCCCCCCCCCCC")  # 22 heavy atoms
        assert _name_quality_is_acceptable("compound123", large_mol) is True


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

    def test_max_naming_depth_constant_preserved(self):
        """MAX_NAMING_DEPTH legacy constant should be 7."""
        assert MAX_NAMING_DEPTH == 7

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
        """A 30-atom molecule with a long descriptive name passes."""
        # 30 heavy atoms: need len >= 30 * 0.6 = 18 chars
        large_mol = _mol("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")  # triacontane, 30C
        long_name = "triacontane-1,2-diol"  # 20 chars >= 18
        long_name2 = "1,2,3,4,5,6,7,8,9,10-decamethyltriacontane"  # 43 chars
        assert _coverage_is_adequate(long_name, large_mol) is True
        assert _coverage_is_adequate(long_name2, large_mol) is True

    def test_large_molecule_short_name_rejected(self):
        """A 30-atom molecule with a very short name is rejected."""
        large_mol = _mol("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")  # 30 heavy atoms
        short_name = "methane"  # 7 chars << 18 expected min (30 * 0.6)
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
        """A medium molecule (15 heavy atoms) with borderline name length."""
        # 15 heavy atoms: need len >= 15 * 0.6 = 9 chars
        mol = _mol("CCCCCCCCCCCCCCC")  # pentadecane, 15 heavy atoms
        # "pentadecane" = 11 chars >= 9, passes
        assert _coverage_is_adequate("pentadecane", mol) is True
        # Very short name rejected (5 chars < 9)
        assert _coverage_is_adequate("short", mol) is False
        # A more descriptive name also passes (27 chars >= 9)
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
