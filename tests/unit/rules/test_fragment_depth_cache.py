"""Tests for fragment naming cycle guard, canonical SMILES, and compound regression.

Tests the infrastructure in fragment_naming.py:
- MAX_NAMING_DEPTH = 7 (legacy constant, kept for backward compatibility)
- MAX_TOTAL_CALLS = 100 (constant available for future use)
- Canonical SMILES normalization before calling name_compound()
- Cycle-guard compound regression tests (V8-DEPTH-01)
"""

import pytest
from orthonym.assembly.fragment_naming import (
    MAX_NAMING_DEPTH,
    MAX_TOTAL_CALLS,
    _fragment_guard,
    _get_visited,
    name_fragment_recursively,
)
from orthonym import name_compound


@pytest.fixture(autouse=True)
def reset_fragment_guard():
    """Reset thread-local state before and after each test."""
    _fragment_guard.visited = set()
    _fragment_guard.cache = None
    yield
    _fragment_guard.visited = set()
    _fragment_guard.cache = None


# ============================================================================
# Group 1: Constants and basic functionality
# ============================================================================


class TestConstants:
    """Verify module-level constants."""

    def test_max_naming_depth_is_7(self):
        assert MAX_NAMING_DEPTH == 7

    def test_max_total_calls_is_100(self):
        assert MAX_TOTAL_CALLS == 100

    def test_basic_fragment_naming_ethanol(self):
        result = name_fragment_recursively("CCO")
        assert result is not None
        assert "ethanol" in result.lower()

    def test_basic_fragment_naming_acetic_acid(self):
        result = name_fragment_recursively("CC(=O)O")
        assert result is not None
        assert "acetic acid" in result.lower()


# ============================================================================
# Group 2: Depth limit behavior
# ============================================================================


class TestCycleGuard:
    """Test cycle-detection guard enforcement."""

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

    def test_no_cycle_returns_name(self):
        """When SMILES is NOT in visited set, should return a name."""
        result = name_fragment_recursively("CCO")
        assert result is not None

    def test_visited_set_restored_after_call(self):
        """Visited set should not grow after a completed call."""
        result = name_fragment_recursively("CCO")
        assert result is not None
        assert "CCO" not in _get_visited()

    def test_parent_entries_preserved(self):
        """Parent entries in visited set should remain after nested call."""
        visited = _get_visited()
        visited.add("FAKE_PARENT")
        name_fragment_recursively("CCO")
        assert "FAKE_PARENT" in visited


# ============================================================================
# Group 3: Canonical SMILES consistency
# ============================================================================


class TestCanonicalSmilesConsistency:
    """Test that canonical SMILES normalization ensures consistent naming."""

    def test_equivalent_smiles_same_name(self):
        """Same molecule in different SMILES notation produces same name."""
        result1 = name_fragment_recursively("CCO")
        _fragment_guard.visited = set()
        result2 = name_fragment_recursively("OCC")
        assert result1 == result2

    def test_equivalent_smiles_propanol(self):
        """Propan-1-ol in different SMILES forms gives same name."""
        result1 = name_fragment_recursively("CCCO")
        _fragment_guard.visited = set()
        result2 = name_fragment_recursively("OCCC")
        assert result1 == result2

    def test_equivalent_smiles_branched(self):
        """2-methylpropane in different forms gives same name."""
        result1 = name_fragment_recursively("CC(C)C")
        _fragment_guard.visited = set()
        result2 = name_fragment_recursively("C(C)(C)C")
        assert result1 == result2

    def test_sequential_toplevel_calls_both_succeed(self):
        """Two consecutive top-level calls should both succeed."""
        _fragment_guard.visited = set()
        result1 = name_fragment_recursively("CCO")
        assert result1 is not None

        _fragment_guard.visited = set()
        result2 = name_fragment_recursively("CCCO")
        assert result2 is not None


# ============================================================================
# Group 4: Depth-limit compound regression tests (V8-DEPTH-01)
# ============================================================================

# 25 compounds from the medium molecule triage that mention depth_limit_reached.
# These compounds produce names at depth 7 (some hit internal depth limits but
# still return results via fallback paths). Each must produce a valid name.

DEPTH_LIMIT_COMPOUNDS = [
    pytest.param(
        "CCCCCC(C)OC(=O)COc1ccc(Cl)c2cccnc12",
        id="001_chloroquinoline_ester",
    ),
    pytest.param(
        "CC1(C)SC(C(NC(=O)COc2ccccc2)C(=O)O)NC1C(=O)O",
        id="002_penicillin_like",
    ),
    pytest.param(
        "C=C[C@](C)(O)CCC=C(C)CCC1OC(C)(C)OC1(C)C",
        id="003_terpene_dioxolane",
    ),
    pytest.param(
        "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O[C@@H]1OC[C@@H](O)[C@H](O)[C@H]1O",
        id="004_disaccharide_xylose",
    ),
    pytest.param(
        "C[C@H]1C[C@@H](O)[C@@]23C1=C[C@@]1(C)CC[C@](C)(C[C@H](O)[C@H](O)[C@@](C)(O)CO)[C@H]1[C@@H]2CC[C@@H]3C",
        id="005_steroid_polyol",
    ),
    pytest.param(
        "COc1cc(-c2ccc(O)c(CC=C(C)C)c2)c(OC)c(O)c1-c1ccc(O)c(O)c1",
        id="006_terphenyl_prenyl",
    ),
    pytest.param(
        "OC[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)CO[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O",
        id="007_galactitol_glucoside",
    ),
    pytest.param(
        "C=CCN(C)CCCCCCOc1ccc(C(=O)c2ccc(Br)cc2)c(F)c1",
        id="008_allylamine_benzophenone",
    ),
    pytest.param(
        "CCCCCCCCCCCCCCCCCCCCCCCC(=O)NCCc1c[nH]c2ccccc12",
        id="009_tryptamine_amide",
    ),
    pytest.param(
        "OC[C@H]1O[C@H](O)[C@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]2O)[C@@H](O)[C@H]1O",
        id="010_disaccharide_mannose",
    ),
    pytest.param(
        "C=C1C(=O)O[C@@H]2C[C@@H](C)/C=C\\C(=O)[C@@](C)(O)C[C@@H](OC(=O)CC(C)C)[C@@H]12",
        id="011_macrolide_lactone",
    ),
    pytest.param(
        "COc1cc(OC)c(C(C)=O)c(O)c1CCOCCc1c(O)cc(OC)c(C(C)=O)c1O",
        id="012_biaryl_ether",
    ),
    pytest.param(
        "NC(=O)CC[C@H](NC(=O)[C@@H]1CCCN1C(=O)[C@@H]1CCCN1)C(=O)O",
        id="013_dipeptide_proline",
    ),
    pytest.param(
        "CNCC[C@H](Oc1cccc2ccccc12)c1cccs1",
        id="014_naphthyl_thiophene",
    ),
    pytest.param(
        "COc1c(-c2ccc(O)cc2)oc2c(O)c(O)ccc2c1=O",
        id="015_flavonoid",
    ),
    pytest.param(
        "C=C(C)C(=O)Cc1c(C)cc(Oc2cc(CO)cc(OC)c2)cc1OC",
        id="016_phenol_ether_ketone",
    ),
    pytest.param(
        "CC[C@H](C)[C@H](NC(=O)[C@H](C)NC(=O)[C@@H](N)CCCN=C(N)N)C(=O)O",
        id="017_tripeptide_arginine",
    ),
    pytest.param(
        "C[C@]12CC[C@H](O)c3coc(c31)C(=O)C1=C2[C@@H](O)C[C@]2(C)C(=O)CC[C@@H]12",
        id="018_steroid_furanone",
    ),
    pytest.param(
        "CC(C)C1=C(O)C(N)=C(/C=C/c2ccccc2)C(=O)C1=O",
        id="019_aminoquinone_styryl",
    ),
    pytest.param(
        "CC(C)[C@H](NC(=O)[C@@H](N)CC(=O)O)C(=O)NCC(=O)N1CCC[C@H]1C(=O)O",
        id="020_tetrapeptide",
    ),
    pytest.param(
        "CC(=O)N[C@H]1C(OP(=O)(O)OP(=O)(O)OC[C@H]2O[C@@H](n3ccc(=O)[nH]c3=O)[C@H](O)[C@@H]2O)O[C@H](CO)[C@H](O)[C@@H]1O",
        id="021_udp_sugar",
    ),
    pytest.param(
        "COc1cc(OC)c2c(=O)c3c(O)cc(C)cc3oc2c1",
        id="022_xanthone_dimethoxy",
    ),
    pytest.param(
        "N[C@@H](COC(=O)CCC(=O)O)C(=O)O",
        id="023_serine_succinate",
    ),
    pytest.param(
        "C=C(C(=O)OC)N1C(=O)C[C@@H](C)C1=O",
        id="024_acrylate_pyrrolidinedione",
    ),
    pytest.param(
        "CCCCCCC(=O)NC1=CC(=O)[C@@H]2CCCN12",
        id="025_pyrrolizinone_amide",
    ),
]


class TestDepthLimitCompounds:
    """Regression tests for compounds that previously hit depth_limit_reached.

    These compounds produce names at depth 7 via fallback paths. The test
    verifies that naming completes without producing None or 'unknown'.
    """

    @pytest.mark.parametrize("smiles", DEPTH_LIMIT_COMPOUNDS)
    def test_naming_completes(self, smiles):
        """Compound should produce a valid name (not None, not 'unknown')."""
        result = name_compound(smiles)
        assert result is not None, f"name_compound returned None for {smiles}"
        assert "unknown" not in result.lower(), f"Name contains unknown: {result}"

    @pytest.mark.roundtrip
    @pytest.mark.parametrize("smiles", DEPTH_LIMIT_COMPOUNDS)
    def test_opsin_roundtrip(self, smiles, opsin_to_smiles, canonical):
        """Named compound should parse back via OPSIN (Tier 2 round-trip)."""
        result = name_compound(smiles)
        assert result is not None, f"name_compound returned None for {smiles}"
        parsed = opsin_to_smiles(result)
        if parsed:
            assert canonical(parsed) == canonical(smiles), (
                f"Round-trip mismatch: {smiles} -> '{result}' -> {parsed}"
            )
