#!/usr/bin/env python3
"""Unit tests for the 5-component graded scoring engine.

Tests cover:
- Tier 1 (OPSIN RT) scoring: parent, substituent, locant, stereo
- Tier 2 (pipeline-internal) scoring: parent, substituent, locant, stereo
- Two-tier dispatch function (score_compound)
- CSV export with correct 15-column header
- Edge cases: naming failure, no stereo, invalid SMILES, empty name
- Helper functions: get_cip_labels, count_name_stereo_descriptors, tanimoto_morgan
"""

import csv
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from benchmark_graded import (
    FIELDNAMES,
    count_name_stereo_descriptors,
    get_cip_labels,
    score_compound,
    score_locant_correctness,
    score_locant_internal,
    score_parent_internal,
    score_parent_structure,
    score_stereo_correctness,
    score_stereo_internal,
    score_substituent_completeness,
    score_substituent_internal,
    tanimoto_morgan,
    write_csv,
)

# ---------------------------------------------------------------------------
# Real SMILES for testing
# ---------------------------------------------------------------------------
ETHANOL = "CCO"
BENZENE = "c1ccccc1"
HEXANE = "CCCCCC"
PHENOL = "c1ccc(O)cc1"
ANILINE = "c1ccc(N)cc1"
ALANINE_S = "N[C@@H](C)C(=O)O"  # L-alanine with (S) stereocenter


# ===========================================================================
# Tier 1: Parent structure scoring
# ===========================================================================

@pytest.mark.unit
class TestParentScoreTier1:
    """Tests for score_parent_structure (Tier 1 OPSIN-based)."""

    def test_parent_score_tier1_ring_match(self):
        """Identical ring molecule returns 1."""
        assert score_parent_structure(BENZENE, BENZENE) == 1

    def test_parent_score_tier1_chain_match(self):
        """Identical chain molecule returns 1."""
        assert score_parent_structure(HEXANE, HEXANE) == 1

    def test_parent_score_tier1_mismatch(self):
        """Ring vs chain returns 0 (cross-match)."""
        assert score_parent_structure(BENZENE, HEXANE) == 0

    def test_parent_score_tier1_similar(self):
        """Hexane vs heptane: Tanimoto >= 0.8 and both are chains -> 1."""
        # Tanimoto(hexane, heptane) = 0.875, both chain molecules
        result = score_parent_structure(HEXANE, "CCCCCCC")
        assert result == 1


# ===========================================================================
# Tier 1: Substituent completeness scoring
# ===========================================================================

@pytest.mark.unit
class TestSubstituentScore:
    """Tests for score_substituent_completeness (Tier 1 OPSIN-based)."""

    def test_substituent_score_identical(self):
        """Same molecule returns 1.0."""
        assert score_substituent_completeness(HEXANE, HEXANE) == 1.0

    def test_substituent_score_partial(self):
        """RT with fewer heavy atoms returns ratio < 1.0."""
        # ethanol (3 HA) vs methanol "CO" (2 HA) -> 2/3 ~ 0.667
        result = score_substituent_completeness("CCO", "CO")
        assert 0.0 < result < 1.0
        assert abs(result - 2.0 / 3.0) < 0.01

    def test_substituent_score_hallucinated(self):
        """RT with more heavy atoms returns 1.0 (clamped)."""
        # methanol (2 HA) original, ethanol (3 HA) RT -> min(3/2, 1.0) = 1.0
        result = score_substituent_completeness("CO", "CCO")
        assert result == 1.0


# ===========================================================================
# Tier 1: Locant correctness scoring
# ===========================================================================

@pytest.mark.unit
class TestLocantScore:
    """Tests for score_locant_correctness (Tier 1 OPSIN-based)."""

    def test_locant_correct(self):
        """Same stereo-stripped SMILES returns 1."""
        assert score_locant_correctness(HEXANE, HEXANE) == 1

    def test_locant_incorrect(self):
        """Different stereo-stripped SMILES returns 0."""
        assert score_locant_correctness(BENZENE, HEXANE) == 0


# ===========================================================================
# Tier 1: Stereo correctness scoring
# ===========================================================================

@pytest.mark.unit
class TestStereoScore:
    """Tests for score_stereo_correctness (Tier 1 OPSIN-based)."""

    def test_stereo_correct(self):
        """Identical CIP labels returns 1."""
        assert score_stereo_correctness(ALANINE_S, ALANINE_S) == 1

    def test_stereo_mismatch(self):
        """Different CIP labels returns 0."""
        alanine_r = "N[C@H](C)C(=O)O"  # R-alanine
        assert score_stereo_correctness(ALANINE_S, alanine_r) == 0

    def test_stereo_no_stereo(self):
        """No stereocenters in original returns 1 (vacuously correct,)."""
        assert score_stereo_correctness(HEXANE, HEXANE) == 1


# ===========================================================================
# Tier 2: Pipeline-internal scoring
# ===========================================================================

@pytest.mark.unit
class TestTier2Scoring:
    """Tests for Tier 2 pipeline-internal scoring functions."""

    def test_tier2_parent_high_coverage(self):
        """atom_coverage >= 0.5 returns 1."""
        assert score_parent_internal({"atom_coverage": 0.6}) == 1

    def test_tier2_parent_low_coverage(self):
        """atom_coverage < 0.5 returns 0."""
        assert score_parent_internal({"atom_coverage": 0.3}) == 0

    def test_tier2_substituent(self):
        """substituent_completeness is passed through."""
        assert score_substituent_internal({"substituent_completeness": 0.7}) == 0.7

    def test_tier2_locant_passes(self):
        """High coverage + completeness returns 1."""
        assert score_locant_internal({"atom_coverage": 0.8, "substituent_completeness": 0.6}) == 1

    def test_tier2_locant_fails(self):
        """Low coverage + completeness returns 0."""
        assert score_locant_internal({"atom_coverage": 0.3, "substituent_completeness": 0.2}) == 0

    def test_tier2_stereo_present(self):
        """Stereocenters + matching descriptor count returns 1."""
        # ALANINE_S has 1 stereocenter, name has 1 descriptor "(2S)"
        result = score_stereo_internal(ALANINE_S, "(2S)-2-aminopropanoic acid")
        assert result == 1

    def test_tier2_stereo_missing(self):
        """Stereocenters but no descriptors in name returns 0."""
        result = score_stereo_internal(ALANINE_S, "2-aminopropanoic acid")
        assert result == 0


# ===========================================================================
# Two-tier dispatch: score_compound
# ===========================================================================

@pytest.mark.unit
class TestScoreCompound:
    """Tests for the two-tier dispatch function."""

    def test_score_compound_naming_failure(self):
        """Naming failure (None name) returns all zeros."""
        result = score_compound(
            orig_smi=HEXANE,
            generated_name=None,
            rt_smi=None,
            opsin_error="",
            confidence_data={},
        )
        assert result["parent_score"] == 0
        assert result["substituent_score"] == 0.0
        assert result["locant_score"] == 0
        assert result["stereo_score"] == 0
        assert result["opsin_score"] == 0
        assert result["graded_total"] == 0.0
        assert result["scoring_tier"] == "failure"

    def test_score_compound_tier1(self):
        """With rt_smi set, uses Tier 1 scoring."""
        result = score_compound(
            orig_smi=HEXANE,
            generated_name="hexane",
            rt_smi=HEXANE,
            opsin_error="",
            confidence_data={},
        )
        assert result["scoring_tier"] == "opsin"
        assert result["opsin_score"] == 1
        assert result["parent_score"] == 1
        assert result["graded_total"] > 0

    def test_score_compound_tier2(self):
        """With rt_smi=None, uses Tier 2 scoring."""
        conf_data = {
            "factors": {
                "atom_coverage": 0.8,
                "substituent_completeness": 0.6,
            }
        }
        result = score_compound(
            orig_smi=HEXANE,
            generated_name="hexane",
            rt_smi=None,
            opsin_error="parse failed",
            confidence_data=conf_data,
        )
        assert result["scoring_tier"] == "internal"
        assert result["opsin_score"] == 0


# ===========================================================================
# CSV export
# ===========================================================================

@pytest.mark.unit
class TestCSVExport:
    """Tests for CSV export function."""

    def test_csv_export(self, tmp_path):
        """write_csv produces file with correct column headers."""
        results = [
            {
                "smiles": HEXANE,
                "generated_name": "hexane",
                "handler": "chain",
                "parent_score": 1,
                "substituent_score": 1.0,
                "locant_score": 1,
                "stereo_score": 1,
                "opsin_score": 1,
                "graded_total": 5.0,
                "inchi_rt": 1,
                "scoring_tier": "opsin",
                "compound_classes": "acyclic",
                "heavy_atoms": 6,
                "opsin_error": "",
                "naming_error": "",
            }
        ]
        csv_path = str(tmp_path / "test_output.csv")
        write_csv(results, csv_path)

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)

        # specifies 15 columns
        assert len(headers) == 15, f"Expected 15 columns, got {len(headers)}: {headers}"
        assert headers == FIELDNAMES
        assert len(rows) == 1


# ===========================================================================
# Edge cases
# ===========================================================================

@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases."""

    def test_edge_case_invalid_smiles(self):
        """Scoring functions return 0 for invalid SMILES without raising."""
        assert score_parent_structure("NOT_A_SMILES", HEXANE) == 0
        assert score_substituent_completeness("NOT_A_SMILES", HEXANE) == 0.0
        assert score_locant_correctness("NOT_A_SMILES", HEXANE) == 0
        assert score_stereo_correctness("NOT_A_SMILES", HEXANE) == 0

    def test_edge_case_empty_name(self):
        """Empty string name returns all zeros."""
        result = score_compound(
            orig_smi=HEXANE,
            generated_name="",
            rt_smi=None,
            opsin_error="",
            confidence_data={},
        )
        assert result["graded_total"] == 0.0
        assert result["scoring_tier"] == "failure"

    def test_ha_ratio_zero_guard(self):
        """Substituent scoring with 0 HA original does not raise ZeroDivisionError."""
        # "[H]" is hydrogen-only (0 heavy atoms)
        result = score_substituent_completeness("[H]", "CCO")
        assert isinstance(result, float)
        # Should not raise


# ===========================================================================
# Helper functions
# ===========================================================================

@pytest.mark.unit
class TestHelpers:
    """Tests for helper functions."""

    def test_count_name_stereo_descriptors_single(self):
        """Single stereodescriptor detected."""
        assert count_name_stereo_descriptors("(2S)-2-aminobutanedioic acid") == 1

    def test_count_name_stereo_descriptors_multiple(self):
        """Multiple stereodescriptors detected."""
        assert count_name_stereo_descriptors("(2R,3S)-tartaric acid") == 2

    def test_count_name_stereo_descriptors_none(self):
        """No stereodescriptors in name."""
        assert count_name_stereo_descriptors("hexane") == 0

    def test_get_cip_labels(self):
        """get_cip_labels returns expected labels for known stereo molecule."""
        atom_labels, bond_labels = get_cip_labels(ALANINE_S)
        assert "S" in atom_labels
        assert len(atom_labels) == 1

    def test_get_cip_labels_no_stereo(self):
        """get_cip_labels returns empty lists for achiral molecule."""
        atom_labels, bond_labels = get_cip_labels(HEXANE)
        assert atom_labels == []
        assert bond_labels == []

    def test_tanimoto_morgan_identical(self):
        """Identical molecules have Tanimoto 1.0."""
        assert tanimoto_morgan(BENZENE, BENZENE) == 1.0

    def test_tanimoto_morgan_different(self):
        """Very different molecules have low Tanimoto."""
        sim = tanimoto_morgan(BENZENE, HEXANE)
        assert sim < 0.5

    def test_tanimoto_morgan_invalid(self):
        """Invalid SMILES returns 0.0."""
        assert tanimoto_morgan("INVALID", BENZENE) == 0.0
