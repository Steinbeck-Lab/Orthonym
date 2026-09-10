"""Unit tests for ``scripts/calibrate_phase_148_1.py`` (Tier 1).

Covers:
- grid topology + sum-to-1.0 filter (50-300 valid configs).
- 3-level tie-breaker logic (parent_correctness > balance > lex).
- anti-overfitting guard (carry-forward of a phase).
- wall-clock guard constants (3-hr soft warning / 5-hr hard abort).
- path constants (BASELINE_CSV → post-148, SPLITS/OUT → 148.1 paths).

Uses ``importlib.import_module`` to load the script under test (same pattern
as a phase analog tests).
"""
import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
cp = importlib.import_module("calibrate_phase_148_1")


class TestGridGeneration:
    """ grid topology + sum-to-1.0 filter."""

    def test_sum_to_1_filter_reduces_to_realistic_count(self):
        """5^5=3125 grid × sum-to-1.0 (±0.01) → between 50 and 300 valid configs."""
        configs = cp.generate_valid_configs({})
        assert 50 <= len(configs) <= 300, (
            f"Expected 50-300 valid configs after sum-to-1.0 filter; got {len(configs)}"
        )

    def test_parent_correctness_floor_is_d02(self):
        """Every valid config has parent_correctness ∈ {0.05, 0.10, 0.15, 0.20, 0.25} (D-02)."""
        configs = cp.generate_valid_configs({})
        valid_pc = {0.05, 0.10, 0.15, 0.20, 0.25}
        for cfg in configs:
            pc = cfg.get("parent_correctness", 0.0)
            assert pc in valid_pc, f"D-02 violation: parent_correctness={pc}"

    def test_grid_version_is_phase148_1_v1(self):
        """ GRID_VERSION lock."""
        assert cp.GRID_VERSION == "phase148_1_v1", (
            f"D-01 GRID_VERSION wrong: {cp.GRID_VERSION}"
        )

    def test_ratio_locked_at_zero(self):
        """a phase -a.1 carry-forward: ratio is permanently 0.0.

        Note: ``ratio`` is not in FACTOR_GRID for a phase; the worker
        sets it to 0.0 in cs.FACTOR_WEIGHTS_V18. The grid only generates
        configs with the 5 active factors (atom_coverage, fg_recognition,
        substituent_completeness, parent_correctness, multiple_bond_count).
        Verify ratio is absent from generated configs (since it's not in
        the GRID dict it cannot leak in).
        """
        configs = cp.generate_valid_configs({})
        for cfg in configs:
            # ratio either absent OR pinned to 0.0 (defensive read)
            assert cfg.get("ratio", 0.0) == 0.0, (
                f"Phase 145.2 D-09-a.1 violation: ratio={cfg.get('ratio')}"
            )


class TestTieBreaker:
    """ 3-level tie-breaker."""

    def _result(self, weights, test_rt):
        return {
            "config": {"ratio": 0.0, **weights},
            "train": {"rt_fraction": 0.0, "n": 5600, "rt_count": 0,
                      "graded_mean": 0.0, "per_handler": {}},
            "val": {"rt_fraction": 0.0, "n": 700, "rt_count": 0,
                    "graded_mean": 0.0, "per_handler": {}},
            "test": {"rt_fraction": test_rt, "n": 700,
                     "rt_count": int(test_rt * 700),
                     "graded_mean": 0.0, "per_handler": {}},
            "test_wilson_95_lower": 0.0,
        }

    def test_level_1_unique_winner(self):
        """One config has strictly higher test rt_fraction → level=1."""
        results = [
            self._result(
                {"atom_coverage": 0.20, "fg_recognition": 0.30,
                 "substituent_completeness": 0.20, "parent_correctness": 0.10,
                 "multiple_bond_count": 0.20},
                test_rt=0.31),
            self._result(
                {"atom_coverage": 0.10, "fg_recognition": 0.40,
                 "substituent_completeness": 0.20, "parent_correctness": 0.10,
                 "multiple_bond_count": 0.20},
                test_rt=0.30),
            self._result(
                {"atom_coverage": 0.15, "fg_recognition": 0.35,
                 "substituent_completeness": 0.20, "parent_correctness": 0.10,
                 "multiple_bond_count": 0.20},
                test_rt=0.29),
        ]
        winner, level = cp._select_winning_config(results, score_tolerance=0.001)
        assert level == "1"
        assert winner["test"]["rt_fraction"] == 0.31

    def test_level_2_higher_parent_correctness_wins(self):
        """Two within score_tolerance: higher parent_correctness wins → level=2."""
        results = [
            self._result(
                {"atom_coverage": 0.20, "fg_recognition": 0.30,
                 "substituent_completeness": 0.20, "parent_correctness": 0.10,
                 "multiple_bond_count": 0.20},
                test_rt=0.3000),
            self._result(
                {"atom_coverage": 0.20, "fg_recognition": 0.30,
                 "substituent_completeness": 0.20, "parent_correctness": 0.20,
                 "multiple_bond_count": 0.10},
                test_rt=0.3005),
        ]
        winner, level = cp._select_winning_config(results, score_tolerance=0.001)
        assert level == "2"
        assert winner["config"]["parent_correctness"] == 0.20

    def test_level_3_lower_variance_wins(self):
        """Tied at L1+L2: lowest variance (most balanced) wins → level=3."""
        results = [
            self._result(
                # High-variance: 0.05/0.45/0.20/0.20/0.10
                {"atom_coverage": 0.05, "fg_recognition": 0.45,
                 "substituent_completeness": 0.20, "parent_correctness": 0.20,
                 "multiple_bond_count": 0.10},
                test_rt=0.30),
            self._result(
                # Zero-variance: 0.20 across all 5 active weights
                {"atom_coverage": 0.20, "fg_recognition": 0.20,
                 "substituent_completeness": 0.20, "parent_correctness": 0.20,
                 "multiple_bond_count": 0.20},
                test_rt=0.30),
        ]
        winner, level = cp._select_winning_config(results, score_tolerance=0.001)
        assert level == "3"
        cfg = winner["config"]
        assert cfg["atom_coverage"] == 0.20 and cfg["fg_recognition"] == 0.20

    def test_level_none_lex_fallback(self):
        """All four levels match: deterministic lex tie-breaker → level='3' or 'none'."""
        # Two identical configs; the function should still return deterministically.
        results = [
            self._result(
                {"atom_coverage": 0.20, "fg_recognition": 0.20,
                 "substituent_completeness": 0.20, "parent_correctness": 0.20,
                 "multiple_bond_count": 0.20},
                test_rt=0.30),
            self._result(
                {"atom_coverage": 0.20, "fg_recognition": 0.20,
                 "substituent_completeness": 0.20, "parent_correctness": 0.20,
                 "multiple_bond_count": 0.20},
                test_rt=0.30),
        ]
        winner, level = cp._select_winning_config(results, score_tolerance=0.001)
        # Two duplicates: variance match → may resolve at L3 (single after
        # first-match) or fall through to "none" (lex). Either is acceptable.
        assert level in {"3", "none"}, f"unexpected level={level}"


class TestAntiOverfittingGuard:
    """ anti-overfitting guard (carry-forward a phase)."""

    def test_guard_passes_when_train_test_aligned(self):
        """Train-best and test-best agree on the same config → 0% direction change."""
        results = [
            {
                "config": {
                    "ratio": 0.0, "atom_coverage": 0.20, "fg_recognition": 0.30,
                    "substituent_completeness": 0.20, "parent_correctness": 0.10,
                    "multiple_bond_count": 0.20,
                },
                "train": {"rt_fraction": 0.40, "n": 5600, "rt_count": 2240,
                          "graded_mean": 0.5, "per_handler": {}},
                "val": {"rt_fraction": 0.35, "n": 700, "rt_count": 245,
                        "graded_mean": 0.5, "per_handler": {}},
                "test": {"rt_fraction": 0.34, "n": 700, "rt_count": 238,
                         "graded_mean": 0.5, "per_handler": {}},
                "test_wilson_95_lower": 0.30,
            },
        ]
        chosen = cp.select_winner_with_anti_overfit(results)
        check = chosen.get("anti_overfitting_check", {})
        # Either key or (legacy) key indicates pass.
        assert (
            check.get("passes_d04_5pct_threshold") is True
            or check.get("passes_d16_5pct_threshold") is True
        ), check

    def test_guard_fires_when_train_test_diverge(self):
        """Train-best ≠ test-best with >=5% delta → guard fires."""
        results = [
            {
                # Config A: high train_rt 0.50, low test_rt 0.20 (overfit pattern)
                "config": {
                    "ratio": 0.0, "atom_coverage": 0.20, "fg_recognition": 0.30,
                    "substituent_completeness": 0.20, "parent_correctness": 0.10,
                    "multiple_bond_count": 0.20,
                },
                "train": {"rt_fraction": 0.50, "n": 5600, "rt_count": 2800,
                          "graded_mean": 0.5, "per_handler": {}},
                "val": {"rt_fraction": 0.20, "n": 700, "rt_count": 140,
                        "graded_mean": 0.5, "per_handler": {}},
                "test": {"rt_fraction": 0.20, "n": 700, "rt_count": 140,
                         "graded_mean": 0.5, "per_handler": {}},
                "test_wilson_95_lower": 0.18,
            },
            {
                # Config B: lower train_rt 0.30, higher test_rt 0.30 (well-regularized)
                "config": {
                    "ratio": 0.0, "atom_coverage": 0.10, "fg_recognition": 0.40,
                    "substituent_completeness": 0.20, "parent_correctness": 0.10,
                    "multiple_bond_count": 0.20,
                },
                "train": {"rt_fraction": 0.30, "n": 5600, "rt_count": 1680,
                          "graded_mean": 0.5, "per_handler": {}},
                "val": {"rt_fraction": 0.30, "n": 700, "rt_count": 210,
                        "graded_mean": 0.5, "per_handler": {}},
                "test": {"rt_fraction": 0.30, "n": 700, "rt_count": 210,
                         "graded_mean": 0.5, "per_handler": {}},
                "test_wilson_95_lower": 0.28,
            },
        ]
        chosen = cp.select_winner_with_anti_overfit(results)
        check = chosen.get("anti_overfitting_check", {})
        # Guard reports the delta is >= 5%
        delta = check.get("train_best_test_direction_change_pct", 0.0)
        assert delta >= 0.05, f"Expected guard to detect divergence; delta={delta}"


class TestWallClockGuard:
    """ wall-clock budget."""

    def test_wallclock_constants_match_d07(self):
        """Source contains 3-hr warn + 5-hr abort constants in literal form."""
        src = (
            Path(__file__).parent.parent.parent.parent
            / "scripts" / "calibrate_phase_148_1.py"
        ).read_text()
        assert "WALLCLOCK_WARN_S = 3 * 3600" in src or "WALLCLOCK_WARN_S = 10800" in src, (
            "D-07: 3-hr soft warning constant missing"
        )
        assert "WALLCLOCK_ABORT_S = 5 * 3600" in src or "WALLCLOCK_ABORT_S = 18000" in src, (
            "D-07: 5-hr abort constant missing"
        )
        # Module attributes match expected values.
        assert cp.WALLCLOCK_WARN_S == 10800
        assert cp.WALLCLOCK_ABORT_S == 18000


class TestPathConstants:
    """D-01 three locked changes — verified via constants."""

    def test_baseline_csv_points_at_post148(self):
        """D-01.1: SOURCE_CSV → post-148 baseline corpus."""
        assert "post148/benchmark_multi_corpus_results.csv" in str(cp.BASELINE_CSV), (
            f"D-01.1 violation: {cp.BASELINE_CSV}"
        )

    def test_splits_json_points_at_148_1(self):
        """D-01.3: SPLITS_JSON → calibration_splits_148_1.json."""
        assert "calibration_splits_148_1.json" in str(cp.SPLITS_JSON), (
            f"D-01.3 violation: {cp.SPLITS_JSON}"
        )

    def test_out_json_points_at_148_1(self):
        """D-01.3: OUT_JSON → calibration_results_148_1.json."""
        assert "calibration_results_148_1.json" in str(cp.OUT_JSON), (
            f"D-01.3 violation: {cp.OUT_JSON}"
        )
