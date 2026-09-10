"""Unit tests for ``scripts/phase146_ship_gate.py`` gate functions.

Covers each Tier-1 ship-gate function (G2/G3/G4/G5), the paired Wilson
95%-lower-bound math, and the diagnostic disclosure enumerator.

Authority: CONTEXT.md (thresholds) + (disclosure).
Source: https://iupac.qmul.ac.uk/BlueBook/P4.html

Dependencies: ``numpy`` + ``pytest`` only (pandas is NOT a project
dependency per ``pyproject.toml``; the ship-gate runner consumes
``List[Dict[str, Any]]`` built via stdlib ``csv.DictReader``).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

# Make ``scripts/`` importable regardless of the invoking pytest cwd.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sg = importlib.import_module("phase146_ship_gate")


def _rows(rows):
    """Pass-through; present for parity with pandas-style test builders."""
    return list(rows)


# ---------------------------------------------------------------------------
# G2: OPSIN self-test (HARD)
# ---------------------------------------------------------------------------


class TestGateG2:
    def test_passes_when_no_regressions_and_positive_delta(self):
        df = _rows(
            [
                {
                    "source_corpus": "opsin_selftest_500",
                    "inchi_rt_v17": 1,
                    "inchi_rt_v18": 1,
                    "smiles_v17": "CCO",
                    "generated_name_v17": "ethanol",
                    "generated_name_v18": "ethanol",
                    "corpus_row_id": "r1",
                },
                {
                    "source_corpus": "opsin_selftest_500",
                    "inchi_rt_v17": 0,
                    "inchi_rt_v18": 1,
                    "smiles_v17": "CC",
                    "generated_name_v17": "ethane",
                    "generated_name_v18": "ethane",
                    "corpus_row_id": "r2",
                },
            ]
        )
        r = sg.gate_g2(df)
        assert r["passes"] is True
        assert r["rt_delta"] == 1
        assert r["regressions_count"] == 0
        assert r["hard"] is True

    def test_fails_when_one_regression(self):
        df = _rows(
            [
                {
                    "source_corpus": "opsin_selftest_500",
                    "inchi_rt_v17": 1,
                    "inchi_rt_v18": 0,
                    "smiles_v17": "CCO",
                    "generated_name_v17": "ethanol",
                    "generated_name_v18": "wrong",
                    "corpus_row_id": "r1",
                },
            ]
        )
        r = sg.gate_g2(df)
        assert r["passes"] is False
        assert r["regressions_count"] == 1

    def test_fails_when_negative_delta(self):
        df = _rows(
            [
                {
                    "source_corpus": "opsin_selftest_500",
                    "inchi_rt_v17": 1,
                    "inchi_rt_v18": 1,
                    "smiles_v17": "X",
                    "generated_name_v17": "x",
                    "generated_name_v18": "x",
                    "corpus_row_id": "r1",
                },
                {
                    "source_corpus": "opsin_selftest_500",
                    "inchi_rt_v17": 1,
                    "inchi_rt_v18": 0,
                    "smiles_v17": "Y",
                    "generated_name_v17": "y",
                    "generated_name_v18": "z",
                    "corpus_row_id": "r2",
                },
            ]
        )
        r = sg.gate_g2(df)
        assert r["passes"] is False

    def test_handles_empty_corpus(self):
        df = _rows(
            [
                {
                    "source_corpus": "chebi_5000",
                    "inchi_rt_v17": 0,
                    "inchi_rt_v18": 1,
                    "smiles_v17": "X",
                    "generated_name_v17": "x",
                    "generated_name_v18": "x",
                    "corpus_row_id": "r1",
                },
            ]
        )
        r = sg.gate_g2(df)
        assert r["passes"] is False
        assert "error" in r


# ---------------------------------------------------------------------------
# G3: ChEBI 5000 co-gate
# ---------------------------------------------------------------------------


class TestGateG3:
    def _make_chebi(self, n=100, gains=40, losses=0, graded_v17=2.0, graded_v18=2.05):
        rows = []
        for i in range(n):
            if i < gains:
                v17, v18 = 0, 1
            elif i < gains + losses:
                v17, v18 = 1, 0
            else:
                v17, v18 = 0, 0
            rows.append(
                {
                    "source_corpus": "chebi_5000",
                    "inchi_rt_v17": v17,
                    "inchi_rt_v18": v18,
                    "graded_total_v17": graded_v17,
                    "graded_total_v18": graded_v18,
                }
            )
        return rows

    def test_rt_delta_and_graded_delta_correct(self):
        # rt_delta = +40, graded_delta = +0.05
        df = self._make_chebi(n=100, gains=40, losses=0, graded_v17=2.0, graded_v18=2.05)
        r = sg.gate_g3(df)
        assert r["rt_delta"] == 40
        assert r["graded_delta"] == pytest.approx(0.05, abs=1e-6)
        # Wilson lower on 40 gains / 100 n should be positive (p_hat = 0.4, se small)
        assert r["wilson_95_lower"] > 0

    def test_fails_below_rt_threshold(self):
        df = self._make_chebi(n=100, gains=39, losses=0)
        r = sg.gate_g3(df)
        assert r["passes"] is False

    def test_fails_below_graded_threshold(self):
        df = self._make_chebi(n=100, gains=50, losses=0, graded_v17=2.0, graded_v18=2.005)
        r = sg.gate_g3(df)
        assert r["passes"] is False

    def test_handles_empty_chebi(self):
        df = _rows(
            [
                {
                    "source_corpus": "opsin_selftest_500",
                    "inchi_rt_v17": 1,
                    "inchi_rt_v18": 1,
                    "graded_total_v17": 5.0,
                    "graded_total_v18": 5.0,
                }
            ]
        )
        r = sg.gate_g3(df)
        assert r["passes"] is False
        assert "error" in r


# ---------------------------------------------------------------------------
# G4: per-handler floor
# ---------------------------------------------------------------------------


class TestGateG4:
    def _make_handlers(self, handler_data):
        """handler_data: list of (handler, n, rt_v17, rt_v18) tuples."""
        rows = []
        for handler, n, rt_v17, rt_v18 in handler_data:
            for i in range(n):
                rows.append(
                    {
                        "handler_v17": handler,
                        "inchi_rt_v17": 1 if i < rt_v17 else 0,
                        "inchi_rt_v18": 1 if i < rt_v18 else 0,
                    }
                )
        return rows

    def test_passes_when_no_handler_violates(self):
        df = self._make_handlers([("chain", 100, 80, 78)])  # delta=-2, not below -5
        r = sg.gate_g4(df)
        assert r["passes"] is True

    def test_fails_when_handler_violates(self):
        # delta=-20 (below -5 absolute), pct=-25% (below -20% relative)
        df = self._make_handlers([("chain", 100, 80, 60)])
        r = sg.gate_g4(df)
        assert r["passes"] is False
        assert r["violations_count"] == 1

    def test_skips_handlers_below_n50(self):
        df = self._make_handlers([("chain", 30, 30, 0)])  # n<50 -> skipped
        r = sg.gate_g4(df)
        assert r["passes"] is True

    def test_combined_floor_absorbs_small_absolute_drop(self):
        # delta=-4 (above -5 absolute) even though pct=-100% -> passes (AND semantics)
        df = self._make_handlers([("rare_handler", 60, 4, 0)])
        r = sg.gate_g4(df)
        assert r["passes"] is True


# ---------------------------------------------------------------------------
# G5: parent_score delta
# ---------------------------------------------------------------------------


class TestGateG5:
    def test_passes_at_threshold(self):
        df = _rows(
            [
                {
                    "inchi_rt_v17": 0,
                    "opsin_score_v17": 1.0,
                    "parent_score_v17": 0.0,
                    "parent_score_v18": 0.10,
                },
                {
                    "inchi_rt_v17": 0,
                    "opsin_score_v17": 1.0,
                    "parent_score_v17": 0.0,
                    "parent_score_v18": 0.10,
                },
            ]
        )
        r = sg.gate_g5(df)
        assert r["passes"] is True
        assert r["parent_score_delta"] == pytest.approx(0.10)

    def test_fails_below_threshold(self):
        df = _rows(
            [
                {
                    "inchi_rt_v17": 0,
                    "opsin_score_v17": 1.0,
                    "parent_score_v17": 0.0,
                    "parent_score_v18": 0.02,
                },
            ]
        )
        r = sg.gate_g5(df)
        assert r["passes"] is False

    def test_handles_empty_subset(self):
        # All rows are RT=1 -> subset empty
        df = _rows(
            [
                {
                    "inchi_rt_v17": 1,
                    "opsin_score_v17": 1.0,
                    "parent_score_v17": 1.0,
                    "parent_score_v18": 1.0,
                },
            ]
        )
        r = sg.gate_g5(df)
        assert r["passes"] is False
        assert "error" in r


# ---------------------------------------------------------------------------
# Paired Wilson lower bound
# ---------------------------------------------------------------------------


class TestPairedWilsonLowerBound:
    def test_zero_changes_returns_zero_or_below(self):
        v17 = np.array([1, 1, 0, 0])
        v18 = np.array([1, 1, 0, 0])
        assert sg.paired_wilson_lower_bound(v17, v18) <= 0.0

    def test_all_gains_positive_lower_bound(self):
        v17 = np.zeros(100, dtype=int)
        v18 = np.ones(100, dtype=int)
        lb = sg.paired_wilson_lower_bound(v17, v18)
        assert lb > 0

    def test_all_losses_negative_lower_bound(self):
        v17 = np.ones(100, dtype=int)
        v18 = np.zeros(100, dtype=int)
        lb = sg.paired_wilson_lower_bound(v17, v18)
        assert lb < 0

    def test_empty_returns_zero(self):
        v17 = np.array([], dtype=int)
        v18 = np.array([], dtype=int)
        assert sg.paired_wilson_lower_bound(v17, v18) == 0.0

    def test_accepts_plain_lists(self):
        # stdlib call path (no numpy arrays at call site) must work since
        # gate_g3 passes plain lists (not arrays).
        lb = sg.paired_wilson_lower_bound([0] * 100, [1] * 100)
        assert lb > 0


# ---------------------------------------------------------------------------
# diagnostic disclosure
# ---------------------------------------------------------------------------


class TestDiagnosticDisclosure:
    def test_returns_only_regressions(self):
        df = _rows(
            [
                {
                    "source_corpus": "x",
                    "corpus_row_id": "1",
                    "smiles_v17": "X",
                    "generated_name_v17": "a",
                    "generated_name_v18": "b",
                    "parent_score_v17": 1.0,
                    "parent_score_v18": 0.0,
                    "handler_v17": "chain",
                    "inchi_rt_v17": 1,
                    "inchi_rt_v18": 0,
                },
                {
                    "source_corpus": "x",
                    "corpus_row_id": "2",
                    "smiles_v17": "Y",
                    "generated_name_v17": "c",
                    "generated_name_v18": "c",
                    "parent_score_v17": 0.0,
                    "parent_score_v18": 0.0,
                    "handler_v17": "chain",
                    "inchi_rt_v17": 0,
                    "inchi_rt_v18": 0,
                },
            ]
        )
        r = sg.diagnostic_disclosure(df)
        assert len(r) == 1
        assert r[0]["corpus_row_id"] == "1"

    def test_handles_missing_columns_gracefully(self):
        # Missing ``parent_score_v17``, ``handler_v17``, etc. — should not crash
        df = _rows(
            [
                {
                    "source_corpus": "x",
                    "corpus_row_id": "1",
                    "smiles_v17": "X",
                    "inchi_rt_v17": 1,
                    "inchi_rt_v18": 0,
                },
            ]
        )
        r = sg.diagnostic_disclosure(df)
        assert len(r) == 1
        # Keys present should be subset of the ``wanted`` list
        assert "corpus_row_id" in r[0]


# ---------------------------------------------------------------------------
# Phase history persistence ()
# ---------------------------------------------------------------------------


class TestUpdatePhaseHistory:
    def test_creates_new_history_file(self, tmp_path):
        df = _rows(
            [
                {
                    "source_corpus": "chebi_5000",
                    "inchi_rt_v17": 1,
                    "inchi_rt_v18": 1,
                    "corpus_row_id": "r1",
                },
                {
                    "source_corpus": "chebi_5000",
                    "inchi_rt_v17": 0,
                    "inchi_rt_v18": 1,
                    "corpus_row_id": "r2",
                },
            ]
        )
        results = {"G2": {"passes": True}, "G3": {"passes": False}}
        history_path = tmp_path / "phase_history.json"
        history = sg.update_phase_history(history_path, results, df)
        assert history_path.exists()
        assert history["baseline_v17"] == 1
        assert len(history["phases"]) == 1
        assert history["phases"][0]["cumulative_delta_since_v17"] == 1

    def test_appends_to_existing_history(self, tmp_path):
        history_path = tmp_path / "phase_history.json"
        # Pre-populate with earlier run.
        import json as _json

        history_path.write_text(
            _json.dumps(
                {
                    "baseline_v17": 10,
                    "phases": [{"phase": "145", "cumulative_delta_since_v17": 0}],
                }
            )
        )
        df = _rows(
            [
                {
                    "source_corpus": "chebi_5000",
                    "inchi_rt_v17": 10,
                    "inchi_rt_v18": 15,
                    "corpus_row_id": "r1",
                }
            ]
        )
        results = {"G2": {"passes": True}, "G3": {"passes": True}}
        history = sg.update_phase_history(history_path, results, df)
        # Baseline preserved from first run (NOT overwritten).
        assert history["baseline_v17"] == 10
        assert len(history["phases"]) == 2
        # Cumulative delta uses preserved baseline.
        assert history["phases"][-1]["cumulative_delta_since_v17"] == 15 - 10


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------


class TestWriteMarkdownReport:
    def test_writes_gate_summary_and_disclosure(self, tmp_path):
        out_path = tmp_path / "146-SHIP-GATE.md"
        results = {
            "G2": {
                "gate": "G2",
                "hard": True,
                "rt_delta": 1,
                "regressions_count": 0,
                "passes": True,
                "n": 100,
            },
            "G3": {
                "gate": "G3",
                "rt_delta": 50,
                "graded_delta": 0.04,
                "wilson_95_lower": 0.01,
                "passes": True,
                "n": 5000,
            },
            "G4": {
                "gate": "G4",
                "violations_count": 0,
                "passes": True,
                "per_handler_summary": [],
                "violations": [],
            },
            "G5": {
                "gate": "G5",
                "parent_score_delta": 0.05,
                "passes": True,
                "n_subset": 1000,
            },
            "diagnostic_regressions": [],
        }
        history = {
            "baseline_v17": 100,
            "phases": [
                {
                    "phase": "146",
                    "rt_count_v18": 105,
                    "cumulative_delta_since_v17": 5,
                }
            ],
        }
        sg.write_markdown_report(results, out_path, history)
        text = out_path.read_text()
        assert "Phase 146 Ship Gate Report" in text
        assert "Triple coordination per D-07" in text
        assert "PASS" in text
        assert "iupac.qmul.ac.uk" in text
        assert "D-12" in text


# ---------------------------------------------------------------------------
# load_both CSV merge helper
# ---------------------------------------------------------------------------


class TestLoadBoth:
    def test_merges_csvs_by_key(self, tmp_path):
        import csv as _csv

        v17 = tmp_path / "v17.csv"
        v18 = tmp_path / "v18.csv"
        cols = [
            "source_corpus",
            "corpus_row_id",
            "smiles",
            "inchi_rt",
            "parent_score",
            "handler",
        ]
        with open(v17, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerow(
                {
                    "source_corpus": "chebi_5000",
                    "corpus_row_id": "c1",
                    "smiles": "CCO",
                    "inchi_rt": "1",
                    "parent_score": "1.0",
                    "handler": "chain",
                }
            )
        with open(v18, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerow(
                {
                    "source_corpus": "chebi_5000",
                    "corpus_row_id": "c1",
                    "smiles": "CCO",
                    "inchi_rt": "1",
                    "parent_score": "1.0",
                    "handler": "chain",
                }
            )
        merged = sg.load_both(v17, v18)
        assert len(merged) == 1
        row = merged[0]
        assert row["source_corpus"] == "chebi_5000"
        assert row["corpus_row_id"] == "c1"
        assert row["inchi_rt_v17"] == 1
        assert row["inchi_rt_v18"] == 1
        assert row["handler_v17"] == "chain"
