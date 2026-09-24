"""Integration test for scripts/benchmark_multi_corpus.py (a phase-05).

Runs the full unified runner against 3 tiny fixture corpora (5 compounds each
= 15 total) and verifies the produced artifacts have the right structure.

Requires OPSIN jar + orthonym import. Skipped if OPSIN jar is missing.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
from tests.support.jars import jar_or_none

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "benchmark_multi_corpus.py"
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "phase_145"
OPSIN_JAR = jar_or_none()

pytestmark = pytest.mark.skipif(
    OPSIN_JAR is None, reason="OPSIN jar required for integration test"
)


@pytest.mark.integration
def test_multi_corpus_produces_unified_csv(tmp_path):
    """Run the benchmark on 3 fixture corpora (5 compounds each, 15 total)
    and verify the unified CSV + summary JSON + reference_consistency.json
    have the expected structure."""
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--output-dir", str(tmp_path),
            "--chebi-csv", str(FIXTURE_DIR / "chebi_fixture.csv"),
            "--pubchem-csv", str(FIXTURE_DIR / "pubchem_fixture.csv"),
            "--opsin-selftest-csv", str(FIXTURE_DIR / "opsin_selftest_fixture.csv"),
            "--ref-consistency-out", str(tmp_path / "reference_consistency.json"),
        ],
        check=True, capture_output=True, text=True, timeout=600,
    )

    csv_out = tmp_path / "benchmark_multi_corpus_results.csv"
    json_out = tmp_path / "benchmark_multi_corpus_summary.json"
    refcon = tmp_path / "reference_consistency.json"

    assert csv_out.exists(), f"CSV missing. stderr: {result.stderr}"
    assert json_out.exists(), f"Summary JSON missing. stderr: {result.stderr}"
    assert refcon.exists(), "reference_consistency.json missing"

    with open(csv_out) as f:
        rows = list(csv.DictReader(f))
    # 5 per corpus × 3 corpora = 15 rows
    assert len(rows) == 15, f"Expected 15 rows, got {len(rows)}"

    sources = {r["source_corpus"] for r in rows}
    expected = {"chebi_5000", "pubchem_2000", "opsin_selftest_500"}
    assert sources == expected, f"Unexpected corpus names: {sources}"

    summary = json.loads(json_out.read_text())
    assert "per_corpus" in summary
    assert set(summary["per_corpus"].keys()) == expected
    assert summary["total_compounds"] == 15

    refcon_data = json.loads(refcon.read_text())
    for corpus_name in expected:
        assert corpus_name in refcon_data
        assert "fraction_consistent" in refcon_data[corpus_name]
        assert "n_compounds" in refcon_data[corpus_name]


@pytest.mark.integration
def test_subset_mode_uses_deterministic_seed(tmp_path):
    """Subset mode (seed=321) produces identical per-compound rows across runs
    on the same 5-row fixtures."""
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    for out in (out1, out2):
        subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--output-dir", str(out),
                "--subset", "--subset-n", "3",
                "--chebi-csv", str(FIXTURE_DIR / "chebi_fixture.csv"),
                "--pubchem-csv", str(FIXTURE_DIR / "pubchem_fixture.csv"),
                "--opsin-selftest-csv", str(FIXTURE_DIR / "opsin_selftest_fixture.csv"),
                "--ref-consistency-out", str(out / "reference_consistency.json"),
            ],
            check=True, capture_output=True, timeout=600,
        )

    with open(out1 / "benchmark_multi_corpus_results.csv") as a, \
         open(out2 / "benchmark_multi_corpus_results.csv") as b:
        rows_a = sorted([(r["source_corpus"], r["smiles"]) for r in csv.DictReader(a)])
        rows_b = sorted([(r["source_corpus"], r["smiles"]) for r in csv.DictReader(b)])
    assert rows_a == rows_b, "Subset must be deterministic across runs"


@pytest.mark.integration
def test_compare_to_baseline_pass_case(tmp_path):
    """compare_benchmark_to_baseline.py exits 0 when current matches baseline."""
    gate_script = PROJECT_ROOT / "scripts" / "compare_benchmark_to_baseline.py"
    summary = {
        "per_corpus": {
            "chebi_5000": {
                "inchi_rt_fraction": 0.30,
                "graded_total_mean": 3.25,
                "opsin_parse_rate": 0.75,
            }
        }
    }
    base_path = tmp_path / "base.json"
    cur_path = tmp_path / "cur.json"
    base_path.write_text(json.dumps(summary))
    cur_path.write_text(json.dumps(summary))
    result = subprocess.run(
        [sys.executable, str(gate_script), str(cur_path), str(base_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"Expected exit 0, got {result.returncode}\n{result.stdout}\n{result.stderr}"


@pytest.mark.integration
def test_compare_to_baseline_fail_case(tmp_path):
    """compare_benchmark_to_baseline.py exits 1 when current regresses > 5pp."""
    gate_script = PROJECT_ROOT / "scripts" / "compare_benchmark_to_baseline.py"
    baseline = {
        "per_corpus": {
            "chebi_5000": {
                "inchi_rt_fraction": 0.30,
                "graded_total_mean": 3.25,
                "opsin_parse_rate": 0.75,
            }
        }
    }
    current = {
        "per_corpus": {
            "chebi_5000": {
                "inchi_rt_fraction": 0.10,  # dropped 20pp -> should fail
                "graded_total_mean": 3.25,
                "opsin_parse_rate": 0.75,
            }
        }
    }
    base_path = tmp_path / "base.json"
    cur_path = tmp_path / "cur.json"
    base_path.write_text(json.dumps(baseline))
    cur_path.write_text(json.dumps(current))
    result = subprocess.run(
        [sys.executable, str(gate_script), str(cur_path), str(base_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1, f"Expected exit 1, got {result.returncode}"
    assert "CI GATE FAILED" in result.stdout
