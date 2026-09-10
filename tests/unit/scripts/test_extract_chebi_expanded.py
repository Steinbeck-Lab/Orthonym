"""Unit tests for scripts/extract_chebi_expanded.py.

a phase-01: stratified ChEBI 5000 corpus extractor.

Fixture (tests/fixtures/phase_145/chebi_sample_5.tsv) contains 5 hand-picked
and classifier-verified CHEBI rows covering distinct (primary_class, size)
cells:

- CHEBI:16236 / CCO / ethanol -> (acyclic, small)
- CHEBI:7 /...bicycloheptene / car-3-ene analog -> (fused-ring, small)
- CHEBI:32 /...piperidine -> (heterocycle, small)
- CHEBI:185681 / O=C(O)NCO / hydroxymethylcarbamic acid -> (polyfunctional, small)
- CHEBI:64451 /...diazoniophenyl arsonate -> (aromatic, small)

Classifications verified by running classify_compound() from
scripts/benchmark_chebi500.py at phase 145-01 execution time (policy).
"""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "scripts" / "extract_chebi_expanded.py"
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "phase_145" / "chebi_sample_5.tsv"


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path / "out.csv"


class TestStratifiedSample:
    def test_fixture_produces_all_5_rows(self, tmp_output):
        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--input", str(FIXTURE),
             "--output", str(tmp_output),
             "--target-total", "5",
             "--seed", "123"],
            check=True, capture_output=True, text=True,
        )
        assert tmp_output.exists(), f"Output not created; stdout={result.stdout}"
        with open(tmp_output) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"
        expected_cols = {
            "chebi_id", "smiles", "reference_name", "name_index",
            "name_type", "compound_classes", "primary_class",
            "size_bucket", "heavy_atoms",
        }
        assert set(rows[0].keys()) >= expected_cols, (
            f"Missing columns: {expected_cols - set(rows[0].keys())}"
        )

    def test_reproducible_with_same_seed(self, tmp_path):
        out1 = tmp_path / "run1.csv"
        out2 = tmp_path / "run2.csv"
        for out in (out1, out2):
            subprocess.run(
                [sys.executable, str(SCRIPT),
                 "--input", str(FIXTURE),
                 "--output", str(out),
                 "--target-total", "5",
                 "--seed", "123"],
                check=True,
            )
        # Byte-identical when seed matches
        assert out1.read_bytes() == out2.read_bytes()

    def test_different_seed_smoke(self, tmp_path):
        # Smoke test that seed parameter is wired up. When the fixture exactly
        # fills the target (5 rows, 5 target), the final deterministic sort
        # makes both seed outputs identical. Just verify no crash.
        out = tmp_path / "s1.csv"
        subprocess.run(
            [sys.executable, str(SCRIPT),
             "--input", str(FIXTURE),
             "--output", str(out),
             "--target-total", "5",
             "--seed", "999"],
            check=True,
        )
        assert out.exists()

    def test_handles_multi_name_compounds(self, tmp_path):
        """Per: a CHEBI_ID appearing on multiple rows emits multiple
        output rows with incrementing name_index (0, 1, 2,...)."""
        multi_fixture = tmp_path / "multi.tsv"
        multi_fixture.write_text(
            "CHEBI:1\tCCO\tethanol\n"
            "CHEBI:1\tCCO\tethyl alcohol\n"
            "CHEBI:2\tCC\tethane\n"
        )
        out = tmp_path / "out.csv"
        subprocess.run(
            [sys.executable, str(SCRIPT),
             "--input", str(multi_fixture),
             "--output", str(out),
             "--target-total", "3",
             "--seed", "123"],
            check=True,
        )
        with open(out) as f:
            rows = list(csv.DictReader(f))
        chebi1_rows = [r for r in rows if r["chebi_id"] == "CHEBI:1"]
        assert len(chebi1_rows) == 2, (
            f"Expected 2 rows for CHEBI:1 (multi-name per D-06), got {len(chebi1_rows)}"
        )
        name_indices = sorted(r["name_index"] for r in chebi1_rows)
        assert name_indices == ["0", "1"], f"name_index not 0,1: {name_indices}"

    def test_primary_class_matches_priority_order(self, tmp_path):
        """CCO is acyclic; classify_compound returns ['acyclic', 'small'].
        The CLASS_PRIORITY chain (aromatic > heterocycle > fused-ring >
        polyfunctional > charged > acyclic) picks 'acyclic' as primary_class."""
        fixture = tmp_path / "fx.tsv"
        fixture.write_text("CHEBI:1\tCCO\tethanol\n")
        out = tmp_path / "out.csv"
        subprocess.run(
            [sys.executable, str(SCRIPT),
             "--input", str(fixture),
             "--output", str(out),
             "--target-total", "1",
             "--seed", "123"],
            check=True,
        )
        with open(out) as f:
            row = next(csv.DictReader(f))
        assert row["primary_class"] == "acyclic", (
            f"ethanol primary_class expected 'acyclic', got '{row['primary_class']}'"
        )
        assert row["size_bucket"] == "small", (
            f"ethanol size_bucket expected 'small', got '{row['size_bucket']}'"
        )

    def test_primary_class_aromatic_beats_acyclic(self, tmp_path):
        """An aromatic heterocycle (pyridine) gets primary_class='aromatic',
        not 'heterocycle'. This locks the priority chain."""
        fixture = tmp_path / "arom.tsv"
        fixture.write_text("CHEBI:1\tc1ccncc1\tpyridine\n")
        out = tmp_path / "out.csv"
        subprocess.run(
            [sys.executable, str(SCRIPT),
             "--input", str(fixture),
             "--output", str(out),
             "--target-total", "1",
             "--seed", "123"],
            check=True,
        )
        with open(out) as f:
            row = next(csv.DictReader(f))
        assert row["primary_class"] == "aromatic", (
            f"pyridine primary_class expected 'aromatic', got '{row['primary_class']}'"
        )

    def test_invalid_smiles_skipped(self, tmp_path):
        """RDKit-unparseable SMILES are dropped; valid rows still emitted."""
        fixture = tmp_path / "invalid.tsv"
        fixture.write_text(
            "CHEBI:1\tXXXNotASmiles\tbad\n"
            "CHEBI:2\tCC\tethane\n"
        )
        out = tmp_path / "out.csv"
        subprocess.run(
            [sys.executable, str(SCRIPT),
             "--input", str(fixture),
             "--output", str(out),
             "--target-total", "5",
             "--seed", "123"],
            check=True,
        )
        with open(out) as f:
            rows = list(csv.DictReader(f))
        # Only CHEBI:2 should survive
        assert len(rows) == 1
        assert rows[0]["chebi_id"] == "CHEBI:2"
