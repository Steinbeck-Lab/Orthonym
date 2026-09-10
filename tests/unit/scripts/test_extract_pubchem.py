"""Unit tests for scripts/extract_pubchem.py.

Exercises the PubChem extractor end-to-end against tiny gzipped fixtures so
that CI never needs network access or the real 1.7 GB PubChem dumps.
"""
import csv
import gzip
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "scripts" / "extract_pubchem.py"
FIXTURE_IUPAC = PROJECT_ROOT / "tests" / "fixtures" / "phase_145" / "pubchem_cid_iupac_sample.tsv"
FIXTURE_SMILES = PROJECT_ROOT / "tests" / "fixtures" / "phase_145" / "pubchem_cid_smiles_sample.tsv"


def _gz(src: Path, dst: Path):
    """Gzip-copy a plain TSV fixture into a.gz file the extractor will read."""
    with open(src, "rb") as r, gzip.open(dst, "wb") as w:
        w.write(r.read())


class TestPubchemExtract:
    def test_fixture_produces_expected_rows(self, tmp_path):
        iupac_gz = tmp_path / "CID-IUPAC.gz"
        smiles_gz = tmp_path / "CID-SMILES.gz"
        _gz(FIXTURE_IUPAC, iupac_gz)
        _gz(FIXTURE_SMILES, smiles_gz)

        out = tmp_path / "out.csv"
        subprocess.run(
            [sys.executable, str(SCRIPT),
             "--cache-dir", str(tmp_path),
             "--output", str(out),
             "--target-total", "10",
             "--seed", "123",
             "--skip-download"],
            check=True, capture_output=True, text=True,
        )
        assert out.exists()
        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 5, "Expected at least 5 valid rows from fixture"
        expected_cols = {"cid", "smiles", "reference_name", "compound_classes",
                         "primary_class", "size_bucket", "heavy_atoms"}
        assert set(rows[0].keys()) >= expected_cols

    def test_reproducible_with_same_seed(self, tmp_path):
        iupac_gz = tmp_path / "CID-IUPAC.gz"
        smiles_gz = tmp_path / "CID-SMILES.gz"
        _gz(FIXTURE_IUPAC, iupac_gz)
        _gz(FIXTURE_SMILES, smiles_gz)

        out1 = tmp_path / "run1.csv"
        out2 = tmp_path / "run2.csv"
        for out in (out1, out2):
            subprocess.run(
                [sys.executable, str(SCRIPT),
                 "--cache-dir", str(tmp_path),
                 "--output", str(out),
                 "--target-total", "5",
                 "--seed", "123",
                 "--skip-download"],
                check=True,
            )
        assert out1.read_bytes() == out2.read_bytes()

    def test_skips_invalid_smiles(self, tmp_path):
        # Create a fixture with an invalid SMILES
        bad_smiles = tmp_path / "bad_smiles.tsv"
        bad_smiles.write_text("1\tCCO\n2\t!!!INVALID!!!\n3\tCC\n")
        bad_iupac = tmp_path / "bad_iupac.tsv"
        bad_iupac.write_text("1\tethanol\n2\tinvalid compound\n3\tethane\n")
        _gz(bad_smiles, tmp_path / "CID-SMILES.gz")
        _gz(bad_iupac, tmp_path / "CID-IUPAC.gz")

        out = tmp_path / "out.csv"
        subprocess.run(
            [sys.executable, str(SCRIPT),
             "--cache-dir", str(tmp_path),
             "--output", str(out),
             "--target-total", "10",
             "--seed", "123",
             "--skip-download"],
            check=True,
        )
        with open(out) as f:
            rows = list(csv.DictReader(f))
        # CID 2 should be skipped
        cids = [r["cid"] for r in rows]
        assert "2" not in cids
        assert "1" in cids
        assert "3" in cids
