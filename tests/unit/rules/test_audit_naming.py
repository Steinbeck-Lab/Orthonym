"""Tests for  audit script."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@pytest.mark.unit
class TestAuditNaming:
    """Test the audit_naming.py script."""

    def _create_fixture(self, tmp_path):
        """Create a minimal benchmark results JSON fixture."""
        results = {
            "summary": {"total": 3},
            "detailed_results": [
                {
                    "smiles": "CCO",
                    "name": "ethanol",
                    "named_ok": True,
                    "opsin_parsed": True,
                    "inchi_match": True,
                },
                {
                    "smiles": "c1ccc2ccccc2c1",
                    "name": "naphthalene",
                    "named_ok": True,
                    "opsin_parsed": True,
                    "inchi_match": True,
                },
                {
                    "smiles": "CC(C)(C)CC(=O)O",
                    "name": "acid",
                    "named_ok": True,
                    "opsin_parsed": False,
                    "inchi_match": False,
                },
            ],
        }
        fixture_path = tmp_path / "test_results.json"
        with open(fixture_path, "w") as f:
            json.dump(results, f)
        return fixture_path

    def _create_debug_log(self, tmp_path):
        """Create a minimal debug log fixture with DROP entries."""
        log_content = (
            "WARNING orthonym.assembly.composer DROP-01 substituent_skip: "
            "reason=zero_carbon position=2 smiles=CC(C)(C)CC(=O)O\n"
            "WARNING orthonym.assembly.composer DROP-04 substituent_skip: "
            "reason=ring_heteroatom_branch position=3 smiles=c1ccc(N)cc1CCO\n"
            "DEBUG orthonym.rules.parent_selection P44_AUDIT: "
            "pg_tied=1 chain_len=5 ring_size=6 chain_would_win_length=False\n"
        )
        log_path = tmp_path / "debug.log"
        with open(log_path, "w") as f:
            f.write(log_content)
        return log_path

    def test_audit_runs_on_fixture(self, tmp_path):
        """audit_naming.py processes benchmark JSON without error."""
        fixture = self._create_fixture(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "audit_naming.py"),
                str(fixture),
                "--top",
                "5",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "Worst Atom Coverage" in result.stdout

    def test_audit_with_debug_log(self, tmp_path):
        """audit_naming.py processes debug log for DROP analysis."""
        fixture = self._create_fixture(tmp_path)
        log_path = self._create_debug_log(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "audit_naming.py"),
                str(fixture),
                "--debug-log",
                str(log_path),
                "--top",
                "3",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "DROP Point Summary" in result.stdout
        assert "DROP-01" in result.stdout
        assert "P-44.1 Audit" in result.stdout

    def test_audit_json_output(self, tmp_path):
        """audit_naming.py writes JSON output when --json-output specified."""
        fixture = self._create_fixture(tmp_path)
        json_out = tmp_path / "audit_out.json"
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "audit_naming.py"),
                str(fixture),
                "--json-output",
                str(json_out),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert json_out.exists()
        with open(json_out) as f:
            audit = json.load(f)
        assert "top_worst_coverage" in audit
        assert "coverage_stats" in audit

    def test_coverage_computation(self):
        """compute_atom_coverage returns correct metrics."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "audit_naming",
            str(PROJECT_ROOT / "scripts" / "audit_naming.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        result = mod.compute_atom_coverage("CCO", "ethanol")
        assert result["total_heavy_atoms"] == 3
        assert result["name_length"] == 7
        assert result["length_ratio"] == round(7 / 3, 3)
        assert "error" not in result

    def test_coverage_invalid_smiles(self):
        """compute_atom_coverage handles invalid SMILES gracefully."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "audit_naming",
            str(PROJECT_ROOT / "scripts" / "audit_naming.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        result = mod.compute_atom_coverage("INVALID", "test")
        assert "error" in result
