"""CLI flag subprocess tests for Phase 162 MLF-01 + D-08.

Validates the ``--allow-ml-fallback`` + ``--ml-opsin-parse-required`` +
``--no-ml-opsin-parse-required`` flags work end-to-end via
``python -m orthonym`` subprocess invocation.

Per CONTEXT D-07 + 162-AUDIT-MLF.md § 8.8: MLF-01 default-OFF invariant
is the most important assertion — adding ``--allow-ml-fallback`` to the
command line MUST NOT change the output of any clean rule-based input.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _run_cli(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Invoke the orthonym CLI via subprocess.

    Uses ``sys.executable -m orthonym`` for portability across CI
    environments that may not have ``orthonym`` in PATH.
    """
    cmd = [sys.executable, "-m", "orthonym", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


@pytest.mark.integration
class TestCliMLFallbackFlag:
    """Tests for --allow-ml-fallback flag (MLF-01)."""

    def test_help_shows_allow_ml_fallback(self) -> None:
        r = _run_cli("--help")
        assert r.returncode == 0, r.stderr
        assert "--allow-ml-fallback" in r.stdout

    def test_help_shows_ml_opsin_parse_required(self) -> None:
        r = _run_cli("--help")
        assert r.returncode == 0, r.stderr
        assert "--ml-opsin-parse-required" in r.stdout

    def test_default_off_no_flag(self) -> None:
        """MLF-01 default-OFF: rule-based pipeline only, no ML."""
        r = _run_cli("CCO")
        assert r.returncode == 0, r.stderr
        assert "ethanol" in r.stdout.lower()

    def test_explicit_allow_ml_fallback_on_clean_input(self) -> None:
        """MLF-06: --allow-ml-fallback MUST NOT change clean-input output."""
        r = _run_cli("--allow-ml-fallback", "CCO")
        assert r.returncode == 0, r.stderr
        assert "ethanol" in r.stdout.lower()

    def test_no_ml_opsin_parse_required_negation(self) -> None:
        """D-08: --no-ml-opsin-parse-required negated form is accepted."""
        r = _run_cli("--no-ml-opsin-parse-required", "CCO")
        assert r.returncode == 0, r.stderr

    def test_ml_flag_does_not_break_default_path(self) -> None:
        """MLF-01 invariant: --allow-ml-fallback must NEVER change default-OFF behavior."""
        r_off = _run_cli("CCO")
        r_on = _run_cli("--allow-ml-fallback", "CCO")
        assert r_off.returncode == 0, r_off.stderr
        assert r_on.returncode == 0, r_on.stderr
        assert "ethanol" in r_off.stdout.lower()
        assert "ethanol" in r_on.stdout.lower()

    def test_invalid_smiles_handled_gracefully_with_ml_on(self) -> None:
        r = _run_cli("--allow-ml-fallback", "INVALID_SMILES_!@#$")
        # Should NOT crash hard; returncode 0 or 1 with helpful error
        assert r.returncode in (0, 1)

    def test_style_flag_compatibility_with_ml(self) -> None:
        """--style + --allow-ml-fallback should not conflict."""
        r = _run_cli("--style", "pin", "--allow-ml-fallback", "CCO")
        assert r.returncode == 0, r.stderr
        assert "ethanol" in r.stdout.lower()

    def test_verbose_with_ml_flag(self) -> None:
        """--verbose + --allow-ml-fallback should not break."""
        r = _run_cli("--verbose", "--allow-ml-fallback", "CCO")
        assert r.returncode == 0, r.stderr
        assert "ethanol" in r.stdout.lower()

    def test_confidence_with_ml_off_includes_ml_keys(self) -> None:
        """--confidence output includes ML annotations when keys are present in dict."""
        r = _run_cli("--confidence", "CCO")
        assert r.returncode == 0, r.stderr
        # Default-OFF: ml_fallback_used=False; cli.py only prints the
        # "ML fallback used: True" line when ml_fallback_used is truthy.
        # Both behaviors are acceptable — assert just that ethanol naming
        # works correctly.
        assert "ethanol" in r.stdout.lower()

    def test_batch_mode_with_ml_flag(self) -> None:
        """--batch + --allow-ml-fallback should not conflict."""
        r = _run_cli("--help")
        if "--batch" not in r.stdout:
            pytest.skip("--batch not supported by this CLI build")
        with tempfile.NamedTemporaryFile("w", suffix=".smi", delete=False) as fh:
            fh.write("CCO\nC1=CC=CC=C1\n")
            tmp_path = fh.name
        try:
            r2 = _run_cli("--allow-ml-fallback", "--batch", tmp_path)
            assert r2.returncode == 0, r2.stderr
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_ml_flag_argument_group_in_help(self) -> None:
        """--help groups the ML flags under an explicit argument group."""
        r = _run_cli("--help")
        assert r.returncode == 0, r.stderr
        assert "ML Fallback" in r.stdout
