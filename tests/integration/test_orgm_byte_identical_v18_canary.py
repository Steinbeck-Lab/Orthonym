"""a phase integration: assert -substrate canary preserved byte-identical.

RESEARCH + Q3 predicts ZERO flips. Per internal notes + +:
- src/orthonym/rules/seniority.py is UNCHANGED (hard invariant)
- src/orthonym/assembly/composer.py is UNCHANGED (enforcement)
- src/orthonym/namer.py orchestration is UNCHANGED

The exceptions file at internal notes
internal notes-ORGM-exceptions.csv MUST stay header-only — ANY non-header rows
indicate a silent canary flip that was absorbed without an explicit
audit-amendment commit. Per a phase + a phase cadence inheritance.

NEVER uses @pytest.mark.xfail (internal notes) — honest-fail-on-data.
"""
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
EXCEPTIONS_FILE = (
    PROJECT_ROOT / ".planning" / "phases"
    / "161-p-69-organometallics-support" / "161-AUDIT-ORGM-exceptions.csv"
)


@pytest.mark.integration
def test_v18_decomp_canary_byte_identical():
    """ inheritance: substrate canary byte-identical post-ORGM landing.

    Runs ``scripts/verify_decomp_byte_identical.py --mode delta`` with the
    EMPTY (header-only) a phase exceptions file. Per RESEARCH
    ZERO-flip prediction: exit code 0 expected.
    """
    result = subprocess.run(
        [
            "python3",
            str(PROJECT_ROOT / "scripts/verify_decomp_byte_identical.py"),
            "--mode", "delta",
            "--exceptions-file", str(EXCEPTIONS_FILE),
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, (
        f"v18 DECOMP canary BYTE-IDENTICAL violation:\n"
        f"stdout (last 2000 chars): {result.stdout[-2000:]}\n"
        f"stderr (last 2000 chars): {result.stderr[-2000:]}"
    )


@pytest.mark.integration
def test_v18_cfr_canary_byte_identical():
    """ inheritance: CFR canary byte-identical post-ORGM landing.

    The full `verify_cfr_byte_identical.py --mode post` re-emit takes
    ~10 minutes (~ 1,282 OPSIN round-trip invocations). For the pytest
    suite (which must stay fast per Plan-04 acceptance), we diff the
    EXISTING pre/post CFR canary CSVs directly. The Plan-03 SUMMARY
    confirmed `--mode post` exits 0 with empty exceptions file (ZERO
    byte-diff) at every Plan-03 commit; Plan-04 inherits this state.
    """
    pre_path = PROJECT_ROOT / "tests" / "canary" / "canary_pre_cfr_158.csv"
    post_path = PROJECT_ROOT / "tests" / "canary" / "canary_post_cfr_158.csv"
    assert pre_path.exists(), f"Missing pre-CFR canary at {pre_path}"
    assert post_path.exists(), f"Missing post-CFR canary at {post_path}"
    pre_md5 = subprocess.check_output(
        ["md5sum", str(pre_path)], text=True
    ).split()[0]
    post_md5 = subprocess.check_output(
        ["md5sum", str(post_path)], text=True
    ).split()[0]
    assert pre_md5 == post_md5, (
        f"v18 CFR canary BYTE-IDENTICAL violation:\n"
        f"  pre  md5: {pre_md5}\n"
        f"  post md5: {post_md5}\n"
        f"Re-run `python3 scripts/verify_cfr_byte_identical.py --mode post "
        f"--exceptions-file {EXCEPTIONS_FILE}` to investigate."
    )


@pytest.mark.integration
def test_exceptions_file_header_only():
    """RESEARCH ZERO-flip prediction: exceptions file MUST stay header-only.

    Non-empty exceptions list = silent canary flip detection per R-10 mitigation.
    """
    lines = EXCEPTIONS_FILE.read_text().strip().splitlines()
    assert len(lines) == 1, (
        f"v18-substrate canary exceptions file has {len(lines)} lines; "
        f"expected 1 (header-only). Non-empty exceptions list = silent "
        f"canary flip detection per R-10 mitigation."
    )


@pytest.mark.integration
def test_seniority_py_unchanged():
    """internal notes hard invariant: ZERO edits to src/orthonym/rules/seniority.py."""
    result = subprocess.run(
        ["git", "diff", "--quiet", "src/orthonym/rules/seniority.py"],
        cwd=str(PROJECT_ROOT), capture_output=True,
    )
    assert result.returncode == 0, (
        "src/orthonym/rules/seniority.py was modified — CONTEXT D-06 hard "
        "invariant violation. The ORGM seniority cascade lives in "
        "src/orthonym/rules/organometallics.py, NEVER in seniority.py."
    )


@pytest.mark.integration
def test_composer_py_unchanged():
    """ structural enforcement: ZERO edits to composer.py."""
    result = subprocess.run(
        ["git", "diff", "--quiet", "src/orthonym/assembly/composer.py"],
        cwd=str(PROJECT_ROOT), capture_output=True,
    )
    assert result.returncode == 0, (
        "src/orthonym/assembly/composer.py was modified — ORGM-03 "
        "structural invariant violation. ORGM ships as 1 CFR row + "
        "4 source modules with ZERO composer mutation."
    )


@pytest.mark.integration
def test_namer_py_unchanged():
    """Orchestration enforcement: ZERO edits to src/orthonym/namer.py."""
    result = subprocess.run(
        ["git", "diff", "--quiet", "src/orthonym/namer.py"],
        cwd=str(PROJECT_ROOT), capture_output=True,
    )
    assert result.returncode == 0, (
        "src/orthonym/namer.py was modified — orchestration must remain "
        "unchanged. ORGM attaches via CFR row in dispatch_table.py, NEVER "
        "via namer.py mutation."
    )
