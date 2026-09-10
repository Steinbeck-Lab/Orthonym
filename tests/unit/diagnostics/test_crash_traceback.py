"""R1 (audit 2026-09-03): a native crash (SIGSEGV inside RDKit or the JVM bridge)
must leave a Python traceback, not only a JVM hs_err file. The 72 hs_err logs of
Aug 30 - Sep 2 named no Python line because nothing enabled faulthandler.
"""
import faulthandler
import subprocess
import sys


def test_enable_crash_traceback_turns_faulthandler_on():
    from orthonym.diagnostics import enable_crash_traceback

    was = faulthandler.is_enabled()
    try:
        faulthandler.disable()
        enable_crash_traceback()
        assert faulthandler.is_enabled()
        enable_crash_traceback()  # idempotent
        assert faulthandler.is_enabled()
    finally:
        if was:
            faulthandler.enable()


def test_cli_enables_faulthandler_before_naming():
    code = (
        "import faulthandler, sys\n"
        "import orthonym.cli as cli\n"
        "faulthandler.disable()\n"
        "rc = cli.main(['CCO'])\n"
        "print('FH', faulthandler.is_enabled(), rc)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert "FH True 0" in out.stdout, out.stdout + out.stderr
