"""``orthonym --batch FILE --jobs N`` names the lines in N worker processes with the output of ``--jobs 1``.

The workers are ``spawn`` processes (``orthonym.batch_pool.run_ordered``), each with its own namer built from the same
settings; a worker returns what it wrote on the error stream for each line, and the parent prints the lines and the
reasons in input order. RDKit stamps its own messages with the time of day, so the error streams are compared
without that stamp. Each run is a fresh ``orthonym`` process.
"""
import os
import re
import subprocess
import sys
import tempfile
import time

import pytest

from orthonym import cli

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

# a blank line, a line RDKit cannot read, a structure with no name (UNSUPPORTED_ELEMENT), one below the PIN
PANEL = ["CCO", "", "c1ccccc1", "not_a_smiles", "CC(=O)O", "F[Xe]F", "C1CC1", "OOc1ccccc1",
         "C=C(C)[C@@H]1CC=C(CO[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)CC1"]
_STAMP = re.compile(r"^\[\d\d:\d\d:\d\d\] ", re.M)


def _batch(tmp_path, lines, *flags, env=None, timeout=300):
    src = tmp_path / "in.smi"
    src.write_text("".join(line + "\n" for line in lines))
    out = subprocess.run([sys.executable, "-m", "orthonym", "--batch", str(src), *flags],
                         capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})})
    return out.returncode, out.stdout, _STAMP.sub("", out.stderr)


@pytest.mark.parametrize("flags", [(), ("--provenance",), ("--emit-tier", "pin"), ("--on-failure", "label")])
def test_jobs_output_equals_serial(tmp_path, flags):
    serial = _batch(tmp_path, PANEL, "-v", *flags)
    assert serial[1].count("\n") == len(PANEL)
    assert _batch(tmp_path, PANEL, "-v", "--jobs", "3", *flags) == serial


def test_jobs_does_not_wait_on_a_busy_one_slot_budget(tmp_path):
    """As on CI (4 cores, one JVM slot), with the slot held by another process: the workers reserve no slot, as the
    serial run reserves none, so the run neither waits for the budget nor writes a warning of it."""
    from orthonym.jvm_budget import jvm_slots
    env = {"ORTHONYM_JVM_SLOTS": "1", "ORTHONYM_JVM_SLOT_DIR": tempfile.mkdtemp()}
    os.environ.update(env)
    try:
        with jvm_slots(1, purpose="test-holder"):
            started = time.monotonic()
            got = _batch(tmp_path, ["CCO"] * 6, "--jobs", "3", env=env, timeout=120)
            took = time.monotonic() - started
    finally:
        for k in env:
            os.environ.pop(k, None)
    assert got == (0, "CCO\tethanol\n" * 6, "") and took < 25, (got, took)


def test_jobs_must_be_at_least_one():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--batch", "x", "--jobs", "0"])
    assert exc.value.code == 2


def test_a_crashed_worker_gives_an_error_line():
    out = cli._worker_crash("CCO")
    assert (out.line, out.error) == ("CCO\tERROR: the naming process ended unexpectedly", 1)
