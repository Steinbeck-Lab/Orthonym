"""The command line and the Python API print no internal telemetry (SF item 3).

``orthonym "<SMILES>"`` used to print, on the error stream, the log lines of the engine's own
refusals and gates ("self_consistency rejected (different molecule)", "OPSIN validity gate
suppressed unparseable name", "Could not name compound substituent at locant 1", the stereo
backstop,...) on 224 of the 500 rows of a dev split at the default tier, and RDKit's own warnings
("Omitted undefined stereo"). Those lines are DEBUG records now. The census tools read them at
DEBUG (``scripts/measure_breadth.py``, ``orthonym.metrics.breadth``), so the message texts did
not change. What stays a warning: the reason line of a structure that gets no name
(``failure_mode``, issue 45), and the records that tell a user about a broken or degraded
environment or an engine fault (see the commit message for the list). The command line also
turns RDKit's warnings AND errors off while it names (the engine's internal SMILES probes make
them); RDKit speaks only when it cannot read the user's own SMILES.

The tests read the message and its level, never a count of rows.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem, rdBase

from orthonym import name_compound
from orthonym.cli import _emit_tier_flags, _readable, _reports_unnamed
from orthonym.rdkit_log import rdkit_logs_off, rdkit_logs_on

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

ROOT = Path(__file__).resolve().parents[2]

# (SMILES from a dev split, the start of the log message it makes at the default tier). Each row logged
# the message at WARNING before this change.
INTERNAL = [
    ("OC(O)c1ccc(Br)c(C(O)O)c1", "self_consistency rejected (different molecule)"),
    ("CC(=O)NCCSC(=O)/C=C/CCC[C@H](O)/C=C/CC(C)O", "OPSIN validity gate suppressed unparseable name"),
    ("C[N+](C)(C)[O-]", "OPSIN grammar validation failed"),
    ("Cc1ccc2cc(N3CC[C@H](NC(=O)Cn4cnc(C#N)n4)C3)ccc2n1", "Stereo backstop"),
    ("NC(CO)C(=O)NNCc1ccc(O)c(O)c1O", "Could not name compound substituent at locant"),
    ("NC(CO)C(=O)NNCc1ccc(O)c(O)c1O", "universal_pipeline_unnameable_substituent substituent_skip"),
    ("CCN(CC)N=O", "substituent_recursion_depth_exceeded substituent_skip"),
    # these two were logged through the ROOT logger (bare ``logging.warning``), outside the
    # ``orthonym`` logger the census tools read
    ("CC(=O)OC/C(C)=C\\C1CCC(C)C2CCC(C)=C12", "Fused ring naming failed for ortho-fused system"),
    ("C[C@@H]1CC[C@@H]2C=C(CO)[C@H]3C[C@](C)(C(=O)O)C[C@]132", "Polycyclic-bridged naming failed for molecule"),
]


def _engine_records(caplog):
    """The records of the ``orthonym`` loggers and of the root logger (the library's own bare
    ``logging.warning`` calls)."""
    return [r for r in caplog.records if r.name == "root" or r.name.startswith("orthonym")]


@pytest.mark.parametrize("smiles,message", INTERNAL)
def test_internal_telemetry_is_logged_at_debug_not_warning(smiles, message, caplog):
    with caplog.at_level(logging.DEBUG):
        name_compound(smiles)
    records = _engine_records(caplog)
    said = [r for r in records if r.getMessage().startswith(message)]
    assert said, f"the census tools lost the line {message!r}: {[r.getMessage()[:60] for r in records][:8]}"
    assert {r.levelno for r in said} == {logging.DEBUG}, message
    # nothing else at WARNING either, except the reason line of a structure that gets no name
    loud = [r for r in records
            if r.levelno >= logging.WARNING and getattr(r, "orthonym_code", None) is None]
    assert not loud, [r.getMessage()[:80] for r in loud]


def test_a_terminal_fragment_refusal_is_debug_not_error(caplog):
    # a best-effort row (pc7) whose composite fragment name failed the completeness check:
    # the producer refuses, the engine names the structure another way
    smiles = ("COC(=O)[C@]1([C@@H]2CCC(=O)O2)CC(=O)c2c(O)cc(C)c(-c3c(C)cc(O)c4c3O[C@@]"
              "(C(=O)OC)([C@@H]3CCC(=O)O3)CC4=O)c2O1")
    with caplog.at_level(logging.DEBUG):
        name_compound(smiles, **_emit_tier_flags("best-effort"))
    said = [r for r in _engine_records(caplog)
            if r.getMessage().startswith("terminal_fragment: composite completeness violated")]
    assert said and {r.levelno for r in said} == {logging.DEBUG}
    assert not [r for r in _engine_records(caplog) if r.levelno >= logging.ERROR]


def test_the_reason_line_of_an_unnamed_structure_stays_a_warning(caplog):
    smiles = "OC(O)c1ccc(Br)c(C(O)O)c1"
    with caplog.at_level(logging.DEBUG):
        assert name_compound(smiles) == ""
    reasons = [r for r in _engine_records(caplog) if getattr(r, "orthonym_code", None)]
    assert [(r.name, r.levelno, r.orthonym_smiles) for r in reasons] == [
        ("orthonym", logging.WARNING, smiles)]


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------


def _cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")])
    proc = subprocess.run([sys.executable, "-m", "orthonym", *args], capture_output=True,
                          text=True, env=env, cwd=ROOT, timeout=600)
    return proc.returncode, proc.stdout, proc.stderr


def test_cli_prints_only_the_reason_line_for_an_unnamed_structure():
    smiles = "OC(O)c1ccc(Br)c(C(O)O)c1"
    code, out, err = _cli(smiles)
    assert code == 0 and out == "\n"
    lines = err.splitlines()
    assert len(lines) == 1 and lines[0].startswith(f"{smiles}\tUNNAMEABLE: "), err


def test_cli_prints_nothing_on_stderr_for_a_name():
    code, out, err = _cli("C[N+](C)(C)[O-]")
    assert code == 0 and out.strip() != ""
    assert err == ""


def test_cli_prints_no_rdkit_error_for_the_engines_own_probes():
    # naming this structure makes RDKit print "Can't kekulize mol" for the engine's internal
    # SMILES probes; it is not the user's input, so the only line is the reason line
    smiles = "Cc1ccc2cc(N3CC[C@H](NC(=O)Cn4cnc(C#N)n4)C3)ccc2n1"
    code, out, err = _cli(smiles)
    assert code == 0
    assert all(ln.startswith(smiles + "\t") for ln in err.splitlines()), err
    assert len(err.splitlines()) <= 1, err


@pytest.mark.parametrize("smiles,rdkit_says", [
    ("C1CC", "SMILES Parse Error"),
    ("c1cccc1", "Can't kekulize"),
    ("C(C)(C)(C)(C)C", "Explicit valence"),
])
def test_cli_an_unreadable_smiles_gets_rdkits_reason_and_the_reason_line(smiles, rdkit_says):
    code, out, err = _cli(smiles)
    assert (code, out) == (1, "")
    lines = err.splitlines()
    assert lines[-1] == f"{smiles}\tINVALID_SMILES: RDKit cannot read this SMILES"
    assert any(rdkit_says in ln for ln in lines[:-1]), err


def test_cli_batch_keeps_the_error_row_of_an_unreadable_line_and_says_why(tmp_path):
    batch = tmp_path / "in.smi"
    batch.write_text("C1CC\nCCO\n")
    code, out, err = _cli("--batch", str(batch))
    assert code == 1
    assert out.splitlines() == ["C1CC\tERROR: Invalid SMILES: C1CC", "CCO\tethanol"]
    assert "SMILES Parse Error" in err
    assert err.splitlines()[-1] == "C1CC\tINVALID_SMILES: RDKit cannot read this SMILES"


def test_cli_does_not_print_rdkit_warnings():
    # RDKit writes "WARNING: Omitted undefined stereo" while the library builds an InChI of
    # this structure
    code, out, err = _cli("NC(CCC(=O)NCCO)C(=O)O")
    assert code == 0 and out.strip() != ""
    assert "WARNING" not in err, err


# ---------------------------------------------------------------------------
# RDKit's own log streams
# ---------------------------------------------------------------------------


@pytest.fixture
def rdkit_warning_and_error_on():
    """Other test modules switch every RDKit stream off for the whole process at import; this
    test needs the two streams on, and gives them back as they were."""
    from rdkit import RDLogger
    was = rdBase.LogStatus()
    RDLogger.EnableLog("rdApp.warning")
    RDLogger.EnableLog("rdApp.error")
    yield
    for stream in ("rdApp.warning", "rdApp.error"):
        if f"{stream}:enabled" not in was:
            RDLogger.DisableLog(stream)


def test_the_cli_wrapper_turns_rdkit_warnings_and_errors_off_and_back_on(capfd, rdkit_warning_and_error_on):
    before = rdBase.LogStatus()
    capfd.readouterr()
    Chem.MolToInchi(Chem.MolFromSmiles("CC(Cl)Br"))
    Chem.MolFromSmiles("c1ccc")
    controls = capfd.readouterr().err
    assert "Omitted undefined stereo" in controls and "SMILES Parse Error" in controls   # RDKit does print both

    @_reports_unnamed
    def run():
        status = rdBase.LogStatus()
        assert "rdApp.warning:disabled" in status
        assert "rdApp.error:disabled" in status
        Chem.MolToInchi(Chem.MolFromSmiles("CC(Cl)Br"))         # an undefined-stereo warning
        Chem.MolFromSmiles("c1ccc")                               # a parse error of an internal probe
        capfd.readouterr()
        assert not _readable("c1ccc")                             # the user's input: RDKit says why
        assert "SMILES Parse Error" in capfd.readouterr().err
        assert "rdApp.error:disabled" in rdBase.LogStatus()       # and is silent again

    run()
    assert rdBase.LogStatus() == before


def test_rdkit_logs_off_puts_each_stream_back_as_it_was():
    before = rdBase.LogStatus()
    with rdkit_logs_off("rdApp.*"):
        assert "enabled" not in rdBase.LogStatus()
    assert rdBase.LogStatus() == before


def test_rdkit_logs_off_does_not_enable_a_stream_that_was_off(rdkit_warning_and_error_on):
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.warning")
    try:
        with rdkit_logs_off("rdApp.warning", "rdApp.error"):
            pass
        status = rdBase.LogStatus()
        assert "rdApp.warning:disabled" in status
        assert "rdApp.error:enabled" in status
    finally:
        RDLogger.EnableLog("rdApp.warning")


def test_building_the_morpheme_lexicon_leaves_rdkit_logging_as_it_found_it():
    # it used to end with EnableLog("rdApp.*"), which also switched the info and debug streams
    # on and undid a caller's rdApp.warning silencing
    from orthonym.validation import name_morphemes
    with rdkit_logs_off("rdApp.debug", "rdApp.info", "rdApp.warning"):
        before = rdBase.LogStatus()
        name_morphemes._build_lexicon()
        assert rdBase.LogStatus() == before


def test_rdkit_logs_on_puts_each_stream_back_as_it_was():
    with rdkit_logs_off("rdApp.*"):
        before = rdBase.LogStatus()
        with rdkit_logs_on("rdApp.error"):
            assert "rdApp.error:enabled" in rdBase.LogStatus()
            assert "rdApp.warning:disabled" in rdBase.LogStatus()
        assert rdBase.LogStatus() == before
