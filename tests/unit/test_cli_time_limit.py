"""``--time-limit SECONDS``: a wall-clock limit on the naming of each structure (``orthonym.cli_time_limit``).

A structure that is not named in time is a line with no name and the reason code ``TIME_LIMIT``, in the
place of its line, and the next line is named as in a fresh process. The slow structure of these tests is a
patched naming call that leaves naming state behind, as a call cut off half way does, and then waits in
Python; one test (``slow``) uses a real structure that the engine needs more than 20 s for.
"""
import json
import threading
import time

import pytest

from orthonym import cli, cli_time_limit
from orthonym.namer import Orthonym
from orthonym.wallclock import leaked_naming_state

# The gate is ON: the structures are named, as in the other tests of the command line, with the OPSIN check.
pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

HANG = "CCCCCCCCCCCCCCCCCC"      # the structure whose naming never ends in these tests
LIMIT = "0.3"                    # seconds
NAMED = "CCO"
NAMED2 = "CC(=O)O"
LABEL = "no name within the time limit of 0.3 s"
REASON = f"{HANG}\tTIME_LIMIT: {LABEL}"
PROVENANCE_KEYS = {"name", "tier", "is_pin", "source", "opsin", "gates_passed", "gate_outcome", "formula",
                   "limit_code", "stereo_unexpressed", "suffix_free_prefix_name", "prefix_order_fallback",
                   "verified", "spelling_failures", "producer"}


def _hang():
    """What a naming call cut off in the middle leaves behind (naming state of the thread and of the
    context), and then no end: a wait in Python, which the alarm interrupts."""
    from orthonym.assembly import fragment_naming
    from orthonym.metrics import provenance
    fragment_naming._fragment_guard.session_depth = 2
    fragment_naming._fragment_guard.name_call_depth = 1
    provenance.best_effort_ctx.set(True)
    provenance.general_fallback_ctx.set(True)
    while True:
        time.sleep(0.005)


@pytest.fixture
def hang(monkeypatch):
    """``HANG`` never gets its name: through the tiered namer (the default, ``--provenance``, every tier above
    pin) and through ``name_compound`` (``--emit-tier pin``, ``--confidence``)."""
    real_tiered, real_compound = Orthonym.name_tiered, cli.name_compound

    def tiered(self, smiles):
        if smiles == HANG:
            _hang()
        return real_tiered(self, smiles)

    def compound(smiles, *args, **kwargs):
        if smiles == HANG:
            _hang()
        return real_compound(smiles, *args, **kwargs)

    monkeypatch.setattr(Orthonym, "name_tiered", tiered)
    monkeypatch.setattr(cli, "name_compound", compound)


def _batch(capsys, tmp_path, lines, *flags):
    """``orthonym --batch <file of lines> *flags`` in this process: (status, output lines, stderr)."""
    src = tmp_path / "in.smi"
    src.write_text("".join(line + "\n" for line in lines))
    status = cli.main(["--batch", str(src), *flags])
    captured = capsys.readouterr()
    return status, captured.out.split("\n")[:-1], captured.err


def _single(capsys, smiles, *flags):
    status = cli.main([smiles, *flags])
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def _reasons(stderr):
    return [line for line in stderr.splitlines() if line.startswith(HANG)]


# --------------------------------------------------------------------------------------------
# the line of a timed-out structure, for each --on-failure
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tier_flags", [(), ("--emit-tier", "pin")], ids=["default-tier", "pin-tier"])
def test_a_timed_out_line_has_an_empty_name_and_a_reason(hang, capsys, tmp_path, tier_flags):
    """``--on-failure empty`` (the default): the line keeps its place with an empty name, the reason is
    ``<SMILES><tab>TIME_LIMIT: <label>`` on the error stream, and the next line is named."""
    status, out, err = _batch(capsys, tmp_path, [NAMED, HANG, NAMED2], "--time-limit", LIMIT, *tier_flags)
    assert status == 0
    assert out == [f"{NAMED}\tethanol", f"{HANG}\t", f"{NAMED2}\tacetic acid"]
    assert _reasons(err) == [REASON]


@pytest.mark.parametrize("tier_flags", [(), ("--emit-tier", "pin")], ids=["default-tier", "pin-tier"])
def test_a_timed_out_line_with_on_failure_label_has_the_label(hang, capsys, tmp_path, tier_flags):
    status, out, err = _batch(capsys, tmp_path, [HANG, NAMED], "--time-limit", LIMIT, "--on-failure", "label",
                              *tier_flags)
    assert status == 0
    assert out == [f"{HANG}\t{LABEL}", f"{NAMED}\tethanol"]
    assert _reasons(err) == []


@pytest.mark.parametrize("tier_flags", [(), ("--emit-tier", "pin")], ids=["default-tier", "pin-tier"])
def test_a_timed_out_line_with_on_failure_raise_is_an_error_row(hang, capsys, tmp_path, tier_flags):
    """As for any other limit: the line gets ``ERROR:``, the run goes on and ends with status 1."""
    status, out, err = _batch(capsys, tmp_path, [HANG, NAMED], "--time-limit", LIMIT, "--on-failure", "raise",
                              *tier_flags)
    assert status == 1
    assert out == [f"{HANG}\tERROR: [TIME_LIMIT] {LABEL}", f"{NAMED}\tethanol"]


def test_a_timed_out_line_with_provenance_has_a_row_that_says_time_limit(hang, capsys, tmp_path):
    """The row has the keys of any other row, tier ``abstain`` and the limit code; ``--on-failure`` does not
    change it (``--provenance`` reports the structure as it is)."""
    for mode in ("empty", "label", "raise"):
        status, out, err = _batch(capsys, tmp_path, [NAMED, HANG, NAMED2], "--time-limit", LIMIT,
                                  "--provenance", "--on-failure", mode)
        assert status == 0
        assert [line.split("\t")[0] for line in out] == [NAMED, HANG, NAMED2]
        row = json.loads(out[1].split("\t")[1])
        assert set(row) == PROVENANCE_KEYS
        assert (row["name"], row["tier"], row["is_pin"], row["limit_code"]) == (None, "abstain", False, "TIME_LIMIT")
        assert row["formula"] == "C18H38" and row["gate_outcome"] == "not_run" and row["producer"] is None
        assert json.loads(out[0].split("\t")[1])["name"] == "ethanol"
        assert json.loads(out[2].split("\t")[1])["name"] == "acetic acid"


def test_a_timed_out_line_with_confidence_keeps_its_columns(hang, capsys, tmp_path):
    status, out, err = _batch(capsys, tmp_path, [NAMED, HANG, NAMED2], "--time-limit", LIMIT, "--confidence")
    assert status == 0
    assert [len(line.split("\t")) for line in out] == [4, 4, 4]
    assert out[1] == f"{HANG}\t\tunverified\tTIME_LIMIT"
    assert out[0].startswith(f"{NAMED}\tethanol\t") and out[2].startswith(f"{NAMED2}\tacetic acid\t")
    assert _reasons(err) == [REASON]


def test_output_goes_to_the_file_and_verbose_counts_the_timeouts(hang, capsys, tmp_path):
    src = tmp_path / "in.smi"
    src.write_text(f"{NAMED}\n{HANG}\n{HANG}\n{NAMED2}\n")
    dest = tmp_path / "names.txt"
    status = cli.main(["--batch", str(src), "--output", str(dest), "--time-limit", LIMIT, "--verbose"])
    captured = capsys.readouterr()
    assert status == 0
    assert dest.read_text().split("\n") == [f"{NAMED}\tethanol", f"{HANG}\t", f"{HANG}\t",
                                           f"{NAMED2}\tacetic acid", ""]
    # the lines that ran out of time are among the unnamed ones
    assert "Processed 4 SMILES, 0 errors, 2 unnamed, 2 timed out" in captured.out
    assert _reasons(captured.err) == [REASON, REASON]
    # without a limit the count has no such field
    status, _, err = _batch(capsys, tmp_path, [NAMED], "--verbose")
    assert "Processed 1 SMILES, 0 errors, 0 unnamed" in err and "timed out" not in err


def test_with_on_failure_raise_a_timeout_counts_as_an_error_not_as_unnamed(hang, capsys, tmp_path):
    status, _, err = _batch(capsys, tmp_path, [NAMED, HANG], "--time-limit", LIMIT, "--on-failure", "raise",
                            "--verbose")
    assert status == 1
    assert "Processed 2 SMILES, 1 errors, 0 unnamed, 1 timed out" in err


# --------------------------------------------------------------------------------------------
# the next line is named as in a fresh process
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("flags", [(), ("--emit-tier", "pin"), ("--provenance",), ("--confidence",)],
                         ids=["default", "pin", "provenance", "confidence"])
def test_a_timed_out_line_leaves_no_state_and_the_next_lines_are_named_as_in_a_fresh_run(
        hang, capsys, tmp_path, flags):
    """The patched call leaves the depth counters of the fragment recursion and the tier contexts set, as a
    call cut off half way does. After the timeout none of it is left, and every line after it has the name
    (and the provenance row, and the score) that a run without the slow structure gives it."""
    others = [NAMED, "CC(=O)Oc1ccccc1C(=O)O", "C[C@H](N)C(=O)O", NAMED2, "c1ccccc1"]
    _, fresh, _ = _batch(capsys, tmp_path, others, *flags)
    status, timed, _ = _batch(capsys, tmp_path, [others[0], HANG, *others[1:], HANG, others[0]],
                              "--time-limit", LIMIT, *flags)
    assert status == 0
    assert leaked_naming_state() == {}
    assert [line for line in timed if not line.startswith(HANG)] == fresh + [fresh[0]]


def test_the_lines_after_a_timeout_are_those_of_a_fresh_process(hang, capsys, tmp_path):
    """The same file named by a process of its own, which never met the slow structure."""
    import subprocess
    import sys
    others = [NAMED, "CC(=O)Oc1ccccc1C(=O)O", "C[C@H](N)C(=O)O", "c1ccc2ccccc2c1"]
    src = tmp_path / "others.smi"
    src.write_text("".join(line + "\n" for line in others))
    fresh = subprocess.run([sys.executable, "-m", "orthonym", "--batch", str(src)], capture_output=True,
                           text=True, check=True, timeout=300).stdout.split("\n")[:-1]
    status, timed, _ = _batch(capsys, tmp_path, [HANG, *others[:2], HANG, *others[2:]], "--time-limit", LIMIT)
    assert status == 0
    assert [line for line in timed if not line.startswith(HANG)] == fresh


# A five-residue peptide that the engine names in 0.7 to 0.9 s (measured warm): a limit of 0.25 s cuts its
# naming a third of the way through, in the real engine. The others are named in 10 to 50 ms.
REAL_SLOW = "C(CCN)CC(C(=O)NC(CC(=O)N)C(=O)NC(CS)C(=O)NC(CCC(=O)O)C(=O)O)N"


def test_real_naming_cut_midway_leaves_the_other_lines_as_in_a_fresh_process(capsys, tmp_path):
    """No patched call: the engine itself is interrupted, three times, between lines that are named. Every
    row of the other lines (name, tier, source, producer, checks) is the row of a process that never met the
    slow structure, and the slow structure, named afterwards without a limit in the same process, has the row
    it has in a fresh one."""
    import subprocess
    import sys
    probes = [NAMED, NAMED2, "c1ccccc1", "CCC", "CCCC"]
    src = tmp_path / "fresh.smi"
    src.write_text("".join(line + "\n" for line in [*probes, REAL_SLOW]))
    done = subprocess.run([sys.executable, "-m", "orthonym", "--batch", str(src), "--provenance"],
                          capture_output=True, text=True, check=True, timeout=300)
    fresh = dict(line.split("\t", 1) for line in done.stdout.split("\n")[:-1])
    assert json.loads(fresh[REAL_SLOW])["name"], "the structure of this test is named in a fresh process"

    lines = [probes[0], REAL_SLOW, probes[1], REAL_SLOW, probes[2], probes[3], REAL_SLOW, probes[4]]
    status, cut, _ = _batch(capsys, tmp_path, lines, "--time-limit", "0.25", "--provenance")
    assert status == 0 and [line.split("\t")[0] for line in cut] == lines
    rows = [(line.split("\t")[0], line.split("\t", 1)[1]) for line in cut]
    cut_slow = [row for smiles, row in rows if smiles == REAL_SLOW]
    # on a fast machine the structure may be named within the limit: then its row is the fresh one
    assert all(json.loads(row)["limit_code"] == "TIME_LIMIT" or row == fresh[REAL_SLOW] for row in cut_slow)
    assert [(smiles, row) for smiles, row in rows if smiles != REAL_SLOW] == [(s, fresh[s]) for s in probes]
    assert leaked_naming_state() == {}
    status, again, _ = _batch(capsys, tmp_path, [REAL_SLOW], "--provenance")
    assert (status, again) == (0, [f"{REAL_SLOW}\t{fresh[REAL_SLOW]}"])
    if not any(json.loads(row)["limit_code"] == "TIME_LIMIT" for row in cut_slow):
        pytest.skip("the structure was named within 0.25 s on this machine: no cut to test")


def test_the_namer_of_a_tiered_batch_is_built_anew_after_a_timeout(hang, capsys, tmp_path, monkeypatch):
    built = []
    real = cli._tiered_namer
    monkeypatch.setattr(cli, "_tiered_namer", lambda *a, **k: built.append(real(*a, **k)) or built[-1])
    status, out, _ = _batch(capsys, tmp_path, [NAMED, NAMED2, HANG, NAMED, HANG], "--time-limit", LIMIT)
    assert status == 0 and len(out) == 5
    assert len(built) == 3      # once, and once after each timeout; a run without a timeout builds one


def test_a_limit_that_never_fires_changes_nothing(capsys, tmp_path):
    """A 50-line file (named, declined, unreadable and blank lines) gives the same output, line for line and
    on the error stream, with a large limit as without one."""
    lines = (["C" * n for n in range(1, 13)] + ["C" * n + "O" for n in range(1, 11)]
             + ["C" * n + "C(=O)O" for n in range(1, 9)] + ["C" * n + "N" for n in range(1, 10)]
             + ["c1ccccc1", "Cc1ccccc1", "c1ccc2ccccc2c1", "C1CCCCC1", "CC(C)(C)c1ccccc1", "O=C=O",
                "CC(=O)Oc1ccccc1C(=O)O", "C[C@H](N)C(=O)O", "F[Xe]F", "not-a-smiles", ""])
    assert len(lines) == 50
    plain = _batch(capsys, tmp_path, lines)
    limited = _batch(capsys, tmp_path, lines, "--time-limit", "600")
    assert limited == plain
    assert plain[1][-2].startswith("not-a-smiles\tERROR")
    assert plain[1][-1] == ""


@pytest.mark.parametrize("flags", [("--emit-tier", "pin"), ("--provenance",), ("--confidence",),
                                   ("--on-failure", "label")])
def test_a_limit_that_never_fires_changes_nothing_for_the_other_modes(capsys, tmp_path, flags):
    lines = [NAMED, NAMED2, "c1ccccc1", "F[Xe]F", "C[C@H](N)C(=O)O", "", "CC(=O)Oc1ccccc1C(=O)O"]
    assert _batch(capsys, tmp_path, lines, "--time-limit", "600", *flags) == _batch(capsys, tmp_path, lines, *flags)


def test_the_start_up_is_paid_before_the_first_limit_begins(monkeypatch, capsys, tmp_path):
    """The JVM, OPSIN's parse tables and the lazily imported modules are brought up outside the limit: a
    first structure is not charged for them, and an alarm cannot land in the start-up."""
    events = []
    real_prepare, real_timed = cli_time_limit.prepare, cli_time_limit.timed
    monkeypatch.setattr(cli_time_limit, "prepare",
                        lambda *a: events.append("prepare") or real_prepare(*a))
    monkeypatch.setattr(cli_time_limit, "timed",
                        lambda *a, **k: events.append("timed") or real_timed(*a, **k))
    status, out, _ = _batch(capsys, tmp_path, [NAMED], "--time-limit", "600")
    assert (status, out) == (0, [f"{NAMED}\tethanol"])
    assert events == ["prepare", "timed"]
    # and it names the small structure of the warm-up with the namer of the run
    seen = []
    real_name = Orthonym.name_tiered
    monkeypatch.setattr(Orthonym, "name_tiered", lambda self, s: seen.append(s) or real_name(self, s))
    _batch(capsys, tmp_path, [NAMED], "--time-limit", "600")
    assert seen == [cli_time_limit.WARM_UP_SMILES, NAMED]


# --------------------------------------------------------------------------------------------
# one SMILES
# --------------------------------------------------------------------------------------------


def test_one_smiles_that_runs_out_of_time_is_a_decline_as_for_a_batch_line(hang, capsys):
    status, out, err = _single(capsys, HANG, "--time-limit", LIMIT)
    assert (status, out) == (0, "\n")
    assert _reasons(err) == [REASON]
    status, out, err = _single(capsys, HANG, "--time-limit", LIMIT, "--emit-tier", "pin")
    assert (status, out, _reasons(err)) == (0, "\n", [REASON])
    status, out, err = _single(capsys, HANG, "--time-limit", LIMIT, "--on-failure", "label")
    assert (status, out, _reasons(err)) == (0, f"{LABEL}\n", [])
    status, out, err = _single(capsys, HANG, "--time-limit", LIMIT, "--on-failure", "raise")
    assert (status, out, _reasons(err)) == (1, "", [REASON])
    status, out, err = _single(capsys, HANG, "--time-limit", LIMIT, "--verbose")
    assert (status, out) == (0, f"SMILES: {HANG}\nStyle:  pin\nName:   \n")


def test_one_smiles_with_provenance_prints_the_row(hang, capsys):
    status, out, err = _single(capsys, HANG, "--time-limit", LIMIT, "--provenance", "--on-failure", "raise")
    row = json.loads(out)
    assert status == 0 and set(row) == PROVENANCE_KEYS
    assert (row["name"], row["tier"], row["limit_code"]) == (None, "abstain", "TIME_LIMIT")
    assert _reasons(err) == []


@pytest.mark.parametrize("flags", [("--confidence",), ("--validation-stats",), ("--dispatch-stats",),
                                   ("--dump-tree",)])
def test_the_flags_that_look_inside_one_naming_have_the_limit_too(monkeypatch, capsys, flags):
    """``name`` (the stats), ``name_with_tree`` and ``name_compound`` are limited as the plain name is."""
    real_name, real_tree, real_compound = Orthonym.name, Orthonym.name_with_tree, cli.name_compound
    monkeypatch.setattr(Orthonym, "name", lambda self, s, **k: _hang() if s == HANG else real_name(self, s, **k))
    monkeypatch.setattr(Orthonym, "name_with_tree", lambda self, s: _hang() if s == HANG else real_tree(self, s))
    monkeypatch.setattr(cli, "name_compound", lambda s, *a, **k: _hang() if s == HANG else real_compound(s, *a, **k))
    status, out, err = _single(capsys, HANG, "--time-limit", LIMIT, *flags)
    assert status == 0
    assert _reasons(err) == [REASON]
    assert leaked_naming_state() == {}


def test_engine_only_has_the_limit_too(monkeypatch, capsys):
    from orthonym.assembly import general_engine
    real = general_engine.name_general
    monkeypatch.setattr(general_engine, "name_general", lambda mol, feats: _hang() if mol.GetNumAtoms() == 18 else real(mol, feats))
    # the JSON of its other declines: the structure as it is, whatever --on-failure says
    for mode in ("empty", "label", "raise"):
        status, out, err = _single(capsys, HANG, "--time-limit", LIMIT, "--engine-only", "--on-failure", mode)
        assert status == 0
        assert json.loads(out) == {"engine": "refused", "limit_code": "TIME_LIMIT", "note": LABEL}
        assert err == ""
    assert leaked_naming_state() == {}
    status, out, err = _single(capsys, NAMED, "--time-limit", "600", "--engine-only")
    assert status == 0 and json.loads(out)["engine"] == "ethan-1-ol"      # the general engine's own name, not the PIN


def test_one_smiles_that_is_named_in_time_prints_what_it_prints_without_a_limit(capsys):
    assert _single(capsys, "C[C@H](N)C(=O)O", "--time-limit", "600") == _single(capsys, "C[C@H](N)C(=O)O")
    assert _single(capsys, NAMED, "--time-limit", "600", "--provenance")[:2] == (
        0, _single(capsys, NAMED, "--provenance")[1])


# --------------------------------------------------------------------------------------------
# the option, and where no alarm can be set
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "-1", "abc", "nan", "inf", ""])
def test_the_limit_must_be_a_number_of_seconds_above_zero(capsys, value):
    with pytest.raises(SystemExit) as exc:
        cli.main([NAMED, "--time-limit", value])
    assert exc.value.code == 2
    assert "--time-limit" in capsys.readouterr().err


def test_without_an_alarm_the_run_says_so_once_and_has_no_limit(monkeypatch, capsys, tmp_path):
    """No ``SIGALRM`` (Windows) or not the main thread: one note on the error stream, then the run names
    every line, with no limit."""
    monkeypatch.setattr(cli_time_limit, "can_limit", lambda: False)
    status, out, err = _batch(capsys, tmp_path, [NAMED, NAMED2, NAMED], "--time-limit", LIMIT, "--verbose")
    assert status == 0
    assert out == [f"{NAMED}\tethanol", f"{NAMED2}\tacetic acid", f"{NAMED}\tethanol"]
    assert err.count("--time-limit needs SIGALRM on the main thread") == 1
    assert "timed out" not in err          # no limit: the counts are those of a run without one
    status, out, err = _single(capsys, NAMED, "--time-limit", LIMIT)
    assert (status, out) == (0, "ethanol\n") and err.count("--time-limit needs SIGALRM") == 1


def test_from_a_thread_other_than_the_main_one_the_run_has_no_limit_and_does_not_crash(capsys, tmp_path):
    src = tmp_path / "in.smi"
    src.write_text(f"{NAMED}\n")
    status = []
    worker = threading.Thread(target=lambda: status.append(cli.main(["--batch", str(src), "--time-limit", LIMIT])))
    worker.start()
    worker.join()
    captured = capsys.readouterr()
    assert status == [0]
    assert captured.out == f"{NAMED}\tethanol\n"
    assert captured.err.count("--time-limit needs SIGALRM on the main thread") == 1


def test_the_help_describes_the_limit(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "--time-limit SECONDS" in text and "TIME_LIMIT" in text and "SIGALRM" in text


# --------------------------------------------------------------------------------------------
# a real slow structure
# --------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_a_real_structure_that_takes_more_than_twenty_seconds_gets_no_name_and_the_run_goes_on(capsys, tmp_path):
    """A 60-heavy-atom macrocycle with a spiro centre and an N=[S-](=O) group, which the engine needs about
    4.5 s to name (an unloaded process, as the -uide anion of a lambda hydride,; it needed 20.7 s to
    decline before that anion had a class). The limit is 1 s, so the margin is four-fold."""
    cage = ("CC1C(C=CC(C2CCC2CN3CC4(CCCC5=C4C=CC(=C5)Cl)COC6=C3C=C(C=C6)C(=O)N=[S-](=O)C1C(=O)N(C)"
            "OC7=NN(C=C7C(=O)N)C)OC)OC(=O)N(C)C")
    started = time.monotonic()
    status, out, err = _batch(capsys, tmp_path, [NAMED, cage, NAMED2], "--time-limit", "1")
    elapsed = time.monotonic() - started
    assert status == 0
    assert out == [f"{NAMED}\tethanol", f"{cage}\t", f"{NAMED2}\tacetic acid"]
    assert f"{cage}\tTIME_LIMIT: no name within the time limit of 1 s" in err
    assert leaked_naming_state() == {}
    assert elapsed < 15         # the structure did not run its 20 s
