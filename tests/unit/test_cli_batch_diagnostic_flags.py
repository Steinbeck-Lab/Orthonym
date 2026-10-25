"""``--batch`` refuses the flags that look inside the naming of one SMILES.

``--validation-stats``, ``--dispatch-stats``, ``--engine-only`` and ``--dump-tree`` (and
``--format``, which only sets the layout of ``--dump-tree``) report on one structure's naming
run. ``main`` used to send ``--batch`` straight to ``_process_batch``, which ignored them: the
file was named and the flag said nothing. Now the pair is a usage error (status 2), made once in
``cli._check_flag_combinations`` before the jars are checked or the file is read, as for
``--confidence`` with a tier (``test_cli_batch_tiers.py``). One SMILES with each flag is
unchanged, and so is ``--batch`` without them.
"""
import json
import re

import pytest

from orthonym import cli

pytestmark = pytest.mark.unit

NAMED = "CCO"

# (the flags given besides --batch, the flags the message names)
REFUSED = [
    (("--validation-stats",), "--validation-stats"),
    (("--dispatch-stats",), "--dispatch-stats"),
    (("--engine-only",), "--engine-only"),
    (("--dump-tree",), "--dump-tree"),
    (("--format", "json"), "--format"),
    # --format text is the layout of --dump-tree without the option, and still not a batch option
    (("--format", "text"), "--format"),
    (("--dump-tree", "--format", "json"), "--dump-tree, --format"),
    (("--emit-tier", "best-effort", "--dump-tree"), "--dump-tree"),
    (("--provenance", "--engine-only"), "--engine-only"),
]


@pytest.fixture
def nothing_is_run(monkeypatch):
    """A refusal comes before the jar check and before the batch run, so neither may be reached."""
    monkeypatch.setattr(cli, "_jars_ready", lambda parsed: pytest.fail("the jar check was reached"))
    monkeypatch.setattr(cli, "_process_batch", lambda *a, **k: pytest.fail("the batch run was reached"))


def _refused(capsys, args):
    with pytest.raises(SystemExit) as stop:
        cli.main(args)
    captured = capsys.readouterr()
    assert captured.out == ""
    return stop.value.code, captured.err


@pytest.mark.parametrize("flags,named", REFUSED)
def test_a_diagnostic_flag_with_batch_is_a_usage_error(capsys, nothing_is_run, tmp_path, flags, named):
    # the file does not exist: the refusal is made before it is read
    status, err = _refused(capsys, ["--batch", str(tmp_path / "absent.smi"), *flags])
    assert status == 2
    assert f"orthonym: error: {named} cannot be combined with --batch" in err


@pytest.mark.parametrize("flags,named", REFUSED)
def test_the_order_of_the_options_does_not_matter(capsys, nothing_is_run, tmp_path, flags, named):
    status, err = _refused(capsys, [*flags, "-b", str(tmp_path / "absent.smi")])
    assert status == 2
    assert f"{named} cannot be combined with --batch" in err


def test_every_flag_given_is_named_in_the_order_of_the_table(capsys, nothing_is_run, tmp_path):
    status, err = _refused(capsys, ["--batch", str(tmp_path / "in.smi"), "--format", "json", "--dump-tree",
                                    "--dispatch-stats", "--engine-only", "--validation-stats"])
    assert status == 2
    assert ("--validation-stats, --dispatch-stats, --engine-only, --dump-tree, --format "
            "cannot be combined with --batch") in err


def test_the_confidence_message_is_unchanged_and_comes_first(capsys, nothing_is_run, tmp_path):
    status, err = _refused(capsys, ["--batch", str(tmp_path / "in.smi"), "--confidence", "--provenance",
                                    "--dump-tree"])
    assert status == 2
    assert cli._CONFIDENCE_NEEDS_PIN in err
    assert "--dump-tree cannot be combined" not in err


# --------------------------------------------------------------------------------------------
# what is not refused
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("flags", [
    (),
    ("--emit-tier", "best-effort"),
    ("--emit-tier", "pin"),
    ("--provenance",),
    ("--confidence",),
    ("--trivial",),
    ("--enable-triviality-controller",),
    ("--binding-proof", "audit"),
    ("--on-failure", "label"),
    ("--verbose",),
])
def test_batch_without_the_diagnostic_flags_still_reaches_the_batch_run(monkeypatch, tmp_path, flags):
    reached = []
    monkeypatch.setattr(cli, "_jars_ready", lambda parsed: None)
    monkeypatch.setattr(cli, "_process_batch", lambda *a, **k: reached.append((a, k)) or 0)
    src = tmp_path / "in.smi"
    assert cli.main(["--batch", str(src), *flags]) == 0
    assert len(reached) == 1 and reached[0][0][0] == str(src)


def test_batch_and_batch_with_a_tier_name_the_file_as_before(capsys, tmp_path):
    src = tmp_path / "in.smi"
    src.write_text(f"{NAMED}\n\n")
    for flags in ((), ("--emit-tier", "best-effort")):
        assert cli.main(["--batch", str(src), *flags]) == 0
        assert capsys.readouterr().out == f"{NAMED}\tethanol\n\n"


# --------------------------------------------------------------------------------------------
# one SMILES with each flag
# --------------------------------------------------------------------------------------------


def _single(capsys, *args):
    status = cli.main(list(args))
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def test_one_smiles_with_validation_stats_is_unchanged(capsys):
    status, out, err = _single(capsys, NAMED, "--validation-stats")
    assert (status, out) == (0, "ethanol\n")
    assert "Validation stats: {" in err


def test_one_smiles_with_dispatch_stats_is_unchanged(capsys):
    status, out, err = _single(capsys, NAMED, "--dispatch-stats")
    assert (status, out) == (0, "ethanol\n")
    assert "CFR Dispatch Stats" in err and "Inner Dispatch Stats" in err


def test_one_smiles_with_engine_only_is_unchanged(capsys):
    status, out, _ = _single(capsys, NAMED, "--engine-only")
    row = json.loads(out)
    assert status == 0 and row["engine"] == "ethan-1-ol" and row["e1_ok"] is True


def test_one_smiles_with_dump_tree_is_unchanged_in_both_layouts(capsys):
    status, text, _ = _single(capsys, NAMED, "--dump-tree")
    assert status == 0 and text.startswith("NameTree: ethanol  (class_id=")
    # --format text is the layout without the option
    assert _single(capsys, NAMED, "--dump-tree", "--format", "text") == (0, text, "")
    status, out, _ = _single(capsys, NAMED, "--dump-tree", "--format", "json")
    payload = json.loads(out)
    assert status == 0 and payload["name"] == "ethanol" and payload["tree"] is not None


def test_one_smiles_with_format_alone_is_unchanged(capsys):
    """For one SMILES without --dump-tree the layout is not used, as before."""
    assert _single(capsys, NAMED, "--format", "json") == (0, "ethanol\n", "")


# --------------------------------------------------------------------------------------------
# the help says it
# --------------------------------------------------------------------------------------------


def test_the_help_of_each_flag_and_of_batch_says_so(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--help"])
    assert stop.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    for ending in ("candidate name. Not with --batch.", "the engine took. Not with --batch.",
                   "instead of the name. Not with --batch.", "do not use it as a name. Not with --batch.",
                   "or json. Not with --batch."):
        assert ending in text, ending
    assert re.search(r"--validation-stats, --dispatch-stats, --engine-only, --dump-tree and --format "
                     r"look inside one SMILES and stop with a usage error here", text)
