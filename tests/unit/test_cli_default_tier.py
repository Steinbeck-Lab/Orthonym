"""The command line runs the best-effort tier by default, and says so once when a name is below ``pin_verified``
(lane DF).

User decisions 2026-10-25: the default is the best-effort tier as it runs when asked for by name; it prints a
bare name as before; ``--provenance`` shows the tier; the run prints ONE note on the error stream, and only
when at least one returned name is below ``pin_verified`` and no ``--emit-tier`` was given; ``--emit-tier pin``
is the old default, byte for byte; ``--confidence`` applies to the default; ``--batch`` has the same default.
"""
import json

import pytest

from orthonym import cli, name_compound

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

PIN = "CCO"                                  # the strict tier names it: pin_verified
BELOW_PIN = "OCCNCc1ccccc1"                  # the strict tier declines it; the best-effort name is pin_unverified
BELOW_PIN_NAME = "2-(benzylamino)ethan-1-ol"
NO_NAME = "O=[U](=O)=O"                      # no tier names it (UNSUPPORTED_ELEMENT)
NOTE = cli._BELOW_PIN_NOTE


def _run(capsys, *argv):
    status = cli.main(list(argv))
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def _notes(stderr: str):
    return [line for line in stderr.splitlines() if line.startswith("orthonym: note:")]


def test_the_default_prints_the_bare_name_and_no_note_for_a_pin(capsys):
    status, out, err = _run(capsys, PIN)
    assert (status, out, err) == (0, "ethanol\n", "")


def test_a_name_below_pin_verified_prints_the_name_and_one_note(capsys):
    status, out, err = _run(capsys, BELOW_PIN)
    assert (status, out) == (0, BELOW_PIN_NAME + "\n")
    assert _notes(err) == [NOTE] and err.count("\n") == 1


def test_the_note_names_the_default_and_the_pin_only_tier():
    assert "--emit-tier pin" in NOTE and "default" in NOTE and "--provenance" in NOTE


def test_emit_tier_pin_is_the_old_default_and_prints_no_note(capsys):
    assert _run(capsys, PIN, "--emit-tier", "pin") == (0, "ethanol\n", "")
    status, out, err = _run(capsys, BELOW_PIN, "--emit-tier", "pin")
    assert (status, out) == (0, "\n")
    assert _notes(err) == [] and "NO_VERIFIED_PIN" in err
    # byte for byte what name_compound(emit_tier="pin") gives
    assert name_compound(BELOW_PIN, emit_tier="pin") == ""
    # the PIN-only tier gives the record of the release before: no tier key, no note
    status, out, err = _run(capsys, PIN, "--confidence", "--emit-tier", "pin")
    assert status == 0 and "Tier:" not in out and err == ""


def test_a_tier_named_with_emit_tier_prints_no_note(capsys):
    """The note says what the default does; a caller who named a tier knows."""
    status, out, err = _run(capsys, BELOW_PIN, "--emit-tier", "best-effort")
    assert (status, out, err) == (0, BELOW_PIN_NAME + "\n", "")


def test_provenance_shows_the_tier_and_prints_no_note(capsys):
    status, out, err = _run(capsys, BELOW_PIN, "--provenance")
    row = json.loads(out)
    assert (status, row["name"], row["tier"]) == (0, BELOW_PIN_NAME, "pin_unverified")
    assert err == ""
    assert json.loads(_run(capsys, PIN, "--provenance")[1])["tier"] == "pin_verified"


def test_a_decline_of_the_default_prints_the_reason_and_no_note(capsys):
    status, out, err = _run(capsys, NO_NAME)
    assert (status, out) == (0, "\n")
    assert _notes(err) == [] and "UNSUPPORTED_ELEMENT" in err


def test_confidence_applies_to_the_default(capsys):
    status, out, err = _run(capsys, PIN, "--confidence")
    assert status == 0 and out.startswith("Name:       ethanol\n") and "Tier:       pin_verified" in out
    assert err == ""
    status, out, err = _run(capsys, BELOW_PIN, "--confidence")
    assert status == 0 and f"Name:       {BELOW_PIN_NAME}" in out and "Tier:       pin_unverified" in out
    assert _notes(err) == [NOTE]
    # the old default with a confidence record, as before
    status, out, err = _run(capsys, PIN, "--confidence", "--emit-tier", "pin")
    assert status == 0 and out.startswith("Name:       ethanol\n") and err == ""


def test_confidence_with_a_tier_named_gives_the_record_of_that_tier_and_no_note(capsys):
    status, out, err = _run(capsys, BELOW_PIN, "--confidence", "--emit-tier", "best-effort")
    assert status == 0 and f"Name:       {BELOW_PIN_NAME}" in out and "Tier:       pin_unverified" in out
    assert err == ""


@pytest.mark.parametrize("flags", [("--provenance",), ("--emit-tier", "best-effort", "--provenance")])
def test_confidence_has_no_row_to_go_with_a_provenance_row(capsys, flags):
    with pytest.raises(SystemExit) as stop:
        cli.main([PIN, "--confidence", *flags])
    assert stop.value.code == 2
    assert "--confidence cannot be combined" in capsys.readouterr().err


@pytest.mark.parametrize("smiles, label", [("N.N.Cl[Ni]Cl", "nickel compound (not supported)"),
                                          ("*CC", "compound with wildcard atoms (not supported)"),
                                          ("O=[U](=O)=O", "inorganic compound (not supported)")])
def test_a_structure_with_no_name_keeps_its_decline_text(capsys, smiles, label):
    """The reason line and the ``--on-failure label`` text are those of the default before it was the
    best-effort tier: the element is named, not a generic catalogue text."""
    status, out, err = _run(capsys, smiles)
    assert (status, out) == (0, "\n") and label in err and _notes(err) == []
    status, out, err = _run(capsys, smiles, "--on-failure", "label")
    assert (status, out.strip()) == (0, label)


def _batch(capsys, tmp_path, lines, *flags):
    src = tmp_path / "in.smi"
    src.write_text("".join(line + "\n" for line in lines))
    status = cli.main(["--batch", str(src), *flags])
    captured = capsys.readouterr()
    return status, captured.out.split("\n")[:-1], captured.err


def test_a_batch_has_the_default_and_prints_the_note_once(capsys, tmp_path):
    lines = [PIN, BELOW_PIN, "", BELOW_PIN, NO_NAME]
    status, out, err = _batch(capsys, tmp_path, lines)
    assert status == 0
    assert out == [f"{PIN}\tethanol", f"{BELOW_PIN}\t{BELOW_PIN_NAME}", "", f"{BELOW_PIN}\t{BELOW_PIN_NAME}",
                   f"{NO_NAME}\t"]
    assert _notes(err) == [NOTE]                                  # once, though two lines are below pin_verified


def test_a_batch_without_a_name_below_pin_verified_prints_no_note(capsys, tmp_path):
    status, out, err = _batch(capsys, tmp_path, [PIN, "c1ccccc1"])
    assert (status, out) == (0, [f"{PIN}\tethanol", "c1ccccc1\tbenzene"])
    assert _notes(err) == []


def test_a_batch_of_the_old_default_and_of_a_named_tier_prints_no_note(capsys, tmp_path):
    status, out, err = _batch(capsys, tmp_path, [PIN, BELOW_PIN], "--emit-tier", "pin")
    assert (status, out) == (0, [f"{PIN}\tethanol", f"{BELOW_PIN}\t"])
    assert _notes(err) == []
    status, out, err = _batch(capsys, tmp_path, [PIN, BELOW_PIN], "--emit-tier", "best-effort")
    assert (status, out) == (0, [f"{PIN}\tethanol", f"{BELOW_PIN}\t{BELOW_PIN_NAME}"])
    assert _notes(err) == []


def test_a_batch_with_confidence_has_the_default_and_the_note(capsys, tmp_path):
    status, out, err = _batch(capsys, tmp_path, [PIN, BELOW_PIN], "--confidence")
    assert status == 0
    assert [line.split("\t")[1] for line in out] == ["ethanol", BELOW_PIN_NAME]
    assert _notes(err) == [NOTE]


def test_the_default_batch_row_of_a_name_equals_the_row_of_one_smiles(capsys, tmp_path):
    status, out, err = _batch(capsys, tmp_path, [PIN, BELOW_PIN], "--provenance")
    assert status == 0 and err == ""
    for smiles, line in zip([PIN, BELOW_PIN], out):
        assert json.loads(line.partition("\t")[2]) == json.loads(_run(capsys, smiles, "--provenance")[1])
