"""``--batch`` honours ``--emit-tier``, ``--provenance`` and ``--enable-triviality-controller``.

Before this, ``orthonym --batch f.smi --emit-tier best-effort`` printed the names of the default
tier and said nothing: ``main`` did not pass the tier, the provenance flag or the triviality
flag to ``_process_batch``. A line of a batch file is now named as one SMILES is: through one
``Orthonym`` built from the tier flags (``cli._tiered_namer``) and its ``name_tiered``, with the
text of the name and the ``--on-failure`` decision shared with the single-SMILES path
(``cli._tiered_name_text``). The output keeps the contract of
``test_cli_batch_line_count.py``: one line per input line, in order.
"""
import json
from unittest import mock

import pytest

from orthonym import cli

# The gate is ON: the default-tier decline below is a decision of the OPSIN-checked PIN path.
pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

# The default tier declines this molecule (NO_VERIFIED_PIN) and best-effort names it. The name is
# the one tests/unit/test_emit_tier_flags.py pins for the single-SMILES path nesting,
# alphanumerical order of the prefixes).
WIDE = "CC(C)(C(=O)Nc1cccc(F)c1)N1CCC(c2nc(-c3cc4ccccc4o3)cs2)CC1"
WIDE_NAME = ("2-{4-[4-(1-benzofuran-2-yl)-1,3-thiazol-2-yl]piperidin-1-yl}"
             "-N-(3-fluorophenyl)-2-methylpropanamide")
# No name at any tier (UNSUPPORTED_ELEMENT), so every tier has a decline to report.
NO_NAME = "O=[U](=O)=O"
NAMED = "CCO"

PROVENANCE_KEYS = {"name", "tier", "is_pin", "source", "opsin", "gates_passed", "gate_outcome",
                   "formula", "limit_code", "stereo_unexpressed", "suffix_free_prefix_name",
                   "prefix_order_fallback", "verified", "spelling_failures", "producer"}


def _batch(capsys, tmp_path, lines, *flags):
    """``orthonym --batch <file of lines> *flags`` in this process: (status, output lines, stderr)."""
    src = tmp_path / "in.smi"
    src.write_text("".join(line + "\n" for line in lines))
    status = cli.main(["--batch", str(src), *flags])
    captured = capsys.readouterr()
    assert captured.out == "" or captured.out.endswith("\n")
    return status, captured.out.split("\n")[:-1], captured.err


def _single(capsys, smiles, *flags):
    """``orthonym <smiles> *flags`` in this process: (status, stdout, stderr)."""
    status = cli.main([smiles, *flags])
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def _reasons(stderr):
    """The '<SMILES><tab><CODE>: <label>' lines of the error stream."""
    return [line for line in stderr.splitlines() if "\t" in line]


# --------------------------------------------------------------------------------------------
# the tier
# --------------------------------------------------------------------------------------------


def test_the_default_tier_declines_the_molecule_the_tests_use(capsys, tmp_path):
    """The control: without a tier the molecule has no name and a reason, so the next test
    cannot pass by naming it at the default tier."""
    status, out, err = _batch(capsys, tmp_path, [NAMED, WIDE], "--emit-tier", "pin")
    assert status == 0
    assert out == [f"{NAMED}\tethanol", f"{WIDE}\t"]
    assert [line.split("\t")[1].split(":")[0] for line in _reasons(err)] == ["NO_VERIFIED_PIN"]


def test_best_effort_batch_names_what_the_default_tier_declines(capsys, tmp_path):
    status, out, err = _batch(capsys, tmp_path, [NAMED, WIDE, NO_NAME], "--emit-tier", "best-effort")
    assert status == 0
    assert out == [f"{NAMED}\tethanol", f"{WIDE}\t{WIDE_NAME}", f"{NO_NAME}\t"]
    # the one line without a name is reported once, as for one SMILES
    assert [line.split("\t")[0] for line in _reasons(err)] == [NO_NAME]


def test_the_namer_is_built_once_for_the_whole_file(capsys, tmp_path):
    built, called = [], []
    real = cli._tiered_namer

    def _spy(*args, **kwargs):
        namer = real(*args, **kwargs)
        real_name_tiered = namer.name_tiered
        namer.name_tiered = lambda smiles: called.append(smiles) or real_name_tiered(smiles)
        built.append(namer)
        return namer

    lines = [NAMED, "", WIDE, NAMED, "  ", NO_NAME]
    with mock.patch.object(cli, "_tiered_namer", _spy):
        status, out, _ = _batch(capsys, tmp_path, lines, "--emit-tier", "best-effort")
    assert status == 0 and len(out) == len(lines)
    assert len(built) == 1
    assert called == [NAMED, WIDE, NAMED, NO_NAME]      # every named line, none of the blanks
    assert built[0]._general_fallback_unverified is True


def test_a_pin_only_batch_does_not_build_a_tiered_namer(capsys, tmp_path):
    """``--emit-tier pin``, the default of releases up to 1.0.x: the lines go through ``name_compound`` as
    they always did. (The default is the best-effort tier since lane DF, named through a tiered namer.)"""
    with mock.patch.object(cli, "_tiered_namer", side_effect=AssertionError("tiered namer built")):
        status, out, err = _batch(capsys, tmp_path, [NAMED, "", NO_NAME], "--emit-tier", "pin")
        assert (status, out) == (0, [f"{NAMED}\tethanol", "", f"{NO_NAME}\t"])
        # --binding-proof alone stays on name_compound with the PIN-only tier too
        assert _batch(capsys, tmp_path, [NAMED], "--emit-tier", "pin")[:2] == (0, [f"{NAMED}\tethanol"])
        assert _batch(capsys, tmp_path, [NAMED], "--emit-tier", "pin", "--binding-proof", "audit")[:2] == (
            0, [f"{NAMED}\tethanol"])


# --------------------------------------------------------------------------------------------
# the provenance row
# --------------------------------------------------------------------------------------------


def test_batch_provenance_gives_the_row_of_one_smiles_on_each_line(capsys, tmp_path):
    lines = [NAMED, "", WIDE, NO_NAME]
    for flags in (("--provenance",), ("--provenance", "--emit-tier", "best-effort")):
        status, out, _ = _batch(capsys, tmp_path, lines, *flags)
        assert status == 0
        assert len(out) == len(lines) and out[1] == ""
        for smiles, line in zip(lines, out):
            if not smiles:
                continue
            first, _, column = line.partition("\t")
            assert first == smiles
            row = json.loads(column)
            assert set(row) == PROVENANCE_KEYS
            # the row of one SMILES, key for key and value for value
            single_status, single_out, _ = _single(capsys, smiles, *flags)
            assert single_status == 0
            assert column == single_out.rstrip("\n")
            assert row == json.loads(single_out)


def test_batch_provenance_rows_report_a_decline_in_the_row(capsys, tmp_path):
    """``--on-failure`` does not change a row: the reason is its ``limit_code``, and nothing is
    added to the error stream (one SMILES prints none either)."""
    for mode in ("empty", "label", "raise"):
        status, out, err = _batch(capsys, tmp_path, [NO_NAME, NAMED], "--provenance",
                                  "--emit-tier", "best-effort", "--on-failure", mode)
        assert status == 0, mode
        row = json.loads(out[0].split("\t", 1)[1])
        assert (row["name"], row["tier"], row["limit_code"]) == (None, "abstain", "UNSUPPORTED_ELEMENT")
        assert json.loads(out[1].split("\t", 1)[1])["name"] == "ethanol"
        assert _reasons(err) == [], mode


# --------------------------------------------------------------------------------------------
# a tier batch is the one-SMILES run, line by line
# --------------------------------------------------------------------------------------------

#: Named at best-effort, declined at every tier, and (WIDE) named only at the wider tiers.
PARITY_LINES = [NAMED, NO_NAME, WIDE, "*CC"]


@pytest.mark.parametrize("mode", ["empty", "label", "raise"])
def test_batch_tier_output_equals_the_single_smiles_output(capsys, tmp_path, mode):
    flags = ("--emit-tier", "best-effort", "--on-failure", mode)
    status, out, err = _batch(capsys, tmp_path, PARITY_LINES, *flags)
    assert len(out) == len(PARITY_LINES)

    expected_rows, expected_reasons, failed = [], [], 0
    for smiles in PARITY_LINES:
        single_status, single_out, single_err = _single(capsys, smiles, *flags)
        reasons = _reasons(single_err)
        if single_status == 0:
            expected_rows.append(f"{smiles}\t{single_out.rstrip(chr(10))}")
            expected_reasons += reasons
        else:
            # --on-failure raise: one SMILES stops with the limit; its batch line says ERROR
            assert (mode, single_out, len(reasons)) == ("raise", "", 1)
            code, label = reasons[0].split("\t", 1)[1].split(": ", 1)
            expected_rows.append(f"{smiles}\tERROR: [{code}] {label}")
            failed += 1
    assert out == expected_rows
    assert _reasons(err) == expected_reasons
    assert status == (1 if failed else 0)
    assert failed == (2 if mode == "raise" else 0)


def test_batch_tier_on_failure_modes_for_the_declined_line(capsys, tmp_path):
    """The decline of the structure above, spelled out for each mode (the parity test compares
    the batch with the single run; this pins what both say)."""
    flags = ("--emit-tier", "valid")
    status, out, err = _batch(capsys, tmp_path, [NO_NAME], *flags, "--on-failure", "empty")
    assert (status, out) == (0, [f"{NO_NAME}\t"])
    assert _reasons(err) == [f"{NO_NAME}\tUNSUPPORTED_ELEMENT: inorganic compound (not supported)"]

    status, out, err = _batch(capsys, tmp_path, [NO_NAME], *flags, "--on-failure", "label")
    assert (status, _reasons(err)) == (0, [])
    assert out[0].startswith(f"{NO_NAME}\t") and out[0] != f"{NO_NAME}\t"

    status, out, err = _batch(capsys, tmp_path, [NO_NAME], *flags, "--on-failure", "raise")
    assert status == 1
    assert out == [f"{NO_NAME}\tERROR: [UNSUPPORTED_ELEMENT] inorganic compound (not supported)"]


# --------------------------------------------------------------------------------------------
# the batch contract
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("flags", [
    ("--emit-tier", "best-effort"),
    ("--provenance",),
    ("--emit-tier", "valid", "--provenance"),
])
def test_blank_lines_keep_their_place_with_a_tier(capsys, tmp_path, flags):
    lines = ["", NAMED, "   ", "", NO_NAME, ""]
    status, out, _ = _batch(capsys, tmp_path, lines, *flags)
    assert status == 0
    assert len(out) == len(lines)
    assert [bool(line) for line in out] == [bool(line.strip()) for line in lines]
    assert out[1].startswith(f"{NAMED}\t") and out[4].startswith(f"{NO_NAME}\t")


def test_an_empty_file_gives_an_empty_output_with_a_tier(capsys, tmp_path):
    for flags in (("--emit-tier", "best-effort"), ("--provenance",)):
        src = tmp_path / "empty.smi"
        src.write_text("")
        assert cli.main(["--batch", str(src), *flags]) == 0
        assert capsys.readouterr().out == ""


def test_an_unreadable_line_keeps_its_place_with_a_tier(capsys, tmp_path):
    for flags in (("--emit-tier", "best-effort"), ("--provenance",)):
        status, out, err = _batch(capsys, tmp_path, [NAMED, "not-a-smiles", NAMED], *flags)
        assert status == 1
        assert len(out) == 3
        assert out[1].startswith("not-a-smiles\tERROR: ")
        assert out[0].startswith(f"{NAMED}\t") and out[2].startswith(f"{NAMED}\t")
        assert "INVALID_SMILES" in err


def test_output_goes_to_the_file_with_a_tier(capsys, tmp_path):
    src = tmp_path / "in.smi"
    src.write_text(f"{NAMED}\n{WIDE}\n")
    dest = tmp_path / "names.txt"
    assert cli.main(["--batch", str(src), "--output", str(dest), "--emit-tier", "best-effort"]) == 0
    assert dest.read_text() == f"{NAMED}\tethanol\n{WIDE}\t{WIDE_NAME}\n"
    assert capsys.readouterr().out == ""


def test_verbose_counts_the_lines_without_a_name_for_a_tier_row(capsys, tmp_path):
    lines = [NAMED, NO_NAME, WIDE]
    # best-effort names WIDE, so one line is unnamed...
    _, _, err = _batch(capsys, tmp_path, lines, "--emit-tier", "best-effort", "--verbose")
    assert "Processed 3 SMILES, 0 errors, 1 unnamed" in err
    _, _, err = _batch(capsys, tmp_path, lines, "--emit-tier", "best-effort", "--provenance", "--verbose")
    assert "Processed 3 SMILES, 0 errors, 1 unnamed" in err
    #... the PIN-only tier declines it too...
    _, _, err = _batch(capsys, tmp_path, lines, "--emit-tier", "pin", "--provenance", "--verbose")
    assert "Processed 3 SMILES, 0 errors, 2 unnamed" in err
    #... and the default (the best-effort tier) names it, as best-effort does
    _, _, err = _batch(capsys, tmp_path, lines, "--provenance", "--verbose")
    assert "Processed 3 SMILES, 0 errors, 1 unnamed" in err
    _, _, err = _batch(capsys, tmp_path, lines, "--emit-tier", "best-effort", "--on-failure", "label",
                       "--verbose")
    assert "Processed 3 SMILES, 0 errors, 1 unnamed" in err


# --------------------------------------------------------------------------------------------
# --confidence has no tier row
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("flags", [
    ("--provenance",),
    ("--emit-tier", "best-effort", "--provenance"),
])
def test_confidence_with_a_tier_row_is_a_usage_error_for_one_smiles_and_for_a_batch(
        capsys, tmp_path, flags):
    src = tmp_path / "in.smi"
    src.write_text(f"{NAMED}\n")
    for argv in ([NAMED, "--confidence", *flags], ["--batch", str(src), "--confidence", *flags]):
        with pytest.raises(SystemExit) as stop:
            cli.main(argv)
        assert stop.value.code == 2
        captured = capsys.readouterr()
        assert "--confidence cannot be combined with --provenance" in captured.err
        assert captured.out == ""


def test_confidence_works_with_the_default_and_with_a_tier(capsys, tmp_path):
    """Lane DF: ``--confidence`` takes the default (the best-effort tier) and any tier named with
    ``--emit-tier``; only ``--provenance`` (a tier row has no coverage score) is a usage error."""
    status, out, _ = _batch(capsys, tmp_path, [NAMED], "--confidence")
    assert status == 0 and out[0].split("\t")[:2] == [NAMED, "ethanol"] and len(out[0].split("\t")) == 4
    status, out, _ = _batch(capsys, tmp_path, [NAMED], "--confidence", "--emit-tier", "valid")
    assert status == 0 and out[0].split("\t")[:2] == [NAMED, "ethanol"]
    assert cli.main([NAMED, "--confidence", "--emit-tier", "valid"]) == 0
    assert capsys.readouterr().out.startswith("Name:       ethanol")


def test_confidence_stays_for_the_pin_only_tier(capsys, tmp_path):
    status, out, _ = _batch(capsys, tmp_path, [NAMED], "--confidence", "--emit-tier", "pin")
    assert status == 0
    assert out[0].split("\t")[:2] == [NAMED, "ethanol"] and len(out[0].split("\t")) == 4
    assert cli.main([NAMED, "--confidence", "--emit-tier", "pin"]) == 0
    assert capsys.readouterr().out.startswith("Name:       ethanol")


def test_confidence_with_binding_proof_is_kept_for_one_smiles_as_for_a_batch(capsys, tmp_path):
    """``--binding-proof`` alone selects the tier branch for one SMILES, which has no coverage
    score: with ``--confidence`` it stays on ``name_compound`` (as ``--batch`` does), so the
    confidence is printed and the mode reaches the namer."""
    seen = []
    real = cli.name_compound

    def _spy(smiles, **kw):
        seen.append((kw.get("include_confidence"), kw.get("binding_proof")))
        return real(smiles, **kw)

    with mock.patch.object(cli, "name_compound", _spy):
        assert cli.main([NAMED, "--confidence", "--binding-proof", "audit"]) == 0
        out = capsys.readouterr().out
        assert out.splitlines()[0] == "Name:       ethanol"
        assert any(line.startswith("Confidence: ") for line in out.splitlines())
        status, rows, _ = _batch(capsys, tmp_path, [NAMED], "--confidence", "--binding-proof", "audit")
    assert seen == [(True, "audit"), (True, "audit")]
    assert status == 0 and rows[0].split("\t")[:2] == [NAMED, "ethanol"]
    # without --confidence the mode still selects the tier branch (the proof is not dropped)
    with mock.patch.object(cli, "name_compound", side_effect=AssertionError("name_compound used")):
        assert cli.main([NAMED, "--binding-proof", "audit"]) == 0
    assert capsys.readouterr().out == "ethanol\n"


def test_process_batch_refuses_confidence_with_a_tier_row_up_front(tmp_path):
    src = tmp_path / "in.smi"
    src.write_text(f"{NAMED}\n")
    with pytest.raises(ValueError, match="--confidence cannot be combined"):
        cli._process_batch(str(src), None, "pin", False, True, provenance=True)


# --------------------------------------------------------------------------------------------
# --enable-triviality-controller reaches a batch
# --------------------------------------------------------------------------------------------


def test_batch_threads_the_triviality_controller_flag(capsys, tmp_path):
    seen = []

    def _fake(smiles, **kw):
        seen.append(kw.get("enable_triviality_controller"))
        return "x"

    src = tmp_path / "in.smi"
    src.write_text(f"{NAMED}\n{NAMED}\n")
    with mock.patch.object(cli, "name_compound", _fake):
        assert cli.main(["--batch", str(src), "--emit-tier", "pin", "--enable-triviality-controller"]) == 0
        assert cli.main(["--batch", str(src), "--emit-tier", "pin"]) == 0
    assert seen == [True, True, False, False]


def test_batch_tier_namer_gets_the_triviality_controller_and_trivial_flags(capsys, tmp_path):
    built = []
    real = cli._tiered_namer

    def _spy(*args, **kwargs):
        built.append(real(*args, **kwargs))
        return built[-1]

    with mock.patch.object(cli, "_tiered_namer", _spy):
        _batch(capsys, tmp_path, [NAMED], "--emit-tier", "valid", "--enable-triviality-controller")
        _batch(capsys, tmp_path, [NAMED], "--provenance", "--trivial")
        _batch(capsys, tmp_path, [NAMED], "--provenance", "--style", "general", "--binding-proof", "audit")
    with_controller, with_trivial, plain = built
    assert with_controller._enable_triviality_controller is True
    assert with_controller._trivial_fallback is False
    assert with_trivial._trivial_fallback is True
    assert (plain._enable_triviality_controller, plain.style, plain._binding_proof) == (
        False, "general", "audit")
