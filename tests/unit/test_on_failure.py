"""``on_failure``: what a public naming call returns for a structure that gets no name.

GitHub issue 45: the labels ('unknown organic compound', 'cobalt compound (not supported)')
are warnings, not IUPAC names. The default is now the empty string, and the reason goes to
the log (``orthonym`` logger, one warning per unnamed structure). ``"label"`` keeps the old
result and ``"raise"`` is ``raise_on_limit=True``.

The limit codes (``errors.LIMIT_CATALOG`` keys) are the stable contract; the texts of the
labels are not, so these tests read the code from the error and the label from the same
error, never from a literal, except where a test is about the label itself.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orthonym import Orthonym, name_compound
from orthonym.errors import LIMIT_CATALOG, OrthonymLimitError, is_failure_name

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

ROOT = Path(__file__).resolve().parents[2]

# (SMILES, the limit code of a structure that gets no name)
UNNAMED = [
    ("C[C@H]1CC[C@H](NC(=O)c2ccccc2)CC1", "UNNAMEABLE"),
    ("O=[U](=O)=O", "UNSUPPORTED_ELEMENT"),
    ("*CC", "WILDCARD_ATOMS"),
]
NAMED = "CCO"


def _label_of(smiles: str) -> str:
    """The label of a structure with no name, from the error the ``"raise"`` mode gives."""
    with pytest.raises(OrthonymLimitError) as err:
        name_compound(smiles, on_failure="raise", emit_tier="pin")
    return err.value.message


# ---------------------------------------------------------------------------
# the three modes
# ---------------------------------------------------------------------------


def test_a_name_is_the_same_in_every_mode():
    for mode in ("empty", "label", "raise"):
        assert name_compound(NAMED, on_failure=mode) == "ethanol"
    assert name_compound(NAMED) == "ethanol"
    assert Orthonym().name(NAMED) == "ethanol"


@pytest.mark.parametrize("smiles,code", UNNAMED)
def test_the_default_is_the_empty_string(smiles, code):
    assert name_compound(smiles, emit_tier="pin") == ""
    assert name_compound(smiles, on_failure="empty", emit_tier="pin") == ""
    assert Orthonym().name(smiles) == ""
    assert is_failure_name(name_compound(smiles, emit_tier="pin"))


@pytest.mark.parametrize("smiles,code", UNNAMED)
def test_label_mode_gives_the_label_of_the_limit(smiles, code):
    label = name_compound(smiles, on_failure="label", emit_tier="pin")
    assert label and is_failure_name(label)
    assert label == _label_of(smiles)
    assert Orthonym().name(smiles, on_failure="label") == label


@pytest.mark.parametrize("smiles,code", UNNAMED)
def test_raise_mode_raises_the_limit(smiles, code):
    assert code in LIMIT_CATALOG
    with pytest.raises(OrthonymLimitError) as err:
        name_compound(smiles, on_failure="raise")
    assert err.value.code == code
    with pytest.raises(OrthonymLimitError) as err:
        Orthonym().name(smiles, on_failure="raise")
    assert err.value.code == code


@pytest.mark.parametrize("smiles,code", UNNAMED)
def test_raise_on_limit_is_the_alias_of_raise_and_wins(smiles, code):
    with pytest.raises(OrthonymLimitError) as err:
        name_compound(smiles, raise_on_limit=True)
    assert err.value.code == code
    with pytest.raises(OrthonymLimitError) as err:
        name_compound(smiles, raise_on_limit=True, on_failure="empty")
    assert err.value.code == code
    with pytest.raises(OrthonymLimitError):
        Orthonym().name(smiles, raise_on_limit=True, on_failure="label")


@pytest.mark.parametrize("bad", ["", "none", "Empty", None, True])
def test_an_unknown_value_raises_value_error(bad):
    with pytest.raises(ValueError, match="on_failure"):
        name_compound(NAMED, on_failure=bad)
    with pytest.raises(ValueError, match="on_failure"):
        Orthonym().name(NAMED, on_failure=bad)


def test_a_mol_is_named_like_its_smiles():
    from rdkit import Chem
    mol = Chem.MolFromSmiles("O=[U](=O)=O")
    assert name_compound(mol) == ""
    assert name_compound(mol, on_failure="label") == _label_of("O=[U](=O)=O")
    assert Orthonym().name(mol) == ""


# ---------------------------------------------------------------------------
# the warning
# ---------------------------------------------------------------------------


def _unnamed_records(caplog):
    return [r for r in caplog.records if r.name == "orthonym"]


@pytest.mark.parametrize("smiles,code", UNNAMED)
def test_one_warning_with_the_code_and_the_label(caplog, smiles, code):
    label = _label_of(smiles)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="orthonym"):
        assert name_compound(smiles, emit_tier="pin") == ""
    records = _unnamed_records(caplog)
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.WARNING
    assert (record.orthonym_smiles, record.orthonym_code, record.orthonym_label) == (
        smiles, code, label)
    message = record.getMessage()
    assert smiles in message and f"{code}: {label}" in message


def test_the_instance_call_warns_once_too(caplog):
    with caplog.at_level(logging.WARNING, logger="orthonym"):
        assert Orthonym().name("O=[U](=O)=O") == ""
    assert [r.orthonym_code for r in _unnamed_records(caplog)] == ["UNSUPPORTED_ELEMENT"]


@pytest.mark.parametrize("mode", ["label", "raise"])
def test_label_and_raise_log_nothing(caplog, mode):
    with caplog.at_level(logging.WARNING, logger="orthonym"):
        try:
            name_compound("O=[U](=O)=O", on_failure=mode)
        except OrthonymLimitError:
            pass
    assert not _unnamed_records(caplog)


def test_a_name_logs_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger="orthonym"):
        assert name_compound(NAMED) == "ethanol"
    assert not _unnamed_records(caplog)


def test_the_confidence_dict_follows_the_mode(caplog):
    with caplog.at_level(logging.WARNING, logger="orthonym"):
        record = name_compound("O=[U](=O)=O", include_confidence=True)
    assert record["name"] == ""
    assert record["limit"]["code"] == "UNSUPPORTED_ELEMENT"     # the shape is unchanged
    assert {"confidence", "handler", "factors"} <= set(record)
    assert [r.orthonym_code for r in _unnamed_records(caplog)] == ["UNSUPPORTED_ELEMENT"]
    labelled = name_compound("O=[U](=O)=O", include_confidence=True, on_failure="label")
    assert labelled["name"] == _label_of("O=[U](=O)=O")
    with pytest.raises(OrthonymLimitError) as err:
        name_compound("O=[U](=O)=O", include_confidence=True, on_failure="raise")
    assert err.value.code == "UNSUPPORTED_ELEMENT"
    assert name_compound(NAMED, include_confidence=True)["name"] == "ethanol"


def test_the_tiered_row_is_unchanged():
    row = Orthonym().name_tiered("O=[U](=O)=O")
    assert row["tier"] == "abstain" and row["limit_code"] == "UNSUPPORTED_ELEMENT"
    assert row["name"] == _label_of("O=[U](=O)=O")


# a structure that the default tier does not name but a wider tier does (pin_unverified)
WIDER_TIER_ONLY = "CC1=CC[C@@](C)([C@@]2(C)CC(=O)C=C2CO)CC1"


def test_a_wider_tier_name_logs_no_unnamed_warning(caplog):
    """``name_tiered`` at a wider tier builds the strict twin's name to tell the tiers
    apart. That is an internal call that reads the twin's result as a string, so it gets the
    label and never logs a 'no name' warning for a molecule that the row names."""
    from orthonym.cli import _emit_tier_flags
    with caplog.at_level(logging.WARNING, logger="orthonym"):
        row = Orthonym(**_emit_tier_flags("best-effort")).name_tiered(WIDER_TIER_ONLY)
    assert row["name"] and not is_failure_name(row["name"])
    assert not _unnamed_records(caplog)


# ---------------------------------------------------------------------------
# a nested call keeps the label
# ---------------------------------------------------------------------------


def test_a_call_inside_a_naming_gets_the_label():
    """The engine names parts of a molecule through the public entries and reads what they
    return as a string; whatever the argument says, such a call gets the label."""
    from orthonym.assembly.fragment_naming import enter_name_scope, exit_name_scope
    label = _label_of("O=[U](=O)=O")
    enter_name_scope()
    try:
        assert name_compound("O=[U](=O)=O") == label
        assert name_compound("O=[U](=O)=O", on_failure="empty") == label
        assert Orthonym().name("O=[U](=O)=O") == label
        assert name_compound("O=[U](=O)=O", include_confidence=True)["name"] == label
        with pytest.raises(OrthonymLimitError):                  # "raise" still raises
            name_compound("O=[U](=O)=O", on_failure="raise")
    finally:
        exit_name_scope()


def test_the_engines_own_nested_calls_get_labels(monkeypatch, caplog):
    """A molecule that the engine names through nested public calls, some of which have no
    name: every nested call still returns its label (none returns ''), the molecule itself
    gets '' for the caller, and only the caller's call is logged."""
    import orthonym
    import orthonym.namer as namer_module
    from orthonym.assembly.fragment_naming import name_scope_depth

    nested = []
    real_nc, real_name = namer_module.name_compound, namer_module.Orthonym.name

    def spy_nc(smiles, *args, **kwargs):
        inside = name_scope_depth() > 0
        result = real_nc(smiles, *args, **kwargs)
        if inside:
            nested.append(result)
        return result

    def spy_name(self, smiles, *args, **kwargs):
        inside = name_scope_depth() > 0
        result = real_name(self, smiles, *args, **kwargs)
        if inside:
            nested.append(result)
        return result

    monkeypatch.setattr(namer_module, "name_compound", spy_nc)
    monkeypatch.setattr(orthonym, "name_compound", spy_nc)
    monkeypatch.setattr(namer_module.Orthonym, "name", spy_name)
    # a glycoside: its aglycone and sugar parts are named through nested calls
    smiles = "N#C[C@]1(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)C=C[C@H](O)C1"
    with caplog.at_level(logging.WARNING, logger="orthonym"):
        assert namer_module.name_compound(smiles, emit_tier="pin") == ""
    assert nested, "the molecule is expected to be named through nested calls"
    assert any(is_failure_name(r) for r in nested), "a nested call with no name is the point"
    assert "" not in nested
    assert len(_unnamed_records(caplog)) == 1


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------


def _cli(*args, input_lines=None, tmp_path=None):
    """Run ``python -m orthonym``: (status, stdout, stderr)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")])
    args = list(args)
    if input_lines is not None:
        batch = tmp_path / "in.smi"
        batch.write_text("".join(line + "\n" for line in input_lines))
        args = ["--batch", str(batch), *args]
    proc = subprocess.run([sys.executable, "-m", "orthonym", *args], capture_output=True,
                          text=True, env=env, cwd=ROOT, timeout=600)
    return proc.returncode, proc.stdout, proc.stderr


def _reasons(stderr: str) -> list:
    """The '<SMILES><tab><CODE>: <label>' lines of the error stream (the library's other
    warnings carry no tab)."""
    return [ln for ln in stderr.splitlines() if "\t" in ln]


def test_cli_single_default_prints_an_empty_line_and_the_reason_on_stderr():
    code, out, err = _cli("O=[U](=O)=O")
    assert code == 0
    assert out == "\n"
    assert _reasons(err) == [f"O=[U](=O)=O\tUNSUPPORTED_ELEMENT: {_label_of('O=[U](=O)=O')}"]


def test_cli_single_label_and_raise():
    label = _label_of("*CC")
    code, out, err = _cli("*CC", "--on-failure", "label")
    assert (code, out, _reasons(err)) == (0, label + "\n", [])
    code, out, err = _cli("*CC", "--on-failure", "raise")
    assert (code, out) == (1, "")
    assert _reasons(err) == [f"*CC\tWILDCARD_ATOMS: {label}"]


def test_cli_a_name_is_unchanged():
    code, out, err = _cli(NAMED)
    assert (code, out, _reasons(err)) == (0, "ethanol\n", [])


def test_cli_confidence_prints_an_empty_name():
    code, out, err = _cli("*CC", "--confidence")
    assert code == 0 and out.splitlines()[0].rstrip() == "Name:"
    assert [ln.split("\t")[0] for ln in _reasons(err)] == ["*CC"]


def test_cli_batch_keeps_one_line_per_input_line(tmp_path):
    lines = [NAMED, "O=[U](=O)=O", "", "*CC", "c1ccccc1"]
    code, out, err = _cli(input_lines=lines, tmp_path=tmp_path)
    assert code == 0
    assert out.split("\n")[:-1] == [f"{NAMED}\tethanol", "O=[U](=O)=O\t", "", "*CC\t",
                                    "c1ccccc1\tbenzene"]
    reasons = _reasons(err)
    assert sorted(ln.split("\t")[0] for ln in reasons) == ["*CC", "O=[U](=O)=O"]
    assert all(": " in ln.split("\t")[1] for ln in reasons)


def test_cli_batch_confidence_columns_stay_aligned(tmp_path):
    code, out, err = _cli("--confidence", input_lines=[NAMED, "O=[U](=O)=O"], tmp_path=tmp_path)
    rows = [row.split("\t") for row in out.split("\n")[:-1]]
    assert code == 0 and len(rows) == 2 and all(len(row) == 4 for row in rows)
    assert rows[0][:2] == [NAMED, "ethanol"]
    assert rows[1][:2] == ["O=[U](=O)=O", ""]
    assert len(_reasons(err)) == 1


def test_cli_batch_label_mode_prints_the_label_and_no_reason(tmp_path):
    label = _label_of("O=[U](=O)=O")
    code, out, err = _cli("--on-failure", "label", input_lines=[NAMED, "O=[U](=O)=O"],
                          tmp_path=tmp_path)
    assert code == 0 and _reasons(err) == []
    assert out.split("\n")[:-1] == [f"{NAMED}\tethanol", f"O=[U](=O)=O\t{label}"]


def test_cli_batch_verbose_counts_the_unnamed_lines(tmp_path):
    code, out, err = _cli("--verbose", input_lines=[NAMED, "O=[U](=O)=O", "*CC"],
                          tmp_path=tmp_path)
    assert code == 0
    assert "Processed 3 SMILES, 0 errors, 2 unnamed" in err


def test_cli_batch_raise_mode_marks_the_line_and_fails(tmp_path):
    code, out, err = _cli("--on-failure", "raise", input_lines=[NAMED, "O=[U](=O)=O"],
                          tmp_path=tmp_path)
    rows = out.split("\n")[:-1]
    assert code == 1
    assert rows[0] == f"{NAMED}\tethanol"
    assert rows[1].startswith("O=[U](=O)=O\tERROR: [UNSUPPORTED_ELEMENT] ")


def test_cli_rejects_an_unknown_value():
    code, out, err = _cli("CCO", "--on-failure", "none")
    assert code == 2 and out == ""
