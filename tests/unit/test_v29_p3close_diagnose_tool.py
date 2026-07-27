"""Item B — tests for the repo's own probe, .

`diagnose.py` is the harness the project's briefs mandate for invariant-11
before/after tables.  It had two defects that made it unable to prove what it
was being used to prove:

1. it named ``Chem.CanonSmiles(input)`` rather than the SMILES as written, so
   **no** table it produced could demonstrate spelling independence — every row
   was naming the identical canonical string; and
2. it passed OPSIN ``-r`` (``--allowRadicals``), so a name that describes only a
   FRAGMENT of the input molecule could score a clean ``OK``.

Both are fixed; these tests pin the fixes, because a shared measurement tool
with no tests is how the defects survived in the first place.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_DIAG = _ROOT / "scripts" / "diagnose.py"


def _load():
    """Import  as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("_diagnose_under_test", _DIAG)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_diagnose_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def diag():
    return _load()


# ---------------------------------------------------------------------------
# Defect 1 — the default must name the SMILES AS WRITTEN.
# ---------------------------------------------------------------------------

def test_default_names_the_smiles_as_written(diag):
    """`OCC` must be handed to the namer as `OCC`, not as `CCO`.

    This is the whole point: if the tool canonicalises, a spelling-dependence
    bug is invisible to it by construction.
    """
    rows = diag.diagnose(["OCC"], use_opsin=False)
    assert rows[0]["named_input"] == "OCC"
    assert rows[0]["canonical"] == "CCO"


def test_canonical_flag_restores_the_old_behaviour(diag):
    rows = diag.diagnose(["OCC"], use_opsin=False, canonical=True)
    assert rows[0]["named_input"] == "CCO"


def test_as_written_and_canonical_differ_on_a_non_canonical_input(diag):
    """A tripwire: if these two ever agree for every input, the flag is inert."""
    as_written = diag.diagnose(["OCC"], use_opsin=False)[0]["named_input"]
    canonical = diag.diagnose(["OCC"], use_opsin=False,
                              canonical=True)[0]["named_input"]
    assert as_written != canonical


def test_bad_smiles_still_reports_bad_smiles(diag):
    rows = diag.diagnose(["this-is-not-smiles"], use_opsin=False)
    assert rows[0]["verdict"] == "BAD-SMILES"


# ---------------------------------------------------------------------------
# Defect 2 — `-r` must be OFF by default.
# ---------------------------------------------------------------------------

def test_opsin_batch_defaults_to_no_radicals(diag, monkeypatch):
    """The default OPSIN invocation must not contain `-r`."""
    seen = {}

    class _P:
        stdout = "CCO\n"

    def _fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return _P()

    monkeypatch.setattr(diag.subprocess, "run", _fake_run)
    diag.opsin_batch(["ethanol"])
    assert "-r" not in seen["cmd"], (
        "-r lets a bare substituent name parse as a radical, so a name "
        "describing only a fragment of the molecule scores a clean round-trip"
    )


def test_opsin_batch_can_opt_in_to_radicals(diag, monkeypatch):
    seen = {}

    class _P:
        stdout = "C\n"

    def _fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return _P()

    monkeypatch.setattr(diag.subprocess, "run", _fake_run)
    diag.opsin_batch(["methyl"], allow_radicals=True)
    assert "-r" in seen["cmd"]


def test_opsin_batch_skips_empty_names_without_spawning_a_jvm(diag, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not spawn a JVM when there is nothing to parse")

    monkeypatch.setattr(diag.subprocess, "run", _boom)
    assert diag.opsin_batch([None, None]) == [None, None]


# ---------------------------------------------------------------------------
# The new `--spelling-independence` mode.
# ---------------------------------------------------------------------------

def test_respell_returns_distinct_strings_for_one_molecule(diag):
    from rdkit import Chem

    spellings = diag.respell("CC(C)CSSSCC(C)CC", n=4, seed=0)
    assert len(spellings) >= 3, "need several spellings for the probe to mean anything"
    assert len(set(spellings)) == len(spellings), "spellings must be distinct"
    ref = Chem.CanonSmiles(spellings[0])
    for s in spellings:
        assert Chem.CanonSmiles(s) == ref, f"{s} is a different molecule"


def test_respell_is_deterministic_for_a_fixed_seed(diag):
    assert diag.respell("CCCC(C)CC", n=4, seed=0) == \
        diag.respell("CCCC(C)CC", n=4, seed=0)


def test_respell_keeps_the_input_spelling_first(diag):
    assert diag.respell("OCC", n=2, seed=0)[0] == "OCC"


def test_respell_includes_the_rdkit_canonical_spelling(diag):
    """The brief requires as-written AND RDKit-canonical AND a re-written form.

    The canonical form is the one an older report would have used, so it must be
    in the comparison set or the probe cannot reproduce that report's row.
    """
    from rdkit import Chem

    spellings = diag.respell("OCC", n=2, seed=0)
    assert "OCC" in spellings                     # as-written
    assert Chem.CanonSmiles("OCC") in spellings   # RDKit-canonical
    assert len(spellings) >= 3                    # at least one re-written form


def test_respell_survives_an_unparseable_input(diag):
    assert diag.respell("not-smiles") == ["not-smiles"]


def test_spelling_independence_reports_stable_for_a_stable_molecule(diag):
    res = diag.spelling_independence(["CCO"], n=3, seed=0)
    assert res[0]["stable"] is True
    assert res[0]["distinct_names"] == ["ethanol"]


def test_spelling_independence_flags_a_divergent_producer(diag, monkeypatch):
    """The mode must actually be able to FAIL — a probe that always says
    STABLE is a mute button, not a check.

    A stub namer that returns a different name per spelling must be reported
    spelling-dependent.
    """
    counter = {"n": 0}

    def _fake_diagnose(smiles_list, **kw):
        out = []
        for s in smiles_list:
            counter["n"] += 1
            out.append({"smiles": s, "name": f"name-{counter['n']}",
                        "verdict": None})
        return out

    monkeypatch.setattr(diag, "diagnose", _fake_diagnose)
    res = diag.spelling_independence(["CCO"], n=3, seed=0)
    assert res[0]["stable"] is False
    assert len(res[0]["distinct_names"]) > 1


def test_spelling_independence_treats_a_naming_failure_as_unstable(diag, monkeypatch):
    def _fake_diagnose(smiles_list, **kw):
        return [{"smiles": s, "name": None, "verdict": "NAMING-FAILED"}
                for s in smiles_list]

    monkeypatch.setattr(diag, "diagnose", _fake_diagnose)
    res = diag.spelling_independence(["CCO"], n=2, seed=0)
    assert res[0]["stable"] is False


# ---------------------------------------------------------------------------
# The documentation obligation: the reasoning must live in --help, not only in
# a report that nobody reads next session.
# ---------------------------------------------------------------------------

def test_help_documents_both_fixes(diag):
    # Collapse whitespace: the prose is hard-wrapped, so "spelling\n
    # independence" must still count as documenting spelling independence.
    doc = " ".join((diag.__doc__ or "").split())
    assert "AS WRITTEN" in doc
    assert "allowRadicals" in doc or "-r" in doc
    assert "spelling independence" in doc.lower()


def test_cli_exposes_the_new_flags(diag):
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with pytest.raises(SystemExit):
            diag.main(["--help"])
    help_text = buf.getvalue()
    for flag in ("--canonical", "--allow-radicals", "--spelling-independence"):
        assert flag in help_text, f"{flag} missing from --help"
