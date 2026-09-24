"""Item B — tests for the repo's own probe, `scripts/diagnose.py`.

`diagnose.py` is the harness the project's briefs mandate for invariant-11
before/after tables. It had two defects that made it unable to prove what it
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
    """Import `scripts/diagnose.py` as a module (it is a script, not a package)."""
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
# Defect 2, CORRECTED (-FINAL Item 2) — `-r` must be ON by default,
# because the SHIPPED validity oracle passes it unconditionally.
#
# The earlier direction of this test (`-r` off by default) was wrong twice over:
#
# * it made the shared probe DISAGREE with production. The oracle behind the
# validity gate spawns OPSIN with `-r` at every entry point --
# `assembly/retained_substitution.py:138` (`["java","-jar",jar,"-r","-osmi"]`),
# `:176` (`get_persistent_opsin(jar, ("-r","-osmi"))`), `:197`, and
# `validation/opsin_server.py:41`/`:140` (default `args=("-r","-osmi")`).
# So a name production ACCEPTS was reported `OPSIN-UNPARSEABLE` by the probe:
# 13 of the 15 gold rows whose `expected_pin` is a bare substituent string
# flipped to a false failure.
# * its stated justification -- that `-r` lets a fragment-only emission score a
# clean `OK` -- cannot happen. `diagnose`'s verdict compares OPSIN's output
# against the canonical form of the INPUT MOLECULE
# (`_canon(opsin_smi) == r["canonical"]`), and a bare substituent name parses
# under `-r` to a strict SUB-structure, which can never compare equal. The
# only input for which `ethyl` compares equal is the ethyl radical itself --
# where the name is right. So `-r` costs no strictness at all and buys
# agreement with the gate.
# ---------------------------------------------------------------------------

def test_opsin_batch_defaults_to_radicals_like_the_shipped_oracle(diag, monkeypatch):
    """The default OPSIN invocation must contain `-r`, as production's does."""
    seen = {}

    class _P:
        stdout = "CCO\n"

    def _fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return _P()

    monkeypatch.setattr(diag.subprocess, "run", _fake_run)
    diag.opsin_batch(["ethanol"])
    assert "-r" in seen["cmd"], (
        "the shipped oracle (retained_substitution.py:138/176/197, "
        "opsin_server.py:41) always passes -r; a probe that does not will "
        "report OPSIN-UNPARSEABLE for names production accepts"
    )


def test_opsin_batch_can_opt_out_of_radicals(diag, monkeypatch):
    """`--no-radicals` is the opt-in strict mode."""
    seen = {}

    class _P:
        stdout = "C\n"

    def _fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return _P()

    monkeypatch.setattr(diag.subprocess, "run", _fake_run)
    diag.opsin_batch(["methyl"], allow_radicals=False)
    assert "-r" not in seen["cmd"]


def test_the_probes_default_opsin_args_match_the_shipped_oracles(diag, monkeypatch):
    """Tool-vs-gate agreement, pinned WITHOUT needing a JVM.

    This is the invariant that was violated: the probe and the production oracle
    must invoke OPSIN with the same radical policy. Comparing the two default
    argument lists states that directly, and cannot deadlock on an OPSIN pipe.
    """
    from orthonym.validation.opsin_server import get_persistent_opsin  # noqa: F401
    import inspect

    from orthonym.validation import opsin_server

    oracle_default = inspect.signature(
        opsin_server.PersistentOpsin.__init__).parameters["args"].default
    assert "-r" in oracle_default, (
        "guard for this test itself: if the shipped oracle ever drops -r, this "
        "test must be revisited rather than silently passing"
    )

    seen = {}

    class _P:
        stdout = "CCO\n"

    monkeypatch.setattr(diag.subprocess, "run",
                        lambda cmd, **kw: (seen.__setitem__("cmd", cmd), _P())[1])
    diag.opsin_batch(["ethanol"])
    assert ("-r" in seen["cmd"]) == ("-r" in oracle_default), (
        f"probe args {seen['cmd']} disagree with oracle default {oracle_default}"
    )


def test_a_gold_radical_pin_is_not_reported_unparseable(diag):
    """The 15-row regression, end to end: a gold `expected_pin` that IS a bare
    substituent string must not be reported `OPSIN-UNPARSEABLE` by the probe.

    Skipped (never silently passed) when Java or the jar is absent, because a
    green result with no OPSIN would be the perfect-harness trap.
    """
    import shutil

    from tests.support.jars import jar_or_skip
    jar_or_skip()  # jar absent -> skip (fail under ORTHONYM_REQUIRE_JARS=1)
    if shutil.which("java") is None:
        pytest.skip("java absent — refusing to report a false pass")
    if diag._gate_running():
        pytest.skip("gate running — never a second OPSIN job")

    # `pentan-3-yl` is a verbatim gold `expected_pin` (benchmarks/the gold set/
    # gold_pins.json) whose SMILES is the radical `CC[CH]CC`.
    got = diag.opsin_batch(["pentan-3-yl"])
    assert got and got[0], (
        "OPSIN returned nothing for the gold PIN 'pentan-3-yl'; with -r "
        "restored it must parse, exactly as the shipped oracle parses it"
    )
    # control: a genuine non-name must still fail, so the assertion above is
    # not passing for a trivial reason.
    assert diag.opsin_batch(["zzz-not-a-name"])[0] is None


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

def test_a_skipped_round_trip_is_never_reported_as_clean(diag, capsys):
    """-FINAL (review claim 41).

    With OPSIN skipped -- e.g. on a false-positive `_gate_running` hit -- every
    row still counted as `clean` and the tool still exited 0, so the summary read
    `-- 1/1 clean` with ZERO round-trips performed. A reader takes that line as
    round-trip evidence. It must say what it actually measured.
    """
    rc = diag.main(["--no-opsin", "CCO"])
    err = capsys.readouterr().err
    assert rc == 0
    assert "clean" not in err, (
        f"a run with no round-trip must not use the word 'clean': {err!r}")
    assert "OPSIN SKIPPED" in err and "no round-trip" in err


def test_spelling_independence_separates_stability_from_refusal(diag, monkeypatch):
    """-FINAL (review claim 40).

    A molecule refused identically for every spelling is 'stable' in the trivial
    sense and carries no information about naming stability. On an 80-row slice,
    40 of the 78 stable rows were of that kind, and the pass rate did not say so.
    """
    monkeypatch.setattr(diag, "respell", lambda s, n=4, seed=0: [s, s])
    monkeypatch.setattr(
        diag, "diagnose",
        lambda spellings, **kw: [{"name": "unknown organic compound"}
                                 for _ in spellings])
    res = diag.spelling_independence(["CCO"])
    assert res[0]["stable"] is True
    assert res[0]["stable_but_refused"] is True, (
        "a row stable only because every spelling was refused must be flagged")

    monkeypatch.setattr(
        diag, "diagnose",
        lambda spellings, **kw: [{"name": "ethanol"} for _ in spellings])
    res2 = diag.spelling_independence(["CCO"])
    assert res2[0]["stable"] is True
    assert res2[0]["stable_but_refused"] is False


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
    for flag in ("--canonical", "--no-radicals", "--allow-radicals",
                 "--spelling-independence"):
        assert flag in help_text, f"{flag} missing from --help"


def test_help_no_longer_asserts_the_false_false_pass_claim(diag):
    """The `--help`/docstring assertion that `-r` produced a false `OK` was
    unsound, and a false in-code assertion is the class this range set out to
    delete. Pin its removal so it cannot creep back.
    """
    doc = " ".join((diag.__doc__ or "").split())
    assert "scored a clean ``OK``" not in doc
    assert "cannot happen" in doc, (
        "the docstring must record WHY -r is safe, not merely drop the claim"
    )
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with pytest.raises(SystemExit):
            diag.main(["--help"])
    help_text = " ".join(buf.getvalue().split())
    assert "scores a clean RT" not in help_text
