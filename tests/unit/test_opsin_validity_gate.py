""" — pre-emission OPSIN-parse validity gate (a phase, Wave 0 scaffold).

RED scaffold per VALIDATION.md "Wave 0 Requirements": the gate function
``orthonym.namer._final_opsin_validity_gate`` does not exist until Plan 06
 lands. Until then this whole module SKIPS via the import guard, so the
``-m "not slow"`` suite stays green. Plan 06 flips these to hard assertions.

The gate contract (internal notes //):
  - a production name that OPSIN parses passes through UNCHANGED;
  - a name OPSIN cannot parse is suppressed -> the EXISTING
    ``_descriptive_fallback(smiles)`` STRING (never None / never a shipped
    invalid IUPAC string);
  - CRITICAL fail-OPEN: when the OPSIN JAR is absent the gate is a NO-OP
    (returns the name unchanged) — the OPPOSITE of OpsinOracle.rt_safe's
    deliberate fail-CLOSED — else a missing-Java env would suppress EVERY name.
  - jar PRESENT but OPSIN 'unavailable' (a timeout / crash on a loaded host,
    after the oracle's retry ladder): fail CLOSED (TRIAGE g7 C01, 2026-09-27;
    it used to fail open under and shipped unverified, sometimes wrong,
    names).

Plan 06 will provide these seams on ``orthonym.namer`` (monkeypatched here):
  - ``_validity_gate_status(name) -> str`` (cached 3-valued parse
    outcome: 'parsed' | 'rejected' | 'unavailable';)
  - ``_validity_gate_jar_present -> bool`` (JAR presence probe)
  - ``_final_opsin_validity_gate(name, smiles) -> str``
"""
import subprocess

import pytest

# Import guard: skip the whole module until (Plan 06) lands the gate.
namer = pytest.importorskip("orthonym.namer")
if not hasattr(namer, "_final_opsin_validity_gate"):
    pytest.skip(
        "SUB-03 validity gate not implemented yet (lands in Plan 06)",
        allow_module_level=True,
    )

from orthonym.namer import _final_opsin_validity_gate, _descriptive_fallback  # noqa: E402


@pytest.mark.unit
class TestOpsinValidityGate:
    """ gate: suppress-malformed / fail-OPEN / parseable-untouched."""

    @pytest.fixture(autouse=True)
    def _force_enable_gate(self, monkeypatch):
        """Re-enable the gate for these tests (the conftest autouse fixture
        disables it for the rest of the suite per)."""
        monkeypatch.setattr(namer, "_DISABLE_VALIDITY_GATE", False)

    def test_parseable_name_passes_through_unchanged(self, monkeypatch):
        """A name OPSIN can parse ('parsed') and that round-trips to the SAME molecule
        is returned verbatim. (reorder: the gate consults name_to_smiles first.)"""
        monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: True)
        monkeypatch.setattr(namer, "_validity_gate_name_to_smiles", lambda n: "CCO")
        monkeypatch.setattr(namer, "_validity_gate_status", lambda n: "parsed")
        assert _final_opsin_validity_gate("ethanol", "CCO") == "ethanol"

    def test_malformed_name_falls_back_to_descriptive_string(self, monkeypatch):
        """A DEFINITIVELY rejected name -> the _descriptive_fallback STRING (not None)."""
        monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: True)
        monkeypatch.setattr(namer, "_validity_gate_name_to_smiles", lambda n: None)  # OPSIN rejects
        monkeypatch.setattr(namer, "_validity_gate_status", lambda n: "rejected")
        smiles = "CC1(C)[C@@H]2CC[C@@]1(C)C(=O)C2"  # camphor
        out = _final_opsin_validity_gate("4,7,7-trimethylanediol", smiles)
        assert out == _descriptive_fallback(smiles)
        assert isinstance(out, str) and out  # never None / never empty

    def test_no_jar_is_fail_open_noop(self, monkeypatch):
        """CRITICAL : JAR absent -> gate is a no-op, name unchanged."""
        monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: False)
        # status would say 'rejected', but the JAR guard must short-circuit FIRST.
        monkeypatch.setattr(namer, "_validity_gate_status", lambda n: "rejected")
        assert _final_opsin_validity_gate("anything-at-all", "CCO") == "anything-at-all"

    def test_unavailable_status_fails_closed(self, monkeypatch):
        """TRIAGE g7 C01 (2026-09-27): OPSIN 'unavailable' with the JAR present (a
        timeout / OSError / JVM crash mid-run that the oracle's retry ladder could
        not clear) verifies NOTHING, so the gate suppresses to the descriptive
        fallback. It used to fail OPEN  and ship the candidate unverified:
        measured in a fresh process with both probes stubbed like this, the PIN
        tier shipped 'benzene' for c1ccc2nc3ccc4ncccc4c3nc2c1. A name that is
        wrong must not ship because OPSIN was slow; a name that is right is kept
        by the retry ladder (TestOpsinOracleRetryLadder), not by failing open.
        'ethanol' is a CORRECT name here on purpose: the gate cannot know that
        without OPSIN, and 0-wrong is absolute -- every emission verified, else
        abstain."""
        monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: True)
        # name_to_smiles returns None on a transient failure too; the status probe
        # then distinguishes 'unavailable' from 'rejected' (reorder).
        monkeypatch.setattr(namer, "_validity_gate_name_to_smiles", lambda n: None)
        monkeypatch.setattr(namer, "_validity_gate_status", lambda n: "unavailable")
        stats = {}
        assert _final_opsin_validity_gate("ethanol", "CCO", stats) == \
            _descriptive_fallback("CCO")
        assert stats == {"opsin_unavailable_suppressed": 1}

    def test_empty_name_is_untouched(self, monkeypatch):
        """A falsy name is returned unchanged (nothing to gate)."""
        monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: True)
        monkeypatch.setattr(namer, "_validity_gate_status", lambda n: "rejected")
        assert _final_opsin_validity_gate("", "CCO") == ""


@pytest.mark.unit
class TestOpsinOracleParseStatus:
    """ gate oracle: the 3-valued parse outcome + no-poison caching.

    These exercise OpsinOracle directly (no Java needed — the subprocess call is
    stubbed), pinning the / contract at its source.
    """

    def _oracle(self):
        from orthonym.assembly.retained_substitution import OpsinOracle
        return OpsinOracle(opsin_jar="/fake/opsin.jar")  # non-None jar so we reach the call

    def test_parsed_when_opsin_emits_smiles(self):
        oracle = self._oracle()
        oracle._invoke_opsin = lambda name: ("CCO", True)
        assert oracle.parse_status("ethanol") == "parsed"

    def test_rejected_when_opsin_emits_nothing(self):
        oracle = self._oracle()
        oracle._invoke_opsin = lambda name: (None, True)
        assert oracle.parse_status("4,7,7-trimethylanediol") == "rejected"

    def test_no_jar_is_unavailable(self):
        from orthonym.assembly.retained_substitution import OpsinOracle
        assert OpsinOracle(opsin_jar=None).parse_status("ethanol") == "unavailable"

    def test_transient_failure_is_unavailable_and_not_cached(self, monkeypatch):
        """: a timeout on EVERY rung of the one-shot retry ladder returns
        'unavailable' and is NOT cached, so a later call (after the transient
        condition clears) re-invokes and can succeed -- a timeout must not
        permanently suppress a name process-wide. (Since TRIAGE g7 C01 a single
        timeout is retried inside the call; see TestOpsinOracleRetryLadder.)"""
        from orthonym.assembly import retained_substitution as rs
        monkeypatch.setattr(rs, "_ONE_SHOT_LADDER", ((0.0, 10.0), (0.0, 60.0)))
        oracle = self._oracle()
        calls = {"n": 0}

        class _Res:
            def __init__(self, out):
                self.stdout = out
                self.returncode = 0  # real CompletedProcess always has one

        def fake_run(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 2:  # both rungs of the first call time out
                raise subprocess.TimeoutExpired(cmd="java", timeout=kwargs["timeout"])
            return _Res("CCO\n")

        monkeypatch.setattr(rs.subprocess, "run", fake_run)
        assert oracle.parse_status("ethanol") == "unavailable"  # transient
        assert "ethanol" not in oracle._parse_status_cache
        # NOT cached -> the second call re-invokes OPSIN and now succeeds.
        assert oracle.parse_status("ethanol") == "parsed"
        assert calls["n"] == 3

    def test_definitive_outcome_is_cached(self, monkeypatch):
        """A definitive 'parsed'/'rejected' IS cached (one subprocess per name)."""
        from orthonym.assembly import retained_substitution as rs
        oracle = self._oracle()
        calls = {"n": 0}

        class _Res:
            def __init__(self, out):
                self.stdout = out
                self.returncode = 0  # real CompletedProcess always has one

        def fake_run(*args, **kwargs):
            calls["n"] += 1
            return _Res("CCO\n")

        monkeypatch.setattr(rs.subprocess, "run", fake_run)
        assert oracle.parse_status("ethanol") == "parsed"
        assert oracle.parse_status("ethanol") == "parsed"
        assert calls["n"] == 1  # cached -> only ONE subprocess call


@pytest.mark.unit
class TestOpsinOracleRetryLadder:
    """TRIAGE g7 C01 (2026-09-27): the one-shot tier retries a transient failure
    with a longer budget before it reports 'unavailable', because the validity
    gate now fails CLOSED on 'unavailable' -- a slow OPSIN on a loaded host must
    not cost a correct name (the worry had), and must not ship an
    unverified one either. The subprocess is stubbed (no Java needed)."""

    def _oracle(self):
        from orthonym.assembly.retained_substitution import OpsinOracle
        return OpsinOracle(opsin_jar="/fake/opsin.jar")

    @staticmethod
    def _fast_ladder(monkeypatch):
        from orthonym.assembly import retained_substitution as rs
        monkeypatch.setattr(rs, "_ONE_SHOT_LADDER", ((0.0, 10.0), (0.0, 60.0)))
        return rs

    def test_one_timeout_then_an_answer_is_a_verdict(self, monkeypatch):
        rs = self._fast_ladder(monkeypatch)
        seen = []

        class _Res:
            returncode = 0
            stdout = "CCO\n"

        def fake_run(*args, **kwargs):
            seen.append(kwargs["timeout"])
            if len(seen) == 1:
                raise subprocess.TimeoutExpired(cmd="java", timeout=kwargs["timeout"])
            return _Res()

        monkeypatch.setattr(rs.subprocess, "run", fake_run)
        oracle = self._oracle()
        assert oracle.parse_status("ethanol") == "parsed"
        assert seen == [10.0, 60.0]  # the retry got the longer budget
        assert oracle._parse_status_cache["ethanol"] == "parsed"

    def test_crash_then_an_answer_is_a_verdict(self, monkeypatch):
        rs = self._fast_ladder(monkeypatch)
        outs = iter([(1, ""), (0, "")])  # JVM crash, then a clean rejection

        class _Res:
            def __init__(self, rc, out):
                self.returncode, self.stdout = rc, out

        monkeypatch.setattr(rs.subprocess, "run",
                            lambda *a, **k: _Res(*next(outs)))
        assert self._oracle().parse_status("not-a-name") == "rejected"

    def test_no_java_executable_is_not_retried(self, monkeypatch):
        rs = self._fast_ladder(monkeypatch)
        calls = {"n": 0}

        def fake_run(*args, **kwargs):
            calls["n"] += 1
            raise FileNotFoundError("java")

        monkeypatch.setattr(rs.subprocess, "run", fake_run)
        assert self._oracle().parse_status("ethanol") == "unavailable"
        assert calls["n"] == 1

    def test_cooldown_after_a_failed_ladder_tries_the_first_rung_only(self, monkeypatch):
        rs = self._fast_ladder(monkeypatch)
        seen = []

        def fake_run(*args, **kwargs):
            seen.append(kwargs["timeout"])
            raise subprocess.TimeoutExpired(cmd="java", timeout=kwargs["timeout"])

        monkeypatch.setattr(rs.subprocess, "run", fake_run)
        oracle = self._oracle()
        assert oracle.parse_status("a") == "unavailable"
        assert seen == [10.0, 60.0]
        assert oracle.parse_status("b") == "unavailable"  # inside the cool-down
        assert seen == [10.0, 60.0, 10.0]
        oracle._one_shot_down_until = 0.0                 # cool-down over
        assert oracle.parse_status("c") == "unavailable"
        assert seen == [10.0, 60.0, 10.0, 10.0, 60.0]

    def test_rt_safe_does_not_cache_a_transient_failure(self, monkeypatch):
        rs = self._fast_ladder(monkeypatch)
        calls = {"n": 0}

        class _Res:
            returncode = 0
            stdout = "CCO\n"

        def fake_run(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise subprocess.TimeoutExpired(cmd="java", timeout=kwargs["timeout"])
            return _Res()

        monkeypatch.setattr(rs.subprocess, "run", fake_run)
        oracle = self._oracle()
        assert oracle.rt_safe("CCO", "ethanol") is False   # fail closed...
        assert ("ethanol", "CCO") not in oracle._cache      #... but not pinned
        oracle._one_shot_down_until = 0.0
        assert oracle.rt_safe("CCO", "ethanol") is True


@pytest.mark.unit
class TestGateUnderLoad:
    """TRIAGE g7 C01 end to end: the REAL gate seams over a REAL OpsinOracle whose
    only live tier is the one-shot subprocess (in-process JVM and persistent
    server switched off), with the subprocess stubbed to behave like OPSIN on a
    loaded host. A name must not depend on machine load: a late answer keeps the
    verified name, no answer at all abstains -- never the unverified candidate."""

    @pytest.fixture
    def loaded_host(self, monkeypatch):
        import orthonym.jvm_bridge as jb
        import orthonym.validation.opsin_server as osrv
        from orthonym.assembly import retained_substitution as rs
        monkeypatch.setattr(namer, "_DISABLE_VALIDITY_GATE", False)
        monkeypatch.setattr(namer, "_SC_MODE", "on")
        monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: True)
        monkeypatch.setattr(namer, "_VALIDITY_ORACLE",
                            rs.OpsinOracle(opsin_jar="/fake/opsin.jar"))
        monkeypatch.setattr(jb, "opsin_stdout", lambda *a, **k: (None, False))
        monkeypatch.setattr(osrv, "get_persistent_opsin", lambda *a, **k: None)
        monkeypatch.setattr(rs, "_ONE_SHOT_LADDER", ((0.0, 10.0), (0.0, 60.0)))
        calls = []

        def install(answers):
            """answers: per subprocess call, a SMILES string or None (= timeout)."""
            it = iter(answers)

            class _Res:
                returncode = 0

            def fake_run(*args, **kwargs):
                calls.append(kwargs["timeout"])
                out = next(it)
                if out is None:
                    raise subprocess.TimeoutExpired(cmd="java", timeout=kwargs["timeout"])
                res = _Res()
                res.stdout = out + "\n"
                return res

            monkeypatch.setattr(rs.subprocess, "run", fake_run)
            return calls

        return install

    def test_a_late_answer_keeps_the_verified_name(self, loaded_host):
        calls = loaded_host([None, "CCO"])  # rung 1 times out, rung 2 answers
        assert _final_opsin_validity_gate("ethanol", "CCO", {}) == "ethanol"
        assert calls == [10.0, 60.0]

    def test_no_answer_abstains_even_for_a_wrong_candidate(self, loaded_host):
        # 'benzene' for the quinoxaline-quinoline: what the fail-open shipped.
        smi = "c1ccc2nc3ccc4ncccc4c3nc2c1"
        loaded_host([None] * 4)  # name_to_smiles: 2 rungs, parse_status: 2 rungs
        stats = {}
        assert _final_opsin_validity_gate("benzene", smi, stats) == \
            _descriptive_fallback(smi)
        assert stats.get("opsin_unavailable_suppressed") == 1


@pytest.mark.unit
class TestOpsinOracleNonZeroExit:
    """-01 close finding: a NON-ZERO OPSIN exit (JVM crash/OOM under
    full-corpus load) must be classified 'unavailable' (transient: retried by the
    one-shot ladder, never cached), NOT 'rejected' -- a crash says nothing about
    the name. Completes the hardening, which previously caught only
    TimeoutExpired/OSError. (What the gate does with 'unavailable' changed in
    TRIAGE g7 C01: it fails closed; the classification here did not change.)
    """

    def _oracle(self):
        from orthonym.assembly.retained_substitution import OpsinOracle
        return OpsinOracle(opsin_jar="/nonexistent/opsin.jar")  # _jar non-None

    def test_nonzero_exit_is_unavailable_not_rejected(self, monkeypatch):
        import orthonym.assembly.retained_substitution as rs

        class _Res:
            returncode = 1          # JVM crash / OOM
            stdout = ""             # empty — looks like a "rejection" to the old code
            stderr = "Error: OutOfMemoryError"

        monkeypatch.setattr(rs.subprocess, "run", lambda *a, **k: _Res())
        oracle = self._oracle()
        # Must be 'unavailable' (transient) -- NOT 'rejected'.
        assert oracle.parse_status("any-valid-name-here") == "unavailable"
        # And it must NOT be cached (a transient failure must not poison later lookups).
        assert "any-valid-name-here" not in oracle._parse_status_cache

    def test_zero_exit_empty_stdout_is_rejected(self, monkeypatch):
        import orthonym.assembly.retained_substitution as rs

        class _Res:
            returncode = 0          # OPSIN ran cleanly
            stdout = ""             #...and definitively rejected (no SMILES)
            stderr = "name is unparsable"

        monkeypatch.setattr(rs.subprocess, "run", lambda *a, **k: _Res())
        oracle = self._oracle()
        assert oracle.parse_status("gibberish-name") == "rejected"

    def test_zero_exit_with_smiles_is_parsed(self, monkeypatch):
        import orthonym.assembly.retained_substitution as rs

        class _Res:
            returncode = 0
            stdout = "CCO\n"
            stderr = ""

        monkeypatch.setattr(rs.subprocess, "run", lambda *a, **k: _Res())
        oracle = self._oracle()
        assert oracle.parse_status("ethanol") == "parsed"
