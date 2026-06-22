"""SUB-03 — pre-emission OPSIN-parse validity gate (Phase 169.5, Wave 0 scaffold).

RED scaffold per VALIDATION.md "Wave 0 Requirements": the gate function
``orthonym.namer._final_opsin_validity_gate`` does not exist until Plan 06
(SUB-03) lands. Until then this whole module SKIPS via the import guard, so the
``-m "not slow"`` suite stays green. Plan 06 flips these to hard assertions.

The gate contract (CONTEXT D-11/D-12/D-13):
  - a production name that OPSIN parses passes through UNCHANGED;
  - a name OPSIN cannot parse is suppressed -> the EXISTING
    ``_descriptive_fallback(smiles)`` STRING (never None / never a shipped
    invalid IUPAC string);
  - CRITICAL fail-OPEN: when the OPSIN JAR is absent the gate is a NO-OP
    (returns the name unchanged) — the OPPOSITE of OpsinOracle.rt_safe's
    deliberate fail-CLOSED — else a missing-Java env would suppress EVERY name.

Plan 06 will provide these seams on ``orthonym.namer`` (monkeypatched here):
  - ``_validity_gate_status(name) -> str``            (cached 3-valued parse
    outcome: 'parsed' | 'rejected' | 'unavailable'; CR-01)
  - ``_validity_gate_jar_present() -> bool``          (JAR presence probe)
  - ``_final_opsin_validity_gate(name, smiles) -> str``
"""
import subprocess

import pytest

# Import guard: skip the whole module until SUB-03 (Plan 06) lands the gate.
namer = pytest.importorskip("orthonym.namer")
if not hasattr(namer, "_final_opsin_validity_gate"):
    pytest.skip(
        "SUB-03 validity gate not implemented yet (lands in Plan 06)",
        allow_module_level=True,
    )

from orthonym.namer import _final_opsin_validity_gate, _descriptive_fallback  # noqa: E402


@pytest.mark.unit
class TestOpsinValidityGate:
    """SUB-03 gate: suppress-malformed / fail-OPEN / parseable-untouched."""

    @pytest.fixture(autouse=True)
    def _force_enable_gate(self, monkeypatch):
        """Re-enable the gate for these tests (the conftest autouse fixture
        disables it for the rest of the suite per D-13)."""
        monkeypatch.setattr(namer, "_DISABLE_VALIDITY_GATE", False)

    def test_parseable_name_passes_through_unchanged(self, monkeypatch):
        """A name OPSIN can parse ('parsed') and that round-trips to the SAME molecule
        is returned verbatim. (SELF-01 reorder: the gate consults name_to_smiles first.)"""
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
        """CRITICAL (D-13): JAR absent -> gate is a no-op, name unchanged."""
        monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: False)
        # status would say 'rejected', but the JAR guard must short-circuit FIRST.
        monkeypatch.setattr(namer, "_validity_gate_status", lambda n: "rejected")
        assert _final_opsin_validity_gate("anything-at-all", "CCO") == "anything-at-all"

    def test_unavailable_status_is_fail_open(self, monkeypatch):
        """CR-01: a transient OPSIN failure ('unavailable' — timeout / OSError mid-run,
        JAR present) must FAIL OPEN and ship the name, NEVER suppress it. The old
        parse-or-None check suppressed here, turning an RT=1 name into RT=0."""
        monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: True)
        # name_to_smiles returns None on a transient failure too; the status probe
        # then distinguishes 'unavailable' -> fail OPEN (SELF-01 reorder).
        monkeypatch.setattr(namer, "_validity_gate_name_to_smiles", lambda n: None)
        monkeypatch.setattr(namer, "_validity_gate_status", lambda n: "unavailable")
        # A perfectly valid name that simply timed out must survive unchanged.
        assert _final_opsin_validity_gate("ethanol", "CCO") == "ethanol"

    def test_empty_name_is_untouched(self, monkeypatch):
        """A falsy name is returned unchanged (nothing to gate)."""
        monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: True)
        monkeypatch.setattr(namer, "_validity_gate_status", lambda n: "rejected")
        assert _final_opsin_validity_gate("", "CCO") == ""


@pytest.mark.unit
class TestOpsinOracleParseStatus:
    """SUB-03 gate oracle: the 3-valued parse outcome + WR-04 no-poison caching.

    These exercise OpsinOracle directly (no Java needed — the subprocess call is
    stubbed), pinning the CR-01/WR-04 contract at its source.
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
        """WR-04: a timeout returns 'unavailable' and is NOT cached, so a later
        call (after the transient condition clears) re-invokes and can succeed —
        a single timeout must not permanently suppress a name process-wide."""
        from orthonym.assembly import retained_substitution as rs
        oracle = self._oracle()
        calls = {"n": 0}

        class _Res:
            def __init__(self, out):
                self.stdout = out
                self.returncode = 0  # real CompletedProcess always has one

        def fake_run(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise subprocess.TimeoutExpired(cmd="java", timeout=10)
            return _Res("CCO\n")

        monkeypatch.setattr(rs.subprocess, "run", fake_run)
        assert oracle.parse_status("ethanol") == "unavailable"  # transient
        # NOT cached -> the second call re-invokes OPSIN and now succeeds.
        assert oracle.parse_status("ethanol") == "parsed"
        assert calls["n"] == 2

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
class TestOpsinOracleNonZeroExit:
    """ADR-21-01 close finding: a NON-ZERO OPSIN exit (JVM crash/OOM under
    full-corpus load) must be classified 'unavailable' (transient, fail-OPEN),
    NOT 'rejected' — else a valid name is suppressed to a descriptive fallback
    on a transient subprocess failure. Completes the CR-01 hardening, which
    previously caught only TimeoutExpired/OSError.
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
        # Must be 'unavailable' (transient) so the gate fails OPEN — NOT 'rejected'.
        assert oracle.parse_status("any-valid-name-here") == "unavailable"
        # And it must NOT be cached (a transient failure must not poison later lookups).
        assert "any-valid-name-here" not in oracle._parse_status_cache

    def test_zero_exit_empty_stdout_is_rejected(self, monkeypatch):
        import orthonym.assembly.retained_substitution as rs

        class _Res:
            returncode = 0          # OPSIN ran cleanly
            stdout = ""             # ...and definitively rejected (no SMILES)
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
