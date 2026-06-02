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
  - ``_validity_gate_parse(name) -> Optional[str]``   (cached parse-or-None)
  - ``_validity_gate_jar_present() -> bool``          (JAR presence probe)
  - ``_final_opsin_validity_gate(name, smiles) -> str``
"""
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
        """A name OPSIN can parse is returned verbatim."""
        monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: True)
        monkeypatch.setattr(namer, "_validity_gate_parse", lambda n: "CCO")
        assert _final_opsin_validity_gate("ethanol", "CCO") == "ethanol"

    def test_malformed_name_falls_back_to_descriptive_string(self, monkeypatch):
        """An unparseable name -> the _descriptive_fallback STRING (not None)."""
        monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: True)
        monkeypatch.setattr(namer, "_validity_gate_parse", lambda n: None)
        smiles = "CC1(C)[C@@H]2CC[C@@]1(C)C(=O)C2"  # camphor
        out = _final_opsin_validity_gate("4,7,7-trimethylanediol", smiles)
        assert out == _descriptive_fallback(smiles)
        assert isinstance(out, str) and out  # never None / never empty

    def test_no_jar_is_fail_open_noop(self, monkeypatch):
        """CRITICAL (D-13): JAR absent -> gate is a no-op, name unchanged."""
        monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: False)
        # parse would return None, but the JAR guard must short-circuit FIRST.
        monkeypatch.setattr(namer, "_validity_gate_parse", lambda n: None)
        assert _final_opsin_validity_gate("anything-at-all", "CCO") == "anything-at-all"

    def test_empty_name_is_untouched(self, monkeypatch):
        """A falsy name is returned unchanged (nothing to gate)."""
        monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: True)
        monkeypatch.setattr(namer, "_validity_gate_parse", lambda n: None)
        assert _final_opsin_validity_gate("", "CCO") == ""
