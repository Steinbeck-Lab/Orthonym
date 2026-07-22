"""v27 Phase 6 T6.4 — surface stereo_unexpressed as METADATA (never in the name).

The flag lives in provenance + the name_tiered row dict only; the emitted name
string stays a clean, OPSIN-parseable constitutional IUPAC name (an appended
"(stereo unexpressed)" token would not be a valid name and would break OPSIN).
"""
import json

import pytest

from orthonym.namer import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.metrics import provenance as pv


def _mk(tier, *, disable_gate=True):
    f = _emit_tier_flags(tier)
    return Orthonym(
        general_fallback=f["general_fallback"],
        general_fallback_unverified=f["general_fallback_unverified"],
        allow_aromatic_general=f["allow_aromatic_general"],
        _disable_opsin_validity_gate=disable_gate,
        _disable_grammar_validation=True,
    )


def test_provenance_roundtrip_and_reset():
    pv.clear_provenance()
    assert pv.get_provenance()["stereo_unexpressed"] is False
    pv.record_stereo_unexpressed(True)
    assert pv.get_provenance()["stereo_unexpressed"] is True
    pv.clear_provenance()
    assert pv.get_provenance()["stereo_unexpressed"] is False


def test_name_tiered_has_flag_field_false_for_pin():
    row = _mk("pin").name_tiered("CCO")
    assert row["name"] == "ethanol"
    assert row["stereo_unexpressed"] is False
    # JSON-serializable (the --provenance contract)
    assert json.loads(json.dumps(row))["stereo_unexpressed"] is False


def test_name_tiered_surfaces_flag_for_shipped_general_engine(monkeypatch):
    """When a general-engine name ships with the flag set, name_tiered reports
    it; the bare name string carries no stereo/parenthetical marker."""
    be = _mk("best-effort")

    def fake_name(smi):
        pv.record_source("general_engine")
        pv.record_stereo_unexpressed(True)
        return "pentan-2-ol"

    monkeypatch.setattr(be, "name", fake_name)
    row = be.name_tiered("C[C@H](O)CCC")
    assert row["source"] == "general_engine"
    assert row["stereo_unexpressed"] is True
    assert "stereo" not in row["name"].lower()
    assert "(" not in row["name"]  # clean constitutional name
    json.loads(json.dumps(row))  # serializes cleanly


def test_name_tiered_flag_not_surfaced_on_suppressed_emission(monkeypatch):
    """If the contextvar was set but the emission was suppressed to a failure
    (downstream constitutional gate rejected it), the row must NOT claim the
    flag — it only rides a name that actually shipped."""
    be = _mk("best-effort")

    def fake_name_fail(smi):
        pv.record_source("general_engine")
        pv.record_stereo_unexpressed(True)
        return "unknown organic compound"  # failure sentinel

    monkeypatch.setattr(be, "name", fake_name_fail)
    row = be.name_tiered("C[C@H](O)CCC")
    assert row["tier"] == "T5"
    assert row["stereo_unexpressed"] is False
