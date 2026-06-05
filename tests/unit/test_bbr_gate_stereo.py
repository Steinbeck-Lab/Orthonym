"""Phase 169.7 BBR-GATE — RED gold-target tripwire (DEF-9) + gate-policy tests.

CONTEXT D-15/D-05/D-06/D-07. The SUB-03 OPSIN-parse validity gate currently
suppresses IUPAC-correct stereo names that OPSIN's generation-side grammar
cannot parse — a correctness inversion (audit Dim-08 §C). The verbatim Blue Book
PIN ``(1s,4s)-cyclohexane-1,4-diol`` (lines 48111-48113) is produced correctly by
the formatter but the gate turns it into ``unknown`` (GATE_SUPPRESSED).

This RED tripwire is the executable form of the DEF-9 gold row. It is
``xfail(strict=True)`` so the moment Plan 04 makes the gate stereo-aware it
XPASSES → hard error → the implementing plan removes the marker. Plan 04 ADDS the
``strip_stereo`` + constitutional-still-suppressed + radical-carve-out assertions
to this file.

The conftest autouse fixture ``_disable_opsin_validity_gate_for_tests`` disables
the gate for the rest of the suite; this gate tripwire RE-ENABLES it and
simulates OPSIN's parse outcome (the gate's own tests do the same — they
monkeypatch ``_validity_gate_status``). The simulation mirrors reality exactly:
OPSIN REJECTS the lowercase-r/s stereo form ``(1s,4s)-cyclohexane-1,4-diol`` but
PARSES the stereo-stripped constitutional form ``cyclohexane-1,4-diol`` (verified
against the real OPSIN oracle in audit Dim-08 §C). Plan 04's fix is precisely:
"rejected, but stereo-stripped form parses → ship the full name".

NO band-aids: the SHIPPED name keeps its stereo; the fix is a gate POLICY change
(decide on WHERE OPSIN fails), not string post-processing.

Gold row: DEF-9  O[C@H]1CC[C@@H](O)CC1 -> (1s,4s)-cyclohexane-1,4-diol  (P-93.5 / P-91)
"""

import pytest

from orthonym import Orthonym


def _simulated_opsin_status(name: str) -> str:
    """Mirror real OPSIN's verdict (audit Dim-08 §C): it REJECTS a name carrying
    a leading lowercase-r/s stereo descriptor block but PARSES the
    stereo-stripped constitutional form. A name beginning with a ``(...)-``
    descriptor block is 'rejected'; otherwise 'parsed'."""
    n = name.strip()
    if n.startswith("(") and ")-" in n[: n.index(")-") + 2]:
        return "rejected"
    return "parsed"


@pytest.fixture
def gate_enabled_real_policy(monkeypatch):
    """Re-enable the production validity gate (conftest disables it suite-wide)
    and force the JAR-present probe True + a deterministic OPSIN status that
    matches the real oracle for the DEF-9 case. This makes the gate's POLICY the
    thing under test — exactly what Plan 04 changes."""
    import orthonym.namer as namer

    monkeypatch.setattr(namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: True, raising=False)
    monkeypatch.setattr(namer, "_validity_gate_status", _simulated_opsin_status, raising=False)
    yield


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason="169.7 BBR-GATE Plan-04: the SUB-03 validity gate suppresses the "
    "correct-by-construction (1s,4s)-cyclohexane-1,4-diol because OPSIN rejects "
    "the lowercase-r/s stereo block (the stereo-stripped form parses). Flips to "
    "PASS when the gate decides on WHERE OPSIN fails; then remove this marker.",
)
def test_cyclohexanediol_not_gate_suppressed(gate_enabled_real_policy):
    # DEF-9: the raw name is already correct; with the gate ENABLED and OPSIN
    # rejecting the stereo block, today it is suppressed to the descriptive
    # fallback. Plan 04 ships it because the stereo-stripped form parses.
    assert Orthonym().name("O[C@H]1CC[C@@H](O)CC1") == "(1s,4s)-cyclohexane-1,4-diol"
