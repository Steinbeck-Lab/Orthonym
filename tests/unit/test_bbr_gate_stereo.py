"""a phase BBR-GATE — RED gold-target tripwire  + gate-policy tests.

internal notes ///. The OPSIN-parse validity gate currently
suppresses IUPAC-correct stereo names that OPSIN's generation-side grammar
cannot parse — a correctness inversion (audit Dim-08 §C). The verbatim Blue Book
PIN ``(1s,4s)-cyclohexane-1,4-diol`` (lines 48111-48113) is produced correctly by
the formatter but the gate turns it into ``unknown`` (GATE_SUPPRESSED).

This RED tripwire is the executable form of the gold row. It is
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

Gold row: O[C@H]1CC[C@@H](O)CC1 -> (1s,4s)-cyclohexane-1,4-diol /
"""

import pytest

from orthonym import Orthonym
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "O[C@H]1CC[C@@H](O)CC1",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset({
    "O[C@H]1CC[C@@H](O)CC1",
})


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)



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
    # Isolate the parseability + stereo-carve-out POLICY under test from the
    # constitutional layer (a separate gate, exercised by test_self_consistency_gate.py).
    monkeypatch.setattr(namer, "_SC_MODE", "off", raising=False)
    monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: True, raising=False)
    monkeypatch.setattr(namer, "_validity_gate_status", _simulated_opsin_status, raising=False)
    # reorder: the gate consults name_to_smiles first. Mirror the simulated
    # verdict — None when OPSIN rejects (the stereo-block names), a SMILES otherwise.
    monkeypatch.setattr(
        namer, "_validity_gate_name_to_smiles",
        lambda n: None if _simulated_opsin_status(n) == "rejected" else "CCO", raising=False)
    yield


@pytest.mark.unit
def test_cyclohexanediol_not_gate_suppressed(gate_enabled_real_policy):
    #: the raw name is already correct; with the gate ENABLED and OPSIN
    # rejecting the stereo block but PARSING the stereo-stripped form, Plan 04's
    # where-it-fails logic ships the full name. FIXED — permanent green tripwire.
    # The gate's output is the strict path's name; with switched off by this
    # fixture no constitution-only verdict is recorded, so the default tier's emission
    # rule declines the name here (NO_VERIFIED_PIN). With the real gate it records
    # 'self_consistency_constitution_only' and the default tier emits the name (one
    # of its exceptions; tests/integration/test_readme_claims.py).
    from tests.support.default_tier import strict_path_name
    assert strict_path_name("O[C@H]1CC[C@@H](O)CC1") == "(1s,4s)-cyclohexane-1,4-diol"
    assert (Orthonym().name_tiered("O[C@H]1CC[C@@H](O)CC1")["limit_code"]
            == "NO_VERIFIED_PIN")


@pytest.mark.unit
@pytest.mark.parametrize("name,expected", [
    ("(1s,4s)-cyclohexane-1,4-diol", "cyclohexane-1,4-diol"),
    ("(2R,3S)-butane-2,3-diol", "butane-2,3-diol"),
    ("(E)-but-2-enoic acid", "but-2-enoic acid"),
    ("hexane", "hexane"),                                       # no over-stripping
    ("1-(2-chloroethyl)-4-methylbenzene", "1-(2-chloroethyl)-4-methylbenzene"),  # substituent group kept
])
def test_strip_stereo(name, expected):
    from orthonym.rules.stereochemistry import strip_stereo
    assert strip_stereo(name) == expected


@pytest.mark.unit
def test_constitutional_fail_still_suppressed(monkeypatch):
    """A name whose stereo-STRIPPED form ALSO fails to parse (a genuine constitutional
    defect, no rescuing stereo block) is STILL suppressed — the gate stays strict."""
    import orthonym.namer as namer
    monkeypatch.setattr(namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: True, raising=False)
    # A malformed constitutional name with NO leading stereo block: strip_stereo is a
    # no-op, so the where-it-fails carve-out cannot rescue it -> suppressed.
    monkeypatch.setattr(namer, "_validity_gate_name_to_smiles", lambda n: None, raising=False)  # OPSIN rejects
    monkeypatch.setattr(namer, "_validity_gate_status", lambda n: "rejected", raising=False)
    out = namer._final_opsin_validity_gate("2-methylhept9ol", "CCCCCCCC")
    assert out != "2-methylhept9ol"  # suppressed to the descriptive fallback


@pytest.mark.unit
def test_fail_open_when_jar_absent(monkeypatch):
    """OPSIN-unavailable (no JAR) -> fail-OPEN: the name ships unchanged (preserved)."""
    import orthonym.namer as namer
    monkeypatch.setattr(namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: False, raising=False)
    out = namer._final_opsin_validity_gate("anything-at-all", "CCO")
    assert out == "anything-at-all"
