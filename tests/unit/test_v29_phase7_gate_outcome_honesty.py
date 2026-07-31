"""v29 Phase 7 Task 1: ``gates_passed`` must report what the GATE DID.

Before this task ``name_tiered`` computed

    gate_active = (not self._disable_opsin_validity_gate
                   and _validity_gate_jar_present())
    opsin = "verified" if gate_active else "unverified"
    if opsin == "verified": gates.append("SELF-01")

``gate_active`` means *"the JAR exists"*. It is not evidence that **this** name
passed anything, so every PIN-path emission on a machine with a jar was stamped
``SELF-01`` — including the ten ``return name`` carve-outs in
``_final_opsin_validity_gate``, which return **before** ``_self_consistency_decision``
is ever called, and the BBR-GATE stereo carve-out, which ships the full name
after SELF-01 judged only the stereo-STRIPPED parse.

Measured at ``a6cf6157`` (`` §2):
three names OPSIN cannot parse at all reported ``gates_passed: ['SELF-01']``.

These tests pin the honest report. **No emitted name changes** — only the label.
"""
from __future__ import annotations

import pytest

from orthonym.namer import Orthonym
from orthonym.metrics import provenance as pv


# The three T1 leaks from FINDINGS.md §2, with the exit each one takes in
# `_final_opsin_validity_gate` (established by the Task 1 spy: a sys.settrace
# line tracer scoped to the function's code object).
#   CC=C.C=C.[Ti+2]        -> exit line 1030, _ORGANOMETALLIC_ADDITIVE_PIN_RE
#   C[C@@H]1C[C@H]1CO      -> exit line 1110, BBR-GATE stereo carve-out
#   C/C(=C\\C1C=CC=C1)/C(=O)O -> exit line 1110, BBR-GATE stereo carve-out
T1_LEAKS = {
    "CC=C.C=C.[Ti+2]": "carveout:organometallic_additive",
    "C[C@@H]1C[C@H]1CO": "self01_verified_constitution_only",
    r"C/C(=C\C1C=CC=C1)/C(=O)O": "self01_verified_constitution_only",
}


# --------------------------------------------------------------------------
# 1. a name that genuinely round-trips still reports SELF-01
# --------------------------------------------------------------------------

@pytest.mark.opsin_gate
def test_genuinely_verified_name_still_reports_self01():
    """`CCO` -> `ethanol` exits the gate through `_self_consistency_decision`
    with verdict "ok". That is the ONE state that may claim SELF-01."""
    row = Orthonym().name_tiered("CCO")
    assert row["name"] == "ethanol"
    assert row["gate_outcome"] == pv.GATE_OUTCOME_SELF01
    assert row["opsin"] == "verified"
    assert "SELF-01" in row["gates_passed"]


# --------------------------------------------------------------------------
# 2. the three measured T1 leaks must NOT claim a bare SELF-01
# --------------------------------------------------------------------------

@pytest.mark.opsin_gate
def test_t1_leaks_do_not_claim_bare_self01():
    """Each measured leak reports the specific outcome its exit path took, and
    none of them contributes a bare ``SELF-01`` token."""
    # Guard against the Phase 6 vacuous-test defect: an empty iterable makes
    # the loop below pass without testing anything.
    assert len(T1_LEAKS) == 3, "the measured T1 leak set must not be empty"
    checked = 0
    for smiles, expected_outcome in T1_LEAKS.items():
        row = Orthonym().name_tiered(smiles)
        assert row["name"], f"{smiles} must still EMIT (Task 1 changes no name)"
        assert row["gate_outcome"] == expected_outcome, (
            f"{smiles}: expected gate outcome {expected_outcome!r}, "
            f"got {row['gate_outcome']!r} (name={row['name']!r})")
        assert "SELF-01" not in row["gates_passed"], (
            f"{smiles} claims a bare SELF-01 it did not earn: "
            f"{row['gates_passed']}")
        checked += 1
    assert checked == 3


@pytest.mark.opsin_gate
def test_stereo_carveout_is_labelled_constitution_only():
    """The BBR-GATE carve-out checked the CONSTITUTION on the stereo-stripped
    parse and never checked the stereo layer. It says so, and it says so with a
    token distinct from a full SELF-01."""
    row = Orthonym().name_tiered("C[C@@H]1C[C@H]1CO")
    assert row["gate_outcome"] == pv.GATE_OUTCOME_SELF01_CONSTITUTION_ONLY
    assert row["opsin"] == "verified_constitution_only"
    assert "SELF-01(constitution)" in row["gates_passed"]
    assert "SELF-01" not in row["gates_passed"]


# --------------------------------------------------------------------------
# 3. a carve-out reports its slug and no bare SELF-01
# --------------------------------------------------------------------------

@pytest.mark.opsin_gate
def test_carveout_reports_its_slug():
    """`CC=C.C=C.[Ti+2]` hits the P-69.2.3 organometallic-additive carve-out
    (`_ORGANOMETALLIC_ADDITIVE_PIN_RE`), which returns the name BEFORE
    `_self_consistency_decision` is called. The name still ships — only the
    label changes."""
    row = Orthonym().name_tiered("CC=C.C=C.[Ti+2]")
    assert row["name"] == "(η²-ethene)(η³-prop-2-en-1-yl)titanium"
    assert row["gate_outcome"] == "carveout:organometallic_additive"
    assert row["gate_outcome"].startswith("carveout:")
    assert row["opsin"] == "unverified"
    assert row["gates_passed"] == [] or "SELF-01" not in row["gates_passed"]


# --------------------------------------------------------------------------
# 4. gate disabled -> gate_disabled / unverified, never verified
# --------------------------------------------------------------------------

def test_gate_disabled_never_reports_verified():
    row = Orthonym(_disable_opsin_validity_gate=True).name_tiered("CCO")
    assert row["name"] == "ethanol"
    assert row["gate_outcome"] == pv.GATE_OUTCOME_DISABLED
    assert row["opsin"] == "unverified"
    assert "SELF-01" not in row["gates_passed"]


# --------------------------------------------------------------------------
# 5. the ContextVar is reset per molecule (a stale outcome must not leak)
# --------------------------------------------------------------------------

@pytest.mark.opsin_gate
def test_outcome_does_not_leak_between_consecutive_molecules():
    """Two molecules named in sequence, in BOTH orders. Each reports its own
    outcome — a contextvar left set by the previous molecule would mislabel the
    next one."""
    nm = Orthonym()

    first = nm.name_tiered("CC=C.C=C.[Ti+2]")          # carve-out
    second = nm.name_tiered("CCO")                     # genuinely verified
    assert first["gate_outcome"] == "carveout:organometallic_additive"
    assert second["gate_outcome"] == pv.GATE_OUTCOME_SELF01
    assert "SELF-01" in second["gates_passed"]

    third = nm.name_tiered("CCO")
    fourth = nm.name_tiered("CC=C.C=C.[Ti+2]")
    assert third["gate_outcome"] == pv.GATE_OUTCOME_SELF01
    assert fourth["gate_outcome"] == "carveout:organometallic_additive"
    assert "SELF-01" not in fourth["gates_passed"]


def test_clear_provenance_resets_the_gate_outcome():
    """Unit-level twin of the above, independent of OPSIN."""
    pv.record_gate_outcome("carveout:phane", "some-phane-name")
    assert pv.get_provenance()["gate_outcome"] == "carveout:phane"
    pv.clear_provenance()
    assert pv.get_provenance()["gate_outcome"] == pv.GATE_OUTCOME_NOT_RUN
    assert pv.get_provenance()["gate_outcome_name"] is None


# --------------------------------------------------------------------------
# 6. deny-by-default: an outcome recorded for a DIFFERENT string never counts
# --------------------------------------------------------------------------

def test_outcome_recorded_for_another_candidate_does_not_verify():
    """`_apply_trivial_fallback` ships a retained name the gate never saw (it
    only fires once the gate has already suppressed the systematic candidate).
    The recorded outcome then belongs to a DIFFERENT string, so it must not be
    used. Pinned on the pure resolver so the rule holds for every such path."""
    assert pv.resolve_gate_outcome(
        pv.GATE_OUTCOME_SELF01, "ethanol", "ethanol") == pv.GATE_OUTCOME_SELF01
    assert pv.resolve_gate_outcome(
        pv.GATE_OUTCOME_SELF01, "ethanol", "caffeine") == pv.GATE_OUTCOME_BYPASSED
    assert pv.resolve_gate_outcome(
        pv.GATE_OUTCOME_NOT_RUN, None, "caffeine") == pv.GATE_OUTCOME_NOT_RUN


def test_only_two_outcomes_may_claim_verification():
    """Fail closed on unknown: anything not explicitly a verified outcome —
    including an outcome string this code has never seen — reports
    ``unverified``."""
    assert pv.opsin_label_for_gate_outcome(pv.GATE_OUTCOME_SELF01) == "verified"
    assert pv.opsin_label_for_gate_outcome(
        pv.GATE_OUTCOME_SELF01_CONSTITUTION_ONLY) == "verified_constitution_only"
    for outcome in (pv.GATE_OUTCOME_NOT_RUN, pv.GATE_OUTCOME_BYPASSED,
                    pv.GATE_OUTCOME_DISABLED, pv.GATE_OUTCOME_UNAVAILABLE,
                    pv.GATE_OUTCOME_SUPPRESSED, pv.GATE_OUTCOME_SELF01_INCONCLUSIVE,
                    pv.GATE_OUTCOME_SELF01_SKIPPED,
                    pv.GATE_OUTCOME_SELF01_WARN_MISMATCH,
                    "carveout:inositol", "carveout:phane",
                    "a-state-that-does-not-exist-yet", None):
        assert pv.opsin_label_for_gate_outcome(outcome) == "unverified", outcome
