""" a phase Task 1: ``gates_passed`` must report what the GATE DID.

Before this task ``name_tiered`` computed

    gate_active = (not self._disable_opsin_validity_gate
                   and _validity_gate_jar_present)
    opsin = "verified" if gate_active else "unverified"
    if opsin == "verified": gates.append("")

``gate_active`` means *"the JAR exists"*. It is not evidence that **this** name
passed anything, so every PIN-path emission on a machine with a jar was stamped
```` — including the ten ``return name`` carve-outs in
``_final_opsin_validity_gate``, which return **before** ``_self_consistency_decision``
is ever called, and the BBR-GATE stereo carve-out, which ships the full name
after judged only the stereo-STRIPPED parse.

Measured at `` (`internal notes`):
three names OPSIN cannot parse at all reported ``gates_passed: ['']``.

These tests pin the honest report. **No emitted name changes** — only the label.
"""
from __future__ import annotations

import pytest

from orthonym.namer import Orthonym
from orthonym.metrics import provenance as pv


# The three leaks from FINDINGS.md, with the exit each one takes in
# `_final_opsin_validity_gate` (established by the Task 1 trace: a sys.settrace
# line tracer scoped to the function's code object).
# CC=C.C=C.[Ti+2] -> exit line 1030, _ORGANOMETALLIC_ADDITIVE_PIN_RE
# C[C@@H]1C[C@H]1CO -> exit line 1110, BBR-GATE stereo carve-out
# C/C(=C\\C1C=CC=C1)/C(=O)O -> exit line 1110, BBR-GATE stereo carve-out
T1_LEAKS = {
    "CC=C.C=C.[Ti+2]": "carveout:organometallic_additive",
    "C[C@@H]1C[C@H]1CO": "self_consistency_constitution_only",
    r"C/C(=C\C1C=CC=C1)/C(=O)O": "self_consistency_constitution_only",
}


# --------------------------------------------------------------------------
# 1. a name that genuinely round-trips still reports
# --------------------------------------------------------------------------

@pytest.mark.opsin_gate
def test_genuinely_verified_name_still_reports_self01():
    """`CCO` -> `ethanol` exits the gate through `_self_consistency_decision`
    with verdict "ok". That is the ONE state that may claim."""
    row = Orthonym().name_tiered("CCO")
    assert row["name"] == "ethanol"
    assert row["gate_outcome"] == pv.GATE_OUTCOME_SELF01
    assert row["opsin"] == "verified"
    assert "self_consistency" in row["gates_passed"]


# --------------------------------------------------------------------------
# 2. the three measured leaks must NOT claim a bare
# --------------------------------------------------------------------------

@pytest.mark.opsin_gate
def test_t1_leaks_do_not_claim_bare_self01():
    """Each measured leak reports the specific outcome its exit path took, and
    none of them contributes a bare ```` token."""
    # Guard against the a phase vacuous-test defect: an empty iterable makes
    # the loop below pass without testing anything.
    assert len(T1_LEAKS) == 3, "the measured T1 leak set must not be empty"
    checked = 0
    for smiles, expected_outcome in T1_LEAKS.items():
        row = Orthonym().name_tiered(smiles)
        assert row["name"], f"{smiles} must still EMIT (Task 1 changes no name)"
        assert row["gate_outcome"] == expected_outcome, (
            f"{smiles}: expected gate outcome {expected_outcome!r}, "
            f"got {row['gate_outcome']!r} (name={row['name']!r})")
        assert "self_consistency" not in row["gates_passed"], (
            f"{smiles} claims a bare self_consistency it did not earn: "
            f"{row['gates_passed']}")
        checked += 1
    assert checked == 3


@pytest.mark.opsin_gate
def test_stereo_carveout_is_labelled_constitution_only():
    """The BBR-GATE carve-out checked the CONSTITUTION on the stereo-stripped
    parse and never checked the stereo layer. It says so, and it says so with a
    token distinct from a full."""
    row = Orthonym().name_tiered("C[C@@H]1C[C@H]1CO")
    assert row["gate_outcome"] == pv.GATE_OUTCOME_SELF01_CONSTITUTION_ONLY
    assert row["opsin"] == "verified_constitution_only"
    assert "self_consistency_constitution_only" in row["gates_passed"]
    assert "self_consistency" not in row["gates_passed"]


# --------------------------------------------------------------------------
# 3. a carve-out reports its slug and no bare
# --------------------------------------------------------------------------

@pytest.mark.opsin_gate
def test_carveout_reports_its_slug():
    """`CC=C.C=C.[Ti+2]` hits the organometallic-additive carve-out
    (`_ORGANOMETALLIC_ADDITIVE_PIN_RE`), which returns the name BEFORE
    `_self_consistency_decision` is called. The name still ships — only the
    label changes."""
    row = Orthonym().name_tiered("CC=C.C=C.[Ti+2]")
    assert row["name"] == "(η²-ethene)(η³-prop-2-en-1-yl)titanium"
    assert row["gate_outcome"] == "carveout:organometallic_additive"
    assert row["gate_outcome"].startswith("carveout:")
    assert row["opsin"] == "unverified"
    assert row["gates_passed"] == [] or "self_consistency" not in row["gates_passed"]


# --------------------------------------------------------------------------
# 4. gate disabled -> gate_disabled / unverified, never verified
# --------------------------------------------------------------------------

def test_gate_disabled_never_reports_verified():
    row = Orthonym(_disable_opsin_validity_gate=True).name_tiered("CCO")
    assert row["name"] == "ethanol"
    assert row["gate_outcome"] == pv.GATE_OUTCOME_DISABLED
    assert row["opsin"] == "unverified"
    assert "self_consistency" not in row["gates_passed"]


# --------------------------------------------------------------------------
# 5. the ContextVar is reset per molecule (a stale outcome must not leak)
# --------------------------------------------------------------------------

# The leak-sensitive probe PAIR, measured at. Both ship the exact
# same string, and that collision is the point — see the test below.
# `[C-]#[O+]` the gate SUPPRESSES the candidate to the descriptive fallback
# and records `suppressed` FOR 'unknown organic compound'.
# the C16H8 PAH ships 'unknown organic compound' having reached NO
# gate-recording branch at all, so its honest outcome is
# `not_run`. (5 of 60 pubchem_2000 rows land here; no EMITTED
# row does — a descriptive-fallback abstain is the realistic
# shape of "the gate was never invoked".)
LEAK_PROBE_RECORDS = "[C-]#[O+]"
LEAK_PROBE_NO_GATE = "C1=CC2=CC3=CC4=CC=CC5=C4C3=C2C1=C5"


@pytest.mark.opsin_gate
def test_consecutive_molecules_each_report_their_own_gate_branch():
    """Two molecules named in sequence, in BOTH orders. Each reports its own
    outcome.

    NOTE — this test does NOT exercise `clear_provenance`'s reset: both probes
    hit a real gate-recording branch on every call, so the contextvar is
    legitimately overwritten whether or not the reset runs (confirmed by
    mutation testing). It pins that the recording sites are wired to the right
    exits. The reset is covered by
    `test_stale_outcome_cannot_leak_onto_a_molecule_the_gate_never_saw` below
    and by `test_clear_provenance_resets_the_gate_outcome`.
    """
    nm = Orthonym()

    first = nm.name_tiered("CC=C.C=C.[Ti+2]")          # carve-out
    second = nm.name_tiered("CCO")                     # genuinely verified
    assert first["gate_outcome"] == "carveout:organometallic_additive"
    assert second["gate_outcome"] == pv.GATE_OUTCOME_SELF01
    assert "self_consistency" in second["gates_passed"]

    third = nm.name_tiered("CCO")
    fourth = nm.name_tiered("CC=C.C=C.[Ti+2]")
    assert third["gate_outcome"] == pv.GATE_OUTCOME_SELF01
    assert fourth["gate_outcome"] == "carveout:organometallic_additive"
    assert "self_consistency" not in fourth["gates_passed"]


@pytest.mark.opsin_gate
def test_stale_outcome_cannot_leak_onto_a_molecule_the_gate_never_saw():
    """`name_tiered` must RESET the gate contextvar per molecule.

    `resolve_gate_outcome`'s deny-by-default name guard is not sufficient on
    its own: it only rejects a stale outcome recorded for a *different* string.
    This probe pair defeats that guard deliberately — the recorder molecule
    records `suppressed` FOR the very string the second molecule ships — so
    `clear_provenance` is the ONLY thing standing between the second row and
    the first molecule's verdict.

    With the reset, the second row honestly says the gate never ran. Without
    it, the row claims the gate examined this molecule and suppressed a
    candidate, which never happened. Verified by mutation: deleting the two
    `_GATE_OUTCOME*` lines from `clear_provenance` makes this test FAIL
    (`assert 'suppressed' == 'not_run'`).
    """
    nm = Orthonym()

    # (a) The recorder molecule records an outcome, and records it FOR the
    # string it ships. Asserted, not assumed: if a future change breaks
    # the collision the failure lands HERE with a clear message rather
    # than silently turning this test back into a tautology.
    recorder = nm.name_tiered(LEAK_PROBE_RECORDS)
    live = pv.get_provenance()
    assert live["gate_outcome"] == pv.GATE_OUTCOME_SUPPRESSED, (
        f"probe pair broken: {LEAK_PROBE_RECORDS} no longer records "
        f"`suppressed` (got {live['gate_outcome']!r}) — pick a new recorder")
    assert live["gate_outcome_name"] == recorder["name"], (
        "probe pair broken: the recorded outcome must belong to the SHIPPED "
        "string, otherwise the name guard — not the reset — is what blocks "
        "the leak")

    # (b) The second molecule reaches NO gate-recording branch, and ships the
    # SAME string, so the name guard cannot fire.
    ungated = nm.name_tiered(LEAK_PROBE_NO_GATE)
    assert ungated["name"] == recorder["name"], (
        f"probe pair broken: {LEAK_PROBE_NO_GATE} must ship the same string "
        f"as the recorder ({recorder['name']!r}) for the name guard to be "
        f"neutralised; got {ungated['name']!r} — pick a new pair")

    # (c) THE LEAK-SENSITIVE ASSERTION. Fails as `suppressed != not_run` if
    # the reset is removed.
    assert ungated["gate_outcome"] == pv.GATE_OUTCOME_NOT_RUN, (
        f"stale gate outcome leaked onto the next molecule: expected "
        f"{pv.GATE_OUTCOME_NOT_RUN!r}, got {ungated['gate_outcome']!r} — "
        f"`clear_provenance` did not reset the contextvar")
    assert "self_consistency" not in ungated["gates_passed"]

    # (d) The reverse order must not leak either: the recorder still reports
    # its own outcome after the ungated molecule left `not_run` behind.
    again = nm.name_tiered(LEAK_PROBE_RECORDS)
    assert again["gate_outcome"] == pv.GATE_OUTCOME_SUPPRESSED


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
