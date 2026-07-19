"""v25 P0 Task 0.1 — typed abstention limit-codes (measurement instrumentation).

Every fail-closed abstention path tags a typed ``AbstentionCode`` into a
per-top-level-naming-session telemetry slot (first-writer-wins = closest to
the root cause; the downstream OPSIN/SELF-01 gate only claims molecules no
earlier site already classified). The instrumentation is side-effect-only:
no emitted name changes (HEAD-A/B byte-identity is part of the task gate).

The slot is read via ``abstention_code_for(result)`` which returns a code
ONLY when ``result`` is the failure sentinel (``is_failure_name``) — for a
successfully named molecule it returns None regardless of any speculative
codes recorded by exploratory paths during naming.

Fixtures (verified against HEAD at authoring time):
  - ``OC(=O)CC1CCN(C)CC1``: acetic-acid parent with an N-methylpiperidinyl
    ring substituent the ring-substituent machinery cannot express ->
    composer emits the 'unknown' branch marker -> BRANCH_UNNAMEABLE.
  - ``[I-](CCO)c1ccccc1``: hypervalent-iodine anion whose charge-blind name
    is vetoed by the Java-free P10 charge-conservation check ->
    GATE_SUPPRESSED (fires with the OPSIN gate disabled, as in unit tests).
  - ``C[C@H]1CC[C@@H](C[C@@H](O)C[C@@H]2CC[C@H](C)CC2)CC1``: W8-P6
    Cluster-E stereo-differing-``bis`` veto raises UNSUPPORTED_RING_SYSTEM
    at the top level (the parent structure is refused) -> NO_PARENT.
"""

import pytest

from orthonym.errors import is_failure_name
from orthonym.metrics.abstention import (
    AbstentionCode,
    abstention_code_for,
    clear_abstention,
    peek_abstention,
    record_abstention,
    record_suppression,
)
from orthonym.namer import Orthonym

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Recorder primitives (pure, no naming involved)
# ---------------------------------------------------------------------------

class TestRecorderPrimitives:
    def setup_method(self):
        clear_abstention()

    def teardown_method(self):
        clear_abstention()

    def test_empty_slot_peeks_none(self):
        assert peek_abstention() is None

    def test_record_then_peek(self):
        record_abstention(AbstentionCode.BRANCH_UNNAMEABLE, detail="x")
        rec = peek_abstention()
        assert rec is not None
        assert rec.code is AbstentionCode.BRANCH_UNNAMEABLE
        assert rec.detail == "x"

    def test_first_writer_wins(self):
        record_abstention(AbstentionCode.BRANCH_UNNAMEABLE, detail="root")
        record_abstention(AbstentionCode.GATE_SUPPRESSED, detail="late")
        rec = peek_abstention()
        assert rec.code is AbstentionCode.BRANCH_UNNAMEABLE
        assert rec.detail == "root"

    def test_suppression_overrides_speculative_branch_for_real_candidate(self):
        # An exploratory dead path recorded BRANCH_UNNAMEABLE, but a gate
        # then suppressed a COMPLETE real candidate -> generation provably
        # succeeded -> the suppression claims the attribution.
        record_abstention(AbstentionCode.BRANCH_UNNAMEABLE, detail="spec")
        record_suppression(AbstentionCode.GATE_SUPPRESSED, detail="self01",
                           candidate="2-iodobenzenylethan-1-ol")
        rec = peek_abstention()
        assert rec.code is AbstentionCode.GATE_SUPPRESSED

    def test_suppression_keeps_branch_for_failure_marked_candidate(self):
        # The suppressed candidate embeds the branch-failure marker
        # ('unknownacetic acid'): the branch IS the root cause -> kept.
        record_abstention(AbstentionCode.BRANCH_UNNAMEABLE, detail="root")
        record_suppression(AbstentionCode.GATE_SUPPRESSED, detail="unparse",
                           candidate="unknownacetic acid")
        rec = peek_abstention()
        assert rec.code is AbstentionCode.BRANCH_UNNAMEABLE

    def test_suppression_never_overrides_post_generation_codes(self):
        # A gate suppressing a downgrade's decomposition fallback keeps the
        # COVERAGE_DOWNGRADE attribution.
        record_suppression(AbstentionCode.COVERAGE_DOWNGRADE, detail="lowconf",
                           candidate="some assembled name")
        record_suppression(AbstentionCode.GATE_SUPPRESSED, detail="self01",
                           candidate="decomp fallback name")
        rec = peek_abstention()
        assert rec.code is AbstentionCode.COVERAGE_DOWNGRADE

    def test_suppression_on_empty_slot_records_plainly(self):
        record_suppression(AbstentionCode.GATE_SUPPRESSED, detail="self01",
                           candidate="benzene")
        assert peek_abstention().code is AbstentionCode.GATE_SUPPRESSED

    def test_clear_resets(self):
        record_abstention(AbstentionCode.OTHER)
        clear_abstention()
        assert peek_abstention() is None

    def test_code_for_success_name_is_none_even_with_recorded_code(self):
        # Speculative failures recorded during a naming that ultimately
        # SUCCEEDS must never surface a code.
        record_abstention(AbstentionCode.BRANCH_UNNAMEABLE)
        assert abstention_code_for("ethanol") is None

    def test_code_for_failure_defaults_to_other(self):
        # Failure sentinel with no recorded site -> OTHER (the residual).
        assert abstention_code_for("unknown organic compound") is AbstentionCode.OTHER

    def test_code_is_json_friendly(self):
        # The census (Task 0.2) writes codes into JSON artifacts.
        assert AbstentionCode.NO_PARENT.value == "NO_PARENT"
        assert isinstance(AbstentionCode.NO_PARENT.value, str)


# ---------------------------------------------------------------------------
# End-to-end: naming a molecule tags the right code
# ---------------------------------------------------------------------------

@pytest.fixture()
def namer():
    return Orthonym()


class TestEndToEndCodes:
    def test_branch_unnameable_fixture(self, namer):
        # N-methylpiperidinyl branch on an acetic-acid parent: the ring-
        # substituent machinery declines -> 'unknown' marker glued into the
        # candidate -> failure sentinel. Root cause = the unnameable BRANCH.
        result = namer.name("OC(=O)CC1CCN(C)CC1")
        assert is_failure_name(result), f"fixture must abstain, got {result!r}"
        assert abstention_code_for(result) is AbstentionCode.BRANCH_UNNAMEABLE

    def test_gate_suppressed_charge_dropped_fixture(self, namer):
        # Documented P10 charge-conservation veto (Java-free, runs with the
        # OPSIN gate disabled): a generated charge-blind name is suppressed.
        result = namer.name("[I-](CCO)c1ccccc1")
        assert is_failure_name(result), f"fixture must abstain, got {result!r}"
        assert abstention_code_for(result) is AbstentionCode.GATE_SUPPRESSED

    def test_no_parent_unsupported_ring_fixture(self, namer):
        # W8-P6 Cluster-E stereo-differing-bis veto: the parent structure is
        # refused at the top level (UNSUPPORTED_RING_SYSTEM) -> NO_PARENT.
        result = namer.name(
            "C[C@H]1CC[C@@H](C[C@@H](O)C[C@@H]2CC[C@H](C)CC2)CC1")
        assert is_failure_name(result), f"fixture must abstain, got {result!r}"
        assert abstention_code_for(result) is AbstentionCode.NO_PARENT

    def test_coverage_downgrade_path_records_code(self, namer, monkeypatch):
        """The >15-HA GENERAL quality/atom-coverage downgrade gates tag
        COVERAGE_DOWNGRADE when they reject the assembled name.

        Exercised via the garbled-name branch (deterministic, corpus-
        independent): force the assembled GENERAL name to look garbled and
        decomposition to fail, so the downgrade machinery engages and the
        molecule ends in a failure sentinel.
        """
        import orthonym.decomposition as decomposition

        # A >15-HA acyclic molecule that routes GENERAL (verified: the CFR
        # GENERAL shim returns None -> the legacy assemble_name pipeline).
        smiles = "CCCCCCCCCCCCCCCC(O)CO"  # C17 diol, 19 HA

        # Force the assembled GENERAL name to look garbled ('cycloane' is a
        # canonical token from the gate's _GARBLED_TOKENS list) and make
        # decomposition fail, so the downgrade machinery engages.
        monkeypatch.setattr("orthonym.namer.assemble_name",
                            lambda features, style="pin", **kw: "cycloane")
        monkeypatch.setattr(decomposition, "try_decompose",
                            lambda mol, style="pin": None)

        result = namer.name(smiles)
        rec = peek_abstention()
        assert rec is not None, "downgrade gate must record a code"
        assert rec.code is AbstentionCode.COVERAGE_DOWNGRADE

    def test_success_molecule_reports_none(self, namer):
        result = namer.name("CCO")
        assert result == "ethanol"
        assert abstention_code_for(result) is None

    def test_slot_resets_between_top_level_calls(self, namer):
        # Failure first, then success: the success call clears the slot at
        # its top-level start, so no stale code leaks across molecules.
        first = namer.name("OC(=O)CC1CCN(C)CC1")
        assert is_failure_name(first)
        assert abstention_code_for(first) is AbstentionCode.BRANCH_UNNAMEABLE

        second = namer.name("CCO")
        assert second == "ethanol"
        assert peek_abstention() is None

        # And a different failure class next records ITS code, not the old one.
        third = namer.name(
            "C[C@H]1CC[C@@H](C[C@@H](O)C[C@@H]2CC[C@H](C)CC2)CC1")
        assert is_failure_name(third)
        assert abstention_code_for(third) is AbstentionCode.NO_PARENT


# ---------------------------------------------------------------------------
# Telemetry surface: name_with_confidence carries the code (additive key)
# ---------------------------------------------------------------------------

class TestConfidenceSurface:
    def test_confidence_dict_carries_abstention_on_failure(self):
        meta = Orthonym().name_with_confidence("OC(=O)CC1CCN(C)CC1")
        assert is_failure_name(meta["name"])
        assert meta.get("abstention") == "BRANCH_UNNAMEABLE"

    def test_confidence_dict_abstention_none_on_success(self):
        meta = Orthonym().name_with_confidence("CCO")
        assert meta["name"] == "ethanol"
        assert meta.get("abstention") is None
