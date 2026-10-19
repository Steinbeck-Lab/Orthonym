""" Task 0.1 — typed abstention limit-codes (measurement instrumentation).

Every fail-closed abstention path tags a typed ``AbstentionCode`` into a
per-top-level-naming-session telemetry slot (first-writer-wins = closest to
the root cause; the downstream OPSIN/ gate only claims molecules no
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


@pytest.fixture()
def be_namer():
    # The max-breadth (best-effort) config — the one the abstention census /
    # frozen sample is built with, and the tier under which the generation-stage
    # blocker codes are meaningfully recorded. Fixtures updated 2026-09-03 (
    # P2): the older default-namer molecules now NAME (the engine improved) or
    # changed code; these fixtures were re-verified to abstain with the asserted
    # code under this config.
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)


# abstaining fixtures re-verified 2026-09-03 under the max-breadth namer
_BRANCH_UNNAMEABLE_SMI = "[B](C1=CC=NN1)C2=CC=NN2"   # bis(pyrazolyl)borane; ring branch declined
_GATE_SUPPRESSED_SMI = "C1=CC=C(C=C1)O[13C](=O)OC2=CC=CC=C2"  # oxoacid/anhydride veto
# re-verified 2026-10-19 (UN lane C): the former fixture, a tetrachloro dispiro cage, is now named
# (best-effort `...dispiro[cyclopropane-1,12'-tetracyclo[6.3.1.0^2,7.0^3,5]dodec-10-ene-6',1''-cyclopropane]`,
# full read-back); PubChem CID 142408528 (a 12-ring polycyclic cage) still abstains with NO_PARENT
_NO_PARENT_SMI = "CC12CC34CC(CCC56C3C1(CCC(C5)C7=CC=CC=C7C8=CC=CC=C68)C9=CC=CC=C9C1=CC=CC=C21)C1=CC=CC=C1C1=C4C=CC(=C1)N(C1=CC=CC=C1)C1=CC=CC2=C1C1=CC=CC=C1C21C2=CC=CC=C2C2=CC=CC=C12"  # unsupported polycyclic parent


class TestEndToEndCodes:
    # Replacement witnesses, re-verified 2026-10-09 under the max-breadth namer (gate off,
    # as the suite runs). The bis(pyrazolyl)boranyl radical of ``_BRANCH_UNNAMEABLE_SMI`` is
    # named now ('di(1H-pyrazol-5-yl)boranyl', the radical build), and the 13C-labelled
    # diphenyl carbonate of ``_GATE_SUPPRESSED_SMI`` is named '{[phenoxy(13C)carbonyl]oxy}benzene',
    # so neither can demonstrate an abstention any more.
    #
    # A phenyl branch on a hypervalent iodide centre that the whole best-effort substituent
    # ladder declines: the ring-fallback census record fires at the ladder's true exhaustion.
    # (It is the 'charge-conservation veto' example of namer.py's P10 veto block, a class the
    # radical/ion builders do not reach.)
    # HU (2026-10-17) names the two iodide fixtures that stood here ('(2-hydroxyethyl)(phenyl)iodanuide',
    # read back through the method (2) twin), so both witnesses were replaced by rows of the frozen
    # unnamed panel (benchmarks/unnamed_panel) that abstain with this code and detail under this config.
    _RING_FALLBACK_SMI = "CC(C)(C)NP(=O)(C1=CC=CC=C1)[N-]C(C)(C)C"   # PubChem CID 101220894
    # The same ladder exhaustion on a ring whose numbering has no tie. ``_RING_FALLBACK_SMI`` still
    # abstains with BRANCH_UNNAMEABLE (asserted above), but its phenyl ring has two equivalent
    # orientations, and the (g) tie-break of ``general_engine._orient_carbocycle`` names the
    # ring's prefix speculatively (``name_substituent_for_ordering``) before the real assembly does.
    # That speculative call reaches ``_descriptive_fallback`` and writes ``enumerator_last_resort``
    # into the first-writer-wins slot ahead of the ladder's own ``enumerator_ring_fallback`` record;
    # the speculative wrapper restores the memos and the output-order props but not that slot (the
    # heteromonocycle tie-break did the same at 0c529ca8d, e.g. 3,5-disubstituted pyridine and
    # 1,4-disubstituted piperazine fixtures). Pyridin-3-yl has no orientation tie, so only the real
    # assembly records, and the detail is the one the test below is about.
    _RING_FALLBACK_DETAIL_SMI = "COC(=O)CC1=C(ON=[N+]1C2CCCCC2)[O-]"   # PubChem CID 24846373
    # A metal coordination compound (cisplatin drawn with explicit NH3 ligands): the
    # Java-free organometallic/coordination veto (namer.py, has_metal_coordination_bond;
    # coordination compounds are out of scope, suppresses the generated name.
    _COORDINATION_VETO_SMI = "[NH3][Pt]([NH3])(Cl)Cl"

    def test_branch_unnameable_fixture(self, be_namer):
        # A ring-bearing branch the whole best-effort substituent ladder declines
        # -> the enumerator_ring_fallback census record fires at the ladder's TRUE
        # exhaustion (fix) -> BRANCH_UNNAMEABLE.
        result = be_namer.name(self._RING_FALLBACK_SMI)
        assert is_failure_name(result), f"fixture must abstain, got {result!r}"
        assert abstention_code_for(result) is AbstentionCode.BRANCH_UNNAMEABLE

    def test_gate_suppressed_fixture(self, be_namer):
        # A Java-free veto (runs with the OPSIN gate disabled): the organometallic /
        # metal-coordination veto suppresses the generated name post-generation ->
        # GATE_SUPPRESSED. (The oxoacid/anhydride witness this test used to carry,
        # a 13C-labelled diphenyl carbonate, is named now.)
        result = be_namer.name(self._COORDINATION_VETO_SMI)
        assert is_failure_name(result), f"fixture must abstain, got {result!r}"
        assert abstention_code_for(result) is AbstentionCode.GATE_SUPPRESSED

    def test_no_parent_unsupported_ring_fixture(self, be_namer):
        # An unsupported polycyclic parent is refused at the top level
        # (UNSUPPORTED_RING_SYSTEM) -> NO_PARENT.
        result = be_namer.name(_NO_PARENT_SMI)
        assert is_failure_name(result), f"fixture must abstain, got {result!r}"
        assert abstention_code_for(result) is AbstentionCode.NO_PARENT

    def test_ring_fallback_recorded_from_best_effort_exhaustion(self, be_namer):
        # census-attribution fix: the enumerator_ring_fallback DETAIL is
        # recorded only when the best-effort substituent ladder is fully exhausted
        # (moved out of _descriptive_fallback, which the PIN Tier-5 path also
        # reaches). A genuinely ring-fallback branch still records it.
        result = be_namer.name(self._RING_FALLBACK_DETAIL_SMI)
        assert is_failure_name(result)
        rec = peek_abstention()
        assert rec is not None and rec.detail == "enumerator_ring_fallback"
        assert rec.code is AbstentionCode.BRANCH_UNNAMEABLE

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

    def test_slot_resets_between_top_level_calls(self, be_namer):
        # Failure first, then success: the success call clears the slot at
        # its top-level start, so no stale code leaks across molecules.
        first = be_namer.name(self._RING_FALLBACK_SMI)
        assert is_failure_name(first)
        assert abstention_code_for(first) is AbstentionCode.BRANCH_UNNAMEABLE

        second = be_namer.name("CCO")
        assert second == "ethanol"
        assert peek_abstention() is None

        # And a different failure class next records ITS code, not the old one.
        third = be_namer.name(_NO_PARENT_SMI)
        assert is_failure_name(third)
        assert abstention_code_for(third) is AbstentionCode.NO_PARENT


# ---------------------------------------------------------------------------
# Telemetry surface: name_with_confidence carries the code (additive key)
# ---------------------------------------------------------------------------

# A titanium complex: out of scope (organometallics,, so it stays unnamed under the
# max-breadth namer whatever the organic producers come to name; its branch is the one that
# records BRANCH_UNNAMEABLE. ``name_with_confidence`` names through ``name``, so the
# organic fixture above (bis(pyrazolyl)boranyl, which ``name`` names as
# 'di(1H-pyrazol-5-yl)boranyl') can no longer stand for an abstention here.
_ORGANOMETALLIC_BRANCH_SMI = "CCC1=CC[C-]=C1C.CCC1=CC[C-]=C1C.CC(=[Ti+2])C.[Cl-].[Cl-]"


class TestConfidenceSurface:
    def test_confidence_dict_carries_abstention_on_failure(self):
        meta = Orthonym(
            general_fallback=True, general_fallback_unverified=True,
            allow_aromatic_general=True).name_with_confidence(
                _ORGANOMETALLIC_BRANCH_SMI)
        assert is_failure_name(meta["name"])
        assert meta.get("abstention") == "BRANCH_UNNAMEABLE"

    def test_confidence_name_is_the_name_of_name(self):
        # One pipeline: name_with_confidence['name'] is what name gives, for a
        # molecule only name's late rescue names (bis(pyrazolyl)boranyl radical,
        # which the producer chain alone declines) and for one the producer chain names.
        be = Orthonym(general_fallback=True, general_fallback_unverified=True,
                      allow_aromatic_general=True)
        for smi in ("[B](C1=CC=NN1)C2=CC=NN2", "[CH2]C", "CCO"):
            assert be.name_with_confidence(smi)["name"] == be.name(smi), smi

    def test_confidence_dict_abstention_none_on_success(self):
        meta = Orthonym().name_with_confidence("CCO")
        assert meta["name"] == "ethanol"
        assert meta.get("abstention") is None
