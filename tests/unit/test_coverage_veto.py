# tests/unit/test_coverage_veto.py
""" a phase Task L1: the coverage AUDIT becomes a real VETO under
`ORTHONYM_COVERAGE_AUDIT=veto`.

L0 built the SHADOW audit at `namer.py::_finish` (telemetry only, never
changed the returned name). L1 flips `veto` mode from a no-op into a real
rejecter: when the computed `CoverageVerdict.complete is False` AND
`.method != "unavailable"`, `_finish` must route to the existing honest
abstain (`_descriptive_fallback`) instead of shipping the incomplete winner.

CARRIED CONSTRAINTS (bind this file too):
  * NEVER veto when `verdict.method == "unavailable"` -- that is the
    fail-OPEN signal (OPSIN absent / a skip-reanchor carve-out / gate
    disabled), and vetoing there would abstain names that are actually fine.
  * `shadow` mode must stay a pure no-op (L0 behaviour untouched).
  * A certificate exception must never veto (fail-open on error).
"""
import pytest

from orthonym.namer import Orthonym, is_failure_name
from orthonym.assembly.coverage_audit import CoverageVerdict

pytestmark = pytest.mark.unit

# The exact canary from a local gap measurement: a general-engine best-effort winner whose
# `certify_general_result` (E1 + binding spine) FAILS -- `e1_spine`
# incomplete -- even though a malformed name shipped under SHADOW.
_TERPENOID_CANARY_SMILES = (
    "C=C1[C@@H](O)CC[C@]2(C)C3=C(CC[C@@H]12)[C@]1(O)[C@@H](O)"
    "C[C@H]([C@H](C)CC[C@H](CC)C(C)C)[C@@]1(C)C[C@H]3O"
)
# The SHADOW-mode winner: the general-engine best-effort name whose E1 spine is
# incomplete. The substituent gained its CIP descriptors + square-bracket nesting
# since this test was authored (stereo composition improved), but it remains the
# same incomplete-spine winner (verdict.complete is False, asserted below) and
# OPSIN round-trips to the canary (verify_or_none True). change-asserted-value:
# updated to the current deterministic shadow-mode emission.
#
# Leads program L3 (N8f): the side chain is '[(2R,5S)-5-ethyl-6-methylheptan-2-yl]',
# the Blue Book, the chain with the greater number of substituents; '6,7-dichloro-5-(2-
# chloropropyl)octan-2-yl (preferred prefix) [not 7-chloro-5-(1,2-dichloropropyl)octan-2-yl]',
#:22746), not the first of two tied arms '[(2R,5S)-5-(propan-2-yl)heptan-2-yl]'; it moves ahead of
# '2,15-dimethyl' in the alphanumerical order. Both names below were read back to the
# canary's full InChIKey by a fresh OPSIN call.
_TERPENOID_MALFORMED_NAME = (
    "(2S,5S,7R,11S,12S,14R,15R,17R)-14-[(2R,5S)-5-ethyl-6-methylheptan-2-yl]-2,15-dimethyl-"
    "6-methylidenetetracyclo[8.7.0.0^2,7.0^11,15]heptadec-1(10)-ene-5,11,12,17-tetrol"
)

# The VETO-mode rescue: when the veto rejects the incomplete-spine winner above,
# the L2-L4 offer layer (select_rt_passing, landed AFTER this test was authored)
# substitutes a DIFFERENT, complete, full-InChIKey-RT-verified floor candidate
# instead of abstaining -- so a name still ships and 0-wrong is held. It uses the
# tetrahydroxy PREFIX + '-ene' form rather than the '-tetrol' suffix.
_TERPENOID_VETO_RESCUE_NAME = (
    "(2S,5S,7R,11S,12S,14R,15R,17R)-14-[(2R,5S)-5-ethyl-6-methylheptan-2-yl]-5,11,12,17-"
    "tetrahydroxy-2,15-dimethyl-6-methylidenetetracyclo[8.7.0.0^2,7.0^11,15]heptadec-1(10)-ene"
)


def _canary_namer():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                      allow_aromatic_general=True)


class TestVetoRejectsIncompleteEGeneralEngineWinner:
    def test_e1_spine_incomplete_veto_mode_rescues_with_verified_complete(
            self, monkeypatch, opsin_gate):
        # Needs the REAL gate on (like L0's own leak-across-molecules test):
        # the suite's autouse gate-off default lets the legacy PIN pipeline
        # ship an un-suppressed WRONG candidate directly for this molecule,
        # so `general_fallback` (and thus the GeneralEngineResult / e1_spine
        # path this test targets) never engages.
        #
        # change-asserted-value: this test originally asserted the veto
        # ABSTAINS. Since then the L2-L4 offer layer (select_rt_passing over
        # the offers, incl. the floor) landed inside the same `_finish`
        # block, running unconditionally under general_fallback_unverified. So
        # the veto still REJECTS the incomplete-spine winner, but instead of
        # abstaining the offer layer now substitutes a DIFFERENT, complete,
        # full-InChIKey-RT-verified candidate -- a strict coverage improvement
        # with 0-wrong held (verify_or_none True). The veto's real invariant
        # (never SHIP the incomplete winner) is unchanged and still asserted.
        monkeypatch.setenv("ORTHONYM_COVERAGE_AUDIT", "veto")
        nm = _canary_namer()
        name = nm.name(_TERPENOID_CANARY_SMILES)
        verdict = nm._last_coverage_verdict
        # Sanity: the certificate really did flag this winner incomplete.
        assert verdict is not None
        assert verdict.method == "e1_spine"
        assert verdict.complete is False
        # The veto must have fired: the incomplete-spine winner must NOT ship.
        assert name != _TERPENOID_MALFORMED_NAME
        #... and the rescue is a real, RT-verified complete name, NOT a
        # sentinel abstain.
        assert not is_failure_name(name), (
            f"expected an RT-verified rescue, got abstain sentinel {name!r}")
        assert name == _TERPENOID_VETO_RESCUE_NAME

    def test_e1_spine_incomplete_shadow_mode_still_ships_it(self, monkeypatch,
                                                             opsin_gate):
        # L0 behaviour must be untouched: shadow NEVER vetoes.
        monkeypatch.setenv("ORTHONYM_COVERAGE_AUDIT", "shadow")
        nm = _canary_namer()
        name = nm.name(_TERPENOID_CANARY_SMILES)
        verdict = nm._last_coverage_verdict
        assert verdict is not None and verdict.complete is False
        assert name == _TERPENOID_MALFORMED_NAME


class TestVetoLeavesCompleteWinnersUnchanged:
    def test_complete_winner_veto_mode_ships_unchanged(self, monkeypatch):
        monkeypatch.setenv("ORTHONYM_COVERAGE_AUDIT", "veto")
        nm = Orthonym()
        name = nm.name("CCO")
        assert name == "ethanol"
        verdict = nm._last_coverage_verdict
        assert verdict is not None and verdict.complete is True


class TestVetoNeverFiresOnUnavailable:
    def test_method_unavailable_incomplete_is_not_vetoed(self, monkeypatch):
        # Fail-OPEN carried constraint: an "unavailable" verdict must never
        # be vetoed even if (hypothetically) `complete` were False on it --
        # monkeypatch `audit_coverage` directly to force this combination,
        # since production code never actually produces
        # complete=False+method=unavailable together (skip_reanchor forces
        # complete=True there) -- this proves the GUARD checks `.method`,
        # not just `.complete`.
        monkeypatch.setenv("ORTHONYM_COVERAGE_AUDIT", "veto")

        def _fake_audit_coverage(mol, name, result_obj, **kwargs):
            return CoverageVerdict(False, "unavailable", "forced for test")

        monkeypatch.setattr(
            "orthonym.assembly.coverage_audit.audit_coverage",
            _fake_audit_coverage)
        nm = Orthonym()
        name = nm.name("CCO")
        assert name == "ethanol", (
            "method=='unavailable' must NEVER be vetoed (fail-open)")


class TestVetoShadowModeParity:
    def test_shadow_mode_never_vetoes_even_when_incomplete(self, monkeypatch):
        monkeypatch.setenv("ORTHONYM_COVERAGE_AUDIT", "shadow")

        def _fake_audit_coverage(mol, name, result_obj, **kwargs):
            return CoverageVerdict(False, "e1_spine", "forced incomplete")

        monkeypatch.setattr(
            "orthonym.assembly.coverage_audit.audit_coverage",
            _fake_audit_coverage)
        nm = Orthonym()
        name = nm.name("CCO")
        assert name == "ethanol"


class TestVetoFailsOpenOnAuditException:
    def test_audit_exception_does_not_veto(self, monkeypatch):
        monkeypatch.setenv("ORTHONYM_COVERAGE_AUDIT", "veto")

        def _boom(mol, name, result_obj, **kwargs):
            raise RuntimeError("certificate blew up")

        monkeypatch.setattr(
            "orthonym.assembly.coverage_audit.audit_coverage", _boom)
        nm = Orthonym()
        name = nm.name("CCO")
        assert name == "ethanol", (
            "an audit exception must fail OPEN -- never veto/crash a name")
