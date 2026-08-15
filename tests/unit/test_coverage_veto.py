# tests/unit/test_coverage_veto.py
"""v33 Phase 0 Task L1: the coverage AUDIT becomes a real VETO under
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

# The exact canary from the L0.4 gap-oracle measurement log
# ( via
# `l0_4_measure.log`): a general-engine best-effort winner whose
# `certify_general_result` (E1 + binding spine) FAILS -- `e1_spine`
# incomplete -- even though a malformed name shipped under SHADOW.
_TERPENOID_CANARY_SMILES = (
    "C=C1[C@@H](O)CC[C@]2(C)C3=C(CC[C@@H]12)[C@]1(O)[C@@H](O)"
    "C[C@H]([C@H](C)CC[C@H](CC)C(C)C)[C@@]1(C)C[C@H]3O"
)
_TERPENOID_MALFORMED_NAME = (
    "(2S,5S,7R,11S,12S,14R,15R,17R)-2,15-dimethyl-6-methylidene-"
    "14-(5-(propan-2-yl)heptan-2-yl)tetracyclo[8.7.0.0^2,7.0^11,15]"
    "heptadec-1(10)-ene-5,11,12,17-tetrol"
)


def _canary_namer():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                      allow_aromatic_general=True)


class TestVetoRejectsIncompleteEGeneralEngineWinner:
    def test_e1_spine_incomplete_veto_mode_abstains(self, monkeypatch, opsin_gate):
        # Needs the REAL gate on (like L0's own leak-across-molecules test):
        # the suite's autouse gate-off default lets the legacy PIN pipeline
        # ship an un-suppressed WRONG candidate directly for this molecule,
        # so `general_fallback` (and thus the GeneralEngineResult / e1_spine
        # path this test targets) never engages.
        monkeypatch.setenv("ORTHONYM_COVERAGE_AUDIT", "veto")
        nm = _canary_namer()
        name = nm.name(_TERPENOID_CANARY_SMILES)
        verdict = nm._last_coverage_verdict
        # Sanity: the certificate really did flag this winner incomplete.
        assert verdict is not None
        assert verdict.method == "e1_spine"
        assert verdict.complete is False
        # The veto must have fired: the malformed name must NOT ship.
        assert name != _TERPENOID_MALFORMED_NAME
        assert is_failure_name(name), (
            f"expected an honest abstain sentinel, got {name!r}")

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
