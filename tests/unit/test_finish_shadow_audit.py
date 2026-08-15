# tests/unit/test_finish_shadow_audit.py
"""v33 Phase 0 Task L0.3: SHADOW coverage audit wired at `_finish`.

Telemetry only -- the returned name must be BYTE-IDENTICAL to the pre-L0.3
behaviour in every mode. Guarded by `ORTHONYM_COVERAGE_AUDIT`
(default "shadow"): off / shadow (L0, this task) / veto (reserved for L1,
behaves like shadow here).
"""
import os

import pytest

from orthonym.namer import Orthonym
from orthonym.assembly.coverage_audit import CoverageVerdict

pytestmark = pytest.mark.unit


class TestShadowAuditDoesNotChangeName:
    def test_default_shadow_mode_records_but_does_not_change_name(self):
        nm = Orthonym()
        before = nm.name("CCO")
        assert before == "ethanol"
        verdict = getattr(nm, "_last_coverage_verdict", None)
        assert verdict is not None
        assert isinstance(verdict, CoverageVerdict)
        assert verdict.complete is True

    def test_off_mode_still_ships_the_identical_name(self, monkeypatch):
        monkeypatch.setenv("ORTHONYM_COVERAGE_AUDIT", "off")
        nm = Orthonym()
        assert nm.name("CCO") == "ethanol"

    def test_veto_mode_behaves_like_shadow_in_l0(self, monkeypatch):
        # L1 (a later task) will turn "veto" into an actual rejecter. In L0 it
        # must be a no-op behaviourally -- same name, same recording.
        monkeypatch.setenv("ORTHONYM_COVERAGE_AUDIT", "veto")
        nm = Orthonym()
        assert nm.name("CCO") == "ethanol"
        verdict = getattr(nm, "_last_coverage_verdict", None)
        assert verdict is not None and verdict.complete is True

    def test_shadow_audit_does_not_leak_across_molecules(self, opsin_gate):
        # A per-top-level-molecule reset is required, mirroring the existing
        # confidence/pool/proof-ledger resets at the top of name() -- else a
        # bare-str winner after a GeneralEngineResult winner would incorrectly
        # inherit the previous molecule's result_obj. Needs the REAL gate on
        # (the `opsin_gate` fixture, not the suite's gate-off default) --
        # otherwise the legacy PIN pipeline ships an un-suppressed WRONG
        # candidate for this molecule and general_fallback never engages,
        # which is a test-harness artifact, not the production path.
        nm = Orthonym(general_fallback=True, general_fallback_unverified=True,
                        allow_aromatic_general=True)
        nm.name("CN(C)CC1CCCCC1O")  # a GENERAL-engine winner (e1_spine)
        v1 = nm._last_coverage_verdict
        nm.name("CCO")  # a plain retained-name (bare-str) winner
        v2 = nm._last_coverage_verdict
        assert v1 is not None and v2 is not None
        assert v1.method == "e1_spine"
        assert v2.method != "e1_spine"

    def test_diverse_molecules_unchanged_with_audit_on_vs_off(self, monkeypatch):
        smiles_list = [
            "CCO", "c1ccccc1", "CC(=O)O", "CC(C)(C)CC(=O)O",
            "c1ccc2ccccc2c1", "CC(N)C(=O)O", "OCC(O)CO", "CCN(CC)CC",
        ]
        nm_on = Orthonym()
        names_on = [nm_on.name(s) for s in smiles_list]
        monkeypatch.setenv("ORTHONYM_COVERAGE_AUDIT", "off")
        nm_off = Orthonym()
        names_off = [nm_off.name(s) for s in smiles_list]
        assert names_on == names_off
