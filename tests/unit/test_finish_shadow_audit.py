# tests/unit/test_finish_shadow_audit.py
""" a phase Task L0.3: SHADOW coverage audit wired at `_finish`.

Telemetry only -- the returned name must be BYTE-IDENTICAL to the pre-L0.3
behaviour in every mode. Guarded by `ORTHONYM_COVERAGE_AUDIT`
(default "shadow"): off / shadow (L0, this task) / veto (reserved for L1,
behaves like shadow here).
"""
import os
import time

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
        # confidence/pool/proof-ledger resets at the top of name -- else a
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


class TestSelf01ReuseFiresEndToEnd:
    """ a phase L0 review fix (I2): a `_finish`-INTEGRATION-level proof that
    the CARRIED-RULING reuse actually fires on the real path -- the gap that
    let review finding C1 through (the unit tests on `audit_coverage` alone
    could not see whether `namer._self01_lookup` was ever actually WIRED to a
    real outcome end-to-end)."""

    def test_bare_str_pin_winner_reuses_self01_with_gate_on(self, opsin_gate):
        nm = Orthonym()
        assert nm.name("CCO") == "ethanol"
        verdict = nm._last_coverage_verdict
        assert verdict is not None
        assert verdict.method == "self01"
        assert verdict.complete is True


class TestSkipReanchorNeverSpawnsOpsin:
    """ a phase L0 review fix (C1/C2 + I2): a resolved gate outcome that
    guarantees a reanchor is useless (a `carveout:*` PIN family, or a
    disabled/unavailable/not-run gate) must skip `validate_atom_coverage`
    ENTIRELY at `_finish` -- no OPSIN subprocess spawned, and FAST.

    OPSIN 2.9.0 in this environment actually parses 'methane-SO-thioperoxol'
    and 'myo-inositol' successfully (verified by direct probe), so those two
    legacy carve-outs are not LIVE-reachable right now to demonstrate this --
    a real, separate finding (the carve-outs' 'OPSIN-unparseable' premise may
    be stale), out of scope for this fix. Instead this primes the exact
    provenance state a genuine carve-out molecule produces
    (`_final_opsin_validity_gate`'s `carveout_outcome(...)` call), which is
    the real production mechanism `_self01_lookup` reads -- so this exercises
    the ACTUAL `_finish` wiring, not a mock of it.
    """

    def _finish_after_carveout(self, nm, name, smiles, slug="thioperoxol"):
        from orthonym.metrics import provenance as _pv
        _pv.clear_provenance()
        _pv.record_gate_outcome(_pv.carveout_outcome(slug), name)
        return nm._finish(name, smiles)

    def test_carveout_outcome_skips_reanchor_and_is_fast(self, monkeypatch):
        def _boom(mol, name):
            raise AssertionError(
                "validate_atom_coverage must NOT be called for a carveout "
                "outcome -- that is exactly the wasted OPSIN spawn review "
                "finding C1 forbids")
        monkeypatch.setattr(
            "orthonym.validation.atom_coverage.validate_atom_coverage", _boom)
        nm = Orthonym()
        name = "methane-SO-thioperoxol"
        t0 = time.perf_counter()
        returned = self._finish_after_carveout(nm, name, "CSO")
        elapsed = time.perf_counter() - t0
        assert returned == name  # SHADOW never changes the name
        verdict = nm._last_coverage_verdict
        assert verdict is not None
        assert verdict.method == "unavailable"
        assert verdict.complete is True
        assert "reanchor skipped" in verdict.detail
        assert elapsed < 0.5, f"expected no OPSIN spawn, took {elapsed:.3f}s"

    def test_gate_disabled_outcome_skips_reanchor(self, monkeypatch):
        # C2: the suite's OWN autouse gate-off default resolves to
        # `gate_disabled`, which must ALSO skip the reanchor -- this is what
        # made every unit test that names a molecule pay a real OPSIN cost
        # before this fix.
        def _boom(mol, name):
            raise AssertionError(
                "validate_atom_coverage must NOT be called when the gate is "
                "disabled -- C2")
        monkeypatch.setattr(
            "orthonym.validation.atom_coverage.validate_atom_coverage", _boom)
        nm = Orthonym()
        t0 = time.perf_counter()
        name = nm.name("CCO")
        elapsed = time.perf_counter() - t0
        assert name == "ethanol"
        verdict = nm._last_coverage_verdict
        assert verdict is not None
        assert verdict.method == "unavailable"
        assert verdict.complete is True
        assert elapsed < 0.5, f"expected no OPSIN spawn, took {elapsed:.3f}s"
