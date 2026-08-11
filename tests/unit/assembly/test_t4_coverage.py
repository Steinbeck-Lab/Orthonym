"""Unit tests for the T4 coverage-by-construction skeleton (Task 2).

These exercise ``name_t4_complete``'s control flow DIRECTLY, via monkeypatch
on ``_best_effort_candidate`` -- no real ``GeneralEngineResult`` is needed
yet (Task 3 builds the producer; Tasks 4-6 wire the cascade + namer.py). This
module is NOT wired into ``Orthonym.name()`` until Task 4, so a test that
went through ``Orthonym(...).name(smi)`` here would exercise only the
EXISTING namer and assert nothing about this skeleton.
"""
import types

import pytest
from rdkit import Chem

from orthonym.assembly import t4_coverage
from orthonym.namer import compute_features

# Confirmed (this session) to abstain today under both general-fallback
# flags -- 2-[(dimethylamino)methyl]cyclohexan-1-ol. Real molecule + real
# perception object for the "stub returns None" case; the monkeypatched
# cases below don't need it (the patched ``_best_effort_candidate`` ignores
# its arguments), so they pass plain ``None`` for ``features``.
_ABSTAINER_SMILES = "CN(C)C[C@H]1CCCC[C@H]1O"


def _mol():
    mol = Chem.MolFromSmiles(_ABSTAINER_SMILES)
    assert mol is not None
    return mol


@pytest.mark.unit
def test_stub_candidate_none_returns_none():
    """With the real Task-2 stub (``_best_effort_candidate`` -> None), the
    namer cleanly abstains and never raises -- None-safety end to end.

    ``compute_features`` is the public perception entry (``namer.py``'s
    ``Orthonym()._perceive`` wrapper); it does NOT run ``_classify``, so the
    ``MolecularFeatures`` here has no ``principal_group`` populated. That's
    fine for today's stub (it ignores ``features`` entirely) -- Task 3's real
    producer will need the fuller perceive+classify pattern that
    ``namer.py``'s ``_try_general_engine_recovery`` uses.
    """
    mol = _mol()
    features = compute_features(mol)
    assert t4_coverage.name_t4_complete(mol, features) is None


@pytest.mark.unit
def test_result_obj_none_bypasses_e1(monkeypatch):
    """A candidate with no proof object ships unchecked -- E1 has nothing to
    verify against."""
    mol = _mol()
    monkeypatch.setattr(
        t4_coverage, "_best_effort_candidate",
        lambda m, f: t4_coverage._Candidate(name="testname", result_obj=None))
    assert t4_coverage.name_t4_complete(mol, None) == "testname"


@pytest.mark.unit
def test_e1_fail_returns_none(monkeypatch):
    """A candidate that fails the E1 atom-coverage certificate is discarded
    -- never patched, never shipped."""
    mol = _mol()
    sentinel_result_obj = object()
    monkeypatch.setattr(
        t4_coverage, "_best_effort_candidate",
        lambda m, f: t4_coverage._Candidate(
            name="bad", result_obj=sentinel_result_obj))
    monkeypatch.setattr(
        t4_coverage, "verify_certificate",
        lambda m, r, allow_charged=False: types.SimpleNamespace(
            ok=False, reason="x"))
    assert t4_coverage.name_t4_complete(mol, None) is None


@pytest.mark.unit
def test_e1_pass_returns_name(monkeypatch):
    """A candidate that passes E1 ships its name unchanged."""
    mol = _mol()
    sentinel_result_obj = object()
    monkeypatch.setattr(
        t4_coverage, "_best_effort_candidate",
        lambda m, f: t4_coverage._Candidate(
            name="good", result_obj=sentinel_result_obj))
    monkeypatch.setattr(
        t4_coverage, "verify_certificate",
        lambda m, r, allow_charged=False: types.SimpleNamespace(
            ok=True, reason="ok"))
    assert t4_coverage.name_t4_complete(mol, None) == "good"
