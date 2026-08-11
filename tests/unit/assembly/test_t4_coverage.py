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
from orthonym.namer import Orthonym
from orthonym.validation.e1_certificate import verify_certificate

# CLASS-A T4 target -- 2-[(dimethylamino)methyl]cyclohexan-1-ol (cid 1542461).
# The PIN/default path abstains on it; the T4 producer names it completely.
_ABSTAINER_SMILES = "CN(C)C[C@H]1CCCC[C@H]1O"
# A molecule whose senior parent is a 2-carbon acetyl chain-ester the general
# engine genuinely DECLINES (``_partition``: "unsupported suffix for
# pg='ester'"); the Task-3 happy-path producer therefore returns None on it
# (clean abstain -- Task 5's route-around cascade handles this class).
_DECLINE_SMILES = "CC(=O)O[C@H]1C(=C)C=C(C=C1OC)OC"  # cid 639588


def _mol():
    mol = Chem.MolFromSmiles(_ABSTAINER_SMILES)
    assert mol is not None
    return mol


def _classified(smi):
    """Build ``(mol, features)`` the way ``namer.py``'s T4 dispatch does --
    ``self._perceive(mol, smiles, canonical); self._classify(feats)``. Task 4
    passes exactly this ``features`` shape, so the producer and its tests must
    use it too (``compute_features`` alone does NOT run ``_classify``, so it
    leaves ``principal_group`` unset -- see the Task-2 report). Perception and
    classification are flag-independent, so a default ``Orthonym()`` builds
    the same object the T4 dispatch would.
    """
    nm = Orthonym()
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    canonical = Chem.MolToSmiles(mol, canonical=True)
    feats = nm._perceive(mol, smi, canonical)
    nm._classify(feats)
    return mol, feats


@pytest.mark.unit
def test_declining_molecule_returns_none():
    """A molecule the general engine genuinely DECLINES yields a clean abstain
    (None) from the Task-3 happy-path producer -- never a raise, never a
    partial. (Task 2's obsolete ``test_stub_candidate_none_returns_none``
    asserted the same None-safety against the stub; Task 3 replaces the stub,
    so the None-safety contract is now proven against a real decline.)
    """
    mol, feats = _classified(_DECLINE_SMILES)
    assert t4_coverage.name_t4_complete(mol, feats) is None


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


@pytest.mark.unit
def test_t4_ring_plus_offring_fg_is_complete():
    """The CLASS-A target (a ring parent + an off-ring FG the PIN path
    abstains on) emits a complete name from the T4 producer.

    ``CN(C)C[C@H]1CCCC[C@H]1O`` -> the cyclohexanol ring parent + the
    ``(dimethylamino)methyl`` off-ring substituent, atom-complete and
    E1-certified.
    """
    mol, feats = _classified(_ABSTAINER_SMILES)
    name = t4_coverage.name_t4_complete(mol, feats)
    assert name is not None
    assert "cyclohex" in name


@pytest.mark.unit
def test_t4_candidate_is_complete_or_none():
    """Invariant: the producer NEVER hands up a partial. For the CLASS-A
    target it returns a candidate whose bindings cover EVERY heavy atom
    (E1-complete); in general it is None OR fully atom-covering, never a
    fragment.
    """
    mol, feats = _classified(_ABSTAINER_SMILES)
    cand = t4_coverage._best_effort_candidate(mol, feats)
    assert cand is not None                      # this target must emit
    assert cand.result_obj is not None
    verdict = verify_certificate(mol, cand.result_obj)
    assert verdict.ok, verdict.reason
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    bound = set()
    for b in cand.result_obj.bindings:
        bound |= set(b.atom_ids)
    assert bound >= heavy, f"unbound heavy atoms: {sorted(heavy - bound)}"
