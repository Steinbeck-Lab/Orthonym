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
# An ester whose senior parent is a 2-carbon acetyl chain the general engine
# genuinely DECLINES on (``_partition``: "unsupported suffix for pg='ester'").
# Task 5's route-around cascade SUPPRESSES that principal group so the ring
# becomes the parent and the acyl-oxy becomes an ``acetyloxy`` PREFIX -- no
# ester suffix needed -- yielding a complete, OPSIN-round-tripping name.
_ESTER_ROUTED_SMILES = "CC(=O)O[C@H]1C(=C)C=C(C=C1OC)OC"  # cid 639588
_ESTER_ROUTED_EXPECTED = (
    "(6S)-6-(acetyloxy)-1,3-dimethoxy-5-methylidenecyclohexa-1,3-diene")

# A molecule NO cascade rung can complete: the ``-OC(=O)NP(=O)(Cl)Cl``
# substituent (a dichlorophosphoryl carbamate) cannot be named -- the engine's
# recursive substituent namer hits its depth cap (DROP-12
# ``recursion_depth_fallback``) on the organophosphorus fragment, so every rung
# fails E1 and the producer HONESTLY abstains (None), never a partial. This is
# the genuine clean-abstain fixture now that the ester class routes around.
_UNROUTABLE_SMILES = "C1CCC(CC1)OC(=O)NP(=O)(Cl)Cl"


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
    """A molecule NO cascade rung can complete yields a clean abstain (None)
    from the producer -- never a raise, never a partial. Task 5's route-around
    cascade TRIES every applicable strategy (PG-suppression, ring-first then
    chain-parent) and, when the remaining substituent is genuinely unnameable
    (the organophosphorus carbamate below), falls through to an honest abstain.
    If it were ever to emit for this input, E1 must still bind every heavy atom.
    """
    mol, feats = _classified(_UNROUTABLE_SMILES)
    name = t4_coverage.name_t4_complete(mol, feats)
    if name is not None:
        # Not expected -- but a non-None here must be atom-complete, never a
        # partial (0-partial invariant). Prove it via the producer's E1 object.
        cand = t4_coverage._best_effort_candidate(mol, feats)
        assert cand is not None and cand.result_obj is not None
        verdict = verify_certificate(mol, cand.result_obj)
        assert verdict.ok, f"emitted a PARTIAL name: {name!r} ({verdict.reason})"
    else:
        assert name is None


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


@pytest.mark.unit
@pytest.mark.opsin_gate
def test_t4_routes_around_ester_decline():
    """Task 5: an ester the general engine DECLINES on (unsupported ester
    suffix) is now COMPLETED by the route-around cascade.

    ``CC(=O)O[C@H]1C(=C)C=C(C=C1OC)OC`` (cid 639588): the primary attempt
    returns None ("unsupported suffix for pg='ester'"). The cascade suppresses
    that principal group so the ring becomes the parent and the acyl-oxy is
    cited as an ``acetyloxy`` PREFIX -- no ester suffix -- giving a complete,
    E1-certified, OPSIN-round-tripping name. Direct producer call with
    classified features (the exact shape namer.py's T4 dispatch passes).

    Gated ``opsin_gate`` because the assertion round-trips through OPSIN; the
    hook skips it when the jar is absent (green-but-blind).
    """
    from orthonym.jvm_budget import jvm_slots
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

    mol, feats = _classified(_ESTER_ROUTED_SMILES)
    name = t4_coverage.name_t4_complete(mol, feats)
    assert name is not None, "cascade failed to route around the ester decline"
    assert "acetyloxy" in name, name          # acyl-oxy cited as a PREFIX
    assert "cyclohexa" in name, name          # the ring is the parent
    assert name == _ESTER_ROUTED_EXPECTED, name
    with jvm_slots(1, purpose="test-t4-ester-routearound"):
        rt = opsin_roundtrip_check(_ESTER_ROUTED_SMILES, name)
    assert rt["passed"], (
        f"cascade name did not round-trip: {rt!r}")


# --- Task 4: the through-namer integration test (Tasks 2/3 deferred this) ---
# The meaningful proof of the wiring: a molecule the PIN/default path abstains
# on must EMIT a complete name once the best-effort/unverified T4 tier is opted
# in -- and it must round-trip OPSIN-exact (SELF-01), since the wiring routes
# the T4 name through the same final round-trip ladder the engine's own name
# gets. Kept in this file (not the isolation file) because it needs a live OPSIN
# JVM; the isolation file stays JVM-cheap. The exact string is Task 3's verified
# emission for cid 1542461.
_T4_TARGET_EXPECTED = "(1R,2R)-2-((dimethylamino)methyl)cyclohexan-1-ol"


@pytest.mark.unit
@pytest.mark.opsin_gate
def test_t4_wired_into_namer_emits_for_abstainer():
    """`Orthonym(general_fallback=True, general_fallback_unverified=True)`
    emits the complete, atom-covering, OPSIN-round-tripping T4 name for the
    CLASS-A abstainer that the default PIN path cannot name.

    Runs with the OPSIN validity gate ON (``opsin_gate`` marker): under the
    conftest default (gate OFF) the PIN path ships an atom-DROPPED wrong name
    ((1R,2R)-2-aminocyclohexan-1-ol) that SELF-01 would suppress, so the
    abstention path the T4 producer sits behind is only reached with the gate
    live. The hook skips this test if the OPSIN jar is absent (green-but-blind).
    """
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="test-t4-wiring"):
        t4 = Orthonym(general_fallback=True, general_fallback_unverified=True)
        name = t4.name(_ABSTAINER_SMILES)
    assert name is not None
    from orthonym.errors import is_failure_name
    assert not is_failure_name(name), f"T4 abstained: {name!r}"
    assert "cyclohex" in name, name
    assert name == _T4_TARGET_EXPECTED, name


@pytest.mark.unit
@pytest.mark.opsin_gate
def test_t4_wiring_does_not_fire_without_unverified_optin():
    """The T4 producer is gated on `general_fallback_unverified`: with only the
    (conservative) verified general-fallback tier on, the abstainer still
    abstains -- the aggressive T4 producer must not run for it. Gate ON for the
    same reason as the emission test above.
    """
    from orthonym.jvm_budget import jvm_slots
    from orthonym.errors import is_failure_name
    with jvm_slots(1, purpose="test-t4-gate"):
        verified_only = Orthonym(general_fallback=True)
        name = verified_only.name(_ABSTAINER_SMILES)
    assert is_failure_name(name), (
        f"verified-only tier must not emit the aggressive T4 name: {name!r}")
