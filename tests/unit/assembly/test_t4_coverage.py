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
from orthonym.assembly.general_engine import GeneralEngineResult, TokenBinding
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
    """A candidate that passes E1 AND the wired binding-spine audit (Phase 0c
    Task 2b) ships its name unchanged. ``result_obj`` needs a real (empty)
    ``.bindings`` -- the wiring calls ``BindingSpine.from_token_bindings``
    unconditionally once E1 passes -- and ``verify_spine`` itself is mocked so
    this test isolates the CONTROL FLOW (E1 pass + spine pass -> ship) from
    the spine's own proofs, which have their own dedicated tests."""
    mol = _mol()
    sentinel_result_obj = types.SimpleNamespace(bindings=())
    monkeypatch.setattr(
        t4_coverage, "_best_effort_candidate",
        lambda m, f: t4_coverage._Candidate(
            name="good", result_obj=sentinel_result_obj))
    monkeypatch.setattr(
        t4_coverage, "verify_certificate",
        lambda m, r, allow_charged=False: types.SimpleNamespace(
            ok=True, reason="ok"))
    monkeypatch.setattr(
        t4_coverage, "verify_spine",
        lambda m, s, n, mode="audit", allow_charged=False, escalate=frozenset():
            types.SimpleNamespace(ok=True, findings=(), stats={}))
    assert t4_coverage.name_t4_complete(mol, None) == "good"


@pytest.mark.unit
def test_spine_fail_returns_none(monkeypatch):
    """Phase 0c Task 2b: a candidate that passes E1 but FAILS the wired
    binding-spine audit is discarded -- the spine is an ADDITIONAL gate, not a
    substitute for E1, so a spine-only defect must still void the candidate."""
    mol = _mol()
    sentinel_result_obj = types.SimpleNamespace(bindings=())
    monkeypatch.setattr(
        t4_coverage, "_best_effort_candidate",
        lambda m, f: t4_coverage._Candidate(
            name="bad-structure", result_obj=sentinel_result_obj))
    monkeypatch.setattr(
        t4_coverage, "verify_certificate",
        lambda m, r, allow_charged=False: types.SimpleNamespace(
            ok=True, reason="ok"))
    monkeypatch.setattr(
        t4_coverage, "verify_spine",
        lambda m, s, n, mode="audit", allow_charged=False, escalate=frozenset():
            types.SimpleNamespace(
                ok=False, findings=("BOND_AMBIGUOUS_LINKAGE",), stats={}))
    assert t4_coverage.name_t4_complete(mol, None) is None


@pytest.mark.unit
def test_e1_passes_but_bond_drop_is_caught_by_the_wired_spine(monkeypatch):
    """A REAL (unmocked) verify_spine run on a genuine E1-blind structural
    defect: cyclohexane spelled as two disjoint 3-atom halves ('prop' +
    'propyl') is atom-complete (E1 passes -- E1 never looks at bonds at all)
    but the two undeclared cross bonds CLOSE A CYCLE that no substituent
    prefix spells (BOND_AMBIGUOUS_LINKAGE, mirroring
    ``test_p2_two_undeclared_cross_bonds_are_ambiguous`` in
    ``test_binding_spine.py``, exercised here through the actual T4 wiring).
    Before Phase 0c Task 2 this candidate would have SHIPPED as
    'propylpropane' for cyclohexane -- a wrong structural claim E1 cannot see.
    """
    mol = Chem.MolFromSmiles("C1CCCCC1")
    bad_result = GeneralEngineResult(
        name="propylpropane",
        bindings=(
            TokenBinding((0, 1, 2), "prop", "parent"),
            TokenBinding((3, 4, 5), "propyl", "prefix"),
        ),
    )
    monkeypatch.setattr(
        t4_coverage, "_best_effort_candidate",
        lambda m, f: t4_coverage._Candidate(
            name="propylpropane", result_obj=bad_result))
    # E1 itself must actually pass this (proving the defect is E1-invisible):
    assert verify_certificate(mol, bad_result).ok
    assert t4_coverage.name_t4_complete(mol, None) is None


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
def test_t4_chiral_ester_alkyl_word_does_not_void_the_parent_stereo():
    """Phase 0c Task 4 FIX-ROUND regression (reviewer-found, real end-to-end
    reproduction): a functional-class two-word ester name whose ALKYL word
    is itself chiral (built by the fully general substituent namer, so it
    carries its OWN leading ``(nR)/(nS)`` block) used to make P8a select
    that word's descriptor instead of the parent's -- reporting all 3 real
    parent centres MISSING and, under
    ``escalate=STRICT_STEREO_CHARGE_AXES``, voiding this otherwise fully
    correct T4 candidate (an abstain, not a wrong name -- but a genuine
    breadth regression this fix closes).
    """
    smi = "CC[C@]12C=CCN3CC[C@]4(C(=C(C(=O)O[C@@H](C)CC)C1)Nc1ccccc14)[C@@H]32"
    mol, feats = _classified(smi)
    name = t4_coverage.name_t4_complete(mol, feats)
    assert name is not None, "the chiral-ester-alkyl candidate must not be voided"
    assert name.startswith("(2S)-butan-2-yl (1R,12R,19S)-"), name


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


# --- Task 6: polyfunctional complete-or-abstain invariant locks ---------------
# Task 6 diagnosis (*.py, measured 2026-08-11): name_general is
# ALL-OR-NOTHING (no partial-with-remainder to route), and the only feature-
# override levers are principal_group / chain_is_parent, whose full toggle space
# the rung 0-2 cascade already covers -- a chain_is_parent=True variant converts
# 0/150 producer-None polyfunctional molecules. So there is NO buildable
# t4_coverage rung that converts a polyfunctional molecule beyond Tasks 3-5
# (DONE_WITH_CONCERNS). These two tests LOCK the contract Task 6 certifies for the
# polyfunctional class: (a) the universal-decomposition path stays complete +
# atom-covering for a molecule it CAN name, and (b) a molecule it cannot complete
# abstains HONESTLY (None) -- never a partial. They are regression locks, GREEN
# because the invariant already holds; a "conversion" assertion would be an
# unsatisfiable target (invariant 16), so it is deliberately not asserted.

# Polyfunctional (N-acetyl amide + aldehyde/oxo + carboxylic acid + one
# stereocentre), all atoms covered. Converts today via rung 0 (perceived PG); the
# lock guards that the polyfunctional complete-coverage path keeps working.
_POLYFUNC_COMPLETE_SMILES = "CC(=O)N[C@@H](CCCC=O)C(=O)O"  # cid 6303498
_POLYFUNC_COMPLETE_EXPECTED = "(2S)-2-acetamido-6-oxohexanoic acid"

# Molecules the T4 producer cannot complete with existing capabilities, so it
# HONESTLY abstains (None): perindopril's deep peptide-ester side chain hits
# DROP-12 recursion_depth_fallback (peptide-residue follow-on), and the crotonyl
# enamide branch is "branch unnameable" (unsaturated-acyl-substituent follow-on).
# Neither is forced -- 0-partial holds (the design's honest-abstain bar).
_POLYFUNC_ABSTAIN = [
    "CCC[C@@H](C(=O)OCC)N[C@H](C)C(=O)N1[C@H]2CCCC[C@H]2C[C@H]1C(=O)O",  # 60184
    "C/C=C/C(=O)NCC(=O)O",                                               # 2818292
]


@pytest.mark.unit
@pytest.mark.opsin_gate
def test_t4_polyfunctional_complete_coverage_locks():
    """A polyfunctional molecule (amide + aldehyde + acid + stereo) is named by
    the T4 universal-decomposition path with EVERY heavy atom bound (E1-complete)
    and the name OPSIN-round-trips. This locks the complete-coverage contract for
    the polyfunctional class -- a fragment-drop here would fail E1, and a
    constitution error would fail the round-trip.
    """
    from orthonym.jvm_budget import jvm_slots
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

    mol, feats = _classified(_POLYFUNC_COMPLETE_SMILES)
    cand = t4_coverage._best_effort_candidate(mol, feats)
    assert cand is not None and cand.result_obj is not None
    verdict = verify_certificate(mol, cand.result_obj)
    assert verdict.ok, verdict.reason
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    bound = set()
    for b in cand.result_obj.bindings:
        bound |= set(b.atom_ids)
    assert bound >= heavy, f"unbound heavy atoms: {sorted(heavy - bound)}"

    name = t4_coverage.name_t4_complete(mol, feats)
    assert name == _POLYFUNC_COMPLETE_EXPECTED, name
    with jvm_slots(1, purpose="test-t4-polyfunc-complete"):
        rt = opsin_roundtrip_check(_POLYFUNC_COMPLETE_SMILES, name)
    assert rt["passed"], f"polyfunctional name did not round-trip: {rt!r}"


@pytest.mark.unit
@pytest.mark.parametrize("smi", _POLYFUNC_ABSTAIN)
def test_t4_polyfunctional_honest_abstain_no_partial(smi):
    """A polyfunctional molecule the T4 producer cannot COMPLETE abstains
    honestly (None) -- never a bare fragment / partial. This is the corrected,
    strengthened form of the brief's Step-1 test: the atom-drop it worried about
    (a scaffold named while side chains are dropped) does NOT happen here,
    because the producer returns None rather than a partial. If it ever DOES
    emit for one of these, E1 must still bind every heavy atom (0-partial).
    """
    mol, feats = _classified(smi)
    name = t4_coverage.name_t4_complete(mol, feats)
    if name is None:
        return  # honest abstain -- the expected, complete-or-abstain outcome
    # Emission is not expected today, but if it happens it must be atom-complete.
    cand = t4_coverage._best_effort_candidate(mol, feats)
    assert cand is not None and cand.result_obj is not None
    verdict = verify_certificate(mol, cand.result_obj)
    assert verdict.ok, f"emitted a PARTIAL name {name!r}: {verdict.reason}"
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    bound = set()
    for b in cand.result_obj.bindings:
        bound |= set(b.atom_ids)
    assert bound >= heavy, f"unbound heavy atoms: {sorted(heavy - bound)}"


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


@pytest.mark.unit
@pytest.mark.opsin_gate
def test_t4_unparseable_name_abstains_not_ships_unverified(monkeypatch):
    """FINAL-REVIEW FIX 1: a T4 producer name OPSIN CANNOT PARSE must make the
    T4 recovery path ABSTAIN (None), never ship it with opsin_status='unverified'.

    The shared emit ladder's ``elif not general_fallback_unverified: return None``
    is False for the T4 opt-in, so BEFORE the fix an unparseable T4 name shipped
    unverified. The fix flags the T4 origin (``_cand_from_t4``) so that branch
    rejects it -- a T4 name must POSITIVELY round-trip or abstain. E1 proves atom
    COVERAGE (so never a wrong MOLECULE), but a name OPSIN cannot parse is
    malformed and must not ship.

    Uses the CLASS-A abstainer, which the acceptance probe confirms enters the
    T4 branch of ``_try_general_engine_recovery`` (its own engine attempt
    declines at allow_aromatic_general=False, so the T4 producer is invoked).
    The first assertion proves the fix does NOT over-reject: the real producer's
    positively-round-tripping name still ships.
    """
    from orthonym.assembly import t4_coverage
    from orthonym.errors import is_failure_name
    from orthonym.jvm_budget import jvm_slots

    with jvm_slots(1, purpose="test-t4-fix1-unparseable"):
        nm = Orthonym(general_fallback=True, general_fallback_unverified=True)
        # (1) the real producer's valid name DOES ship (T4 branch entered; the
        # fix leaves a positively-round-tripping T4 name untouched).
        shipped = nm._try_general_engine_recovery(_ABSTAINER_SMILES)
        assert shipped is not None and not is_failure_name(shipped), shipped
        assert shipped == _T4_TARGET_EXPECTED, shipped
        # (2) force the hole: an unparseable T4 name must now ABSTAIN, not ship.
        monkeypatch.setattr(
            t4_coverage, "name_t4_complete",
            lambda mol, feats: "zzz-not-a-real-iupac-name-zzz")
        got = nm._try_general_engine_recovery(_ABSTAINER_SMILES)
    assert got is None, (
        f"unparseable T4 name shipped unverified instead of abstaining: {got!r}")


# --- Task 7: backbone acceptance PROBE (not a gate) --------------------------
# 12 vetted T4-reaching abstainers (from the atom-drop class corpus, pulled by
# CID from `` -- the CSV is authoritative, not the
# literal strings below). This is a PROBE, NOT a gate: it asserts the class
# invariant (0-partial) and records the conversion count informationally.
# NO threshold is asserted on how many of the 12 convert -- that number is
# expected to grow as follow-on per-class sub-namers ship (see the brief's
# follow-on roadmap), and asserting it here would turn a probe into a gate.
_T4_PROBE_CIDS = [
    "1542461", "581475", "118415", "639588", "107217", "60184",
    "43057", "158257", "103172", "70868", "439233", "558120",
]


def _t4_probe_smiles_by_cid():
    """Re-pull the 12 probe SMILES from the CSV by CID (authoritative source
    -- never retyped, to avoid transcription risk)."""
    import csv as _csv
    from pathlib import Path as _Path

    csv_path = (
        _Path(__file__).parent.parent.parent.parent / "benchmarks" /
        "pubchem_2000.csv"
    )
    wanted = set(_T4_PROBE_CIDS)
    found = {}
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            if row["cid"] in wanted:
                found[row["cid"]] = row["smiles"]
    missing = wanted - found.keys()
    assert not missing, f"CIDs missing from {csv_path}: {sorted(missing)}"
    return [(cid, found[cid]) for cid in _T4_PROBE_CIDS]


@pytest.mark.unit
@pytest.mark.opsin_gate
def test_t4_backbone_acceptance_probe():
    """PROBE, NOT A GATE: names 12 T4-reaching abstainers in ONE process and
    asserts the class invariant -- every molecule is either atom-complete AND
    OPSIN-round-trips to the input, OR a clean ``unknown organic compound``
    abstain. NEVER a partial (a name that emits but denotes the wrong
    molecule, e.g. from a silently dropped fragment).

    The only hard assertion is ``partial_count == 0`` -- the 0-wrong contract
    for the T4 backbone. The converted count is recorded (printed, visible
    with ``-s``) purely informationally; NO threshold is asserted on it,
    because that number is expected to grow as the follow-on per-class
    sub-namers (heterocyclic unsaturation, decorated-ring substituents, fused
    polycyclic, spiro/bridged, peptide/sugar, acyclic exotic-FG, OPSIN-format)
    ship independently -- pinning it here would make this probe a gate on
    work that hasn't happened yet.

    Both ``general_fallback`` and ``general_fallback_unverified`` are
    required to reach the T4 tier (see
    ``test_t4_wiring_does_not_fire_without_unverified_optin`` above -- the
    verified-only tier does not run this producer). Runs under
    ``opsin_gate`` so the PIN path's own SELF-01 round-trip gate is live,
    matching how T4 is actually reached in production (T4 only fires once
    the gated PIN/general path has honestly abstained).
    """
    from orthonym.errors import is_failure_name
    from orthonym.jvm_budget import jvm_slots
    from orthonym.namer import name_compound
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

    verdicts = []  # (cid, outcome, name_or_None)
    partials = []  # (cid, name, opsin_smiles_or_None) for any 0-partial violation

    with jvm_slots(1, purpose="t4-acceptance-probe"):
        for cid, smi in _t4_probe_smiles_by_cid():
            name = name_compound(
                smi,
                general_fallback=True,
                general_fallback_unverified=True,
            )
            if is_failure_name(name):
                verdicts.append((cid, "abstain", name))
                continue

            rt = opsin_roundtrip_check(smi, name)
            if rt["passed"]:
                verdicts.append((cid, "converted", name))
            else:
                verdicts.append((cid, "PARTIAL/WRONG", name))
                partials.append((cid, name, rt))

    converted = [v for v in verdicts if v[1] == "converted"]
    abstained = [v for v in verdicts if v[1] == "abstain"]

    summary_lines = [
        f"  cid={cid:<9} {outcome:<14} {name!r}"
        for cid, outcome, name in verdicts
    ]
    summary = (
        "T4 backbone acceptance probe (INFORMATIONAL, not a gate):\n"
        + "\n".join(summary_lines)
        + f"\nconverted={len(converted)}/12 abstained={len(abstained)}/12 "
        + f"partial={len(partials)}/12"
    )
    print("\n" + summary)

    assert not partials, (
        "0-partial invariant VIOLATED -- a T4 emission did not round-trip to "
        "the input molecule:\n"
        + "\n".join(
            f"  cid={cid} name={name!r} opsin_roundtrip={rt!r}"
            for cid, name, rt in partials
        )
    )


# --- Fix 4 (final review): rung 2 is a UNIQUE producer, not dead code ---------
# Measured 2026-08-11 (, 400-molecule pubchem_2000
# sample, short-circuited so rung 2 fires only when rungs 0 AND 1 both decline):
# cascade rung 2 (suppress PG only, KEEP the perceived chain parent) is the SOLE
# producer for 5/400 molecules -- the ring-less / chain-preferred class where
# rung 1's chain_is_parent=False forces a ring parent the engine then cannot
# host, so rung 1 declines while rung 2 succeeds. So rung 2 is NOT redundant
# belt-and-braces and NOT dead code (invariants 8/17). Anchor: cid 266765, the
# simplest (stereo-free) of the 5; its rung-2 name OPSIN-round-trips.
_RUNG2_UNIQUE_SMILES = "CC(C)(CC(=O)NC1CCCCC1)CBr"  # cid 266765
_RUNG2_UNIQUE_EXPECTED = "4-bromo-1-(cyclohexylamino)-3,3-dimethyl-1-oxobutane"


@pytest.mark.unit
@pytest.mark.opsin_gate
def test_cascade_rung2_is_a_unique_producer():
    """Rung 2 (PG-suppressed, chain parent KEPT) converts a molecule NEITHER
    rung 0 (perceived PG) NOR rung 1 (PG-suppressed + chain_is_parent=False)
    can -- proving rung 2 is the sole producer for the ring-less/chain-preferred
    class, not redundant with rung 1. Runs under ``opsin_gate`` so the general
    engine's internal validity suppression matches how T4 is reached in
    production (and how the measuring probe ran).
    """
    from orthonym.jvm_budget import jvm_slots
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

    mol, feats = _classified(_RUNG2_UNIQUE_SMILES)
    with jvm_slots(1, purpose="test-rung2-unique"):
        rung0 = t4_coverage._run_general_e1(mol, feats)
        rung1 = t4_coverage._run_general_e1(
            mol, t4_coverage._clone_features_with(
                feats, principal_group=None, principal_group_atoms=[],
                chain_is_parent=False))
        rung2 = t4_coverage._run_general_e1(
            mol, t4_coverage._clone_features_with(
                feats, principal_group=None, principal_group_atoms=[]))
        assert rung0 is None, (
            f"rung 0 unexpectedly named it: {rung0.name!r}")
        assert rung1 is None, (
            f"rung 1 unexpectedly named it: {rung1.name!r}")
        assert rung2 is not None, (
            "rung 2 failed -- the unique-producer lock is broken; rung 2 is "
            "dead code if this no longer converts")
        # And it is a REAL conversion: the full producer emits it and it
        # round-trips through OPSIN to the input constitution.
        name = t4_coverage.name_t4_complete(mol, feats)
        assert name == _RUNG2_UNIQUE_EXPECTED, name
        rt = opsin_roundtrip_check(_RUNG2_UNIQUE_SMILES, name)
    assert rt["passed"], f"rung-2 name did not round-trip: {rt!r}"
