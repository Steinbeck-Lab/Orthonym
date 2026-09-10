""" a phase T5: the ring parent emits one binding per token it spells.

Before this phase the general ring producer appended a SINGLE binding whose
token was ``cage.descriptor`` alone -- ``'bicyclo[2.2.1]'`` -- claiming every
cage atom, while the spelled word is ``hetero_prefix + descriptor + stem +
(ene/yne block)``. Two audit-mode consequences, both measured:

* the token had no chain stem, so the arity oracle could not decide what it
  spells (``ARITY_UNVERIFIED``); on a bare cage that was the ONLY textual
  token, so the whole proof reported ``PROOF_UNSUBSTANTIATED``; and
* the replacement morphemes were bound to nothing, so P5 reported the leftover
  ``'xa'`` of ``7-oxa`` as ``UNBOUND_MORPHEME``.

Neither was a wrongness bug -- the names are correct -- which is exactly why
they mattered: they are false alarms standing between the spine and
``binding_proof='enforce'``.

Reach limit, stated so these tests are not over-read: both ``record_spine``
sites sit behind the general engine, which is opt-in and default OFF, so no
default-path emission records a spine at all. This is a correctness
prerequisite, not a live-defect fix.

Harness note: the conftest force-disables the production gate
suite-wide, so these tests drive the PRODUCER directly (the established
pattern in ``tests/unit/rules/test_v26_p2_aromatic_vonbaeyer.py``) and assert
on bindings + ``verify_spine``, never on a round trip.
"""
import dataclasses

import pytest
from rdkit import Chem

import orthonym.assembly.general_engine as general_engine
from orthonym.assembly.general_engine import (name_general_ring,
                                               name_general_spiro)
from orthonym.namer import Orthonym
from orthonym.rules.vonbaeyer_universal import (RingAnalysis, SpiroSystem,
                                                 UniversalCage,
                                                 analyze_cage_universal,
                                                 analyze_spiro_universal)
from orthonym.validation.binding_spine import (BindingKind, BindingSpine,
                                                verify_spine)
from orthonym.validation.name_morphemes import token_arity

pytestmark = pytest.mark.unit


def _run(producer, smiles):
    """Run a general-engine ring producer and return (mol, result)."""
    nm = Orthonym(style="pin", general_fallback=True,
                   allow_aromatic_general=True)
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    feats = nm._perceive(mol, smiles, Chem.MolToSmiles(mol))
    nm._classify(feats)
    return mol, producer(mol, feats, allow_aromatic_general=True)


def _emit(smiles):
    """Run the von-Baeyer CAGE producer. Complete tier."""
    return _run(name_general_ring, smiles)


def _emit_spiro(smiles):
    """Run the SPIRO producer -- the sibling that shares the emission tail.

    ``name_general_ring`` will not reach a spiro system (the cage analyzer
    refuses ``<2 bridgeheads`` by design), so the spiro half of the shared tail
    has to be driven through its own entry point.
    """
    return _run(name_general_spiro, smiles)


def _roles(result, role):
    return [b for b in result.bindings if b.role == role]


OXA_CAGE = "C1CC2CCC1O2"          # 7-oxabicyclo[2.2.1]heptane
DIAZA_CAGE = "C1CC2CCC1NN2"       # 7,8-diazabicyclo[2.2.2]octane
MIXED_CAGE = "C1CC2CCC1ON2"       # 7-oxa-8-azabicyclo[2.2.2]octane
NORBORNANE = "C1CC2CCC1C2"        # bicyclo[2.2.1]heptane (carbocyclic control)
NORBORNENE = "C1=CC2CCC1C2"       # bicyclo[2.2.1]hept-2-ene (unsaturated)

# The SPIRO sibling. ``SpiroSystem`` and ``UniversalCage`` are two analysis
# forms feeding ONE emission tail, so every property asserted of the cage above
# has to hold here too -- the field contract was previously stated twice and
# drifted twice (see ``RingAnalysis``).
OXA_SPIRO = "C1CCC2(C1)OCCCC2"    # 6-oxaspiro[4.5]decane
DIOXA_SPIRO = "O1CCC2(C1)COCC2"   # 2,7-dioxaspiro[4.4]nonane (one morpheme x2)
OXA_THIA_SPIRO = "O1CCC2(C1)CSCC2"  # 2-oxa-7-thiaspiro[4.4]nonane (two morphemes)
SPIRO_CARBO = "C1CCC2(C1)CCCCC2"  # spiro[4.5]decane (carbocyclic control)
# SUBSTITUTED hetero spiros. These are not decoration: the spiro analyzer works
# on a ring-only SUBMOL, so ``hetero_per_atom`` has to be mapped back to original
# indices -- and in an all-ring molecule the two index spaces coincide, which
# makes that mapping untestable. A non-ring substituent shifts them apart (the
# methyl takes index 0, so every ring atom's original index is its submol index
# + 1) and is the only fixture shape under which an unmapped index is visible.
SUBST_OXA_SPIRO = "CC1CCC2(C1)OCCCC2"     # 2-methyl-6-oxaspiro[4.5]decane
SUBST_OXA_THIA_SPIRO = "CC1COC2(CSCC2)C1"  # 3-methyl-1-oxa-7-thiaspiro[4.4]nonane
# The nitrogen twin of DIOXA_SPIRO, same skeleton, same locants -- the CANARY
# subject below (P5's aza blind spot: see that section).
DIAZA_SPIRO = "N1CCC2(C1)CNCC2"   # 2,7-diazaspiro[4.4]nonane


# --------------------------------------------------------------------------
# The parent token now carries the stem, so its arity is decidable
# --------------------------------------------------------------------------
def test_parent_token_is_descriptor_plus_stem():
    """The parent token must be ``descriptor + stem``, not the descriptor alone.

    ``'bicyclo[2.2.1]'`` on its own is arity-undecidable ("descriptor not
    followed by a chain stem"); with the stem the oracle confirms 7 skeletal
    atoms two independent ways (descriptor arithmetic and the stem).
    """
    mol, res = _emit(OXA_CAGE)
    assert res is not None
    parents = _roles(res, 'parent')
    assert len(parents) == 1
    assert parents[0].token == "bicyclo[2.2.1]hept"
    # The token really is a span of the emitted name.
    assert parents[0].token in res.name


def test_parent_token_arity_is_confident_and_matches_the_claim():
    """P6 must be able to corroborate the parent, and must agree with it."""
    mol, res = _emit(OXA_CAGE)
    parent = _roles(res, 'parent')[0]
    estimate = token_arity(parent.token, BindingKind.PARENT)
    assert estimate.confident, estimate.basis
    assert estimate.heavy_atoms == len(parent.atom_ids) == 7


def test_parent_claims_every_cage_atom_including_the_heteroatom():
    """The stem counts skeletal POSITIONS, so the parent keeps all of them.

    A replacement prefix says which element sits at a position the stem has
    already counted -- it does not add an atom. Handing the heteroatom to the
    'oxa' binding instead would make the parent claim 6 while its morphemes
    spell 7, i.e. an ``ARITY_MISMATCH`` *error* against a correct name.
    """
    mol, res = _emit(OXA_CAGE)
    parent = _roles(res, 'parent')[0]
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    assert set(parent.atom_ids) == ring_atoms
    oxygen = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'O')
    assert oxygen in parent.atom_ids


# --------------------------------------------------------------------------
# Replacement morphemes are bound, and claim no atom
# --------------------------------------------------------------------------
def test_replacement_morpheme_is_bound_claiming_zero_atoms():
    mol, res = _emit(OXA_CAGE)
    reps = _roles(res, 'replacement')
    assert len(reps) == 1
    assert reps[0].token == "oxa"
    assert reps[0].atom_ids == ()


def test_one_replacement_binding_per_heteroatom_not_per_morpheme():
    """``diaza`` is spelled once but counts twice; P4 counts multiplicands.

    One binding for the two nitrogens would trade the old over-claim for
    ``MULTIPLICITY_MISMATCH`` ("spells 'aza' 2 times but 1 binding claims it").
    """
    mol, res = _emit(DIAZA_CAGE)
    reps = _roles(res, 'replacement')
    assert [b.token for b in reps] == ["aza", "aza"]
    assert all(b.atom_ids == () for b in reps)


def test_distinct_morphemes_each_get_their_own_binding():
    mol, res = _emit(MIXED_CAGE)
    reps = _roles(res, 'replacement')
    assert sorted(b.token for b in reps) == ["aza", "oxa"]


# --------------------------------------------------------------------------
# The role must survive the flat adapter
# --------------------------------------------------------------------------
def test_replacement_role_is_not_coerced_to_prefix():
    """``'replacement'`` must map to ``BindingKind.REPLACEMENT``.

    The flat adapter coerces any role it cannot map to PREFIX and records the
    raw string in ``legacy_role_coerced``. A non-empty list here means the
    producer is emitting a role the spine vocabulary does not accept.
    """
    mol, res = _emit(OXA_CAGE)
    spine = BindingSpine.from_token_bindings(res.bindings)
    assert spine.legacy_role_coerced == ()
    kinds = {b.token: b.kind for b in spine.walk()}
    assert kinds["oxa"] is BindingKind.REPLACEMENT
    assert kinds["bicyclo[2.2.1]hept"] is BindingKind.PARENT


# --------------------------------------------------------------------------
# Whole-proof: P1/P2 stay clean and the false alarms are gone
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles", [OXA_CAGE, DIAZA_CAGE, MIXED_CAGE,
                                    NORBORNANE, NORBORNENE])
def test_audit_has_no_error_or_warning_findings(smiles):
    """The whole point: correct names must stop raising proof findings.

    ``ARITY_UNVERIFIED`` on a bare replacement morpheme is expected and is
    ``info`` -- the oracle refuses a prefix that qualifies no skeleton by
    design. Everything above info level must be gone.
    """
    mol, res = _emit(smiles)
    assert res is not None
    proof = verify_spine(mol, BindingSpine.from_token_bindings(res.bindings),
                         res.name, mode='audit')
    loud = [(f.code, f.detail) for f in proof.findings
            if f.severity in ('error', 'warn')]
    assert loud == [], loud
    assert proof.ok


@pytest.mark.parametrize("smiles", [OXA_CAGE, DIAZA_CAGE, MIXED_CAGE,
                                    NORBORNANE, NORBORNENE])
def test_p1_and_p2_stay_clean(smiles):
    """Atom partition and bond totality must not regress from the split."""
    mol, res = _emit(smiles)
    proof = verify_spine(mol, BindingSpine.from_token_bindings(res.bindings),
                         res.name, mode='audit')
    codes = {f.code for f in proof.findings}
    for code in ("ATOM_DOUBLE_BOUND", "ATOM_UNBOUND", "ATOM_PHANTOM",
                 "BOND_UNCLAIMED", "BOND_DOUBLE_CLAIMED",
                 "BOND_AMBIGUOUS_LINKAGE"):
        assert code not in codes, (code, [f.detail for f in proof.findings])


def test_no_longer_reports_unbound_replacement_morpheme():
    """The specific P5 finding this task removes."""
    mol, res = _emit(OXA_CAGE)
    proof = verify_spine(mol, BindingSpine.from_token_bindings(res.bindings),
                         res.name, mode='audit')
    assert "UNBOUND_MORPHEME" not in {f.code for f in proof.findings}


def test_no_longer_reports_proof_unsubstantiated():
    """A bare cage used to corroborate nothing at all; now the parent does."""
    mol, res = _emit(NORBORNANE)
    proof = verify_spine(mol, BindingSpine.from_token_bindings(res.bindings),
                         res.name, mode='audit')
    assert "PROOF_UNSUBSTANTIATED" not in {f.code for f in proof.findings}
    assert proof.stats["arity_confident_atom_frac"] == 1.0


# --------------------------------------------------------------------------
# The unsaturation block spells bonds, so it is bound by no token
# --------------------------------------------------------------------------
def test_unsaturation_block_is_not_folded_into_the_parent_token():
    """``hept-2-ene``: the parent token stops at the stem."""
    mol, res = _emit(NORBORNENE)
    parent = _roles(res, 'parent')[0]
    assert res.name == "bicyclo[2.2.1]hept-2-ene"
    assert parent.token == "bicyclo[2.2.1]hept"
    assert "ene" not in parent.token
    # No binding claims the ene block, and no binding claims zero-atom bonds
    # other than the replacement morphemes (none here).
    assert _roles(res, 'replacement') == []


# ==========================================================================
# The SPIRO sibling. ``hetero_per_atom`` was added to ``UniversalCage`` only,
# so the stem-upgrade half of this fix reached the spiro producer through the
# shared tail while the replacement-binding half silently did not: every
# hetero spiro kept the exact ``UNBOUND_MORPHEME`` finding this task removes.
#
# Which morphemes are OBSERVABLE through P5 is a lexicon accident worth knowing
# when reading these fixtures: P5 strips "glue" greedily from a residue run, and
# ``_STEREO_WORDS`` contributes the bare letters r/s/e/z, so ``aza`` happens to
# strip clean (a + z + a) and raises nothing even when unbound, while ``oxa``
# leaves ``'xa'`` and ``thia`` leaves ``'thia'``. The fixtures below therefore
# use oxa/thia, which actually exercise the finding; an aza-only spiro would
# pass this file before AND after the fix.
# ==========================================================================
def test_spiro_replacement_morpheme_is_bound_claiming_zero_atoms():
    mol, res = _emit_spiro(OXA_SPIRO)
    assert res is not None
    assert res.name == "6-oxaspiro[4.5]decane"
    reps = _roles(res, 'replacement')
    assert len(reps) == 1
    assert reps[0].token == "oxa"
    # ZERO atoms, exactly as on the cage path: a replacement prefix says which
    # element sits at a position the stem already counted.
    assert reps[0].atom_ids == ()


def test_spiro_parent_still_claims_every_ring_atom():
    """Splitting the morpheme off must not shrink the parent's claim.

    ``token_arity('spiro[4.5]dec')`` is confidently 10 skeletal POSITIONS,
    blind to which element occupies them. Handing the oxygen to the ``oxa``
    binding would leave the parent claiming 9 against a text that spells 10 --
    an ``ARITY_MISMATCH`` at severity *error* on a correct name.
    """
    mol, res = _emit_spiro(OXA_SPIRO)
    parents = _roles(res, 'parent')
    assert len(parents) == 1
    assert parents[0].token == "spiro[4.5]dec"
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    assert set(parents[0].atom_ids) == ring_atoms
    oxygen = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'O')
    assert oxygen in parents[0].atom_ids
    estimate = token_arity(parents[0].token, BindingKind.PARENT)
    assert estimate.confident, estimate.basis
    assert estimate.heavy_atoms == len(parents[0].atom_ids) == 10


def test_spiro_one_binding_per_heteroatom_not_per_morpheme():
    """``dioxa`` is written once and spells ``oxa`` twice (P4 counts
    multiplicands), so two oxygens need two bindings."""
    mol, res = _emit_spiro(DIOXA_SPIRO)
    assert res.name == "2,7-dioxaspiro[4.4]nonane"
    reps = _roles(res, 'replacement')
    assert [b.token for b in reps] == ["oxa", "oxa"]
    assert all(b.atom_ids == () for b in reps)


def test_spiro_distinct_morphemes_each_get_their_own_binding():
    mol, res = _emit_spiro(OXA_THIA_SPIRO)
    assert res.name == "2-oxa-7-thiaspiro[4.4]nonane"
    reps = _roles(res, 'replacement')
    assert sorted(b.token for b in reps) == ["oxa", "thia"]
    assert all(b.atom_ids == () for b in reps)


def test_spiro_replacement_role_is_not_coerced_to_prefix():
    mol, res = _emit_spiro(OXA_SPIRO)
    spine = BindingSpine.from_token_bindings(res.bindings)
    assert spine.legacy_role_coerced == ()
    kinds = {b.token: b.kind for b in spine.walk()}
    assert kinds["oxa"] is BindingKind.REPLACEMENT
    assert kinds["spiro[4.5]dec"] is BindingKind.PARENT


@pytest.mark.parametrize("smiles", [OXA_SPIRO, DIOXA_SPIRO, OXA_THIA_SPIRO,
                                    SUBST_OXA_SPIRO, SUBST_OXA_THIA_SPIRO,
                                    SPIRO_CARBO])
def test_spiro_audit_has_no_error_or_warning_findings(smiles):
    """Correct spiro names must raise no proof finding above ``info``.

    ``SPIRO_CARBO`` is the carbocyclic control: it has no morpheme to bind and
    must stay at zero findings, so a fix that manufactured bindings would show
    up here rather than only on the hetero cases.
    """
    mol, res = _emit_spiro(smiles)
    assert res is not None
    proof = verify_spine(mol, BindingSpine.from_token_bindings(res.bindings),
                         res.name, mode='audit')
    loud = [(f.code, f.detail) for f in proof.findings
            if f.severity in ('error', 'warn')]
    assert loud == [], loud
    assert proof.ok


@pytest.mark.parametrize("smiles", [OXA_SPIRO, DIOXA_SPIRO, OXA_THIA_SPIRO,
                                    SUBST_OXA_SPIRO, SUBST_OXA_THIA_SPIRO,
                                    SPIRO_CARBO])
def test_spiro_p1_and_p2_stay_clean(smiles):
    mol, res = _emit_spiro(smiles)
    proof = verify_spine(mol, BindingSpine.from_token_bindings(res.bindings),
                         res.name, mode='audit')
    codes = {f.code for f in proof.findings}
    for code in ("ATOM_DOUBLE_BOUND", "ATOM_UNBOUND", "ATOM_PHANTOM",
                 "BOND_UNCLAIMED", "BOND_DOUBLE_CLAIMED",
                 "BOND_AMBIGUOUS_LINKAGE"):
        assert code not in codes, (code, [f.detail for f in proof.findings])


@pytest.mark.parametrize("smiles,residues", [
    (OXA_SPIRO, ['xa']),
    (DIOXA_SPIRO, ['xa']),
    (OXA_THIA_SPIRO, ['xa', 'thia']),
    (SUBST_OXA_SPIRO, ['xa']),
    (SUBST_OXA_THIA_SPIRO, ['xa', 'thia']),
])
def test_spiro_no_longer_reports_unbound_replacement_morpheme(smiles, residues):
    """The specific P5 finding, and the residue text it used to name."""
    mol, res = _emit_spiro(smiles)
    proof = verify_spine(mol, BindingSpine.from_token_bindings(res.bindings),
                         res.name, mode='audit')
    assert "UNBOUND_MORPHEME" not in {f.code for f in proof.findings}
    for residue in residues:
        assert residue not in proof.stats["residue_runs"], proof.stats


# ==========================================================================
# CANARY: P5's aza blind spot, pinned as it is TODAY.
#
# `` recorded (as a comment only, above at DIAZA_CAGE) that this
# defect class is invisible for nitrogen: P5's glue lexicon lets 'aza' strip
# cleanly ('a' + 'z' + 'a', all three in ``_GLUE_MORPHEMES`` -- 'a' is
# left/right connective glue, 'z' rides in on ``_STEREO_WORDS``), so an
# aza-only fixture reports zero findings whether or not the replacement
# binding exists. A comment is not a test: nobody would notice if a future
# change to ``_STEREO_WORDS`` or the lexicon silently widened or closed this
# hole. These two tests turn the comment into a living assertion.
#
# ``test_canary_*`` deliberately REMOVES the 'aza' replacement bindings the
# real producer emits (reproducing the exact pre-fix shape -- a parent
# claiming every atom with NO binding corroborating the replacement morpheme)
# and asserts P5 stays silent. If a lexicon change ever makes this fail, that
# is GOOD NEWS (the blind spot closed) -- re-derive the test against the new
# behaviour, do not just delete it or loosen the assertion back to green.
#
# ``test_contrast_*`` is the same mutilation on the oxygen twin of the exact
# same skeleton and locants (``DIOXA_SPIRO``/``DIAZA_SPIRO`` differ only in
# element), and IS caught -- P5 reports ``UNBOUND_MORPHEME`` for the leftover
# 'xa'. The contrast is the point: identical defect shape, element-dependent
# detectability, because the oracle's blind spot is lexical (which residue
# letters happen to be glue words) and not structural.
# ==========================================================================
def test_canary_aza_replacement_binding_omission_is_invisible_to_p5():
    """PINS the known blind spot: an unbound 'aza' morpheme raises nothing.

    Do not "fix" this test by making it pass some other way if a real aza
    spiro regresses -- that is the false-negative this test exists to record.
    It should only ever fail because the LEXICON changed, in which case
    re-derive it (and celebrate: the blind spot just closed).
    """
    mol, res = _emit_spiro(DIAZA_SPIRO)
    assert res is not None
    assert res.name == "2,7-diazaspiro[4.4]nonane"
    reps = _roles(res, 'replacement')
    assert [b.token for b in reps] == ["aza", "aza"]

    # Deliberately drop the replacement bindings -- the exact pre-fix
    # shape (parent-only spine) reproduced by construction, not by reverting
    # source. The parent alone still claims every atom, so P1/P2/P3/P7 stay
    # clean; only P5 (name residue) is under test here.
    parent_only = [b for b in res.bindings if b.role != 'replacement']
    spine = BindingSpine.from_token_bindings(parent_only)
    proof = verify_spine(mol, spine, res.name, mode='audit')
    loud = [(f.code, f.detail) for f in proof.findings
            if f.severity in ('error', 'warn')]
    assert loud == [], (
        "canary tripped: P5 now sees the unbound 'aza' -- the lexicon "
        f"changed and this test must be re-derived, not silenced: {loud}"
    )
    assert "UNBOUND_MORPHEME" not in {f.code for f in proof.findings}


def test_contrast_oxa_replacement_binding_omission_IS_visible_to_p5():
    """The oxygen twin of the canary above: the SAME mutilation IS caught.

    Same skeleton, same locants (``DIOXA_SPIRO`` vs ``DIAZA_SPIRO`` differ
    only in element), same removed-bindings construction. 'oxa' does not
    strip through the glue lexicon the way 'aza' does ('o' consumes as a
    connective vowel, but the trailing 'xa' matches no glue morpheme), so P5
    reports the leftover -- proving the canary's silence above is a genuine
    lexicon accident, not a broken test harness.
    """
    mol, res = _emit_spiro(DIOXA_SPIRO)
    assert res is not None
    assert res.name == "2,7-dioxaspiro[4.4]nonane"
    reps = _roles(res, 'replacement')
    assert [b.token for b in reps] == ["oxa", "oxa"]

    parent_only = [b for b in res.bindings if b.role != 'replacement']
    spine = BindingSpine.from_token_bindings(parent_only)
    proof = verify_spine(mol, spine, res.name, mode='audit')
    codes = {f.code for f in proof.findings}
    assert "UNBOUND_MORPHEME" in codes, (
        "expected the visible half of the contrast to trip -- if it no "
        "longer does, the glue lexicon widened enough to swallow 'xa' too, "
        "and the canary above needs re-checking against the same change"
    )
    assert 'xa' in proof.stats["residue_runs"], proof.stats


# ==========================================================================
# Anti-drift: the contract, and the requirement that each form FILLS it.
#
# ``RingAnalysis`` makes the field list impossible to diverge (both forms
# inherit it and add nothing), but declaring a field is not populating one --
# a third analysis form could inherit ``hetero_per_atom`` and still report .
# These two tests are the other half of the guarantee and are what fails if
# either happens again.
# ==========================================================================
def test_both_ring_analysis_forms_share_one_field_contract():
    """The shape is INHERITED, never re-declared per form."""
    for form in (UniversalCage, SpiroSystem):
        assert issubclass(form, RingAnalysis)
        # Adds no field of its own: the subclass's field list IS the base's.
        assert ([f.name for f in dataclasses.fields(form)]
                == [f.name for f in dataclasses.fields(RingAnalysis)])
    assert 'hetero_per_atom' in {f.name for f in dataclasses.fields(RingAnalysis)}


@pytest.mark.parametrize("analyze,smiles", [
    (analyze_cage_universal, OXA_CAGE),
    (analyze_cage_universal, DIAZA_CAGE),
    (analyze_cage_universal, MIXED_CAGE),
    (analyze_cage_universal, NORBORNANE),
    (analyze_spiro_universal, OXA_SPIRO),
    (analyze_spiro_universal, DIOXA_SPIRO),
    (analyze_spiro_universal, OXA_THIA_SPIRO),
    # The two fixtures whose submol and original index spaces DIFFER; without
    # them the index assertion below is vacuous (verified by mutation).
    (analyze_spiro_universal, SUBST_OXA_SPIRO),
    (analyze_spiro_universal, SUBST_OXA_THIA_SPIRO),
    (analyze_spiro_universal, SPIRO_CARBO),
])
def test_every_ring_analysis_form_decomposes_its_replacement_prefix(
        analyze, smiles):
    """A spelled replacement prefix must always come with its decomposition.

    ``hetero_prefix`` non-empty and ``hetero_per_atom`` empty is precisely the
    defect state: morphemes in the name that no binding can account for. The
    reported atoms must also be the real heteroatoms, in ORIGINAL indices, with
    the morpheme the prefix actually spells -- which is what catches an index
    space left unmapped (the spiro analyzer works on a ring-only submol).
    """
    mol = Chem.MolFromSmiles(smiles)
    analysis = analyze(mol)
    assert analysis is not None, smiles
    if not analysis.hetero_prefix:
        assert analysis.hetero_per_atom == ()
        return
    assert analysis.hetero_per_atom, (smiles, analysis.hetero_prefix)
    hetero_atoms = {a.GetIdx() for a in mol.GetAtoms()
                    if a.GetSymbol() != 'C' and a.IsInRing()}
    assert {idx for idx, _m in analysis.hetero_per_atom} == hetero_atoms
    for idx, morpheme in analysis.hetero_per_atom:
        assert morpheme in analysis.hetero_prefix, (idx, morpheme)


# ==========================================================================
# Defence-in-depth: ``_ring_parent_bindings`` reads ``hetero_per_atom`` via
# ``getattr(cage, 'hetero_per_atom', )`` rather than a hard attribute
# access. The review verdict: sufficient for present scope (``RingAnalysis``
# makes the field un-omittable by either real form, and the failure mode
# degrades to the pre-af0d7262 audit finding, never a wrong name), but a
# THIRD analysis form that inherits the field and never populates it should
# be noisy, not silent. These tests exercise that ``logger.warning``.
# ==========================================================================
def test_ring_parent_bindings_warns_when_hetero_prefix_outruns_hetero_per_atom(
        caplog):
    """Simulate the un-reachable-today gap and confirm it is now LOGGED.

    Neither ``UniversalCage`` nor ``SpiroSystem`` can produce this state (see
    ``test_every_ring_analysis_form_decomposes_its_replacement_prefix``
    above), so it is manufactured with ``dataclasses.replace`` on a REAL
    ``SpiroSystem`` -- same type, ``hetero_per_atom`` forced back to ````
    -- which is exactly "inherits the field, never fills it". ``name`` and
    ``parent_block`` are captured from the actual production call via a trace
    rather than reconstructed by hand, so this exercises the real argument
    shapes, not a guess at them.
    """
    captured = {}
    original = general_engine._ring_parent_bindings

    def _spy(cage, name, parent_block):
        captured['cage'] = cage
        captured['name'] = name
        captured['parent_block'] = parent_block
        return original(cage, name, parent_block)

    general_engine._ring_parent_bindings = _spy
    try:
        mol, res = _emit_spiro(OXA_SPIRO)
    finally:
        general_engine._ring_parent_bindings = original
    assert res is not None
    assert captured['cage'].hetero_prefix, "OXA_SPIRO must exercise the hetero path"

    starved = dataclasses.replace(captured['cage'], hetero_per_atom=())

    with caplog.at_level("WARNING", logger="orthonym.assembly.general_engine"):
        bindings = general_engine._ring_parent_bindings(
            starved, captured['name'], captured['parent_block'])

    warnings = [r.getMessage() for r in caplog.records
                if r.levelname == "WARNING"]
    assert any("hetero_per_atom" in msg and "hetero_prefix" in msg
               for msg in warnings), warnings
    # Degrades to the pre-fix audit gap (no replacement binding) -- never a
    # crash, never a fabricated binding.
    assert [b.role for b in bindings] == ['parent']


def test_ring_parent_bindings_stays_silent_when_the_fields_agree(caplog):
    """Contrast: the real (unstarved) producer path must NOT warn.

    Guards against the warning test above passing vacuously because the
    warning fires unconditionally rather than on the specific mismatch.
    """
    with caplog.at_level("WARNING", logger="orthonym.assembly.general_engine"):
        mol, res = _emit_spiro(OXA_SPIRO)
    assert res is not None
    warnings = [r.getMessage() for r in caplog.records
                if r.levelname == "WARNING"]
    assert not any("hetero_per_atom" in msg for msg in warnings), warnings
