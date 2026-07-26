"""v29 Phase 2 T5: the ring parent emits one binding per token it spells.

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

Harness note: the conftest force-disables the production SELF-01 gate
suite-wide, so these tests drive the PRODUCER directly (the established
pattern in ``tests/unit/rules/test_v26_p2_aromatic_vonbaeyer.py``) and assert
on bindings + ``verify_spine``, never on a round trip.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.general_engine import name_general_ring
from orthonym.namer import Orthonym
from orthonym.validation.binding_spine import (BindingKind, BindingSpine,
                                                verify_spine)
from orthonym.validation.name_morphemes import token_arity

pytestmark = pytest.mark.unit


def _emit(smiles):
    """Run the ring producer and return (mol, result). Complete tier."""
    nm = Orthonym(style="pin", general_fallback=True,
                   allow_aromatic_general=True)
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    feats = nm._perceive(mol, smiles, Chem.MolToSmiles(mol))
    nm._classify(feats)
    return mol, name_general_ring(mol, feats, allow_aromatic_general=True)


def _roles(result, role):
    return [b for b in result.bindings if b.role == role]


OXA_CAGE = "C1CC2CCC1O2"          # 7-oxabicyclo[2.2.1]heptane
DIAZA_CAGE = "C1CC2CCC1NN2"       # 7,8-diazabicyclo[2.2.2]octane
MIXED_CAGE = "C1CC2CCC1ON2"       # 7-oxa-8-azabicyclo[2.2.2]octane
NORBORNANE = "C1CC2CCC1C2"        # bicyclo[2.2.1]heptane (carbocyclic control)
NORBORNENE = "C1=CC2CCC1C2"       # bicyclo[2.2.1]hept-2-ene (unsaturated)


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
