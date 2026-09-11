"""v47 P1 — the ring cis/trans COMPLETENESS predicate + reclaim.

_ring_cistrans_is_complete(mol, a, b) is True iff flipping BOTH ring stereocentres
yields the same molecule (up,up)≅(down,down) -- i.e. cis/trans loses no absolute
information. This is the SOUND replacement for the CIP-case gate (refuted) and the
arc-symmetry gate (refuted: a stereo-blind rank match is not a 3D automorphism).

Spec: internal notes
"""
import re
import pytest
from rdkit import Chem
from orthonym import Orthonym
from orthonym.assembly.universal_substituent import (
    _ring_cistrans_is_complete,
    _rings_jointly_complete,
)
from orthonym.cli import _emit_tier_flags
from orthonym.perception.stereo import assign_stereochemistry
from orthonym.validation.reconstruct import verify_or_none


def _centres_of_first_sat_ring(smi):
    """Return (mol, a, b): the 2 stereocentres of the first small saturated ring
    with exactly 2 same-ring _CIPCode atoms. Helper for the predicate test."""
    mol = Chem.MolFromSmiles(smi)
    assign_stereochemistry(mol)
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        n = len(ring)
        if not (3 <= n <= 6):
            continue
        if any(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            continue
        cen = [a for a in ring if mol.GetAtomWithIdx(a).HasProp("_CIPCode")]
        if len(cen) == 2 and ri.AreAtomsInSameRing(cen[0], cen[1]):
            return mol, cen[0], cen[1]
    raise AssertionError(f"no 2-centre saturated ring in {smi}")


# (smiles, expected completeness) — the ring under test is the small saturated one
PREDICATE_CASES = [
    # symmetric rings: cis/trans is complete -> True
    ("OC(=O)[C@H]1C[C@@H](C(=O)NC)C1", True),          # 1,3-cyclobutane (meso-type)
    ("CNC(=O)[C@H]1CC[C@@H](O)CC1", True),             # 1,4-cyclohexane
    # a lowercase-pseudoasymmetric M2.3 positive ring is symmetric -> True
    ("Cn1cc(Br)c(CN[C@H]2C[C@H](NC(=O)C)C2)n1", True), # cyclobutane-1,3 (simplified W3)
    # true chiral pairs: cis/trans under-specified -> False
    ("CNC(=O)N[C@H]1CCCC[C@@H]1O", False),             # 1,2-cyclohexane
    ("CNC(=O)[C@H]1C[C@@H]1O", False),                 # cyclopropane-1,2 (adjacent)
    # a review's diazinane leak: opposite-sense N-pendant chirals -> False
    ("C[C@H](O)C(=O)NC[C@H]1CN([C@H](C)c2ccccc2)[C@@H]([C@@H](C)O)N([C@@H](C)c2ccccc2)C1", False),
]


@pytest.mark.parametrize("smi,expected", PREDICATE_CASES)
def test_ring_cistrans_completeness_predicate(smi, expected):
    mol, a, b = _centres_of_first_sat_ring(smi)
    assert _ring_cistrans_is_complete(mol, a, b) is expected


def _all_sat_2centre_rings(smi):
    """Return (mol, [ring_atomset,...]) for every small saturated ring with
    exactly 2 same-ring _CIPCode stereocentres. For the joint-completeness test."""
    mol = Chem.MolFromSmiles(smi)
    assign_stereochemistry(mol)
    ri = mol.GetRingInfo()
    rings = []
    for ring in ri.AtomRings():
        n = len(ring)
        if not (3 <= n <= 6):
            continue
        if any(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            continue
        if not all(mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % n]).GetBondTypeAsDouble() == 1.0
                   for i in range(n)):
            continue
        cen = [a for a in ring if mol.GetAtomWithIdx(a).HasProp("_CIPCode")]
        if len(cen) == 2 and ri.AreAtomsInSameRing(cen[0], cen[1]):
            rings.append(frozenset(ring))
    return mol, rings


def test_rings_jointly_complete_two_symmetric_rings():
    """A molecule with TWO disjoint symmetric substituent rings (two 1,3-cyclobutanes
    linked by an amide chain): each ring is individually complete AND the pair is
    jointly complete (flip ALL 4 centres -> same molecule), so a 2-descriptor
    relative name is 0-wrong. Guards the rung-1c k=2 path.

    NB: the whole-branch reviewer could not construct an individually-complete-but-
    jointly-INCOMPLETE pair (the only rings that could need a global-swap rescue are
    rejected individually by _ring_cistrans_is_complete), so only the True case is
    asserted here; the guard's value is rejecting the (unconstructed) coupling case
    at 0 cost to the common single-ring path."""
    smi = "O=C([C@H]1C[C@@H](C(=O)NCC)C1)NC[C@H]1C[C@@H](F)C1"
    mol, rings = _all_sat_2centre_rings(smi)
    assert len(rings) == 2, f"fixture must have two 2-centre saturated rings, got {len(rings)}"
    # each ring individually complete
    for rs in rings:
        a, b = [x for x in rs if mol.GetAtomWithIdx(x).HasProp("_CIPCode")]
        assert _ring_cistrans_is_complete(mol, a, b) is True
    # and the pair jointly complete
    assert _rings_jointly_complete(mol, rings) is True
    # empty ring list -> no certifiable centres -> False (reject-safe)
    assert _rings_jointly_complete(mol, []) is False


# The best-effort tier ships with the OPSIN validity gate ON in production
# (``namer._DISABLE_VALIDITY_GATE`` defaults False); the suite's autouse fixture
# turns it OFF unless a test opts back in. Without the gate, an OPSIN-unparseable
# absolute ``(1R,3S)-`` general-engine candidate leaks through BEFORE the floor's
# cis/trans reclaim is observed, so the engine integration tests below MUST run
# gate-ON to see the real production emission (skipped when the OPSIN jar is
# absent, per conftest). Applied per-test rather than module-wide so the pure-
# RDKit predicate test above still runs jar-absent. Same rationale as
# test_m2_cistrans_substituent.py.
ABS_BLOCK = re.compile(r"\(\d+[RS],\d+[RS]\)")

# (row, cis/trans PARTNER) -- the partner is the DISTINCT diastereomer the row's
# cis/trans name must not collide with. It is built by flipping ONE ring sense
# (one ``@``->``@@`` on the second ring stereocentre), i.e. cis<->trans. NB the
# partner is NOT ``flip-both``: for a COMPLETE ring flip-both is the identity
# (that is exactly what ``_ring_cistrans_is_complete`` certifies), so a flip-both
# "diastereomer" is the SAME molecule and would trivially share the name. The
# flip-one partner is a genuinely distinct molecule (asserted below) that must
# receive the opposite ``cis``/``trans`` prefix.
RECLAIM = [  # symmetric substituent ring -> reclaim as a verified cis/trans name
    ("C[C@H](CSc1ccc(F)cc1)C(=O)NCC[C@H]1C[C@H](NC(=O)[C@@H]2CNC(=O)C2)C1",    # head-0
     "C[C@H](CSc1ccc(F)cc1)C(=O)NCC[C@H]1C[C@@H](NC(=O)[C@@H]2CNC(=O)C2)C1"),  # partner
    ("O=C1CC[C@@H](CC(=O)NC[C@H]2C[C@H](NC(=O)[C@@H](O)c3ccc(Cl)cc3)C2)C1",    # head-2
     "O=C1CC[C@@H](CC(=O)NC[C@H]2C[C@@H](NC(=O)[C@@H](O)c3ccc(Cl)cc3)C2)C1"),  # partner
]
BOTH_KINDS = "C[C@H](Cc1cccc(Cl)c1)C(=O)N[C@H]1C[C@H](NC(=O)[C@@H]2CCCC[C@H]2O)C1"  # head-1
TRUE_PAIR_ONLY = "CNC(=O)N[C@H]1CCCC[C@@H]1O"
PIN_ISOLATION = [
    ("CNC(=O)[C@@H]1C[C@@H](O)CN1", "(2S,4R)-N-methyl-4-hydroxypyrrolidine-2-carboxamide"),
    ("CNC(=O)[C@H]1O[C@H]1C", "(2S,3S)-N-methyl-2-methyloxirane-3-carboxamide"),
]


@pytest.fixture(scope="module")
def engine():
    return Orthonym(style="pin", **_emit_tier_flags("best-effort"))


def _inchikey(smi):
    return Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(smi))


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
@pytest.mark.parametrize("smi,partner", RECLAIM)
def test_symmetric_ring_reclaims_verified_cistrans(engine, smi, partner):
    name = engine.name_tiered(smi).get("name")
    assert name is not None, f"should reclaim: {smi}"
    assert ("cis" in name) or ("trans" in name), f"needs a relative ring descriptor: {name}"
    assert verify_or_none(name, smi) == name, f"must round-trip to input: {name}"
    # 0-wrong (non-under-specification): the cis/trans PARTNER is a DISTINCT
    # diastereomer (a name shared by two distinct molecules would be
    # under-specified). It must NOT receive the same name -- it gets the opposite
    # cis/trans prefix. This is the RIGHT test: verify_or_none(name, partner) is
    # vacuous for a shipped name, and flip-BOTH is the identity for a complete
    # ring, so the partner must be the flip-ONE (cis<->trans) molecule.
    assert _inchikey(smi) != _inchikey(partner), \
        f"test bug: partner is not a distinct diastereomer of {smi}"
    partner_name = engine.name_tiered(partner).get("name")
    # partner_name must be non-None (else `!= name` would pass vacuously on an
    # abstain), proving the reclaim is symmetric across the cis/trans pair...
    assert partner_name is not None, f"partner should also reclaim, not abstain: {partner}"
    #...and it must differ from the row's name (a shared name = under-specified).
    assert partner_name != name, \
        f"UNDER-SPECIFIED: cis/trans partner got the same name {name}"


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
def test_both_kinds_keeps_true_pair_absolute_block(engine):
    name = engine.name_tiered(BOTH_KINDS).get("name")
    assert name is not None
    assert ("cis" in name) or ("trans" in name), f"symmetric cyclobutane should be relative: {name}"
    assert ABS_BLOCK.search(name), f"true-pair cyclohexane must keep its absolute block: {name}"
    assert verify_or_none(name, BOTH_KINDS) == name


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
def test_true_pair_only_never_ships_bare_cistrans(engine):
    name = engine.name_tiered(TRUE_PAIR_ONLY).get("name")
    if name is not None:
        assert "cis" not in name and "trans" not in name, f"true pair must stay absolute: {name}"
        assert verify_or_none(name, TRUE_PAIR_ONLY) == name


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
@pytest.mark.parametrize("smi,expected", PIN_ISOLATION)
def test_pin_path_absolute_ring_unchanged(engine, smi, expected):
    assert engine.name_tiered(smi).get("name") == expected
