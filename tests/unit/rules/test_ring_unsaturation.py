""" a phase (H2): the shared ring-unsaturation locant primitive.

The hole this closes, reproduced before the fix
-----------------------------------------------
``vonbaeyer_universal.analyze_cage_universal`` recomputed ``double_bonds`` into
the von-Baeyer COMPOSITE form (``1(6)`` for a bond whose two atoms are not
consecutively numbered) but left ``triple_bonds`` exactly as
``polycyclic.get_polycyclic_unsaturation`` produced it -- ``min(loc1, loc2)``, a
bare lower locant with **no composite form and no guard**. A cage carrying a
non-consecutively-numbered triple bond would therefore have emitted a ``-yne``
locant denoting a DIFFERENT bond than the one present. The spiro sibling refused
that case outright; the cage path was silent.

``rules/ring_unsaturation.py::render_ring_unsaturation`` is now the single
producer for both bond orders, on both paths.

The ``-yne`` policy: a non-consecutive triple bond REFUSES (``None``).
(1) grants the compound ``x(y)`` locant to DOUBLE bonds only, and every
Blue Book ``-yne`` example carries a plain locant -- so the bare lower locant is
correct on every structure the Blue Book covers, and no compound ``-yne`` form
exists to fall back on. The state is moreover unreachable for standard bonding
numbers +: every non-consecutively-numbered bond is incident to
a bridgehead, a bridgehead has >= 3 skeletal neighbours, a C(triple)C carbon has 2
sigma bonds), so reaching it means the upstream NUMBERING is wrong and citing the
bare locant would be a genuine wrong-structure emission (a reader parses
``-8-yne`` as the 8-9 bond). The refusal is an assertion against that bug.
"""
import pytest
from rdkit import Chem

from orthonym.rules.polycyclic import VonBaeyerAnalyzer
from orthonym.rules.ring_unsaturation import (
    RingUnsaturation, render_ring_unsaturation,
)
from orthonym.rules.vonbaeyer_universal import analyze_cage_universal


pytestmark = pytest.mark.unit


def _cage(smiles):
    """(kekulized mol, numbering) for a von-Baeyer cage, prepared exactly the way
    ``analyze_cage_universal`` prepares it (canonical reparse -> kekulize ->
    analyze), so the locants under test are the ones that ship."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"unparseable test SMILES {smiles!r}"
    canon = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
    kek = Chem.RWMol(canon)
    Chem.Kekulize(kek, clearAromaticFlags=True)
    ring_atoms = set()
    for ring in kek.GetRingInfo().AtomRings():
        ring_atoms.update(ring)
    desc = VonBaeyerAnalyzer().analyze(kek, ring_atoms)
    assert desc and desc.numbering, f"no von-Baeyer descriptor for {smiles!r}"
    return kek, desc.numbering


# ---------------------------------------------------------------------------
# 1. Double bonds: plain locant when consecutive, composite when not.
# ---------------------------------------------------------------------------
def test_consecutive_ene_is_a_plain_locant():
    """norbornene: the ring double bond spans consecutively-numbered atoms."""
    kek, numbering = _cage("C1=CC2CCC1C2")
    res = render_ring_unsaturation(kek, numbering)
    assert res is not None
    assert len(res.double_pairs) == 1
    lo, hi = res.double_pairs[0]
    assert hi == lo + 1
    assert res.double_locants == (str(lo),)
    assert res.triple_pairs == ()
    assert res.triple_locants == ()


def test_non_consecutive_ene_is_a_composite_locant():
    """octalin: the ene sits on the bridgehead-bridgehead bond -> ``1(6)``."""
    kek, numbering = _cage("C1CCC2=C1CCCC2")
    res = render_ring_unsaturation(kek, numbering)
    assert res is not None
    assert res.double_locants == ("1(6)",)
    assert res.double_pairs == ((1, 6),)


def test_norbornadiene_two_plain_locants():
    kek, numbering = _cage("C1=CC2C=CC1C2")
    res = render_ring_unsaturation(kek, numbering)
    assert res is not None
    assert res.double_locants == ("2", "5")
    assert res.double_pairs == ((2, 3), (5, 6))


def test_saturated_cage_is_empty():
    kek, numbering = _cage("C1CC2CCC1C2")
    res = render_ring_unsaturation(kek, numbering)
    assert res is not None
    assert res.double_pairs == ()
    assert res.double_locants == ()
    assert res.triple_pairs == ()
    assert res.triple_locants == ()


def test_mancude_cage_locants_match_the_shipped_polyene():
    """heptalene, whose exact polyene string is gold-locked in
    test_v26_p2_aromatic_vonbaeyer.py: ``bicyclo[6.4.0]dodeca-1,3,5,7,9,11-hexaene``."""
    kek, numbering = _cage("C1=CC=CC2=CC=CC=CC=C12")
    res = render_ring_unsaturation(kek, numbering)
    assert res is not None
    assert res.double_locants == ("1", "3", "5", "7", "9", "11")


# ---------------------------------------------------------------------------
# 2. Triple bonds: plain locant when consecutive; REFUSE when not.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles", [
    "C1CCC#CCCC2CCCCC12",   # yne in the 8-ring of a bicyclo[7.4.0]
    "C1CC#CCCC2C1CCCC2",    # yne in the 8-ring of a bicyclo[6.4.0]
])
def test_consecutive_yne_is_a_plain_locant(smiles):
    kek, numbering = _cage(smiles)
    res = render_ring_unsaturation(kek, numbering)
    assert res is not None
    assert len(res.triple_pairs) == 1
    lo, hi = res.triple_pairs[0]
    assert hi == lo + 1, "corpus expectation: this yne is consecutively numbered"
    assert res.triple_locants == (str(lo),)


def test_non_consecutive_yne_refuses():
    """The keystone. ``render_ring_unsaturation`` is a pure function of
    ``(mol, numbering)``, so the case is exhibited by handing it a numbering under
    which the real triple bond is non-consecutive -- no fabricated molecule
    needed, and the rule is tested rather than the corpus."""
    kek, numbering = _cage("C1CCC#CCCC2CCCCC12")
    yne = next(b for b in kek.GetBonds()
               if b.GetBondType() == Chem.BondType.TRIPLE)
    i, j = yne.GetBeginAtomIdx(), yne.GetEndAtomIdx()
    assert abs(numbering[i] - numbering[j]) == 1, "precondition: consecutive"
    assert render_ring_unsaturation(kek, numbering) is not None

    # Swap one yne endpoint's locant with a distant atom's -> the SAME molecule,
    # a still-bijective numbering, but the yne now spans non-consecutive locants.
    far = next(a for a in numbering
               if a not in (i, j) and abs(numbering[a] - numbering[i]) > 2)
    swapped = dict(numbering)
    swapped[i], swapped[far] = numbering[far], numbering[i]
    assert abs(swapped[i] - swapped[j]) != 1
    assert render_ring_unsaturation(kek, swapped) is None


def test_refusal_is_not_triggered_by_a_non_consecutive_ene():
    """The conservative ``-yne`` policy must not leak into ``-ene``: octalin's
    ``1(6)`` is emitted, not refused."""
    kek, numbering = _cage("C1CCC2=C1CCCC2")
    assert render_ring_unsaturation(kek, numbering) is not None


# ---------------------------------------------------------------------------
# 3. Scope + determinism.
# ---------------------------------------------------------------------------
def test_bonds_outside_the_numbering_are_ignored():
    """Only bonds with BOTH atoms in ``numbering`` count -- an exocyclic ene on a
    substituent is not ring unsaturation."""
    kek, numbering = _cage("C1CC2CCC1C2")          # norbornane, saturated
    rw = Chem.RWMol(kek)
    c1 = rw.AddAtom(Chem.Atom(6))
    c2 = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(0, c1, Chem.BondType.SINGLE)
    rw.AddBond(c1, c2, Chem.BondType.DOUBLE)       # exocyclic vinyl
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    res = render_ring_unsaturation(mol, numbering)
    assert res is not None
    assert res.double_pairs == ()


@pytest.mark.parametrize("smiles", [
    "C1=CC2C=CC1C2",            # norbornadiene
    "C1CCC2=C1CCCC2",           # octalin (compound locant)
    "C1=CC=CC2=CC=CC=CC=C12",   # mancude fused cage -> kekulized polyene
    "C1CCC#CCCC2CCCCC12",       # cage carrying a yne
])
def test_determinism_across_smiles_spellings(smiles):
    """Different atom orderings of the SAME molecule must give identical locants
    (the cage path canonical-reparses before numbering, so this holds through the
    helper). Alternative spellings are generated FROM the molecule and each is
    checked back against its canonical form, so the test cannot silently compare
    two different structures -- which an earlier hand-written pair did."""
    mol = Chem.MolFromSmiles(smiles)
    canon = Chem.MolToSmiles(mol)
    spellings = {smiles}
    for k in range(mol.GetNumAtoms()):
        alt = Chem.MolToSmiles(mol, canonical=False, rootedAtAtom=k)
        assert Chem.MolToSmiles(Chem.MolFromSmiles(alt)) == canon, (
            f"generated spelling {alt!r} is not the same molecule")
        spellings.add(alt)
    assert len(spellings) > 1, "no alternative spelling generated"

    results = set()
    for spelling in sorted(spellings):
        kek, numbering = _cage(spelling)
        res = render_ring_unsaturation(kek, numbering)
        assert res is not None, spelling
        results.add((res.double_locants, res.triple_locants))
    assert len(results) == 1, f"{smiles}: non-deterministic -> {results}"


def test_result_is_frozen():
    kek, numbering = _cage("C1=CC2CCC1C2")
    res = render_ring_unsaturation(kek, numbering)
    assert isinstance(res, RingUnsaturation)
    with pytest.raises(Exception):
        res.double_locants = ()


# ---------------------------------------------------------------------------
# 4. The STRUCTURAL claim behind the refusal, tested rather than assumed.
#
#: a bridge connects two bridgeheads, so von Baeyer numbering is a
# concatenation of runs each ending at a bridgehead and every
# non-consecutively-numbered bond is incident to a bridgehead.: a
# bridgehead has >= 3 skeletal neighbours. A C(triple)C carbon has 2 sigma bonds.
# Hence no triple bond can be non-consecutive -- for standard bonding numbers.
# (A lambda-n heteroatom that is both 3-connected and triply bonded would escape
# the argument; the Blue Book has no rule for citing that bond, so the primitive's
# refusal is what covers it.) These two tests check the claim on real cages rather
# than resting on it.
# ---------------------------------------------------------------------------
#: Cages whose skeletal atoms all have standard bonding numbers.
STANDARD_BONDING_CORPUS = [
    "C1CC2CCC1C2",              # norbornane
    "C1=CC2C=CC1C2",            # norbornadiene
    "C1CCC2=C1CCCC2",           # octalin
    "C1CC2CCC1CC2",             # bicyclo[2.2.2]octane
    "C1CCC2CCCCC2C1",           # decalin
    "C1CC2CC1CC2",              # bicyclo[2.1.1]hexane
    "C1CC2CC3CC1CC(C2)C3",      # adamantane
    "C1=CC=CC2=CC=CC=CC=C12",   # heptalene
    "C1=CC=Cc2ccccc2C1",        # benzocycloheptene
    "C1CCC#CCCC2CCCCC12",       # cage with a yne
    "C1CC#CCCC2C1CCCC2",        # cage with a yne
    "O1CC2CCC1C2",              # hetero cage
    "C1CC2CCOC2C1",             # hetero cage
]

#: FOUND COUNTEREXAMPLE (a phase T1). The lambda-n hole the Blue Book research
#: flagged as NOT ESTABLISHED is REACHABLE: a pnictogen bridgehead carrying two
#: ring bonds plus a triple bond is 3-connected (a bridgehead, at valence
#: 5, so it escapes the degree-2 argument. In these bicyclo[6.6.0] cages the triple
#: bond spans locants 1<->8. HEAD cited ``[1]`` for it -- i.e. named the 1-2 bond,
#: a wrong structure. So the fail-closed branch is LOAD-BEARING, not defensive.
LAMBDA_N_YNE_COUNTEREXAMPLES = [
    "P12#P(CCCCCC1)CCCCCC2",
    "[As]12#[As](CCCCCC1)CCCCCC2",
    "[Sb]12#[Sb](CCCCCC1)CCCCCC2",
]

#: The bridgehead-degree claim is asserted over BOTH -- it holds for the
#: counterexample too (that is exactly why the counterexample exists: the bond is
#: incident to a bridgehead, and here the bridgehead can carry the triple bond).
STRUCTURAL_CORPUS = STANDARD_BONDING_CORPUS + LAMBDA_N_YNE_COUNTEREXAMPLES


@pytest.mark.parametrize("smiles", STRUCTURAL_CORPUS)
def test_non_consecutive_bonds_always_touch_a_high_degree_atom(smiles):
    kek, numbering = _cage(smiles)
    cage = set(numbering)
    for bond in kek.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i not in cage or j not in cage:
            continue
        lo, hi = sorted((numbering[i], numbering[j]))
        if hi == lo + 1:
            continue
        deg_i = sum(1 for n in kek.GetAtomWithIdx(i).GetNeighbors()
                    if n.GetIdx() in cage)
        deg_j = sum(1 for n in kek.GetAtomWithIdx(j).GetNeighbors()
                    if n.GetIdx() in cage)
        assert max(deg_i, deg_j) >= 3, (
            f"{smiles}: non-consecutive bond {lo}({hi}) has cage degrees "
            f"{deg_i}/{deg_j} -- both < 3, so the bridgehead argument fails "
            f"and a non-consecutive yne may be reachable")


@pytest.mark.parametrize("smiles", STANDARD_BONDING_CORPUS)
def test_no_carbon_triple_bond_sits_on_a_non_consecutive_bond(smiles):
    """The corollary for STANDARD bonding numbers: with every non-consecutive bond
    touching a degree->=3 atom, and a C(triple)C carbon capped at degree 2, no yne
    in this corpus is non-consecutive."""
    kek, numbering = _cage(smiles)
    for bond in kek.GetBonds():
        if bond.GetBondType() != Chem.BondType.TRIPLE:
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i not in numbering or j not in numbering:
            continue
        lo, hi = sorted((numbering[i], numbering[j]))
        assert hi == lo + 1, f"{smiles}: COUNTEREXAMPLE -- yne at {lo}({hi})"


@pytest.mark.parametrize("smiles", LAMBDA_N_YNE_COUNTEREXAMPLES)
def test_lambda_n_bridgehead_yne_is_reachable_and_refused(smiles):
    kek, numbering = _cage(smiles)
    yne = [b for b in kek.GetBonds()
           if b.GetBondType() == Chem.BondType.TRIPLE]
    assert len(yne) == 1
    i, j = yne[0].GetBeginAtomIdx(), yne[0].GetEndAtomIdx()
    lo, hi = sorted((numbering[i], numbering[j]))

    # (a) the triple bond really is non-consecutively numbered...
    assert hi != lo + 1, "corpus expectation: this yne must be non-consecutive"
    # (b)... because an endpoint is a 3-connected (bridgehead) skeletal atom,
    # which the neutral-carbon degree-2 argument cannot exclude...
    assert max(kek.GetAtomWithIdx(i).GetDegree(),
               kek.GetAtomWithIdx(j).GetDegree()) >= 3
    # (c)... so the primitive must refuse rather than cite the bare lower locant,
    # which would denote the lo/lo+1 bond -- a different molecule.
    assert render_ring_unsaturation(kek, numbering) is None


@pytest.mark.parametrize("smiles", LAMBDA_N_YNE_COUNTEREXAMPLES)
def test_cage_analysis_refuses_the_counterexample(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert analyze_cage_universal(mol) is None
    assert analyze_cage_universal(mol, allow_mancude=True) is None


# ---------------------------------------------------------------------------
# 5. Both call sites are wired to the primitive (no forked second producer).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected_ene", [
    ("C1CCC2=C1CCCC2", ["1(6)"]),
    ("C1=CC2C=CC1C2", ["2", "5"]),
    ("C1CC2CCC1C2", []),
])
def test_cage_payload_uses_the_primitive(smiles, expected_ene):
    mol = Chem.MolFromSmiles(smiles)
    cage = analyze_cage_universal(mol)
    assert cage is not None
    assert list(cage.unsaturation['double_bonds']) == expected_ene
    # the engine's oxo/ene valence guard reads the RAW pairs
    assert all(isinstance(p, tuple) and len(p) == 2
               for p in cage.unsaturation['double_bond_pairs'])
    assert list(cage.unsaturation['triple_bonds']) == []


def test_cage_payload_yne_is_a_consecutive_locant_string():
    mol = Chem.MolFromSmiles("C1CCC#CCCC2CCCCC12")
    cage = analyze_cage_universal(mol)
    assert cage is not None
    ynes = list(cage.unsaturation['triple_bonds'])
    assert len(ynes) == 1
    # a locant that _build_parent_with_unsaturation can spell directly
    assert str(ynes[0]).isdigit()
