"""The centres labeller beside rings that are not mancude.

 "Mancude rings and ring systems" (the Blue Book): "Mancude rings, i.e.,
rings or ring systems having the maximum number of noncumulative double bonds, are treated
as Kekulé structures. For mancude heterocycles, each duplicate atom is given an atomic
number that is the mean of what it would have if the double bonds were located at each of
the possible positions."

Two things are checked against the pinned ``centres`` 1.2.1 jar:

* ``perception.centres_mancude`` repeats the typing and pruning of
  ``com.simolecule.centres.Mancude``: its parts give, for every atom, the mean atomic number
  the jar computes (``Mancude.CalcFracAtomNums``), on a list of ring systems of every kind
  the typing has a case for;
* the labels beside a ring that is not mancude are those of its Kekulé structure (Sequence
  Rule 1a with the atomic numbers of the structure as drawn), and the labels beside a mancude
  ring are those the plain string gave.

Tests that need the jar skip cleanly when Java, JPype or the jar is absent.
"""
from fractions import Fraction

import pytest
from rdkit import Chem

from orthonym.perception import centres_mancude as cm
from orthonym.perception.centres_bridge import (
    _centres_input,
    _smiles_output_order,
    centres_label_batch,
    centres_label_mol,
)
from orthonym.perception.stereo import assign_stereochemistry
from tests.support.jars import jar_or_skip

pytestmark = pytest.mark.integration


def _jvm_classes():
    """(SmilesParser, SilentChemObjectBuilder, CdkMol) of the in-process JVM, or skip."""
    jar_or_skip("centres")
    jpype = pytest.importorskip("jpype")
    from orthonym import jvm_bridge
    if not jvm_bridge.centres_available():
        pytest.skip("the in-process JVM is not available")
    return (jpype.JClass("org.openscience.cdk.smiles.SmilesParser"),
            jpype.JClass("org.openscience.cdk.silent.SilentChemObjectBuilder"),
            jpype.JClass("com.simolecule.centres.CdkMol"))


def _jar_means(smiles):
    """The jar's mean atomic number of every atom of ``smiles``, as (numerator, denominator)."""
    parser_cls, builder_cls, mol_cls = _jvm_classes()
    parser = parser_cls(builder_cls.getInstance())
    mol = mol_cls(parser.parseSmiles(smiles))
    out = []
    for i in range(int(mol.getNumAtoms())):
        frac = mol.getFractionalAtomicNum(mol.getAtom(i))
        out.append((int(frac.getNum()), int(frac.getDen())))
    return out


def _replica_means(mol):
    """``Mancude.CalcFracAtomNums`` as repeated by ``centres_mancude``: ordinary parts give
    each atom the mean atomic number of its neighbours in the part; an anionic part gives
    all its atoms the sum over its double bonds (the lone pair counting zero) over its size."""
    kek = Chem.Mol(mol)
    Chem.Kekulize(kek, clearAromaticFlags=True)
    n = mol.GetNumAtoms()
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    means = {i: (z[i], 1) for i in range(n)}
    seed = cm._seed_types(mol, kek)
    if all(t == 'C' for t in seed.values()):
        return means
    nbrs = [[o.GetIdx() for o in a.GetNeighbors()] for a in mol.GetAtoms()]
    ring = [[] for _ in nbrs]
    for b in mol.GetBonds():
        if b.IsInRing():
            ring[b.GetBeginAtomIdx()].append(b.GetEndAtomIdx())
            ring[b.GetEndAtomIdx()].append(b.GetBeginAtomIdx())
    types = dict(seed)
    cm._relax(types, nbrs)
    parts = cm._parts(types, ring)
    part_of = {a: k for k, part in enumerate(parts) for a in part}
    for a, k in part_of.items():
        around = [o for o in nbrs[a] if part_of.get(o) == k]
        means[a] = (sum(z[o] for o in around), len(around))
    for k, part in enumerate(parts):
        if any(types[a] in cm._ANIONIC for a in part):
            total = sum((int(kek.GetBondBetweenAtoms(a, o).GetBondTypeAsDouble()) - 1) * z[o]
                        for a in part for o in nbrs[a]
                        if part_of.get(o) == k
                        and kek.GetBondBetweenAtoms(a, o).GetBondTypeAsDouble() > 1)
            for a in part:
                means[a] = (total, len(part))
    return means


def _same_mean(a, b):
    """Two (numerator, denominator) means; a denominator of 0 is its own value."""
    if a[1] == 0 or b[1] == 0:
        return a == b
    return Fraction(*a) == Fraction(*b)


#: Ring systems of every kind ``Mancude.SeedTypes`` has a case for: carbocycles, azines and
#: azoles (``Nv3D2``), iminium and pyrylium rings (``Nv4D3Plus``, ``Ov3D2Plus``), carbanion and
#: amide rings (``Cv3D3Minus``, ``Nv2D2Minus``), silicon and phosphorus analogues, rings that
#: are pruned (an N-H, an sp3 atom), a ring with exocyclic double bonds, and those of the
#: labels below.
_RING_SYSTEMS = [
    "c1ccccc1", "c1ccncc1", "c1cnccn1", "c1ncncn1", "c1ccc2ncccc2c1", "c1ccc2ncncc2c1",
    "c1ccc2[nH]ccc2c1", "c1ccc2occc2c1", "c1cc2ccc3cccc4ccc(c1)c2c34",
    "c1ccc2c(c1)[nH]c1ccccc12", "c1ncc2nc[nH]c2n1", "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "C[n+]1ccccc1", "c1cc[o+]cc1", "c1cc[s+]cc1", "[cH-]1cccc1", "[c-]1ccc[nH]1",
    "c1cc[n-]c1", "C1=CC=C[CH-]1", "c1ccc[nH]c1=O", "O=c1cc[nH]cc1",
    "O=c1[nH]cnc2ccccc12", "O=c1c2ccccc2[nH]c2ccccc12", "C1=CC=CC=C1C=O",
    "c1cc[nH]c1", "c1cscn1", "c1c[nH]cn1", "c1cnoc1", "c1c[nH]nc1", "c1ccpcc1",
    "C=C1C=CC(=C)C=C1", "C=C1C(=C)N=CN=C1", "C/C=C\\1/C(=NC=NC1=C)C", "CC=C1C=NC=NC1=O",
    "C[C@@H]1N=C2C=CC=CC2=CN1C", "C[C@@H]2N=C3C=CC=CC3=CN2c1ccccc1",
    "C=C1C=Nc2cccnc12", "O=C1C=CC=CC=N1", "C1=CC=CC=CC=C1", "c1ccc2cc3ccccc3cc2c1",
    "CC/C=C/1C(=C/C)N=C(C2=C1N(C(=N2)COCC)CC(C)(C)O)N",
    "CC[C@@]1\\2C3=CC=CC=C3C(=N1)N=C4C5=CC=CC=C5C(=NC6=CC(=CC=C6)N=C7C8=CC=CC=C8/C(=N2)/N7)N4",
    "C[Si]1=CC=CC=C1", "N#Cc1ccncc1", "C[S+](C)c1ccncc1",
]


@pytest.mark.parametrize("smiles", _RING_SYSTEMS)
def test_the_repeated_typing_gives_the_means_of_the_jar(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    text = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(text)      # atoms in the order of the string the jar reads
    jar = _jar_means(text)
    assert len(jar) == mol.GetNumAtoms()
    replica = _replica_means(mol)
    bad = [(i, jar[i], replica[i]) for i in range(len(jar)) if not _same_mean(jar[i], replica[i])]
    assert not bad, (text, bad)


# ---------------------------------------------------------------------------
# What centres does with the string the bridge writes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smiles", [
    "C/C=C\\1/C(=NC=NC1=C)C",
    "C[C@@H]1N=C2C=CC=CC2=CN1C",
    "C[C@@H]2N=C3C=CC=CC3=CN2c1ccccc1",
    "CC/C=C/1C(=C/C)N=C(C2=C1N(C(=N2)COCC)CC(C)(C)O)N",
])
def test_no_atom_of_the_string_is_left_with_a_zero_mean_and_held_atoms_are_literal(smiles):
    """The jar gives an atom that no part pruned but that has no neighbour in its part a
    mean of 0/0; fixing the neighbours of a held atom must not leave one."""
    mol = Chem.MolFromSmiles(smiles)
    held = cm.non_mancude_ring_atoms(mol)
    assert held
    text, order = _centres_input(mol)
    jar = _jar_means(text)
    for pos, (num, den) in enumerate(jar):
        assert den >= 1, (text, pos)
        if order[pos] in held:
            assert den == 1, (text, pos)


def _labels(mol):
    return ({a.GetIdx(): a.GetProp('_CIPCode') for a in mol.GetAtoms() if a.HasProp('_CIPCode')},
            {b.GetIdx(): b.GetProp('_CIPCode') for b in mol.GetBonds() if b.HasProp('_CIPCode')})


@pytest.mark.parametrize("smiles, atoms, bonds", [
    # The 4,5-diylidene pyrimidine, C-1=C-2 the ethylidene bond. C-2 bears the ring atoms C-3
    # {N, N (duplicate), C} and C-7 {N, C, C (duplicate)}: C-3 outranks C-7 with the
    # duplicate of N carrying 7 as drawn, and the CH3 on C-1 is trans to C-3: E. The mean of
    # the ring (6.5 at C-3 and C-7) made the labellers say Z. (the ring has two
    # double bonds, not the three of a mancude ring) and Sequence Rule 1a; OPSIN reads
    # '(5E)-' to this input.
    ("C/C=C\\1/C(=NC=NC1=C)C", {}, {1: 'E'}),
    # 1-phenyl-2-methyl-1,2-dihydro-quinazoline-like ring with an o-quinoid carbocycle.
    # Centre C-1 (H, CH3, N=C, N-phenyl): at sphere III the =N branch has C-3 with
    # {N (duplicate), C, C} and the N-phenyl branch has the ipso carbon with {C, C, C};
    # N, C, C > C, C, C, so the order is N= > N-Ph > CH3 > H, with the duplicate of N on C-3
    # carrying 7 (the ring is not mancude, so no mean). SMILES order of the neighbours of
    # the centre: CH3, H, ring-closure N-Ph, N=; @@ puts H, N-Ph, N= clockwise seen from
    # CH3, so CH3, N-Ph, N= is clockwise seen with H behind and N=, N-Ph, CH3 is
    # anticlockwise: S.
    ("C[C@@H]2N=C3C=CC=CC3=CN2c1ccccc1", {1: 'S'}, {}),
    ("C[C@H]2N=C3C=CC=CC3=CN2c1ccccc1", {1: 'R'}, {}),
    # A held ring atom that is a charged nitrogen (N+ with an exocyclic =CH2, charge +1 -> +2
    # in the string): the centre bears NH2 > ring C {N, C, C} > CH3 > H; SMILES order CH3, H,
    # N, ring C with @ puts CH3, N, ring C anticlockwise seen with H behind, and N, ring C,
    # CH3 is the same cycle: S.
    ("C[C@H](N)C1=CC(=C)C=C[N+]1=C", {1: 'S'}, {}),
    # A zwitterion (ammonium and carboxylate) beside the held ring. The ring centre is the one
    # of the N-phenyl case above (N= > N-CH3 > C > H, the same neighbour order): S. The
    # ammonium centre: N > the aminal carbon {N, N, H} > CH2 {C, H, H} > H, and its
    # neighbours in SMILES order (N, H, CH2, aminal C) with @@ give S.
    ("[NH3+][C@@H](CC(=O)[O-])[C@@H]1N=C2C=CC=CC2=CN1C", {1: 'S', 6: 'S'}, {}),
    # An isotope: the ring carbon {N, N, H} outranks the methyls, and 13CH3 outranks CH3 by
    # Sequence Rule 2 (after Rule 1 ties). Neighbours in SMILES order (13CH3, H, CH3, ring C)
    # with @ give ring C > 13CH3 > CH3 anticlockwise seen with H behind: S; the ring centre
    # keeps S.
    ("[13CH3][C@H](C)[C@@H]1N=C2C=CC=CC2=CN1C", {1: 'S', 3: 'S'}, {}),
])
def test_labels_beside_a_ring_that_is_not_mancude_are_those_of_its_kekule_structure(
        smiles, atoms, bonds):
    mol = Chem.MolFromSmiles(smiles)
    assign_stereochemistry(mol)
    assert _labels(mol) == (atoms, bonds)


def _plain_labels(mol):
    """What ``centres_label_mol`` gave before the bridge told centres anything: the labels
    of RDKit's canonical SMILES of ``mol``, mapped back by its atom order."""
    text = Chem.MolToSmiles(mol)
    order = _smiles_output_order(mol)
    raw = centres_label_batch([text])[text]
    out = Chem.Mol(mol)
    from orthonym.perception.centres_bridge import apply_centres_labels
    apply_centres_labels(out, {order[pos - 1] + 1: lab for pos, lab in raw.items()})
    return _labels(out)


@pytest.mark.parametrize("smiles", [
    "C[C@H](N)C(=O)O",
    "C[C@H](c1ccncc1)NC(=O)c1ccc2ncccc2c1",
    "C[C@@H](Cn1cnc2c1c(=O)n(C)c(=O)n2C)O",
    "C/C=C/c1ccc2[nH]ccc2c1",
    "O=c1cc[nH]cc1C[C@H](N)C",
    "C[C@@H]1CC[C@H](CC1)c1ccc2c(c1)C(=O)c1ccccc1C2=O",
    "C=C1C=CC(=C)C=C1.C[C@H](O)c1ccncc1",     # a ring that is not mancude but is carbon only
])
def test_labels_beside_mancude_rings_are_the_plain_labels(smiles):
    jar_or_skip("centres")
    mol = Chem.MolFromSmiles(smiles)
    assert cm.non_mancude_ring_atoms(mol) == ()
    labelled = Chem.Mol(mol)
    assert centres_label_mol(labelled)
    assert _labels(labelled) == _plain_labels(mol)
