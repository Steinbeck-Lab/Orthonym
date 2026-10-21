"""Ring atoms the centres labeller must not average, and the string it is given.

 "Mancude rings and ring systems" (the Blue Book): "Mancude rings, i.e.,
rings or ring systems having the maximum number of noncumulative double bonds, are treated
as Kekulé structures. For mancude heterocycles, each duplicate atom is given an atomic
number that is the mean of what it would have if the double bonds were located at each of
the possible positions." The mean is for mancude rings only. ``centres`` averages every
ring atom of the valence signature of a mancude ring, so ``perception.centres_mancude``
names the atoms of a part that is not mancude (its ring double bonds are fewer than the
maximum matching of its atoms) and ``centres_bridge._centres_input`` tells ``centres``.

These tests need no JVM. The labels themselves are in
``tests/integration/test_centres_mancude_parts.py``.
"""
import random

import pytest
from rdkit import Chem

from orthonym.perception import centres_bridge
from orthonym.perception.centres_bridge import _centres_input
from orthonym.perception.centres_mancude import STATS, non_mancude_ring_atoms

pytestmark = pytest.mark.unit


def _held(smiles):
    return non_mancude_ring_atoms(Chem.MolFromSmiles(smiles))


# Rings that are not mancude: two ring double bonds against the three of a six-membered
# mancude ring, the rest of the ring's atoms bearing an exocyclic double bond.
@pytest.mark.parametrize("smiles, held", [
    # 4,5-diylidene pyrimidine: the two ring atoms (2 and 7) that bear the ylidene groups
    # are the atoms whose double bond leaves the ring
    ("C/C=C\\1/C(=NC=NC1=C)C", (2, 7)),
    # the same ring without stereo: ring atoms 1 and 2 bear the methylidene groups
    ("C=C1C(=C)N=CN=C1", (1, 2)),
    # a 4-oxo-5-ylidene pyrimidine: the ring atom bearing the ylidene group and the
    # carbonyl carbon
    ("CC=C1C=NC=NC1=O", (2, 7)),
    # an o-quinoid carbocycle fused to a dihydropyrimidine (the 2,3-dihydroquinazoline
    # family): the carbocycle's two atoms with a double bond to an atom outside it, and
    # the =N and =CH- atoms that fixing them leaves with no neighbour in a part
    ("C[C@@H]1N=C2C=CC=CC2=CN1C", (2, 3, 8, 9)),
    ("C[C@@H]2N=C3C=CC=CC3=CN2c1ccccc1", (2, 3, 8, 9)),
])
def test_atoms_of_a_non_mancude_part_are_held(smiles, held):
    assert _held(smiles) == held


# Mancude parts, parts without an atom whose double bond leaves them, parts that are
# not judged, and molecules with no ring atom that could be averaged.
@pytest.mark.parametrize("smiles", [
    "c1ccncc1",                                  # pyridine
    "c1ccc2ncccc2c1",                            # quinoline
    "C[n+]1ccccc1",                              # pyridinium
    "c1cc[o+]cc1",                               # pyrylium
    "c1c[nH]cn1",                                # 1H-imidazole
    "[cH-]1cccc1",                               # cyclopentadienide (anionic part)
    "O=c1cc[nH]cc1",                             # 4-pyridone: pruned, no part
    "O=c1[nH]cnc2ccccc12",                       # quinazolin-4(3H)-one
    "O=c1c2ccccc2[nH]c2ccccc12",                 # acridone
    "O=C1CCCCC1",                                # cyclohexanone
    "N=C1C=CNC=C1",                              # 4-imino-1,4-dihydropyridine
    # a part whose double bonds are a maximum matching of its atoms keeps the mean, as
    # before: 3H-pyrrolo[3,2-b]pyridin-3-ylidene has the four ring double bonds that nine
    # atoms allow
    "C=C1C=Nc2cccnc12",
    # the ring with two ylidene groups is carbon only and so is its exocyclic partner:
    # averaging changes no atomic number, and the pyridine beside it is mancude
    "C=C1C=CC(=C)C=C1.c1ccncc1",
])
def test_other_parts_are_left_as_they_were(smiles):
    assert _held(smiles) == ()


def test_held_atoms_do_not_depend_on_the_atom_order():
    """Determinism: the held atoms of a renumbered molecule are the renumbered atoms."""
    rng = random.Random(20261021)
    for smiles in ("C/C=C\\1/C(=NC=NC1=C)C", "C[C@@H]2N=C3C=CC=CC3=CN2c1ccccc1",
                   "CC=C1C=NC=NC1=O", "CC/C=C/1C(=C/C)N=C(C2=C1N(C(=N2)COCC)CC(C)(C)O)N"):
        mol = Chem.MolFromSmiles(smiles)
        held = non_mancude_ring_atoms(mol)
        assert held
        for _ in range(8):
            order = list(range(mol.GetNumAtoms()))
            rng.shuffle(order)
            renumbered = Chem.RenumberAtoms(mol, order)
            where = {old: new for new, old in enumerate(order)}
            assert non_mancude_ring_atoms(renumbered) == tuple(sorted(where[a] for a in held))


def test_ambiguous_kekule_structure_is_not_judged():
    """A part with an atom whose double bond leaves it, and an aromatic atom whose own
    double bond could lie inside the part or outside it in different Kekulé structures
    (a benzodithiophene with ylidene groups, fused to a thiadiazole), is not judged: how
    many double bonds the part holds would depend on the Kekulé structure chosen."""
    smiles = ("CCCCCCCCCCCC[C@@H]1CC(CC(C1)/C=C2/C3=CC(=C)SC3=C4C2=C/C(=C5/C6=NC(=C(N=C6C(=C)"
              "C7=NSN=C57)C)C)/S4)CCCCCCCCCCCC")
    assert _held(smiles) == ()


# ---------------------------------------------------------------------------
# The string centres is given
# ---------------------------------------------------------------------------


def _read_unsanitized(smiles):
    """The string as centres reads it: RDKit's valence check would refuse a carbon with a
    charge and four bonds, centres does not check."""
    parsed = Chem.MolFromSmiles(smiles, sanitize=False)
    parsed.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(parsed)
    Chem.AssignStereochemistry(parsed, cleanIt=True, force=True)
    return parsed


def _charged_positions(smiles, order, mol):
    """Atom indices of ``mol`` whose atom in ``smiles`` has a formal charge other than
    the one ``mol`` has."""
    parsed = _read_unsanitized(smiles)
    return tuple(sorted(
        order[pos] for pos, atom in enumerate(parsed.GetAtoms())
        if atom.GetFormalCharge() != mol.GetAtomWithIdx(order[pos]).GetFormalCharge()))


def test_centres_string_raises_the_charge_of_the_held_atoms_only():
    mol = Chem.MolFromSmiles("C[C@@H]2N=C3C=CC=CC3=CN2c1ccccc1")
    held = non_mancude_ring_atoms(mol)
    smiles, order = _centres_input(mol)
    assert sorted(order) == list(range(mol.GetNumAtoms()))
    assert _charged_positions(smiles, order, mol) == held
    parsed = _read_unsanitized(smiles)
    assert not any(a.GetIsAromatic() for a in parsed.GetAtoms()), "the string is Kekulé"
    for pos, atom in enumerate(parsed.GetAtoms()):
        own = mol.GetAtomWithIdx(order[pos])
        assert atom.GetAtomicNum() == own.GetAtomicNum()
        assert atom.GetTotalNumHs() == own.GetTotalNumHs()
        if order[pos] not in held:
            assert atom.GetFormalCharge() == own.GetFormalCharge()
    # the stereo of the molecule is the stereo of the string (one centre, no stereo bond)
    assert [a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in parsed.GetAtoms()
            ].count(True) == 1


def test_centres_string_keeps_a_double_bond_configuration():
    mol = Chem.MolFromSmiles("C/C=C\\1/C(=NC=NC1=C)C")
    smiles, order = _centres_input(mol)
    parsed = _read_unsanitized(smiles)
    stereo = [b for b in parsed.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE]
    assert len(stereo) == 1
    ends = sorted((order[stereo[0].GetBeginAtomIdx()], order[stereo[0].GetEndAtomIdx()]))
    assert ends == [1, 2]       # the ethylidene C=C of the input, C-1 and C-2


@pytest.mark.parametrize("smiles", [
    "C[C@H](N)C(=O)O",                             # no ring
    "C[C@H](c1ccncc1)N",                           # a mancude ring beside a centre
    "C[C@@H]1CC[C@H](CC1)c1ccc2ncccc2c1",
    "C/C=C/c1ccncc1",
    "C=C1C(=C)N=CN=C1",                            # a non-mancude ring but no stereo unit
])
def test_the_ordinary_string_is_kept_without_held_atoms_or_stereo(smiles):
    mol = Chem.MolFromSmiles(smiles)
    text, order = _centres_input(mol)
    assert text == Chem.MolToSmiles(mol)
    assert order == [int(i) for i in mol.GetProp("_smilesAtomOutputOrder").strip("[]").split(",")
                     if i.strip()]


# ---------------------------------------------------------------------------
# The cache of held atoms and the count of parts not judged
# ---------------------------------------------------------------------------

_HELD_SMILES = (
    "C/C=C\\1/C(=NC=NC1=C)C",
    "C[C@@H]2N=C3C=CC=CC3=CN2c1ccccc1",
    "CC/C=C/1C(=C/C)N=C(C2=C1N(C(=N2)COCC)CC(C)(C)O)N",
    "[NH3+][C@@H](CC(=O)[O-])[C@@H]1N=C2C=CC=CC2=CN1C",
)


def _spellings(smiles, count, seed):
    mol = Chem.MolFromSmiles(smiles)
    rng = random.Random(seed)
    out = [mol]
    for _ in range(count):
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        out.append(Chem.RenumberAtoms(mol, order))
    return out


def test_a_served_input_is_the_one_computed_for_every_spelling():
    """The cache is keyed by the canonical SMILES and holds positions in it: a hit for one
    spelling of a molecule gives another spelling exactly what its own cold computation gives."""
    for smiles in _HELD_SMILES:
        spellings = _spellings(smiles, 6, 7)
        cold = []
        for mol in spellings:
            centres_bridge._HELD_CACHE.clear()
            cold.append(_centres_input(mol))
        centres_bridge._HELD_CACHE.clear()
        _centres_input(spellings[0])                 # filled by the first spelling only
        assert len(centres_bridge._HELD_CACHE) == 1
        for mol, expected in zip(spellings, cold):
            assert _centres_input(mol) == expected   # served for the others
            assert _centres_input(mol) == expected


def test_the_cache_is_bounded_and_keeps_the_most_recent(monkeypatch):
    monkeypatch.setattr(centres_bridge, "_HELD_CACHE_MAX", 2)
    centres_bridge._HELD_CACHE.clear()
    mols = [Chem.MolFromSmiles(s) for s in _HELD_SMILES[:3]]
    for mol in mols:
        _centres_input(mol)
    assert list(centres_bridge._HELD_CACHE) == [Chem.MolToSmiles(m) for m in mols[1:]]
    _centres_input(mols[1])                           # a hit makes it the most recent
    _centres_input(mols[0])
    assert list(centres_bridge._HELD_CACHE) == [Chem.MolToSmiles(mols[1]),
                                                Chem.MolToSmiles(mols[0])]
    centres_bridge._HELD_CACHE.clear()


def test_a_part_that_cannot_be_judged_is_counted():
    smiles = ("CCCCCCCCCCCC[C@@H]1CC(CC(C1)/C=C2/C3=CC(=C)SC3=C4C2=C/C(=C5/C6=NC(=C(N=C6C(=C)"
              "C7=NSN=C57)C)C)/S4)CCCCCCCCCCCC")
    STATS.clear()
    assert non_mancude_ring_atoms(Chem.MolFromSmiles(smiles)) == ()
    assert STATS["parts_not_judged"] >= 1
    STATS.clear()
    assert _held("C/C=C\\1/C(=NC=NC1=C)C") == (2, 7)
    assert not STATS
