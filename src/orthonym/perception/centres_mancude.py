"""Ring atoms the ``centres`` labeller must not average.

 "Mancude rings and ring systems" (the Blue Book): "Mancude rings, i.e.,
rings or ring systems having the maximum number of noncumulative double bonds, are treated
as Kekulé structures. For mancude heterocycles, each duplicate atom is given an atomic
number that is the mean of what it would have if the double bonds were located at each of
the possible positions." The mean belongs to mancude rings only; a ring whose double bonds
are fixed below that maximum keeps the atomic numbers of its own Kekulé structure.

``centres`` 1.2.1 (``com.simolecule.centres.Mancude``, read with ``javap -c``) does not
judge that. It gives each ring atom whose valence signature fits a mancude ring (``C``,
``Si``, ``Ge`` with one double bond and two single ones; ``N``, ``P``, ``As`` with one
double bond and one single one, or charged as an iminium; ``O+``) a type, drops by
repeated pruning the atoms left with a single typed neighbour, and takes the connected
groups of what remains (its "parts"). The duplicate atom that a double bond adds to an
atom of a part then gets the mean atomic number of that atom's neighbours in the part,
whatever the atom's double bond actually is (measured on a digraph dump: the duplicate
of ``N`` on a ring carbon of such a part carries 12/2, not 7). A ring atom that bears an
exocyclic double bond has the signature, so a ring with two ylidene or imino groups (two
ring double bonds against the three a six-membered mancude ring has) is averaged as if
it were mancude, and the labels beside it come out as those of another stereoisomer.
``rdCIPLabeler``, to which the engine falls back only when ``centres`` is unavailable (no
JVM), is a separate implementation that treats these rings in its own way (on the labels
this module changes it gives the Kekulé label for every centre and for about half of the
double bonds); this module does not reach that fallback.

The only per-atom inputs ``Mancude`` reads are the element, the valence and the formal
charge; no sequence rule reads a charge. ``getCharge`` / ``getFormalCharge`` is called by
``Mancude.SeedTypes``, by ``Digraph`` (for a negative charge on an averaged atom) and by
``CdkLabeller.handleSruRepeat`` (the copy of a ctab structure-repeat-unit Sgroup, never
reached from a SMILES). So the bridge tells
``centres`` that an atom is not part of a mancude system by raising its formal charge by
one in the string it labels (``centres_bridge.centres_label_mol``): no typed signature
has a charge one higher, and the pruning then removes what hangs on it.

This module decides which atoms those are. It repeats ``SeedTypes``, ``RelaxTypes`` and
``VisitParts`` (the check of that repetition is in
``tests/integration/test_centres_mancude_parts.py``), and for every part in which the mean
would change an atomic number asks the definition above: the part is mancude when the
double bonds the structure holds inside it are a maximum matching of its atoms (the
predicate of ``perception.mancude``). Of a part that is not, the atoms whose double bond
leaves it (the ylidene and imino carbons) are fixed in place; the pruning is run again
and the remaining parts are asked in turn, so a benzo ring beside a quinoid ring keeps
its own mean.
"""
import logging
from collections import Counter, deque
from typing import Dict, List, Set, Tuple

from rdkit import Chem

from .mancude import max_matching_size

logger = logging.getLogger(__name__)

#: Parts left alone because they cannot be judged, by cause (``parts_not_judged``: the number
#: of double bonds the part holds would depend on the Kekulé structure chosen, or an atom has
#: two fixed double bonds; ``parts_odd``: its atoms cannot all be matched). The bridge serves
#: repeats of one molecule from a cache, so each is counted once per molecule and process.
STATS: Counter = Counter()

#: ``centres`` seeds ``C``, ``Si`` and ``Ge`` alike (atomic numbers 6, 14, 32), ``N``,
#: ``P`` and ``As`` alike (7, 15, 33), and ``O`` alone (8).
_C_LIKE = frozenset((6, 14, 32))
_N_LIKE = frozenset((7, 15, 33))

#: ``Mancude.SeedTypes`` adds 1 per single bond (and per hydrogen), 256 per double bond,
#: 65536 per triple bond and 16777216 for any other order; a type is a (charge, sum) pair.
_ORDER_WEIGHT = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 256,
                 Chem.BondType.TRIPLE: 65536}
_OTHER_ORDER_WEIGHT = 16777216

#: The anionic seed types (``Cv3D3Minus``, ``Nv2D2Minus``): ``centres`` averages their
#: parts over the lone pair, a case this module leaves to it.
_ANIONIC = frozenset(('C-', 'N-'))


def _seed_types(mol, kek) -> Dict[int, str]:
    """``Mancude.SeedTypes``: the typed atoms of ``mol`` (an atom absent from the result
    has type ``Other``). ``kek`` is a Kekulé copy of ``mol``; ring bonds are those of
    ``mol``."""
    types: Dict[int, str] = {}
    ring_bond = {b.GetIdx() for b in mol.GetBonds() if b.IsInRing()}
    for atom in kek.GetAtoms():
        bonds = atom.GetBonds()
        if not any(b.GetIdx() in ring_bond for b in bonds):
            continue
        v = atom.GetTotalNumHs()
        for b in bonds:
            v += _ORDER_WEIGHT.get(b.GetBondType(), _OTHER_ORDER_WEIGHT)
        z, q = atom.GetAtomicNum(), atom.GetFormalCharge()
        if z in _C_LIKE:
            if q == 0 and v == 258:
                types[atom.GetIdx()] = 'C'
            elif q == -1 and v == 3:
                types[atom.GetIdx()] = 'C-'
        elif z in _N_LIKE:
            if q == 0 and v == 257:
                types[atom.GetIdx()] = 'N'
            elif q == -1 and v == 2:
                types[atom.GetIdx()] = 'N-'
            elif q == 1 and v == 258:
                types[atom.GetIdx()] = 'N+'
        elif z == 8 and q == 1 and v == 257:
            types[atom.GetIdx()] = 'O+'
    return types


def _relax(types: Dict[int, str], nbrs: List[List[int]]) -> None:
    """``Mancude.RelaxTypes``: an atom with exactly one typed neighbour (over every bond,
    ring or not) becomes ``Other``, and that is repeated for its neighbours."""
    count = [sum(1 for o in around if o in types) for around in nbrs]
    queue = deque(a for a, c in enumerate(count) if c == 1)
    while queue:
        a = queue.popleft()
        if a not in types:
            continue
        del types[a]
        for o in nbrs[a]:
            count[o] -= 1
            if count[o] == 1:
                queue.append(o)


def _parts(types: Dict[int, str], ring_nbrs: List[List[int]]) -> List[List[int]]:
    """``Mancude.VisitParts``: the connected groups of typed atoms over ring bonds, each
    sorted, in order of their lowest atom."""
    seen: Set[int] = set()
    parts: List[List[int]] = []
    for start in sorted(types):
        if start in seen:
            continue
        seen.add(start)
        part, stack = [start], [start]
        while stack:
            a = stack.pop()
            for o in ring_nbrs[a]:
                if o in types and o not in seen:
                    seen.add(o)
                    part.append(o)
                    stack.append(o)
        parts.append(sorted(part))
    return parts


def _flexible(mol, kek, i: int) -> bool:
    """Atom ``i`` takes a double bond in every Kekulé structure and ``mol`` does not fix
    it (its double bond is on an aromatic bond)."""
    double = Chem.BondType.DOUBLE
    return (not any(b.GetBondType() == double for b in mol.GetAtomWithIdx(i).GetBonds())
            and any(b.GetBondType() == double for b in kek.GetAtomWithIdx(i).GetBonds()))


def _can_pair_across(mol, kek, a: int, o: int) -> bool:
    """Some Kekulé structure puts the double bond of ``a`` on its aromatic bond to ``o``.

    The Kekulé structures of the flexible atoms joined to ``a`` by aromatic bonds are the
    perfect matchings of that group; one uses the bond ``a``-``o`` exactly when the group
    without the two has one."""
    group, stack = {a}, [a]
    while stack:
        x = stack.pop()
        for b in mol.GetAtomWithIdx(x).GetBonds():
            if b.GetBondType() != Chem.BondType.AROMATIC:
                continue
            y = b.GetOtherAtomIdx(x)
            if y not in group and _flexible(mol, kek, y):
                group.add(y)
                stack.append(y)
    rest = group - {a, o}
    adjacent = {x: [b.GetOtherAtomIdx(x) for b in mol.GetAtomWithIdx(x).GetBonds()
                    if b.GetBondType() == Chem.BondType.AROMATIC
                    and b.GetOtherAtomIdx(x) in rest] for x in rest}
    return 2 * max_matching_size(rest, adjacent) == len(rest)


def _fixed_atoms(mol, kek, part: List[int], members: Set[int]):
    """``{atom of part whose double bond leaves it: atomic number at the other end}``, or
    None when that cannot be said.

    A double bond ``mol`` holds as such (not an aromatic bond) is fixed: to an atom of the
    part it lies inside the part, to any other atom it leaves it. An atom with no such
    bond has its double bond on an aromatic bond, which a Kekulé structure may put on any
    of them. When a part has an atom with a fixed double bond that leaves it, each such
    atom must have that bond on an aromatic bond inside the part in every Kekulé structure
    (``_can_pair_across``); otherwise the choice of Kekulé structure would decide how many
    double bonds the part holds, and None is returned."""
    leaving: Dict[int, int] = {}
    double = Chem.BondType.DOUBLE
    for a in part:
        atom = mol.GetAtomWithIdx(a)
        doubles = [b for b in atom.GetBonds() if b.GetBondType() == double]
        if len(doubles) > 1:
            return None
        if doubles and doubles[0].GetOtherAtomIdx(a) not in members:
            leaving[a] = mol.GetAtomWithIdx(doubles[0].GetOtherAtomIdx(a)).GetAtomicNum()
    if not leaving:
        return leaving
    for a in part:
        if a in leaving or not _flexible(mol, kek, a):
            continue
        inside = False
        for b in mol.GetAtomWithIdx(a).GetBonds():
            if b.GetBondType() != Chem.BondType.AROMATIC:
                continue
            o = b.GetOtherAtomIdx(a)
            if o in members:
                inside = True
            elif _flexible(mol, kek, o) and _can_pair_across(mol, kek, a, o):
                return None
        if not inside:
            return None
    return leaving


def _non_mancude_part_atoms(mol, kek, part: List[int], types: Dict[int, str],
                            ring_nbrs: List[List[int]]) -> List[int]:
    """The atoms to fix in place when ``part`` is not mancude: those whose
    double bond leaves it. Empty when the part is mancude, when its mean would change no
    atomic number (every atom of the part and every atom at the far end of a leaving
    double bond is one element), or when it cannot be judged."""
    if any(types[a] in _ANIONIC for a in part):
        return []
    members = set(part)
    leaving = _fixed_atoms(mol, kek, part, members)
    if leaving is None:
        STATS['parts_not_judged'] += 1
        logger.debug("centres_mancude: a part of %d atoms is not judged (its Kekule structure "
                     "is ambiguous)", len(part))
        return []
    if not leaving:
        return []
    if (len(part) - len(leaving)) % 2:
        STATS['parts_odd'] += 1
        logger.debug("centres_mancude: a part of %d atoms with %d leaving is not judged "
                     "(odd)", len(part), len(leaving))
        return []
    elements = {mol.GetAtomWithIdx(a).GetAtomicNum() for a in part}
    if len(elements | set(leaving.values())) < 2:
        return []
    held = (len(part) - len(leaving)) // 2
    adjacent = {a: [o for o in ring_nbrs[a] if o in members] for a in part}
    if max_matching_size(part, adjacent) > held:
        return sorted(leaving)
    return []


def _may_leave_a_part(mol) -> bool:
    """A cheap screen before the Kekulé copy: some ring atom has a double bond (the only
    bond that can leave a part, see ``_fixed_atoms``) and two neighbours that can take
    part in a ring double-bond system (aromatic, or with a double bond of their own); a
    ring atom with fewer is pruned and no part holds it."""
    info = mol.GetRingInfo()
    double = Chem.BondType.DOUBLE
    for atom in mol.GetAtoms():
        if not info.NumAtomRings(atom.GetIdx()):
            continue
        if not any(b.GetBondType() == double for b in atom.GetBonds()):
            continue
        like = sum(1 for n in atom.GetNeighbors()
                   if n.GetIsAromatic() or any(b.GetBondType() == double for b in n.GetBonds()))
        if like >= 2:
            return True
    return False


def non_mancude_ring_atoms(mol) -> Tuple[int, ...]:
    """Atom indices of ``mol`` that ``centres`` would average as part of a mancude system
    but that are not part of one, sorted; empty when there are none.

    Never raises: a molecule the analysis cannot read gets the empty answer, which leaves
    ``centres`` as it was."""
    try:
        return _non_mancude_ring_atoms(mol)
    except Exception:
        return ()


def _non_mancude_ring_atoms(mol) -> Tuple[int, ...]:
    if not _may_leave_a_part(mol):
        return ()
    kek = Chem.Mol(mol)
    Chem.Kekulize(kek, clearAromaticFlags=True)
    seed = _seed_types(mol, kek)
    if all(t == 'C' for t in seed.values()):
        return ()          # centres averages nothing when every typed atom is a carbon
    nbrs = [[o.GetIdx() for o in atom.GetNeighbors()] for atom in mol.GetAtoms()]
    ring_nbrs = [[] for _ in nbrs]
    for b in mol.GetBonds():
        if b.IsInRing():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            ring_nbrs[i].append(j)
            ring_nbrs[j].append(i)
    fixed: Set[int] = set()
    alone: Set[int] = set()
    first = True
    while True:
        types = {a: t for a, t in seed.items() if a not in fixed}
        _relax(types, nbrs)
        parts = _parts(types, ring_nbrs)
        if first:
            alone = {p[0] for p in parts if len(p) == 1}
            first = False
        new: Set[int] = set()
        for part in parts:
            if len(part) == 1:
                # a typed atom left with no neighbour in its part would carry no mean
                # (centres gives it 0/0): fixing the neighbours made it so, so it is
                # fixed too, unless it was alone before anything was fixed (an anion is
                # never touched: its charge is its own)
                if part[0] not in alone and types[part[0]] not in _ANIONIC:
                    new.add(part[0])
            else:
                new.update(_non_mancude_part_atoms(mol, kek, part, types, ring_nbrs))
        new -= fixed
        if not new:
            return tuple(sorted(fixed))
        fixed |= new
