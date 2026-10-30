"""The open hydrocarbon series of: polyacenes, polyaphenes, polyalenes,
polyphenylenes, polynaphthylenes and polyhelicenes.

 "Systematically named hydrocarbon parent components" (the Blue Book)
names a parent component of the maximum number of noncumulative double bonds by "a
prefix and an ending or term representing the nature and arrangement of the component
rings". Each rule defines a CLASS, one member for each count of rings, so none of them
is a list:

- Polyacenes (:11438): "four or more ortho-fused benzene rings in a straight
  linear arrangement... a numerical prefix ('tetra', 'penta', etc.)... followed by the
  ending 'acene'... with elision of a letter 'a'" -- tetracene, pentacene,...;
- Polyaphenes (:11444): "n ortho-fused benzene rings (n > 3) forming two
  straight linear arrangements of (n + 1)/2 rings (if n is odd) or n/2 and (n/2) + 1
  rings (if n is even) with a common benzene ring and that make a formal angle of 120
  degrees with each other... 'phene'" -- tetraphene, pentaphene,...;
- Polyalenes (:11455): "two identical ortho-fused monocyclic hydrocarbon
  rings... the numerical prefix ('penta', 'hepta', etc.) that denotes the number of
  carbon atoms in each ring followed by the ending 'alene'... with elision of a letter
  'a'. The name naphthalene is retained" -- pentalene, heptalene, octalene,... (the bare
  ring system from octalene on: see ``_substituted_polyalene``);
- Polyphenylenes (:11461): "a monocyclic hydrocarbon with an even number of
  carbon atoms and benzene rings ortho-fused to alternate sides... 'phenylene'. The
  traditional name biphenylene is retained" -- biphenylene, triphenylene, tetraphenylene,
  ...;
- Polynaphthylenes (:11469): "a monocyclic hydrocarbon with an even number of
  carbon atoms ortho-fused on alternate sides to the 2,3-positions of naphthalene rings
  ... 'naphthylene'. The series begins with three naphthalene rings, trinaphthylene";
- Polyhelicenes (:11477): "a hydrocarbon parent component of six or more rings
  that consists of a benzene ring ortho-fused to the 3,4-position of phenanthrene and
  further benzene rings fused in a similar way... 'helicene'" -- hexahelicene,....

This module holds the rule once for every member: it recognises a member from the ring
graph of a fused ring system (:func:`classify_ring_system`,:func:`identify_series`), spells its
name (:func:`series_name`, read back by:func:`parse_series_name`), and numbers it
(:func:`series_entry`), so a member the tables never listed (octacene, a
pentadecaphenylene) is named, numbered and substituted by the same rule as tetracene.

The numbering of every series is (the Blue Book): "The numbering of
peripheral atoms in the preferred orientation starts from the uppermost ring... Numbering
starts from the nonfused atom most counterclockwise in the ring selected and proceeds in a
clockwise direction around the system... Each fusion carbon atom is given the same number
as the immediately preceding nonfusion skeletal atom, modified by a Roman letter 'a', 'b',
'c', 'd', etc." Polyacenes, polyaphenes and polyalenes take the orientation and numbering
engine of ``rules/fusion_numbering.py``. That engine cannot place a ring of eight or more
members next to benzene rings (polyphenylenes, polynaphthylenes) or a helicene (which
overlaps itself in the plane), so for those three the start atom of the walk is fixed here
from the Blue Book's own statement of the orientation:

- polyphenylenes and polynaphthylenes: every benzene or naphthalene ring is equivalent, so
  the walk starts at the first of the four consecutive CH of a terminal benzene ring;
- polyhelicenes (:12531): "A helicene is oriented so that a terminal ring is located in
  the upper right quadrant; numbering always begins in this terminal ring" -- the walk
  starts in a terminal ring and runs along the outside of the helix first ('hexahelicene
  (PIN)',:12517 figure; 1-4, 4a, 5, 6, 6a... 16, 16a... 16e).

The numbering of every member up to 20 rings was read against OPSIN 2.9.0's own locants
(``<name> -o extendedsmi``) when this was written; the unit tests pin a few members.
"""

from __future__ import annotations

import functools
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Set, Tuple

from rdkit import Chem

from ..data.chain_names import multiplying_prefix
from .fusion_numbering import (
    _assign_from_start,
    _peripheral_cycle,
    compute_fused_numbering,
)

# OPSIN 2.9.0 reads a simple numerical prefix only up to 50 ('pentaconta'); a longer
# member cannot be read back, so it is not a member this module names.
_MAX_N = 50

# series -> (the ending that follows the numerical prefix, the fewest rings / carbon atoms).
# 'acene' and 'alene' take the ending with the 'a' of the prefix elided,
#, so the prefix 'tetra' + 'acene' is spelled 'tetracene'.
_FORMS: Dict[str, Tuple[str, int]] = {
    'acene': ('cene', 4),
    'aphene': ('phene', 4),
    'alene': ('lene', 5),
    'phenylene': ('phenylene', 2),
    'naphthylene': ('naphthylene', 3),
    'helicene': ('helicene', 6),
}

# A polyalene of two rings of 5 or 7 members keeps the producers that name it today
# (data/fused_heterocycles.py and data/retained_names.py hold pentalene and heptalene, with a
# numbering tied to the drawing), and of the open class only the bare ring system is
# recognised (``_substituted_polyalene``); the class takes over from octalene on.
_ALENE_FIRST_RECOGNISED = 8


class SeriesSpec(NamedTuple):
    """One member of an open series: ``n`` is the count the numerical prefix cites (rings,
    or the carbon atoms of each ring of a polyalene)."""

    series: str
    n: int


class SeriesHit(NamedTuple):
    """A ring system of a molecule that is a member of a series."""

    name: str
    spec: SeriesSpec
    atoms: FrozenSet[int]
    rings: int


def _numerical_prefix(series: str, n: int) -> Optional[str]:
    if series == 'phenylene' and n == 2:
        return 'bi'  #: "biphenylene (PIN) (not diphenylene)" (:11465)
    try:
        return multiplying_prefix(n)
    except ValueError:
        return None


@functools.lru_cache(maxsize=None)
def _prefix_to_n() -> Dict[str, int]:
    table = {}
    for n in range(3, _MAX_N + 1):
        table[multiplying_prefix(n)] = n
    table['bi'] = 2
    return table


def _valid(series: str, n: int) -> bool:
    ending_minimum = _FORMS[series][1]
    if n < ending_minimum or n > _MAX_N:
        return False
    if series == 'alene' and n == 6:
        return False  # six carbon atoms in each ring is naphthalene, retained (:11455)
    return True


def series_name(series: str, n: int) -> Optional[str]:
    """The name of a member, or ``None`` when ``n`` is not a member of ``series``."""
    if series not in _FORMS or not _valid(series, n):
        return None
    prefix = _numerical_prefix(series, n)
    if prefix is None:
        return None
    return prefix + _FORMS[series][0]


def parse_series_name(name: str) -> Optional[SeriesSpec]:
    """The member a name spells, or ``None`` (the inverse of:func:`series_name`)."""
    if not name:
        return None
    table = _prefix_to_n()
    for series, (ending, _minimum) in _FORMS.items():
        if not name.endswith(ending):
            continue
        n = table.get(name[:-len(ending)])
        if n is None or not _valid(series, n):
            continue
        if series_name(series, n) == name:
            return SeriesSpec(series, n)
    return None


# ---------------------------------------------------------------------------
# Recognition of a member from a ring system of a molecule
# ---------------------------------------------------------------------------

def _is_mancude_hydrocarbon(mol: Chem.Mol, atoms: Set[int]) -> bool:
    """True iff every atom of the ring system is an uncharged carbon with three
    neighbours (counting hydrogen) whose bonds to atoms outside the system are single,
    i.e. a system of the maximum number of noncumulative double bonds with no ring
    position that is saturated, charged or double-bonded to a substituent."""
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge() != 0
                or atom.GetNumRadicalElectrons() != 0 or atom.GetTotalDegree() != 3):
            return False
        for bond in atom.GetBonds():
            other = bond.GetOtherAtomIdx(idx)
            if other not in atoms and bond.GetBondType() != Chem.BondType.SINGLE:
                return False
    return True


def _fused_components(rings: List[Tuple[int, ...]]) -> List[List[int]]:
    """Indices of the rings of each fused ring system (rings that share at least two atoms)."""
    parent = list(range(len(rings)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sets = [set(r) for r in rings]
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(sets[i] & sets[j]) >= 2:
                parent[find(j)] = find(i)
    groups: Dict[int, List[int]] = {}
    for i in range(len(rings)):
        groups.setdefault(find(i), []).append(i)
    return [groups[k] for k in sorted(groups)]


def _bond_index(cycle: List[int], a: int, b: int) -> Optional[int]:
    """The position ``t`` with ``{cycle[t], cycle[t + 1]} == {a, b}``."""
    n = len(cycle)
    for t in range(n):
        if {cycle[t], cycle[(t + 1) % n]} == {a, b}:
            return t
    return None


def _classify_path(mol: Chem.Mol, rings: List[Tuple[int, ...]],
                   adj: Dict[int, List[int]], shared: Dict[Tuple[int, int], Tuple[int, int]]
                   ) -> Optional[SeriesSpec]:
    """Polyacene, polyaphene or polyhelicene: all-six-membered rings in a chain."""
    k = len(rings)
    if any(len(r) != 6 for r in rings) or any(len(v) > 2 for v in adj.values()):
        return None
    ends = sorted(i for i in range(k) if len(adj[i]) == 1)
    if len(ends) != 2:
        return None
    order = [ends[0]]
    while len(order) < k:
        nxt = [j for j in adj[order[-1]] if j not in order]
        if len(nxt) != 1:
            return None
        order.append(nxt[0])

    # Orient every ring cycle the same way round the plane: a ring traverses the bond it
    # shares with the previous ring in the opposite direction to that ring (two faces
    # that share an edge run along it in opposite senses). Then the distance, in bonds,
    # between the two fusion bonds of an inner ring says how the chain turns there: 3 is
    # straight on (linear), 2 and 4 are the 120-degree turns to the one side or the other.
    cycles: List[List[int]] = []
    for pos, ring_index in enumerate(order):
        cyc = list(rings[ring_index])
        if pos:
            a, b = shared[(order[pos - 1], ring_index)]
            prev = cycles[-1]
            t = _bond_index(prev, a, b)
            x, y = prev[t], prev[(t + 1) % 6]
            q = cyc.index(y)
            if cyc[(q + 1) % 6] != x:
                cyc.reverse()
        cycles.append(cyc)

    turns: List[int] = []
    for pos in range(1, k - 1):
        cyc = cycles[pos]
        i = _bond_index(cyc, *shared[(order[pos - 1], order[pos])])
        j = _bond_index(cyc, *shared[(order[pos], order[pos + 1])])
        if i is None or j is None:
            return None
        turns.append((j - i) % 6)
    if any(t in (1, 5, 0) for t in turns):
        return None

    angular = [pos for pos, t in enumerate(turns) if t != 3]
    if not angular:
        return SeriesSpec('acene', k) if k >= 4 else None
    if len(angular) == 1:
        # inner ring number ``angular[0] + 1`` (0-based) is the common ring; the arms count it
        arm1, arm2 = angular[0] + 2, k - angular[0] - 1
        if k >= 4 and abs(arm1 - arm2) <= 1:
            return SeriesSpec('aphene', k)
        return None
    if len(set(turns)) == 1 and k >= 6:
        return SeriesSpec('helicene', k)
    return None


def _classify_hub(mol: Chem.Mol, rings: List[Tuple[int, ...]],
                  adj: Dict[int, List[int]], shared: Dict[Tuple[int, int], Tuple[int, int]]
                  ) -> Optional[SeriesSpec]:
    """Polyphenylene or polynaphthylene: benzene (or naphthalene) rings ortho-fused to
    alternate sides of one monocyclic ring of twice as many carbon atoms."""
    k = len(rings)
    for hub in range(k):
        degree = len(adj[hub])
        if degree < 2 or len(rings[hub]) != 2 * degree:
            continue
        arms = adj[hub]
        # the rings fused to the hub sit on alternate bonds of it
        cyc = list(rings[hub])
        sites = [_bond_index(cyc, *shared[(hub, r)]) for r in arms]
        if any(t is None for t in sites):
            continue
        if len(set(sites)) != degree or len({t % 2 for t in sites}) != 1:
            continue
        if any(len(rings[r]) != 6 for r in arms):
            continue
        if all(len(adj[r]) == 1 for r in arms) and k == degree + 1:
            return SeriesSpec('phenylene', degree)
        if degree >= 3 and k == 2 * degree + 1 and all(len(adj[r]) == 2 for r in arms):
            linear = True
            for r in arms:
                (pendant,) = [x for x in adj[r] if x != hub]
                if len(adj[pendant]) != 1 or len(rings[pendant]) != 6:
                    linear = False
                    break
                i = _bond_index(list(rings[r]), *shared[(r, hub)])
                j = _bond_index(list(rings[r]), *shared[(r, pendant)])
                if i is None or j is None or (j - i) % 6 != 3:
                    linear = False
                    break
            if linear:
                return SeriesSpec('naphthylene', degree)
    return None


def classify_ring_system(mol: Chem.Mol, ring_indices: List[int]) -> Optional[SeriesSpec]:
    """The series member that the rings ``ring_indices`` (indices into the molecule's
    ``AtomRings``) form, or ``None``. The rings must be a fused ring system with the
    maximum number of noncumulative double bonds and no ring atom shared by three rings."""
    all_rings = mol.GetRingInfo().AtomRings()
    rings = [tuple(all_rings[i]) for i in ring_indices]
    k = len(rings)
    if k < 2 or k > 2 * _MAX_N + 1:
        return None
    atoms: Set[int] = set()
    count: Dict[int, int] = {}
    for r in rings:
        for a in r:
            atoms.add(a)
            count[a] = count.get(a, 0) + 1
    if max(count.values()) >= 3 or not _is_mancude_hydrocarbon(mol, atoms):
        return None

    adj: Dict[int, List[int]] = {i: [] for i in range(k)}
    shared: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for i in range(k):
        for j in range(i + 1, k):
            common = set(rings[i]) & set(rings[j])
            if not common:
                continue
            if len(common) != 2:
                return None
            a, b = sorted(common)
            if mol.GetBondBetweenAtoms(a, b) is None:
                return None
            adj[i].append(j)
            adj[j].append(i)
            shared[(i, j)] = shared[(j, i)] = (a, b)

    spec: Optional[SeriesSpec] = None
    if k == 2:
        if len(rings[0]) == len(rings[1]) and len(rings[0]) >= _ALENE_FIRST_RECOGNISED:
            spec = SeriesSpec('alene', len(rings[0]))
    else:
        spec = _classify_path(mol, rings, adj, shared) or _classify_hub(mol, rings, adj, shared)
    if spec is None or series_entry(series_name(spec.series, spec.n)) is None:
        return None
    return spec


def _substituted_polyalene(spec: SeriesSpec, mol: Chem.Mol, atoms: FrozenSet[int]) -> bool:
    """A polyalene (the series of localized double bonds) is recognised only as the bare ring
    system. Its bond-shift drawings are different compounds that 'Localized double
    bonds' (the Blue Book) tells apart with a Delta descriptor, so which locant a
    substituent takes depends on the drawing, a question the numbering of this module does not
    answer (the catalogue entries of pentalene and heptalene decide it from the drawing:
    ``fused_heterocycles.numbering_held_by_drawn_alternation``). The benzenoid series have one
    Kekule-independent numbering."""
    return spec.series == 'alene' and len(atoms) != mol.GetNumAtoms()


def identify_series(mol: Chem.Mol) -> Optional[SeriesHit]:
    """The largest ring system of ``mol`` that is a member of a series, or ``None``.

    The systems are the fused ring systems of the molecule. Substituents are free to be
    anything attached by a single bond; a ring system that is saturated in part, bears a
    double bond to a substituent, or carries an atom that is not carbon is never a member. No member is
    found inside ``isotope_alternative.alternative_skeleton``."""
    from .isotope_alternative import alternative_skeleton_active
    if alternative_skeleton_active():
        return None
    rings = mol.GetRingInfo().AtomRings()
    if len(rings) < 2:
        return None
    hits: List[SeriesHit] = []
    for comp in _fused_components(list(rings)):
        if len(comp) < 2:
            continue
        spec = classify_ring_system(mol, comp)
        atoms = frozenset(a for i in comp for a in rings[i])
        if spec is not None and not _substituted_polyalene(spec, mol, atoms):
            hits.append(SeriesHit(series_name(spec.series, spec.n), spec, atoms, len(comp)))
    if not hits:
        return None
    # (d) the greater number of rings, then (e) the greater number of skeletal atoms
    # (the Blue Book); systems equal in both are told apart by name, never by atom order
    return min(hits, key=lambda h: (-h.rings, -len(h.atoms), h.name))


def ring_system_series(mol: Chem.Mol, atoms) -> Optional[SeriesHit]:
    """The member that the fused ring system with exactly these ``atoms`` is, or ``None``."""
    wanted = set(atoms)
    rings = mol.GetRingInfo().AtomRings()
    for comp in _fused_components(list(rings)):
        members = {a for i in comp for a in rings[i]}
        if members != wanted or len(comp) < 2:
            continue
        spec = classify_ring_system(mol, comp)
        if spec is None:
            return None
        return SeriesHit(series_name(spec.series, spec.n), spec, frozenset(members), len(comp))
    return None


def series_ring_system_atoms(mol: Chem.Mol, name: str) -> Optional[Set[int]]:
    """The atoms of the first ring system of ``mol`` that is the member ``name``."""
    spec = parse_series_name(name)
    if spec is None:
        return None
    rings = mol.GetRingInfo().AtomRings()
    for comp in _fused_components(list(rings)):
        if len(comp) >= 2 and classify_ring_system(mol, comp) == spec:
            return {a for i in comp for a in rings[i]}
    return None


# ---------------------------------------------------------------------------
# The skeleton of a member and its numbering
# ---------------------------------------------------------------------------

class _Skeleton:
    """A carbon skeleton built by ortho-fusing rings, bond orders not yet assigned."""

    def __init__(self) -> None:
        self.n = 0
        self.edges: Set[Tuple[int, int]] = set()

    def atom(self) -> int:
        self.n += 1
        return self.n - 1

    def bond(self, a: int, b: int) -> None:
        self.edges.add((min(a, b), max(a, b)))

    def ring(self, size: int) -> List[int]:
        atoms = [self.atom() for _ in range(size)]
        for i in range(size):
            self.bond(atoms[i], atoms[(i + 1) % size])
        return atoms

    def fuse(self, cycle: List[int], i: int, size: int) -> List[int]:
        """Fuse a ring on the bond (cycle[i], cycle[i + 1]); the new ring's cycle starts with it."""
        a, b = cycle[i % len(cycle)], cycle[(i + 1) % len(cycle)]
        new = [self.atom() for _ in range(size - 2)]
        chain = [b] + new + [a]
        for x, y in zip(chain, chain[1:]):
            self.bond(x, y)
        return [a, b] + new

    def kekule_smiles(self) -> Optional[str]:
        adj: Dict[int, List[int]] = {i: [] for i in range(self.n)}
        for a, b in self.edges:
            adj[a].append(b)
            adj[b].append(a)
        partner = _perfect_matching(adj)
        if partner is None:
            return None
        rw = Chem.RWMol()
        for _ in range(self.n):
            rw.AddAtom(Chem.Atom(6))
        for a, b in sorted(self.edges):
            kind = Chem.BondType.DOUBLE if partner[a] == b else Chem.BondType.SINGLE
            rw.AddBond(a, b, kind)
        mol = rw.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
        return Chem.MolToSmiles(mol)


def _perfect_matching(adj: Dict[int, List[int]]) -> Optional[Dict[int, int]]:
    """A set of disjoint bonds that covers every atom (a Kekule structure), or ``None``."""
    partner: Dict[int, Optional[int]] = {v: None for v in adj}

    def solve() -> bool:
        best, options = None, None
        for v in adj:
            if partner[v] is not None:
                continue
            free = [w for w in adj[v] if partner[w] is None]
            if best is None or len(free) < len(options):
                best, options = v, free
                if len(free) <= 1:
                    break
        if best is None:
            return True
        for w in options:
            partner[best], partner[w] = w, best
            if solve():
                return True
            partner[best], partner[w] = None, None
        return False

    if not solve():
        return None
    return {v: w for v, w in partner.items() if w is not None}


def _template_smiles(spec: SeriesSpec) -> Optional[str]:
    n = spec.n
    s = _Skeleton()
    if spec.series == 'acene':
        cur = s.ring(6)
        cur = s.fuse(cur, 0, 6)
        for _ in range(n - 2):
            cur = s.fuse(cur, 3, 6)
    elif spec.series == 'aphene':
        #: arms of (n + 1)/2 rings each (n odd), or n/2 and n/2 + 1 (n even), the
        # common ring counted in both
        arm1, arm2 = ((n + 1) // 2, (n + 1) // 2) if n % 2 else (n // 2, n // 2 + 1)
        cur = s.ring(6)
        cur = s.fuse(cur, 0, 6)
        for _ in range(arm1 - 2):
            cur = s.fuse(cur, 3, 6)
        cur = s.fuse(cur, 2, 6)
        for _ in range(arm2 - 2):
            cur = s.fuse(cur, 3, 6)
    elif spec.series == 'helicene':
        cur = s.ring(6)
        cur = s.fuse(cur, 0, 6)
        for t in range(n - 2):
            # the cycles alternate in sense, so one sense of turn alternates between bond 2 and 4
            cur = s.fuse(cur, 2 if t % 2 == 0 else 4, 6)
    elif spec.series == 'alene':
        s.fuse(s.ring(n), 0, n)
    elif spec.series in ('phenylene', 'naphthylene'):
        hub = [s.atom() for _ in range(2 * n)]
        for i in range(2 * n):
            s.bond(hub[i], hub[(i + 1) % (2 * n)])
        for j in range(n):
            ring = s.fuse(hub, 2 * j, 6)
            if spec.series == 'naphthylene':
                s.fuse(ring, 3, 6)
    else:
        return None
    return s.kekule_smiles()


def _runs(periphery: List[int], fusion: Set[int]) -> List[Tuple[int, int]]:
    """The runs of consecutive non-fusion atoms around the periphery as (first position, length)."""
    n = len(periphery)
    runs = []
    for i in range(n):
        if periphery[i] in fusion or periphery[(i - 1) % n] not in fusion:
            continue
        length = 0
        while periphery[(i + length) % n] not in fusion:
            length += 1
        runs.append((i, length))
    return runs


def _walk_numbering(mol: Chem.Mol, spec: SeriesSpec) -> Optional[Dict[int, object]]:
    """ for the series the engine of ``fusion_numbering`` cannot orient."""
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    periphery = _peripheral_cycle(mol, ring_atoms)
    if periphery is None:
        return None
    count: Dict[int, int] = {}
    for ring in mol.GetRingInfo().AtomRings():
        for a in ring:
            count[a] = count.get(a, 0) + 1
    fusion = {a for a, c in count.items() if c >= 2}
    n = len(periphery)
    start = None
    for first, length in _runs(periphery, fusion):
        if length != 4:
            continue
        if spec.series == 'helicene':
            # outside of the helix first: the atom after the fusion atom that closes the
            # terminal ring is a CH, not another fusion atom (the inside of the helix).
            if periphery[(first + 5) % n] in fusion:
                continue
        start = (periphery[first], 1)
        break
    if start is None:
        # the run is entered from its other end when the periphery was read the other way
        for first, length in _runs(periphery[::-1], fusion):
            if length != 4:
                continue
            rev = periphery[::-1]
            if spec.series == 'helicene' and rev[(first + 5) % n] in fusion:
                continue
            start = (rev[first], -1)
            break
    if start is None:
        return None
    locants = _assign_from_start(mol, periphery, start[0], start[1], fusion)
    return locants or None


def _label(locant) -> Optional[object]:
    if isinstance(locant, tuple):
        base, letter = locant
        if len(letter) != 1 or not letter.isalpha() or letter > 'z':
            return None
        return f'{base}{letter}'
    return locant


def _generic_smarts(mol: Chem.Mol) -> str:
    rw = Chem.RWMol()
    for _ in range(mol.GetNumAtoms()):
        rw.AddAtom(Chem.Atom(6))
    for bond in mol.GetBonds():
        rw.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    return Chem.MolToSmarts(rw.GetMol()).replace('-', '~')


@functools.lru_cache(maxsize=None)
def series_entry(name: str) -> Optional[Dict[str, object]]:
    """A ``POLYCYCLIC_DATA``-shaped entry for a member, or ``None`` when ``name`` is not a
    member this module can build and number.

    The entry's ``canonical_smiles`` is a Kekule drawing of the bare member, its
    ``iupac_numbering`` keys are the atom indices of that SMILES, and ``generic_bonds`` says
    that a match onto a molecule must ignore bond orders: a member the Blue Book draws with
    localized double bonds (octalene) has more than one drawing, and the numbering is the
    same for all of them "Localized double bonds", the Blue Book)."""
    spec = parse_series_name(name)
    if spec is None:
        return None
    smiles = _template_smiles(spec)
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms()}
    if spec.series in ('acene', 'aphene', 'alene'):
        engine = compute_fused_numbering(mol, ring_atoms)
        if not engine:
            return None
        numbering = {i: _label(loc) for i, loc in engine.items()}
    else:
        walked = _walk_numbering(mol, spec)
        if not walked:
            return None
        numbering = {i: _label(loc) for i, loc in walked.items()}
    if len(numbering) != mol.GetNumAtoms() or any(v is None for v in numbering.values()):
        return None
    return {
        'canonical_smiles': smiles,
        'smarts': _generic_smarts(mol),
        'num_atoms': mol.GetNumAtoms(),
        'iupac_numbering': numbering,
        'substituent_positions': sorted(v for v in numbering.values() if isinstance(v, int)),
        'num_rings': mol.GetRingInfo().NumRings(),
        'generic_bonds': True,
        'series': spec,
    }


__all__ = [
    'SeriesHit', 'SeriesSpec', 'classify_ring_system', 'identify_series', 'parse_series_name',
    'ring_system_series', 'series_entry', 'series_name', 'series_ring_system_atoms',
]
