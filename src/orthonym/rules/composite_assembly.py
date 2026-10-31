"""An unbranched ring assembly of three to six identical cyclic systems, numbered with composite locants.

``## **** UNBRANCHED RING ASSEMBLIES OF THREE THROUGH SIX IDENTICAL CYCLIC SYSTEMS`` (the Blue Book),
``****`` (:15653): "Each cyclic system of the assembly is numbered consecutively and each ring or ring system is
numbered in its usual way. Composite locants (see are formed by citing the locants denoting positions in each
ring or ring system as superscripts to the locants indicating the position of a cyclic system in the assembly (ref. 26).
Locants indicating points of attachment are placed before the name of the assembly in ascending order; locants denoting
junctions are separated by a comma and sets of junction locants are separated by a colon." and (:15655) "This is a new
numbering system recommended for ring assemblies consisting of more than two rings or ring systems. The new system, which
uses composite locants composed of primary locants and superscript numbers attached to them as described above, is
recommended for preferred IUPAC names. The numbering method using serially primed locants used in earlier recommendations
(refs. 1 and 2) may be used in general nomenclature." ``### **** Preferred IUPAC names and numbering for ring
assemblies`` (:24070), ``****`` (:24078): "The preferred numbering for ring assemblies composed of three or more
identical cyclic systems uses composite locants rather than primed locants (see ". The PINs are
``1^1,2^1:2^4,3^1-terphenyl (PIN)`` (:20598), ``[1^1,2^1:2^4,3^1-terphenyl]-2^2-ol (PIN)`` (:27220),
``[1^1,2^1:2^4,3^1-terphenyl]-1^2,1^6,2^3,2^5-tetrol (PIN)`` (:26899) and ``1^2,2^2:2^6,3^2:3^6,4^2-quaterpyridine (PIN)``
(:15667); ``rules.ring_assemblies`` writes the primed names (``1,1':4',1''-terphenyl``) of general nomenclature and labels
them below the PIN.

The order of the locants. ``****`` (:15115), under ``### **** Numbering of phane parent hydrides``: "The
seniority of a composite locant is determined first on the basis of its primary locants... and, if these locants are
identical, on the basis of the complete composite locant itself, i.e., the primary locant and its superscripts." So the
ring of the assembly counts before the position in the ring: 1^9 is lower than 2^1, which is NOT the order of the primed
locants (2 < 2' < 3; the two orders choose differently for 2',4''-dichloro-1,1':4',1''-terphenyl, which is
1^4,2^3-dichloro-1^1,2^1:2^4,3^1-terphenyl). The book's tetrol (:26899) is consistent with it: of its two numberings the one
with the locant set 1^2,1^6,2^3,2^5 is the PIN, not 2^2,2^6,3^2,3^6 (the lower set if the superscripts alone were compared).
The criteria are those of the primed names: the junctions first (``****``,:15567: "Lowest possible locants must be used to denote the
positions of attachment"; ``### ****``:16109: "Low locants are assigned to ring junctions, then to free valences";
the tetraphenylterphenyl of:15729 gives the junction set 1^1,2^1:2^2,3^1 before the substituents), the set of the junctions
and then the junctions in the order they are written ('1^2,2^2:2^6,3^2:3^6,4^2-quaterpyridine',:15667, not 1^2,2^6:2^2,3^6:
3^2,4^2; ``****``:15663 "the locant set 1^1,2^1:2^2,3^1:3^3,4^1 is lower than the locant set 1^1,2^1:2^3,3^1:3^2,
4^1"), then the principal characteristic groups (``### **** NUMBERING`` (:3219) (c)), the detachable prefixes together
(f) and the prefix cited first in alphanumerical order (g).

The numbering. The ring systems are numbered along the chain, from either end (two orientations); each ring takes a
numbering of the usual kind: the numbering of the component ``rules.ring_assemblies`` gives it, composed with every
automorphism of the skeleton of the ring system (the numberings that differ by a symmetry of the component). The junction
atoms and the substituents of one ring are placed by the numbering of that ring alone, so the ring that has the lowest
criteria tuple for the junctions, the groups and the prefixes is chosen for every ring on its own, and the orientations are
then compared as wholes in the order above. A numbering is never chosen by the order of the atoms of the input.

The scope. An unbranched assembly of three to six rings joined by single bonds, of identical ring systems that are their
own parent hydrides (a benzene or other mancude monocycle or fused system, a saturated monocycle, with no indicated
hydrogen and no hydro prefix), with no stereo, charge, isotope or radical, whose substituents are one suffix-expressible
principal group (``-ol``, ``-amine``, ``-carboxylic acid``, ``-carbaldehyde``, ``-carbonitrile``: the most senior one is
the suffix and the others its prefixes) and detachable prefixes of the plain kinds (halogen, nitro, alkyl, and the alkoxy
groups with a retained name,. Every other molecule is declined and keeps the primed name. The name is offered only where the OPSIN in use reads composite
locants (``opsin_capabilities``), and only when that OPSIN reads it back to the full InChIKey of the molecule; with OPSIN
2.9.0 nothing here runs. Nothing is offered while the isotope decorator names the stripped skeleton of a labelled molecule
(``ring_assemblies.hydro_assembly_withheld``): its search for the locant of a descriptor tries no composite locant, so a
labelled assembly keeps its primed name.
"""
from __future__ import annotations

from functools import cmp_to_key
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

from rdkit import Chem

__all__ = ["MAX_RINGS", "MIN_RINGS", "composite_assembly_name"]

#: ```` ("THREE THROUGH SIX IDENTICAL CYCLIC SYSTEMS"): the sizes of an unbranched assembly named here.
MIN_RINGS, MAX_RINGS = 3, 6
#: Automorphisms of one ring system beyond which it is not numbered (declined).
MAX_AUTOMORPHISMS = 256


#: ``### **** Retained names`` (the Blue Book): the contracted names of R-O- groups that are preferred
#: prefixes; any other alkoxy group is cited as '(hexyloxy)', a spelling this module does not write
_ALKOXY = frozenset({'methoxy', 'ethoxy', 'propoxy', 'butoxy', 'tert-butoxy'})


class _Sub(NamedTuple):
    """A group on a ring atom: ``kind`` is its suffix kind (None for a plain prefix), ``name`` its prefix name."""
    system: int
    atom: int
    kind: Optional[str]
    name: Optional[str]


def composite_assembly_name(mol, assembly_info: Dict) -> Optional[str]:
    """The PIN of the unbranched ring assembly ``assembly_info`` (``rules.ring_assemblies.detect_ring_assembly``) of
    ``mol``, with composite locants, or None: the OPSIN in use does not read them, the molecule is outside the scope of
    the module (see its docstring), or the name does not read back to the molecule."""
    from ..opsin_capabilities import composite_assembly_enabled
    if not composite_assembly_enabled():
        return None
    from .ring_assemblies import _HYDRO_WITHHELD
    if _HYDRO_WITHHELD.get():
        # the isotope decorator is naming the stripped skeleton of a labelled molecule (``hydro_assembly_withheld``):
        # its search for the locants of the descriptor tries integer, primed and letter locants only
        # (``rules.isotopes``), never a composite one, so the descriptor could not enter a composite name
        return None
    try:
        name = _composite_name(mol, assembly_info)
    except Exception:  # noqa: BLE001 - a structure this reader cannot number is declined, and keeps its name
        return None
    if name is None:
        return None
    from .ring_assemblies import _hydro_certified
    return name if _hydro_certified(mol, name) else None


#: the nitro group as RDKit reads it, with its formal charges (the one charged group of the plain kinds)
_NITRO = Chem.MolFromSmarts("[N+](=O)[O-]")


def _plain(mol, systems: List[Set[int]]) -> bool:
    """One component; every atom neutral (but for the atoms of a nitro group), unlabelled, with no radical electron and
    no stereo; every ring atom belongs to a system of the chain (no other ring: the assembly is the whole ring skeleton,
    as ``****`` takes it)."""
    if len(Chem.GetMolFrags(mol)) != 1:
        return False
    charged = {i for match in mol.GetSubstructMatches(_NITRO) for i in match}
    for atom in mol.GetAtoms():
        if (atom.GetFormalCharge() and atom.GetIdx() not in charged) or atom.GetIsotope() \
                or atom.GetNumRadicalElectrons():
            return False
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return False
    if any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds()):
        return False
    chain = set().union(*systems)
    return all(a in chain for ring in mol.GetRingInfo().AtomRings() for a in ring)


def _chain(info: Dict):
    """``(junction, count)``: ``junction[(i, j)]`` is the atom of ring system ``i`` bonded to ring system ``j``, for the
    systems of ``info`` in the order of the chain; None when ``info`` is not an unbranched chain of 3-6 systems joined
    by single bonds that no flag of the detector marks as another kind of assembly (``branched`` is no such flag: it marks
    the chain of a tree and an assembly renumbered by ``rules.branched_assembly``, and ``_plain`` declines the tree, whose
    other rings are no part of the chain)."""
    if any(info.get(k) for k in ('double_bond_junction', 'mancude_tautomer', 'replacement', 'hydro')):
        return None
    systems, connections = info['ring_systems'], info['connections']
    count = len(systems)
    if not MIN_RINGS <= count <= MAX_RINGS or info.get('count') != count or len(connections) != count - 1:
        return None
    junction: Dict[Tuple[int, int], int] = {}
    for a1, a2, s1, s2 in connections:
        if abs(s1 - s2) != 1 or (s1, s2) in junction:
            return None
        junction[(s1, s2)], junction[(s2, s1)] = a1, a2
    return junction, count


def _component(mol, systems: List[Set[int]]) -> Optional[str]:
    """The name of the component (the same for every system), or None when it is not a parent hydride this module
    numbers: a name with indicated hydrogen, a hydro prefix or a von Baeyer descriptor, or a ring system that is
    neither mancude (every atom aromatic) nor a saturated monocycle."""
    from .ring_assemblies import _INDICATED_H_EMBEDDED_RE, _INDICATED_H_RE, _get_ring_parent_name
    names = {_get_ring_parent_name(mol, s) for s in systems}
    if len(names) != 1:
        return None
    name = names.pop()
    if not name or _INDICATED_H_RE.match(name) or _INDICATED_H_EMBEDDED_RE.search(name) or '[' in name \
            or 'hydro' in name:
        return None
    rings = mol.GetRingInfo().AtomRings()
    for system in systems:
        aromatic = all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in system)
        single = len([r for r in rings if set(r) <= system]) == 1 and all(
            b.GetBondType() == Chem.BondType.SINGLE for b in mol.GetBonds()
            if b.GetBeginAtomIdx() in system and b.GetEndAtomIdx() in system)
        if not (aromatic or single):
            return None
    return name


def _automorphisms(mol, atoms: List[int]) -> Optional[List[Dict[int, int]]]:
    """Every permutation of ``atoms`` that keeps each atom's element and each bond (its order) among them: the
    symmetries of the ring system, as ``{atom: image}``; None when there are more than:data:`MAX_AUTOMORPHISMS`."""
    inside = set(atoms)
    adjacent = {a: {n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() in inside} for a in atoms}
    element = {a: mol.GetAtomWithIdx(a).GetAtomicNum() for a in atoms}
    order = [min(atoms)]
    for a in order:                       # breadth first: every atom after the first has a placed neighbour
        order.extend(sorted(b for b in adjacent[a] if b not in order))
    found: List[Dict[int, int]] = []
    image: Dict[int, int] = {}

    def bond(a, b):
        return mol.GetBondBetweenAtoms(a, b).GetBondType()

    def place(i: int) -> bool:
        if i == len(order):
            found.append(dict(image))
            return len(found) <= MAX_AUTOMORPHISMS
        a = order[i]
        for c in sorted(set(atoms) - set(image.values())):
            if element[c] != element[a] or len(adjacent[c]) != len(adjacent[a]):
                continue
            if any((p in adjacent[a]) != (image[p] in adjacent[c])
                   or (p in adjacent[a] and bond(a, p) != bond(c, image[p])) for p in image):
                continue
            image[a] = c
            ok = place(i + 1)
            del image[a]
            if not ok:
                return False
        return True

    return found if place(0) else None


def _monocycle_numberings(mol, ring) -> List[Dict[int, int]]:
    """The numberings of a monocycle of the usual kind (``{atom: locant}``): any atom may be 1 and either way round for a
    carbocycle; for a heteromonocycle those that give the heteroatoms the lowest locants as a set and then, in the
    order O > S > Se > Te > N > P..., the lowest locants to the senior element, Hantzsch-Widman and
    retained ring names alike: 'pyridine' N1, '1,3-thiazole' S1 and N3)."""
    from ..data.hw_heteroatoms import get_heteroatom_priority
    size = len(ring)
    options = [{ring[(start + step * i) % size]: i + 1 for i in range(size)}
               for start in range(size) for step in (1, -1)]
    hetero = [a for a in ring if mol.GetAtomWithIdx(a).GetAtomicNum() != 6]
    if not hetero:
        return options

    def lowest(numbering):
        senior_first = sorted(hetero, key=lambda a: (get_heteroatom_priority(mol.GetAtomWithIdx(a).GetSymbol()),
                                                     numbering[a]))
        return sorted(numbering[a] for a in hetero), [numbering[a] for a in senior_first]
    best = min(lowest(o) for o in options)
    return [o for o in options if lowest(o) == best]


def _families(mol, info: Dict) -> Optional[List[List[Dict[int, int]]]]:
    """For each ring system the numberings of its usual kind: ``{atom: locant}``. A monocycle takes
    :func:`_monocycle_numberings`. A fused system takes the numbering of the component that
    ``ring_assemblies._compute_per_system_ring_locants`` gives it composed with each symmetry of the ring system (an
    atom with no locant, a fusion atom, has none in any of them)."""
    from .ring_assemblies import _compute_per_system_ring_locants
    rings = mol.GetRingInfo().AtomRings()
    base = None
    families = []
    for k, system in enumerate(info['ring_systems']):
        inside = [r for r in rings if set(r) <= system]
        if len(inside) == 1:
            families.append(_monocycle_numberings(mol, list(inside[0])))
            continue
        base = base if base is not None else _compute_per_system_ring_locants(mol, info)
        symmetries = _automorphisms(mol, sorted(system))
        if not base or len(base) != len(info['ring_systems']) or not base[k] or symmetries is None:
            return None
        seen, family = set(), []
        for sigma in symmetries:
            numbering = {a: base[k][sigma[a]] for a in system if sigma[a] in base[k]}
            marker = tuple(sorted(numbering.items()))
            if marker not in seen:
                seen.add(marker)
                family.append(numbering)
        families.append(family)
    return families


def _groups(mol, info: Dict) -> Optional[List[_Sub]]:
    """The groups on the ring atoms, named: one suffix kind (the most senior present, every group of that kind), the
    others prefixes of the plain kinds. None when a group is attached by a multiple bond or has no such name."""
    from .ring_assemblies import (
        _RING_ASSEMBLY_PCG_SENIORITY_KEY,
        _mixed_ring_assembly_prefix_name,
        _ring_assembly_pcg_suffix,
        _walk_substituent,
    )
    from .seniority import compare_seniority
    systems = info['ring_systems']
    ring_atoms = set().union(*systems)
    owner = {a: k for k, system in enumerate(systems) for a in system}
    found, taken = [], set()
    for a in sorted(ring_atoms):
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            n = nb.GetIdx()
            if n in ring_atoms:
                continue
            atoms = _walk_substituent(mol, n, ring_atoms)
            if mol.GetBondBetweenAtoms(a, n).GetBondType() != Chem.BondType.SINGLE or taken & set(atoms):
                return None
            taken.update(atoms)
            found.append((owner[a], a, n, atoms, _ring_assembly_pcg_suffix(mol, atoms, n)))
    present = {kind for *_, kind in found if kind is not None}
    senior = min(present, key=cmp_to_key(
        lambda x, y: compare_seniority(_RING_ASSEMBLY_PCG_SENIORITY_KEY[x], _RING_ASSEMBLY_PCG_SENIORITY_KEY[y]))
    ) if present else None
    groups = []
    for system, a, n, atoms, kind in found:
        if kind is not None and kind == senior:
            groups.append(_Sub(system, a, kind, None))
            continue
        name = _mixed_ring_assembly_prefix_name(mol, atoms, n, kind)
        if not name or 'unknown' in name:
            return None
        if kind is None and mol.GetAtomWithIdx(n).GetSymbol() == 'O' and name not in _ALKOXY:
            return None                       # an R-O- group whose name is not a retained one
        groups.append(_Sub(system, a, None, name))
    return groups


def _ring_pick(family, system: int, junctions: List[int], groups: List[_Sub], ranks):
    """``(key, numbering)`` of the numbering of one ring with the lowest criteria tuple: the set of its junction locants,
    the junctions in the order they are written (the junction to the previous ring first), the locants of its suffix
    groups, of all its prefixes together, of its prefixes in alphanumerical order, then the canonical atom ranks (which
    decide nothing the book decides). None when no numbering gives every junction and group a locant."""
    from ..assembly.naming_utils import alpha_sort_key
    best = None
    for numbering in family:
        try:
            written = tuple(numbering[j] for j in junctions)
            suffix = tuple(sorted(numbering[g.atom] for g in groups if g.system == system and g.name is None))
            prefixes = sorted((alpha_sort_key(g.name), numbering[g.atom])
                              for g in groups if g.system == system and g.name is not None)
        except KeyError:
            continue
        key = (tuple(sorted(written)), written, suffix, tuple(sorted(loc for _, loc in prefixes)),
               tuple(loc for _, loc in prefixes), tuple(ranks[a] for a in sorted(numbering, key=numbering.get)))
        if best is None or key < best[0]:
            best = (key, numbering)
    return best


def _oriented(order: List[int], junction, families, groups: List[_Sub], ranks):
    """The best numbering of the chain read in ``order`` (system indices from ring 1): ``(key, numberings)`` with the
    numbering of each ring and the criteria of the whole name, in the order the book applies them, each a list of
    composite locants ``(ring, superscript)`` compared by ring first; None when a ring cannot be numbered."""
    from ..assembly.naming_utils import alpha_sort_key
    picks = []
    for position, system in enumerate(order):
        ends = [junction[(system, order[position + step])] for step in (-1, 1) if 0 <= position + step < len(order)]
        pick = _ring_pick(families[system], system, ends, groups, ranks)
        if pick is None:
            return None
        picks.append(pick)

    def composite(slot):
        return [(ring + 1, loc) for ring, (key, _) in enumerate(picks) for loc in key[slot]]
    prefixed = sorted((alpha_sort_key(g.name), (order.index(g.system) + 1, picks[order.index(g.system)][1][g.atom]))
                      for g in groups if g.name is not None)
    key = (composite(0), composite(1), composite(2), composite(3), [loc for _, loc in prefixed])
    return key, [numbering for _, numbering in picks]


def _composite_name(mol, info: Dict) -> Optional[str]:
    chain = _chain(info)
    systems = info['ring_systems']
    if chain is None or not _plain(mol, systems):
        return None
    junction, count = chain
    ring_name = _component(mol, systems)
    families = _families(mol, info) if ring_name else None
    groups = _groups(mol, info) if families else None
    if groups is None:
        return None
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    forward = list(range(count))
    candidates = [(r, order) for order in (forward, forward[::-1])
                  for r in [_oriented(order, junction, families, groups, ranks)] if r is not None]
    if not candidates:
        return None
    (key, numberings), order = min(candidates, key=lambda c: c[0][0])
    return _spell(ring_name, count, order, junction, numberings, groups)


def _spell(ring_name: str, count: int, order: List[int], junction, numberings, groups: List[_Sub]) -> Optional[str]:
    """The name of the chain read in ``order`` with the numbering of each ring: ``<prefixes>[<junctions>-<multiplier>
    <component>]-<suffix locants>-<suffix>`` with a suffix group, ``<prefixes>-<junctions>-<multiplier><component>``
    without: no hyphen before an enclosing mark)."""
    from ..assembly.naming_utils import (
        _join_multiplied_suffix,
        alpha_sort_key,
        enclose_if_compound,
        multiplied_component,
    )
    from ..data.multipliers import multiplier_table
    from .ring_assemblies import (
        ASSEMBLY_MULTIPLIERS,
        _component_before_suffix,
        _enclose_component,
        _multiplied_assembly,
    )
    number = {system: (ring + 1, numberings[ring]) for ring, system in enumerate(order)}

    def locant(g: _Sub) -> Tuple[int, int]:
        ring, numbering = number[g.system]
        return ring, numbering[g.atom]

    def cite(locants) -> str:
        return ",".join(f"{ring}^{loc}" for ring, loc in sorted(locants))

    pairs = []
    for ring in range(count - 1):
        here, there = number[order[ring]][1], number[order[ring + 1]][1]
        pairs.append(f"{ring + 1}^{here[junction[(order[ring], order[ring + 1])]]},"
                     f"{ring + 2}^{there[junction[(order[ring + 1], order[ring])]]}")
    junctions, multiplier = ":".join(pairs), ASSEMBLY_MULTIPLIERS[count]
    by_name: Dict[str, List[Tuple[int, int]]] = {}
    for g in groups:
        if g.name is not None:
            by_name.setdefault(g.name, []).append(locant(g))
    parts = []
    for name in sorted(by_name, key=alpha_sort_key):
        where = cite(by_name[name])
        parts.append(f"{where}-{multiplied_component(len(by_name[name]), name, enclose_if_compound(name))}"
                     if len(by_name[name]) > 1 else f"{where}-{enclose_if_compound(name)}")
    prefix_block = "-".join(parts)
    suffixed = [g for g in groups if g.name is None]
    if not suffixed:
        base = _multiplied_assembly(junctions, multiplier, _enclose_component(ring_name), count)
        return f"{prefix_block}-{base}" if prefix_block else base
    multiple = multiplier_table(4, one='').get(len(suffixed))
    if multiple is None:
        return None
    suffix = _join_multiplied_suffix(multiple, suffixed[0].kind)
    core = _multiplied_assembly(junctions, multiplier, _component_before_suffix(ring_name, suffix), count)
    return f"{prefix_block}[{core}]-{cite(locant(g) for g in suffixed)}-{suffix}"
