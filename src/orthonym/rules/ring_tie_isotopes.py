"""Two rings of equal seniority joined by =CH-: criterion (k), the isotopically modified atoms of the parents.

 (the Blue Book, under " SENIORITY CRITERIA APPLICABLE TO RINGS, RING SYSTEMS, OR ACYCLIC
CHAINS",:21014) applies criterion (k) "has one or more isotopically modified atoms " (:21030) before any
criterion of. (:21473) chooses the senior parent structure "between isotopically modified and
isotopically unmodified compounds or between isotopically modified compounds", by these criteria "applied
successively until a decision can be made":

*.1 (:21477) "the greater number of isotopically modified atoms or groups" ('(1,1-2H2)cyclohexane' is senior to
  '(14C1)cyclohexane', 'two 2H atoms are senior to one 14C');
*.2 (:21490) "the greater number of nuclides of higher atomic number" ('(14C1)cyclopentane' is senior to
  '(2H1)cyclopentane');
*.3 (:21496) "the greater number of nuclides of higher mass number" ('(3H1)benzene' is senior to '(2H1)benzene',
  '[1-14C]pentane' to '[1-13C]pentane');
*.4 (:21510) "the lowest locant(s) for isotopically modified atoms or groups" ('[4-2H1]pentanoic acid' is senior to
  '[5-2H1]pentanoic acid');
*.5 (:21516) "the lower locant(s) for nuclides of higher atomic number" ('[4-13C,5-2H1]pentanoic acid' is senior to
  '[5-13C,4-2H1]pentanoic acid');
*.6 (:21523) "the lowest locant(s) for nuclides of higher mass number" ('[4-13C,3-14C]butan-1-ol' is senior to
  '[3-13C,4-14C]butan-1-ol').

The parent of a ring is the ring with its principal groups. A nuclide ranks above another by its atomic number and then
its mass number;.2 and.3 compare the nuclides of the two parents from the highest rank down,.5 and.6 compare the
locants of each rank of nuclide in the same order. The locants are those of the parent as it is numbered in its name
,:3219): the suffix (c), all the prefixes together (f), the prefix cited first (g), and then (i) (:3336), "the
starting point and the direction of numbering... lowest locants to the modified atoms or groups considered together as
a set in increasing numerical order. If a choice still remains, the lower locant is given to the nuclide of higher atomic
number", and to the higher mass number of one element. Where the numbering is not fixed by (c) and (f) alone, the
prefixes are named for (g) (``ring_tie_names.named_prefixes``).

Worked out for the saturated carbocyclic monocycles ``ring_selection`` numbers; a label on an atom of a principal group
(a locant it has no numbering for here), a ring that is not such a monocycle when.4 to.6 are reached, or a labelled
molecule that cannot be read leave the tie ``unknown`` and the caller keeps its honest decline. All of.1 to.6 equal is
a ``tie``: the parents carry the same labels and the criteria of go on.

The isotope decorator names the molecule without its labels, so the labels are read from the labelled molecule it
declares (``locant_omission.skeleton_label_maps``).
"""
from typing import Dict, List, Optional, Set, Tuple

#: ``(atom, atomic number, mass number)`` of one isotopically modified atom
Entry = Tuple[int, int, int]


def _entries(mol, labels: Dict[int, List[Tuple[int, int]]], atoms: Set[int]) -> List[Entry]:
    """The isotopically modified atoms of ``atoms``: those the labelled molecule declares for them, and those
    ``mol`` itself carries (a labelled atom, an isotopic hydrogen on it)."""
    out: List[Entry] = []
    for a in sorted(atoms):
        atom = mol.GetAtomWithIdx(a)
        found = list(labels.get(a, []))
        if atom.GetIsotope():
            found.append((atom.GetAtomicNum(), atom.GetIsotope()))
        found += [(1, nb.GetIsotope()) for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1 and nb.GetIsotope()]
        out += [(a, z, m) for z, m in found]
    return out


def _lower_first(a: List[Tuple[int, ...]], b: List[Tuple[int, ...]]) -> Optional[int]:
    return None if a == b else (0 if a < b else 1)


def _higher_first(a: List[Tuple[int, ...]], b: List[Tuple[int, ...]]) -> Optional[int]:
    return None if a == b else (0 if a > b else 1)


def _locants(mol, ring: Set[int], cyc: List[int], suffix: List[int], group_atoms: Set[int],
             entries: List[Entry], get_named) -> Optional[List[Tuple[int, int, int]]]:
    """``(locant, atomic number, mass number)`` of the modified atoms of a ring under the numbering of its name,
    or None when the numbering cannot be told; ``get_named`` the prefixes of the ring with their names (None
    when they have none), asked only when (c) and (f) leave numberings that put the labels differently."""
    from .ring_tie_names import _cited_groups, prefix_attachments, tied_numberings
    n = len(cyc)

    def read(numbering):
        start, step = numbering
        return sorted((((cyc.index(a) - start) * step) % n + 1, z, m) for a, z, m in entries)

    def choose(numberings):
        """ (i): the lowest locants together, then to the nuclide of higher atomic number, then of higher mass."""
        def key(read_):
            zs = sorted({z for _, z, _ in read_}, reverse=True)
            by_z = [tuple(loc for loc, zz, _ in read_ if zz == z) for z in zs]
            by_zm = [tuple(loc for loc, zz, mm in read_ if (zz, mm) == nz)
                     for nz in sorted({(z, m) for _, z, m in read_}, reverse=True)]
            return (tuple(loc for loc, _, _ in read_), tuple(by_z), tuple(by_zm))
        reads = [read(nb) for nb in numberings]
        return min(reads, key=key)

    firsts = tied_numberings(cyc, suffix, prefix_attachments(mol, ring, group_atoms))
    if len({tuple(read(nb)) for nb in firsts}) == 1:
        return read(firsts[0])
    named = get_named()
    if named is None:
        return None
    return choose(tied_numberings(cyc, suffix, named, _cited_groups(named)))


def _verdict(mol, labels: Dict[int, List[Tuple[int, int]]], rings: List[Set[int]], parents: List[Set[int]],
             cycles: List[Optional[List[int]]], suffix: List[List[int]], group_atoms: Set[int],
             names) -> Tuple[str, Optional[int]]:
    ent = [_entries(mol, labels, parents[k]) for k in (0, 1)]
    if not ent[0] and not ent[1]:
        return "tie", None
    if len(ent[0]) != len(ent[1]):                                                       #.1
        return "decided", 0 if len(ent[0]) > len(ent[1]) else 1
    z = [sorted(((e[1],) for e in ent[k]), reverse=True) for k in (0, 1)]
    zm = [sorted(((e[1], e[2]) for e in ent[k]), reverse=True) for k in (0, 1)]
    for compare, ranked in ((_higher_first, z), (_higher_first, zm)):                    #.2,.3
        senior = compare(ranked[0], ranked[1])
        if senior is not None:
            return "decided", senior
    if any(e[0] not in rings[k] for k in (0, 1) for e in ent[k]) or None in cycles:
        return "unknown", None
    loc = []
    for k in (0, 1):
        got = _locants(mol, rings[k], cycles[k], suffix[k], group_atoms, ent[k], lambda k=k: names(k))
        if got is None:
            return "unknown", None
        loc.append(got)
    senior = _lower_first([(x[0],) for x in loc[0]], [(x[0],) for x in loc[1]])         #.4
    if senior is not None:
        return "decided", senior
    for z_ in sorted({x[1] for x in loc[0]}, reverse=True):                              #.5
        senior = _lower_first([(x[0],) for x in loc[0] if x[1] == z_], [(x[0],) for x in loc[1] if x[1] == z_])
        if senior is not None:
            return "decided", senior
    for nz in sorted({(x[1], x[2]) for x in loc[0]}, reverse=True):                      #.6
        senior = _lower_first([(x[0],) for x in loc[0] if (x[1], x[2]) == nz],
                              [(x[0],) for x in loc[1] if (x[1], x[2]) == nz])
        if senior is not None:
            return "decided", senior
    return "tie", None


def senior_ring_by_isotopes(mol, rings: List[Set[int]], parents: List[Set[int]], cycles: List[Optional[List[int]]],
                            suffix: List[List[int]], group_atoms: Set[int]) -> Tuple[str, Optional[int]]:
    """``("decided", index of the senior ring)``, ``("tie", None)`` (the parents carry the same labels, or none) or
    ``("unknown", None)`` (a criterion this does not work out would decide).

    ``rings`` are the atoms of the two rings, ``parents`` the same with their principal groups, ``cycles`` the rings in
    the order round the ring (None for a ring that is not a saturated carbocyclic monocycle), ``suffix`` the ring atoms
    each cites its principal groups at, ``group_atoms`` the atoms of the principal groups of both."""
    from ..assembly.locant_omission import skeleton_label_maps
    maps = skeleton_label_maps(mol)
    if maps is None:
        return "unknown", None
    cache: Dict[str, Optional[list]] = {}

    def names(k):
        if "named" not in cache:
            from .ring_tie_names import named_prefixes
            cache["named"] = named_prefixes(mol, rings, group_atoms)
        return None if cache["named"] is None else cache["named"][k]

    try:
        verdicts = {_verdict(mol, labels, rings, parents, cycles, suffix, group_atoms, names) for labels in maps}
    except Exception:  # noqa: BLE001 -- a comparison that cannot be made decides nothing
        return "unknown", None
    return verdicts.pop() if len(verdicts) == 1 else ("unknown", None)
