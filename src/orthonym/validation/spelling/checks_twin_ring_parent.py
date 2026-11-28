"""Spelling check: of two identical benzene rings, the parent is the one with more prefixes, then the one with
the lower locant set (registered with:func:`orthonym.validation.pin_spelling.register`).

 "CRITERIA RELATED TO NUMBER AND LOCATION OF SUBSTITUENT GROUPS" (the Blue Book): (:21604) the
preferred name is "based on the senior parent structure that has the maximum number of substituents cited as prefixes"
('1-methyl-4-(phenoxymethyl)benzene (PIN) [not [(4-methylphenyl)methoxy]benzene]',:21618); (:21698) "... that
has the lower locant or set of locants for substituents cited as prefixes". So '[(4-chlorophenoxy)methyl]benzene' (PIN
'1-(benzyloxy)-4-chlorobenzene') or '1-(2,4-dinitrophenoxy)-4-methylbenzene' (PIN '1-(4-methylphenoxy)-2,4-
dinitrobenzene') is correct but no PIN.

Read only for a narrow shape: the molecule has exactly two rings, both isolated benzene rings not bonded to each other,
no group that a suffix would express, no charge, isotope or second fragment; the name is one word on a bare 'benzene'
parent. Equal prefix counts with equal locant sets (decided by later criteria) are not judged.
"""
from __future__ import annotations

from itertools import product

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register
from .models import P10, read_units

_BENZENE = Chem.MolFromSmarts('c1ccccc1')
_NITRO = Chem.MolFromSmarts('[N+](=O)[O-]')
#: an atom a suffix would express (the name of such a molecule is not on a bare 'benzene')
_SUFFIXABLE = Chem.MolFromSmarts('[$([#6]=[O,S,Se,N]),$([OX2H1]),$([SX2H1]),$([NX3;H1,H2]),$([#6]#[NX1])]')


def _ring_prefixes(mol, ring):
    """The ring atoms (in ring order) and, for each, the number of exocyclic heavy neighbours."""
    order = list(ring)
    return order, [sum(1 for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() not in ring
                       and n.GetAtomicNum() > 1) for i in order]


def _lowest_locants(mol, ring) -> tuple:
    """The lowest locant set of the ring's substituents over the twelve numberings of a benzene ring."""
    info = mol.GetRingInfo()
    cycle = next(list(r) for r in info.AtomRings() if set(r) == set(ring))
    counts = [sum(1 for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() not in ring and n.GetAtomicNum() > 1)
              for i in cycle]
    best = None
    for start, step in product(range(6), (1, -1)):
        locs = sorted(pos + 1 for pos in range(6) for _ in range(counts[(start + step * pos) % 6]))
        if best is None or locs < best:
            best = locs
    return tuple(best)


def _name_parent(name):
    """(prefix count, locant set) of the top-level unit when it is a bare 'benzene' parent, else None."""
    parsed = read_units(name)
    if parsed is None:
        return None
    tops = [p for p in parsed[2] if p.unit.encl is None]
    if len(tops) != 1:
        return None
    top = tops[0]
    if top.parent != 'benzene' or top.endings or top.hydro or top.repl or top.ih or top.multiplied:
        return None
    count, locs = 0, []
    for x in top.prefixes:
        if any(not loc.isdigit() for loc in x.locs):
            return None
        count += max(len(x.locs), x.mult)
        locs += [int(loc) for loc in x.locs]
    if count == 1 and not locs:
        locs = [1]
    if len(locs) != count:
        return None
    return count, tuple(sorted(locs))


@register('P-45.2')
def twin_ring_parent_check(mol, name):
    """Of two identical benzene rings, the name's parent has fewer prefixes, or as many with a higher locant set."""
    if not name or ' ' in name.strip() or "'" in name or P10.search(name):
        return None
    nitro = {i for m in mol.GetSubstructMatches(_NITRO) for i in m}   # a nitro group drawn charge-separated is no ion
    if len(Chem.GetMolFrags(mol)) != 1 or any((a.GetFormalCharge() and a.GetIdx() not in nitro)
                                              or a.GetNumRadicalElectrons() or a.GetIsotope() for a in mol.GetAtoms()):
        return None
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    if len(rings) != 2 or rings[0] & rings[1] or mol.HasSubstructMatch(_SUFFIXABLE):
        return None
    if not all(any(set(m) == r for m in mol.GetSubstructMatches(_BENZENE)) for r in rings):
        return None
    if any(mol.GetBondBetweenAtoms(i, j) for i in rings[0] for j in rings[1]):
        return None                     # a ring assembly ('1,1'-biphenyl')
    read = _name_parent(name)
    if read is None:
        return None
    count, locs = read
    n = [sum(_ring_prefixes(mol, r)[1]) for r in rings]
    if n[0] != n[1]:
        if count == min(n) and count != max(n):
            return SpellingFailure('P-45.2.1', f"the parent ring carries {count} prefixes, the other benzene ring "
                                               f"{max(n)}: the ring with more prefixes is the parent")
        return None
    if count != n[0]:
        return None
    sets = [_lowest_locants(mol, r) for r in rings]
    if sets[0] != sets[1] and locs == max(sets) and locs != min(sets):
        return SpellingFailure('P-45.2.2', f"the parent ring has the locant set {','.join(map(str, locs))}, the other "
                                           f"benzene ring {','.join(map(str, min(sets)))}: the lower set is the parent")
    return None


# ------------------------------------------------------------------ twin rings that carry the suffix group
#: the retained benzene parents and the group each expresses as its suffix (on the ring carbon numbered 1)
_SUFFIX_PARENT = {
    'phenol': Chem.MolFromSmarts('c[OX2H1]'),
    'aniline': Chem.MolFromSmarts('c[NX3H2]'),
    'benzoic': Chem.MolFromSmarts('c[CX3](=O)[OX2H1]'),
    'benzonitrile': Chem.MolFromSmarts('c[CX2]#[NX1]'),
    'benzaldehyde': Chem.MolFromSmarts('c[CX3H1]=O'),
}


def _suffix_ring_locants(mol, ring, suffix_atom) -> tuple:
    """The lowest locant set of the ring's other substituents with the suffix carbon numbered 1 (two directions)."""
    cycle = next(list(r) for r in mol.GetRingInfo().AtomRings() if set(r) == set(ring))
    start = cycle.index(suffix_atom)
    counts = [sum(1 for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() not in ring and n.GetAtomicNum() > 1)
              for i in cycle]
    counts[start] -= 1
    best = None
    for step in (1, -1):
        locs = sorted(pos + 1 for pos in range(6) for _ in range(counts[(start + step * pos) % 6]))
        if best is None or locs < best:
            best = locs
    return tuple(best)


@register('P-45.2')
def twin_suffix_ring_parent_check(mol, name):
    """Of two benzene rings that each carry the suffix group of a retained benzene parent ('phenol', 'aniline',
    'benzoic acid', 'benzonitrile', 'benzaldehyde'), the name's parent has fewer prefixes, or as many with a higher
    locant set: '4-chloro-2-[(3-cyanophenyl)methyl]benzonitrile (PIN) [not 3-[(5-chloro-2-cyanophenyl)methyl]
    benzonitrile]' (the Blue Book), (:21698)."""
    if not name or ' ' in name.strip().replace(' acid', '') or "'" in name or P10.search(name):
        return None
    parsed = read_units(name)
    if parsed is None:
        return None
    tops = [p for p in parsed[2] if p.unit.encl is None and p.parent in _SUFFIX_PARENT]
    if len(tops) != 1:
        return None
    top = tops[0]
    if top.endings or top.hydro or top.repl or top.ih or top.multiplied:
        return None
    count, locs = 0, []
    for x in top.prefixes:
        if any(not loc.isdigit() for loc in x.locs) or not x.locs:
            return None
        count += max(len(x.locs), x.mult)
        locs += [int(loc) for loc in x.locs]
    if not count or len(locs) != count:
        return None
    nitro = {i for m in mol.GetSubstructMatches(_NITRO) for i in m}
    if len(Chem.GetMolFrags(mol)) != 1 or any((a.GetFormalCharge() and a.GetIdx() not in nitro)
                                              or a.GetNumRadicalElectrons() or a.GetIsotope() for a in mol.GetAtoms()):
        return None
    ring_count = {}
    for r in mol.GetRingInfo().AtomRings():
        for i in r:
            ring_count[i] = ring_count.get(i, 0) + 1
    # isolated benzene rings only: the benzo ring of a fused system is no 'phenol' parent ('...-1H-inden-3-yl]phenol')
    benzenes = [set(m) for m in mol.GetSubstructMatches(_BENZENE) if all(ring_count.get(i) == 1 for i in m)]
    on_ring = {}
    for m in mol.GetSubstructMatches(_SUFFIX_PARENT[top.parent]):
        ring = next((frozenset(r) for r in benzenes if m[0] in r), None)
        if ring is None or ring in on_ring:
            return None                 # a group off a benzene ring, or two on one ring
        on_ring[ring] = m[0]
    if len(on_ring) != 2:
        return None
    (ra, sa), (rb, sb) = on_ring.items()
    if ra & rb or any(mol.GetBondBetweenAtoms(i, j) for i in ra for j in rb):
        return None
    n = [sum(1 for i in r for nb in mol.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() not in r and nb.GetAtomicNum() > 1)
         - 1 for r in (ra, rb)]
    if n[0] != n[1]:
        if count == min(n) and count != max(n):
            return SpellingFailure('P-45.2.1', f"the parent ring carries {count} prefixes, the other ring with the "
                                               f"same suffix {max(n)}: the ring with more prefixes is the parent")
        return None
    if count != n[0]:
        return None
    sets = [_suffix_ring_locants(mol, ra, sa), _suffix_ring_locants(mol, rb, sb)]
    mine = tuple(sorted(locs))
    if sets[0] != sets[1] and mine == max(sets) and mine != min(sets):
        return SpellingFailure('P-45.2.2', f"the parent ring has the locant set {','.join(map(str, mine))}, the other "
                                           f"ring with the same suffix {','.join(map(str, min(sets)))}")
    return None


# ------------------------------------------------------------------ aniline twins with N-substituents
_AMINE_N = Chem.MolFromSmarts('[NX3;!R;+0;!$(N-[#6,#16,#15]=[O,S,N]);!$(N-[#7,#8])]')


def _aniline_key(mol, ring, n_idx, ring_atom):
    """(prefix count, lowest locant set with 'N' as 0) of the ring as an aniline parent: its ring prefixes plus the
    substituents on the nitrogen,:3195, an italic letter locant is lower than a numeral)."""
    n_subs = sum(1 for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors() if nb.GetIdx() != ring_atom and nb.GetAtomicNum() > 1)
    cycle = next(list(r) for r in mol.GetRingInfo().AtomRings() if set(r) == set(ring))
    start = cycle.index(ring_atom)
    counts = [sum(1 for nb in mol.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() not in ring and nb.GetAtomicNum() > 1)
              for i in cycle]
    counts[start] -= 1
    best = None
    for step in (1, -1):
        locs = sorted([0] * n_subs + [pos + 1 for pos in range(6) for _ in range(counts[(start + step * pos) % 6])])
        if best is None or locs < best:
            best = locs
    return n_subs + sum(counts), tuple(best)


@register('P-45.2')
def aniline_twin_parent_check(mol, name):
    """Of two benzene rings that each carry an amine nitrogen (their own, or one N they share), the 'aniline' name's
    parent has fewer prefixes (N-substituents counted), or as many with a higher locant set: '4-methoxy-N-
    phenylaniline (PIN)' (the Blue Book); "the locant set 'N,2' in the PIN is lower than '2,5'" (:21704)."""
    if not name or ' ' in name.strip() or "'" in name or P10.search(name):
        return None
    parsed = read_units(name)
    if parsed is None:
        return None
    tops = [p for p in parsed[2] if p.unit.encl is None and p.parent == 'aniline']
    if len(tops) != 1:
        return None
    top = tops[0]
    if top.endings or top.hydro or top.repl or top.ih or top.multiplied:
        return None
    count, locs = 0, []
    for x in top.prefixes:
        if not x.locs or any(not (loc.isdigit() or loc == 'N') for loc in x.locs):
            return None
        count += max(len(x.locs), x.mult)
        locs += [0 if loc == 'N' else int(loc) for loc in x.locs]
    if not count or len(locs) != count:
        return None
    nitro = {i for m in mol.GetSubstructMatches(_NITRO) for i in m}
    if len(Chem.GetMolFrags(mol)) != 1 or any((a.GetFormalCharge() and a.GetIdx() not in nitro)
                                              or a.GetNumRadicalElectrons() or a.GetIsotope() for a in mol.GetAtoms()):
        return None
    ring_count = {}
    for r in mol.GetRingInfo().AtomRings():
        for i in r:
            ring_count[i] = ring_count.get(i, 0) + 1
    benzenes = [frozenset(m) for m in mol.GetSubstructMatches(_BENZENE) if all(ring_count.get(i) == 1 for i in m)]
    carriers = []                       # (ring, amine N, ring atom on the N)
    for (n_idx,) in mol.GetSubstructMatches(_AMINE_N):
        for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors():
            ring = next((r for r in benzenes if nb.GetIdx() in r), None)
            if ring is not None:
                carriers.append((ring, n_idx, nb.GetIdx()))
    if len(carriers) != 2 or carriers[0][0] == carriers[1][0]:
        return None                     # exactly two benzene rings, each on one amine N
    (ra, na, aa), (rb, nb_, ab) = carriers
    if ra & rb or any(mol.GetBondBetweenAtoms(i, j) for i in ra for j in rb):
        return None
    keys = [_aniline_key(mol, ra, na, aa), _aniline_key(mol, rb, nb_, ab)]
    n = [k[0] for k in keys]
    if n[0] != n[1]:
        if count == min(n) and count != max(n):
            return SpellingFailure('P-45.2.1', f"the aniline parent carries {count} prefixes (N-substituents counted), "
                                               f"the other amine-bearing ring {max(n)}")
        return None
    if count != n[0]:
        return None
    sets = [k[1] for k in keys]
    mine = tuple(sorted(locs))
    if sets[0] != sets[1] and mine == max(sets) and mine != min(sets):
        return SpellingFailure('P-45.2.2', "the aniline parent has the higher locant set ('N' is lower than any "
                                           "numeral); the other amine-bearing ring is the parent")
    return None
