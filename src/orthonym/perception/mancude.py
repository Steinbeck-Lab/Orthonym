"""Mancude and ortho-fusion predicates for ring systems (large-polycycle plan, Task 1).

 (the Blue Book): 'ortho-fused' or 'ortho- and peri-fused'
polycyclic ring systems "with the maximum number of noncumulative double bonds
(mancude)". A ring system is mancude iff its ring double bonds (on a Kekule
structure) form a MAXIMUM matching of the ring atoms that can bear a ring double
bond. (:11865): "Two rings that have only two atoms and one bond in
common are said to be ortho-fused".
"""
from typing import Iterable, Mapping, Optional, Set

from rdkit import Chem


def max_matching_size(nodes: Iterable[int], adj: Mapping[int, Iterable[int]]) -> int:
    """Maximum-cardinality matching size (Edmonds' blossom algorithm, O(V^3))."""
    order = sorted(set(nodes))
    idx = {v: i for i, v in enumerate(order)}
    n = len(order)
    g = [sorted(idx[w] for w in adj.get(v, ()) if w in idx and w != v) for v in order]
    match = [-1] * n

    def find_path(root):
        used = [False] * n
        p = [-1] * n
        base = list(range(n))
        used[root] = True
        q = [root]
        qi = 0

        def lca(a, b):
            seen = [False] * n
            while True:
                a = base[a]
                seen[a] = True
                if match[a] == -1:
                    break
                a = p[match[a]]
            while True:
                b = base[b]
                if seen[b]:
                    return b
                b = p[match[b]]

        def mark(v, b, child, blossom):
            while base[v] != b:
                blossom[base[v]] = blossom[base[match[v]]] = True
                p[v] = child
                child = match[v]
                v = p[match[v]]

        while qi < len(q):
            v = q[qi]
            qi += 1
            for to in g[v]:
                if base[v] == base[to] or match[v] == to:
                    continue
                if to == root or (match[to] != -1 and p[match[to]] != -1):
                    cur = lca(v, to)
                    blossom = [False] * n
                    mark(v, cur, to, blossom)
                    mark(to, cur, v, blossom)
                    for i in range(n):
                        if blossom[base[i]]:
                            base[i] = cur
                            if not used[i]:
                                used[i] = True
                                q.append(i)
                elif p[to] == -1:
                    p[to] = v
                    if match[to] == -1:
                        return to, p
                    used[match[to]] = True
                    q.append(match[to])
        return -1, p

    for v in range(n):
        if match[v] == -1:
            to, p = find_path(v)
            while to != -1:
                pv = p[to]
                ppv = match[pv]
                match[to] = pv
                match[pv] = to
                to = ppv
    return sum(1 for v in range(n) if match[v] != -1) // 2


def _ring_bonds(mol, cage):
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in cage and j in cage and b.IsInRing():
            yield i, j, b


def is_mancude_ring_system(mol, ring_atoms: Set[int]) -> Optional[bool]:
    """True/False, or None when undecidable (kekulization fails, ring triple bond)."""
    cage = set(ring_atoms)
    kek = Chem.Mol(mol)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        return None
    pt = Chem.GetPeriodicTable()
    adj = {i: set() for i in cage}
    ring_deg = {i: 0 for i in cage}
    in_double = set()
    n_double = 0
    for i, j, b in _ring_bonds(kek, cage):
        adj[i].add(j); adj[j].add(i)
        ring_deg[i] += 1; ring_deg[j] += 1
        if b.GetBondType() == Chem.BondType.DOUBLE:
            n_double += 1
            in_double.update((i, j))
        elif b.GetBondType() == Chem.BondType.TRIPLE:
            return None

    def eligible(i):
        if i in in_double:
            return True
        a = kek.GetAtomWithIdx(i)
        if a.GetFormalCharge() != 0:
            return False
        dv = pt.GetDefaultValence(a.GetAtomicNum())
        return dv >= 0 and dv - ring_deg[i] >= 1

    nodes = {i for i in cage if eligible(i)}
    sub = {i: {j for j in adj[i] if j in nodes} for i in nodes}
    return max_matching_size(nodes, sub) == n_double


def _induced_rings(mol, atoms, min_size):
    """SSSR of the subgraph INDUCED by `atoms` (the fused core's rings are not in the
    whole molecule's SSSR when bridges are present: trimethanoanthracene's SSSR is six
    5-rings, its anthracene 6-rings are not among them)."""
    idx = sorted(atoms)
    pos = {a: i for i, a in enumerate(idx)}
    rw = Chem.RWMol()
    for _ in idx:
        at = Chem.Atom(6)
        at.SetNoImplicit(True)
        rw.AddAtom(at)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in pos and j in pos:
            rw.AddBond(pos[i], pos[j], Chem.BondType.SINGLE)
    sub = rw.GetMol()
    sub.UpdatePropertyCache(strict=False)
    out = []
    for r in Chem.GetSymmSSSR(sub):
        ra = [idx[k] for k in r]
        if len(ra) >= min_size:
            out.append((frozenset(ra), {frozenset((ra[k], ra[(k + 1) % len(ra)])) for k in range(len(ra))}))
    return out


def has_ortho_fused_pair(mol, atoms: Set[int], min_size: int = 5) -> bool:
    rings = _induced_rings(mol, set(atoms), min_size)
    for x in range(len(rings)):
        for y in range(x + 1, len(rings)):
            if len(rings[x][0] & rings[y][0]) == 2 and len(rings[x][1] & rings[y][1]) == 1:
                return True
    return False


def count_rings_at_least(mol, atoms: Set[int], size: int = 5) -> int:
    return len(_induced_rings(mol, set(atoms), size))


def mancude_fused_core(mol, ring_atoms: Set[int]) -> Optional[Set[int]]:
    """The candidate fused parent of a bridged fused name:14023):
    the ring atoms minus the BRIDGE atoms. A bridge atom has only single bonds on a
    Kekule structure AND spare standard valence for a ring double bond (so a divalent
    ring O/S, which can never carry one, always stays in the fused parent). A
    connected group of such atoms is put back when removing it would leave some
    remaining atom in no ring: that group is a hydro / indicated-hydrogen position
    of the fused ring (the N-H of 1H-indole, a CH2 of 1H-indene), not a bridge."""
    ring_atoms = set(ring_atoms)
    kek = Chem.Mol(mol)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        return None
    pt = Chem.GetPeriodicTable()

    def ring_deg(i):
        return sum(1 for n in kek.GetAtomWithIdx(i).GetNeighbors()
                   if n.GetIdx() in ring_atoms and kek.GetBondBetweenAtoms(i, n.GetIdx()).IsInRing())

    def bridge_candidate(i):
        a = kek.GetAtomWithIdx(i)
        if a.GetFormalCharge() != 0:
            return False
        if not all(b.GetBondType() == Chem.BondType.SINGLE for b in a.GetBonds()):
            return False
        dv = pt.GetDefaultValence(a.GetAtomicNum())
        return dv >= 0 and dv - ring_deg(i) >= 1

    rem = {i for i in ring_atoms if bridge_candidate(i)}
    groups = []
    while rem:
        stack = [min(rem)]
        group = set(stack)
        rem -= group
        while stack:
            x = stack.pop()
            for n in kek.GetAtomWithIdx(x).GetNeighbors():
                j = n.GetIdx()
                if j in rem:
                    rem.discard(j)
                    group.add(j)
                    stack.append(j)
        groups.append(group)
    removed = set().union(*groups) if groups else set()
    while True:
        core = ring_atoms - removed
        in_ring = set().union(*[r for r, _ in _induced_rings(mol, core, 3)]) if core else set()
        acyclic = core - in_ring
        if not acyclic:
            return core
        back = [g for g in groups if g <= removed
                and any(kek.GetBondBetweenAtoms(a, b) is not None for a in g for b in acyclic)]
        if not back:
            return set(ring_atoms)          # cannot explain the acyclic atoms: excise nothing
        for g in back:
            removed -= g


def fusion_pin_class(mol, ring_atoms: Set[int]) -> Optional[bool]:
    """True when (:23710) gives this cage a fusion or bridged-fused PIN:
    its fused core is mancude:11903) and contains an ortho-fused pair
     :11865) of rings with >= 5 members. None = undecidable."""
    core = mancude_fused_core(mol, ring_atoms)
    if core is None:
        return None
    if not has_ortho_fused_pair(mol, core, 5):
        return False
    m = is_mancude_ring_system(mol, core)
    return None if m is None else m
