"""How many indicated hydrogens each ring system of a structure needs, and how many a name cites.

The rule (the Blue Book):

* (:8318-:8320): "After the maximum number of noncumulative double bonds has been
  assigned..., any ring atom with a bonding number of three or higher connected to adjacent ring
  atoms by single bonds only, and carrying one or more hydrogen atoms, is designated by indicated
  hydrogen"; for fused systems:14551.
* (:3721): "In general nomenclature, indicated hydrogen may be omitted... However, in a
  preferred IUPAC name a locant and the symbol 'H' must be cited"; (:14607),
   (:24639): in preferred IUPAC names all indicated hydrogen is cited.
* (:10152) spiro, / (:14245,:14640) bridged fused,
  (:15593) ring assemblies: the whole skeleton is made mancude first, one indicated-hydrogen group
  in front of the name of that system; saturated spiro components keep their saturation; a
  bridge keeps the saturation of its own name.
*: a heteroatom cited with lambda-n has bonding number n.
*, (:3725,:24687): added indicated hydrogen ``n(mH)`` after a suffix locant
  is not the parent's indicated hydrogen and does not replace it.

Structure side: the RDKit graph. For the atoms of a mancude ring system that can carry a ring
double bond (bonding number >= 3 and at least one bond left after the ring bonds), k = capable
atoms - 2 x (maximum matching of ring bonds among them). Name side: the front ``nH-`` groups.
Validated on the 3,243 Blue Book PIN rows (agreement 99.3 %, every disagreement explained:
listing rows that print a ring component without its hydrogen, charged rings, two-name rows).
"""
from __future__ import annotations

import re
from collections import defaultdict

from rdkit import Chem

STD_BOND = {
    'B': 3, 'Al': 3, 'Ga': 3, 'In': 3, 'Tl': 3,
    'C': 4, 'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 4,
    'N': 3, 'P': 3, 'As': 3, 'Sb': 3, 'Bi': 3,
    'O': 2, 'S': 2, 'Se': 2, 'Te': 2, 'Po': 2,
    'F': 1, 'Cl': 1, 'Br': 1, 'I': 1, 'Hg': 2, 'Be': 2, 'Zn': 2, 'Mg': 2,
}


# ---------------------------------------------------------------- graph helpers
def max_matching(nodes, adj) -> int:
    """Number of edges of a maximum-cardinality matching (Edmonds' blossom algorithm)."""
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    g = [[idx[w] for w in adj[v] if w in idx] for v in nodes]
    match = [-1] * n
    base = list(range(n))
    p = [-1] * n

    def lca(a, b):
        used = [False] * n
        while True:
            a = base[a]
            used[a] = True
            if match[a] == -1:
                break
            a = p[match[a]]
        while True:
            b = base[b]
            if used[b]:
                return b
            b = p[match[b]]

    def mark_path(v, b, child, blossom):
        while base[v] != b:
            blossom[base[v]] = blossom[base[match[v]]] = True
            p[v] = child
            child = match[v]
            v = p[match[v]]

    def find_path(root):
        used = [False] * n
        for i in range(n):
            p[i] = -1
            base[i] = i
        used[root] = True
        q = [root]
        qh = 0
        while qh < len(q):
            v = q[qh]
            qh += 1
            for to in g[v]:
                if base[v] == base[to] or match[v] == to:
                    continue
                if to == root or (match[to] != -1 and p[match[to]] != -1):
                    curbase = lca(v, to)
                    blossom = [False] * n
                    mark_path(v, curbase, to, blossom)
                    mark_path(to, curbase, v, blossom)
                    for i in range(n):
                        if blossom[base[i]]:
                            base[i] = curbase
                            if not used[i]:
                                used[i] = True
                                q.append(i)
                elif p[to] == -1:
                    p[to] = v
                    if match[to] == -1:
                        return to
                    used[match[to]] = True
                    q.append(match[to])
        return -1

    for v in range(n):
        if match[v] == -1:
            for to in g[v]:
                if match[to] == -1:
                    match[to] = v
                    match[v] = to
                    break
    for v in range(n):
        if match[v] == -1:
            u = find_path(v)
            while u != -1:
                pv = p[u]
                ppv = match[pv]
                match[u] = pv
                match[pv] = u
                u = ppv
    return sum(1 for v in range(n) if match[v] != -1) // 2


def ring_systems(mol):
    """Connected components of ring atoms joined by ring bonds."""
    parent = {}

    def f(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in mol.GetAtoms():
        if a.IsInRing():
            parent[a.GetIdx()] = a.GetIdx()
    for b in mol.GetBonds():
        if b.IsInRing():
            x, y = f(b.GetBeginAtomIdx()), f(b.GetEndAtomIdx())
            if x != y:
                parent[x] = y
    comps = defaultdict(set)
    for a in parent:
        comps[f(a)].add(a)
    return [sorted(c) for c in comps.values()]


def ring_adj(mol, atoms):
    s = set(atoms)
    adj = {a: set() for a in atoms}
    for b in mol.GetBonds():
        if not b.IsInRing():
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in s and j in s:
            adj[i].add(j)
            adj[j].add(i)
    return adj


def blocks(atoms, adj):
    """Biconnected components of the ring graph, as atom sets (iterative, no recursion limit)."""
    disc, low = {}, {}
    out = []
    counter = 0
    for root in atoms:
        if root in disc:
            continue
        disc[root] = low[root] = counter
        counter += 1
        stack = [(root, None, iter(sorted(adj[root])))]
        edges = []
        while stack:
            u, par, it = stack[-1]
            advanced = False
            for w in it:
                if w not in disc:
                    disc[w] = low[w] = counter
                    counter += 1
                    edges.append((u, w))
                    stack.append((w, u, iter(sorted(adj[w]))))
                    advanced = True
                    break
                if w != par and disc[w] < disc[u]:
                    edges.append((u, w))
                    low[u] = min(low[u], disc[w])
            if advanced:
                continue
            stack.pop()
            if stack:
                pu = stack[-1][0]
                low[pu] = min(low[pu], low[u])
                if low[u] >= disc[pu]:
                    comp = set()
                    while True:
                        e = edges.pop()
                        comp.update(e)
                        if e == (pu, u):
                            break
                    out.append(comp)
        if edges:
            comp = set()
            while edges:
                comp.update(edges.pop())
            out.append(comp)
    return out


def kekule_ring_double(mol, atoms) -> int:
    """Number of endocyclic double bonds (Kekule form) inside the atom set.

    A structure without a Kekule form raises (RDKit's KekulizeException): counting its aromatic
    bonds as single would read a mancude ring as saturated.:func:`analyse` reads a sanitized
    structure, which always has one; elsewhere the error reaches
    :func:`orthonym.validation.pin_spelling.check_pin_spelling`, which logs it and lets the check
    abstain (``strict=True`` re-raises)."""
    m = Chem.Mol(mol)
    Chem.Kekulize(m, clearAromaticFlags=True)
    s = set(atoms)
    n = 0
    for b in m.GetBonds():
        if b.IsInRing() and b.GetBondType() == Chem.BondType.DOUBLE:
            if b.GetBeginAtomIdx() in s and b.GetEndAtomIdx() in s:
                n += 1
    return n


# ---------------------------------------------------------------- name side
_LOC = r"\d+(?:'|′)*[a-z]?(?:'|′)*"
_FRONT_IH = re.compile(r"(?:(?<=^)|(?<=[\s\-\(\[\{,—–/]))((?:" + _LOC + r"H,)*" + _LOC
                       + r"H)-(?=[A-Za-z\[\(λ\{0-9])")
_ADDED_IH = re.compile(r"(?<=[\d'a-z′])\(((?:" + _LOC + r"H,)*" + _LOC + r"H)\)")
_VB = re.compile(r"((?:undeca|dodeca|trideca|tetradeca|bi|tri|tetra|penta|hexa|hepta|octa|nona|deca)cyclo|"
                 r"(?:di|tri|tetra|penta|hexa)?spiro)\[([0-9.,^\s]+(?:<sup>[0-9,\s]*</sup>[0-9.,\s]*)*)\]")
_NR = {'bi': 2, 'tri': 3, 'tetra': 4, 'penta': 5, 'hexa': 6, 'hepta': 7, 'octa': 8, 'nona': 9,
       'deca': 10, 'undeca': 11, 'dodeca': 12, 'trideca': 13, 'tetradeca': 14}
#: retained names of stereoparents and nucleosides,: their skeleton carries no
#: indicated hydrogen of its own
_STEREOPARENT_STEMS = ('gonan', 'gona-', 'estra', 'estr-', 'androst', 'pregn', 'chola', 'chol-', 'cholan',
                       'cholest', 'ergost', 'stigmast', 'campest', 'poriferast', 'gorgost', 'cardanolid',
                       'bufanolid', 'spirostan', 'furostan', 'morphinan', 'ergolin', 'yohimban',
                       'cobamide', 'cobinamide', 'cobyrinic', 'corrin', 'aspidospermidin', 'atisan',
                       'kauran', 'gibban', 'taxan', 'tubocuraran', 'strychnidin', 'vincaleukoblastin')
_NUCLEOSIDE_STEMS = ('adenosine', 'guanosine', 'inosine', 'xanthosine', 'cytidine', 'uridine',
                     'thymidine', 'adenylic', 'guanylic', 'inosinic', 'cytidylic', 'uridylic',
                     'thymidylic', 'xanthylic')
_RETAINED_CATIONS = ('pyrylium', 'xanthylium', 'chromenylium', 'flavylium', 'thiopyrylium')
_SAT_RETAINED = ('adamantan', 'cuban', 'morphinan', 'ergolin', 'estran', 'androstan',
                 'pregnan', 'cholestan', 'gonan', 'yohimban', 'quinuclidin', 'cholan', 'ergostan',
                 'stigmastan', 'campestan', 'cardanolid', 'bufanolid', 'spirostan', 'furostan',
                 'aspidospermidin', 'atisan', 'kauran', 'labdan', 'abietan', 'pimaran', 'taxan')
_ASSEMBLY = re.compile(r"\d+(?:'|′)*[a-z]?(?:'|′)*,\d+(?:'|′)+[a-z]?(?:'|′)*-(?:bi|ter|quater)|"
                       r"\d+[a-z]?(?:'|′)+,\d+[a-z]?(?:'|′)*-(?:bi|ter|quater)")


def front_ih_groups(name):
    return [(m.start(1), m.group(1), len(m.group(1).split(','))) for m in _FRONT_IH.finditer(name)]


def added_ih_groups(name):
    return [(m.start(1), m.group(1), len(m.group(1).split(','))) for m in _ADDED_IH.finditer(name)]


def _vb_descriptors(name):
    out = []
    for m in _VB.finditer(name):
        kind = m.group(1)
        body = re.sub(r'<sup>[\d,\s]*</sup>', '', m.group(2))
        body = re.sub(r'\^\{?[\d,]+\}?', '', body)
        nums = []
        for part in (x for x in body.split('.') if x.strip()):
            mm = re.match(r'(\d+)', part.strip())
            if mm:
                nums.append(int(mm.group(1)))
        if 'spiro' in kind:
            out.append(('spiro', sum(nums) + 1 + kind.count('di') + (2 if kind.startswith('tri') else 0), None))
        else:
            out.append((kind, nums, _NR.get(kind[:-len('cyclo')])))
    return out


def _lambda_tokens(name):
    return [(m.group(1), int(m.group(2)))
            for m in re.finditer(r"(\d+[a-z]?(?:'|′)*)?λ(?:<sup>)?\s*(\d)", name)]


def _bonding(mol, a, lam_map, charge_mode):
    at = mol.GetAtomWithIdx(a)
    if a in lam_map:
        return lam_map[a]
    bn = STD_BOND.get(at.GetSymbol(), 4)
    q = at.GetFormalCharge()
    if charge_mode and q:
        sym = at.GetSymbol()
        if sym in ('N', 'P', 'As', 'Sb', 'O', 'S', 'Se', 'Te'):
            bn = bn + q
        elif sym == 'B':
            bn = bn - q
        elif sym == 'C':
            bn = 3
    return bn


def _n_ih(mol, sub_atoms, deg, lam_map, charge_mode=False) -> int:
    cap = [a for a in sub_atoms
           if _bonding(mol, a, lam_map, charge_mode) >= 3
           and _bonding(mol, a, lam_map, charge_mode) - deg[a] >= 1]
    cs = set(cap)
    adj = {a: set() for a in cap}
    for b in mol.GetBonds():
        if b.IsInRing():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in cs and j in cs:
                adj[i].add(j)
                adj[j].add(i)
    return len(cap) - 2 * max_matching(cap, adj)


def _sub_rings(mol, bl):
    m = Chem.RWMol()
    mp = {}
    for a in sorted(bl):
        mp[a] = m.AddAtom(Chem.Atom(6))
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if b.IsInRing() and i in bl and j in bl:
            m.AddBond(mp[i], mp[j], Chem.BondType.SINGLE)
    inv = {v: k for k, v in mp.items()}
    mm = m.GetMol()
    mm.UpdatePropertyCache(strict=False)
    return [set(inv[x] for x in r) for r in Chem.GetSymmSSSR(mm)]


def _is_fused(mol, bl, adj) -> bool:
    """ortho- or ortho-and-peri-fused: no bond in more than two rings, no two rings
    sharing more than one bond."""
    bl = set(bl)
    rings = _sub_rings(mol, bl)
    e = sum(len(adj[a] & bl) for a in bl) // 2
    if e - len(bl) + 1 != len(rings):
        return False
    bond_count = defaultdict(int)
    for r in rings:
        for a in r:
            for b in adj[a]:
                if b in r and a < b:
                    bond_count[(a, b)] += 1
    if any(v > 2 for v in bond_count.values()):
        return False
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            sh = rings[i] & rings[j]
            if len(sh) > 2:
                return False
            if len(sh) == 2:
                a, b = tuple(sh)
                if b not in adj[a]:
                    return False
    return True


_BRIDGE_MORPH = [
    ('epoxymethano', 'OC'), ('methanoxy', 'CO'), ('epoxyethano', 'OCC'), ('epiminomethano', 'NC'),
    ('epithiomethano', 'SC'), ('methanooxymethano', 'COC'), ('epidioxy', 'OO'),
    ('epoxy', 'O'), ('epimino', 'N'), ('epithio', 'S'), ('episulfano', 'S'), ('epiazano', 'N'),
    ('methano', 'C'), ('ethano', 'CC'), ('propano', 'CCC'), ('butano', 'CCCC'), ('pentano', 'CCCCC'),
    ('hexano', 'CCCCCC'), ('metheno', 'C='), ('etheno', 'CC'), ('propeno', 'CCC'), ('buteno', 'CCCC'),
    ('benzeno', 'CCCCCC'),
]
_BM = "|".join(sorted({m for m, _ in _BRIDGE_MORPH}, key=len, reverse=True))
_BRIDGE_RE = re.compile(r"(?:^|[-\s\(\[\]])((?:" + _LOC + r",)+" + _LOC + r"(?::(?:" + _LOC + r",)+" + _LOC
                        + r")*)-(?:\[[\d,]+\])?(di|tri|tetra)?\(?((?:" + _BM + r"))\)?(?=[a-z\[\d\-])")


def _parse_bridges(name):
    out = []
    seen = set()
    for m in _BRIDGE_RE.finditer(name):
        tok = (m.group(1), m.group(2), m.group(3))
        if tok in seen:
            continue
        seen.add(tok)
        locsets = m.group(1).split(':')
        mult = {'di': 2, 'tri': 3, 'tetra': 4}.get(m.group(2) or '', 1)
        el = dict(_BRIDGE_MORPH)[m.group(3)]
        for k in range(max(mult, len(locsets))):
            out.append((locsets[min(k, len(locsets) - 1)], m.group(3), el.replace('=', '')))
    return out


def _bridged_needs(mol, bl, adj, deg, lam_map, name):
    """k of a bridged fused system: the fused system without the bridges named in ``name``
    ; (None, reason) when the bridges cannot be placed in exactly one way."""
    brs = _parse_bridges(name)
    if not brs:
        return None, 'no bridge prefix parsed'
    if any(b[1] == 'metheno' for b in brs):
        return None, 'metheno bridge'
    paths = []
    for start in sorted(bl):
        if len(adj[start] & bl) < 3:
            continue
        stack = [(start, [start])]
        while stack:
            v, path = stack.pop()
            for w in adj[v] & bl:
                if w in path:
                    continue
                if len(path) >= 2 and len(adj[w] & bl) >= 3:
                    if path[0] < w:
                        paths.append((path[0], tuple(path[1:]), w))
                    continue
                if len(adj[w] & bl) == 2 and len(path) <= 8:
                    stack.append((w, path + [w]))
    cands = []
    for _locs, _morph, el in brs:
        cs = []
        for s, inner, e in paths:
            if len(inner) != len(el):
                continue
            seq = ''.join(mol.GetAtomWithIdx(a).GetSymbol() for a in inner)
            if sorted(seq) == sorted(el):
                cs.append((s, inner, e))
        cands.append(cs)
    results = set()

    def loc_ok(loc, atom, sub):
        if re.search(r'[a-z]', loc):
            return len(sub[atom]) >= 3
        return len(sub[atom]) == 2 or mol.GetAtomWithIdx(atom).GetSymbol() != 'C'

    def rec(i, used, chosen):
        if i == len(cands):
            fused = bl - used
            sub = {a: adj[a] & fused for a in fused}
            if not fused or not _is_fused(mol, fused, sub):
                return
            lmap = {}
            for (locs, _morph, _el), (s0, _inner, e0) in zip(brs, chosen):
                ls = locs.split(',')
                if len(ls) != 2:
                    return
                ok_any = False
                for x, y in ((s0, e0), (e0, s0)):
                    trial = dict(lmap)
                    good = True
                    for loc, at in ((ls[0], x), (ls[1], y)):
                        if loc in trial and trial[loc] != at:
                            good = False
                        trial[loc] = at
                        if at not in fused or not loc_ok(loc, at, sub):
                            good = False
                    if good and len(set(trial.values())) == len(trial):
                        lmap = trial
                        ok_any = True
                        break
                if not ok_any:
                    return
            results.add(_n_ih(mol, fused, deg, lam_map))
            return
        for s0, inner, e0 in cands[i]:
            if used & set(inner):
                continue
            rec(i + 1, used | set(inner), chosen + [(s0, inner, e0)])

    rec(0, set(), [])
    if not results:
        return None, f'no consistent bridge choice {brs}'
    if len(results) > 1:
        return None, f'ambiguous {sorted(results)}'
    return next(iter(results)), 'bridged resolved'


def _assign_lambda(mol, name):
    toks = _lambda_tokens(name)
    if not toks:
        return {}
    cands = []
    for at in mol.GetAtoms():
        if not at.IsInRing() or at.GetSymbol() == 'C':
            continue
        if at.GetTotalValence() != STD_BOND.get(at.GetSymbol(), 4) or at.GetFormalCharge() != 0:
            cands.append(at.GetIdx())
    lam = {}
    for _loc, v in toks:
        best = None
        for a in cands:
            if a in lam:
                continue
            d = abs(mol.GetAtomWithIdx(a).GetTotalValence() - v)
            if best is None or d < best[0]:
                best = (d, a)
        if best is not None and best[0] <= 2:
            lam[best[1]] = v
    return lam


def _skel_key(mol, atoms):
    sub = Chem.RWMol()
    mp = {}
    for a in atoms:
        mp[a] = sub.AddAtom(Chem.Atom(mol.GetAtomWithIdx(a).GetAtomicNum()))
    s = set(atoms)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in s and j in s and b.IsInRing():
            sub.AddBond(mp[i], mp[j], Chem.BondType.SINGLE)
    try:
        return Chem.MolToSmiles(sub.GetMol())
    except Exception:
        return str(sorted(atoms))


def analyse(smiles: str, name: str) -> dict:  # noqa: C901 -- the rule's case analysis
    """verdict 'OMISSION' (fewer front groups than a ring system needs), 'extra_front_ih' (more
    front groups than ring systems that need one), 'unresolved' (a bridged system the reader
    cannot place) or 'ok'; with the requirement per ring skeleton (``reqd``) and the front and
    added groups read from the name."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return dict(error='bad smiles')
    vbs = _vb_descriptors(name)
    lam_map = _assign_lambda(mol, name)
    lname = name.lower()
    rss = ring_systems(mol)
    rs_of = {a: i for i, rs in enumerate(rss) for a in rs}
    keys = [_skel_key(mol, rs) for rs in rss]
    # ring-assembly bonds: identical ring systems joined directly
    asm_order = defaultdict(int)
    asm_links = []
    if _ASSEMBLY.search(name) or 'spirobi' in name:
        for b in mol.GetBonds():
            if b.IsInRing():
                continue
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in rs_of and j in rs_of and rs_of[i] != rs_of[j] and keys[rs_of[i]] == keys[rs_of[j]]:
                order = 2 if b.GetBondType() == Chem.BondType.DOUBLE else 1
                asm_order[i] += order
                asm_order[j] += order
                asm_links.append((rs_of[i], rs_of[j]))

    def _nrings(rs_):
        a_ = ring_adj(mol, sorted(rs_))
        return sum(len(v) for v in a_.values()) // 2 - len(rs_) + 1

    poly = [(len(rs_), i_) for i_, rs_ in enumerate(rss) if _nrings(rs_) >= 2]
    stereo_rs = max(poly)[1] if poly else -1
    rs_items = []
    for ri, rs in enumerate(rss):
        rs = set(rs)
        adj = ring_adj(mol, sorted(rs))
        deg = {a: len(adj[a]) + asm_order.get(a, 0) for a in rs}
        blks = blocks(sorted(rs), adj)

        def rings_in(bl):
            return sum(len(adj[a] & bl) for a in bl) // 2 - len(bl) + 1

        units = []
        total_rings = sum(rings_in(set(b)) for b in blks)
        charged = any(mol.GetAtomWithIdx(a).GetFormalCharge() and a not in lam_map for a in rs)
        retained_cation = any(t in lname for t in _RETAINED_CATIONS)
        need = 0
        note = ''
        if total_rings >= 2 and any(t in lname for t in _STEREOPARENT_STEMS) and ri == stereo_rs:
            units.append(('stereoparent', 0))
        elif total_rings >= 2 and any(t in lname for t in _NUCLEOSIDE_STEMS) and \
                {mol.GetAtomWithIdx(a).GetSymbol() for a in rs} <= {'C', 'N'}:
            units.append(('nucleoside-retained', 0))
        elif len(blks) > 1 and all(rings_in(set(b)) == 1 for b in blks):
            units.append(('spiro-vonbaeyer', 0))
        elif len(blks) == 1 and any(k.endswith('cyclo') and n and sum(n) + 2 == len(rs) and r == total_rings
                                    for k, n, r in vbs):
            units.append(('vonbaeyer', 0))
        elif any(t in lname for t in _SAT_RETAINED) and kekule_ring_double(mol, rs) == 0 and total_rings > 1:
            units.append(('retained-saturated', 0))
        else:
            mancude_atoms = set()
            for bl in blks:
                bl = set(bl)
                r = rings_in(bl)
                hetero = {mol.GetAtomWithIdx(a).GetSymbol() for a in bl} - {'C'}
                if r == 1:
                    if not hetero:
                        units.append(('carbocycle-mono', 0))
                    elif len(bl) > 10:
                        units.append(('large-heteromono', 0))
                    elif kekule_ring_double(mol, bl) == 0:
                        units.append(('saturated-heteromono', 0))
                    else:
                        units.append(('heteromono', None))
                        mancude_atoms |= bl
                    continue
                fused = _is_fused(mol, bl, adj)
                vb_exact = any(k.endswith('cyclo') and n and sum(n) + 2 == len(bl) and rr == r for k, n, rr in vbs)
                vb_loose = (not fused) and any(k.endswith('cyclo') and rr == r for k, n, rr in vbs)
                if vb_exact or vb_loose:
                    units.append(('vonbaeyer', 0))
                    continue
                if fused:
                    units.append(('fused', None))
                    mancude_atoms |= bl
                else:
                    k, bnote = _bridged_needs(mol, bl, adj, deg, lam_map, name)
                    units.append(('bridged', k, bnote))
                    if k is None:
                        need = None
                        note = bnote
                    elif need is not None:
                        need += k
            if mancude_atoms and need is not None:
                cm = retained_cation and charged
                k = _n_ih(mol, mancude_atoms, deg, lam_map, charge_mode=cm)
                need += k
                units = [(u[0], k) if u[1] is None else u for u in units]
                if charged and not cm:
                    note = 'ionic (neutral parent hydride rule, P-73/P-72 operations)'
        rs_items.append(dict(rs=ri, key=keys[ri], units=units, need=need, note=note, atoms=sorted(rs)))
    parent = list(range(len(rss)))

    def f(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for a, b in asm_links:
        parent[f(a)] = f(b)
    merged = defaultdict(list)
    for it in rs_items:
        merged[f(it['rs'])].append(it)
    reqs = []
    unresolved = []
    for its in merged.values():
        if any(i['need'] is None for i in its):
            unresolved.append(its)
            continue
        tot = sum(i['need'] for i in its)
        if tot > 0:
            reqs.append((tuple(sorted(i['key'] for i in its)), tot))
    # identical parents (multiplied substituents) share one group
    reqd = {}
    for key, tot in reqs:
        reqd[key] = max(reqd.get(key, 0), tot)
    fg = front_ih_groups(name)
    pool = sorted(g[2] for g in fg)
    ok = True
    for nreq in sorted(reqd.values(), reverse=True):
        fit = [g for g in pool if g >= nreq]
        if not fit:
            ok = False
            break
        pool.remove(min(fit))
    if not ok:
        verdict = 'OMISSION'
    elif unresolved:
        verdict = 'unresolved'
    elif len(fg) > len(reqs):
        verdict = 'extra_front_ih'
    else:
        verdict = 'ok'
    return dict(verdict=verdict, reqd=reqd, front=fg, added=added_ih_groups(name), items=rs_items,
                unresolved=[[(i['key'], i['units'], i['note']) for i in its] for its in unresolved],
                lam=lam_map)
