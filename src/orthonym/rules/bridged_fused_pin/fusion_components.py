"""The fusion components of a fused ring skeleton, their PIN names, their
own numberings and their seniority, for the two-component fusion names of
``hetero_fusion``.

 (the Blue Book): "Fusion components are mancude or ring systems that can
be named without the application of any fusion nomenclature principles." The components are:

- a heteromonocycle of 3 to 10 members: its Hantzsch-Widman name, or a Table 2.2
  retained name, except that 'isothiazole', 'isoxazole', 'thiazole' and 'oxazole' are never
  used: (:11982) "The Hantzsch-Widman names 1,2-thiazole, 1,2-oxazole, 1,3-
  thiazole, and 1,3-oxazole, respectively, must be used; the locants are enclosed in square
  brackets in the completed fusion name." Numbered by (:8284): "The locant '1' is
  given to a heteroatom that occurs first in the seniority sequence... The numbering is then
  chosen to give lowest locants to heteroatoms considered as a set";
- a heteromonocycle of more than ten members: (:11984) the 'ine' names of
   (:11735) "changing the ending 'ane' of the corresponding saturated
  heteromonocycle... to 'ine'. Their locants are cited in front of the name... in the
  order of the appearance of the corresponding replacement ('a') prefixes";
- a monocyclic hydrocarbon: 'benzo' or a (:12000) 'cyclo...a' prefix;
- a polycyclic system with a retained or systematic one-component name of /
  (the engine's parent tables: ``parents._table_index``), numbered by the table;
- a benzo name (:11815), formed only by ``hetero_fusion`` (the second stage):
  "a benzene ring... ortho-fused to a heteromonocyclic component of five or more members
  ... treated together as a one-component unit... The locant '1' is always assigned to the
  atom of the heterocyclic component next to a fusion atom. Heteroatoms are allocated lowest
  locants as a set, without regard to kind; if there is a choice, lowest locants are
  assigned in accordance with the seniority of the 'a' prefixes... for preferred IUPAC names
  locants must be cited. The letter 'o' of the 'benzo' prefix is elided when followed by a
  vowel."

Seniority, (:12137) "the following criteria are considered, in order, until a
decision can be made": (a):12139 the senior heteroatom N > F > Cl > Br > I > O > S > Se > Te
> P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In > Tl; (b):12163 more rings; (c)
:12234 the larger ring at the first point of difference; (d):12260 more heteroatoms; (e)
:12282 more kinds of heteroatom; (f):12298 more heteroatoms first in F > Cl > Br > I > O > S >
Se > Te > N > P...; (g):12317 more rings in a horizontal row; (h):12392 lower heteroatom
locants; (i):12407 lower heteroatom locants in the order of (f); (j):12418 lower locants
of the peripheral fusion carbon atoms. (h)-(j) read each component's own numbering (the
book's:12403 "locants '1,2' of pyridazine preferred to locants '1,4' of pyrazine").
"""
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from rdkit import Chem

#: (:8284) citation order of the 'a' prefixes, also the order of
#: (f) (:12298) and (i) (:12407)
HW_ORDER = ("F", "Cl", "Br", "I", "O", "S", "Se", "Te", "N", "P", "As", "Sb", "Bi", "Si", "Ge",
            "Sn", "Pb", "B", "Al", "Ga", "In", "Tl")
#: (a) (:12139)
SENIOR_A = ("N", "F", "Cl", "Br", "I", "O", "S", "Se", "Te", "P", "As", "Sb", "Bi", "Si", "Ge",
            "Sn", "Pb", "B", "Al", "Ga", "In", "Tl")
#: Table 2.2 (:8111-:8178) retained names of mancude heteromonocycles, keyed by the
#: Hantzsch-Widman name they replace; 'oxazole', 'isoxazole', 'thiazole', 'isothiazole' are
#: not used for fusion PINs (:11982), so 1,3-/1,2-oxazole and -thiazole keep their
#: Hantzsch-Widman names. The single-heteroatom stems (azole -> pyrrole, oxole -> furan,
#: azine -> pyridine, oxine -> pyran,...) are ``heterocycles._apply_retained_stem``'s.
_TABLE_2_2 = {"1,3-diazole": "imidazole", "1,2-diazole": "pyrazole", "1,2-diazine": "pyridazine",
              "1,3-diazine": "pyrimidine", "1,4-diazine": "pyrazine"}
#: (:12047): the only contracted prefixes of PINs
CONTRACTED = {"anthracene": "anthra", "naphthalene": "naphtho", "benzene": "benzo",
              "phenanthrene": "phenanthro", "furan": "furo", "imidazole": "imidazo",
              "pyridine": "pyrido", "pyrimidine": "pyrimido", "thiophene": "thieno"}
#: (Table 1.4) numerical terms 3-20 (21 and above: ``chain_names.numerical_term``)
_NUMERAL = {3: "propa", 4: "buta", 5: "penta", 6: "hexa", 7: "hepta", 8: "octa", 9: "nona",
            10: "deca", 11: "undeca", 12: "dodeca", 13: "trideca", 14: "tetradeca",
            15: "pentadeca", 16: "hexadeca", 17: "heptadeca", 18: "octadeca", 19: "nonadeca",
            20: "icosa"}
#: / Table 1.5 'a' prefixes (replacement nomenclature keeps every final 'a')
_A_PREFIX = {"O": "oxa", "S": "thia", "Se": "selena", "Te": "tellura", "N": "aza", "P": "phospha",
             "As": "arsa", "Sb": "stiba", "Bi": "bisma", "Si": "sila", "Ge": "germa", "Sn": "stanna",
             "Pb": "plumba", "B": "bora", "Al": "aluma", "Ga": "galla", "In": "indiga", "Tl": "thalla"}
_MULT = {2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa", 7: "hepta", 8: "octa", 9: "nona",
         10: "deca"}


def numeral(n: int) -> Optional[str]:
    if n in _NUMERAL:
        return _NUMERAL[n]
    if n < 3 or n > 9999:
        return None
    from ...data.chain_names import numerical_term
    return numerical_term(n)


@dataclass(frozen=True)
class Component:
    """One fusion component on the skeleton: ``kind`` 'hetero' (a heteromonocycle), 'carbo'
    (a monocyclic hydrocarbon), 'retained' (a one-component polycyclic name of the tables) or
    'benzo' (a benzo name); ``name`` its PIN parent name without indicated
    hydrogen ('1,2-oxazole', 'pyridine', 'quinoline', '1-benzopyran', 'benzene');
    ``rings`` indices into the skeleton's ring list; ``numberings`` every numbering the
    component's own rules allow, as {skeleton atom: locant string}."""
    kind: str
    name: str
    rings: FrozenSet[int]
    atoms: FrozenSet[int]
    numberings: Tuple[Tuple[Tuple[int, str], ...], ...]

    def numbering_dicts(self) -> List[Dict[int, str]]:
        return [dict(n) for n in self.numberings]


def split_locants(name: str) -> Tuple[Optional[str], str]:
    """'1,2-oxazole' -> ('1,2', 'oxazole'); 'quinoline' -> (None, 'quinoline')."""
    head, sep, rest = name.partition("-")
    if sep and head and all(p.isdigit() for p in head.split(",")):
        return head, rest
    return None, name


def base_form(comp: Component) -> str:
    """The component as the parent of a fusion name: its heteroatom locants in square
    brackets:11907 "Locants that describe structural features of components,
    such as positions of heteroatoms, are kept with the name of the component and are
    enclosed within square brackets"): '[1,2]oxazole', '[2,1]benzothiazole', 'pyridine'."""
    loc, stem = split_locants(comp.name)
    return f"[{loc}]{stem}" if loc else stem


def prefix_form(comp: Component) -> Optional[str]:
    """The attached-component prefix: (:12047) the contracted prefixes, else
     (:11905) / (:12024) "changing the final letter 'e' into the letter
    'o' or by adding the letter 'o' when no final letter 'e' is present"; (:12000)
    'cyclo...a' for a monocyclic hydrocarbon other than benzene; heteroatom locants in
    brackets. None for a benzo name on furan or thiophene: the book prints no PIN with
    '[1]benzofuro' / '[1]benzofurano' (its only such spelling,:13461, is a rejected name), so
    that prefix is not certified."""
    if comp.kind == "carbo":
        size = len(comp.atoms)
        if size == 6:
            return "benzo"
        num = numeral(size)
        return f"cyclo{num}" if num else None
    loc, stem = split_locants(comp.name)
    if stem in CONTRACTED:
        pre = CONTRACTED[stem]
    elif comp.kind == "benzo" and (stem.endswith("furan") or stem.endswith("thiophene")):
        return None
    elif stem.endswith("e"):
        pre = stem[:-1] + "o"
    else:
        pre = stem + "o"
    return f"[{loc}]{pre}" if loc else pre


# --------------------------------------------------------------------------------------
# rings of a skeleton
# --------------------------------------------------------------------------------------

def ring_cycle(km, ring: FrozenSet[int]) -> Optional[List[int]]:
    """The atoms of ``ring`` in bond order around it."""
    atoms = sorted(ring)
    start = atoms[0]
    order = [start]
    prev = None
    cur = start
    while True:
        nbs = [nb.GetIdx() for nb in km.GetAtomWithIdx(cur).GetNeighbors()
               if nb.GetIdx() in ring and nb.GetIdx() != prev]
        if prev is None:
            nbs = sorted(nbs)[:1]
        nxt = [x for x in nbs if x != prev]
        if not nxt:
            return None
        prev, cur = cur, nxt[0]
        if cur == start:
            break
        if cur in order:
            return None
        order.append(cur)
    return order if len(order) == len(ring) else None


def ortho_rings(km) -> Optional[List[FrozenSet[int]]]:
    """The rings of an ortho-fused skeleton (every atom in at most two rings, two rings share
    nothing or one bond), or None (peri-fusion and interior atoms are slice S2c-3)."""
    km = Chem.Mol(km)
    if not km.GetRingInfo().NumRings():
        Chem.FastFindRings(km)
    rings = [frozenset(r) for r in Chem.GetSymmSSSR(km)]
    if not rings or len(rings) != km.GetNumBonds() - km.GetNumAtoms() + 1:
        return None
    if set().union(*rings) != set(range(km.GetNumAtoms())):
        return None
    for a in range(km.GetNumAtoms()):
        if sum(1 for r in rings if a in r) > 2:
            return None
    for i, x in enumerate(rings):
        for y in rings[i + 1:]:
            s = x & y
            if len(s) not in (0, 2):
                return None
            if len(s) == 2 and km.GetBondBetweenAtoms(*tuple(s)) is None:
                return None
    return sorted(rings, key=lambda r: tuple(sorted(r)))


# --------------------------------------------------------------------------------------
# monocycles
# --------------------------------------------------------------------------------------

def mono_numberings(km, ring: FrozenSet[int], hw: bool) -> List[Dict[int, int]]:
    """Every numbering of a heteromonocycle its rules allow. ``hw`` (3-10 members):
     locant 1 on an atom of the element first in ``HW_ORDER``, then the lowest
    heteroatom locant set, then the lowest locants in ``HW_ORDER``; otherwise (the 'ine'
    names) the lowest set first, then the same order."""
    cyc = ring_cycle(km, ring)
    if cyc is None:
        return []
    n = len(cyc)
    het = {a: km.GetAtomWithIdx(a).GetSymbol() for a in cyc if km.GetAtomWithIdx(a).GetAtomicNum() != 6}
    if not het:
        return []
    if any(e not in HW_ORDER for e in het.values()):
        return []                         # no 'a' prefix of (:8284): not named here
    senior = min(het.values(), key=HW_ORDER.index)
    best, out = None, []
    for s in range(n):
        for d in (1, -1):
            order = [cyc[(s + d * k) % n] for k in range(n)]
            if hw and het.get(order[0]) != senior:
                continue
            num = {a: k + 1 for k, a in enumerate(order)}
            key = (tuple(sorted(num[a] for a in het)),
                   tuple(tuple(sorted(num[a] for a in het if het[a] == e)) for e in HW_ORDER))
            if best is None or key < best:
                best, out = key, [num]
            elif key == best:
                out.append(num)
    return out


def hetero_mono_name(km, ring: FrozenSet[int], numbering: Dict[int, int]) -> Optional[str]:
    """The PIN name of the mancude heteromonocycle (no indicated hydrogen), as a fusion
    ``Component`` (:func:`base_form` /:func:`prefix_form`): (:11907) "Locants
    that describe structural features of components, such as positions of heteroatoms,
    are kept with the name of the component" -- every component of a fusion name keeps its
    heteroatom locants, whether cited as the parent or an attached component, even for a
    ring (:8470) calls unambiguous standing alone ('[1,2,3,4]tetrazolo[1,5-
    a]pyridine', not 'tetrazolo[1,5-a]pyridine', although bare '1H-tetrazole' cites none)."""
    n = len(ring)
    het = sorted(((numbering[a], km.GetAtomWithIdx(a).GetSymbol()) for a in ring
                  if km.GetAtomWithIdx(a).GetAtomicNum() != 6))
    if n <= 10:
        from ..heterocycles import _apply_retained_stem, build_hw_name
        name = build_hw_name(het, n, False, True, describe_structure=True)
        if not name:
            return None
        name = _apply_retained_stem(name)
        return _TABLE_2_2.get(name, name)
    num = numeral(n)
    if num is None:
        return None
    by_el: Dict[str, List[int]] = {}
    for loc, el in het:
        by_el.setdefault(el, []).append(loc)
    parts, locs = [], []
    for el in sorted(by_el, key=HW_ORDER.index):
        pre = _A_PREFIX.get(el)
        if pre is None:
            return None
        cnt = len(by_el[el])
        parts.append((_MULT[cnt] if cnt > 1 else "") + pre)
        locs.extend(sorted(by_el[el]))
    body = "".join(parts) + "cyclo" + num[:-1] + "ine"
    return ",".join(str(x) for x in locs) + "-" + body


def hetero_mono_stem(km, ring: FrozenSet[int], numbering: Dict[int, int]) -> Optional[str]:
    """The heteromonocycle name without its heteroatom locants (the stem of a benzo name):
    'dioxole', 'oxazole', 'furan', 'imidazole', 'dioxacyclooctadecine'."""
    name = hetero_mono_name(km, ring, numbering)
    if name is None:
        return None
    return split_locants(name)[1]


# --------------------------------------------------------------------------------------
# benzo names
# --------------------------------------------------------------------------------------

def benzo_numberings(km, benzene: FrozenSet[int], hetero: FrozenSet[int]) -> List[Dict[int, str]]:
    """ (:11815): the heterocyclic ring numbered first from an atom next to a fusion
    atom, then the fusion atom, the four benzene atoms and the other fusion atom ('4a', '8a'
    style letters); lowest heteroatom locants as a set, then in 'a'-prefix seniority."""
    shared = benzene & hetero
    if len(shared) != 2:
        return []
    hcyc = ring_cycle(km, hetero)
    bcyc = ring_cycle(km, benzene)
    if hcyc is None or bcyc is None:
        return []
    k = len(hetero)
    het = {a: km.GetAtomWithIdx(a).GetSymbol() for a in hetero if km.GetAtomWithIdx(a).GetAtomicNum() != 6}
    best, out = None, []
    for f_start in shared:
        f_end = next(iter(shared - {f_start}))
        # walk the hetero ring from f_start away from f_end
        i = hcyc.index(f_start)
        for d in (1, -1):
            if hcyc[(i + d) % k] == f_end:
                continue
            path = [hcyc[(i + d * s) % k] for s in range(1, k - 1)]
            if hcyc[(i + d * (k - 1)) % k] != f_end:
                continue
            num: Dict[int, str] = {a: str(s + 1) for s, a in enumerate(path)}
            num[f_end] = f"{k - 2}a"
            j = bcyc.index(f_end)
            for d2 in (1, -1):
                if bcyc[(j + d2) % 6] == f_start:
                    continue
                walk = [bcyc[(j + d2 * s) % 6] for s in range(1, 5)]
                if bcyc[(j + d2 * 5) % 6] != f_start:
                    continue
                full = dict(num)
                for s, a in enumerate(walk):
                    full[a] = str(k - 1 + s)
                full[f_start] = f"{k + 2}a"
                key = (tuple(sorted(int(full[a]) for a in het)),
                       tuple(tuple(sorted(int(full[a]) for a in het if het[a] == e)) for e in HW_ORDER))
                if best is None or key < best:
                    best, out = key, [full]
                elif key == best:
                    out.append(full)
    return out


def benzo_name(km, benzene: FrozenSet[int], hetero: FrozenSet[int],
               numbering: Dict[int, str]) -> Optional[str]:
    """'1,3-benzodioxole', '2,1-benzoxazole', '1-benzopyran', '3-benzazacycloundecine'."""
    hn = mono_numberings(km, hetero, hw=len(hetero) <= 10)
    if not hn:
        return None
    stem = hetero_mono_stem(km, hetero, hn[0])
    if stem is None:
        return None
    het = [(int(numbering[a]), km.GetAtomWithIdx(a).GetSymbol()) for a in hetero
           if km.GetAtomWithIdx(a).GetAtomicNum() != 6]
    locs = [str(loc) for el in HW_ORDER for loc, e in sorted(het) if e == el]
    benz = "benz" if stem[0] in "aeiou" else "benzo"
    return ",".join(locs) + "-" + benz + stem


# --------------------------------------------------------------------------------------
# retained polycyclic components (the parent tables)
# --------------------------------------------------------------------------------------

def is_one_component_name(name: str) -> bool:
    """A retained or systematic / name: no fusion descriptor, not a benzo name
    (benzo names are formed in the second stage)."""
    import re
    if "[" in name:
        return False
    return not re.match(r"^\d+(?:,\d+)*-benz", name)


def retained_component(km, atoms: FrozenSet[int]):
    """(name, numberings) of the one-component table name of the sub-skeleton on ``atoms``,
    or None. Only skeletons a table names are looked up, so ``parents._parent_for_key``
    never reaches a producer from here."""
    from .parents import _IH, _key_of, _mancude, _skeleton, _table_index, fused_parent, is_pin_parent_name
    sk, _ = _skeleton(km, atoms)
    key = Chem.MolToSmiles(sk)
    names = {e[0] for e in _table_index().get(key, ())}
    if not names:
        from ...data import get_retained_name
        std = _mancude(Chem.MolFromSmiles(key))
        retained = get_retained_name(Chem.MolToSmiles(std)) if std is not None else None
        if retained and is_pin_parent_name(_IH.sub("", retained)):
            names = {_IH.sub("", retained)}
    names = {n for n in names if is_one_component_name(n)}
    if len(names) != 1:
        return None
    got = fused_parent(km, set(atoms))
    if got is None or got.name not in names:
        return None
    return got.name, [{a: str(l) for a, l in num.items()} for num in got.numberings]


# --------------------------------------------------------------------------------------
# seniority
# --------------------------------------------------------------------------------------

def _loc_key(loc: str) -> Tuple[int, str]:
    digits = "".join(c for c in loc if c.isdigit())
    return (int(digits), loc[len(digits):])


def rank_key(km, comp: Component, rings: Sequence[FrozenSet[int]]) -> Optional[tuple]:
    """ (a)-(f), (h)-(j) as a key whose minimum is the senior component; (g) is
    ``rows_in_line``, asked by the caller only when (a)-(f) tie. None for a component with a
    heteroatom outside the order of (a) (:12139): the caller declines."""
    sym = [km.GetAtomWithIdx(a).GetSymbol() for a in comp.atoms]
    het = [s for s in sym if s != "C"]
    if any(s not in SENIOR_A for s in het):
        return None
    a = min((SENIOR_A.index(s) for s in het), default=99)
    sizes = tuple(sorted((len(rings[i]) for i in comp.rings), reverse=True))
    b = -len(comp.rings)
    c = tuple(-s for s in sizes)
    d = -len(het)
    e = -len(set(het))
    f = tuple(-het.count(x) for x in HW_ORDER)
    hij = []
    for num in comp.numbering_dicts():
        hl = tuple(sorted(_loc_key(num[x]) for x in comp.atoms if km.GetAtomWithIdx(x).GetAtomicNum() != 6))
        il = tuple(tuple(sorted(_loc_key(num[x]) for x in comp.atoms
                                if km.GetAtomWithIdx(x).GetSymbol() == el)) for el in HW_ORDER)
        jl = tuple(sorted(_loc_key(num[x]) for x in comp.atoms
                          if km.GetAtomWithIdx(x).GetAtomicNum() == 6
                          and sum(1 for i in comp.rings if x in rings[i]) == 2))
        hij.append((hl, il, jl))
    h, i, j = min(hij) if hij else ((), (), ())
    return (a, b, c, d, e, f), (h, i, j)


def rows_in_line(km, comp: Component) -> Optional[int]:
    """ (g) (:12317) "the greatest number of rings in a horizontal row when it is
    drawn in the preferred orientation according to ", from the grid drawing of
    ``fusion_orientation`` (None when the component cannot be drawn)."""
    if len(comp.rings) <= 2:
        return len(comp.rings)
    from .parents import _skeleton
    from ..fusion_orientation import _grid_orientations_uncached, score_orientation_grid
    sk, _ = _skeleton(km, comp.atoms)
    sk = Chem.MolFromSmiles(Chem.MolToSmiles(sk))      # with its rings perceived
    got = _grid_orientations_uncached(sk, set(range(sk.GetNumAtoms()))) if sk is not None else None
    if not got:
        return None
    graph, layouts = got
    centres, coords = layouts[0]
    return -score_orientation_grid(graph, centres, coords)[0]


def peripheral_cycle(km, comp_atoms: FrozenSet[int], ring_sets: Sequence[FrozenSet[int]]) -> Optional[List[int]]:
    """The periphery of a cata-condensed component: the cycle of its bonds that lie in one of
    its rings only."""
    def in_rings(u, v):
        return sum(1 for r in ring_sets if u in r and v in r)
    adj: Dict[int, List[int]] = {a: [] for a in comp_atoms}
    for b in km.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if u in comp_atoms and v in comp_atoms and in_rings(u, v) == 1:
            adj[u].append(v)
            adj[v].append(u)
    if any(len(x) != 2 for x in adj.values()):
        return None
    start = min(comp_atoms)
    order, prev, cur = [start], None, start
    while True:
        nxt = [x for x in adj[cur] if x != prev]
        if prev is None:
            nxt = sorted(nxt)[:1]
        prev, cur = cur, nxt[0]
        if cur == start:
            break
        order.append(cur)
    return order if len(order) == len(comp_atoms) else None


def side_letters(km, comp: Component, numbering: Dict[int, str],
                 ring_sets: Sequence[FrozenSet[int]]) -> Optional[Dict[FrozenSet[int], Tuple[str, Tuple[int, int]]]]:
    """ (:11911): the peripheral sides lettered 'a' (1-2), 'b' (2-3),... in the
    order of the periphery from locant 1 towards the lower of its neighbours' locants (the
    sides of fusion atoms included; acridine and carbazole keep their own numbering).
    {bond atoms: (letter, (first atom, second atom) in the direction of lettering)}."""
    sets = [ring_sets[i] for i in comp.rings]
    per = peripheral_cycle(km, comp.atoms, sets)
    if per is None:
        return None
    one = [a for a, l in numbering.items() if l == "1"]
    if len(one) != 1:
        return None
    n = len(per)
    i = per.index(one[0])
    fwd, back = per[(i + 1) % n], per[(i - 1) % n]
    d = 1 if _loc_key(numbering[fwd]) < _loc_key(numbering[back]) else -1
    walk = [per[(i + d * k) % n] for k in range(n)]
    out = {}
    for k in range(min(n, 26)):          # a side past 'z' gets no letter here
        u, v = walk[k], walk[(k + 1) % n]
        out[frozenset((u, v))] = (chr(ord("a") + k), (u, v))
    return out
