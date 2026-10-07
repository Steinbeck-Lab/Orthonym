"""Two-component fusion names with a heterocyclic component, and benzo names, for
the ortho-fused parents no table holds (slice S2c-1), numbered by OPSIN's reading of the name.

 (the Blue Book): ring systems "that have no accepted retained or systematic
name described in sections and are named by prefixing to the name of a
component ring or ring system (the parent component) designations of the other component(s)
(attached components)." The name is formed in two stages:

1. Components (``fusion_components``): the monocycles and the one-component polycyclic names
   of the tables, retained names used whole. The parent component is the senior component of
   the whole skeleton by (a)-(j) (:12137), every component that occurs counted
   (the book's own reasons rank components that are not in the final name: carbazole over
   quinoxaline:12232, acridine over phenanthridine:12386). The rest of the skeleton must be
   one component, fused to the parent by one bond, or two components of which one is a benzene
   ring that the second stage merges.
2. Benzo names: (:11815) "If the initial identified preferred components for
   naming a fused ring system include an isolated benzo component (i.e. not forming part of a
   component with a retained name such as quinoline or anthracene) ortho-fused to a
   heteromonocyclic component of five or more members, these two components are treated
   together as a one-component unit"; (:13437) "They may be treated as a parent
   component or an attached component depending on the order of seniority... However, this
   approach is not used if it disrupts a multiparent system... or the use of multiplicative
   prefixes";:13443 'thieno[3,2-f][2,1]benzothiazole (PIN) (2,1-benzothiazole is senior to
   1-benzothiophene)',:13449 '[1,2]benzoxazolo[6,5-g]quinoline (PIN)',:13459
   '4H-[1,4]thiazino[2,3-g]quinoline (PIN) (the retained name quinoline must be used)',:13463
   '6H-dibenzo[b,d]pyran (PIN) (not 6H-benzo[c][1]benzopyran...)'. A whole skeleton made of
   one benzene ring and one heteromonocycle of five or more members is a benzo name.

Declined (other slices): multiparent names;:13453 'benzo[1,2-b:4,5-c']difuran
(PIN) (not furo[3,4-f][1]benzofuran...)',:41087 'benzo[1,2-c:4,5-c']dipyrrole (not
pyrrolo[3,4-f]isoindole...)'), multiplied attached components:13467), three or
more components, peri-fusion and interior atoms, a heteroatom that needs the lambda-convention,
a skeleton the tables name, and any name OPSIN 2.9.0 cannot read back to the same skeleton.

The descriptor: (:11911) the parent sides lettered "a for the side numbered '1,2', b
for '2,3'... To the letter as early in the alphabet as possible that denotes the side where
the fusion occurs are prefixed, if necessary, the numbers of the positions of attachment of the
other component. These numbers are chosen to be as low as is consistent with the numbering of
the compound and their order conforms to the direction of lettering of the parent component";
 (:11917) brackets, no space or hyphen around them; (:13970) the numbers
are omitted for 'benzo' and the monocyclic hydrocarbon prefixes ('benzo[g]quinoline (PIN)'
:13978); (:14007) both terminal locants, a fusion atom's included
('naphtho[1,8a-b]azirine (PIN)':14011); Note (:11909) no elision;
(:23710) at least two rings of five or more members.
"""
from functools import lru_cache
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from rdkit import Chem

from . import fusion_components as fc


def _atoms_of(rings: Sequence[FrozenSet[int]], idx) -> FrozenSet[int]:
    return frozenset().union(*(rings[i] for i in idx))


def _connected(rings, idx) -> bool:
    idx = list(idx)
    seen, stack = {idx[0]}, [idx[0]]
    while stack:
        c = stack.pop()
        for j in idx:
            if j not in seen and len(rings[c] & rings[j]) == 2:
                seen.add(j)
                stack.append(j)
    return len(seen) == len(idx)


def _is_benzene(km, ring) -> bool:
    return len(ring) == 6 and all(km.GetAtomWithIdx(a).GetAtomicNum() == 6 for a in ring)


def _is_hetero(km, ring) -> bool:
    return any(km.GetAtomWithIdx(a).GetAtomicNum() != 6 for a in ring)


def _frozen(num: Dict[int, object]) -> Tuple[Tuple[int, str], ...]:
    return tuple(sorted((a, str(l)) for a, l in num.items()))


def _mono(km, rings, i) -> Optional[fc.Component]:
    ring = rings[i]
    if not _is_hetero(km, ring):
        return fc.Component("carbo", "benzene" if len(ring) == 6 else f"cyclo{len(ring)}",
                            frozenset([i]), ring, ())
    nums = fc.mono_numberings(km, ring, hw=len(ring) <= 10)
    if not nums:
        return None
    names = {fc.hetero_mono_name(km, ring, n) for n in nums}
    if len(names) != 1 or None in names:
        return None
    return fc.Component("hetero", names.pop(), frozenset([i]), ring,
                        tuple(_frozen(n) for n in nums))


def _stage1_components(km, rings) -> Dict[FrozenSet[int], fc.Component]:
    """Every component occurring in the skeleton: each ring, and each connected set of
    rings with a one-component table name."""
    out: Dict[FrozenSet[int], fc.Component] = {}
    n = len(rings)
    for i in range(n):
        c = _mono(km, rings, i)
        if c is not None:
            out[frozenset([i])] = c
    for size in range(2, n + 1):
        for idx in combinations(range(n), size):
            if not _connected(rings, idx):
                continue
            atoms = _atoms_of(rings, idx)
            got = fc.retained_component(km, atoms)
            if got is None:
                continue
            name, nums = got
            out[frozenset(idx)] = fc.Component("retained", name, frozenset(idx), atoms,
                                               tuple(_frozen(x) for x in nums))
    return out


def _benzo_unit(km, rings, b, h) -> Optional[fc.Component]:
    nums = fc.benzo_numberings(km, rings[b], rings[h])
    if not nums:
        return None
    names = {fc.benzo_name(km, rings[b], rings[h], x) for x in nums}
    if len(names) != 1 or None in names:
        return None
    return fc.Component("benzo", names.pop(), frozenset([b, h]), rings[b] | rings[h],
                        tuple(_frozen(x) for x in nums))


def _min_partitions(rings, comps, rest: FrozenSet[int]) -> List[List[fc.Component]]:
    """Every partition of the rings ``rest`` into the fewest stage-1 components."""
    best: List[List[fc.Component]] = []
    best_n = [99]

    def go(left: FrozenSet[int], acc: List[fc.Component]):
        if len(acc) > best_n[0]:
            return
        if not left:
            if len(acc) < best_n[0]:
                best_n[0] = len(acc)
                best.clear()
            if len(acc) == best_n[0]:
                best.append(list(acc))
            return
        low = min(left)
        for key, comp in comps.items():
            if low in key and key <= left:
                go(left - key, acc + [comp])

    go(rest, [])
    return best


def _senior(km, rings, cands: List[fc.Component]) -> Optional[List[fc.Component]]:
    """The senior candidates by (a)-(j): all candidates of the senior name kind
    that tie through every criterion (identical components in other places), or None when a
    tie of different names is left or (g) is needed and cannot be drawn."""
    keyed = [(fc.rank_key(km, c, rings), c) for c in cands]
    if any(k is None for k, _ in keyed):
        return None
    top_af = min(k[0] for k, _ in keyed)
    left = [(k, c) for k, c in keyed if k[0] == top_af]
    if len({c.name for _, c in left}) > 1:
        rows = {}
        for _k, c in left:
            r = fc.rows_in_line(km, c)
            if r is None:
                return None
            rows[id(c)] = r
        top_g = max(rows.values())
        left = [(k, c) for k, c in left if rows[id(c)] == top_g]
    top_hij = min(k[1] for k, _ in left)
    left = [c for k, c in left if k[1] == top_hij]
    if len({c.name for c in left}) != 1:
        return None
    return left


def _one_bond(rings, a: fc.Component, b: fc.Component) -> Optional[Tuple[int, int]]:
    shared = a.atoms & b.atoms
    if len(shared) != 2:
        return None
    pairs = [(i, j) for i in a.rings for j in b.rings if len(rings[i] & rings[j]) == 2]
    if len(pairs) != 1:
        return None
    return tuple(sorted(shared))


def _multi(km, rings, base: fc.Component, att: fc.Component) -> bool:
    """ / on the two-component reading [monocycle M, polycycle P]: a ring of P
    with M's size and elements, not fused to M, both fused to one ring of P, makes a
    multiparent name (M a heteromonocycle::13453,:41087) or two 'benzo' prefixes (M benzene
    and P a benzo name: 'dibenzo[b,d]furan':7475); M benzene and P a retained name keeps the
    two-component name ('benzo[b]phosphindole':41804)."""
    for m, p in ((base, att), (att, base)):
        if len(m.rings) != 1 or len(p.rings) < 2:
            continue
        mi = next(iter(m.rings))
        mr = rings[mi]
        sig = (len(mr), tuple(sorted(km.GetAtomWithIdx(a).GetAtomicNum() for a in mr)))
        for k in p.rings:
            r = rings[k]
            if r & mr:
                continue
            if (len(r), tuple(sorted(km.GetAtomWithIdx(a).GetAtomicNum() for a in r))) != sig:
                continue
            if not any(len(rings[y] & mr) == 2 and len(rings[y] & r) == 2 for y in p.rings if y != k):
                continue
            if _is_hetero(km, mr):
                return True
            if p.kind == "benzo":
                return True
    return False


def _merged(km, rings, base: fc.Component, part: List[fc.Component]):
    """The second stage on a stage-1 reading of three or more components: every isolated
    benzene (a component of its own) merges with the senior benzo name it can form with an
    adjacent heteromonocycle of five or more members (:13443); the reading is kept only when
    two components are left. Identical components both fused to the parent are multiplied
    instead:13467,:13463 '6H-dibenzo[b,d]pyran')."""
    for x, y in combinations(part, 2):
        if x.name == y.name and _one_bond(rings, base, x) and _one_bond(rings, base, y):
            return None
    pieces = [base] + list(part)
    benzenes = [p for p in pieces if p.kind == "carbo" and len(p.atoms) == 6]
    if not benzenes:
        return None
    merged: Dict[int, Tuple[fc.Component, fc.Component]] = {}
    used = set()
    for b in benzenes:
        bi = next(iter(b.rings))
        units = []
        for h in pieces:
            if h.kind != "hetero" or len(h.atoms) < 5:
                continue
            hi = next(iter(h.rings))
            if len(rings[bi] & rings[hi]) != 2:
                continue
            u = _benzo_unit(km, rings, bi, hi)
            if u is not None:
                units.append((u, h))
        if not units:
            return None
        top = _senior(km, rings, [u for u, _ in units])
        if top is None or len(top) != 1:
            return None
        u, h = next((u, h) for u, h in units if u is top[0])
        if id(h) in used:
            return None
        used.add(id(h))
        merged[id(b)] = (u, h)
    gone = {id(b) for b in benzenes} | used
    left = [p for p in pieces if id(p) not in gone] + [u for u, _ in merged.values()]
    if len(left) != 2:
        return None
    pair = _senior(km, rings, left)
    if pair is None or len(pair) != 1:
        return None
    p = pair[0]
    a = left[1] if p is left[0] else left[0]
    if _one_bond(rings, p, a) is None:
        return None
    return p, a


def _readings(km, rings, comps) -> Optional[List[Tuple[fc.Component, fc.Component]]]:
    """Every (parent, attached) two-component reading the two stages allow, or None."""
    allr = frozenset(range(len(rings)))
    seniors = _senior(km, rings, list(comps.values()))
    if seniors is None:
        return None
    out = []
    for base in seniors:
        rest = allr - base.rings
        parts = _min_partitions(rings, comps, rest)
        for part in parts:
            if len(part) == 1:
                att = part[0]
                if _one_bond(rings, base, att) is None:
                    continue
                out.append((base, att))
            elif len(part) >= 2:
                got = _merged(km, rings, base, part)
                if got is not None:
                    out.append(got)
    out = [(b, a) for b, a in out if not _multi(km, rings, b, a)]
    return out or None


def _descriptor(km, rings, base: fc.Component, att: fc.Component):
    """(key, name) of the lowest descriptor over every numbering of the two components."""
    shared = _one_bond(rings, base, att)
    if shared is None:
        return None
    prefix = fc.prefix_form(att)
    if prefix is None:
        return None
    omit = att.kind == "carbo"
    best = None
    for bnum in base.numbering_dicts():
        sides = fc.side_letters(km, base, bnum, rings)
        if sides is None:
            continue
        got = sides.get(frozenset(shared))
        if got is None:
            continue
        letter, (u, v) = got
        if omit:
            key = (letter,)
            text = f"{prefix}[{letter}]{fc.base_form(base)}"
            if best is None or key < best[0]:
                best = (key, text)
            continue
        for anum in att.numbering_dicts():
            lu, lv = anum[u], anum[v]
            key = (letter, tuple(sorted((fc._loc_key(lu), fc._loc_key(lv)))),
                   (fc._loc_key(lu), fc._loc_key(lv)))
            text = f"{prefix}[{lu},{lv}-{letter}]{fc.base_form(base)}"
            if best is None or key < best[0]:
                best = (key, text)
    return best


def _lambda_free(km) -> bool:
    """Every heteroatom has its standard bonding number on the skeleton (the
    lambda-convention,:12452, is not built here)."""
    pt = Chem.GetPeriodicTable()
    for a in km.GetAtoms():
        if a.GetAtomicNum() != 6 and a.GetDegree() > pt.GetDefaultValence(a.GetAtomicNum()):
            return False
    return True


def rule_name(key_mol) -> Optional[str]:
    """The name the two stages give the skeleton ``key_mol`` (an element-aware
    all-single-bond graph, ``parents._key_of``), before the tables are asked: a benzo name
    for one benzene ring and one heteromonocycle, a two-component fusion name otherwise, or
    None outside the class of the module docstring."""
    if all(a.GetAtomicNum() == 6 for a in key_mol.GetAtoms()) or not _lambda_free(key_mol):
        return None
    rings = fc.ortho_rings(key_mol)
    if rings is None or not 2 <= len(rings) <= 6:
        return None
    # (:23710)
    if sum(1 for r in rings if len(r) >= 5) < 2:
        return None
    comps = _stage1_components(key_mol, rings)
    if len(comps) < len(rings) or frozenset(range(len(rings))) in comps:
        return None                       # a ring without a name, or one component (tables)
    if len(rings) == 2:
        for b, h in ((0, 1), (1, 0)):
            if _is_benzene(key_mol, rings[b]) and _is_hetero(key_mol, rings[h]) and len(rings[h]) >= 5:
                u = _benzo_unit(key_mol, rings, b, h)
                return u.name if u is not None else None
    readings = _readings(key_mol, rings, comps)
    if not readings:
        return None
    named = []
    for base, att in readings:
        d = _descriptor(key_mol, rings, base, att)
        if d is not None:
            named.append(d)
    if not named:
        return None
    best = min(k for k, _ in named)
    texts = {t for k, t in named if k == best}
    if len(texts) != 1:
        return None
    return texts.pop()


def hetero_component_name(key_mol) -> Optional[str]:
    """``rule_name`` for a skeleton no parent table names:11903 puts every
    retained, systematic or Blue Book name first)."""
    from ...data import get_retained_name
    from .parents import _IH, _mancude, _table_index, is_pin_parent_name
    if _table_index().get(Chem.MolToSmiles(key_mol)):
        return None
    std = _mancude(key_mol)
    retained = get_retained_name(Chem.MolToSmiles(std)) if std is not None else None
    if retained and is_pin_parent_name(_IH.sub("", retained)):
        return None
    return rule_name(key_mol)


@lru_cache(maxsize=512)
def opsin_structure(name: str, n_atoms: int) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """``fusion_names.opsin_structure`` with the indicated-hydrogen forms widened: a
    heterocyclic parent may need one indicated hydrogen at any position or two
    ('4H,5H-pyrano[4,3-d][1,2,3]dioxathiine',:21239). The first form OPSIN reads decides,
    as there."""
    from .fusion_names import _AV, opsin_structure as narrow
    got = narrow(name, n_atoms)
    if got is not None:
        return got
    from ...validation.opsin_roundtrip import extended_smiles_or_unavailable
    forms = [f"{i}H-" for i in range(3, n_atoms + 1)]
    forms += [f"{i}H,{j}H-" for i in range(1, min(n_atoms, 16) + 1) for j in range(i + 1, min(n_atoms, 16) + 1)]
    for ih in forms:
        line = extended_smiles_or_unavailable(f"{ih}{name}")
        m = _AV.match((line or "").strip())
        if not m:
            continue
        mol = Chem.MolFromSmiles(m.group(1))
        if mol is None:
            continue
        locants = tuple(m.group(2).split(";"))
        if mol.GetNumAtoms() == len(locants) == n_atoms:
            return m.group(1), locants
        return None
    return None
