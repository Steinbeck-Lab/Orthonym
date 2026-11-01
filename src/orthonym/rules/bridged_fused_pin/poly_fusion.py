"""Polycomponent fusion names whose attached components are all first-order (LO4c), for the
ortho-fused parents no table holds, numbered by OPSIN's reading of the name.

``hetero_fusion`` builds a name of two components: the parent and ONE attached component (a benzo
unit merged beforehand). A skeleton of three or more components fused to one parent component,
each attached directly to it, has a name the Blue Book gives but that module declined:
"Constructing polycomponent fusion names" (the Blue Book): "When several nonidentical
components must be considered, one, and only one, can be the parent component. All other
components are attached components. A component attached directly to the parent component is
called a 'first-order attached component'". This module builds the names of that class, and only
that class; a component that is not fused to the parent (a second-order one) declines.

1. Parent component. (:12956) "When there are two or more locations for a parent
   component in a fused ring structure, the following criteria are applied sequentially":
   (b) (:12970) "the location that results in a name that does not require attached components
   higher than first-order" (only such readings are built), (c) (:12976) "the location that
   results in the maximum number of first-order attached components", (d) (:12984) "a location
   that permits the expression of the maximum number of identical attached components with
   multiplicative prefixes". The parent is the senior component of the whole skeleton by
    (``hetero_fusion._senior``).
2. Attached components. Identical ones fused to the parent are multiplied: (:12909)
   "A multiplicity of components that are identical and are all fused to a parent component or an
   attached component is indicated by the use of the prefix 'di', 'tri', etc.... The multiplying
   prefix is not considered in determining the alphabetical order of attached components... A
   colon separates the sets of locants, and a comma is used when letters only are present"
   ('dibenzo[c,g]phenanthrene', 'dipyrano[2,3-b:3',4',5'-de]quinoline'). A single benzene ring
   fused to a heteromonocycle of five or more members is one component with it, a benzo name
    :13437, ``hetero_fusion._merged``), which a reading with one benzo prefix would
   disrupt: declined. Two or more benzo prefixes stay, (:13437) "this approach is not
   used if it disrupts... the use of multiplicative prefixes (see ". A ring of the
   parent's own size and elements elsewhere in the skeleton may make a multiparent name
   ,:13530;:13451 "A multiparent name is preferred to a fused ring system,
   when there is a choice"): declined (``_multiparent_risk``, ``hetero_fusion._multi``).
3. Citation. (:13040) "All attached components are cited in front of the parent
   component... If there are two or more different components... attached to a lower order
   component, they are cited in alphabetical order" ('furo[3,2-b]thieno[2,3-e]pyridine (PIN)'
   :13044); (:13070) "If two or more groups of components only differ by virtue of
   heteroatom locants, these locants are used to determine the order; lower locant sets are cited
   earlier" ('[1,2]oxazolo[5,4-c][1,3]oxazolo[3,2-a]pyridine (PIN)':13074).
4. Locants. The parent's peripheral sides are lettered:11911) and the attached
   component's attachment locants are "as low as is consistent with the numbering of the compound
   and their order conforms to the direction of lettering of the parent component". When the
   numbering of the parent offers a choice, (:13076) "the lower letters or numbers
   are selected in accordance with the following criteria,... in order": (a) (:13080) "parent
   component letters as a set" ('furo[3,2-h]pyrrolo[3,4-a]carbazole (PIN)', not 'furo[2,3-b]
   pyrrolo[3,4-i]carbazole'; a,h is lower than b,i), (b) (:13088) "Parent component letters in
   order of citation" ('furo[3,4-b]thieno[2,3-e]pyrazine (PIN)', not 'furo[3,4-e]thieno[2,3-b]
   pyrazine'), (c) (:13102) "Set of locants of the first-order attached components for fusion to
   the parent component", (d) (:13403) "First-order attached component locants for fusion to the
   parent component in order of citation".

Second-order attached components (DT4-A, ``_tree_name``): when no reading with first-order components
only exists, a component fused to a first-order attached component is second-order.
(:12718) "numerical locants are used instead of letters and the two sets of locants are separated by a
colon. The locants of second-order attached components are primed"; (:13040) "Each
second-order attached component is cited in front of the first-order attached component to which it is
fused"; (e)-(h) (:13417-:13433) order the locants of the lower-order components for fusion to
higher-order ones, then those of the higher-order ones. See ``_tree_name`` for the class and its limits.

Declined: a hydrocarbon (``fusion_names``), peri-fusion and interior atoms, third-order and higher
attached components, a tie the rules above leave open between names that differ, and any name OPSIN 2.9.0
cannot read back to the same skeleton (the caller, ``parents._parent_for_key``).
"""
import re
from itertools import product
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from rdkit import Chem

from . import fusion_components as fc
from . import hetero_fusion as hf

#: /: the skeleton has three to this many rings (``hetero_fusion.rule_name``'s bound)
MAX_RINGS = 6
_MULT = fc._MULT


def _stem(prefix: str) -> str:
    """The prefix without its bracketed heteroatom locants: alphabetical order ignores them."""
    return re.sub(r"\[[^\]]*\]", "", prefix)


def _bracket_locants(prefix: str) -> Tuple[int, ...]:
    m = re.match(r"\[([\d,]+)\]", prefix)
    return tuple(int(x) for x in m.group(1).split(",")) if m else ()


def _first_order_only(rings, base: fc.Component, parts: Sequence[fc.Component]) -> bool:
    """Every part is fused to the parent through one shared bond and no two parts share an atom
    (an attached component fused to another attached component would be second-order)."""
    if any(hf._one_bond(rings, base, p) is None for p in parts):
        return False
    return all(not (x.atoms & y.atoms) for i, x in enumerate(parts) for y in parts[i + 1:])


def _benzo_policy_ok(km, base: fc.Component, parts: Sequence[fc.Component]) -> bool:
    """ (:13437): one benzene part on a heteromonocycle of five or more members is a benzo
    name, built by ``hetero_fusion``; two or more benzene parts are multiplied ('dibenzo')."""
    n_benzo = sum(1 for p in parts if p.kind == "carbo" and len(p.atoms) == 6)
    return not (n_benzo == 1 and base.kind == "hetero" and len(base.atoms) >= 5)


def _ring_signature(km, ring) -> Tuple[int, Tuple[int, ...]]:
    return len(ring), tuple(sorted(km.GetAtomWithIdx(a).GetAtomicNum() for a in ring))


def _multiparent_risk(km, rings, base: fc.Component) -> bool:
    """A ring outside the parent with the parent's size and elements:13530)."""
    if len(base.rings) != 1:
        return False
    sig = _ring_signature(km, rings[next(iter(base.rings))])
    return any(i not in base.rings and _ring_signature(km, r) == sig for i, r in enumerate(rings))


def _readings(km, rings, comps) -> List[Tuple[fc.Component, Tuple[fc.Component, ...]]]:
    """The (parent, parts) readings with first-order attached components only, after
    (b), (c), (d)."""
    allr = frozenset(range(len(rings)))
    seniors = hf._senior(km, rings, list(comps.values()))
    if seniors is None:
        return []
    out = []
    for base in seniors:
        rest = allr - base.rings
        if not rest or _multiparent_risk(km, rings, base):
            continue
        for part in hf._min_partitions(rings, comps, rest):
            if (_first_order_only(rings, base, part) and _benzo_policy_ok(km, base, part)
                    and not any(hf._multi(km, rings, base, p) for p in part)):
                out.append((base, tuple(part)))
    if not out:
        return []

    def identical(parts):
        names = [(p.kind, p.name) for p in parts]
        return sum(names.count(n) for n in set(names) if names.count(n) > 1)

    top_c = max(len(p) for _, p in out)
    out = [r for r in out if len(r[1]) == top_c]
    top_d = max(identical(p) for _, p in out)
    return [r for r in out if identical(r[1]) == top_d]


def _descriptor(km, rings, base: fc.Component, parts: Sequence[fc.Component]):
    """(key, text) of the lowest descriptor of one reading over every numbering of the parent and
    of each attached component, or None."""
    groups: Dict[Tuple[str, str], List[fc.Component]] = {}
    for p in parts:
        groups.setdefault((p.kind, p.name), []).append(p)
    prefix_of = {}
    for key, members in groups.items():
        pf = fc.prefix_form(members[0])
        if pf is None or (len(members) > 1 and pf.startswith("[")):
            return None                    # no 'bis' / bracketed multiples here
        prefix_of[key] = pf
    order = sorted(groups, key=lambda k: (_stem(prefix_of[k]), _bracket_locants(prefix_of[k])))
    best = None
    for bnum in base.numbering_dicts():
        sides = fc.side_letters(km, base, bnum, rings)
        if sides is None:
            continue
        per_group = []
        for key in order:
            members = []
            for p in groups[key]:
                shared = hf._one_bond(rings, base, p)
                got = sides.get(frozenset(shared))
                if got is None:
                    members = None
                    break
                letter, (u, v) = got
                if p.kind == "carbo":
                    members.append((letter, None))
                    continue
                opts = []
                for anum in p.numbering_dicts():
                    lu, lv = anum[u], anum[v]
                    opts.append(((tuple(sorted((fc._loc_key(lu), fc._loc_key(lv)))),
                                  (fc._loc_key(lu), fc._loc_key(lv))), (lu, lv)))
                members.append((letter, min(opts)[1]))
            if members is None:
                per_group = None
                break
            per_group.append(sorted(members, key=lambda m: m[0]))
        if per_group is None:
            continue
        letters_set = tuple(sorted(m[0] for g in per_group for m in g))
        letters_cited = tuple(m[0] for g in per_group for m in g)
        locs = [fc._loc_key(x) for g in per_group for m in g if m[1] for x in m[1]]
        locs_cited = tuple(fc._loc_key(x) for g in per_group for m in g if m[1] for x in m[1])
        # (a) letters as a set, (b) in order of citation, (c) locants as a set, (d) in order
        key = (letters_set, letters_cited, tuple(sorted(locs)), locs_cited)
        pieces = []
        for gkey, members in zip(order, per_group):
            pf = prefix_of[gkey]
            if len(members) == 1:
                letter, lc = members[0]
                pieces.append(f"{pf}[{letter}]" if lc is None else f"{pf}[{lc[0]},{lc[1]}-{letter}]")
                continue
            mult = _MULT.get(len(members))
            if mult is None:
                return None
            if members[0][1] is None:
                pieces.append(f"{mult}{pf}[{','.join(m[0] for m in members)}]")
            else:
                # (:13467): "the locants of the second component are primed..., the third
                # double primed, and so on": member i takes i primes
                sets = [f"{lc[0] + chr(39) * i},{lc[1] + chr(39) * i}-{letter}"
                        for i, (letter, lc) in enumerate(members)]
                pieces.append(f"{mult}{pf}[{':'.join(sets)}]")
        text = "".join(pieces) + fc.base_form(base)
        if best is None or key < best[0]:
            best = (key, {text})
        elif key == best[0]:
            best[1].add(text)
    if best is None or len(best[1]) != 1:
        return None
    return best[0], next(iter(best[1]))


# --------------------------------------------------------------------------------------
# second-order attached components (DT4-A)
# --------------------------------------------------------------------------------------

#: the product of the numbering choices of the components of one reading is searched exhaustively up to this size
_MAX_COMBOS = 20000
_PRIME = "'"   # the engine's spelling of the prime (as the multiplied prefixes above)


def _numberings(km, rings, comp: fc.Component, leaf: bool) -> List[Optional[Dict[int, str]]]:
    """Every numbering of ``comp`` for the descriptor. A carbocyclic leaf cites no locants of its own
     :13970: the numbers are omitted for 'benzo' and the monocyclic hydrocarbon prefixes), so
    it has none (``None``); a carbocycle that carries a second-order component takes every numbering of
    its ring, the lowest locants being chosen by."""
    if comp.kind != "carbo":
        return list(comp.numbering_dicts())
    if leaf:
        return [None]
    cyc = fc.ring_cycle(km, rings[next(iter(comp.rings))])
    if cyc is None:
        return []
    n = len(cyc)
    return [{a: str(k + 1) for k, a in enumerate([cyc[(s + d * k) % n] for k in range(n)])}
            for s in range(n) for d in (1, -1)]


def _tree_of(rings, base: fc.Component, part: Sequence[fc.Component]):
    """{component: its parent component} of the reading, or None when the components are not fused as a
    tree of single shared bonds (a cycle of components, a peri-fusion)."""
    pieces = [base] + list(part)
    nbrs = {id(p): [] for p in pieces}
    edges = 0
    for i, x in enumerate(pieces):
        for y in pieces[i + 1:]:
            if not (x.atoms & y.atoms):
                continue
            if hf._one_bond(rings, x, y) is None:
                return None
            nbrs[id(x)].append(y)
            nbrs[id(y)].append(x)
            edges += 1
    if edges != len(pieces) - 1:
        return None
    parent = {id(base): None}
    todo = [base]
    while todo:
        c = todo.pop(0)
        for y in nbrs[id(c)]:
            if id(y) not in parent:
                parent[id(y)] = c
                todo.append(y)
    return parent if len(parent) == len(pieces) else None


def _benzo_unit_risk(km, rings, base: fc.Component, pieces: Sequence[fc.Component]) -> bool:
    """ (:13437): a benzene component fused to a heteromonocycle component of five or more members
    is one component with it (a benzo name), which this module does not build. The one pair that stays
    separate is two or more benzene components on the parent ('dibenzo',:13467)."""
    benz = [p for p in pieces if p.kind == "carbo" and len(p.atoms) == 6]
    het = [p for p in pieces if p.kind == "hetero" and len(p.atoms) >= 5]
    for b in benz:
        near = [h for h in het if b.atoms & h.atoms]
        if not near:
            continue
        if len(near) == 1 and near[0] is base and sum(1 for x in benz if x.atoms & base.atoms) >= 2:
            continue
        return True
    return False


def _tree_readings(km, rings, comps):
    """The (parent, {id: parent component}, parts) readings with second-order attached components, or
    None when the class does not apply (a first-order-only reading exists, whether or not the
    multiparent and benzo policies keep it: (b):12970 "the location that results in a name
    that does not require attached components higher than first-order")."""
    allr = frozenset(range(len(rings)))
    # a three- or four-membered ring (an epoxide, a cyclopropane) is not built here: the engine names a
    # saturated oxirane on a ring system by its bridge prefix or the von Baeyer name, and those names stand
    if any(len(r) <= 4 for r in rings):
        return None
    seniors = hf._senior(km, rings, list(comps.values()))
    if not seniors:
        return None
    found = []
    for base in seniors:
        rest = allr - base.rings
        if not rest or _multiparent_risk(km, rings, base):
            return None                       # multiparent names are not built
        for part in hf._min_partitions(rings, comps, rest):
            if _first_order_only(rings, base, part):
                return None
            tree = _tree_of(rings, base, part)
            if tree is None:
                continue
            pieces = [base] + list(part)
            if _benzo_unit_risk(km, rings, base, pieces):
                return None
            if any(hf._multi(km, rings, p, q) for i, p in enumerate(pieces) for q in pieces[i + 1:]
                   if p.atoms & q.atoms):
                return None
            found.append((base, tree, list(part)))
    return found or None


def _levels(base: fc.Component, tree, part):
    """[first-order components, second-order components,...] of a reading."""
    depth = {id(base): 0}

    def d(c):
        if id(c) not in depth:
            depth[id(c)] = d(tree[id(c)]) + 1
        return depth[id(c)]
    out: List[List[fc.Component]] = []
    for c in part:
        k = d(c)
        while len(out) < k:
            out.append([])
        out[k - 1].append(c)
    return out


def _choose_reading(km, rings, found):
    """ (c) (:12976) the maximum number of first-order, then second-order,...
    attached components; (a) the senior component of each order. None when a tie is left
    between readings that the remaining text comparison cannot be trusted to settle."""
    lev = [(r, _levels(r[0], r[1], r[2])) for r in found]
    top = max(tuple(len(x) for x in lv) for _, lv in lev)
    lev = [(r, lv) for r, lv in lev if tuple(len(x) for x in lv) == top]
    for k in range(len(top)):
        if len({tuple(sorted(c.name for c in lv[k])) for _, lv in lev}) == 1:
            continue
        if top[k] != 1:
            return None
        sen = hf._senior(km, rings, [lv[k][0] for _, lv in lev])
        if not sen:
            return None
        lev = [(r, lv) for r, lv in lev if any(lv[k][0] is s for s in sen)]
    return [r for r, _ in lev]


def _stem_text(text: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", text)


def _prime(loc: str, n: int) -> str:
    return loc + _PRIME * n


#: the deepest order built; the key of (e)-(h) is the book's for two orders and read the same way for more
MAX_ORDER = 3   # the book's one third-order example (:13433) is rebuilt exactly; none of fourth order is printed


def _chain_options(km, rings, chain: Sequence[fc.Component], prefixes: Sequence[str], letter: str, uv):
    """Every numbering of the components of one chain (a first-order component and the single component
    fused to each, outwards), as (parent locants of the first, [locants of each component for fusion to the
    next], [own locants of each higher-order component], text). Text: the deepest component is cited
    first, (:13040); the locants of the order-n component are primed n-1 times, those of the
    lower-order component it is fused to n-2 times, (:12718); a carbocyclic leaf cites only the
    lower-order locants:13970), 'cyclopenta[4,5]oxepino'."""
    out = []
    k = len(chain)

    def rec(i, nums, sides_uv):
        # nums: numberings chosen so far; sides_uv[i] = (u, v) of the bond of component i fused to the next
        if i == k:
            first = nums[0]
            f_par = None if first is None else (first[uv[0]], first[uv[1]])
            # the direction of the pair of locants of a higher-order bond is free: (f) (:13423)
            # 'pyrrolo[3',2':4,5]cyclohepta[1,2-b]quinoline (PIN) (not pyrrolo[2',3':5,4]...; the locant set
            # '4,5' is lower than '5,4')'; both directions are offered and the key chooses
            for flips in product((False, True), repeat=k - 1):
                toward, own, texts = [], [], []
                for j in range(1, k):
                    p, q = sides_uv[j - 1]
                    if flips[j - 1]:
                        p, q = q, p
                    lo = (nums[j - 1][p], nums[j - 1][q])
                    toward.append(lo)
                    plo = (_prime(lo[0], j - 1), _prime(lo[1], j - 1))
                    if nums[j] is None:
                        texts.append(f"{prefixes[j]}[{plo[0]},{plo[1]}]")
                        own.append(None)
                    else:
                        g = (nums[j][p], nums[j][q])
                        own.append(g)
                        texts.append(f"{prefixes[j]}[{_prime(g[0], j)},{_prime(g[1], j)}:{plo[0]},{plo[1]}]")
                if k == 1:
                    head = f"{prefixes[0]}[{letter}]" if first is None else \
                        f"{prefixes[0]}[{f_par[0]},{f_par[1]}-{letter}]"
                else:
                    head = f"{prefixes[0]}[{f_par[0]},{f_par[1]}-{letter}]"
                out.append((f_par, toward, own, "".join(reversed(texts)) + head))
            return
        leaf = i == k - 1
        for num in _numberings(km, rings, chain[i], leaf):
            nxt = list(sides_uv)
            if not leaf:
                sf = fc.side_letters(km, chain[i], num, rings)
                got = sf.get(frozenset(hf._one_bond(rings, chain[i], chain[i + 1]))) if sf else None
                if got is None:
                    continue
                nxt.append(got[1])
            rec(i + 1, nums + [num], nxt)

    rec(0, [], [])
    return out


def _tree_descriptor(km, rings, base: fc.Component, tree, part):
    """(key, name) of one tree reading (first- and higher-order attached components), the lowest
    descriptor by (a)-(h) over every numbering, or None. Shapes built: every first-order
    component distinct, each carrying a chain of single higher-order components (no carbocycle with a
    component of order two or more fused to it)."""
    firsts = [c for c in part if tree[id(c)] is base]
    if len({(c.kind, c.name) for c in firsts}) != len(firsts) or not firsts:
        return None
    chains = []
    for f in firsts:
        chain = [f]
        while True:
            nxt = [c for c in part if tree[id(c)] is chain[-1]]
            if not nxt:
                break
            if len(nxt) > 1:
                return None
            chain.append(nxt[0])
        if len(chain) > MAX_ORDER or any(c.kind == "carbo" for c in chain[1:-1]):
            return None
        prefixes = [fc.prefix_form(c) for c in chain]
        if None in prefixes:
            return None
        chains.append((_stem_text("".join(reversed(prefixes))), chain, prefixes))
    chains.sort(key=lambda t: t[0])
    if len({t[0] for t in chains}) != len(chains):
        return None                           # /.4: told apart by locants, not built
    combos = len(base.numbering_dicts())
    for _, chain, _p in chains:
        for i, c in enumerate(chain):
            combos *= max(1, len(_numberings(km, rings, c, i == len(chain) - 1)))
    if combos > _MAX_COMBOS:
        return None
    best = None
    for bnum in base.numbering_dicts():
        sides = fc.side_letters(km, base, bnum, rings)
        if sides is None:
            continue
        per_set = []
        for _, chain, prefixes in chains:
            got = sides.get(frozenset(hf._one_bond(rings, base, chain[0])))
            if got is None:
                per_set = None
                break
            letter, uv = got
            opts = [(letter,) + o for o in _chain_options(km, rings, chain, prefixes, letter, uv)]
            if not opts:
                per_set = None
                break
            per_set.append(opts)
        if per_set is None:
            continue

        def flat(items):
            return tuple(fc._loc_key(x) for it in items if it for x in it)
        for pick in product(*per_set):
            letters = tuple(o[0] for o in pick)
            c_list = [o[1] for o in pick]
            # (a):13080 letters as a set, (b):13088 in order of citation, (c):13102 locants of the
            # first-order components for fusion to the parent as a set, (d):13403 in order; then order by order
            # outwards, for the fusion of order n to order n+1: (e):13417 locants of the lower-order components as
            # a set, (f):13423 in order of citation, (g):13427 locants of the higher-order components as a set,
            # (h):13433 in order ('pyrano[2'',3'':6',7']thiepino[4',5':4,5]furo[3,2-c]pyrazole (PIN) (not
            # pyrano[3'',2'':2',3']thiepino[5',4':4,5]...; the locant set '4',5'' is lower than '5',4'')': the
            # first fusion decides before the second is looked at)
            key = [tuple(sorted(letters)), letters, tuple(sorted(flat(c_list))), flat(c_list)]
            for n in range(max(len(o[2]) for o in pick)):
                lows = [o[2][n] if len(o[2]) > n else None for o in pick]
                highs = [o[3][n] if len(o[3]) > n else None for o in pick]
                key += [tuple(sorted(flat(lows))), flat(lows), tuple(sorted(flat(highs))), flat(highs)]
            key = tuple(key)
            text = "".join(o[4] for o in pick) + fc.base_form(base)
            if best is None or key < best[0]:
                best = (key, {text})
            elif key == best[0]:
                best[1].add(text)
    if best is None or len(best[1]) != 1:
        return None
    return best[0], next(iter(best[1]))


def _tree_name(key_mol, rings, comps) -> Optional[str]:
    """The name of a skeleton whose attached components are not all first-order (DT4-A), or None.

    Built: one parent, first-order attached components that are all different, and at most
    a chain of higher-order attached components on each; named ``X[g,g':c,c]Y[c,c-b]parent``:12718),
    the sets cited alphabetically:13040). Declined: any skeleton that has a first-order-only
    reading (``_tree_readings``), multiparent names, benzo units, identical first-order
    components, third order, and a tie the rules leave open."""
    found = _tree_readings(key_mol, rings, comps)
    if not found:
        return None
    chosen = _choose_reading(key_mol, rings, found)
    if not chosen:
        return None
    # readings left are the same components at other locations of the parent; (b), (c) (:13003,
    #:13020 'naphtho[2',1':3,4]phenanthro[1,2-b]thiophene (PIN) (not... phenanthro[2,1-b]thiophene...)') "the
    # location that has lowest locants... for fusion to the parent component", as the key of the descriptor
    named = []
    for base, tree, part in chosen:
        d = _tree_descriptor(key_mol, rings, base, tree, part)
        if d is None:
            return None
        named.append(d)
    low = min(k for k, _ in named)
    texts = {t for k, t in named if k == low}
    if len(texts) != 1:
        return None
    text = texts.pop()
    # an attached component fused at a bond of its own fusion atom ('indeno[1,7a-c]') is a peri-fusion descriptor the
    # book prints no example of at higher orders; declined until one is derived
    return None if _FUSION_ATOM_LOCANT.search(text) else text


#: a locant of a component's own fusion atom ('7a', '4b') inside a fusion descriptor
_FUSION_ATOM_LOCANT = re.compile(r"\[[^\]]*\d[a-z]")

def rule_name(key_mol, carbocyclic: bool = False) -> Optional[str]:
    """The name this module's rules give the skeleton ``key_mol`` (an element-aware all-single-bond
    graph, ``parents._key_of``), before the tables are asked, or None outside the class.

    ``carbocyclic``: the other half of the class, a skeleton of carbon atoms only. Only the shape
    ``fusion_names.retained_parent_name`` documents is built (a retained polycyclic hydrocarbon parent
    and monocyclic hydrocarbon attached components, every one fused to the parent); the heterocyclic
    skeletons are left to the call without it, and the other carbocycles to ``fusion_names``. The parent
    is the senior component of the whole skeleton by (b):12163, (c):12234, (g):12317 "the
    greatest number of rings in a horizontal row" and (j):12418 (``hetero_fusion._senior``). A skeleton
    with a tetravalent fusion atom (the angular triquinane 'cyclopenta[c]pentalene') is declined: its
    numbering (OPSIN's, checked only for a periphery walk, ``fc.periphery_numbering_ok``) is not
    derived by here, and the von Baeyer name stands."""
    if all(a.GetAtomicNum() == 6 for a in key_mol.GetAtoms()) != carbocyclic or not hf._lambda_free(key_mol):
        return None
    rings = fc.ortho_rings(key_mol)
    if rings is None or not 3 <= len(rings) <= MAX_RINGS:
        return None
    if carbocyclic and fc.tetravalent_fusion_atoms(key_mol, rings):
        return None
    # (:23710)
    if sum(1 for r in rings if len(r) >= 5) < 2:
        return None
    comps = hf._stage1_components(key_mol, rings)
    if len(comps) < len(rings) or frozenset(range(len(rings))) in comps:
        return None
    texts = set()
    readings = _readings(key_mol, rings, comps)
    if not readings and not carbocyclic:
        return _tree_name(key_mol, rings, comps)
    for base, parts in readings:
        if carbocyclic and not (base.kind == "retained" and all(p.kind == "carbo" for p in parts)):
            return None
        d = _descriptor(key_mol, rings, base, parts)
        if d is None:
            return None
        texts.add(d[1])
    # readings at different locations of the parent are told apart by (:13000), which
    # is not applied: they must all give one name
    return texts.pop() if len(texts) == 1 else None


def polycomponent_name(key_mol) -> Optional[str]:
    """``rule_name`` for a skeleton no parent table names:11903 puts every retained,
    systematic or Blue Book name first), as ``hetero_fusion.hetero_component_name``."""
    from ...data import get_retained_name
    from .parents import _IH, _mancude, _table_index, is_pin_parent_name
    if _table_index().get(Chem.MolToSmiles(key_mol)):
        return None
    std = _mancude(key_mol)
    retained = get_retained_name(Chem.MolToSmiles(std)) if std is not None else None
    if retained and is_pin_parent_name(_IH.sub("", retained)):
        return None
    return rule_name(key_mol)
