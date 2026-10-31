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

Declined: a hydrocarbon (``fusion_names``), peri-fusion and interior atoms, second-order attached
components, a tie the rules above leave open between names that differ, and any name OPSIN 2.9.0
cannot read back to the same skeleton (the caller, ``parents._parent_for_key``).
"""
import re
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
