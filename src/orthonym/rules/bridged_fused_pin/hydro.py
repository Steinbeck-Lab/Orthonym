"""Hydrogenation state of a bridged fused ring system: mancude parent, indicated hydrogen,
hydro prefixes.

 (the Blue Book): "The maximum number of noncumulative double bonds is
assigned to the fused ring system after the insertion of the bridge." (:14583):
"In bridged fused systems, the distribution of noncumulative double bonds in the parent
fused ring system is accomplished after allowance has been made for the bonds existing
between the bridge(s) and the fused ring system." So the mancude parent is a maximum
matching over the parent atoms that keep a free valence after the bridge bonds (a fusion
atom that carries a bridge has four ring bonds and no double bond, as 4a and 8a of
4a,8a-ethanonaphthalene); the atoms the matching leaves out carry indicated hydrogen,
"cited in front of the completed bridged fused ring name" (:14245;:14642
"All indicated hydrogen atoms are indicated at the front of the complete name").
 (:16880): hydro prefixes come in pairs, "Indicated hydrogen atoms have priority
over 'hydro' prefixes for low locants. If indicated hydrogen atoms are present in a name,
the 'hydro' prefixes precede them." (:17026): "Total hydrogenation is
indicated by the appropriate multiplicative prefixes indicating the total number of
hydrogen atoms attached, but locants are omitted".

Suffixes that need hydrogen the mancude parent does not have (slice S3, ``accommodate``):
 (:24766) "Indicated hydrogen is cited at any position of a ring system in order
to accommodate principal characteristic groups or free valences expressed as suffixes,
provided that there are an equal or greater number of indicated hydrogen atoms available";
 (:24768), (:24794), (:24806) (1)-(4); the groups
left over take 'added indicated hydrogen':24693, lowest locants:24695), except
a pair that "simply removes a double bond (directly or after rearrangement of double
bonds)":24721).
"""
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from rdkit import Chem

from ...perception.mancude import max_matching_size
from .numbering import loc_key, locant_tuple
from .selection import Split

#: at most this many indicated-hydrogen positions (three: '5,6-dihydro-1H,3H,4H-3a,6a-
#: methanocyclopenta[c]furan-1,3-dione (PIN)', the Blue Book)
MAX_INDICATED_H = 3
#: (the Blue Book) hydro prefixes come in pairs; the multiplying prefixes
#: are those of (di, tetra,..., icosa, docosa,..., triaconta)
_HYDRO = {2: "dihydro", 4: "tetrahydro", 6: "hexahydro", 8: "octahydro", 10: "decahydro",
          12: "dodecahydro", 14: "tetradecahydro", 16: "hexadecahydro", 18: "octadecahydro",
          20: "icosahydro", 22: "docosahydro", 24: "tetracosahydro", 26: "hexacosahydro",
          28: "octacosahydro", 30: "triacontahydro"}


@dataclass(frozen=True)
class HydroState:
    eligible: FrozenSet[int]          # parent atoms that can carry a ring double bond
    saturated: FrozenSet[int]         # eligible atoms without a ring double bond in the input
    n_double: int                     # ring double bonds of the parent in the input
    mancude_double: int               #... in the mancude bridged parent
    ih_sets: Tuple[FrozenSet[int], ...]   # the indicated-hydrogen sets the parent allows
    # bonds between eligible atoms (the mancude parent's possible double bonds)
    adj: Dict[int, FrozenSet[int]] = field(default_factory=dict, compare=False, hash=False)
    # free valences left on each eligible atom by its ring-system bonds (valence - bonds)
    free: Dict[int, int] = field(default_factory=dict, compare=False, hash=False)


def hydro_state(mol, split: Split) -> Optional[HydroState]:
    residual = set(split.residual)
    system = residual | {a for chain in split.bridges for a in chain}
    pt = Chem.GetPeriodicTable()
    eligible = set()
    free: Dict[int, int] = {}
    for a in residual:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None
        n_sys = sum(1 for nb in atom.GetNeighbors() if nb.GetIdx() in system)
        if pt.GetDefaultValence(atom.GetAtomicNum()) - n_sys >= 1:
            eligible.add(a)
            free[a] = pt.GetDefaultValence(atom.GetAtomicNum()) - n_sys
    if not (residual & split.unsaturated) <= eligible:
        return None
    adj: Dict[int, set] = {a: set() for a in eligible}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in eligible and j in eligible:
            adj[i].add(j)
            adj[j].add(i)
    saturated = eligible - split.unsaturated
    mancude = max_matching_size(eligible, adj)
    n_ih = len(eligible) - 2 * mancude
    if n_ih > MAX_INDICATED_H or n_ih > len(saturated) or (len(saturated) - n_ih) % 2:
        return None
    ih_sets = []
    for combo in combinations(sorted(saturated), n_ih):
        rest = eligible - set(combo)
        sub = {a: adj[a] & rest for a in rest}
        if 2 * max_matching_size(rest, sub) == len(rest):
            ih_sets.append(frozenset(combo))
    if not ih_sets:
        return None
    return HydroState(frozenset(eligible), frozenset(saturated), split.n_double, mancude,
                      tuple(ih_sets), {a: frozenset(v) for a, v in adj.items()}, free)


def choose_indicated_hydrogen(state: HydroState, atom_to_locant: Dict[int, Any]):
    """(indicated-hydrogen atoms, hydro atoms) under this numbering.
    (the Blue Book): indicated hydrogen is cited "if posible, at the lowest
    nonfusion peripheral atom", also when hydro prefixes are present (:24685
    '3a,5-dihydro-4H-indene (PIN) (not 4,5-dihydro-3aH-indene)'); among the allowed
    sets, the fewest fusion atoms, then the lowest locants:16880)."""
    def n_fusion(atoms):
        return sum(1 for a in atoms if not isinstance(atom_to_locant[a], int))
    ih = min(state.ih_sets,
             key=lambda s: (n_fusion(s), locant_tuple(atom_to_locant[a] for a in s)))
    return ih, state.saturated - ih


#: at most this many 'added indicated hydrogen' atoms (the book's largest is three,
#: '...-1,5,11,28,29(4H,6H,31H)-pentone (PIN)', the Blue Book)
MAX_ADDED_H = 4


def lacks_hydrogen(state: HydroState, atom: int, n_h: int) -> bool:
    """True when parent atom ``atom`` has fewer than ``n_h`` hydrogen atoms in the mancude
    parent unless it carries indicated hydrogen: an eligible atom keeps one double bond
    there, so it has (free valence - 1) hydrogen atoms. (:28386-:28388): a ketone
    needs a >CH2 group, "Compounds not having suitably located indicated hydrogen atoms or
    composed only of =CH- groups, must be hydrogenated"; a monovalent suffix or free valence
    needs one hydrogen atom (b):3246 '1H-phenalen-4-ol (PIN)': none is needed on a
    =CH- atom;:24717 '1,3,4,5-tetrahydronaphthalene-4a(2H)-carboxylic acid
    (PIN)': a fusion atom has none). A bridge atom is not a parent atom (a saturated bridge
    carbon is a >CH2 group, slice S1.8b)."""
    return atom in state.eligible and state.free[atom] - 1 < n_h


def _added_hydrogen(state: HydroState, ih: FrozenSet[int], groups: FrozenSet[int],
                    atom_to_locant: Dict[int, Any]) -> Optional[FrozenSet[int]]:
    """ (:24693): 'added indicated hydrogen' for the groups no indicated hydrogen
    accommodates; (:24695) "with the lowest locants consistent with the
    arrangement of double bonds in the compound"; (:24721) none for a pair of
    groups that "simply removes a double bond (directly or after rearrangement of double
    bonds)". So: the fewest atoms of the compound's saturated parent atoms, lowest locants,
    whose removal with the groups leaves the rest of the parent able to carry its double
    bonds (a perfect matching). None when more than MAX_ADDED_H would be needed."""
    base = state.eligible - ih - groups
    pool = sorted(state.saturated - ih - groups, key=lambda a: loc_key(atom_to_locant[a]))
    for k in range(len(groups) % 2, min(len(pool), MAX_ADDED_H) + 1, 2):
        found = [frozenset(c) for c in combinations(pool, k)
                 if 2 * max_matching_size(base - set(c), state.adj) == len(base) - k]
        if found:
            return min(found, key=lambda c: locant_tuple(atom_to_locant[a] for a in c))
    return None


def accommodate(state: HydroState, atom_to_locant: Dict[int, Any], need: FrozenSet[int]):
    """(indicated-hydrogen atoms, added-hydrogen atoms, hydro atoms) under this numbering,
    for suffix atoms ``need`` that lack hydrogen in the mancude parent (``lacks_hydrogen``),
    or None. (:24768): with enough indicated hydrogen atoms they "are placed at
    peripheral atoms that will accommodate these principal characteristic groups" ('1,2,3,7,
    8,8a-hexahydro-4H-3a,7-methanoazulene-4,9-dione (PIN)':24792); (:24794)
    the others go "to the lowest nonfusion peripheral atom" ('5,6-dihydro-1H,3H,4H-3a,6a-
    methanocyclopenta[c]furan-1,3-dione (PIN)':24800). When they cannot accommodate every
    group, (:24806) and (:24841): "(1) at least one of the indicated
    hydrogen atoms is assigned to a nonfusion peripheral atom having the lowest locants
    consistent with the mancude system", "(2) other indicated hydrogen atoms are assigned to
    other positions that accommodate characteristic groups", "(3)... 'added indicated
    hydrogen atoms'", "(4) indicated hydrogen atoms that cannot be used to accommodate...
    has seniority over 'added indicated hydrogen' for lower locants" ('1H-cyclopenta[a]-
    naphthalene-1,2(3H)-dione (PIN) (not 3H-cyclopenta[a]naphthalene-1,2-dione)':24822,
    'hexahydro-1H-isoindole-1,3(2H)-dione (PIN)':33849)."""
    if not need:
        ih, rest = choose_indicated_hydrogen(state, atom_to_locant)
        return ih, frozenset(), rest
    if not need <= state.saturated:
        return None

    def fusion(a):
        return not isinstance(atom_to_locant[a], int)

    sets = state.ih_sets
    cands = [s for s in sets if need <= s]
    if not cands and sets and sets[0]:
        low = min((fusion(a), loc_key(atom_to_locant[a])) for s in sets for a in s)
        cands = [s for s in sets if any((fusion(a), loc_key(atom_to_locant[a])) == low for a in s)]
    elif not cands:
        cands = [frozenset()]
    # Among the sets (1) allows: an indicated hydrogen that accommodates no group sits on a
    # nonfusion atom:24641 "if posible, at the lowest nonfusion peripheral
    # atom"); then the fewest added hydrogen atoms ('1H,5H-pyrido[3,2,1-ij]quinoline-6,8-dione
    # (PIN)':24858: '1H,8H' would accommodate the 8-one but need '6,8(5H)'; '4,4a-dihydro-
    # 2H,5H-...-5,6(6aH)-dione (PIN) (not...-2H,4H-...-5,6(4aH,6aH)-dione)':24854); then (2)
    # the most groups accommodated; then the lowest locants for the indicated hydrogen before
    # the added hydrogen (4).
    best = None
    for s in cands:
        added = _added_hydrogen(state, s, need - s, atom_to_locant)
        if added is None:
            continue
        key = (sum(1 for a in s - need if fusion(a)), len(added), -len(s & need),
               locant_tuple(atom_to_locant[a] for a in s),
               locant_tuple(atom_to_locant[a] for a in added))
        if best is None or key < best[0]:
            best = (key, s, added)
    if best is None:
        return None
    _, s, added = best
    return s, added, state.saturated - s - (need - s) - added


def hydro_text(state: HydroState, hydro_atoms, atom_to_locant: Dict[int, Any],
               ih=frozenset(), need=frozenset()) -> Optional[str]:
    """'' (mancude), '1,2,3,4-tetrahydro', or 'decahydro' for total hydrogenation. Total
    hydrogenation keeps its locants when the indicated hydrogen atoms accommodate suffix
    groups and one of them accommodates none: the book prints '5,6-dihydro-
    1H,3H,4H-3a,6a-methanocyclopenta[c]furan-1,3-dione (PIN)' (:24800) but 'hexahydro-1H-
    isoindole-1,3(2H)-dione (PIN)' (:33849), 'tetrahydro-4,8-ethanopyrano[4,3-c]pyran-1,3,5,7-
    tetrone (PIN)' (:32535), 'hexadecahydro-1H-8,12-methanobenzo[13]annulene (PIN)'
    (:23875)."""
    n = len(hydro_atoms)
    if n == 0:
        return ""
    word = _HYDRO.get(n)
    if word is None:
        return None
    if state.n_double == 0 and not (set(ih) & set(need) and set(ih) - set(need)):
        return word
    locs = sorted((atom_to_locant[a] for a in hydro_atoms), key=loc_key)
    return f"{','.join(str(x) for x in locs)}-{word}"


def ih_text(ih_atoms, atom_to_locant: Dict[int, Any]) -> str:
    """'' or '1H' or '2H,7H'; also the 'added indicated hydrogen' list ('2H', '1H,5H')."""
    locs = sorted((atom_to_locant[a] for a in ih_atoms), key=loc_key)
    return ",".join(f"{x}H" for x in locs)
