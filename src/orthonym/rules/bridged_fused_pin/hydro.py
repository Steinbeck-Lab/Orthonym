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
"""
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from rdkit import Chem

from ...perception.mancude import max_matching_size
from .numbering import loc_key, locant_tuple
from .selection import Split

#: at most this many indicated-hydrogen positions (slice S1)
MAX_INDICATED_H = 2
_HYDRO = {2: "dihydro", 4: "tetrahydro", 6: "hexahydro", 8: "octahydro", 10: "decahydro",
          12: "dodecahydro", 14: "tetradecahydro", 16: "hexadecahydro"}


@dataclass(frozen=True)
class HydroState:
    eligible: FrozenSet[int]          # parent atoms that can carry a ring double bond
    saturated: FrozenSet[int]         # eligible atoms without a ring double bond in the input
    n_double: int                     # ring double bonds of the parent in the input
    mancude_double: int               #... in the mancude bridged parent
    ih_sets: Tuple[FrozenSet[int], ...]   # the indicated-hydrogen sets the parent allows


def hydro_state(mol, split: Split) -> Optional[HydroState]:
    residual = set(split.residual)
    system = residual | {a for chain in split.bridges for a in chain}
    pt = Chem.GetPeriodicTable()
    eligible = set()
    for a in residual:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None
        n_sys = sum(1 for nb in atom.GetNeighbors() if nb.GetIdx() in system)
        if pt.GetDefaultValence(atom.GetAtomicNum()) - n_sys >= 1:
            eligible.add(a)
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
                      tuple(ih_sets))


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


def hydro_text(state: HydroState, hydro_atoms, atom_to_locant: Dict[int, Any]) -> Optional[str]:
    """'' (mancude), '1,2,3,4-tetrahydro', or 'decahydro' for total hydrogenation."""
    n = len(hydro_atoms)
    if n == 0:
        return ""
    word = _HYDRO.get(n)
    if word is None:
        return None
    if state.n_double == 0:
        return word
    locs = sorted((atom_to_locant[a] for a in hydro_atoms), key=loc_key)
    return f"{','.join(str(x) for x in locs)}-{word}"


def ih_text(ih_atoms, atom_to_locant: Dict[int, Any]) -> str:
    """'' or '1H' or '2H,7H'."""
    locs = sorted((atom_to_locant[a] for a in ih_atoms), key=loc_key)
    return ",".join(f"{x}H" for x in locs)
