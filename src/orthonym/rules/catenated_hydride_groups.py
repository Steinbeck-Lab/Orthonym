"""Organyl groups on the hub atoms of an a(ba)n chain: 'hexamethyldisiloxane'.

``### ****`` (the Blue Book), under ```` Heterogenous acyclic parent hydrides: the unbranched
chain of alternating atoms is named by "citing successively a multiplying prefix denoting the number of atoms of the
terminal element followed by the 'a' term for that element, then the 'a' term of the other element in the chain and the
ending 'ane'" ('disiloxane'), and "These parent hydrides are preselected parent hydrides (see and have priority to
receive preferred IUPAC names, as long as they are used to name compounds containing carbon" (:8016). A group on such a
parent is a detachable prefix of a substitutive name: 'chlorodisiloxane' (:2945), 'disiloxanecarboxylic acid (PIN)'
(:29896), and (:23535) 'tristannoxane (preselected name) [not bis(stannyloxy)stannane]'.

The chain stays the parent for a compound with carbon groups on its hub atoms: ``### **** Skeletal replacement
('a') nomenclature in acyclic chains``, **** (:23348) asks for the 'a' name of a chain "when four or more
heterounits are present in an unbranched chain containing at least one carbon atom", and "A heterounit is a set of
heteroatoms having a name of its own such as... -SiH2-O-SiH2-, disiloxane-1,3-diyl" (:23350): the a(ba)n chain of a
carbon-substituted siloxane is ONE heterounit, so the substitutive name stands.

Locants. The chain is numbered 1 to 2n-1 along its atoms (the hub atoms take the odd numbers), and the orientation is
the one with the lowest locants for the prefixes together (``### **** NUMBERING`` (f),:3301) and then for the
prefix cited first (g,:3303). ``****`` (:3007): "All locants are omitted in compounds or substituent groups in
which all substitutable positions are completely substituted or modified... in the same way" ('hexamethyldisiloxane',
'octamethyltrisiloxane'); ``****`` (:2938): the locant is omitted "in monosubstituted symmetrical parent
hydrides... where there is only one kind of substitutable hydrogen" ('chlorodisiloxane',:2945), which is the
disiloxane (two equivalent hub atoms) and no longer chain. Every other name cites all its locants,:2869).

Scope (fail closed, ``None`` and the molecule keeps the namers it had): silicon or germanium hub atoms, oxygen,
sulfur, selenium or tellurium bridge atoms (a nitrogen bridge is an amine,, single bonds from the hub atoms
to groups whose first atom is carbon, each group named by ``substituent_purity.organyl_prefix_name`` (the chokepoint
that refuses a group it cannot name or whose stereo it cannot express); no atom of the chain in a ring; one fragment, no
charge. Tin and lead are metals of ``data/metal_scope.py`` and are not read here.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from rdkit import Chem

__all__ = ["name_hub_chain_with_groups"]

#: the hub elements read here (the metals of Group 14 are the table of ``data/metal_scope.py``)
_HUBS = frozenset({'Si', 'Ge'})
_BRIDGES = frozenset({'O', 'S', 'Se', 'Te'})


def _chain_order(mol, hubs, bridges) -> Optional[List[int]]:
    """The atoms of the chain, hub first, in order from one end, or None when they are not one unbranched path that
    alternates hub and bridge atoms and ends in two hub atoms."""
    hub_set, bridge_set = {a.GetIdx() for a in hubs}, {a.GetIdx() for a in bridges}
    if len(hubs) < 2 or len(bridges) != len(hubs) - 1:
        return None
    chain = hub_set | bridge_set
    adjacency = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in chain] for i in chain}
    for i in hub_set:
        if any(n not in bridge_set for n in adjacency[i]) or not 1 <= len(adjacency[i]) <= 2:
            return None
    for i in bridge_set:
        if len(adjacency[i]) != 2 or any(n not in hub_set for n in adjacency[i]):
            return None
    ends = [i for i in hub_set if len(adjacency[i]) == 1]
    if len(ends) != 2:
        return None
    order, prev, cur = [ends[0]], None, ends[0]
    while True:
        following = [n for n in adjacency[cur] if n != prev]
        if not following:
            break
        prev, cur = cur, following[0]
        order.append(cur)
    return order if len(order) == len(chain) else None


def _groups(mol, order) -> Optional[Dict[int, List[str]]]:
    """``hub atom -> the names of the groups on it``; None when a group is not an organyl the chokepoint names, or is
    not bonded by a single bond from a carbon atom."""
    from .substituent_purity import organyl_prefix_name
    chain = set(order)
    out: Dict[int, List[str]] = {}
    for idx in order[::2]:
        names = []
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetIdx() in chain or nb.GetAtomicNum() == 1:
                continue
            bond = mol.GetBondBetweenAtoms(idx, nb.GetIdx())
            if nb.GetSymbol() != 'C' or bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            name = organyl_prefix_name(mol, nb.GetIdx(), idx)
            if name is None:
                return None
            names.append(name)
        out[idx] = names
    return out


def _hub_slots(mol, idx: int) -> int:
    """The groups a hub atom can carry: its bonding number less its two (or one) chain bonds."""
    from .lambda_convention import nonstandard_bonding_number
    if nonstandard_bonding_number(mol, idx) is not None:
        return -1
    atom = mol.GetAtomWithIdx(idx)
    return 4 - sum(1 for n in atom.GetNeighbors() if n.GetSymbol() in _BRIDGES)


def name_hub_chain_with_groups(mol, hubs, bridges, hub_stems, bridge_suffixes,
                               parent_name: Callable[[str, str, int], str]) -> Optional[str]:
    """The PIN of an a(ba)n chain with organyl groups on its hub atoms, or None (see the module docstring)."""
    if not hubs or not bridges:
        return None
    hub_sym, bridge_sym = hubs[0].GetSymbol(), bridges[0].GetSymbol()
    if (hub_sym not in _HUBS or bridge_sym not in _BRIDGES or any(a.GetSymbol() != hub_sym for a in hubs)
            or any(a.GetSymbol() != bridge_sym for a in bridges)):
        return None
    if (any(atom.IsInRing() for atom in hubs + bridges) or any(a.GetIsotope() for a in mol.GetAtoms())
            or any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms())
            or any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds())):
        return None                      # a ring through the chain, a label, a stereo element: not named here
    order = _chain_order(mol, hubs, bridges)
    groups = _groups(mol, order) if order else None
    if groups is None or any(_hub_slots(mol, i) < 0 for i in groups):
        return None
    # every heavy atom is on the chain or in a group named above (the chokepoint reads the whole group)
    from ..assembly.naming_utils import alpha_sort_key
    best = None
    for seq in (order, order[::-1]):
        locant = {idx: 2 * k + 1 for k, idx in enumerate(seq[::2])}
        pairs = [(name, locant[i]) for i in groups for name in groups[i]]
        by_name: Dict[str, List[int]] = {}
        for name, loc in pairs:
            by_name.setdefault(name, []).append(loc)
        cited = tuple(loc for name in sorted(by_name, key=alpha_sort_key) for loc in sorted(by_name[name]))
        key = (sorted(loc for _, loc in pairs), cited)
        if best is None or key < best[0]:
            best = (key, seq, pairs)
    _, seq, pairs = best
    parent = parent_name(hub_sym, bridge_sym, len(hubs))
    if not pairs:
        return None
    from .locanted_prefixes import locanted_prefix_block
    names = {name for name, _ in pairs}
    complete = all(len(groups[i]) == _hub_slots(mol, i) for i in groups) and len(names) == 1
    monosubstituted_disiloxane = len(hubs) == 2 and len(pairs) == 1
    if complete or monosubstituted_disiloxane:
        from ..assembly.prefix_marks import mononuclear_substituent_block
        block = mononuclear_substituent_block([name for name, _ in pairs], max_count=max(len(pairs), 1))
        return None if block is None else f"{block}{parent}"
    block = locanted_prefix_block([(name, str(loc)) for name, loc in pairs])
    return None if block is None else f"{block}{parent}"
