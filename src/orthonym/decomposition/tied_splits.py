"""Which of several equally ranked ester splits the decomposition names the molecule from.

``engine._select_best_bond`` ranks the cleavable bonds by type and by how evenly the split halves
the molecule. Two ester bonds can tie on both (the two acetates of a diacetate, the two
but-2-enoates of a piperazine bis-enoate, the five acetates of a polyol), and the first of them in
the atom order of the input was cleaved. Every such bond gives a name of the molecule, and the
names differ: ``…-2-methylpiperazin-1-yl}…`` against ``…-3-methylpiperazin-1-yl}…``, ``…-en-1-yl
acetate`` against ``…-en-11-yl acetate``, a butenoate chain against a propenoate chain.

The Blue Book decides between such names. The criteria are applied in this order
(the Blue Book, " SENIORITY OF ACYCLIC CHAINS (THE PRINCIPAL CHAIN)":20916, "
SENIORITY CRITERIA APPLICABLE TO RINGS, RING SYSTEMS, OR ACYCLIC CHAINS":21014, " CRITERIA
RELATED TO ALPHANUMERICAL ORDER OF NAMES":22234):

1. The acid component of the split: "Systems composed of rings and chains" (1) "Within
   the same class, a ring or ring system has seniority over a chain" (:19336); then "The
   principal chain has the greater number of skeletal atoms [criterion (b) in ]" (:20938).
2. "The senior ring or ring system has the lowest locants for points of attachment
   (if a substituent group)" (:21354, ``pyridin-2-yl`` before ``pyridin-3-yl``): the alkyl
   component, when the two names cite the same skeleton.
3. "The preferred IUPAC name is the name that is earlier in alphanumerical order...
   Alphabetic letters are considered first in the order that they appear in the name... Then, if
   still there is a choice, numerical locants are considered in the order of their appearance in
   the name", then (:22603) the order of the descriptors R and S.

Only the splits that tie are named and compared; a name is taken from the split that the order
puts first, and every other criterion of ``_select_best_bond`` stays as it was. What this does not
model: the heteroatoms of an acid chain (a)), the multiple bonds of the chain,
and the seniority of two different ring skeletons.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

#: the most tied splits that are named
MAX_TIED_SPLITS = 6

_ATTACHMENT_RE = re.compile(r"(\d+)(?:\(\d+H\))?-yl$")


def acyl_chain_atoms(mol, acyl_c: int) -> int:
    """The number of skeletal atoms of the longest chain of acyclic carbon atoms that starts at the
    acyl carbon (the carboxyl carbon is the first atom of the chain), 0 for an acyl carbon on a
    ring. The acyclic carbon atoms of a molecule form a forest, so a walk away from the start
    cannot meet itself."""
    atom = mol.GetAtomWithIdx(acyl_c)
    if any(nb.GetAtomicNum() == 6 and nb.IsInRing() for nb in atom.GetNeighbors()):
        return 0

    def depth(a: int, parent: int) -> int:
        longest = 0
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            j = nb.GetIdx()
            if j != parent and nb.GetAtomicNum() == 6 and not nb.IsInRing():
                longest = max(longest, depth(j, a))
        return 1 + longest

    return depth(acyl_c, -1)


def _parent_key(mol, bond: Dict) -> Tuple[int, int]:
    """(0 for a ring acid, 1 for a chain acid; minus the skeletal atoms of the acid chain)."""
    acyl_c = bond["acid_atom"]
    chain = acyl_chain_atoms(mol, acyl_c)
    return (0, 0) if chain == 0 else (1, -chain)


def attachment_locant(name: str) -> int:
    """The locant of the free valence of the alkyl word of an ester name (0 when it is not cited)."""
    found = _ATTACHMENT_RE.search(name.split(" ", 1)[0])
    return int(found.group(1)) if found else 0


def name_order_key(name: str) -> Tuple[str, int, tuple, tuple]:
    """``(letters, locant of the free valence, locants in order of appearance, R/S letters)``: the
    order of criteria 2 and 3 of the module docstring. The letters of a name are compared first
    , so the locant of the free valence is compared only between names that cite the same
    skeleton."""
    from ..rules.benzene import _alphanumerical_name_key
    letters, locants, stereo = _alphanumerical_name_key(name)
    return (letters, attachment_locant(name), locants, stereo)


def _symmetry_classes(mol, bonds: List[Dict]) -> List[Dict]:
    """One bond of each class of bonds that the symmetry of the molecule maps onto each other (they
    give one name), ordered by the canonical ranks of their atoms (not by the input order)."""
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    seen = {}
    for bond in bonds:
        b = mol.GetBondWithIdx(bond["bond_idx"])
        key = tuple(sorted((ranks[b.GetBeginAtomIdx()], ranks[b.GetEndAtomIdx()])))
        seen.setdefault(key, bond)
    return [seen[key] for key in sorted(seen)]


def tied_ester_splits(mol, bonds: List[Dict], best: Dict, balance) -> List[Dict]:
    """The ester bonds that rank with ``best`` on every criterion of ``_select_best_bond`` (one per
    symmetry class), or ``[best]`` when ``best`` is not an ester or stands alone.

    ``balance(mol, bond)`` is the engine's evenness score. An ester whose acid ring carries fewer
    substituents than another of the same skeleton is left out, as the engine does), and
    so is one whose parent roles are swapped when ``best``'s are not (the two names are not of one
    kind)."""
    if best.get("type") != "ester":
        return [best]
    from . import engine
    level = balance(mol, best)
    peers = [b for b in bonds
             if b.get("type") == "ester" and balance(mol, b) == level
             and bool(b.get("roles_swapped")) == bool(best.get("roles_swapped"))
             and not engine._ester_peers_by_parent_substituents(mol, bonds, b)]
    if len(peers) < 2 or not any(b["bond_idx"] == best["bond_idx"] for b in peers):
        return [best]
    return _symmetry_classes(mol, peers)[:MAX_TIED_SPLITS]


def choose_split(mol, group: List[Dict], best: Dict, style: str, name_split, adequate, exhausted
                 ) -> Tuple[Dict, Optional[str]]:
    """``(bond, name)`` of the split of ``group`` whose name the criteria of the module docstring put
    first; ``(best, its name)`` when none of them gives an adequate name.

    ``name_split(mol, bond, style)`` names the molecule from a split, ``adequate(name, mol, bond)``
    is the engine's coverage gate and ``exhausted`` its exploration budget."""
    named = {}
    for bond in group:
        if named and exhausted():
            break
        named[bond["bond_idx"]] = name_split(mol, bond, style)
    scored = [((_parent_key(mol, b), name_order_key(named[b["bond_idx"]])), b)
              for b in group
              if named.get(b["bond_idx"]) and adequate(named[b["bond_idx"]], mol, b)]
    if not scored:
        if best["bond_idx"] not in named:
            named[best["bond_idx"]] = name_split(mol, best, style)
        return best, named[best["bond_idx"]]
    _, winner = min(scored, key=lambda s: s[0])
    return winner, named[winner["bond_idx"]]
