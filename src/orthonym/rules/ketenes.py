"""Ketene namer (P-64.2.2.4).

Ketene is the class name for H2C=C=O and its derivatives. The unsubstituted
structure and its halogen derivatives are named on the ``ethenone`` parent
(BB verbatim examples)::

    C=C=O -> ethenone (PIN)
    BrC(Br)=C=O -> dibromoethenone (PIN; "not dibromoketene")

SCOPE (fail-closed, accuracy-first): the exact terminal heterocumulene
O=C=C< where the sp carbon carries NOTHING else and the terminal carbon
carries only hydrogen and/or halogens. Halogens are the P-15.1.8.2
"compulsory prefix" substituents the retained class name allows, and the
single substitutable carbon makes the prefix locants unambiguous (BB cites
``dibromoethenone`` with no locants). Everything else fails a guard and
cascades onward — alkyl/aryl ketenes (2-butylhex-1-en-1-one) and ylidene
ketenes (cyclohexylidenemethanone) are named by the general ketone
principles, which are NOT built here; never a wrong name.

Graph/atom classifier (no SMARTS broadening); pure — no mol mutation.
"""

from collections import deque
from typing import Optional

from rdkit import Chem

_HALO_PREFIX = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}


def _branch_atoms(mol, start: int, exclude: int) -> set:
    """Connected atom set reachable from ``start`` without crossing
    ``exclude`` (the shared terminal carbon)."""
    seen = {start}
    queue = deque([start])
    while queue:
        idx = queue.popleft()
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            j = nbr.GetIdx()
            if j != exclude and j not in seen:
                seen.add(j)
                queue.append(j)
    return seen


def name_ketene(mol) -> Optional[str]:
    """Return the ethenone-parent PIN for a (halo)ketene, else ``None``."""
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    # NOTE: rings ARE allowed now — Branch 3 (cyclohexylidenemethanone) and
    # Branch 2 (diphenylethenone) need ring substituents. The halogen branch
    # still fails closed on any ring atom via its own coverage check.
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    # Locate the cumulated sp carbon: C(=O)(=C), degree exactly 2, no H.
    sp_carbon = None
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetTotalNumHs() != 0:
            continue
        nbrs = atom.GetNeighbors()
        if len(nbrs) != 2:
            continue
        symbols = sorted(n.GetSymbol() for n in nbrs)
        if symbols != ['C', 'O']:
            continue
        bonds = [mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx())
                 for n in nbrs]
        if all(b.GetBondType() == Chem.BondType.DOUBLE for b in bonds):
            sp_carbon = atom
            break
    if sp_carbon is None:
        return None

    oxygen = next(n for n in sp_carbon.GetNeighbors() if n.GetSymbol() == 'O')
    terminal_c = next(n for n in sp_carbon.GetNeighbors()
                      if n.GetSymbol() == 'C')
    if oxygen.GetDegree() != 1:
        return None

    # Branch 3 (P-64.5(3) oxomethylidene on a ring): terminal C is a RING
    # atom (spiro-exocyclic cumulene) -> '<ring-ylidene>methanone'
    # (cyclohexylidenemethanone). Ring must be a plain cycloalkane fragment
    # nameable as an '-ylidene' substituent; else fail closed.
    if terminal_c.IsInRing():
        ring_atoms = ({a.GetIdx() for a in mol.GetAtoms()}
                      - {sp_carbon.GetIdx(), oxygen.GetIdx()})
        from ..assembly.substituent_enumerator import name_ylidene_substituent
        # The ring is DOUBLE-bonded to the sp carbon, so the pipeline returns
        # the '-ylidene' already formed; appending 'idene' to a '-yl' token
        # here would be the morphology decided twice.
        base = name_ylidene_substituent(mol, ring_atoms, terminal_c.GetIdx())
        if base is None:
            return None
        if mol.GetNumHeavyAtoms() != len(ring_atoms) + 2:
            return None
        return f"{base}methanone"

    # Branch 2 (P-64.2.2.4): BOTH substituents identical ARYL groups, zero H
    # on the terminal C -> '<di><name>ethenone' (BB/scope-decision verbatim:
    # diphenylethenone). Only aromatic-ring substituents take this
    # ethenone form; ACYCLIC alkyl cumulated chains use ordinary ketone
    # '-one' numbering (scope decision #4: 2-butylhex-1-en-1-one) and are
    # NOT this namer's class -> fail closed. Any asymmetry or H fails too.
    heavy_subs = [n for n in terminal_c.GetNeighbors()
                  if n.GetIdx() != sp_carbon.GetIdx()
                  and n.GetSymbol() not in _HALO_PREFIX]
    if heavy_subs:
        if len(heavy_subs) != 2 or terminal_c.GetTotalNumHs() != 0:
            return None
        # Both attachment atoms must be aromatic ring atoms (aryl only).
        if not all(n.GetIsAromatic() and n.IsInRing() for n in heavy_subs):
            return None
        from ..assembly.substituent_enumerator import name_substituent
        names, covered = [], {sp_carbon.GetIdx(), oxygen.GetIdx(),
                              terminal_c.GetIdx()}
        for nbr in heavy_subs:
            frag = _branch_atoms(mol, nbr.GetIdx(),
                                 exclude=terminal_c.GetIdx())
            nm = name_substituent(mol, frag, nbr.GetIdx())
            if not nm:
                return None
            names.append(nm)
            covered |= frag
        if names[0] != names[1]:
            return None
        if covered != {a.GetIdx() for a in mol.GetAtoms()}:
            return None
        return f"di{names[0]}ethenone"

    # The terminal carbon may carry only H and/or single-bonded halogens.
    halo_counts: dict = {}
    for nbr in terminal_c.GetNeighbors():
        if nbr.GetIdx() == sp_carbon.GetIdx():
            continue
        symbol = nbr.GetSymbol()
        if symbol not in _HALO_PREFIX or nbr.GetDegree() != 1:
            return None
        bond = mol.GetBondBetweenAtoms(terminal_c.GetIdx(), nbr.GetIdx())
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return None
        halo_counts[symbol] = halo_counts.get(symbol, 0) + 1

    # Whole-molecule coverage: sp C + terminal C + O + counted halogens.
    if mol.GetNumHeavyAtoms() != 3 + sum(halo_counts.values()):
        return None

    prefix_parts = []
    for symbol in sorted(halo_counts, key=lambda s: _HALO_PREFIX[s]):
        halo_name = _HALO_PREFIX[symbol]
        count = halo_counts[symbol]
        multiplier = {1: '', 2: 'di'}[count]
        prefix_parts.append(f"{multiplier}{halo_name}")
    return f"{''.join(prefix_parts)}ethenone"


__all__ = ["name_ketene"]
