"""Ketene namer (P-64.2.2.4).

Ketene is the class name for H2C=C=O and its derivatives. The unsubstituted
structure and its halogen derivatives are named on the ``ethenone`` parent
(BB verbatim examples)::

    C=C=O          -> ethenone         (PIN)
    BrC(Br)=C=O    -> dibromoethenone  (PIN; "not dibromoketene")

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

from typing import Optional

from rdkit import Chem

_HALO_PREFIX = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}


def name_ketene(mol) -> Optional[str]:
    """Return the ethenone-parent PIN for a (halo)ketene, else ``None``."""
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if mol.GetRingInfo().NumRings() > 0:
        return None
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
