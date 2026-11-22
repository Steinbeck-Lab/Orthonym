"""A bis(azo) compound without a suffix group is named on 'diazene'; Orthonym builds no such name yet.

 (the Blue Book): "Bis(azo) compounds and more complex analogues, in the absence of a characteristic
group having seniority to be cited as suffix, are named: (1) on the basis of 'diazene'... Method (1) generates
preferred IUPAC names": '{7-[(anthracen-2-yl)diazenyl]naphthalen-2-yl}(phenyl)diazene (PIN)' (:38829). The ring
parent names ('1,4-bis(phenyldiazenyl)benzene', '2-{[7-(phenyldiazenyl)naphthalen-2-yl]diazenyl}anthracene') were
labelled as PINs;:func:`note_bis_azo` records them below the PIN.
"""
from __future__ import annotations

from rdkit import Chem

_HALOGENS = frozenset({"F", "Cl", "Br", "I"})


def _azo_bonds(mol) -> int:
    """The acyclic N=N bonds whose two nitrogens are neutral and each bonded to one carbon (C-N=N-C)."""
    count = 0
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if (bond.GetBondType() == Chem.BondType.DOUBLE and a.GetSymbol() == b.GetSymbol() == "N"
                and not bond.IsInRing() and not a.GetFormalCharge() and not b.GetFormalCharge()
                and all(at.GetDegree() == 2 and sum(nb.GetSymbol() == "C" for nb in at.GetNeighbors()) == 1
                        for at in (a, b))):
            count += 1
    return count


def _prefix_only(mol) -> bool:
    """No atom that could carry a suffix group: carbon, the azo nitrogens, halogens, ether oxygen or sulfur (bonded to
    two carbons) and the skeletal heteroatoms of a ring ('2-{[4-(phenyldiazenyl)phenyl]diazenyl}pyridine': a ring
    nitrogen is no characteristic group) only."""
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if atom.GetFormalCharge() or atom.GetNumRadicalElectrons() or atom.GetIsotope():
            return False
        if sym == "C" or sym in _HALOGENS:
            continue
        if sym == "N" and atom.GetDegree() == 2 and not atom.IsInRing() and any(
                b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(atom).GetSymbol() == "N"
                for b in atom.GetBonds()):
            continue
        if sym in ("O", "S") and atom.GetDegree() == 2 and all(nb.GetSymbol() == "C" for nb in atom.GetNeighbors()):
            continue
        if sym in ("N", "O", "S") and atom.IsInRing() and all(nb.IsInRing() for nb in atom.GetNeighbors()):
            continue                            # a skeletal ring atom without an exocyclic group on it
        return False
    return True


def note_bis_azo(mol) -> None:
    """Record 'diazenyl' below the PIN for a molecule (named at the top level) with two or more azo groups and no atom
    that could carry a suffix group: its PIN is a 'diazene' name, which no producer builds."""
    from ..assembly.fragment_naming import is_top_level_naming
    if mol is None or not is_top_level_naming() or _azo_bonds(mol) < 2 or not _prefix_only(mol):
        return
    from ..metrics.provenance import record_non_pin_label
    record_non_pin_label("diazenyl", rule="P-68.3.1.3.2.3", detail="a bis(azo) compound is named on 'diazene'")
