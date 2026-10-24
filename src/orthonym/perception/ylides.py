"""An uncharged phosphorus (or As, Sb, Bi) ylide drawn with its double bond: R3P=CR2.

 'Ylides' (the Blue Book): "Compounds in which an anionic site 'Y-'... is attached directly
to a heteroatom 'X+' (usually nitrogen, phosphorus, sulfur, selenium, or tellurium) carrying a formal
positive charge are 1,2-dipolar species of the type RmX+-Y--Rn.... if 'X' is a third, fourth, etc. row
element uncharged canonical forms are usually shown, RmX=YRn. These 'ylides'... may be named in different
ways...: (1) as zwitterionic compounds; (2) by applying the λ-convention when X = P, As, Sb, Bi, S, Se or
Te; (3) by functional class nomenclature... Method (1) is applicable to all 'ylides' and leads to
preferred IUPAC names." The phosphorus example (:42519): '(1) 2-(trimethylphosphaniumyl)propan-2-ide (PIN)';
'(2) trimethyl(propan-2-ylidene)-λ5-phosphane' is not marked a PIN. A name of such a structure that is not
the zwitterion is therefore not the PIN, whichever way it is built.
"""
from __future__ import annotations

from rdkit import Chem

#: the Group 15 elements of the λ5 ylides of (the sulfur ylides have a PIN of their own in
#:, 'dimethyl(methylidene)-λ4-sulfane', the Blue Book; they are left out here)
_GROUP15 = frozenset({"P", "As", "Sb", "Bi"})


def is_uncharged_group15_ylide(mol: Chem.Mol) -> bool:
    """True when ``mol`` has no charged atom and an atom of P, As, Sb or Bi with four neighbours (the
    bonding number 5 of the λ5 form) that is double-bonded to a carbon atom by a bond outside every
    ring. A ring bond P=C is the mancude ring of a λ5-phosphinine, not an ylide centre: its name is a
    ring name."""
    if mol is None or any(atom.GetFormalCharge() for atom in mol.GetAtoms()):
        return False
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.IsInRing():
            continue
        first, second = bond.GetBeginAtom(), bond.GetEndAtom()
        for centre, carbon in ((first, second), (second, first)):
            if (carbon.GetSymbol() == "C" and centre.GetSymbol() in _GROUP15
                    and centre.GetDegree() >= 4):
                return True
    return False
