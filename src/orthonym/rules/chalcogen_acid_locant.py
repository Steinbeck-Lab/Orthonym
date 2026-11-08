"""The italic element locant of a mixed chalcocarbonic or chalcocarbamic acid: 'carbamothioic O-acid'.

 (the Blue Book): "Tautomeric groups in mixed chalcocarbonic acids, such as HO-CO-SH or HO-CS-OH, are
distinguished by prefixing italic element symbols, such as 'S' or 'O', respectively, to the term 'acid'":
'H2N-CS-OH carbamothioic O-acid (PIN)' (:30774), 'H2N-CO-SeH carbamoselenoic Se-acid (PIN)' (:30776), 'HO-CO-SH
carbonothioic S-acid (PIN)' (:30778), 'HSe-CO-SeH carbonodiselenoic Se,Se-acid (PIN)' (:30780), 'carbonothioic O,O-acid
(PIN)' (:30861). Only those shapes are given a locant here: the one acidic atom of a carbamic acid; for a carbonic acid
two acidic atoms of one element other than O, one such atom beside an -OH and =O, or two -OH beside =X. Any other mixed
shape (HS-CS-OH) has no locant from this module, and its names are not the PIN (``validation.spelling.
checks_carbamothioic``).
"""
from typing import Optional

from rdkit import Chem

_CHALCOGENS = ("O", "S", "Se", "Te")


def acid_centre_elements(atom):
    """``(double-bonded element, [acidic elements], has_nitrogen)`` of a carbamic/carbonic acid carbon, else None."""
    if atom.GetSymbol() != "C" or atom.IsInRing() or atom.GetDegree() != 3 or atom.GetFormalCharge():
        return None
    double = [b.GetOtherAtom(atom) for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]
    single = [b.GetOtherAtom(atom) for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE]
    if len(double) != 1 or double[0].GetSymbol() not in _CHALCOGENS or len(single) != 2:
        return None
    acidic = [a.GetSymbol() for a in single
              if a.GetSymbol() in _CHALCOGENS and a.GetDegree() == 1 and a.GetTotalNumHs() == 1]
    nitrogens = [a for a in single if a.GetSymbol() == "N" and not a.GetIsAromatic()]
    if len(acidic) == 1 and len(nitrogens) == 1:
        return double[0].GetSymbol(), acidic, True
    if len(acidic) == 2:
        return double[0].GetSymbol(), sorted(acidic, key=_CHALCOGENS.index), False
    return None


def acid_locant(atom) -> Optional[str]:
    """'O', 'Se', 'S', 'Se,Se', 'O,O' for a mixed chalcogen acid carbon of a book shape; '' when no locant is needed
    (all its chalcogen atoms are of one element); None for a mixed shape the book gives no example of."""
    found = acid_centre_elements(atom)
    if found is None:
        return None
    double, acidic, has_n = found
    if len({double, *acidic}) == 1:
        return ""
    if has_n:
        return acidic[0]
    if acidic[0] == acidic[1] != "O":
        return f"{acidic[0]},{acidic[1]}"
    if acidic[0] == "O" and acidic[1] != "O" and double == "O":
        return acidic[1]
    if acidic == ["O", "O"]:
        return "O,O"
    return None


def mol_acid_locant(mol) -> Optional[str]:
    """:func:`acid_locant` of the one carbamic/carbonic acid carbon of ``mol``; None when there is not exactly one."""
    centres = [a for a in mol.GetAtoms() if acid_centre_elements(a) is not None]
    return acid_locant(centres[0]) if len(centres) == 1 else None
