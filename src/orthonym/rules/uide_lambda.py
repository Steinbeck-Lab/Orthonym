"""The -uide anion of a hypervalent centre: the hydride adduct of its lambda hydride.

 'ANIONS FORMED BY ADDITION OF HYDRIDE IONS' (the Blue Book): the suffix 'uide' describes an anion
formally derived by adding a hydride ion, H-, to a parent hydride name, and a hypervalent centre adds it to its
lambda hydride: 'hexafluoro-λ5-iodanuide (preselected name)' (:41112, IF6-), 'octafluoro-λ6-tellanediuide
(preselected name)' (:41114). The bonding number of the anion is one above the lambda number of the hydride, so
a sulfur anion with three bonds is 'sulfanuide' (:41139, 'sulfanuida (preselected name)'), with five bonds
'λ4-sulfanuide' and with seven 'λ6-sulfanuide'.
"""
from typing import Optional

# Standard bonding number of each element that has a -uide anion (degree + H of the anion == standard + 1).
UIDE_STD_VALENCE = {
    'B': 3, 'Al': 3, 'Ga': 3, 'In': 3, 'Tl': 3,
    'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 4,
    'P': 3, 'As': 3, 'Sb': 3,
    'S': 2, 'Se': 2, 'Te': 2,
    'I': 1, 'Br': 1, 'Cl': 1,
}

# How many steps of two above the standard bonding number the lambda hydride of the element may take:
# the chalcogens up to lambda 6, the pnictogens lambda 5, the halogens lambda 7.
UIDE_LAMBDA_STEPS = {'S': 2, 'Se': 2, 'Te': 2, 'P': 1, 'As': 1, 'Sb': 1, 'I': 3, 'Br': 3, 'Cl': 3}


def uide_bonding_number(atom) -> int:
    """The bonding number of the centre: the bond orders and hydrogens for an element of lambda hydrides, the
    neighbours and hydrogens for the others (a multiple bond on boron or silicon is no -uide centre)."""
    if atom.GetSymbol() in UIDE_LAMBDA_STEPS:
        return sum(int(b.GetBondTypeAsDouble()) for b in atom.GetBonds()) + atom.GetTotalNumHs()
    return atom.GetDegree() + atom.GetTotalNumHs()


def uide_lambda(symbol: str, bonding_number: int) -> Optional[int]:
    """The bonding number of the hydride that an anion of ``symbol`` with ``bonding_number`` adds H- to, or None
    when it is no -uide centre of the element. The name carries the λ of it only when it is not the standard
    bonding number (``UIDE_STD_VALENCE``)."""
    std = UIDE_STD_VALENCE.get(symbol)
    if std is None:
        return None
    step = bonding_number - 1 - std
    if step == 0:
        return std
    if step > 0 and step % 2 == 0 and step // 2 <= UIDE_LAMBDA_STEPS.get(symbol, 0):
        return bonding_number - 1
    return None


def double_bonded_ligand(atom) -> bool:
    """True when a ligand of the centre holds a multiple bond to it ('oxo', 'imino', 'methylidene' groups)."""
    return any(b.GetBondTypeAsDouble() > 1 for b in atom.GetBonds())


def _is_sulfinyl_anion(mol, center_idx: int) -> bool:
    """The sulfur centre of the sulfinyl anion R-S(=O)(-) of the book's example: one terminal oxygen by a double bond
    and one carbon by a single bond, no hydrogen."""
    centre = mol.GetAtomWithIdx(center_idx)
    if centre.GetSymbol() != 'S':
        return False
    ligands = [(n, mol.GetBondBetweenAtoms(center_idx, n.GetIdx()).GetBondTypeAsDouble()) for n in centre.GetNeighbors()]
    return (not centre.GetTotalNumHs() and len(ligands) == 2
            and sorted((n.GetSymbol(), order, n.GetDegree() if n.GetSymbol() == 'O' else 0) for n, order in ligands)
            == [('C', 1.0, 0), ('O', 2.0, 1)])


def lambda_uide_name(mol, center_idx: int, lam: int, suffix: str) -> str:
    """The name of the anion of a hypervalent centre or of one that holds a multiply bonded ligand: the ligands of
    the centre as prefixes (``centre_name``), '-λ<n>-' and the ending of the parent hydride.

    A centre above its standard bonding number is the -uide of its lambda hydride, 'amino(imino)(oxo)-λ4-sulfanuide'
    , 'hexafluoro-λ5-iodanuide (preselected name)', the Blue Book): the name of a parent hydride, not the PIN
    of a substituted one, so the name is recorded as not the PIN.

    A centre at its standard bonding number that holds a double bond ('oxo') is the 'ide' of the lambda hydride that has
    the two more bonds: 'benzenesulfinyl anion oxo(phenyl)-λ4-sulfanide (PIN)' Functional class
    nomenclature,:40880). That name is the PIN of a sulfinyl anion R-S(=O)(-); for any other group on such a centre
    the book gives no example and the name is recorded as not the PIN. '' when a ligand has no name."""
    from ..metrics.provenance import record_non_pin_fragment
    from .centre_ligands import centre_name
    from .lambda_convention import LAMBDA
    if mol.GetAtomWithIdx(center_idx).GetSymbol() in UIDE_STD_VALENCE and lam == UIDE_STD_VALENCE[
            mol.GetAtomWithIdx(center_idx).GetSymbol()]:
        lam, suffix, pin = lam + 2, suffix[:-len('uide')] + 'ide', _is_sulfinyl_anion(mol, center_idx)
    else:
        pin = False
    name = centre_name(mol, center_idx, f"-{LAMBDA}{lam}-{suffix}", f"{LAMBDA}{lam}-{suffix}")
    if name and not pin:
        record_non_pin_fragment(name)
    return name
