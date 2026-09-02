"""λ-hydride heteroatom perception — IUPAC P-14.1.3 / P-45.3.1.

A neutral skeletal atom whose bonding number exceeds its standard value (Table
1.3) carries the λ-convention, e.g. a tetrahydridophosphorus substituent -PH4
is ``λ5-phosphanyl``. This module answers the narrow perception question the
substituent namer needs: "is this phosphorus a λ5 *hydride* (-PH4)?", as opposed
to a phosphoryl / phosphonic ``P=O`` (also bonding number 5, but named by the
oxoacid subsystem — P-67).

The distinction is load-bearing (the P-45.3.1 collision): both have bonding
number 5, so the λ-hydride detector fires ONLY when every non-attachment bond of
the phosphorus is to hydrogen. Fail-closed: a charged P, a radical, a ring P, or
any O/N/S neighbour / multiple bond declines.

References:
    IUPAC 2013 Blue Book, P-14.1.1/P-14.1.2 (bonding number, Table 1.3)
    IUPAC 2013 Blue Book, P-14.1.3 (λ-convention on neutral atoms)
    IUPAC 2013 Blue Book, P-45.3.1 (λ5-phosphanyl substituent)
"""

from rdkit import Chem

from ..rules.lambda_convention import nonstandard_bonding_number


def is_lambda_hydride_phosphorus(mol, atom_idx: int) -> bool:
    """True iff ``atom_idx`` is a λ5-hydride phosphorus (-PH4 / PH5).

    Requires: symbol P; neutral; no radical; NOT in a ring; nonstandard bonding
    number of exactly 5 (P-14.1.3); and every non-attachment bond to hydrogen —
    i.e. at most one heavy-atom neighbour (the parent attachment) and all bonds
    single. This EXCLUDES a phosphoryl / phosphonic ``P=O`` (multiple heavy
    neighbours and a P=O double bond), which the oxoacid subsystem names.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() != 'P':
        return False
    if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
        return False
    if atom.IsInRing():
        return False
    if nonstandard_bonding_number(mol, atom_idx) != 5:
        return False
    # Every non-attachment bond must be to H: at most one heavy neighbour (the
    # parent attachment), and no multiple bonds (no P=O phosphoryl).
    heavy_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
    if len(heavy_nbrs) > 1:
        return False
    for bond in atom.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return False
    return True
