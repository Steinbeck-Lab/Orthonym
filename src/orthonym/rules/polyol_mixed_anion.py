"""The label of method (2) of for the routes that name an ester by its acid (carbamate, noncarbon ester)."""
from typing import Optional


def record_mixed_anion_polyol_non_pin(mol, name: Optional[str], centre_symbols) -> None:
    """:func:`record_polyol_mixed_anion_non_pin` for every ester bond of ``mol`` whose acid centre is one of
    ``centre_symbols`` (the acid the name is built on: 'C' for a carbamate, 'S'/'P'/... for a noncarbon ester): the routes
    that name an ester by its acid without knowing which ester bond holds the polyol ('2-(nitrooxy)ethyl carbamate',
    '2-(nitrooxy)ethyl methanesulfonate', method (2) of "When anions are different, two methods are
    used", the Blue Book). An ester cited as a prefix in an N-substituent keeps its name ('tert-butyl
    [2,3-bis(nitrooxy)propyl]carbamate')."""
    from .esters import _ester_bond_atoms, record_polyol_mixed_anion_non_pin
    if not name or mol is None:
        return
    for atom in mol.GetAtoms():
        pair = _ester_bond_atoms(mol, atom.GetIdx())
        if pair is not None and mol.GetAtomWithIdx(pair[0]).GetSymbol() in centre_symbols:
            record_polyol_mixed_anion_non_pin(mol, pair[1], atom.GetIdx(), name)
