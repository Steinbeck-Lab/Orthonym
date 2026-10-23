"""The floor's name of a molecule whose one radical centre is a skeletal heteroatom of a ring.

A ring Si, Ge or B atom that holds fewer bonds than its standard valence and carries the rest as radical electrons
(a silylene drawn ``[Si]`` in a ring, a ring ``[B]``) is the atom of its parent hydride with a lowered valence, which
the lambda convention writes: ``2λ2-sila...`` "Nonstandard bonding numbers", the Blue Book: "A nonstandard
bonding number of a neutral skeletal atom of a parent hydride is indicated by the symbol 'λn', cited in
conjunction with an appropriate locant"). The radical electrons are then already cited, and a '-ylidene' beside the lambda cites them a
second time: the floor's radical rung names the whole heavy-atom set as an '-ylidene' prefix with the centre as its
attachment atom, which gave ``spiro[2λ2-sila...]-2-ylidene``, a name OPSIN reads as another structure. The name of
the molecule itself, with the lambda convention and no ylidene, is the name of the species.
"""
from typing import Optional

from rdkit import Chem

#: skeletal heteroatoms whose lowered valence the floor writes with the lambda convention
_LAMBDA_ELEMENTS = frozenset({"Si", "Ge", "B"})


def skeletal_lambda_floor_name(mol, radical_site) -> Optional[str]:
    """The whole-molecule floor name of ``mol`` when its only radical centre is a ring Si, Ge or B atom whose
    radical electrons are its missing valence, else None."""
    atom = mol.GetAtomWithIdx(radical_site["atom_idx"])
    if (atom.GetSymbol() not in _LAMBDA_ELEMENTS or not atom.IsInRing() or atom.GetFormalCharge()
            or atom.GetIsAromatic()):
        return None
    standard = Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())
    held = atom.GetTotalValence()
    if atom.GetNumRadicalElectrons() != radical_site["n_electrons"] or held + radical_site["n_electrons"] != standard:
        return None
    from .universal_substituent import name_universal_substitutive
    named = name_universal_substitutive(mol)
    name = getattr(named, "name", None)
    return name if name and "λ" in name else None
