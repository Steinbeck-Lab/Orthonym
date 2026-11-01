"""The prefix of a carbon group on a mononuclear chalcogen onium cation, the Blue Book).

'ethyl(propan-2-ylidene)oxidanium (PIN)' (:41465): the groups on the cationic centre are prefixes, a carbon group
by its substituent name ('propan-2-yl', '2-methylpropyl', 'tert-butyl', 'prop-2-en-1-yl', '2-chloroethyl').
``perception.chains.classify_substituent`` names an 'alkyl' group by its carbon count ('propyl' for propan-2-yl: another
constitution) and cannot name a heteroatom in it, so ``ions.emit_onium_hydride_parent`` asks the shared substituent
namer first and keeps the count-based name as the fallback.
"""
from typing import Any, Optional


def onium_carbon_group_name(mol, sub_atoms, neighbor, info: Any) -> Optional[str]:
    """The prefix of the group ``sub_atoms`` bonded to the centre through ``neighbor`` (``info`` is its
    ``classify_substituent`` row): the name of the substituent namer for a carbon group, else the count-based name.
    False when only the count-based name was left for an 'alkyl' group that holds a heteroatom (an atom drop)."""
    name = info.get('name')
    if info.get('type') != 'alkyl':
        return name
    if neighbor.GetSymbol() == 'C':
        from ..assembly.substituent_enumerator import name_substituent
        own = name_substituent(mol, set(sub_atoms), neighbor.GetIdx())
        if own and own != 'substituent':
            return own
    if any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in sub_atoms):
        return False
    return name
