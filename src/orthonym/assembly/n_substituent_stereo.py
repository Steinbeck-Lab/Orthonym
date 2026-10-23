"""The stereodescriptors of an N-substituent of an amine, cited as the amide's N-substituents cite them.

 (the Blue Book): the descriptors of a substituent group "are cited at the front of the corresponding
prefix" with a locant: 'N-[(1R)-1-phenylethyl]'. ``rules.amides._name_n_substituent`` gets that spelling from the
substituent cascade (``substituent_enumerator.name_substituent``, whose stereo route derives the locant from the
structure); the amine assembler names its N-substituents from their atoms and cites the descriptors afterwards.
This is the one step that gives it the cascade's spelling when the cascade cites every stereogenic unit of the group,
and the bare '(R)-' of ``_add_substituent_stereo`` (a name below the PIN) when it does not.
"""
from typing import Iterable, Optional

from ..rules.stereochemistry import count_expressed_stereo_descriptors
from .substituent_naming import _add_substituent_stereo


def _stereo_units(mol, frag) -> int:
    atoms = sum(1 for a in frag if mol.GetAtomWithIdx(a).HasProp('_CIPCode'))
    members = set(frag)
    bonds = sum(1 for b in mol.GetBonds()
                if b.GetBeginAtomIdx() in members and b.GetEndAtomIdx() in members and b.HasProp('_CIPCode'))
    return atoms + bonds


def cite_n_substituent_stereo(mol, frag: Iterable[int], attach_idx: int, name: Optional[str]) -> Optional[str]:
    """``name`` of the N-substituent ``frag`` (attached at ``attach_idx``) with its stereodescriptors."""
    if not name:
        return name
    frag = list(frag)
    units = _stereo_units(mol, frag)
    if not units:
        return name
    from .substituent_enumerator import name_substituent
    cascade = name_substituent(mol, set(frag), attach_idx)
    if (cascade and isinstance(cascade, str) and not cascade.startswith('substituent')
            and count_expressed_stereo_descriptors(cascade) >= units):
        return cascade
    return _add_substituent_stereo(mol, frag, name, attach_idx=attach_idx)
