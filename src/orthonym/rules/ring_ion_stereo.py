"""The stereodescriptors of a ring ion named on its bare parent ring names, ``rules.ions``).

``ions._p74_bare_ring_stem`` names the parent ring with every exocyclic substituent deleted. The deleted atoms leave the
ring atoms' chiral tags in place, so the stem can come back with its own block: '(1R,5S)-8-azabicyclo[3.2.1]octane'.
That block is wrong twice over for the ion. It was computed on the bare ring, not on the ion (a ring atom whose
substituent was deleted loses its centre: C3 of tropan-3-ol), and the ion's prefixes are then joined in front of it:
'3-hydroxy-8,8-dimethyl(1R,5S)-8-azabicyclo[3.2.1]octan-8-ium', a name no parser reads.

 "NAMING OF STEREOISOMERS" (the Blue Book): stereodescriptors "are placed at the front of the complete name
when related to the parent structure", with the locants of the name's own numbering. So the stem is used without its
block, and the block of the ring atoms is computed on the ion, in the ion's numbering, and cited at the front."""
import re
from typing import Dict, Optional, Tuple

_LEADING_BLOCK = re.compile(r"^\([^()]*\)-")


def ring_stereo_front(mol, stem: str, locants: Dict[int, int]) -> Tuple[Optional[str], str]:
    """``(stem without its leading block, the ion's block for the ring atoms)``.

    A stem without a leading block is returned unchanged with an empty block (the ring carries no stereo). The ion's
    block comes from ``ring_cation_hydro.ring_stereo_prefix`` on the ion with its CIP labels assigned, ``locants``
    mapping the ion's ring atoms to the locants the ion name cites. ``(None, '')`` where that block cannot be given
    (a stereogenic ring nitrogen with a hydrogen, a labeller failure): the caller declines rather than cite the bare
    ring's block."""
    if not stem or not _LEADING_BLOCK.match(stem):
        return stem, ""
    from rdkit import Chem

    from ..perception.stereo import assign_stereochemistry
    from .ring_cation_hydro import ring_stereo_prefix
    work = Chem.Mol(mol)
    try:
        assign_stereochemistry(work)
        front = ring_stereo_prefix(work, dict(locants))
    except Exception:   # noqa: BLE001 -- any labeller failure declines the name, never ships the bare ring's block
        return None, ""
    if front is None:
        return None, ""
    return _LEADING_BLOCK.sub("", stem, count=1), front
