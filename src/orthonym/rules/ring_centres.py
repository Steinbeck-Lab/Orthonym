"""Ring-centre helpers of the charged and radical ring suffixes (``ions``): the ring system of a centre, and a ring
carbenium the cumulative-suffix path cannot number, named through its '-yl' group."""
import re

from rdkit import Chem


def ylium_as_ring_substituent(mol, center_idx: int) -> str:
    """A ring carbenium the in-place path cannot number, named through the ring's
    substituent group at the same atom: '<group>-<n>-yl' + 'ium'.

    : a cation formed by loss of a hydride is named with the suffix 'ylium'
    on the parent hydride, in the way the suffix 'yl' names the substituent group of the
    same atom ('9H-fluoren-9-yl', '9H-fluoren-9-ylium'). The in-place path has no
    numbering for a fused system outside its tables, or no neutral ring for a centre
    perceived as aromatic (``c1ccc2ccc[cH+]c-2cc1``); the ring-system substituent namer
    numbers it, so the cation is '1H-benzo[7]annulen-1-ylium', the localized fused form
    of ('1-methyl-5H-cyclohepta[b]pyridin-1-ium-5-ylium (PIN)', the Blue Book).
    The centre becomes a radical with its own hydrogen (the namers of
    ``radicals._placeholder_prefix`` name it). Returns '' unless the PIN route builds a
    locanted '-yl' group."""
    from .radicals import _placeholder_prefix
    work = Chem.RWMol(mol)
    try:
        h = work.GetAtomWithIdx(center_idx).GetTotalNumHs()
        Chem.Kekulize(work, clearAromaticFlags=True)
        a = work.GetAtomWithIdx(center_idx)
        a.SetFormalCharge(0)
        a.SetNumRadicalElectrons(1)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(h)
        radical = work.GetMol()
        Chem.SanitizeMol(radical)
        prefix, from_general = _placeholder_prefix(radical, center_idx, general=False)
    except (RuntimeError, ValueError):
        return ''
    if not prefix or from_general or not re.search(r'-\d+[a-z]?-yl$', prefix):
        return ''
    if 'hydro' in prefix:
        #: the cation of a mancude ring at a fusion atom takes added hydrogen ('naphthalen-4a(8aH)-ylium'),
        # where the '-yl' group takes hydro prefixes ('4a,8a-dihydronaphthalen-4a-yl'): not the same parent
        return ''
    return prefix + 'ium'


def _ring_system_of(mol, center_idx: int):
    from ..perception.rings import get_ring_systems
    try:
        for rs in get_ring_systems(mol, include_spiro=True):
            if center_idx in rs:
                return rs
    except (RuntimeError, ValueError):
        return None
    return None
