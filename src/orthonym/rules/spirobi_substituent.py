"""A 'spirobi' ring system as a substituent prefix: ``9,9'-spirobi[fluoren]-2-yl``.

``### **** MONOSPIRO RING SYSTEMS CONTAINING TWO IDENTICAL POLYCYCLIC COMPONENTS`` (the Blue Book)
builds the ring system, (:10146): the prefix 'spirobi' before the component name in square
brackets, the spiro locants unprimed first; the numbering of the component is kept. The free valence is a
suffix ``yl`` of the parent hydride name (``****`` "General methodology for naming substituent groups",
:15800) and takes the lowest locant its numbering allows: "lowest locants are assigned to all free
valences as a set"; (c),:3244); the final 'e' of the component stem is elided where a suffix follows
('2,2'-spirobi[bicyclo[2.2.1]heptan]-5-ene (PIN)',:16771). The naming of the ring system itself is the one of
the parent (``composer._assemble_complex_ring_name``); this module numbers its free valence over the symmetry
of the ring system and writes the prefix. Declined (None, the caller keeps its path): a component with a
heteroatom, indicated hydrogen, hydro prefixes or a bracket of its own, or any ring system not named
'<locants>-spirobi[<hydrocarbon stem>]'.
"""
import re
from typing import Optional

#: the spirobi names of a mancude hydrocarbon component: '9,9'-spirobi[fluorene]'
_SPIROBI = re.compile(r"^(\d+[a-z]?,\d+[a-z]?')-spirobi\[([a-z]+)e\]$")


#: the prefix of each ring-only structure (the attachment atom marked) already written: an oligomer of
#: spirobi[fluorene] units asks for the same few structures dozens of times, and each is assembled and
#: numbered from scratch (a spirobi system assembles in about 70 ms); the prefix depends on the
#: isomorphism class of the ring system and of the atom that carries the free valence, and on nothing else
_WRITTEN: dict = {}
_WRITTEN_LIMIT = 512


def spirobi_substituent_name(sub, attach_sub) -> Optional[str]:
    """``9,9'-spirobi[fluoren]-2-yl`` for the ring system ``sub`` (a ring-only molecule) whose free
    valence is on the atom ``attach_sub``, or None."""
    from ..perception.rings import get_spiro_atoms
    if any(a.GetAtomicNum() != 6 for a in sub.GetAtoms()) or not get_spiro_atoms(sub):
        return None                                  # a spirobi system has a spiro atom
    from rdkit import Chem
    marked = Chem.RWMol(sub)
    marked.GetAtomWithIdx(attach_sub).SetAtomMapNum(1)
    key = Chem.MolToSmiles(marked)
    if key not in _WRITTEN:
        if len(_WRITTEN) >= _WRITTEN_LIMIT:
            _WRITTEN.clear()
        _WRITTEN[key] = _write(sub, attach_sub)
    return _WRITTEN[key]


def _write(sub, attach_sub) -> Optional[str]:
    from ..assembly.composer import _assemble_complex_ring_name, _primed_locant_sort_key
    from ..assembly.prefix_derivation import built
    from ..namer import compute_features
    try:
        result = _assemble_complex_ring_name(sub, compute_features(sub))
    except Exception:  # noqa: BLE001 -- a namer that raises declines the prefix
        return None
    if (result is None or not result.atom_to_locant
            or set(result.atom_to_locant) != {a.GetIdx() for a in sub.GetAtoms()}):
        return None
    shape = _SPIROBI.match(result.name)
    if shape is None:
        return None
    locants = result.atom_to_locant
    best = min((locants[m[attach_sub]] for m in sub.GetSubstructMatches(
        sub, uniquify=False, maxMatches=10000, useChirality=False)), key=_primed_locant_sort_key,
        default=None)
    base, primes = best if isinstance(best, tuple) else (best, "")
    if not isinstance(base, int):
        return None                                  # a fusion atom takes no free valence
    return built(f"{shape.group(1)}-spirobi[{shape.group(2)}]-{base}{primes}-yl", substituted=False)
