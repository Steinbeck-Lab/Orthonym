"""Tier 4.4 of the substituent cascade (DT4 F2b): a fragment rooted on a fused ring system is that system's '-yl'
fusion prefix with its own prefixes and stereodescriptors ('cholest-5-en-3β-yl' as '(3S,...)-10,13-dimethyl-17-[(2R)-6-
methylheptan-2-yl]-2,3,4,7,8,9,10,11,12,13,14,15,16,17-tetradecahydro-1H-cyclopenta[a]phenanthren-3-yl').
(the Blue Book) the free valence; (:23710) a fusion name, not von Baeyer, for a system of two or more
rings of five or more members. Asked at every tier, before the general tiers' von Baeyer composition
(``substituent_enumerator._recursive_fragment_substituent_name``)."""
from __future__ import annotations

from typing import Optional


def fused_ring_system_prefix(mol, frag_atoms, attach_idx) -> Optional[str]:
    """``bridged_fused_pin.build_fused_decorated_substituent`` for a fragment rooted on a ring atom, the atoms outside
    the fragment as the parent side, or None (the builder declines, or fails)."""
    if attach_idx is None or attach_idx not in frag_atoms or not mol.GetAtomWithIdx(attach_idx).IsInRing():
        return None
    from ..rules.bridged_fused_pin import build_fused_decorated_substituent
    try:
        return build_fused_decorated_substituent(mol, attach_idx, set(range(mol.GetNumAtoms())) - set(frag_atoms))
    except Exception:  # noqa: BLE001 - a builder bug declines, the cascade goes on
        return None
