"""a phase polycyclic handler — direct-return shim (LIFT body alongside).

Verbatim move of composer.py:1100-1114 dispatch logic. Wraps the inline
``_assemble_polycyclic_name(features, style)`` call + ``pool.add`` +
``pool.best.name`` return.

Per internal notes incremental migration: the underlying
``_assemble_polycyclic_name`` body STAYS in composer.py during Plan-02 and
moves to this module in Plan-03 commit 03-10 (composer.py thinning).

IUPAC cite: (polycyclic aromatics + retained names).

Byte-identical contract: this handler fires only when ``polycyclic_name``
is set AND complex_ring would NOT take the dispatch — encoded as
``not _is_complex_ring_system(features.mol)`` in the predicate to preserve
the inline gate ``not _complex_ring_accepted`` semantics. The cases where
complex_ring fires-and-rejects (low ratio) still fall through
to the inline fallback path further down composer.py.

References:
- composer.py:1100-1114 (inline polycyclic branch; REMOVED at this commit).
- composer.py:_assemble_polycyclic_name (STAYS until 03-10).
- composer.py:_is_complex_ring_system (mutex helper for the predicate).
- internal notes-DECOMP.md row 'polycyclic' + purity proof.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_polycyclic(features: Any) -> bool:
    """Predicate: polycyclic_name set AND not chain_is_parent AND not a complex ring system.

    The inline composer.py gate is nested under
    ``if not _complex_ring_accepted and not chain_is_parent`` and only fires
    when complex_ring did NOT short-circuit. We encode the conservative
    half of that gate here: when ``_is_complex_ring_system`` returns True,
    the inline complex_ring block at composer.py:1056+ handles the
    molecule (or falls through to the original inline polycyclic path,
    which is now NOT present because this handler replaces it). To preserve
    byte-identical behavior we require ``not _is_complex_ring_system`` —
    cases that would have fallen through after complex_ring rejected are
    captured by the legacy inline fallback at composer.py later in the
    cascade.
    """
    from ._handler_shared import cached_is_complex_ring_system

    if getattr(features, 'polycyclic_name', None) is None:
        return False
    if getattr(features, 'chain_is_parent', False):
        return False
    mol = getattr(features, 'mol', None)
    if mol is None:
        return False
    # A mancude ring system with a retained PAH name, 'pleiadene (PIN)':11360) is named on that parent even
    # when RDKit leaves a seven-membered ring of it non-aromatic and the system therefore counts as complex: the complex
    # path refuses such a cage (a von Baeyer name would saturate it).
    if _mancude_ring_system(mol):
        return True
    #: shared memoization with partial_sat / ring_ester predicates.
    if cached_is_complex_ring_system(features):
        return False
    return True


def _mancude_ring_system(mol) -> bool:
    """True when ``mol`` is one fused ring system, mancude, with a ring atom RDKit leaves non-aromatic.

    An all-aromatic system (a substituted naphthalene) and two systems joined by a single bond (a biaryl) are not
    this case and keep the complex-ring gate below.
    """
    from rdkit import Chem
    rings = mol.GetRingInfo().AtomRings()
    ring = {i for r in rings for i in r}
    if not ring or all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
        return False
    km = Chem.RWMol(mol)
    try:
        Chem.Kekulize(km, clearAromaticFlags=True)
    except Exception:  # noqa: BLE001 -- a structure RDKit cannot kekulize is not this case
        return False
    in_double = {i for b in km.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
                 and b.GetBeginAtomIdx() in ring and b.GetEndAtomIdx() in ring
                 for i in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())}
    if in_double != ring:
        return False
    seen, stack = {next(iter(ring))}, [next(iter(ring))]
    while stack:
        at = stack.pop()
        for b in mol.GetAtomWithIdx(at).GetBonds():
            nb = b.GetOtherAtomIdx(at)
            if b.IsInRing() and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen == ring


def name_polycyclic(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return polycyclic handler — wraps _assemble_polycyclic_name."""
    from ._handler_shared import pool_polycyclic_candidate

    polycyclic_name = getattr(features, 'polycyclic_name', None)
    if polycyclic_name is None:
        return None

    if logger.isEnabledFor(logging.DEBUG):
        _ha = features.mol.GetNumHeavyAtoms()
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
            "polycyclic", _ha, (polycyclic_name or "")[:60],
        )

    # composer.py:1114 inline: return pool.best.name (NO _inject_stereo wrap)
    _nm = pool_polycyclic_candidate(features, style)
    return NamingResult(
        name=_nm,
        tree=NameTreeNode(parent_stem=_nm, class_id="polycyclic", iupac_section_cite="P-25", fragment_legacy=_nm),
        atom_to_locant_hint=None,
    )


__all__ = ["name_polycyclic", "_is_polycyclic"]
