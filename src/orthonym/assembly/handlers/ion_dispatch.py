"""a phase ion_dispatch handler — pre-pool bypass wrapper.

Per internal notes, ion / salt / zwitterion / radical species use a
PRE-POOL bypass at composer.py:751-768. The pre-pool call site STAYS
in composer.py (lines 751-768); only the body of ``assemble_ion_name``
is hosted here as ``name_ion_dispatch``.

This module hosts 4 sub-types in ONE file per internal notes (a single
``ion_dispatch`` HANDLER_POLICIES key spans salt + zwitterion + radical +
ion). All four sub-types share the same body (the inner switch on
``features.species_type`` is preserved verbatim).

IUPAC cites:
- / / + / (salts: cation word(s) + anion)
  (: corrected 169.6-04 — the old conjunctive-nomenclature cite was wrong)
- (zwitterions)
- (radicals)
- + (anions / cations)

Byte-identical contract: the inline pre-pool call site at composer.py:
751-768 stays IDENTICAL — those 4 ``if species_type...`` blocks are the
actual control-flow paths in production. The inner-dispatch registration
of ion_dispatch at priority 50 is structurally complete (so future
architectural reviewers see ion_dispatch in the table) but unreachable
in practice because the inline bypass runs FIRST.

References:
- composer.py:751-768 (inline pre-pool call site; PRESERVED per internal notes).
- composer.py:assemble_ion_name (body STAYS until Plan-03 commit 03-10).
- internal notes-DECOMP.md row 'ion_dispatch' + purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NamingResult


def _is_ion_dispatch(features: Any) -> bool:
    """Predicate: species_type in (salt, zwitterion, radical, ion).

    Mirrors composer.py:751 / 756 / 763 inline bypass blocks. The
    ``not _composing_ion`` clause from the inline-ion check at composer.py:
    763 is NOT encoded here because ``_composing_ion`` is a function-local
    kwarg on ``_assemble_name_impl``, not a feature attribute. inner-dispatch
    callers from ion-composition recursion paths bypass this handler
    via the inline guard (they enter via `_composing_ion=True` which routes
    through the neutral path).
    """
    species_type = getattr(features, 'species_type', 'neutral')
    return species_type in ('salt', 'zwitterion', 'radical', 'ion')


def name_ion_dispatch(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Pre-pool ion / salt / zwitterion / radical naming dispatcher.

    Wraps ``composer.assemble_ion_name`` per internal notes. The inline
    bypass at composer.py:751-768 is the primary call path; this handler
    function exists as a callable target so the architecture is uniform
    across all HANDLER_POLICIES keys.
    """
    from ..composer import assemble_ion_name

    name = assemble_ion_name(features, features.mol, style)
    if not name:
        return None
    return NamingResult(name=name, tree=None, atom_to_locant_hint=None)


__all__ = ["name_ion_dispatch", "_is_ion_dispatch"]
