"""Phase 160 shared handler helpers (DECOMP-01 + CONTEXT D-08).

Substrate commit 02-00: lazy re-export wrappers around the canonical
implementations in ``composer.py``. Per CONTEXT D-24 incremental-migration
discipline, composer.py STILL OWNS:

* ``_name_iso_x_cyanate`` (composer.py:2204; 27 LOC) — shared between
  isocyanate (commit 02-04) and isothiocyanate (commit 02-05) handlers.
* ``_name_r_group`` (composer.py:2233; 217 LOC) — shared between urea
  (commit 02-08), guanidine (commit 02-09), and the general_acyclic
  catch-all (Plan-03 commit 03-09).

The substrate ships THIS module so handler files can write the
forward-looking import path::

    from ._handler_shared import name_iso_x_cyanate
    from ._handler_shared import name_r_group

while internally the symbols delegate (via lazy import inside each
function body) to composer.py. When Plan-03 commit 03-10 lands, the
function BODIES move here verbatim and composer.py's
``_name_iso_x_cyanate`` + ``_name_r_group`` definitions delete. The
re-export shape ensures handler files do NOT need to change import paths
at thinning time — only this delegation layer flips.

Per CONTEXT D-08 catch-all helper convention: shared logic between
multiple handlers MUST live in this module (not duplicated across
handlers/). The "≥ 2 handler" threshold is per CONTEXT D-03 +
160-AUDIT-DECOMP.md § 1.2 in-file-handler-body inventory.

Anti-pattern hygiene:
- AP-160-02 banned: do NOT group two handlers into one extraction commit
  because they share a body. The handlers are separate files; the SHARED
  helper lives HERE; each handler imports the helper but keeps its own
  file + own atomic commit.
- AP-160-12 banned: handler logic in shim (must be 1-3-line wrapper).
  This module's helpers ARE the multi-line logic; handlers import them.

References:
- composer.py:2204-2230 (``_name_iso_x_cyanate``) — verbatim source.
- composer.py:2233-2449 (``_name_r_group``) — verbatim source.
- 160-PATTERNS.md § "Common Conventions" + line 458.
- 160-CONTEXT.md D-08 + D-24 — catch-all helper convention + migration.
"""
from __future__ import annotations

from typing import Any, Optional


def name_iso_x_cyanate(
    features: Any, fg_key: str, suffix_word: str,
) -> Optional[str]:
    """Common implementation for isocyanate and isothiocyanate naming.

    Lazy delegate to ``composer.py:_name_iso_x_cyanate`` (composer.py:2204).
    Per CONTEXT D-24 + PATTERNS § 5 first-wave guidance, composer.py owns
    the canonical body at this commit; this wrapper provides the
    forward-looking import path ``handlers._handler_shared.name_iso_x_cyanate``
    for handler files that want stable paths now.

    SMARTS pattern: ``[#6][NX2]=[CX2]=[OX1]`` (isocyanate) or
    ``[#6][NX2]=[CX2]=[SX1]`` (isothiocyanate). Match tuple:
    ``(R_carbon, N, C, O/S)``.

    Args:
        features: MolecularFeatures object.
        fg_key: Either ``'isocyanate'`` or ``'isothiocyanate'``.
        suffix_word: The functional-class suffix word
            (``'isocyanate'`` / ``'isothiocyanate'``).

    Returns:
        Functional class name like ``'methyl isocyanate'``, or None.

    See Also:
        composer.py:_name_iso_x_cyanate — canonical implementation.
        composer.py:_name_isocyanate / _name_isothiocyanate — single-line
            callers; both move to handlers/{isocyanate,isothiocyanate}.py.
    """
    # Lazy import per PATTERNS § Lazy Import.
    from ..composer import _name_iso_x_cyanate
    return _name_iso_x_cyanate(features, fg_key, suffix_word)


def name_r_group(
    mol: Any, start_idx: int, exclude_atoms: set,
) -> Optional[str]:
    """Name an R group (substituent fragment) starting from start_idx.

    Lazy delegate to ``composer.py:_name_r_group`` (composer.py:2233).
    Per CONTEXT D-24 + PATTERNS § 5 first-wave guidance.

    Args:
        mol: RDKit Mol object.
        start_idx: Atom index of the R-group attachment point.
        exclude_atoms: Set of atom indices to exclude from the R-group walk
            (e.g., the functional group atoms already named).

    Returns:
        Substituent name as IUPAC P-29 substituent prefix (e.g., ``'methyl'``,
        ``'phenyl'``, ``'4-chlorophenyl'``), or None on failure.

    See Also:
        composer.py:_name_r_group — canonical implementation; consumed by
            urea, guanidine, isocyanate, isothiocyanate, and the
            general_acyclic catch-all.
    """
    # Lazy import per PATTERNS § Lazy Import.
    from ..composer import _name_r_group
    return _name_r_group(mol, start_idx, exclude_atoms)


def cached_is_complex_ring_system(features: Any) -> bool:
    """WR-02: per-features memoization of composer._is_complex_ring_system.

    The SMARTS-based complex-ring check is heavy; predicates that call it
    inside the dispatch loop violate the spirit of CONTEXT D-25 (predicates
    are pure read-only over already-perceived state). Cache the result on
    the features object as a private attribute so partial_sat / polycyclic /
    ring_ester predicates share a single SMARTS evaluation per features
    instance instead of running it three times per dispatch.

    The cache is per-features-instance state owned by features itself; the
    predicate remains pure with respect to shared/global state.

    Args:
        features: MolecularFeatures-like object with a ``mol`` attribute.

    Returns:
        True iff the molecule is a complex ring system per the SMARTS check.
    """
    cached = getattr(features, "_cached_complex_ring_system_result", None)
    if cached is not None:
        return cached
    from ..composer import _is_complex_ring_system
    result = bool(_is_complex_ring_system(features.mol))
    try:
        features._cached_complex_ring_system_result = result
    except (AttributeError, TypeError):
        # Frozen / immutable features: fall back to per-call computation.
        pass
    return result


__all__ = [
    "name_iso_x_cyanate",
    "name_r_group",
    "cached_is_complex_ring_system",
]
