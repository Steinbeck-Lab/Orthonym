""" handler enrichment helpers.

Substrate: lazy re-export wrappers around the canonical
implementations in ``composer.py``. Per CONTEXT incremental-migration
discipline, composer.py STILL OWNS ``_enrich_handler_name`` (composer.py:191-277)
and ``_integrate_universal_prefixes`` (composer.py:109-183) at this commit;
those functions stay until (composer.py thinning).

The substrate ships THIS module so handler files can write the
forward-looking import path::

    from.._enrichment import enrich_handler_name # eventual public path

while internally the symbols delegate (via lazy import inside each
function body) to composer.py. When lands, the
function BODIES move here verbatim and composer.py's `_enrich_handler_name`
+ `_integrate_universal_prefixes` definitions delete. The re-export
shape ensures handler files do NOT need to change import paths at thinning
time — only this delegation layer flips.

Per PATTERNS § 5 line 474 first-wave guidance: both lazy-from-composer
AND eventual-handlers/_enrichment paths work. We choose the lazy-from-
composer path now to minimize commit-02-00 risk: zero copy of composer.py
logic; zero risk of stale-closure drift; the byte-identical canary delta
gate is trivially satisfied.

Anti-pattern hygiene (inheritance):
- banned: pure read-only on ``features``; never mutates
  ``features.mol`` or ``features.functional_groups``. (The lazy delegate
  inherits composer.py's purity verbatim.)
- banned: no regex band-aid / postprocessor on inner-dispatch
  output — the helpers are pure wrappers.
- ``logger.debug`` for HANDLER_COVERAGE telemetry; default-OFF.
- IUPAC P-31.1 cite stays in ``_integrate_universal_prefixes`` docstring
  in composer.py (unchanged at this commit).

References:
- composer.py:109-183 (``_integrate_universal_prefixes``) — verbatim source.
- composer.py:191-277 (``_enrich_handler_name``) — verbatim source.
- 160-PATTERNS.md § 5 — analog: composer.py:109-277.
- 160-CONTEXT.md — incremental-migration discipline.
"""
from __future__ import annotations

from typing import Any, Optional


def enrich_handler_name(
    features: Any, base_name: str, handler_id: str = "unknown",
    atom_to_locant: Optional[Any] = None,
) -> str:
    """Enrich a handler's base name with non-principal substituents.

    Lazy delegate to ``composer.py:_enrich_handler_name`` (composer.py:191-277).
    Per CONTEXT + PATTERNS § 5 first-wave guidance, composer.py owns
    the canonical body at this commit; this wrapper provides the
    forward-looking import path ``handlers._enrichment.enrich_handler_name``
    for handler files that want stable paths now.

    Args:
        features: MolecularFeatures object.
        base_name: The handler's base name (e.g., "carbamic acid").
        handler_id: Handler identifier for logging.
        atom_to_locant: Optional producer-supplied ``{atom idx -> locant}``
            numbering — THE one ``base_name`` was spelled from. Overrides the
            ``features``-derived fallback; ``None`` keeps existing behaviour.

    Returns:
        Enriched name with prefixes, or base_name if no enrichment needed.

    See Also:
        composer.py:_enrich_handler_name — canonical implementation;
            moves here verbatim at (composer.py thinning).
    """
    # Lazy import per PATTERNS § Lazy Import (avoid composer.py -> handlers
    # -> composer.py cycle at module load).
    from ..composer import _enrich_handler_name
    return _enrich_handler_name(
        features, base_name, handler_id, atom_to_locant=atom_to_locant,
    )


def integrate_universal_prefixes(
    mol: Any,
    parent_atoms: Any,
    parent_type: str = "auto",
    oriented_ring: Optional[Any] = None,
    principal_chain: Optional[Any] = None,
    atom_to_locant: Optional[Any] = None,
    ring_atom_to_locant: Optional[Any] = None,
    exclude_atoms: Optional[Any] = None,
) -> str:
    """Discover and format all substituents on a parent structure.

    Lazy delegate to ``composer.py:_integrate_universal_prefixes``
    (composer.py:109-183). Per CONTEXT + PATTERNS § 5 first-wave guidance.

    Args:
        mol: RDKit Mol object.
        parent_atoms: Set of atom indices defining the parent structure.
        parent_type: ``"ring"``, ``"chain"``, or ``"auto"`` (auto-detects).
        oriented_ring: Ring atom indices in IUPAC order (for ring parents).
        principal_chain: Chain atom indices in order (for chain parents).
        atom_to_locant: Optional mapping of atom idx -> IUPAC locant (CHAIN
            parents only).
        ring_atom_to_locant: Optional inherited mapping for a RING parent —
            overrides the ``oriented_ring`` position arithmetic per atom.
        exclude_atoms: Atoms already accounted for.

    Returns:
        Prefix string ready to prepend to the handler's core name; empty
        string if no substituents are found.

    See Also:
        composer.py:_integrate_universal_prefixes — canonical implementation.
    """
    # Lazy import per PATTERNS § Lazy Import.
    from ..composer import _integrate_universal_prefixes
    return _integrate_universal_prefixes(
        mol, parent_atoms,
        parent_type=parent_type,
        oriented_ring=oriented_ring,
        principal_chain=principal_chain,
        atom_to_locant=atom_to_locant,
        ring_atom_to_locant=ring_atom_to_locant,
        exclude_atoms=exclude_atoms,
    )


__all__ = [
    "enrich_handler_name",
    "integrate_universal_prefixes",
]
