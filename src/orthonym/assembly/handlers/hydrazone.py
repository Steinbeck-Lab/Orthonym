"""a phase hydrazone handler — Tier B retained-name (gate 0.40).

Verbatim lift of the hydrazone dispatch logic from composer.py:785-794
(inline branch) + composer.py:1906-2042 (shared body of
_name_oxime_or_hydrazone with second arg ``'hydrazone'``). Per internal notes
 incremental-migration discipline, the body stays in composer.py
until Plan-03 commit 03-10 thinning.

IUPAC cite: (hydrazone functional class naming).

References:
- composer.py:785-794 (inline dispatch branch; REMOVED at this commit).
- composer.py:1906-2042 (shared with oxime; second arg dispatches naming).
- internal notes-DECOMP.md row 'hydrazone' + predicate purity proof.
- internal notes (Tier-B lift handler pattern).
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_hydrazone(features: Any) -> bool:
    """Mirrors composer.py:785 (``features.principal_group == 'hydrazone'``).

    Pure read-only per internal notes / -26.
    """
    return getattr(features, 'principal_group', None) == 'hydrazone'


def name_hydrazone(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """a phase Tier-B hydrazone handler.

    Verbatim semantics of composer.py:785-794 (inline branch). Returns
    ``NamingResult(name, tree=None, atom_to_locant_hint=None)`` on success;
    None on gate-fail (Pool's HANDLER_POLICIES['hydrazone'] gate 0.40).
    """
    # Lazy imports per PATTERNS § Lazy Import.
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _enrich_handler_name,
        _inject_stereo_if_missing,
        _name_oxime_or_hydrazone,
        _try_name_acylhydrazone,
        _try_name_hydrazone_substitutive,
        _try_name_semicarbazone,
    )

    # (W2E-P1FG Task 12): a semicarbazone R2C=N-NH-CO-NH2 is
    # perceived as principal_group='hydrazone' (the C=N-N arm) whose N-NH2
    # tail is acylated by a carbamoyl. Name it substitutively as
    # '2-(ylidene)hydrazine-1-carboxamide' BEFORE the generic hydrazone
    # functional-class rebuild (which drops the carbamoyl -> a different
    # molecule). Fail-closed -> falls through to the plain hydrazone path.
    hydrazone_name = _try_name_semicarbazone(features)
    # + (the Blue Book): an acylhydrazone R'2C=N-NH-C(=O)-R
    # is a hydrazide (the senior acyl group is the parent) with an N'-ylidene
    # substituent -> 'N'-({ylidene}){acyl}hydrazide', NOT an 'ylidene'-hydrazine
    # nor a functional-class hydrazone. Tried after the semicarbazone builder
    # (whose acyl-C-on-N motif this one excludes: acyl C on carbon) and before
    # the bare-hydrazine substitutive path (whose [NX3H2] guard the acylated
    # terminal N already fails). Fail-closed -> the substitutive/functional path.
    if not hydrazone_name:
        hydrazone_name = _try_name_acylhydrazone(features)
    # (W3-P15): a bare hydrazone R2C=N-NH2 -> the SUBSTITUTIVE PIN
    # ('propylidenehydrazine'), an 'ylidene' derivative of hydrazine (method (1)
    # = PIN), NOT the functional-class 'propanal hydrazone'. Tried after the
    # semicarbazone builder (whose N-CO-NH2 tail its [NX3H2] guard excludes) and
    # before the functional-class rebuild. Fail-closed -> functional-class path.
    if not hydrazone_name:
        hydrazone_name = _try_name_hydrazone_substitutive(features)
    if not hydrazone_name:
        hydrazone_name = _name_oxime_or_hydrazone(features, 'hydrazone')
    if not hydrazone_name:
        return None

    hydrazone_name = _enrich_handler_name(features, hydrazone_name, "hydrazone")

    pool = get_current_pool()
    cand = pool.add(hydrazone_name, "hydrazone", features)
    if cand is None:
        return None

    # Per composer.py:793 inline branch: wrap in _inject_stereo_if_missing
    # with atom_to_locant=None for byte-identical preservation per internal notes.
    final_name = _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="hydrazone", iupac_section_cite="P-68.3.1.2", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_hydrazone", "_is_hydrazone"]
