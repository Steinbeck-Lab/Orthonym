"""N-substituted sulfonamide handler — P-66.1.1.3.1.1.

Mirrors ``handlers/hydroximic_acid.py``: a thin dispatch shim over a rule module
that builds the COMPLETE name and refuses rather than skipping any part of it.
The rule, the Blue Book derivation and the measured class boundary all live in
``rules/sulfonamides.py``.

Priority 5213 — immediately after ``sulfonimidic_n_hydroxy``@5212 and before
``amine``@5300, so an N-substituted sulfonamide is claimed before the amine
path can treat its nitrogen as an amine parent. Fail-safe: returns None, so
``general_acyclic``@99999 remains the backstop.

Why this is NOT enriched: ``n_substituted_sulfonamide_name`` already spells the
parent AND every N-substituent, so running the generic enricher over it could
only re-discover the handler's own core and spell an atom twice — the second
anti-pattern recorded in
``.planning/audit-v29/FINDING-count-based-naming-sites.md``. Same reasoning as
``handlers/thiourea.py``.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

_N_SUBSTITUTED_SULFONAMIDE_PGS = (
    'secondary_sulfonamide',
    'tertiary_sulfonamide',
)


def _is_n_substituted_sulfonamide(features: Any) -> bool:
    """Principal group is an N-substituted sulfonamide.

    ``primary_sulfonamide`` is deliberately absent: the unsubstituted parent is
    already named correctly by the existing suffix paths
    (``methanesulfonamide``, ``benzenesulfonamide``) and this handler must not
    compete with them.
    """
    return (getattr(features, 'principal_group', None)
            in _N_SUBSTITUTED_SULFONAMIDE_PGS)


def name_n_substituted_sulfonamide(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Tier-B N-substituted sulfonamide handler (P-66.1.1.3.1.1)."""
    from ...rules.sulfonamides import n_substituted_sulfonamide_name
    from ..candidate_pool import get_current_pool

    _mol = mol if mol is not None else getattr(features, 'mol', None)
    if _mol is None:
        return None

    name = n_substituted_sulfonamide_name(_mol, style=style)
    if not name:
        return None

    pool = get_current_pool()
    cand = pool.add(name, "n_substituted_sulfonamide", features)
    if cand is None:
        return None

    return NamingResult(
        name=cand.name,
        tree=NameTreeNode(
            parent_stem=cand.name,
            class_id="n_substituted_sulfonamide",
            iupac_section_cite="P-66.1.1.3.1.1",
            fragment_legacy=cand.name,
        ),
        atom_to_locant_hint=None,
    )


__all__ = ["name_n_substituted_sulfonamide", "_is_n_substituted_sulfonamide"]
