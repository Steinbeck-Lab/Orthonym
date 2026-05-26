"""Phase 160 N-oxide handler — direct-return; recursive name_compound() permitted.

Verbatim lift of the N-oxide dispatch logic from composer.py:813-820
(inline branch) + composer.py:2044-2174 (body of _try_name_n_oxide +
_name_aromatic_n_oxide + _name_aliphatic_n_oxide). Per CONTEXT D-24,
the body stays in composer.py until Plan-03 commit 03-10 thinning.

IUPAC cite: P-62.5 (N-oxides; functional class naming).

Special case (per CONTEXT D-25 + 160-AUDIT-DECOMP.md § 2.4):
- _try_name_n_oxide() makes a recursive name_fragment_recursively() call
  on a reduced (N-oxide → parent amine) RWMol copy. The recursion is
  PERMITTED because:
  (a) the recursive call instantiates a FRESH push_pool/pop_pool lifecycle
      per Phase 145.1 D-09;
  (b) RWMol mutation operates on a COPY (Chem.RWMol(features.mol)),
      NOT the input features.mol;
  (c) the recursive name_fragment_recursively() runs in an isolated
      pool scope; no cross-handler shared state leaks.
- side_effect_inventory remains () per D-25 hard invariant (recursion
  is not a side effect on the outer features / outer pool).

Predicate strategy (per CONTEXT D-25):
- The predicate checks ``features.functional_groups.get('n_oxide_aromatic')``
  or ``features.functional_groups.get('n_oxide_aliphatic')`` (the keys
  populated by perception/functional_groups.py:70-71). This filters out
  non-N-oxide molecules at the predicate layer so dispatch_inner can
  proceed to the next priority entry. The handler body still calls
  ``_try_name_n_oxide(features)`` which re-runs the SMARTS pattern AND
  applies the is_top_level_naming() guard + reduced-molecule recursion;
  the handler returns None if those guards fail (e.g., during fragment
  naming recursion), in which case the dispatch loop's caller (composer.py)
  falls through to the inline cascade.

References:
- composer.py:813-820 (inline dispatch branch; REMOVED at this commit).
- composer.py:2044-2174 (_try_name_n_oxide + aromatic/aliphatic helpers).
- 160-AUDIT-DECOMP.md § 1 row 'n_oxide' + § 2.4 (recursive call audit).
- 160-PATTERNS.md § 7 (n_oxide direct-return shape).
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_n_oxide(features: Any) -> bool:
    """Predicate: features has an aromatic or aliphatic N-oxide SMARTS match.

    Cheap-check against ``features.functional_groups`` (populated by
    perception/functional_groups.py:70-71 for SMARTS keys 'n_oxide_aromatic'
    and 'n_oxide_aliphatic'). The full N-oxide naming logic
    (_try_name_n_oxide at composer.py:2044-2174) is hosted in the handler
    body; this predicate just gates on the perception-flag presence so
    dispatch_inner can proceed to the next priority entry for non-N-oxide
    molecules.

    Pure read-only per CONTEXT D-25 / AP-160-26: reads
    ``features.functional_groups`` dict (set by perception layer before
    assembly); no writes.
    """
    fg = getattr(features, 'functional_groups', None) or {}
    return bool(fg.get('n_oxide_aromatic')) or bool(fg.get('n_oxide_aliphatic'))


def name_n_oxide(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Phase 160 direct-return N-oxide handler.

    Verbatim semantics of composer.py:813-820 (inline branch). Returns
    ``NamingResult(name, tree=None, atom_to_locant_hint=<heterocycle locant
    map>)`` on success; None when not an N-oxide.

    Per 160-AUDIT-DECOMP.md § 2.4: the recursive name_fragment_recursively
    call inside _try_name_n_oxide is PERMITTED under D-25 because it
    instantiates a fresh push_pool/pop_pool lifecycle per Phase 145.1 D-09
    and operates on a COPY of features.mol (no outer mutation).
    """
    # Lazy imports per PATTERNS § Lazy Import.
    from ..candidate_pool import get_current_pool
    from ..composer import _try_name_n_oxide, _inject_stereo_if_missing

    n_oxide_name = _try_name_n_oxide(features)
    if not n_oxide_name:
        return None

    # The heterocycle locant map (composer.py:816) becomes the
    # atom_to_locant for _inject_stereo_if_missing.
    locant_map = (
        getattr(features, 'heterocycle_atom_to_locant', None)
        or features.atom_to_locant
    )

    # Phase 145.1: direct_return handler routes through pool.add()
    # (composer.py:818-819 inline equivalent).
    pool = get_current_pool()
    cand = pool.add(n_oxide_name, "n_oxide", features)
    # WR-03: use the candidate returned by pool.add() instead of pool.best()
    # to prevent silently returning a DIFFERENT handler's name when this
    # handler's candidate is rejected by the gate. Matches sibling pattern
    # in oxime.py. Per CONTEXT D-13: handlers own the name string they
    # place in NamingResult.name.
    name_to_inject = cand.name if cand is not None else n_oxide_name

    # Per composer.py:820 inline branch: wrap in _inject_stereo_if_missing
    # with the heterocycle locant_map for byte-identical preservation.
    final_name = _inject_stereo_if_missing(
        features, name_to_inject, atom_to_locant=locant_map,
    )
    _nm = final_name
    return NamingResult(
        name=_nm,
        tree=NameTreeNode(parent_stem=_nm, class_id="n_oxide", iupac_section_cite="P-74.2", fragment_legacy=_nm),
        atom_to_locant_hint=locant_map,
    )


__all__ = ["name_n_oxide", "_is_n_oxide"]
