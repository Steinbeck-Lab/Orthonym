"""WSD-05 (Phase 175) nitrite-ester handler — P-67 functional-class naming.

A nitrite ester R-O-N=O is named in the two-word functional-class form
``<R> nitrite`` (P-67 / P-65.5; e.g. ``ethyl nitrite``), NOT as a C-nitroso
compound. This handler is the genuinely-missing emitter co-shipped with the
PERC-03 ``nitroso`` ``[#6]`` guard: without it, guarding ``nitroso`` would leave
``CCON=O`` nameless (there is no nitrite/nitrous seniority entry; the
``nitric acid -> nitrate`` mapping was deferred — resolvers.py:299).

Modeled on handlers/imidate.py (two-word functional-class, predicate-pure +
pool.add + stereo injection) and registered in inner_dispatch at the specialty
tier (priority 2960, after chalcogen_ester@2950).
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_nitrite_ester(features: Any) -> bool:
    """Predicate (D-07 pure): the nitrite ester R-O-N=O is the senior/sole
    characteristic group. Defers (False) when a higher-seniority PG is present,
    so the handler never claims a polyfunctional molecule where nitrite loses.
    NO mol/features mutation; NO module state."""
    fg = getattr(features, 'functional_groups', None) or {}
    if not fg.get('nitrite'):
        return False
    pg = getattr(features, 'principal_group', None)
    return pg in (None, 'nitrite')


def name_nitrite_ester(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Emit ``<R> nitrite`` (P-67) for a nitrite ester R-O-N=O.

    Algorithm: from the ``nitrite`` SMARTS match (R_carbon, O, N, O), name the
    R (alkyl/aryl) fragment via the universal substituent pipeline and join as
    ``"<R> nitrite"``. Returns None (defer) if the R cannot be named.
    Style is ignored (single PIN per compound, CONTEXT D-04).
    """
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing
    from ..substituent_enumerator import name_substituent

    matches = (getattr(features, 'functional_groups', None) or {}).get('nitrite', [])
    if not matches:
        return None
    if mol is None:
        mol = getattr(features, 'mol', None)
    if mol is None:
        return None

    # nitrite SMARTS = [#6][OX2][NX2]=[OX1] -> (R_carbon, ester_O, N, =O)
    match = matches[0]
    if len(match) < 4:
        return None
    r_carbon, ester_o = match[0], match[1]

    # R fragment = subgraph anchored at R_carbon, NOT crossing the ester oxygen.
    r_atoms = _collect_subgraph(mol, r_carbon, exclude={ester_o})
    if not r_atoms:
        return None
    try:
        r_word = name_substituent(mol, set(r_atoms), r_carbon)
    except Exception:
        r_word = None
    if not r_word:
        return None

    name = f"{r_word} nitrite"
    pool = get_current_pool()
    cand = pool.add(name, "nitrite_ester", features)
    if cand is None:
        return None

    final_name = _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem=final_name, class_id="nitrite_ester",
            iupac_section_cite="P-67", fragment_legacy=final_name,
        ),
        atom_to_locant_hint=None,
    )


def _collect_subgraph(mol: Any, anchor_idx: int, exclude: "set[int]") -> "tuple[int, ...]":
    """BFS subgraph from anchor, excluding given atom indices (pure, read-only)."""
    visited: "set[int]" = set()
    stack = [anchor_idx]
    while stack:
        idx = stack.pop()
        if idx in visited or idx in exclude:
            continue
        visited.add(idx)
        for bond in mol.GetAtomWithIdx(idx).GetBonds():
            other_idx = bond.GetOtherAtomIdx(idx)
            if other_idx not in visited and other_idx not in exclude:
                stack.append(other_idx)
    return tuple(sorted(visited))


__all__ = ["name_nitrite_ester", "_is_nitrite_ester"]
