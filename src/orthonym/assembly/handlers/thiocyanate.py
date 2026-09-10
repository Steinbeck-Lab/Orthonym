"""W3-P08 thiocyanate-ester handler — functional-class naming.

A thiocyanate ester R-S-C#N is named in the two-word functional-class form
``<R> thiocyanate``; e.g. ``propan-2-yl thiocyanate``).
Thiocyanic acid (HS-C#N) is a retained PIN (BBv2 L30987) and its esters are
functional-class PINs (BBv2 L32033 '(CH3)2CH-S-CN propan-2-yl thiocyanate
(PIN)').

The thiocyanate FG is detected but has no seniority principal-group entry (it is
a pseudohalide prefix), so ``principal_group`` is None and — without this
emitter — R-S-C#N reads as 'unknown' (the substitutive 'thiocyanato' prefix
path mis-builds and is self-consistency-suppressed).

CRITICAL — distinct from isothiocyanate: the isothiocyanate ester C-N=C=S has
its PIN as the substitutive ``isothiocyanato`` prefix and is NOT
handled here (different SMARTS/connectivity; ``fg['thiocyanate']`` never matches
it). Only S-C#N connectivity fires this handler.

Modeled on handlers/nitrite_ester.py (two-word functional-class, predicate-pure
+ pool.add + stereo injection) and registered in inner_dispatch at the specialty
tier (priority 2965, after nitrite_ester, before tier_a_ring).
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_thiocyanate(features: Any) -> bool:
    """Predicate (pure): the thiocyanate ester R-S-C#N is the sole
    characteristic group. Fires ONLY when principal_group is None so a
    polyfunctional molecule carrying a senior group (where thiocyanate is a
    plain prefix) is never hijacked. NO mol/features mutation; NO module state.
    """
    fg = getattr(features, 'functional_groups', None) or {}
    if not fg.get('thiocyanate'):
        return False
    pg = getattr(features, 'principal_group', None)
    return pg is None


def name_thiocyanate(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Emit ``<R> thiocyanate`` for a thiocyanate ester R-S-C#N.

    Algorithm: from the ``thiocyanate`` SMARTS match (S, C, N), take the S's
    carbon neighbour that is NOT the nitrile carbon (the organyl R), name that
    R fragment via the universal substituent pipeline, and join as
    ``"<R> thiocyanate"``. Returns None (defer / fail-closed) if the R cannot
    be named. Style is ignored (single PIN per compound, internal notes); the
    thiocyanate ester PIN IS functional-class, so it is NOT declined under 'pin'.
    """
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing
    from ..substituent_enumerator import name_substituent

    matches = (getattr(features, 'functional_groups', None) or {}).get('thiocyanate', [])
    if not matches:
        return None
    if mol is None:
        mol = getattr(features, 'mol', None)
    if mol is None:
        return None

    # thiocyanate SMARTS = [SX2][CX2]#[NX1] -> (S, nitrile_C, N).
    match = matches[0]
    if len(match) < 2:
        return None
    s_idx, nitrile_c = match[0], match[1]

    # Organyl carbon = the S neighbour that is not the nitrile carbon.
    organyl = None
    for nbr in mol.GetAtomWithIdx(s_idx).GetNeighbors():
        ni = nbr.GetIdx()
        if ni != nitrile_c and nbr.GetAtomicNum() > 1:
            organyl = ni
            break
    if organyl is None:
        return None

    # R fragment = subgraph anchored at the organyl carbon, NOT crossing the S.
    r_atoms = _collect_subgraph(mol, organyl, exclude={s_idx})
    if not r_atoms:
        return None
    try:
        r_word = name_substituent(mol, set(r_atoms), organyl)
    except Exception:
        r_word = None
    # Fail closed on an un-nameable organyl (name_substituent may return the
    # 'substituent' sentinel or a spaced junk string rather than None).
    if not r_word or r_word == "substituent" or " " in r_word:
        return None

    name = f"{r_word} thiocyanate"
    pool = get_current_pool()
    cand = pool.add(name, "thiocyanate", features)
    if cand is None:
        return None

    final_name = _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem=final_name, class_id="thiocyanate",
            iupac_section_cite="P-65.6.3.3.7.2", fragment_legacy=final_name,
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


__all__ = ["name_thiocyanate", "_is_thiocyanate"]
