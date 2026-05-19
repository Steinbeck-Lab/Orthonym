"""Phase 163.1 chalcogen-ester handler — Tier FRN-E functional-class naming.

Handles selenoester (R-C(=O)-Se-R') and telluroester (R-C(=O)-Te-R') per
IUPAC P-65.3 + P-65.6 functional-replacement nomenclature. Emits names of the
form "X-{alkyl} {chain}X{oate}" where X is "Se" or "Te":

    CC(=O)[Se]C        -> "Se-methyl ethaneselenoate"
    CCC(=O)[Se]C       -> "Se-methyl propaneselenoate"
    CCC(=O)[Se]CC      -> "Se-ethyl propaneselenoate"
    CCC(=O)[Te]C       -> "Te-methyl propanetelluroate"

Structurally mirrors handlers/imidate.py:
- Predicate consults features.principal_group so the handler defers to higher-
  seniority groups (acid / amide / etc.).
- Universal substituent enumeration for the alkyl side.
- Local longest-carbon-chain DFS + composer._integrate_universal_prefixes for
  the chain (acyl) side with branched/substituted support.

References:
- handlers/imidate.py (Tier FRN-D predicate pattern)
- handlers/isothiocyanate.py (Tier-B functional-class direct-return template)
- functional_groups.py selenoester / telluroester SMARTS (Plan-02 commit 98e46cbb)
- 163-AUDIT-FRN.md § 2.5 + § 5.3 + § 6 (audit baseline; Phase 163.1 closure
  ships the dedicated handler that was originally backlogged).
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NamingResult


# Map principal-group FG name -> ("X" prefix token, suffix token).
_CHALCOGEN_ESTER_TOKENS = {
    "selenoester":   ("Se", "selenoate"),
    "telluroester":  ("Te", "telluroate"),
}


def _is_chalcogen_ester(features: Any) -> bool:
    """Predicate: selenoester OR telluroester is the principal group.

    Mirrors handlers/imidate._is_imidate predicate contract. Fires when
    features.principal_group identifies a chalcogen ester AND no higher-
    seniority group outranks it. Side-effect-free per D-07.
    """
    pg = getattr(features, "principal_group", None)
    if pg not in _CHALCOGEN_ESTER_TOKENS:
        return False
    fg = getattr(features, "functional_groups", None) or {}
    return bool(fg.get(pg))


def name_chalcogen_ester(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Phase 163.1 selenoester / telluroester functional-class handler.

    Algorithm:
    1. Identify (C_carbonyl, O, X_chalcogen, C_alkyl) from the SMARTS match.
    2. Render the alkyl side via substituent_enumerator.name_substituent so
       branched/substituted alkyls (isopropyl, benzyl, ...) render correctly.
    3. Render the chain side by walking the longest carbon chain through
       C_carbonyl and enumerating branch substituents via
       composer._integrate_universal_prefixes (same path as anhydrides + ester
       chain renderers).
    4. Emit "{X}-{alkyl} {chain-prefix}{chain-stem}{X-suffix}".
    """
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _enrich_handler_name, _inject_stereo_if_missing,
        _integrate_universal_prefixes,
    )

    pg = features.principal_group
    tokens = _CHALCOGEN_ESTER_TOKENS.get(pg)
    if tokens is None:
        return None
    x_prefix, x_suffix = tokens

    matches = features.functional_groups.get(pg, [])
    if not matches:
        return None
    match = matches[0]
    if len(match) < 4:
        return None
    c_carbonyl_idx, o_idx, x_idx, c_alkyl_idx = match[0], match[1], match[2], match[3]

    if mol is None:
        mol = getattr(features, "mol", None)
    if mol is None:
        return None

    # Branch 1: alkyl word ({X}-{alkyl}) - render the R' on the chalcogen side.
    alkyl_atoms = _collect_subgraph(mol, c_alkyl_idx, exclude={x_idx})
    alkyl_word = _name_alkyl_side(mol, alkyl_atoms, anchor=c_alkyl_idx)
    if alkyl_word is None:
        return None

    # Branch 2: chain stem with {X-suffix} - render the acyl (chain) side.
    stem_word = _name_chalcogen_chain(
        mol, c_carbonyl_idx, exclude={o_idx, x_idx, c_alkyl_idx},
        x_suffix=x_suffix,
    )
    if stem_word is None:
        return None

    # Compose: "{X}-{alkyl} {chain}{X-suffix}" - e.g., "Se-methyl propaneselenoate".
    name = f"{x_prefix}-{alkyl_word} {stem_word}"
    name = _enrich_handler_name(features, name, "chalcogen_ester")

    pool = get_current_pool()
    cand = pool.add(name, "chalcogen_ester", features)
    if cand is None:
        return None

    final_name = _inject_stereo_if_missing(
        features, cand.name, atom_to_locant=None,
    )
    return NamingResult(name=final_name, tree=None, atom_to_locant_hint=None)


def _collect_subgraph(
    mol: Any, anchor_idx: int, exclude: "set[int]",
) -> "tuple[int, ...]":
    """BFS-walk a fragment from anchor, halting at excluded atoms. Pure."""
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


def _name_alkyl_side(
    mol: Any, atoms: "tuple[int, ...]", anchor: int,
) -> Optional[str]:
    """Render the {X}-alkyl-side via substituent_enumerator.name_substituent.

    Handles retained names (methyl, ethyl, isopropyl, benzyl, phenyl, ...) +
    systematic + recursive compound names automatically.
    """
    if not atoms:
        return None
    try:
        from ..substituent_enumerator import name_substituent
        return name_substituent(mol, set(atoms), anchor)
    except Exception:
        return None


def _name_chalcogen_chain(
    mol: Any, anchor_idx: int, exclude: "set[int]", x_suffix: str,
) -> Optional[str]:
    """Render the acyl (chain) side with {x_suffix} (selenoate / telluroate).

    Walks the longest carbon chain from C_carbonyl excluding the =O / chalcogen
    / alkyl-side atoms. Enumerates branch substituents via the composer's
    universal prefix integrator so branched + substituted chains render
    correctly (mirrors handlers/imidate._name_chain_with_imidate_suffix).
    """
    # Collect the candidate fragment (chain side atoms only).
    frag = _collect_subgraph(mol, anchor_idx, exclude)
    if not frag:
        return None

    principal_chain = _find_longest_carbon_chain(mol, anchor_idx, set(frag))
    chain_len = len(principal_chain)
    if chain_len < 1:
        return None

    # Chain prefix: methane / ethane / propane / butane / pentane / ...
    try:
        from ...data.chain_names import get_chain_prefix
        chain_prefix = get_chain_prefix(chain_len)
    except (ValueError, KeyError, ImportError):
        return None

    # IUPAC P-16.3.3: 'e' of '{prefix}ane' preserved because '-selenoate' /
    # '-telluroate' start with consonants ('s' / 't').
    base = f"{chain_prefix}ane{x_suffix}"

    chain_set = set(principal_chain)
    has_branches = any(
        sum(
            1 for n in mol.GetAtomWithIdx(idx).GetNeighbors()
            if n.GetIdx() in set(frag) and n.GetIdx() not in chain_set
        ) > 0
        for idx in principal_chain
    )
    if not has_branches:
        return base

    atom_to_locant = {idx: pos + 1
                      for pos, idx in enumerate(principal_chain)}
    all_atom_idxs = set(range(mol.GetNumAtoms()))
    exclude_atoms = all_atom_idxs - set(frag)

    try:
        from ..composer import _integrate_universal_prefixes
        prefix_str = _integrate_universal_prefixes(
            mol, chain_set,
            parent_type="chain",
            principal_chain=principal_chain,
            atom_to_locant=atom_to_locant,
            exclude_atoms=exclude_atoms,
        )
    except Exception:
        prefix_str = ""

    if prefix_str:
        return f"{prefix_str}{base}"
    return base


def _find_longest_carbon_chain(
    mol: Any, start: int, frag_atoms: "set[int]",
) -> list:
    """DFS longest simple carbon path from `start` within `frag_atoms`. Pure."""
    carbon_set = {
        i for i in frag_atoms
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 6
    }
    best_path = [start]

    def dfs(current: int, visited: "set[int]", path: list) -> None:
        nonlocal best_path
        if len(path) > len(best_path):
            best_path = list(path)
        for nbr in mol.GetAtomWithIdx(current).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in visited or nidx not in carbon_set:
                continue
            visited.add(nidx)
            path.append(nidx)
            dfs(nidx, visited, path)
            path.pop()
            visited.discard(nidx)

    dfs(start, {start}, [start])
    return best_path


__all__ = ["name_chalcogen_ester", "_is_chalcogen_ester"]
