"""Phase 163 imidate handler — Tier-D iminoester functional-class naming.

IUPAC cite: P-65.1.7 (imidic acids and imidates; "alkyl alkanimidate"
functional-class naming for R-C(=NH)-O-R').

Mirrors handlers/isothiocyanate.py structurally: predicate-pure + direct-
return + pool.add() + _inject_stereo_if_missing wrapping.

References:
- handlers/isothiocyanate.py (Tier-B retained-name handler; structural template per CONTEXT D-03)
- functional_groups.py iminoester SMARTS (Phase 163 Plan-02 commit 163-02-04 addition)
- 163-AUDIT-FRN.md § 6 (per-fixture spec + intercept analysis + predicate purity proof)
- 163-AUDIT-FRN.md § 7 (INNER_DISPATCH priority 2900 LOCK)
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NamingResult


def _is_imidate(features: Any) -> bool:
    """Predicate: iminoester FG present (D-07 predicate-pure).

    Reads features.functional_groups.get('iminoester', []); returns bool(...).
    Does NOT consult features.principal_group because iminoester is a
    FUNCTIONAL-CLASS handler (it bypasses principal-group seniority cascade
    for compounds where iminoester is the sole defining feature).

    Per D-07 + Phase 158 D-26 + Phase 160 D-25 hard invariant: NO mol
    mutation; NO features mutation; NO module-global state R/W; NO
    exception swallowing.
    """
    fg = getattr(features, 'functional_groups', None) or {}
    return bool(fg.get('iminoester'))


def name_imidate(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Phase 163 Tier-D iminoester functional-class handler.

    Emits "alkyl alkanimidate" PIN per IUPAC P-65.1.7. Style parameter is
    ignored per CONTEXT D-04 (single PIN per compound).

    Algorithm (per AUDIT § 6):
    1. Get iminoester atom-match: (C_carbonyl, =NH, O, C_alkyl) tuple.
    2. Identify the alkyl word (R'): atoms reachable from C_alkyl
       excluding the C_carbonyl -> O path. Generate substituent name via
       carbon-count lookup.
    3. Identify the chain stem (R): atoms reachable from C_carbonyl
       excluding =NH and =O paths. Generate chain stem with -imidate suffix.
    4. Emit two-word name: "{alkyl} {stem}imidate".

    Returns None if either branch fails (gate-fail per ADR-19-04 contract;
    dispatch_inner cascades to next entry).
    """
    from ..candidate_pool import get_current_pool
    from ..composer import _enrich_handler_name, _inject_stereo_if_missing

    matches = features.functional_groups.get('iminoester', [])
    if not matches:
        return None

    # SMARTS match: (C_carbonyl, =NH, O, C_alkyl) per iminoester SMARTS
    # [CX3](=[NX2H1])[OX2][#6] - atom_indices order matches SMARTS atom order
    c_carbonyl_idx, nh_idx, o_idx, c_alkyl_idx = matches[0]

    if mol is None:
        mol = getattr(features, 'mol', None)
    if mol is None:
        return None

    # Branch 1: alkyl word (R')
    # Collect subgraph from C_alkyl excluding the O atom (which separates
    # the alkyl side from the stem side)
    alkyl_atoms = _collect_subgraph(mol, c_alkyl_idx, exclude={o_idx})
    alkyl_word = _name_alkyl_fragment(mol, alkyl_atoms, anchor=c_alkyl_idx)
    if alkyl_word is None:
        return None

    # Branch 2: chain stem with -imidate suffix
    # Collect subgraph from C_carbonyl excluding =NH and =O paths
    stem_atoms = _collect_subgraph(mol, c_carbonyl_idx,
                                   exclude={nh_idx, o_idx})
    # Chain length includes c_carbonyl (it's the locant-1 carbon of the stem)
    stem_word = _name_chain_with_imidate_suffix(mol, stem_atoms,
                                                anchor=c_carbonyl_idx)
    if stem_word is None:
        return None

    # Compose two-word name: "{alkyl} {stem}imidate"
    name = f"{alkyl_word} {stem_word}"
    name = _enrich_handler_name(features, name, "imidate")

    # Pool insertion + stereo injection (mirror isothiocyanate.py)
    pool = get_current_pool()
    cand = pool.add(name, "imidate", features)
    if cand is None:
        return None

    final_name = _inject_stereo_if_missing(features, cand.name,
                                           atom_to_locant=None)
    return NamingResult(name=final_name, tree=None, atom_to_locant_hint=None)


def _collect_subgraph(mol: Any, anchor_idx: int,
                      exclude: "set[int]") -> "tuple[int, ...]":
    """BFS subgraph collection from anchor, excluding given atom indices.

    PURE per D-07: read-only mol traversal; no mutation.
    """
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


def _name_alkyl_fragment(mol: Any, atoms: "tuple[int, ...]",
                         anchor: int) -> Optional[str]:
    """Generate the alkyl-word for the R'-O- side of the imidate.

    Maps the alkyl carbon-count to retained alkyl name (methyl/ethyl/propyl/etc.)
    or to the systematic alkyl name for aromatic/non-linear cases.

    Per AUDIT § 6: FRN-D-01..06 baseline supports linear alkyl 1-5 carbons
    + phenyl (FRN-D-04 methyl benzimidate uses 'benzimidate' stem on the
    other side; the alkyl word is always linear alkyl in the audit corpus).

    PURE per D-07: read-only mol queries; no mutation.
    """
    # Aromatic check: 6 aromatic C carbons -> phenyl
    n_aromatic_c = sum(
        1 for idx in atoms
        if mol.GetAtomWithIdx(idx).GetIsAromatic()
        and mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
    )
    if n_aromatic_c == 6:
        # Pure aromatic ring on the alkyl side -> phenyl
        # (rare in iminoester corpus but supported for safety)
        return "phenyl"

    # Count non-aromatic carbon atoms in the fragment
    n_carbons = sum(
        1 for idx in atoms
        if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
        and not mol.GetAtomWithIdx(idx).GetIsAromatic()
    )

    # Linear alkyl retained names per IUPAC P-29.2 (chain prefixes from
    # orthonym.data.chain_names get_alkyl_name() also work, but for the
    # audit corpus baseline (1-5 carbons) the inline map is faster and
    # avoids a circular-import risk)
    alkyl_names = {1: "methyl", 2: "ethyl", 3: "propyl", 4: "butyl", 5: "pentyl"}
    if n_carbons in alkyl_names:
        return alkyl_names[n_carbons]

    # Fall back to the canonical alkyl-name lookup for longer chains.
    if n_carbons > 0:
        try:
            from ...data.chain_names import get_alkyl_name
            return get_alkyl_name(n_carbons)
        except (ValueError, KeyError, ImportError):
            return None

    return None  # Empty fragment — out of baseline scope


def _name_chain_with_imidate_suffix(mol: Any, atoms: "tuple[int, ...]",
                                    anchor: int) -> Optional[str]:
    """Generate the chain-stem-with-imidate-suffix for the R-C(=N)- side.

    Maps the stem carbon-count to "{stem}imidate" form:
    - 2C -> "acetimidate" (retained per IUPAC P-65.1.7 + RESEARCH §6.4 FRN-D-02)
    - 3C -> "propanimidate"
    - 4C -> "butanimidate"
    - 5C -> "pentanimidate"
    - 6C (aromatic ring) -> "benzimidate" (retained per RESEARCH §6.4 FRN-D-04)

    PURE per D-07: read-only mol queries; no mutation.
    """
    # Aromatic-ring case: phenyl ring (6 aromatic C) plus the anchor C
    # gives 7 C atoms total -> "benzimidate"
    n_aromatic_c = sum(
        1 for idx in atoms
        if mol.GetAtomWithIdx(idx).GetIsAromatic()
        and mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
    )
    n_carbons = sum(
        1 for idx in atoms
        if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
    )
    # 6 aromatic ring C + 1 anchor C = 7 total
    if n_aromatic_c == 6 and n_carbons == 7:
        return "benzimidate"

    # Linear chain case: {alkan}imidate per IUPAC P-65.1.7
    stem_map = {
        2: "acetimidate",       # PIN per IUPAC P-65.1.7 (acetimidate retained)
        3: "propanimidate",
        4: "butanimidate",
        5: "pentanimidate",
        6: "hexanimidate",
    }
    if n_carbons in stem_map:
        return stem_map[n_carbons]

    # Fall back to chain-name + 'imidate' suffix for longer chains.
    if n_carbons > 1:
        try:
            from ...data.chain_names import get_chain_prefix
            # P-29.2 + P-65.1.7: ethanimidate / propanimidate / butanimidate ...
            return get_chain_prefix(n_carbons) + "animidate"
        except (ValueError, KeyError, ImportError):
            return None

    return None  # Out of baseline scope


__all__ = ["name_imidate", "_is_imidate"]
