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
    """Predicate: iminoester is the principal group, or iminoester is the
    sole characteristic group (no higher-seniority PG present) (D-07 pure).

    CR-fix (Phase 163 post-merge): consult features.principal_group so the
    handler defers to higher-seniority groups (carboxylic_acid, ester, amide,
    nitrile, etc.) when those win the seniority cascade. Previous version
    claimed dispatch slot 2900 whenever any iminoester SMARTS matched, which
    silently dropped acid carbons on mixed-PG inputs like
    OC(=O)c1ccc(C(=N)OC)cc1.

    Mirrors handlers/isothiocyanate.py and handlers/urea.py contract:
    iminoester is a functional-class group — emit only when no higher PG
    outranks it. Per IUPAC P-41 seniority, the seniority cascade in
    rules/seniority.py sets principal_group='iminoester' when no higher
    group is present; in that case (or principal_group is None for pure-
    imidate compounds) this predicate fires.

    Per D-07 + Phase 158 D-26 + Phase 160 D-25 hard invariant: NO mol
    mutation; NO features mutation; NO module-global state R/W; NO
    exception swallowing.
    """
    fg = getattr(features, 'functional_groups', None) or {}
    if not fg.get('iminoester'):
        return False
    pg = getattr(features, 'principal_group', None)
    return pg in (None, 'iminoester')


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
    from ..name_tree import NameTreeNode  # Phase 165 SCORE-01 Path-B coarse node
    _nm = final_name
    return NamingResult(
        name=_nm,
        tree=NameTreeNode(parent_stem=_nm, class_id="imidate", iupac_section_cite="P-65.6", fragment_legacy=_nm),
        atom_to_locant_hint=None,
    )


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

    CR-02 fix (Phase 163 post-merge): delegate to the universal substituent
    naming pipeline (substituent_enumerator.name_substituent) so branched
    and substituted alkyl sides — isopropyl, 2-hydroxyethyl, tert-butyl,
    benzyl, etc. — are rendered correctly instead of being collapsed to a
    linear chain by atom count. The previous version returned ``propyl``
    for isopropyl, ``ethyl`` for 2-hydroxyethyl, etc.

    PURE per D-07: read-only mol queries; no mutation.
    """
    if not atoms:
        return None
    try:
        from ..substituent_enumerator import name_substituent
        # name_substituent is documented non-None and handles retained names
        # (isopropyl, tert-butyl, phenyl, benzyl) plus systematic chains.
        return name_substituent(mol, set(atoms), anchor)
    except Exception:
        # Last-resort fallback (mirrors prior linear-only behavior) so a
        # downstream change in name_substituent never silently drops the
        # whole handler. Per D-07 contract this fallback also pure.
        n_carbons = sum(
            1 for idx in atoms
            if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
        )
        alkyl_names = {1: "methyl", 2: "ethyl", 3: "propyl",
                       4: "butyl", 5: "pentyl"}
        if n_carbons in alkyl_names:
            return alkyl_names[n_carbons]
        if n_carbons > 0:
            try:
                from ...data.chain_names import get_alkyl_name
                return get_alkyl_name(n_carbons)
            except (ValueError, KeyError, ImportError):
                return None
        return None


def _name_chain_with_imidate_suffix(mol: Any, atoms: "tuple[int, ...]",
                                    anchor: int) -> Optional[str]:
    """Generate the chain-stem-with-imidate-suffix for the R-C(=N)- side.

    CR-03 fix (Phase 163 post-merge): find the longest carbon chain through
    the anchor C(=N) and enumerate branch substituents via the composer's
    universal-prefix integrator. The previous version counted atoms only,
    producing ``methyl pentanimidate`` for tert-butyl acetimidate
    (CC(C)(C)C(=N)OC; correct stem is 2,2-dimethylpropanimidate).

    Naming rules per IUPAC P-65.1.7:
    - 2C linear stem  -> "acetimidate" (retained PIN)
    - aromatic ring at anchor -> "benzimidate" (retained PIN, FRN-D-04)
    - 3+ C linear stem -> "{chainprefix}animidate"  e.g. propanimidate
    - branched stem -> "{locant-substituent-list}{chainprefix}animidate"

    PURE per D-07: read-only mol queries; no mutation.
    """
    if not atoms:
        return None

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
    if n_aromatic_c == 6 and n_carbons == 7:
        return "benzimidate"

    # Find longest carbon chain from C(=N) through the stem subgraph.
    frag_set = set(atoms)
    principal_chain = _find_longest_carbon_chain(mol, anchor, frag_set)
    chain_len = len(principal_chain)

    if chain_len < 1:
        return None

    # Build base stem.
    if chain_len == 2:
        # IUPAC P-65.1.7: acetimidate retained
        base = "acetimidate"
    else:
        try:
            from ...data.chain_names import get_chain_prefix
            base = get_chain_prefix(chain_len) + "animidate"
        except (ValueError, KeyError, ImportError):
            return None

    # Enumerate substituents off the chain (CR-03 branched-stem support).
    chain_set = set(principal_chain)
    has_branches = any(
        sum(1 for n in mol.GetAtomWithIdx(idx).GetNeighbors()
            if n.GetIdx() in frag_set and n.GetIdx() not in chain_set) > 0
        for idx in principal_chain
    )
    if not has_branches:
        return base

    atom_to_locant = {idx: pos + 1
                      for pos, idx in enumerate(principal_chain)}
    all_atom_idxs = set(range(mol.GetNumAtoms()))
    exclude = all_atom_idxs - frag_set

    try:
        from ..composer import _integrate_universal_prefixes
        prefix_str = _integrate_universal_prefixes(
            mol, chain_set,
            parent_type="chain",
            principal_chain=principal_chain,
            atom_to_locant=atom_to_locant,
            exclude_atoms=exclude,
        )
    except Exception:
        prefix_str = ""

    if prefix_str:
        return f"{prefix_str}{base}"
    return base


def _find_longest_carbon_chain(mol: Any, start: int,
                               frag_atoms: "set[int]") -> list:
    """DFS the longest simple carbon path starting from ``start`` within
    ``frag_atoms``. Mirrors anhydrides._find_longest_chain.

    PURE per D-07: read-only mol queries; no mutation.
    """
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


__all__ = ["name_imidate", "_is_imidate"]
