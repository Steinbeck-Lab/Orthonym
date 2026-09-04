"""Phase 163 imidate handler — Tier-D iminoester functional-class naming.

IUPAC cite: P-65.1.7 (imidic acids and imidates; "alkyl alkanimidate"
functional-class naming for R-C(=NH)-O-R').

Mirrors handlers/isothiocyanate.py structurally: predicate-pure + direct-
return + pool.add() + _inject_stereo_if_missing wrapping.

References:
- handlers/isothiocyanate.py (Tier-B retained-name handler; structural template per CONTEXT)
- functional_groups.py iminoester SMARTS (Phase 163 Plan-02 commit 163-02-04 addition)
- 163-AUDIT-FRN.md § 6 (per-fixture spec + intercept analysis + predicate purity proof)
- 163-AUDIT-FRN.md § 7 (INNER_DISPATCH priority 2900 LOCK)
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_imidate(features: Any) -> bool:
    """Predicate: iminoester is the principal group, or iminoester is the
    sole characteristic group (no higher-seniority PG present) (pure).

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

    Per + Phase 158 + Phase 160 hard invariant: NO mol
    mutation; NO features mutation; NO module-global state R/W; NO
    exception swallowing.
    """
    fg = getattr(features, 'functional_groups', None) or {}
    # W2F-P6 (P-66.1.6.1.2.1): also fire on the N-substituted carbamimidate
    # ester (dedicated SMARTS; amidine is suppressed on its atoms so pg is None).
    if not (fg.get('iminoester') or fg.get('carbamimidate')):
        return False
    pg = getattr(features, 'principal_group', None)
    return pg in (None, 'iminoester', 'carbamimidate')


def name_imidate(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Phase 163 Tier-D iminoester functional-class handler.

    Emits "alkyl alkanimidate" PIN per IUPAC P-65.1.7. Style parameter is
    ignored per CONTEXT (single PIN per compound).

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

    if mol is None:
        mol = getattr(features, 'mol', None)
    if mol is None:
        return None

    # W2F-P6 (P-66.1.6.1.2.1): the N-substituted carbamimidate ester
    # R''2N-C(=NR')-O-R owns the whole -O-C(=N-)-N unit (a diamino imidate;
    # the central C carries NO carbon, so the parent acid is carbamimidic acid
    # -> the retained stem 'carbamimidate'). When present, name it here with
    # N/N' substituent citation. Fail closed (return None) rather than falling
    # back to the iminoester path, which would drop the amino-N substituents.
    carb_matches = features.functional_groups.get('carbamimidate', [])
    if carb_matches:
        name = _name_carbamimidate(mol, carb_matches[0])
        if name is None:
            return None
        name = _enrich_handler_name(features, name, "imidate")
        pool = get_current_pool()
        cand = pool.add(name, "imidate", features)
        if cand is None:
            return None
        final_name = _inject_stereo_if_missing(features, cand.name,
                                               atom_to_locant=None)
        return NamingResult(
            name=final_name,
            tree=NameTreeNode(parent_stem=final_name, class_id="imidate",
                              iupac_section_cite="P-66.1.6.1.2.1",
                              fragment_legacy=final_name),
            atom_to_locant_hint=None,
        )

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

    # P-66.1.6.1.2.1 (BB 33398): the imidic-ester tautomer of urea,
    # H2N-C(OH)=NH -> 'carbamimidic acid', has the retained stem
    # 'carbamimidate' (NOT the systematic '1-aminomethanimidate' the chain
    # path would emit). Detect it by building the acid analog (ester O-alkyl
    # -> OH) and consulting the inorganic-acids exact-SMILES table. Only
    # fires when the whole acid analog is EXACTLY a tabled retained acid.
    stem_word = _retained_imidate_stem(mol, c_carbonyl_idx, nh_idx,
                                       o_idx, c_alkyl_idx)
    if stem_word is None:
        # Branch 2: chain stem with -imidate suffix
        # Collect subgraph from C_carbonyl excluding =NH and =O paths
        stem_atoms = _collect_subgraph(mol, c_carbonyl_idx,
                                       exclude={nh_idx, o_idx})
        # Chain length includes c_carbonyl (locant-1 carbon of the stem)
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
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="imidate", iupac_section_cite="P-65.6", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


def _name_carbamimidate(mol: Any, match: "tuple[int, ...]") -> Optional[str]:
    """P-66.1.6.1.2.1 (BB 33408): R''2N-C(=NR')-O-R -> the carbamimidate ester
    PIN 'R N'-R'-N,N-R''-carbamimidate'.

    N = amino (sp3) N substituents; N' = imino (=N-) N substituent. The whole
    prefix set is cited alphanumerically by substituent name (methyl < phenyl).
    Unsubstituted -> 'R carbamimidate'. Fail closed (None) on any un-nameable
    ester-alkyl side or N-substituent fragment.

    PURE per: read-only mol queries; no mutation.

    ``match`` = (central_C, imino_N, amino_N, ester_O, alkyl_C) per the
    carbamimidate SMARTS ``[CX3](=[NX2])([NX3])[OX2][#6]``.
    """
    from collections import OrderedDict

    from ..naming_utils import (
        alpha_sort_key,
        apply_enclosing_marks,
        get_multiplier_prefix,
        needs_brackets,
    )
    from ..substituent_enumerator import name_substituent

    central_c, imino_n, amino_n, o_idx, alkyl_c = match

    # Leading word: the ester O-alkyl side (R).
    alkyl_atoms = _collect_subgraph(mol, alkyl_c, exclude={o_idx})
    alkyl_word = _name_alkyl_fragment(mol, alkyl_atoms, anchor=alkyl_c)
    if alkyl_word is None:
        return None

    # Enumerate N (amino, sp3) and N' (imino, =N-) substituents. Each distinct
    # substituent NAME collects its locant symbols ('N' / "N'").
    by_name: "OrderedDict[str, list]" = OrderedDict()
    for n_idx, locant in ((amino_n, 'N'), (imino_n, "N'")):
        for nbr in mol.GetAtomWithIdx(n_idx).GetNeighbors():
            ni = nbr.GetIdx()
            if ni == central_c or nbr.GetAtomicNum() <= 1:
                continue
            frag = _collect_subgraph(mol, ni, exclude={n_idx})
            sub_name = name_substituent(mol, set(frag), ni)
            # Fail closed on an un-nameable N-fragment: name_substituent returns
            # the documented 'substituent' sentinel (or junk like
            # 'boronic acidyl' containing a space) as a last resort rather than
            # None. A genuine single-substituent name never contains a space,
            # so refuse both -> the whole carbamimidate name is never truncated.
            if (not sub_name or sub_name == "substituent"
                    or " " in sub_name):
                return None
            by_name.setdefault(sub_name, []).append(locant)

    prefix_parts = []
    for nm in sorted(by_name, key=alpha_sort_key):
        locs = sorted(by_name[nm])   # 'N' sorts before "N'"
        loc_str = ','.join(locs)
        disp = apply_enclosing_marks(nm, -1) if needs_brackets(nm) else nm
        if len(locs) == 1:
            prefix_parts.append(f"{loc_str}-{disp}")
        else:
            mult = get_multiplier_prefix(len(locs), nm)
            prefix_parts.append(f"{loc_str}-{mult}{disp}")
    n_prefix = '-'.join(prefix_parts)

    if n_prefix:
        return f"{alkyl_word} {n_prefix}carbamimidate"
    return f"{alkyl_word} carbamimidate"


def _retained_imidate_stem(mol: Any, c_carbonyl_idx: int, nh_idx: int,
                           o_idx: int, c_alkyl_idx: int) -> Optional[str]:
    """P-66.1.6.1.2.1 (BB 33398): if the imidate's ACID analog (ester
    -O-alkyl replaced by -OH) is EXACTLY a tabled retained / functional-
    replacement inorganic acid (carbamimidic acid, carbonimidic acid, ...),
    return its '-ate' stem ('carbamimidate'), else None (chain path owns it).

    PURE: builds a throwaway RWMol copy; no mutation of the input mol."""
    from rdkit import Chem

    from ...rules.esters import _acid_name_to_ate
    from ...rules.inorganic_acids import lookup_exact_acid_name

    # The alkyl side must carry no extra functional atoms (else the acid
    # analog would silently drop them -> a different molecule -> fail closed).
    alkyl_atoms = set(_collect_subgraph(mol, c_alkyl_idx, exclude={o_idx}))
    for a in alkyl_atoms:
        if mol.GetAtomWithIdx(a).GetAtomicNum() not in (1, 6):
            return None

    rw = Chem.RWMol(mol)
    # Remove alkyl atoms (high->low so indices stay valid); the ester O keeps
    # its remaining valence and RDKit fills it with an implicit H -> -OH.
    for idx in sorted(alkyl_atoms, reverse=True):
        rw.RemoveAtom(idx)
    acid = rw.GetMol()
    try:
        Chem.SanitizeMol(acid)
        acid_smiles = Chem.MolToSmiles(acid)
    except Exception:
        return None
    retained = lookup_exact_acid_name(acid_smiles)
    if retained is None:
        return None
    return _acid_name_to_ate(retained)


def _collect_subgraph(mol: Any, anchor_idx: int,
                      exclude: "set[int]") -> "tuple[int, ...]":
    """BFS subgraph collection from anchor, excluding given atom indices.

    PURE per: read-only mol traversal; no mutation.
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

    PURE per: read-only mol queries; no mutation.
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
        # whole handler. Per contract this fallback also pure.
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

    Naming rules per IUPAC P-65.6.3.3.7.1:
    - aromatic ring at anchor -> "benzimidate" (retained PIN, FRN-)
    - N-C linear stem -> "{chainprefix}animidate" (SYSTEMATIC PIN; BBv2 L31993:
      'methyl ethanimidate (PIN) methyl acetimidate' -> acetimidate is general-
      nomenclature only, so 2C uses 'ethanimidate' like every other length)
    - branched stem -> "{locant-substituent-list}{chainprefix}animidate"

    PURE per: read-only mol queries; no mutation.
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

    # Build base stem. W3-P08 (P-65.6.3.3.7.1, BBv2 L31993): the PIN is the
    # SYSTEMATIC '{chainprefix}animidate' for EVERY chain length. 'acetimidate'
    # (2C) is general-nomenclature only, so 2C -> 'ethanimidate' via the same
    # path as all other lengths (get_chain_prefix(2)='eth').
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

    PURE per: read-only mol queries; no mutation.
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
