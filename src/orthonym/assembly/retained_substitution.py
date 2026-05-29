"""Phase 168 — Continuous Triviality Controller (P2; HERITAGE-derived tree visitor).

Implements TRIV-01/02/03 via a pure functional transform ``NameTreeNode -> NameTreeNode'``:
  - Depth-first leaves-first traversal (Pitfall 4 avoidance: rewrite children BEFORE the
    parent's swap-decision so the parent sees the post-swap children).
  - Per-Type runtime dispatch (D-02; P-15.1.8.1..3): Type 1 unconditional, Type 2a
    principal-group-bound, Type 2b SMARTS closed-list, Type 2c per-entry-override-or-Type-3,
    Type 3 bare-only + locant_context.
  - Multiplier-feedback after every swap (D-05; TRIV-02): re-derive the di<->bis multiplier
    on the swapped node via ``get_multiplier_prefix`` (which consults ``is_complex_substituent``)
    — never a static per-entry flag.
  - Re-alphabetization after every swap (D-06): ``_alphabetize_prefixes``.
  - T2 runtime OPSIN-RT cache (D-07; TRIV-03; memoized per ``(post_swap_subtree_str,
    pre_swap_canon)`` in a plain dict). On RT mismatch: silently keep systematic form +
    emit ``ControllerEvent(kind='swap_reject_rt_unsafe')``.

CRITICAL serializer invariant (#2 fix, reviews iter 1): the Pass-2 serializer short-circuits
on a non-None ``fragment_legacy`` (name_tree_to_string.py:96) and IGNORES ``node.prefixes``. So
ANY rewrite that changes ``parent_stem`` (the swap) OR changes a child
(``new_prefixes != node.prefixes``) MUST set ``fragment_legacy=None``, or the serializer
renders the stale legacy string and discards the rewrite. The helper
``_replace_preserving_or_resetting_legacy`` encodes the conditional: reset when a child
changed, preserve when nothing changed.

SMILES recovery is a 3-path cascade (#3 fix, reviews iter 1): an earlier hint-based
reverse-mapping path was REMOVED — the per-atom locant hint is NOT a field of the frozen
``NameTreeNode`` (name_tree.py:90-103); it lives on ``NamingResult`` (name_tree.py:128) and is
never forwarded onto nodes, so the lookup always returned None and that path was dead code.
Recovery therefore starts at token-match (labelled Path B for cascade continuity), then
reverse-OPSIN (Path C), then bail (Path D).

Recovery reach is bounded (#4, reviews iter 1): ``match_token_atoms_in_mol`` cannot map a
specific IR node to a specific physical duplicate fragment; for duplicate substituents it picks
a deterministic-but-arbitrary match. The recovered fragment SMILES is correct for true
duplicates (same canonical SMILES), but recovery can MISS for nested substituents — so
``controller_fired_count`` may trail ``controller_reach_count`` even for seed members. Plan-04
reports BOTH counts (CONTEXT D-04) and never claims reach the recovery cannot deliver.

Thread-safety (#6 REJECTED, reviews iter 1): the OPSIN-RT cache is a plain per-instance dict.
The benchmark (benchmark_multi_corpus.py) is single-threaded over rows (max_workers=1 timeout
wrapper at :191; serial rows at :446; no --threads arg) and each OpsinOracle belongs to one
Orthonym instance — no lock needed.

Module boundary preserved per D-12: this module is the ONLY new code in the assembly package;
name_tree.py, name_tree_to_string.py, naming_utils.py stay UNTOUCHED.
"""

from __future__ import annotations

import dataclasses
import logging
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from rdkit import Chem

from .name_tree import NameTreeNode, _alphabetize_prefixes, is_coarse_node
from .naming_utils import get_multiplier_prefix, is_complex_substituent

# Lazy import of SEED_TABLE / SubstitutionType inside function bodies (Pattern S3)
# to avoid circular-import risk if data/ initialization is not finished.

logger = logging.getLogger(__name__)

# Multiplier -> occurrence count, for TRIV-02 di<->bis re-derivation after a swap.
_MULT_COUNT = {
    "di": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6, "hepta": 7,
    "octa": 8, "nona": 9, "deca": 10,
    "bis": 2, "tris": 3, "tetrakis": 4, "pentakis": 5, "hexakis": 6,
    "heptakis": 7, "octakis": 8, "nonakis": 9, "decakis": 10,
}


@dataclass(frozen=True)
class ControllerEvent:
    """Phase 168 diagnostic event (RESEARCH section 4.5)."""

    kind: str  # "swap_emit" | "swap_reject_type_check" | "swap_reject_rt_unsafe" |
    # "passthrough_coarse" | "passthrough_not_in_seed" | "recovery_miss"
    canonical_smiles: Optional[str]
    seed_entry_name: Optional[str]
    substitution_type: Optional[str]
    rt_oracle_result: Optional[bool]
    p_section_cite: Optional[str]


class OpsinOracle:
    """T2 runtime OPSIN-RT cache (D-07 + RESEARCH R-02 + Pitfall 5).

    Plain per-instance dict cache keyed by ``(post_swap_subtree_str, pre_swap_canon)``.
    #6 REJECTED (reviews iter 1): no lock needed — the benchmark is single-threaded over rows.
    """

    def __init__(self, opsin_jar: Optional[str] = None):
        self._jar = opsin_jar
        self._cache: Dict[Tuple[str, str], bool] = {}
        self._name_cache: Dict[str, Optional[str]] = {}
        self._events: list = []  # ControllerEvent diagnostic log

    def rt_safe(self, pre_swap_canon: str, post_swap_subtree_str: str) -> bool:
        """CONTEXT D-07 T2: parse ``post_swap_subtree_str`` via OPSIN, canonicalize, compare to
        ``pre_swap_canon``. Cached per ``(post_swap_subtree_str, pre_swap_canon)``."""
        key = (post_swap_subtree_str, pre_swap_canon)
        if key in self._cache:
            return self._cache[key]
        if self._jar is None:
            # Graceful fallback (RESEARCH section 3.8): T2 cannot run; permit the swap (T1 already
            # verified RT safety at seed load). Stage A canary stays byte-identical anyway because
            # Stage A does not surface the rewrite.
            self._cache[key] = True
            return True
        try:
            result = subprocess.run(
                ["java", "-jar", self._jar, "-osmi"],
                input=post_swap_subtree_str + "\n",
                capture_output=True, text=True, timeout=10,
            )
            opsin_smiles = result.stdout.strip()
            if not opsin_smiles:
                self._cache[key] = False
                return False
            actual_canon = Chem.CanonSmiles(opsin_smiles)
            ok = (actual_canon == pre_swap_canon)
        except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
            logger.debug("OpsinOracle T2 RT check failed for %r: %s", post_swap_subtree_str, exc)
            ok = False
        self._cache[key] = ok
        return ok

    def name_to_smiles(self, name: str) -> Optional[str]:
        """Path C reverse-OPSIN helper. Returns canonical SMILES or None on failure. Cached."""
        if self._jar is None or not name:
            return None
        if name in self._name_cache:
            return self._name_cache[name]
        try:
            result = subprocess.run(
                ["java", "-jar", self._jar, "-osmi"],
                input=name + "\n",
                capture_output=True, text=True, timeout=10,
            )
            smi = result.stdout.strip()
            canon = Chem.CanonSmiles(smi) if smi else None
        except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
            logger.debug("OpsinOracle.name_to_smiles failed for %r: %s", name, exc)
            canon = None
        self._name_cache[name] = canon
        return canon


def _recompute_multiplicative_prefix(old_mult: Optional[str], new_name: str) -> Optional[str]:
    """TRIV-02 (D-05): re-derive the di<->bis multiplier for a repeated substituent whose name
    changed, using ``get_multiplier_prefix`` (which consults ``is_complex_substituent``
    internally) — never a static per-entry flag. Preserves the occurrence count encoded by the
    old multiplier; returns None when the node carried no multiplier.

    This matters because the serializer renders ``node.multiplicative_prefix`` verbatim
    (name_tree_to_string.py:164-165) — it does NOT recompute from sibling counts. So a swap that
    flips a substituent simple<->complex (e.g. a hyphenated retained form) MUST re-derive the
    multiplier here or the di/bis prefix would be dropped or stale.
    """
    if not old_mult:
        return None
    count = _MULT_COUNT.get(old_mult.strip().lower())
    if count is None:
        return None
    return get_multiplier_prefix(count, new_name)


def _replace_preserving_or_resetting_legacy(
    node: NameTreeNode, new_prefixes: Tuple[NameTreeNode, ...],
) -> NameTreeNode:
    """#2 FIX (reviews iter 1): no-swap branches (seed-miss / recovery-fail / type-refuse /
    RT-unsafe) return the node with re-alphabetized children. BUT if a CHILD actually changed
    (``new_prefixes != node.prefixes``), the serializer's ``fragment_legacy`` short-circuit
    (name_tree_to_string.py:96) would discard the child rewrite — so we MUST reset
    ``fragment_legacy=None`` in that case. If nothing changed, preserve ``fragment_legacy``
    (original NameTreeNode contract)."""
    alphabetized = _alphabetize_prefixes(new_prefixes)
    if alphabetized != node.prefixes:
        # A child changed (or re-sorted) => force the serializer onto explicit fields.
        return dataclasses.replace(node, prefixes=alphabetized, fragment_legacy=None)
    # Nothing changed => preserve fragment_legacy.
    return dataclasses.replace(node, prefixes=alphabetized)


def apply_triviality_controller(
    tree: NameTreeNode,
    mol: "Chem.Mol",
    principal_group: Optional[str],
    opsin_oracle: Optional[OpsinOracle] = None,
    *,
    enabled: bool = False,
) -> NameTreeNode:
    """Phase 168 entry point (CONTEXT D-01 + TRIV-01).

    - When ``enabled=False``: returns the input tree UNCHANGED (Stage A default-OFF invariant per D-08).
    - When ``enabled=True`` + ``is_coarse_node(tree)``: returns the input tree UNCHANGED (D-04 coarse passthrough).
    - Otherwise: depth-first leaves-first walk; per-Type dispatch; rewrite-on-match + RT-gate.
    """
    if not enabled:
        return tree
    if is_coarse_node(tree):
        return tree
    return _walk(tree, mol, principal_group, opsin_oracle)


def _walk(
    node: NameTreeNode,
    mol: "Chem.Mol",
    principal_group: Optional[str],
    opsin_oracle: Optional[OpsinOracle],
) -> NameTreeNode:
    """Depth-first LEAVES-FIRST traversal (Pitfall 4 mitigation per RESEARCH section 4.1).

    Children are rewritten FIRST so the parent's swap-decision sees the post-swap children.
    """
    if is_coarse_node(node):
        return node  # D-04 coarse passthrough

    # Rewrite children first (leaves-first).
    new_prefixes = tuple(
        _walk(p, mol, principal_group, opsin_oracle) for p in node.prefixes
    )

    # Lazy import of seed table (Pattern S3).
    from ..data.triviality_controller_seed import SEED_TABLE

    # Recover canonical SMILES of this node's substructure (RESEARCH section 4.2 3-path cascade).
    canonical = _recover_canonical_smiles(node, mol, opsin_oracle)
    if canonical is None:
        # Can't recover SMILES => can't look up => no swap of THIS node; preserve children rewrite
        # + re-alphabetize. #2 fix: reset fragment_legacy IFF a child changed.
        return _replace_preserving_or_resetting_legacy(node, new_prefixes)

    seed_entry = SEED_TABLE.get(canonical)
    if seed_entry is None:
        # Not in seed table => keep systematic (preserve children rewrite + re-alphabetize per D-06).
        return _replace_preserving_or_resetting_legacy(node, new_prefixes)

    # Per-Type runtime dispatch (CONTEXT D-02; RESEARCH section 2.1).
    if not _type_check(node, mol, seed_entry, principal_group, opsin_oracle):
        # Type check refused: keep systematic. #2 fix: reset fragment_legacy IFF a child changed.
        return _replace_preserving_or_resetting_legacy(node, new_prefixes)

    # Build the candidate rewrite (this DOES change parent_stem => fragment_legacy=None per #2/Pitfall 1).
    rewritten = _build_rewrite(node, seed_entry, new_prefixes)

    # T2 runtime RT-safety check (CONTEXT D-07 + TRIV-03).
    if opsin_oracle is not None:
        from .name_tree_to_string import name_tree_to_string
        post_swap_str = name_tree_to_string(rewritten, style="pin")
        if not opsin_oracle.rt_safe(canonical, post_swap_str):
            # RT-unsafe => silently reject swap (the systematic form is kept). CONTEXT D-07 + Pitfall 6.
            # #2 fix: reset fragment_legacy IFF a child changed (the swap of THIS node is rejected,
            # but a child below may have changed).
            return _replace_preserving_or_resetting_legacy(node, new_prefixes)

    return rewritten


def _build_rewrite(
    node: NameTreeNode,
    seed_entry,
    new_prefixes: Tuple[NameTreeNode, ...],
) -> NameTreeNode:
    """Rewrite a node to its retained-PIN form (CONTEXT D-01 + RESEARCH section 4.1).

    CRITICAL INVARIANT per #2 + Pitfall 1: this function DOES change ``parent_stem``, so it MUST
    set ``fragment_legacy=None`` to force the serializer to use explicit fields rather than the
    stale legacy string at name_tree_to_string.py:96.

    TRIV-02 (D-05): the di<->bis multiplier is re-derived via ``_recompute_multiplicative_prefix``
    (which calls ``get_multiplier_prefix``), and the parenthesization hint is re-derived via
    ``is_complex_substituent`` (P-16.3.4) — both single-source-of-truth predicates, never a static
    flag.
    """
    new_name = seed_entry.retained_pin_name
    return dataclasses.replace(
        node,
        parent_stem=new_name,
        locants=(tuple(seed_entry.locant_context)
                 if seed_entry.locant_context else node.locants),
        prefixes=_alphabetize_prefixes(new_prefixes),  # D-06 always-re-run
        multiplicative_prefix=_recompute_multiplicative_prefix(
            node.multiplicative_prefix, new_name),  # TRIV-02 (D-05)
        parenthesization_hint=is_complex_substituent(new_name),  # P-16.3.4 (D-05)
        iupac_section_cite=seed_entry.iupac_p_section,
        fragment_legacy=None,  # CRITICAL — force serializer to use explicit fields (#2 / Pitfall 1)
    )


def _type_check(
    node: NameTreeNode,
    mol: "Chem.Mol",
    seed_entry,
    principal_group: Optional[str],
    opsin_oracle: Optional[OpsinOracle],  # WARNING #7 fix: propagated for Type 2b prefix recovery
) -> bool:
    """Per-Type runtime check (CONTEXT D-02 + RESEARCH section 2.1)."""
    from ..data.triviality_controller_seed import SubstitutionType

    st = seed_entry.substitution_type
    if st == SubstitutionType.TYPE_1:
        return _type_1_check(node, seed_entry)
    if st == SubstitutionType.TYPE_2A:
        return _type_2a_check(node, principal_group, seed_entry)
    if st == SubstitutionType.TYPE_2B:
        return _type_2b_check(node, mol, seed_entry, opsin_oracle)  # WARNING #7 — oracle propagated
    if st == SubstitutionType.TYPE_2C:
        return _type_2c_check(node, mol, seed_entry)
    if st == SubstitutionType.TYPE_3:
        return _type_3_check(node, seed_entry)
    raise ValueError(f"unknown substitution_type {st!r}")


def _type_1_check(node: NameTreeNode, seed_entry) -> bool:
    """Type 1 (P-15.1.8.1): unconditional swap on canonical-SMILES match (RESEARCH section 2.1)."""
    return True


def _type_2a_check(node: NameTreeNode, principal_group: Optional[str], seed_entry) -> bool:
    """Type 2a (P-15.1.8.2.1): principal-group-bound (RESEARCH section 2.1 + Pitfall 2).

    NOTE (#8, reviews iter 1): ``principal_group`` is the MOLECULE-LEVEL PCG
    (features.principal_group), not a fragment-level group. So a phenol/aniline fragment occurring
    as a NON-principal substituent on a higher-seniority parent (e.g. an ester) will NOT swap (the
    required PG won't match). This is the correct conservative PIN behavior per CONTEXT D-02; it
    bounds Type 2a coverage. A Plan-03 fixture pins the intended stay-systematic behavior.
    """
    required = seed_entry.principal_group_required
    if not required:
        return False
    if principal_group is None:
        return False
    return _principal_groups_match(principal_group, required)


def _type_2b_check(
    node: NameTreeNode,
    mol: "Chem.Mol",
    seed_entry,
    opsin_oracle: Optional[OpsinOracle],  # WARNING #7 fix: oracle propagated for prefix SMILES recovery
) -> bool:
    """Type 2b (P-15.1.8.2.2): closed-substituent SMARTS allow-list (RESEARCH section 2.1).

    WARNING #7 FIX: ``opsin_oracle`` is propagated into prefix SMILES recovery (formerly hard-coded
    to None, which disabled Path C and caused Type 2b to silently fail on any prefix where Path B
    token-match had no in-mol atoms). The 3-path cascade is used per prefix.
    """
    smarts_list = seed_entry.compulsory_prefix_smarts
    if smarts_list is None:
        return False
    if not node.prefixes:
        return True  # bare => Type 2b reduces to Type 1 unconditional
    for prefix_node in node.prefixes:
        prefix_canon = _recover_canonical_smiles(prefix_node, mol, opsin_oracle)
        if prefix_canon is None:
            return False
        prefix_mol = Chem.MolFromSmiles(prefix_canon)
        if prefix_mol is None:
            return False
        matched = False
        for smarts in smarts_list:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None and prefix_mol.HasSubstructMatch(pattern):
                matched = True
                break
        if not matched:
            return False
    return True


def _type_2c_check(node: NameTreeNode, mol: "Chem.Mol", seed_entry) -> bool:
    """Type 2c (P-15.1.8.2.3): per-name-specific (CONTEXT D-02 default-to-Type-3)."""
    if seed_entry.locus_override_rule_id is None:
        return _type_3_check(node, seed_entry)
    return False


def _type_3_check(node: NameTreeNode, seed_entry) -> bool:
    """Type 3 (P-15.1.8.3): bare-only + locant_context match (RESEARCH section 2.1 + Pitfall 3)."""
    if node.prefixes:
        return False
    if seed_entry.locant_context is not None:
        if set(node.locants) != set(seed_entry.locant_context):
            return False
    return True


def _principal_groups_match(inferred: str, required: str) -> bool:
    """Normalize PG strings for comparison (case-insensitive, strip whitespace)."""
    return inferred.strip().lower() == required.strip().lower()


def _recover_canonical_smiles(
    node: NameTreeNode,
    mol: "Chem.Mol",
    opsin_oracle: Optional[OpsinOracle],
) -> Optional[str]:
    """Recover the canonical SMILES of the substructure this node represents.

    3-PATH cascade per RESEARCH section 4.2 + #3 fix (the earlier hint-based reverse-mapping path
    was REMOVED — the per-atom locant hint is NOT a field of the frozen NameTreeNode; it lives on
    NamingResult and is never forwarded to nodes):
      Path B — rules.parent_correctness.match_token_atoms_in_mol(node.parent_stem, mol)
      Path C — opsin_oracle.name_to_smiles(node.parent_stem) reverse-OPSIN
      Path D — bail (return None; controller keeps systematic form)

    #4 reach-limit: Path B picks a deterministic-but-arbitrary match for duplicate fragments; it
    cannot map a specific IR node to a specific physical duplicate. Recovery can MISS for nested
    substituents, so controller_fired_count may trail controller_reach_count.
    """
    # Path B: token-match the parent_stem against atoms in the mol.
    try:
        from ..rules.parent_correctness import match_token_atoms_in_mol
        token = (node.parent_stem or "").strip()
        if token and mol is not None:
            atoms = match_token_atoms_in_mol(token, mol)
            if atoms:
                frag = Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), canonical=True)
                if frag:
                    return Chem.CanonSmiles(frag)
    except Exception as exc:
        logger.debug("Path B SMILES recovery failed for %r: %s", node.parent_stem, exc)

    # Path C: reverse-OPSIN the parent_stem (REQUIRES opsin_oracle propagation per WARNING #7).
    if opsin_oracle is not None:
        try:
            canon = opsin_oracle.name_to_smiles(node.parent_stem or "")
            if canon:
                return canon
        except Exception as exc:
            logger.debug("Path C SMILES recovery failed for %r: %s", node.parent_stem, exc)

    # Path D: bail.
    return None
