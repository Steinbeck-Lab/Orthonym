""": per-substring (per-node) scoring on the Name-Tree IR.

Generalizes the scalar ``ParentCorrectnessScorer`` (rules/parent_correctness.py)
to NODE granularity (CONTEXT): for each STRUCTURED ``NameTreeNode``,
attribute a binary ``parent_score`` / ``locant_score`` / ``substituent_score``
(1.0 match / 0.0 mismatch / 0.5 no-decision) by aligning the node against the
OPSIN-parsed reference name.

Mechanism (REUSED, not re-invented — RESEARCH §Don't-Hand-Roll):
  - The reference name is OPSIN-parsed ONCE per compound (A1 strategy, audit
    §A1 OPSIN-Cost Prototype) via ``opsin_reference_mol``; per-node atom
    alignment is RDKit substructure matching via ``match_token_atoms_in_mol``
    (``CanonicalRankAtoms(breakTies=True)`` deterministic tiebreak — RESEARCH
    Pitfall 4).
  - Reads the reference name from the SAME thread-local as the parent oracle
    (``parent_correctness._pc_context``). Production (no reference set) -> ``{}``
    with ZERO OPSIN cost -> byte-identical.
  - Coarse nodes (``is_coarse_node``, WR-4 single source of truth) are OMITTED
    from the score dict: they carry no scoreable substrings; the
    aggregate confidence is retained for them.

POST-HOC contract (audit §POST-HOC Byte-Identical Contract): the
``node_scores`` dict is attached on ``CandidateName`` AFTER ``compute_confidence``
returns; it NEVER feeds back into ``confidence`` (contrast the V18
``multiple_bond_count`` recompute at ``candidate_pool.py:810-823`` — the
explicit anti-model).

Binary ``substituent_score`` is locked (audit §Binary-vs-Graded); a graded
variant is a documented escape hatch only if the curated near-tie set
proves binary insufficient (Plan 03 owns that validation).
"""
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .name_tree import NameTreeNode, is_coarse_node

logger = logging.getLogger(__name__)

# Integer locants appearing in an IUPAC name (e.g. 2 and 3 in "2,3-dimethyl...").
_LOCANT_INT_RE = re.compile(r"\d+")

# Leading locant cluster on a substituent stem ("2-methyl" -> "methyl").
_LOCANT_PREFIX_RE = re.compile(r"^\d+[a-z]?,?(\d+[a-z]?,?)*-?")


@dataclass(frozen=True)
class NodeScores:
    """Per-node parent/locant/substituent scores (binary: 1.0 / 0.0 / 0.5)."""

    parent_score: float       # node's parent atom set vs reference parent
    locant_score: float       # node's locants vs reference numbering
    substituent_score: float  # do node.prefixes[] subtrees match (binary)


def _ref_locant_set(ref_name: str) -> set:
    """All integer locants appearing in the reference name."""
    return {int(m) for m in _LOCANT_INT_RE.findall(ref_name or "")}


def _node_parent_token(node: "NameTreeNode") -> Optional[str]:
    """Best-effort OPSIN-parseable parent-hydride stem for a node.

    A structured root carries a bare stem ('but') with the saturated suffix
    implicit; substituent prefix nodes carry e.g. '2-methyl'. Strip any leading
    locant cluster so a prefix resolves to its hydride stem ('methyl'). The
    matcher (``_candidate_parent_atoms``) then also tries the stem+'ane' alkane
    form; if neither parses, the scorer treats it as no-decision (0.5).
    """
    stem = (node.parent_stem or "").strip()
    if not stem:
        return None
    stem_core = _LOCANT_PREFIX_RE.sub("", stem).strip()
    return stem_core or stem


def _candidate_parent_atoms(node: "NameTreeNode", mol: Any) -> Optional[set]:
    """Match the node's parent-hydride atoms in ``mol`` (None on any failure)."""
    # Lazy import to avoid any assembly<->rules import cycle at module load
    # (mirrors name_tree._alphabetize_prefixes:168).
    from ..rules.parent_correctness import match_token_atoms_in_mol

    token = _node_parent_token(node)
    if not token:
        return None
    # Try the token as-is, then the saturated alkane form (bare stem -> +'ane').
    for candidate_token in (token, token + "ane"):
        atoms = match_token_atoms_in_mol(candidate_token, mol)
        if atoms:
            return atoms
    return None


class PerNodeScorer:
    """Attributes per-node parent/locant/substituent scores off the IR."""

    @staticmethod
    def score_tree(tree: "NameTreeNode", mol: Any) -> Dict[int, NodeScores]:
        """Return ``{id(node): NodeScores}`` for each STRUCTURED node, or ``{}``.

        ``{}`` is returned when: no reference name is set (production
        byte-identical guard, zero OPSIN cost), the root is coarse, or
        the reference name does not OPSIN-parse (no-decision).
        """
        # Lazy import to avoid any assembly<->rules import-cycle at module load.
        from ..rules.parent_correctness import (
            _extract_reference_parent_atoms,
            _pc_context,
            opsin_reference_mol,
        )

        ref_name = getattr(_pc_context, "reference_name", None)
        if ref_name is None:
            return {}                    # production: no signal, zero OPSIN cost
        if tree is None or is_coarse_node(tree):
            return {}                    # coarse -> aggregate retained
        if opsin_reference_mol(ref_name) is None:
            return {}                    # reference unparseable -> no-decision
        ref_parent_atoms = _extract_reference_parent_atoms(ref_name, mol)
        ref_locants = _ref_locant_set(ref_name)
        scores: Dict[int, NodeScores] = {}
        PerNodeScorer._walk(tree, mol, ref_parent_atoms, ref_locants, scores)
        return scores

    @staticmethod
    def _walk(node, mol, ref_parent_atoms, ref_locants, scores) -> None:
        if is_coarse_node(node):
            return                       # skip coarse sub-nodes
        scores[id(node)] = PerNodeScorer._score_node(
            node, mol, ref_parent_atoms, ref_locants
        )
        for prefix in node.prefixes:
            PerNodeScorer._walk(prefix, mol, ref_parent_atoms, ref_locants, scores)

    @staticmethod
    def _score_node(node, mol, ref_parent_atoms, ref_locants) -> NodeScores:
        return NodeScores(
            parent_score=PerNodeScorer._parent_score(node, mol, ref_parent_atoms),
            locant_score=PerNodeScorer._locant_score(node, ref_locants),
            substituent_score=PerNodeScorer._substituent_score(
                node, mol, ref_parent_atoms
            ),
        )

    @staticmethod
    def _parent_score(node, mol, ref_parent_atoms) -> float:
        if ref_parent_atoms is None:
            return 0.5                   # reference alignment unavailable
        node_atoms = _candidate_parent_atoms(node, mol)
        if node_atoms is None:
            return 0.5                   # node token unmatched -> no-decision
        return 1.0 if node_atoms == ref_parent_atoms else 0.0

    @staticmethod
    def _locant_score(node, ref_locants) -> float:
        if not node.locants:
            return 0.5                   # no locants on this node -> no-decision
        if not ref_locants:
            return 0.5
        return 1.0 if set(node.locants) <= ref_locants else 0.0

    @staticmethod
    def _substituent_score(node, mol, ref_parent_atoms) -> float:
        structured = [p for p in node.prefixes if not is_coarse_node(p)]
        if not structured:
            return 0.5                   # no substituents -> no-decision
        prefix_scores = [
            PerNodeScorer._parent_score(p, mol, ref_parent_atoms)
            for p in structured
        ]
        if all(s == 1.0 for s in prefix_scores):
            return 1.0
        if any(s == 0.0 for s in prefix_scores):
            return 0.0
        return 0.5                       # mixed no-decision (binary, per audit)


# ---------------------------------------------------------------------------
# lexicographic near-tie comparator
# ---------------------------------------------------------------------------

# Sentinel for an unscored candidate: all-no-decision -> ties everything ->
# the comparator returns 0 and the caller defers to the aggregate (strict
# refinement). An unscored candidate never wins or loses on per-substring.
_NO_DECISION = NodeScores(0.5, 0.5, 0.5)


def _scores_from(node_scores, tree) -> NodeScores:
    """Aggregate an explicit ``(node_scores, tree)`` pair into comparison-level NodeScores.

     STAGE B: extracted from ``_candidate_scores`` so the Stage-B selector can build a
    comparison key from ``(node_scores_rewritten, tree_rewritten)`` WITHOUT mutating
    ``cand.node_scores`` (no aliasing, no restore). On ``(cand.node_scores, cand.tree)`` this is
    byte-identical to the prior inline body — the refactor is behavior-preserving.
    """
    if not node_scores or tree is None:
        return _NO_DECISION
    root = node_scores.get(id(tree))
    if root is None:
        return _NO_DECISION
    real_locants = [
        ns.locant_score for ns in node_scores.values() if ns.locant_score != 0.5
    ]
    locant = min(real_locants) if real_locants else 0.5
    return NodeScores(root.parent_score, locant, root.substituent_score)


def _candidate_scores(cand: Any) -> NodeScores:
    """Aggregate a candidate's tree into comparison-level NodeScores.

    ``parent_score`` and ``substituent_score`` are read from the ROOT node (the
    main parent + whether its immediate substituents match). ``locant_score``
    is the WORST real (non-0.5) locant decision ANYWHERE in the tree, so a wrong
    locant on a child substituent node still surfaces at comparison level — e.g.
    ``2-methylbutane`` vs ``3-methylbutane``, where the distinguishing locant
    lives on the ``methyl`` prefix node, not the root. (A root-only read would
    tie these, since the root chain carries no locant.) Returns the all-0.5
    no-decision sentinel when the candidate has no scored tree.
    """
    # STAGE B: delegate to _scores_from (behavior-preserving; the explicit-pair form lets
    # the Stage-B selector key on the rewritten tree without mutating cand.node_scores).
    return _scores_from(getattr(cand, "node_scores", None), getattr(cand, "tree", None))


def compare_scores(sa: NodeScores, sb: NodeScores) -> int:
    """Pure lexicographic first-point-of-difference on two NodeScores (CONTEXT).

     STAGE B: extracted from ``compare_by_node_scores`` so the Stage-B selector can compare
    explicit keys (e.g. rewritten-tree scores from ``_scores_from``) without re-reading candidate
    attributes. Behavior on ``_candidate_scores(a/b)`` is byte-identical.
    """
    for key in ("parent_score", "locant_score", "substituent_score"):
        va, vb = getattr(sa, key), getattr(sb, key)
        if va != vb:
            return -1 if va > vb else 1
    return 0


def compare_by_node_scores(a: Any, b: Any) -> int:
    """Lexicographic first-point-of-difference comparator (CONTEXT).

    Compares two near-tie candidates by their ROOT NodeScores in the FIXED
    priority order ``parent_score -> locant_score -> substituent_score``. At
    the first key where they differ, the HIGHER score wins. Returns:

        -1 a is better
        +1 b is better
         0 full per-substring tie -> caller defers to the aggregate
            (``select_best_candidate``), a STRICT refinement.

    This is NOT a weighted sum (the anti-pattern this phase exists to kill — a
    high ``substituent_score`` must never mask a zero ``parent_score``; the
    parent must be right before locants matter, mirroring IUPAC
    first-point-of-difference, Blue Book P-31.1.4). Pure function: no OPSIN, no
    mutation, deterministic.
    """
    return compare_scores(_candidate_scores(a), _candidate_scores(b))
