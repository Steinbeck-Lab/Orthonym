"""a phase Plan 01 Wave 0 RED scaffold — SCORE-03 per-node scorer.

These tests import ``PerNodeScorer`` / ``NodeScores`` from
``orthonym.assembly.per_substring_scoring``, which does NOT exist until Plan 02.
Collection therefore ERRORS (ModuleNotFoundError) — the expected Wave 0 RED
state. Plan 02 creates the module and turns these GREEN.

Locked contracts under test (internal notes-per-substring-scoring.md):
  - ``score_tree`` returns ``{}`` when no reference name is set (production
    byte-identical guard — the production path never calls set_reference_name).
  - ``score_tree`` returns ``{}`` on a coarse tree even WITH a reference set
    (coarse fallback).
  - ``node_scores`` attach POST-HOC: ``.confidence`` / ``.factors`` are unchanged
    with vs without a reference set (POST-HOC byte-identical contract).
  - a structured node scored with a reference yields ``NodeScores`` whose
    parent/locant/substituent values are in ``{0.0, 0.5, 1.0}`` (binary scorer).
"""
import pytest
from rdkit import Chem

from orthonym.assembly.name_tree import NameTreeNode, is_coarse_node
from orthonym.assembly.candidate_pool import CandidatePool
from orthonym.assembly.coverage_scoring import CandidateName
from orthonym.namer import Orthonym, name_with_tree
from orthonym.rules.parent_correctness import (
    clear_reference_name,
    set_reference_name,
)
from tests.support.jars import jar_or_none

# RED dependency: this module is the Plan 02/03 deliverable. Importing it at
# module top makes collection fail (ModuleNotFoundError) until it lands — the
# intended Wave 0 RED state.
from orthonym.assembly.per_substring_scoring import (
    NodeScores,
    PerNodeScorer,
    compare_by_node_scores,
)

# OPSIN-required guard (mirrors test_parent_correctness.py:36-38).
opsin_required = pytest.mark.skipif(
    jar_or_none() is None, reason="OPSIN jar required for subprocess tests"
)


@pytest.mark.unit
class TestPerNodeScorer:
    def setup_method(self):
        clear_reference_name()

    def teardown_method(self):
        clear_reference_name()

    def test_no_reference_noop(self):
        """No set_reference_name -> score_tree returns {} (production guard).

        The production naming path never calls set_reference_name, so the scorer
        must short-circuit to {} with zero OPSIN cost — this is what keeps the
        production output byte-identical (SCORE-05).
        """
        node = NameTreeNode(parent_stem="ethan", suffix="ol")  # structured
        assert not is_coarse_node(node)
        mol = Chem.MolFromSmiles("CCO")
        assert PerNodeScorer.score_tree(node, mol) == {}

    def test_coarse_tree_returns_empty(self):
        """Coarse tree -> {} even WITH a reference set (coarse fallback).

        A coarse node carries only the final string (parent_stem ==
        fragment_legacy, no prefixes, no suffix), so there are no scoreable
        substrings; the scorer abstains and the aggregate confidence stands.
        """
        coarse = NameTreeNode(parent_stem="butane", fragment_legacy="butane")
        assert is_coarse_node(coarse)
        set_reference_name("butane")
        mol = Chem.MolFromSmiles("CCCC")
        assert PerNodeScorer.score_tree(coarse, mol) == {}

    def test_posthoc_no_confidence_change(self):
        """node_scores attach POST-HOC:.confidence /.factors are byte-equal
        with vs without a reference set (POST-HOC byte-identical contract).

        Uses the EXISTING ``'score_based'`` mode (NOT the per-substring mode that
        Plan 03 introduces) — proving the no-confidence-change contract does not
        require the new mode, so this scaffold stays self-contained.
        """
        mol = Chem.MolFromSmiles("CCO")
        canonical = Chem.MolToSmiles(mol)
        feats = Orthonym()._perceive(mol, "CCO", canonical)
        node = NameTreeNode(parent_stem="ethan", suffix="ol")
        clear_reference_name()
        without_ref = CandidatePool('score_based').add(
            'ethanol', 'chain', feats, tree=node)
        set_reference_name("ethanol")
        try:
            with_ref = CandidatePool('score_based').add(
                'ethanol', 'chain', feats, tree=node)
        finally:
            clear_reference_name()
        assert with_ref.confidence == without_ref.confidence
        assert with_ref.factors == without_ref.factors

    @opsin_required
    def test_structured_node_scored_with_reference(self):
        """A structured chain tree scored with a reference yields a non-empty
        dict whose values are NodeScores with parent/locant/substituent in
        {0.0, 0.5, 1.0} (binary scorer per the audit doc).

        Uses 2-methylbutane: a bare unbranched alkane like butane is COARSE
        (parent_stem == fragment_legacy), but a substituted chain is structured
        (root 'but' + a '2-methyl' prefix node carrying locants).
        """
        res = name_with_tree("CC(C)CC")  # 2-methylbutane: STRUCTURED chain
        assert res.tree is not None and not is_coarse_node(res.tree)
        mol = Chem.MolFromSmiles("CC(C)CC")
        set_reference_name("2-methylbutane")
        try:
            scores = PerNodeScorer.score_tree(res.tree, mol)
        finally:
            clear_reference_name()
        assert scores, "structured tree + reference must produce node scores"
        for ns in scores.values():
            assert isinstance(ns, NodeScores)
            for v in (ns.parent_score, ns.locant_score, ns.substituent_score):
                assert v in (0.0, 0.5, 1.0)


# ---------------------------------------------------------------------------
# a phase SCORE-04: lexicographic near-tie comparator (no OPSIN needed)
# ---------------------------------------------------------------------------

def _cand(parent, locant, subst, name="x"):
    """A synthetic CandidateName whose ROOT node carries the given scores."""
    cand = CandidateName(name=name, handler="chain", confidence=0.5)
    root = NameTreeNode(parent_stem="x")
    cand.tree = root
    cand.node_scores = {id(root): NodeScores(parent, locant, subst)}
    return cand


@pytest.mark.unit
class TestCompareByNodeScores:
    def test_higher_parent_wins(self):
        a = _cand(1.0, 0.0, 0.0)
        b = _cand(0.0, 1.0, 1.0)
        assert compare_by_node_scores(a, b) == -1
        assert compare_by_node_scores(b, a) == 1

    def test_parent_breaks_first_aggregate_blindness_kill(self):
        """A parent-correct / substituent-WRONG candidate MUST beat a
        parent-wrong / substituent-RIGHT candidate (parent breaks first; NOT a
        weighted sum). This is the exact aggregate-blindness this phase kills."""
        parent_right = _cand(1.0, 0.5, 0.0)   # right parent, wrong substituents
        parent_wrong = _cand(0.0, 0.5, 1.0)   # wrong parent, lucky substituents
        assert compare_by_node_scores(parent_right, parent_wrong) == -1

    def test_locant_breaks_when_parent_ties(self):
        a = _cand(1.0, 1.0, 0.0)
        b = _cand(1.0, 0.0, 1.0)
        assert compare_by_node_scores(a, b) == -1

    def test_substituent_breaks_when_parent_and_locant_tie(self):
        a = _cand(1.0, 1.0, 1.0)
        b = _cand(1.0, 1.0, 0.0)
        assert compare_by_node_scores(a, b) == -1

    def test_full_tie_returns_zero(self):
        a = _cand(1.0, 0.5, 0.0)
        b = _cand(1.0, 0.5, 0.0)
        assert compare_by_node_scores(a, b) == 0

    def test_unscored_candidates_tie(self):
        """No node_scores -> all-0.5 sentinel -> tie -> aggregate fallback
        (an unscored candidate never wins/loses on per-substring;)."""
        a = CandidateName(name="a", handler="chain", confidence=0.7)
        b = CandidateName(name="b", handler="chain", confidence=0.3)
        assert compare_by_node_scores(a, b) == 0
