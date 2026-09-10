"""a phase SCORE-04: curated near-tie selection proof set.

Each fixture is a structurally-equal candidate PAIR differing in exactly one
substring; the selector (the default-OFF ``score_based_per_substring`` mode, or
the comparator directly) must pick the OPSIN-RT-correct member. PRODUCTION stays
``first_applicable`` — this is a demonstrated, non-production path.

Comparator order (, lexicographic first-point-of-difference):
    parent_score -> locant_score -> substituent_score -> aggregate fallback.
A full per-substring tie defers to ``select_best_candidate`` (the aggregate) —
a STRICT refinement: no behaviour change on a tie.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.candidate_pool import CandidatePool
from orthonym.assembly.coverage_scoring import CandidateName, select_best_candidate
from orthonym.assembly.name_tree import NameTreeNode
from orthonym.namer import Orthonym, name_with_tree
from orthonym.rules.parent_correctness import (
    OPSIN_JAR,
    clear_reference_name,
    set_reference_name,
)
from orthonym.assembly.per_substring_scoring import (
    NodeScores,
    PerNodeScorer,
    compare_by_node_scores,
)

opsin_required = pytest.mark.skipif(
    not OPSIN_JAR.exists(), reason="OPSIN jar required for subprocess tests"
)


@pytest.mark.integration
class TestNearTieSelection:
    def setup_method(self):
        clear_reference_name()

    def teardown_method(self):
        clear_reference_name()

    @opsin_required
    def test_locant_near_tie(self):
        """Locant near-tie: the selector prefers correct-locant ``2-methylbutane``
        over wrong-locant ``3-methylbutane`` — same parent (butane), same methyl
        substituent, differing ONLY in the substituent locant. ``locant_score``
        breaks it (parent ties). The distinguishing locant lives on the methyl
        PREFIX node, so the comparator's tree-aggregated locant_score surfaces it.

        Exercises the REAL pipeline end-to-end: ``CandidatePool`` in the new mode,
        ``pool.add`` (which attaches node_scores via the real OPSIN-driven
        ``PerNodeScorer``), and ``best`` -> ``_best_two_tier(per_substring=True)``.

        OPSIN-RT verify (A5, 2026-05-26): name_compound('CC(C)CC') == '2-methylbutane';
        both '2-methylbutane' and '3-methylbutane' OPSIN-round-trip to the CC(C)CC
        graph (3-methyl is a non-preferred locant for the identical constitution),
        so the correct IUPAC locant is 2.
        """
        smi = "CC(C)CC"
        mol = Chem.MolFromSmiles(smi)
        feats = Orthonym()._perceive(mol, smi, Chem.MolToSmiles(mol))
        correct_tree = name_with_tree(smi).tree            # real 2-methylbutane tree
        assert correct_tree is not None
        wrong_tree = NameTreeNode(
            parent_stem="but",
            fragment_legacy="3-methylbutane",
            prefixes=(NameTreeNode(parent_stem="3-methyl", locants=(3,)),),
        )
        set_reference_name("2-methylbutane")
        try:
            pool = CandidatePool("score_based_per_substring")
            pool.add("2-methylbutane", "chain", feats, tree=correct_tree)
            pool.add("3-methylbutane", "chain", feats, tree=wrong_tree)
            winner = pool.best()
        finally:
            clear_reference_name()
        assert winner.name == "2-methylbutane"

    def test_parent_near_tie(self):
        """Parent near-tie (the aggregate-blindness kill): a parent-CORRECT /
        substituent-WRONG candidate must beat a parent-WRONG / substituent-RIGHT
        candidate — EVEN when the parent-wrong candidate is added first AND has
        the higher aggregate confidence. ``parent_score`` breaks FIRST , so
        a wrong parent can never win on a lucky substituent match: the exact
        failure this phase exists to kill.

        Synthetic node_scores (no OPSIN) make the parent/substituent contrast
        crisp and deterministic; the candidates go through ``pool.add`` (real
        features) so the full selection wiring is exercised, then node_scores +
        confidence are set to the controlled scenario.
        """
        smi = "CCCC"
        mol = Chem.MolFromSmiles(smi)
        feats = Orthonym()._perceive(mol, smi, Chem.MolToSmiles(mol))
        pool = CandidatePool("score_based_per_substring")
        # Parent-wrong added FIRST and with the HIGHER aggregate confidence.
        wrong = pool.add(
            "parent-wrong", "chain", feats,
            tree=NameTreeNode(parent_stem="w", suffix="x"))
        right = pool.add(
            "parent-correct", "chain", feats,
            tree=NameTreeNode(parent_stem="y", suffix="z"))
        wrong.node_scores = {id(wrong.tree): NodeScores(0.0, 0.5, 1.0)}
        wrong.confidence = 0.9
        right.node_scores = {id(right.tree): NodeScores(1.0, 0.5, 0.0)}
        right.confidence = 0.3
        # Comparator: parent breaks first -> parent-correct wins despite lower aggregate.
        assert compare_by_node_scores(right, wrong) == -1
        # End-to-end through the pool selection wiring.
        assert pool.best().name == "parent-correct"

    def test_full_tie_aggregate_fallback(self):
        """Full per-substring tie -> defers to the aggregate (STRICT refinement,
        NO behaviour change). Two candidates with IDENTICAL per-substring scores:
        ``compare_by_node_scores`` == 0, and the pool's winner equals EXACTLY what
        ``select_best_candidate`` (the aggregate) alone would pick. No OPSIN."""
        smi = "CCCC"
        mol = Chem.MolFromSmiles(smi)
        feats = Orthonym()._perceive(mol, smi, Chem.MolToSmiles(mol))
        pool = CandidatePool("score_based_per_substring")
        a = pool.add("cand-a", "chain", feats,
                     tree=NameTreeNode(parent_stem="a", suffix="x"))
        b = pool.add("cand-b", "chain", feats,
                     tree=NameTreeNode(parent_stem="b", suffix="x"))
        a.node_scores = {id(a.tree): NodeScores(1.0, 0.5, 0.5)}
        b.node_scores = {id(b.tree): NodeScores(1.0, 0.5, 0.5)}
        assert compare_by_node_scores(a, b) == 0           # identical per-substring
        # What the aggregate alone would pick from the same (post-tie) candidate set.
        expected = select_best_candidate(list(pool.all_candidates()))
        assert pool.best().name == expected.name           # tie -> aggregate decides
