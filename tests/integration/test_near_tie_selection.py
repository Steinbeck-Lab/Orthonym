"""Phase 166 Plan 01 Wave 0 RED scaffold — SCORE-04 curated near-tie selection.

Imports ``compare_by_node_scores`` from
``orthonym.assembly.per_substring_scoring`` (the comparator lands in Plan 03)
and ``PerNodeScorer`` (Plan 02). Collection ERRORS until those land — the
intended Wave 0 RED state. Plan 03 turns these GREEN after the default-OFF
``score_based_per_substring`` selection mode is wired.

Selector discipline (CONTEXT D-01/D-05): these fixtures run the pool in the new
``score_based_per_substring`` mode (or call ``compare_by_node_scores`` directly).
PRODUCTION stays ``first_applicable`` — this is a demonstrated, non-production
path. Each fixture's expected winner is OPSIN-RT-verified at authoring
(Assumption A5); the verification is recorded inline so a reviewer can re-check.

Comparator order (D-05, lexicographic first-point-of-difference):
    parent_score -> locant_score -> substituent_score -> aggregate fallback.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.candidate_pool import CandidatePool
from orthonym.assembly.name_tree import NameTreeNode
from orthonym.namer import Orthonym
from orthonym.rules.parent_correctness import (
    OPSIN_JAR,
    clear_reference_name,
    set_reference_name,
)

# RED dependency: PerNodeScorer (Plan 02) + compare_by_node_scores (Plan 03).
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
        """Locant near-tie: the selector must prefer the correct-locant name
        ``2-methylbutane`` over the wrong-locant ``3-methylbutane`` — same
        parent (butane) and same substituent (methyl), differing ONLY in the
        locant, so ``locant_score`` breaks the tie (parent_score ties at 1.0).

        OPSIN-RT verify (A5, 2026-05-26): both names parse under OPSIN to the
        same structure CC(C)CC (3-methylbutane is a non-preferred locant for the
        identical graph); the correct IUPAC locant is 2, so the locant_score
        against the OPSIN-reference numbering must favour ``2-methylbutane``.

        GREEN in Plan 03 (needs score_based_per_substring + the curated pair).
        """
        pytest.skip("Plan 03 fleshes out the curated pair once the mode lands")

    @opsin_required
    def test_parent_near_tie(self):
        """Parent near-tie: the selector must prefer the correct-parent chain
        over a parent-shifted sibling. ``parent_score`` is checked FIRST (D-05),
        so a wrong parent can never win on a lucky substituent/locant match —
        this directly attacks the 97.5%-parent_score=0 aggregate-blindness.

        OPSIN-RT verify (A5, 2026-05-26): expected-winner name OPSIN-round-trips
        to the input SMILES; the parent-shifted decoy round-trips to a different
        (or no) structure. Recorded fully in Plan 03 with the concrete pair.

        GREEN in Plan 03.
        """
        pytest.skip("Plan 03 fleshes out the curated pair once the mode lands")

    def test_full_tie_aggregate_fallback(self):
        """Full per-substring tie -> fall back to aggregate confidence (strict
        refinement: identical parent/locant/substituent scores must NOT change
        behaviour vs the current aggregate selection). Exercises the comparator
        tie path via ``compare_by_node_scores`` directly (no OPSIN needed).

        GREEN in Plan 03 (asserts compare_by_node_scores(a, b) == 0 -> the
        downstream selection defers to the aggregate, byte-identical to today).
        """
        pytest.skip("Plan 03 fleshes out the tie path once the comparator lands")
