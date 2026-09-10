"""Unit tests for a phase Tier-1 cascade filters.

Tests each filter,,,,
in isolation, plus the cascade short-circuit behavior of
CandidatePool._best_two_tier.

All filter functions have signature (List[CandidateName]) -> List[CandidateName]
and NEVER return empty list — if all candidates tie, all are returned.

Source for IUPAC rules:
  https://iupac.qmul.ac.uk/BlueBook/P4.html cascade)
  https://iupac.qmul.ac.uk/BlueBook/P5.html ring-on-tie)
"""
import pytest
from rdkit import Chem

from orthonym.assembly.candidate_pool import (
    CandidatePool,
    HANDLER_POLICIES,
    HandlerPolicy,
    P_44_1_2_ELEMENT_RANK,
    P_44_1_2_ELEMENT_SENIORITY,
    _TIER1_FILTERS,
    _filter_lowest_locants,
    _filter_max_multiple_bonds,
    _filter_max_pcg_count,
    _filter_max_skeletal_atoms,
    _filter_ring_over_chain_on_tie,
    _filter_senior_heteroatom_class,
    _has_iupac_locants,
)
from orthonym.assembly.coverage_scoring import CandidateName


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cand(
    name="x",
    handler="chain",
    confidence=0.5,
    factors=None,
    parent_atom_indices=None,
    parent_pcg_count=None,
    features_mol=None,
    ring_info=None,
):
    """Build a CandidateName with optional Tier-1 cascade fields."""
    cand = CandidateName(
        name=name,
        handler=handler,
        confidence=confidence,
        factors=factors or {},
    )
    cand.parent_atom_indices = parent_atom_indices
    cand.parent_pcg_count = parent_pcg_count
    if features_mol is not None:
        # _filter_senior_heteroatom_class falls back to this attr in Plan 02.
        cand._features_mol = features_mol
    if ring_info is not None:
        cand.ring_info = ring_info
    return cand


def _mol(smi):
    """Build an RDKit Mol from SMILES."""
    return Chem.MolFromSmiles(smi)


# ---------------------------------------------------------------------------
# TestFilterMaxPCGCount —
# ---------------------------------------------------------------------------

class TestFilterMaxPCGCount:
    """: max principal characteristic group count wins."""

    def test_single_winner(self):
        """: candidate with strictly higher PCG count wins alone."""
        cands = [
            _make_cand(name="a", parent_pcg_count=2),
            _make_cand(name="b", parent_pcg_count=1),
            _make_cand(name="c", parent_pcg_count=1),
        ]
        result = _filter_max_pcg_count(cands)
        assert len(result) == 1, (
            f"P-44.1.1 expected single winner, got {len(result)}"
        )
        assert result[0].name == "a"

    def test_tie_returns_all_tied(self):
        """: candidates tied at max are all returned."""
        cands = [
            _make_cand(name="a", parent_pcg_count=1),
            _make_cand(name="b", parent_pcg_count=1),
            _make_cand(name="c", parent_pcg_count=1),
        ]
        result = _filter_max_pcg_count(cands)
        assert len(result) == 3, (
            f"P-44.1.1 expected 3-way tie returns all, got {len(result)}"
        )

    def test_zero_count_returns_all(self):
        """: None counts as 0; all-None means all-tied at 0."""
        cands = [
            _make_cand(name="a", parent_pcg_count=None),
            _make_cand(name="b", parent_pcg_count=None),
        ]
        result = _filter_max_pcg_count(cands)
        assert len(result) == 2, (
            "P-44.1.1: all-None should be treated as all-zero (tied)"
        )

    def test_empty_input_returns_empty(self):
        """: empty input returns empty (degenerate case)."""
        assert _filter_max_pcg_count([]) == []


# ---------------------------------------------------------------------------
# TestFilterSeniorHeteroatomClass —
# ---------------------------------------------------------------------------

class TestFilterSeniorHeteroatomClass:
    """: senior heteroatom class wins (N > P >... > C)."""

    def test_n_beats_c(self):
        """: pyridine (N in ring) beats benzene (C-only ring)."""
        # Pyridine: c1ccncc1 — atom indices 0,1,2,3(N),4,5
        pyr_mol = _mol("c1ccncc1")
        # Benzene: c1ccccc1 — all C
        ben_mol = _mol("c1ccccc1")
        cands = [
            _make_cand(
                name="pyridine",
                handler="heterocycle",
                parent_atom_indices={0, 1, 2, 3, 4, 5},
                features_mol=pyr_mol,
            ),
            _make_cand(
                name="benzene",
                handler="benzene",
                parent_atom_indices={0, 1, 2, 3, 4, 5},
                features_mol=ben_mol,
            ),
        ]
        result = _filter_senior_heteroatom_class(cands)
        assert len(result) == 1, (
            f"P-44.1.2 expected single winner (N>C), got {len(result)}"
        )
        assert result[0].name == "pyridine"

    def test_p_beats_o(self):
        """: P (rank=1) beats O (rank=14)."""
        # PH3: just to give P-bearing parent (atoms = {0})
        p_mol = _mol("P")
        # H2O: just O
        o_mol = _mol("O")
        cands = [
            _make_cand(
                name="phosphine",
                handler="phosphine",
                parent_atom_indices={0},
                features_mol=p_mol,
            ),
            _make_cand(
                name="water",
                handler="chain",
                parent_atom_indices={0},
                features_mol=o_mol,
            ),
        ]
        result = _filter_senior_heteroatom_class(cands)
        assert len(result) == 1, (
            f"P-44.1.2 expected P>O, got {len(result)}"
        )
        assert result[0].name == "phosphine"

    def test_tie_same_class_returns_all(self):
        """P-44.1.2: two N-bearing parents tie at rank 0 — both returned."""
        pyr_mol = _mol("c1ccncc1")
        cands = [
            _make_cand(
                name="pyridineA",
                handler="heterocycle",
                parent_atom_indices={0, 1, 2, 3, 4, 5},
                features_mol=pyr_mol,
            ),
            _make_cand(
                name="pyridineB",
                handler="heterocycle",
                parent_atom_indices={0, 1, 2, 3, 4, 5},
                features_mol=pyr_mol,
            ),
        ]
        result = _filter_senior_heteroatom_class(cands)
        assert len(result) == 2, (
            "P-44.1.2: tied seniority returns all"
        )

    def test_unknown_element_treated_as_least_senior(self):
        """: F not in P_44_1_2_ELEMENT_SENIORITY -> sentinel rank;
        a candidate with C (in the table at rank=18) beats fluorine-only.
        """
        c_mol = _mol("C")
        # Fluorine ion — RDKit fragment with F atom only
        f_mol = _mol("F[H]")  # H-F, but parent atoms = {0} (the F)
        cands = [
            _make_cand(
                name="methane",
                handler="chain",
                parent_atom_indices={0},
                features_mol=c_mol,
            ),
            _make_cand(
                name="hf",
                handler="chain",
                parent_atom_indices={0},
                features_mol=f_mol,
            ),
        ]
        result = _filter_senior_heteroatom_class(cands)
        # C is at rank 18 (in the table); F gets sentinel rank 19 (not in
        # table). C is more senior -> single winner.
        assert len(result) == 1, (
            f"P-44.1.2 expected C beats unknown F, got {len(result)}"
        )
        assert result[0].name == "methane"

    def test_empty_parent_treated_as_C(self):
        """: candidate with empty parent_atom_indices ranks as
        sentinel (least senior); a candidate with C beats it."""
        c_mol = _mol("C")
        cands = [
            _make_cand(
                name="methane",
                handler="chain",
                parent_atom_indices={0},
                features_mol=c_mol,
            ),
            _make_cand(
                name="empty",
                handler="chain",
                parent_atom_indices=set(),
                features_mol=c_mol,
            ),
        ]
        result = _filter_senior_heteroatom_class(cands)
        # C in parent (rank 18) beats empty parent (sentinel rank 19).
        assert len(result) == 1, (
            f"P-44.1.2 expected C in parent beats empty, got {len(result)}"
        )
        assert result[0].name == "methane"


# ---------------------------------------------------------------------------
# TestFilterRingOverChainOnTie — /
# ---------------------------------------------------------------------------

class TestFilterRingOverChainOnTie:
    """: ring beats chain on tie. If any ring candidate exists,
    drop chain candidates."""

    def test_drop_chain_when_ring_present(self):
        """: chain + ring_a mix -> only ring kept."""
        cands = [
            _make_cand(name="chain", handler="chain"),
            _make_cand(name="ring", handler="benzene"),
        ]
        result = _filter_ring_over_chain_on_tie(cands)
        assert len(result) == 1, (
            f"P-52.2.8 expected ring wins, got {len(result)}"
        )
        assert result[0].name == "ring"

    def test_all_ring_returns_all(self):
        """: all-ring input returns all (identity)."""
        cands = [
            _make_cand(name="ring1", handler="benzene"),
            _make_cand(name="ring2", handler="heterocycle"),
        ]
        result = _filter_ring_over_chain_on_tie(cands)
        assert len(result) == 2, "P-52.2.8: all-ring returns all"

    def test_all_chain_returns_all(self):
        """: all-chain input returns all (identity)."""
        cands = [
            _make_cand(name="chainA", handler="chain"),
            _make_cand(name="chainB", handler="chain"),
        ]
        result = _filter_ring_over_chain_on_tie(cands)
        assert len(result) == 2, "P-52.2.8: all-chain returns all"

    def test_unknown_handler_treated_as_non_chain(self):
        """: handler not in HANDLER_POLICIES -> treated as
        non-chain (defensive default; kept in result)."""
        cands = [
            _make_cand(name="chain", handler="chain"),
            _make_cand(name="mystery", handler="zz_unknown_handler_zz"),
        ]
        result = _filter_ring_over_chain_on_tie(cands)
        # Unknown is non-chain; chain dropped.
        assert len(result) == 1, (
            "P-52.2.8: unknown handler is non-chain; chain dropped"
        )
        assert result[0].name == "mystery"


# ---------------------------------------------------------------------------
# TestFilterMaxSkeletalAtoms —
# ---------------------------------------------------------------------------

class TestFilterMaxSkeletalAtoms:
    """: max number of skeletal atoms in the parent wins."""

    def test_max_size_wins(self):
        """: largest parent wins alone."""
        cands = [
            _make_cand(name="ten", parent_atom_indices=set(range(10))),
            _make_cand(name="five", parent_atom_indices=set(range(5))),
            _make_cand(name="eight", parent_atom_indices=set(range(8))),
        ]
        result = _filter_max_skeletal_atoms(cands)
        assert len(result) == 1, (
            f"P-44.4.1.1 expected single winner, got {len(result)}"
        )
        assert result[0].name == "ten"

    def test_tie_returns_all_tied(self):
        """: candidates tied at max size all returned."""
        cands = [
            _make_cand(name="fiveA", parent_atom_indices=set(range(5))),
            _make_cand(name="fiveB", parent_atom_indices=set(range(5, 10))),
            _make_cand(name="three", parent_atom_indices=set(range(3))),
        ]
        result = _filter_max_skeletal_atoms(cands)
        assert len(result) == 2, (
            f"P-44.4.1.1 expected 2-way tie returns 2, got {len(result)}"
        )

    def test_none_parent_counts_as_zero(self):
        """: parent_atom_indices=None -> size 0; loses to size-1."""
        cands = [
            _make_cand(name="none", parent_atom_indices=None),
            _make_cand(name="one", parent_atom_indices={0}),
        ]
        result = _filter_max_skeletal_atoms(cands)
        assert len(result) == 1, (
            f"P-44.4.1.1 expected size-1 beats None, got {len(result)}"
        )
        assert result[0].name == "one"


# ---------------------------------------------------------------------------
# TestFilterMaxMultipleBonds —
# ---------------------------------------------------------------------------

class TestFilterMaxMultipleBonds:
    """: max (double + triple bonds in parent) wins."""

    def test_max_count_wins(self):
        """: highest multiple_bond_count wins alone."""
        cands = [
            _make_cand(name="three", factors={'multiple_bond_count': 3}),
            _make_cand(name="one",   factors={'multiple_bond_count': 1}),
            _make_cand(name="zero",  factors={'multiple_bond_count': 0}),
        ]
        result = _filter_max_multiple_bonds(cands)
        assert len(result) == 1, (
            f"P-44.4.1.2 expected single winner, got {len(result)}"
        )
        assert result[0].name == "three"

    def test_tie_returns_all_tied(self):
        """: candidates tied at max count all returned."""
        cands = [
            _make_cand(name="twoA", factors={'multiple_bond_count': 2}),
            _make_cand(name="twoB", factors={'multiple_bond_count': 2}),
            _make_cand(name="one",  factors={'multiple_bond_count': 1}),
        ]
        result = _filter_max_multiple_bonds(cands)
        assert len(result) == 2, (
            f"P-44.4.1.2 expected 2-way tie returns 2, got {len(result)}"
        )

    def test_missing_factor_treated_as_zero(self):
        """: missing 'multiple_bond_count' key -> 0."""
        cands = [
            _make_cand(name="missing", factors={}),
            _make_cand(name="one", factors={'multiple_bond_count': 1}),
        ]
        result = _filter_max_multiple_bonds(cands)
        assert len(result) == 1, (
            f"P-44.4.1.2 expected count-1 beats missing, got {len(result)}"
        )
        assert result[0].name == "one"


# ---------------------------------------------------------------------------
# TestTier1Cascade — full cascade on CandidatePool._best_two_tier
# ---------------------------------------------------------------------------

class TestTier1Cascade:
    """Cascade short-circuit behavior + Tier-2 fall-through."""

    def test_cascade_short_circuits_on_single_winner(self):
        """: when filter 1 produces single winner, later filters
        should NOT be invoked (short-circuit on len <= 1)."""
        pool = CandidatePool(selection_mode='score_based')
        # Build candidates where filter 1 (PCG count) produces a single winner.
        cand_a = _make_cand(
            name="winner",
            handler="chain",
            parent_pcg_count=2,
            parent_atom_indices={0, 1},
        )
        cand_b = _make_cand(
            name="loser1",
            handler="chain",
            parent_pcg_count=1,
            parent_atom_indices={0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        )
        cand_c = _make_cand(
            name="loser2",
            handler="chain",
            parent_pcg_count=0,
            parent_atom_indices={0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        )
        pool._candidates = [cand_a, cand_b, cand_c]
        result = pool._best_two_tier()
        # Filter 1 picks cand_a (PCG=2). Even though cand_b/cand_c have
        # bigger parent (would win step 4), cascade short-circuits.
        assert result is cand_a, (
            f"P-44.1 cascade: filter 1 winner should short-circuit; "
            f"got {result.name!r}"
        )

    def test_cascade_falls_through_to_tier2_on_full_tie(self):
        """ + Tier 2: when all 5 filters tie, defer to
        select_best_candidate. With identical confidences and same handler,
        select_best_candidate falls back to first-added or HANDLER_PRIORITY.
        Either way, a valid winner is returned (not None)."""
        pool = CandidatePool(selection_mode='score_based')
        # Two identical candidates (same fields => all 5 filters return both).
        cand_a = _make_cand(
            name="alphaA",
            handler="chain",
            confidence=0.5,
            parent_pcg_count=0,
            parent_atom_indices={0, 1, 2},
            factors={'multiple_bond_count': 0},
        )
        cand_b = _make_cand(
            name="alphaB",
            handler="chain",
            confidence=0.5,
            parent_pcg_count=0,
            parent_atom_indices={0, 1, 2},
            factors={'multiple_bond_count': 0},
        )
        pool._candidates = [cand_a, cand_b]
        result = pool._best_two_tier()
        # After all 5 filters tie, select_best_candidate is invoked.
        # Result must be one of the two (deterministic per HANDLER_PRIORITY
        # / first-added ordering).
        assert result is not None, "Tier 2 fall-through must return a winner"
        assert result in (cand_a, cand_b), (
            f"Tier 2 must return one of the input candidates, got {result}"
        )

    def test_step6_skipped_when_no_iupac_locants(self):
        """: step 6 (lowest locants) is gated on iupac_locants being
        populated. In a phase, no candidate has iupac_locants -> step 6
        is a no-op; cascade falls through to Tier 2."""
        pool = CandidatePool(selection_mode='score_based')
        cand_a = _make_cand(
            name="a",
            handler="chain",
            confidence=0.5,
            parent_pcg_count=1,
            parent_atom_indices={0, 1},
            factors={'multiple_bond_count': 0},
        )
        cand_b = _make_cand(
            name="b",
            handler="chain",
            confidence=0.5,
            parent_pcg_count=1,
            parent_atom_indices={0, 1},
            factors={'multiple_bond_count': 0},
        )
        pool._candidates = [cand_a, cand_b]
        # Without iupac_locants, step 6 is no-op; Tier 2 chooses winner.
        # (If step 6 incorrectly fired with proxy locants, behavior would
        # be unpredictable. Here we just verify a winner is returned.)
        result = pool._best_two_tier()
        assert result is not None, (
            "D-02: step 6 no-op + Tier 2 must produce a winner"
        )
        assert result in (cand_a, cand_b)

    def test_empty_pool_returns_none(self):
        """Degenerate case: empty pool -> None."""
        pool = CandidatePool(selection_mode='score_based')
        assert pool._best_two_tier() is None


# ---------------------------------------------------------------------------
# TestHasIupacLocants — / safe probe
# ---------------------------------------------------------------------------

class TestHasIupacLocants:
    """: Safe probe for whether step 6 (lowest locants) can run."""

    def test_returns_false_when_all_missing(self):
        """: no candidate has ring_info['iupac_locants'] -> False."""
        cands = [
            _make_cand(name="a"),
            _make_cand(name="b"),
        ]
        assert _has_iupac_locants(cands) is False, (
            "D-02: all-missing iupac_locants -> step 6 should be no-op"
        )

    def test_returns_false_when_one_missing(self):
        """: ALL candidates must have iupac_locants for True."""
        cands = [
            _make_cand(name="a", ring_info={'iupac_locants': {0: 1, 1: 2}}),
            _make_cand(name="b", ring_info={'iupac_locants': {0: 1, 1: 2}}),
            _make_cand(name="c"),  # missing
        ]
        assert _has_iupac_locants(cands) is False, (
            "D-02: even one missing -> step 6 should be no-op"
        )

    def test_returns_true_when_all_present(self):
        """: ALL candidates have non-empty iupac_locants -> True."""
        cands = [
            _make_cand(name="a", ring_info={'iupac_locants': {0: 1, 1: 2}}),
            _make_cand(name="b", ring_info={'iupac_locants': {3: 1, 4: 2}}),
        ]
        assert _has_iupac_locants(cands) is True, (
            "D-02: all-populated iupac_locants -> step 6 may run "
            "(Phase 147 activation path)"
        )

    def test_empty_dict_is_treated_as_missing(self):
        """: empty iupac_locants dict is falsy -> False."""
        cands = [
            _make_cand(name="a", ring_info={'iupac_locants': {}}),
            _make_cand(name="b", ring_info={'iupac_locants': {}}),
        ]
        assert _has_iupac_locants(cands) is False, (
            "D-02: empty iupac_locants dict treated as missing"
        )
