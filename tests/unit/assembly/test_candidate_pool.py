"""Unit tests for CandidatePool, HandlerPolicy, HANDLER_POLICIES, and the
thread-local pool store (Phase 145.1 SC-1).

Tests behavior-preserving extraction of composer.py handler cascade.
Critical regression tests:
- test_pool_add_does_NOT_pass_parent_atom_indices_* proves Risk 1 mitigation
- test_module_level_parent_correctness_binding proves ISS-005 remediation
"""

import inspect
import threading
from unittest.mock import patch

import pytest
from rdkit import Chem

from orthonym.assembly.candidate_pool import (
    CandidatePool,
    HandlerPolicy,
    HANDLER_POLICIES,
    clear_pool,
    get_current_pool,
    _pool_store,
)
import orthonym.assembly.candidate_pool as cp_module
from orthonym.assembly.coverage_scoring import (
    CandidateName,
    HANDLER_PRIORITY,
    CONFIDENCE_GATE_THRESHOLD,
)
from orthonym.namer import MolecularFeatures


# ---------------------------------------------------------------------------
# Test helper (mirrors tests/unit/rules/test_coverage_scoring.py pattern)
# ---------------------------------------------------------------------------

def _make_features(smiles, functional_groups=None, principal_group=None):
    """Create a minimal MolecularFeatures for testing pool.add()."""
    mol = Chem.MolFromSmiles(smiles)
    f = MolecularFeatures(mol=mol, smiles=smiles)
    if functional_groups is not None:
        f.functional_groups = functional_groups
    if principal_group is not None:
        f.principal_group = principal_group
    return f


# ---------------------------------------------------------------------------
# HandlerPolicy + HANDLER_POLICIES tests
# ---------------------------------------------------------------------------

class TestHandlerPolicy:
    def test_handler_policy_creation(self):
        """Test 1: HandlerPolicy dataclass instantiation with defaults."""
        p = HandlerPolicy(handler_id='x', tier='ring_a', priority=4)
        assert p.handler_id == 'x'
        assert p.tier == 'ring_a'
        assert p.priority == 4
        assert p.min_ratio_accept is None
        assert p.cascade_ratio_min is None
        assert p.gate_threshold is None
        assert p.direct_return is False

    def test_handler_policies_dict_populated(self):
        """Test 2: HANDLER_POLICIES contains all expected handler IDs."""
        expected_handlers = {
            'complex_ring', 'heterocycle', 'benzene', 'chain',
            'oxime', 'hydrazone', 'isocyanate', 'isothiocyanate',
            'carbamic_acid', 'carbamate', 'urea', 'guanidine',
            'sulfoxide', 'sulfone', 'thioether', 'boronic_acid',
            'partial_sat', 'polycyclic', 'acid_halide', 'anhydride',
            'lactone', 'lactam', 'ring_ester', 'polyfunctional',
            'multi_ester', 'ester', 'phosphine_oxide', 'phosphate_ester',
            'phosphine', 'phosphinic_acid', 'ring_assembly',
            'ring_nitrile', 'amide', 'amine', 'n_oxide',
        }
        missing = expected_handlers - set(HANDLER_POLICIES.keys())
        assert not missing, f"Missing handlers: {missing}"

    def test_handler_policies_complex_ring_gates_match_composer(self):
        """Test 3: complex_ring carries BOTH gates verbatim from composer.py."""
        p = HANDLER_POLICIES['complex_ring']
        assert p.cascade_ratio_min == 0.40, "composer.py:1199 _CASCADE_RATIO_MIN"
        assert p.min_ratio_accept == 0.30, "composer.py:1323 _MIN_RATIO_ACCEPT"
        assert p.direct_return is False, "Tier A — pool decides"
        assert p.tier == 'ring_a'

    @pytest.mark.parametrize("handler_id", [
        'oxime', 'hydrazone', 'isocyanate', 'isothiocyanate',
        'carbamic_acid', 'carbamate', 'urea', 'guanidine',
        'sulfoxide', 'sulfone', 'thioether', 'boronic_acid', 'partial_sat',
    ])
    def test_handler_policies_tier_b_gates_match_composer(self, handler_id):
        """Test 4: All 13 Tier B handlers carry gate_threshold=0.40."""
        p = HANDLER_POLICIES[handler_id]
        assert p.gate_threshold == 0.40, f"{handler_id}: must mirror CONFIDENCE_GATE_THRESHOLD"
        assert p.gate_threshold == CONFIDENCE_GATE_THRESHOLD, \
            f"{handler_id}: must equal coverage_scoring.CONFIDENCE_GATE_THRESHOLD"
        assert p.direct_return is True, f"{handler_id}: Tier B is first-applicable"
        assert p.tier == 'ring_b'

    def test_handler_policies_chain_priority_one(self):
        """Test 5: chain has priority=1 (D-02 fallback; Phase 146 raises)."""
        p = HANDLER_POLICIES['chain']
        assert p.priority == 1
        assert p.tier == 'chain'
        assert p.direct_return is False
        # Gate fields must all be None (no chain gate in current composer)
        assert p.cascade_ratio_min is None
        assert p.min_ratio_accept is None
        assert p.gate_threshold is None

    @pytest.mark.parametrize("handler_id", [
        'complex_ring', 'heterocycle', 'benzene', 'chain', 'oxime',
        'polycyclic', 'n_oxide', 'amine', 'polyfunctional',
    ])
    def test_handler_policies_priority_pulled_from_handler_priority(self, handler_id):
        """Test 18 (ISS-002 REGRESSION): every priority pulled from HANDLER_PRIORITY.

        Includes n_oxide, amine, polyfunctional which Plan 02 Task 2 added to
        HANDLER_PRIORITY. Proves single-source-of-truth invariant — no
        hardcoded priority literals in HANDLER_POLICIES."""
        assert HANDLER_POLICIES[handler_id].priority == HANDLER_PRIORITY[handler_id]

    def test_handler_policies_excludes_ion_classes(self):
        """Test 19: salt/ion/radical/simple_molecule excluded per RESEARCH §9.2 + ISS-001."""
        for excluded in ('salt', 'ion', 'radical', 'simple_molecule'):
            assert excluded not in HANDLER_POLICIES, \
                f"{excluded!r} should NOT be in HANDLER_POLICIES (different naming semantics)"


# ---------------------------------------------------------------------------
# CandidatePool init tests
# ---------------------------------------------------------------------------

class TestCandidatePoolInit:
    def test_pool_init_first_applicable(self):
        """Test 6: default selection_mode is 'first_applicable'."""
        pool = CandidatePool()
        assert pool.selection_mode == 'first_applicable'
        assert pool._candidates == []
        assert pool._direct_return_winner is None

    def test_pool_init_score_based(self):
        """Test 7: score_based mode accepted."""
        pool = CandidatePool(selection_mode='score_based')
        assert pool.selection_mode == 'score_based'

    def test_pool_init_invalid_mode_raises(self):
        """Test 8: unknown selection_mode raises AssertionError."""
        with pytest.raises(AssertionError):
            CandidatePool(selection_mode='bogus')


# ---------------------------------------------------------------------------
# CandidatePool selection tests
# ---------------------------------------------------------------------------

class TestCandidatePoolSelection:
    def test_pool_first_applicable_returns_first_added(self):
        """Test 9: first_applicable returns first-added regardless of confidence."""
        pool = CandidatePool(selection_mode='first_applicable')
        # Add three candidates manually (bypassing add() to control confidence)
        c1 = CandidateName(name="ethanol", handler='chain', confidence=0.4)
        c2 = CandidateName(name="ethyl alcohol", handler='chain', confidence=0.9)
        c3 = CandidateName(name="hydroxyethane", handler='chain', confidence=0.7)
        pool._candidates = [c1, c2, c3]
        best = pool.best()
        assert best is c1, "first_applicable returns first-added (not highest confidence)"

    def test_pool_score_based_returns_highest_confidence(self):
        """Test 10: score_based returns highest-confidence via select_best_candidate."""
        pool = CandidatePool(selection_mode='score_based')
        c1 = CandidateName(name="a", handler='chain', confidence=0.4)
        c2 = CandidateName(name="b", handler='benzene', confidence=0.9)
        c3 = CandidateName(name="c", handler='heterocycle', confidence=0.7)
        pool._candidates = [c1, c2, c3]
        best = pool.best()
        assert best is c2, "score_based returns highest-confidence candidate"

    def test_pool_best_empty_returns_none(self):
        """pool.best() with no candidates returns None."""
        pool = CandidatePool()
        assert pool.best() is None

    def test_pool_direct_return_winner_short_circuits(self):
        """Test 17: in first_applicable, first-added direct_return wins."""
        pool = CandidatePool(selection_mode='first_applicable')
        c1 = CandidateName(name="polycyclic_name", handler='polycyclic', confidence=0.6)
        c2 = CandidateName(name="complex_ring_name", handler='complex_ring', confidence=0.9)
        pool._candidates = [c1, c2]
        assert pool.best() is c1


# ---------------------------------------------------------------------------
# CandidatePool.add() — gate behavior + post-hoc parent_atom_indices
# ---------------------------------------------------------------------------

class TestCandidatePoolAdd:
    def test_pool_add_returns_candidate_for_non_gated_handler(self):
        """pool.add() with non-Tier-B handler stores and returns candidate."""
        features = _make_features("CCO")
        pool = CandidatePool()
        cand = pool.add("ethanol", "chain", features)
        assert cand is not None
        assert cand.name == "ethanol"
        assert cand.handler == "chain"
        assert len(pool.all_candidates()) == 1

    def test_pool_tier_b_gate_rejects_below_threshold(self):
        """Test 11: Tier B handler with confidence < 0.40 returns None."""
        features = _make_features("CCO")
        pool = CandidatePool()
        # Patch compute_confidence to return a candidate with confidence < 0.40
        with patch(
            "orthonym.assembly.candidate_pool.compute_confidence",
            return_value=CandidateName(name="x", handler="oxime", confidence=0.3),
        ):
            result = pool.add("x", "oxime", features)
        assert result is None, "gate should reject confidence < 0.40"
        assert len(pool.all_candidates()) == 0, "rejected candidate not stored"

    def test_pool_tier_b_gate_accepts_at_threshold(self):
        """Test 12: Tier B handler with confidence == 0.40 accepts."""
        features = _make_features("CCO")
        pool = CandidatePool()
        with patch(
            "orthonym.assembly.candidate_pool.compute_confidence",
            return_value=CandidateName(name="x", handler="oxime", confidence=0.40),
        ):
            result = pool.add("x", "oxime", features)
        assert result is not None
        assert len(pool.all_candidates()) == 1

    def test_pool_parent_atom_indices_propagated_post_hoc(self):
        """Test 15: parent_atom_indices set POST-HOC on returned candidate."""
        features = _make_features("CCO")
        pool = CandidatePool()
        parent_atoms = {0, 1, 2}
        cand = pool.add("ethanol", "chain", features, parent_atom_indices=parent_atoms)
        assert cand is not None
        assert cand.parent_atom_indices == parent_atoms

    def test_pool_add_does_NOT_pass_parent_atom_indices_to_compute_confidence(self):
        """Test 16 (Risk 1 REGRESSION): byte-identical proof.

        compute_confidence MUST NOT receive parent_atom_indices as a
        kwarg/positional arg, because doing so changes atom_coverage
        (coverage_scoring.py:284-287) and breaks byte-identical for
        handlers that previously passed None.
        """
        features = _make_features("CCO")
        pool = CandidatePool()
        with patch(
            "orthonym.assembly.candidate_pool.compute_confidence",
            return_value=CandidateName(name="ethanol", handler="chain", confidence=0.5),
        ) as mock_cc:
            pool.add("ethanol", "chain", features, parent_atom_indices={0, 1, 2})
        # Verify compute_confidence was called WITHOUT parent_atom_indices
        assert mock_cc.called
        args, kwargs = mock_cc.call_args
        # parent_atom_indices must not appear in kwargs (or, if Python
        # passed it, must be None/absent — but the safer assertion is
        # that the call signature only includes name/handler/features)
        assert 'parent_atom_indices' not in kwargs, \
            "RISK 1 VIOLATION: pool.add() must not pass parent_atom_indices to compute_confidence"

    def test_pool_unknown_handler_id_no_gate(self):
        """Test 20: unknown handler_id accepted with no gate enforcement."""
        features = _make_features("CCO")
        pool = CandidatePool()
        cand = pool.add("ethanol", "totally_unknown_handler_xyz", features)
        # No policy → no gate → candidate stored regardless of confidence
        assert cand is not None
        assert len(pool.all_candidates()) == 1


# ---------------------------------------------------------------------------
# Thread-local pool lifecycle tests
# ---------------------------------------------------------------------------

class TestPoolLifecycle:
    def test_clear_pool_resets(self):
        """Test 14: clear_pool() resets thread-local state."""
        features = _make_features("CCO")
        clear_pool()
        pool = get_current_pool()
        pool.add("ethanol", "chain", features)
        assert len(pool.all_candidates()) == 1
        clear_pool()
        new_pool = get_current_pool()
        assert len(new_pool.all_candidates()) == 0

    def test_get_current_pool_lazy_init(self):
        """get_current_pool() creates pool on first access in a fresh thread."""
        # Reset by deleting the attribute (simulates fresh thread)
        if hasattr(_pool_store, 'pool'):
            delattr(_pool_store, 'pool')
        pool = get_current_pool()
        assert isinstance(pool, CandidatePool)
        assert pool.selection_mode == 'first_applicable'

    def test_pool_thread_local_isolated(self):
        """Test 13: pools in different threads do not share candidates."""
        features = _make_features("CCO")
        results = {}
        results_lock = threading.Lock()

        def worker(name, handler, key):
            clear_pool()
            pool = get_current_pool()
            pool.add(name, handler, features)
            with results_lock:
                results[key] = len(pool.all_candidates())

        t1 = threading.Thread(target=worker, args=("ethanol", "chain", "t1"))
        t2 = threading.Thread(target=worker, args=("methanol", "chain", "t2"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Each thread sees its own pool with exactly 1 candidate
        assert results == {"t1": 1, "t2": 1}, \
            f"Thread isolation broken or test races; got {results!r}"


# ---------------------------------------------------------------------------
# ISS-005 module-level binding regression test
# ---------------------------------------------------------------------------

class TestModuleLevelImportBinding:
    def test_module_level_parent_correctness_binding_exists(self):
        """Test 21 (ISS-005 REGRESSION): ParentCorrectnessScorer is bound at
        MODULE LEVEL via try/except ImportError. Either the class or None
        — both indicate the binding lives at module scope (not per-call)."""
        assert hasattr(cp_module, 'ParentCorrectnessScorer'), \
            "Module must expose ParentCorrectnessScorer binding (class or None)"
        # Binding must be either the class or None (not undefined / missing).
        assert cp_module.ParentCorrectnessScorer is None \
            or callable(cp_module.ParentCorrectnessScorer), \
            f"Binding must be class or None; got {cp_module.ParentCorrectnessScorer!r}"

    def test_pool_add_does_not_re_import_inside_body(self):
        """Test 21 follow-up (ISS-005 REGRESSION): pool.add() body must NOT
        contain a per-call `from ..rules.parent_correctness import` statement.
        That would defeat the module-level binding optimization."""
        src = inspect.getsource(CandidatePool.add)
        # Must not have a per-call import inside the method body
        assert 'from ..rules.parent_correctness import' not in src, (
            "ISS-005 VIOLATION: pool.add() must reference the MODULE-LEVEL "
            "ParentCorrectnessScorer binding, not re-import per call. "
            "Move the try/except ImportError to module scope."
        )
        # Must reference the module-level binding (either by name check or call)
        assert 'ParentCorrectnessScorer' in src, (
            "pool.add() must reference ParentCorrectnessScorer (module-level binding)"
        )
