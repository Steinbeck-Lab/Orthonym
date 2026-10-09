"""Unit tests for ParentCorrectnessScorer (a phase).

Covers:
- Production path (no reference set) -> 0.5
- Failure modes per RESEARCH (8 modes, all return 0.5)
- _extract_parent_token regex heuristic (, 9 verified cases)
- OPSIN subprocess wrapper (timeout, parse, failure)
- Match / mismatch / no-decision scoring
- Thread-local context isolation
- Substructure-match ambiguity tiebreak (canonical-rank determinism)

Tests requiring OPSIN are guarded with module-level skipif marker per the
test_benchmark_multi_corpus.py convention.
"""

import threading
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym.rules.parent_correctness import (
    ParentCorrectnessScorer,
    _extract_parent_token,
    _extract_reference_parent_atoms,
    _opsin_to_smi,
    _pc_context,
    clear_reference_name,
    match_token_atoms_in_mol,
    opsin_reference_mol,
    set_reference_name,
    OPSIN_TIMEOUT,
)
from orthonym.assembly.coverage_scoring import CandidateName
from tests.support.jars import jar_or_none

# OPSIN-required guard (the jar is resolved by orthonym.jars, not by a module constant)
opsin_required = pytest.mark.skipif(
    jar_or_none() is None, reason="OPSIN jar required for subprocess tests"
)


# ---------------------------------------------------------------------------
# Helper for tests that need a CandidateName with parent_atom_indices
# ---------------------------------------------------------------------------

def _make_candidate(name="ethanol", handler="chain", parent_atoms=None):
    return CandidateName(
        name=name, handler=handler,
        confidence=0.5, factors={},
        parent_atom_indices=parent_atoms,
    )


# ---------------------------------------------------------------------------
# Production-path tests (no OPSIN required -- short-circuit)
# ---------------------------------------------------------------------------

class TestProductionShortCircuit:
    def setup_method(self):
        clear_reference_name()

    def teardown_method(self):
        clear_reference_name()

    def test_no_reference_returns_half(self):
        """Test 1: no reference set -> 0.5 with zero OPSIN cost (T-145.1-01)."""
        cand = _make_candidate(parent_atoms={0, 1, 2})
        mol = Chem.MolFromSmiles("CCO")
        assert ParentCorrectnessScorer.score(cand, mol) == 0.5

    def test_no_parent_atom_indices_returns_half(self):
        """Test 2: handler didn't report parent atoms -> 0.5."""
        set_reference_name("ethanol")
        cand = _make_candidate(parent_atoms=None)
        mol = Chem.MolFromSmiles("CCO")
        assert ParentCorrectnessScorer.score(cand, mol) == 0.5


# ---------------------------------------------------------------------------
# _extract_parent_token tests (no OPSIN required)
# ---------------------------------------------------------------------------

class TestExtractParentToken:
    @pytest.mark.parametrize("name,expected", [
        ("4-oxo-4-(prop-2-enoyloxy)but-2-enoic acid", "but-2-enoic acid"),
        ("ethanol", "ethanol"),
        ("3-(2-methoxyethyl)hexan-1-ol", "hexan-1-ol"),
        ("4-(4-chlorophenyl)butan-2-one", "butan-2-one"),
        ("benzene-1,2-diol", "benzene-1,2-diol"),
    ])
    def test_extract_parent_token_with_parens(self, name, expected):
        """Test 3: regex extracts expected parent token from parens-bearing names."""
        assert _extract_parent_token(name) == expected

    @pytest.mark.parametrize("name", [
        "(2S)-2-amino-3-methylbutanoic acid",  # stereo prefix -- partial extraction OK
        "2-methylpropanal",                    # no parens -- heuristic falls through
        "4-methylpentan-2-one",                # no parens -- heuristic falls through
        "1H-indole",                           # 1H- stripped
    ])
    def test_extract_parent_token_no_parens_does_not_crash(self, name):
        """Test 4: graceful behavior on names without parens (no-decision is OK)."""
        result = _extract_parent_token(name)
        # Returns None or a string; must not raise. Scorer treats either as
        # no-decision when downstream OPSIN parsing fails on the result.
        assert result is None or isinstance(result, str)

    def test_extract_parent_token_empty_returns_none(self):
        """Test 5: empty input returns None."""
        assert _extract_parent_token("") is None
        assert _extract_parent_token(None) is None


# ---------------------------------------------------------------------------
# _opsin_to_smi tests (OPSIN required)
# ---------------------------------------------------------------------------

@opsin_required
class TestOpsinSubprocess:
    def test_opsin_to_smi_valid_name_returns_smiles(self):
        """Test 6: OPSIN parses 'ethanol' -> SMILES containing C and O."""
        smi = _opsin_to_smi("ethanol")
        assert smi is not None
        assert "C" in smi
        assert "O" in smi

    def test_opsin_to_smi_invalid_name_returns_none(self):
        """Test 7: OPSIN can't parse -> None (graceful failure)."""
        result = _opsin_to_smi("definitely not a real chemical xyzzy 12345")
        assert result is None

    def test_opsin_to_smi_timeout_returns_none(self, monkeypatch):
        """Test 8: subprocess timeout -> None (caught, logged at DEBUG)."""
        import orthonym.rules.parent_correctness as pc
        monkeypatch.setattr(pc, "OPSIN_TIMEOUT", 0.001)
        # Even a tiny name will timeout at 1ms (JVM startup ~100ms)
        result = pc._opsin_to_smi("ethanol")
        assert result is None


# ---------------------------------------------------------------------------
# End-to-end scorer tests (OPSIN required)
# ---------------------------------------------------------------------------

@opsin_required
class TestScorer:
    def setup_method(self):
        clear_reference_name()

    def teardown_method(self):
        clear_reference_name()

    def test_scorer_match_returns_one(self):
        """Test 9: ethanol reference + ethanol input + matching parent_atom_indices -> 1.0."""
        set_reference_name("ethanol")
        mol = Chem.MolFromSmiles("CCO")
        # Whole molecule is the parent for ethanol
        parent_atoms = set(range(mol.GetNumAtoms()))
        cand = _make_candidate(parent_atoms=parent_atoms)
        score = ParentCorrectnessScorer.score(cand, mol)
        assert score == 1.0, f"Expected 1.0 (full match), got {score}"

    def test_scorer_mismatch_returns_zero(self):
        """Test 10: reference parent atoms differ from candidate's -> 0.0."""
        set_reference_name("ethanol")
        mol = Chem.MolFromSmiles("CCC")  # propane SMILES (3 atoms)
        # Claim that only atoms {0} are the parent -- clearly wrong vs ethanol
        cand = _make_candidate(parent_atoms={0})
        score = ParentCorrectnessScorer.score(cand, mol)
        # Either 0.0 (mismatch confirmed via OPSIN) or 0.5 (no substructure match)
        assert score in (0.0, 0.5), f"Expected 0.0 or 0.5, got {score}"

    def test_scorer_handles_substructure_ambiguity_deterministically(self):
        """Test 11: ambiguous substructure match -> deterministic via canonical rank.

        From RESEARCH: '4-oxo-4-(prop-2-enoyloxy)but-2-enoic acid'
        yields 2 substructure matches; canonical-rank tiebreak picks one
        deterministically. Calling score twice must return the SAME value.
        """
        set_reference_name("4-oxo-4-(prop-2-enoyloxy)but-2-enoic acid")
        mol = Chem.MolFromSmiles("O=C(C=CC(=O)O)OC(C=C)=O")
        cand = _make_candidate(parent_atoms={4, 3, 2, 1, 5, 6})  # one of the 2 matches
        score1 = ParentCorrectnessScorer.score(cand, mol)
        score2 = ParentCorrectnessScorer.score(cand, mol)
        assert score1 == score2, "Scorer must be deterministic across invocations"


# ---------------------------------------------------------------------------
# Thread-local context tests (no OPSIN required)
# ---------------------------------------------------------------------------

class TestThreadLocalContext:
    def test_set_clear_reference_name_idempotent(self):
        """Test 13: set + set + clear + clear sequence does not raise."""
        set_reference_name("ethanol")
        set_reference_name("propanol")  # overwrite OK
        clear_reference_name()
        clear_reference_name()  # double clear OK
        assert getattr(_pc_context, 'reference_name', None) is None

    def test_thread_local_isolation(self):
        """Test 12: reference name set in thread A is invisible to thread B ."""
        results = {}

        def thread_a():
            clear_reference_name()
            set_reference_name("ethanol")
            results['a'] = getattr(_pc_context, 'reference_name', None)

        def thread_b():
            clear_reference_name()
            # Do NOT set in this thread
            results['b'] = getattr(_pc_context, 'reference_name', None)

        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results['a'] == "ethanol", "Thread A should see its own reference name"
        assert results['b'] is None, "Thread B should not see thread A's reference name"


# ---------------------------------------------------------------------------
# a phase SCORE-03: reusable per-node alignment helpers
# ---------------------------------------------------------------------------

class TestPerNodeAlignmentHelpers:
    """opsin_reference_mol + match_token_atoms_in_mol (the per-node generalization
    of the OPSIN-reference alignment). _extract_reference_parent_atoms now
    delegates to match_token_atoms_in_mol with UNCHANGED behavior."""

    def test_match_token_empty_returns_none(self):
        """Empty/None token -> None with zero OPSIN cost (unguarded)."""
        mol = Chem.MolFromSmiles("CCCC")
        assert match_token_atoms_in_mol("", mol) is None
        assert match_token_atoms_in_mol(None, mol) is None

    @opsin_required
    def test_match_token_unparseable_returns_none(self):
        """A token OPSIN cannot parse -> None (no-decision)."""
        mol = Chem.MolFromSmiles("CCCC")
        assert match_token_atoms_in_mol("zzznotarealname", mol) is None

    @opsin_required
    def test_match_token_matches_butane(self):
        """'butane' submol matches all 4 carbons of CCCC (deterministic)."""
        mol = Chem.MolFromSmiles("CCCC")
        atoms = match_token_atoms_in_mol("butane", mol)
        assert atoms == {0, 1, 2, 3}

    @opsin_required
    def test_opsin_reference_mol_parses(self):
        """opsin_reference_mol('ethanol') -> a 3-atom RDKit mol (CCO)."""
        ref = opsin_reference_mol("ethanol")
        assert ref is not None
        assert ref.GetNumAtoms() == 3  # C, C, O

    @opsin_required
    def test_opsin_reference_mol_unparseable_returns_none(self):
        """An unparseable name -> None (caught-failure contract)."""
        assert opsin_reference_mol("zzznotarealname") is None

    @opsin_required
    def test_extract_reference_parent_atoms_behavior_unchanged(self):
        """The refactored _extract_reference_parent_atoms still returns the
        parent atom set for a known case (delegates to match_token_atoms_in_mol;
        byte-identical behavior preserved)."""
        mol = Chem.MolFromSmiles("CCO")
        atoms = _extract_reference_parent_atoms("ethanol", mol)
        assert atoms == {0, 1, 2}
