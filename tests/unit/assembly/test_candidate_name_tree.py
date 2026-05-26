"""Phase 165 Plan 02 Task 1: CandidateName.tree field + pool.add(tree=...)
POST-HOC attachment + best().tree surfacing.

Mirrors test_candidate_name_ring_info.py (the ring_info post-hoc analog).
Source: Phase 165 SCORE-01 (D-03 per-candidate tree carry);
        Phase 146 CONTEXT D-19 (Risk 1 byte-identical guarantee — tree MUST be
        POST-HOC like parent_atom_indices / ring_info, never into compute_confidence).
"""

import pytest


def test_candidate_name_default_tree_is_none():
    """CandidateName instantiates with tree defaulting to None."""
    from orthonym.assembly.coverage_scoring import CandidateName
    c = CandidateName(name='foo', handler='chain')
    assert c.tree is None


def test_candidate_name_tree_post_hoc_assignment():
    """tree can be set POST-HOC after construction (D-03 attach pattern)."""
    from orthonym.assembly.coverage_scoring import CandidateName
    from orthonym.assembly.name_tree import NameTreeNode
    c = CandidateName(name='ethanol', handler='chain')
    assert c.tree is None
    node = NameTreeNode(parent_stem='ethan', suffix='ol')
    c.tree = node
    assert c.tree is node


def test_pool_add_attaches_tree_post_hoc():
    """pool.add(..., tree=node) attaches the NameTreeNode POST-HOC and
    best().tree surfaces it for the winning candidate."""
    from rdkit import Chem
    from orthonym.assembly.candidate_pool import CandidatePool
    from orthonym.assembly.name_tree import NameTreeNode
    from orthonym.namer import Orthonym
    mol = Chem.MolFromSmiles('CCO')
    canonical = Chem.MolToSmiles(mol)
    feats = Orthonym()._perceive(mol, 'CCO', canonical)
    node = NameTreeNode(parent_stem='ethan', suffix='ol')
    pool = CandidatePool(selection_mode='score_based')
    cand = pool.add('ethanol', 'chain', feats, tree=node)
    assert cand is not None
    assert cand.tree is node
    assert pool.best().tree is node


def test_pool_add_without_tree_kwarg_leaves_none():
    """pool.add() without the tree kwarg leaves cand.tree as None (back-compat:
    existing call sites need not pass the new kwarg; that candidate is coarse-
    bucket counted in Plan 04)."""
    from rdkit import Chem
    from orthonym.assembly.candidate_pool import CandidatePool
    from orthonym.namer import Orthonym
    mol = Chem.MolFromSmiles('CCO')
    canonical = Chem.MolToSmiles(mol)
    feats = Orthonym()._perceive(mol, 'CCO', canonical)
    pool = CandidatePool(selection_mode='score_based')
    cand = pool.add('ethanol', 'chain', feats)
    assert cand is not None
    assert cand.tree is None
    assert pool.best().tree is None


def test_pool_tree_not_threaded_into_compute_confidence():
    """Risk 1 guard: adding a tree does NOT change the candidate's confidence
    (tree is POST-HOC; never passed into compute_confidence per D-19)."""
    from rdkit import Chem
    from orthonym.assembly.candidate_pool import CandidatePool
    from orthonym.assembly.name_tree import NameTreeNode
    from orthonym.namer import Orthonym
    mol = Chem.MolFromSmiles('CCO')
    canonical = Chem.MolToSmiles(mol)
    feats = Orthonym()._perceive(mol, 'CCO', canonical)
    node = NameTreeNode(parent_stem='ethan', suffix='ol')
    with_tree = CandidatePool('score_based').add('ethanol', 'chain', feats, tree=node)
    without_tree = CandidatePool('score_based').add('ethanol', 'chain', feats)
    assert with_tree.confidence == without_tree.confidence
    assert with_tree.factors == without_tree.factors
