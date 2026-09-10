"""a phase Plan 02 Task 1: tests for CandidateName.ring_info field
+ pool.add(ring_info=...) POST-HOC attachment + _has_iupac_locants probe.

Source: https://iupac.qmul.ac.uk/BlueBook/P4.html +
Source: a phase internal notes (dispatch helper consumer);
        a phase internal notes (Risk 1 byte-identical guarantee on
        compute_confidence — ring_info MUST be POST-HOC like
        parent_atom_indices and parent_pcg_count).
"""

import pytest


def test_candidate_name_default_ring_info_is_none():
    """CandidateName instantiates with ring_info defaulting to None.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html +
    Source: a phase internal notes.
    """
    from orthonym.assembly.coverage_scoring import CandidateName
    c = CandidateName(name='foo', handler='benzene')
    assert c.ring_info is None


def test_candidate_name_ring_info_constructor_kwarg():
    """CandidateName accepts ring_info via constructor kwarg.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html +
    Source: a phase internal notes.
    """
    from orthonym.assembly.coverage_scoring import CandidateName
    c = CandidateName(
        name='foo',
        handler='benzene',
        ring_info={'iupac_locants': {0: 1, 1: 2}},
    )
    assert c.ring_info == {'iupac_locants': {0: 1, 1: 2}}


def test_candidate_name_ring_info_post_hoc_assignment():
    """ring_info can be set POST-HOC after construction (attach pattern).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html +
    Source: a phase internal notes.
    """
    from orthonym.assembly.coverage_scoring import CandidateName
    c = CandidateName(name='foo', handler='benzene')
    assert c.ring_info is None
    c.ring_info = {'iupac_locants': {0: 1}}
    assert c.ring_info == {'iupac_locants': {0: 1}}


def test_compute_confidence_byte_identical_with_ring_info_field():
    """Risk 1 guard: compute_confidence output is byte-identical regardless
    of whether the new ring_info field exists on CandidateName.

    Calling compute_confidence twice on the same SMILES must produce
    identical confidence values — this protects against accidental wiring
    of ring_info INTO compute_confidence (which would break the
    byte-identical contract per a phase).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html +
    Source: a phase internal notes.
    """
    from rdkit import Chem
    from orthonym.assembly.coverage_scoring import compute_confidence
    from orthonym.namer import Orthonym
    mol = Chem.MolFromSmiles('CCO')
    canonical = Chem.MolToSmiles(mol)
    feats = Orthonym()._perceive(mol, 'CCO', canonical)
    a = compute_confidence('ethanol', 'chain', feats)
    b = compute_confidence('ethanol', 'chain', feats)
    assert a.confidence == b.confidence
    assert a.factors == b.factors


def test_pool_add_attaches_ring_info_post_hoc():
    """pool.add(..., ring_info={'iupac_locants': {...}}) attaches the dict
    to the returned CandidateName POST-HOC (after compute_confidence).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html +
    Source: a phase internal notes; a phase internal notes.
    """
    from rdkit import Chem
    from orthonym.assembly.candidate_pool import CandidatePool
    from orthonym.namer import Orthonym
    mol = Chem.MolFromSmiles('CCO')
    canonical = Chem.MolToSmiles(mol)
    feats = Orthonym()._perceive(mol, 'CCO', canonical)
    pool = CandidatePool(selection_mode='score_based')
    cand = pool.add(
        'ethanol', 'chain', feats,
        ring_info={'iupac_locants': {0: 1}},
    )
    assert cand is not None
    assert cand.ring_info == {'iupac_locants': {0: 1}}


def test_pool_add_without_ring_info_kwarg_leaves_none():
    """pool.add without the ring_info kwarg leaves cand.ring_info as None
    (back-compat: existing call sites need not pass the new kwarg).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html +
    Source: a phase internal notes.
    """
    from rdkit import Chem
    from orthonym.assembly.candidate_pool import CandidatePool
    from orthonym.namer import Orthonym
    mol = Chem.MolFromSmiles('CCO')
    canonical = Chem.MolToSmiles(mol)
    feats = Orthonym()._perceive(mol, 'CCO', canonical)
    pool = CandidatePool(selection_mode='score_based')
    cand = pool.add('ethanol', 'chain', feats)
    assert cand is not None
    assert cand.ring_info is None


def test_has_iupac_locants_truth_table():
    """_has_iupac_locants returns False if ANY candidate has unpopulated
    iupac_locants (per / conservative gate — spiro/VB stubs and
    candidates with ring_info=None correctly disable cascade step 6).

    Truth table:
      both None -> False
      one None, one populated -> False
      both populated-non-empty -> True
      one None-stub (iupac_locants=None inside dict) + populated -> False

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html +
    Source: a phase internal notes,.
    """
    from orthonym.assembly.coverage_scoring import CandidateName
    from orthonym.assembly.candidate_pool import _has_iupac_locants
    a_none = CandidateName(name='a', handler='x')
    b_none = CandidateName(name='b', handler='x')
    a_pop = CandidateName(
        name='a', handler='benzene',
        ring_info={'iupac_locants': {0: 1, 1: 2}},
    )
    b_pop = CandidateName(
        name='b', handler='benzene',
        ring_info={'iupac_locants': {0: 1, 1: 2}},
    )
    a_stub = CandidateName(
        name='a', handler='spiro',
        ring_info={'iupac_locants': None},
    )

    assert _has_iupac_locants([a_none, b_none]) is False
    assert _has_iupac_locants([a_none, b_pop]) is False
    assert _has_iupac_locants([a_pop, b_pop]) is True
    assert _has_iupac_locants([a_stub, b_pop]) is False
    # Empty list: vacuously True (no failing candidate).
    assert _has_iupac_locants([]) is True
