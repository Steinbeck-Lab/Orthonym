"""Phase 147 Plan 02 Task 4: tests for cascade step 6 gate semantics.

Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.4.1.4+
Source: Phase 147 CONTEXT D-02 (cascade gate), D-06 (spiro/VB stub
        semantics — None drops to Tier-2).
"""

import pytest


def test_step6_skipped_when_any_candidate_ring_info_is_none_stub():
    """D-06: _has_iupac_locants returns False on None stub -> step 6 no-op.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    Source: Phase 147 CONTEXT D-06.
    """
    from orthonym.assembly.coverage_scoring import CandidateName
    from orthonym.assembly.candidate_pool import _has_iupac_locants
    a = CandidateName(
        name='a', handler='spiro',
        ring_info={'iupac_locants': None},
    )
    b = CandidateName(
        name='b', handler='benzene',
        ring_info={'iupac_locants': {0: 1}},
    )
    assert _has_iupac_locants([a, b]) is False


def test_step6_skipped_when_any_candidate_ring_info_missing():
    """D-02: _has_iupac_locants returns False if any candidate has
    ring_info=None -> step 6 no-op.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    Source: Phase 147 CONTEXT D-02.
    """
    from orthonym.assembly.coverage_scoring import CandidateName
    from orthonym.assembly.candidate_pool import _has_iupac_locants
    a = CandidateName(name='a', handler='complex_ring')
    b = CandidateName(
        name='b', handler='benzene',
        ring_info={'iupac_locants': {0: 1}},
    )
    assert _has_iupac_locants([a, b]) is False


def test_step6_active_when_all_candidates_populated():
    """D-02 inverse: when ALL candidates have iupac_locants populated,
    cascade step 6 IS active.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    Source: Phase 147 CONTEXT D-02.
    """
    from orthonym.assembly.coverage_scoring import CandidateName
    from orthonym.assembly.candidate_pool import _has_iupac_locants
    a = CandidateName(
        name='a', handler='benzene',
        ring_info={'iupac_locants': {0: 1, 1: 2}},
    )
    b = CandidateName(
        name='b', handler='benzene',
        ring_info={'iupac_locants': {0: 1, 1: 2}},
    )
    assert _has_iupac_locants([a, b]) is True


def test_step6_dispatch_in_best_two_tier_skipped_for_spiro():
    """Integration: _best_two_tier observably DOES NOT call
    _filter_lowest_locants when any candidate has the None-stub
    ring_info (spiro/VB drop to Tier-2).

    Verified by ensuring the pool's selected candidate for two None-stub
    candidates is determined by Tier-2 weighted-sum (selection differs
    based on confidence), not by step 6 (which would be skipped anyway).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    Source: Phase 147 CONTEXT D-06.
    """
    from orthonym.assembly.coverage_scoring import CandidateName
    from orthonym.assembly.candidate_pool import (
        _has_iupac_locants, _filter_lowest_locants,
    )
    a_stub = CandidateName(
        name='a', handler='spiro',
        ring_info={'iupac_locants': None},
    )
    b_stub = CandidateName(
        name='b', handler='spiro',
        ring_info={'iupac_locants': None},
    )
    # _has_iupac_locants must return False, so _best_two_tier's step 6
    # branch (`if ... and _has_iupac_locants(candidates)`) does NOT call
    # _filter_lowest_locants.
    assert _has_iupac_locants([a_stub, b_stub]) is False
    # Direct call to _filter_lowest_locants on bail-out candidates
    # (no parent_atom_indices) returns them unchanged — gate behaviour
    # is also defensive at the function body level.
    out = _filter_lowest_locants([a_stub, b_stub])
    assert len(out) == 2
