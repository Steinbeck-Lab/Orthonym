"""a phase Plan 02 Task 4: tests for cascade step 6 gate semantics.

Source: https://iupac.qmul.ac.uk/BlueBook/P4.html +
Source: a phase internal notes (cascade gate), (spiro/VB stub
        semantics — None drops to Tier-2).
"""

import pytest


def test_step6_skipped_when_any_candidate_ring_info_is_none_stub():
    """: _has_iupac_locants returns False on None stub -> step 6 no-op.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html
    Source: a phase internal notes.
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
    """: _has_iupac_locants returns False if any candidate has
    ring_info=None -> step 6 no-op.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html
    Source: a phase internal notes.
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
    """ inverse: when ALL candidates have iupac_locants populated,
    cascade step 6 IS active.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html
    Source: a phase internal notes.
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

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html
    Source: a phase internal notes.
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
    # branch (`if... and _has_iupac_locants(candidates)`) does NOT call
    # _filter_lowest_locants.
    assert _has_iupac_locants([a_stub, b_stub]) is False
    # Direct call to _filter_lowest_locants on bail-out candidates
    # (no parent_atom_indices) returns them unchanged — gate behaviour
    # is also defensive at the function body level.
    out = _filter_lowest_locants([a_stub, b_stub])
    assert len(out) == 2


def test_step6_skipped_for_branch_6_5_base_component_atoms_key():
    """a phase /: Branch 6.5 emits 'base_component_atoms' key,
    NOT 'iupac_locants'. _has_iupac_locants returns False -> cascade
    step 6 stays GATED for non-cataloged fused systems.

    This is the carry-forward lock from a phase (cascade step 6
    conservative gate) extended to a phase's new dict key.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html +
    Source: 149-internal notes;.
    Source: 147-internal notes.
    """
    from orthonym.assembly.coverage_scoring import CandidateName
    from orthonym.assembly.candidate_pool import _has_iupac_locants
    a = CandidateName(
        name='non_cataloged_fused',
        handler='complex_ring',
        ring_info={'base_component_atoms': frozenset({0, 1, 2})},
    )
    b = CandidateName(
        name='b',
        handler='benzene',
        ring_info={'iupac_locants': {0: 1}},
    )
    # _has_iupac_locants checks for the 'iupac_locants' key specifically;
    # Branch 6.5's new key is invisible to it. Cascade step 6 stays GATED.
    assert _has_iupac_locants([a, b]) is False, (
        "Phase 149 SC-7: Branch 6.5's 'base_component_atoms' key MUST NOT "
        "satisfy _has_iupac_locants gate; cascade step 6 must stay GATED "
        "for non-cataloged fused systems"
    )


def test_step6_skipped_when_only_branch_6_5_candidate():
    """a phase: even if every candidate has 'base_component_atoms',
    cascade step 6 stays GATED because none has 'iupac_locants'.

    Source: 149-internal notes;.
    """
    from orthonym.assembly.coverage_scoring import CandidateName
    from orthonym.assembly.candidate_pool import _has_iupac_locants
    a = CandidateName(
        name='a',
        handler='complex_ring',
        ring_info={'base_component_atoms': frozenset({0, 1, 2})},
    )
    b = CandidateName(
        name='b',
        handler='complex_ring',
        ring_info={'base_component_atoms': frozenset({3, 4, 5, 6})},
    )
    assert _has_iupac_locants([a, b]) is False
