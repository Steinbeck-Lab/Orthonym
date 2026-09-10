""" wiring: select_best_candidate resolves full ties alphanumerically.

BB (the Blue Book): the PIN is the name earlier in
alphanumerical order — replaces the first-wins list-order fallback.
"""
from orthonym.assembly.coverage_scoring import (
    CandidateName,
    select_best_candidate,
)


def _cand(name, handler="benzene", confidence=0.8):
    return CandidateName(name=name, handler=handler, confidence=confidence,
                         factors={})


class TestSelectBestAlphanumericTie:
    def test_p455_tie_break_replaces_first_wins(self):
        late = _cand("2,4-dibromo-N-(2-bromo-4-chlorophenyl)aniline")
        early = _cand("2-bromo-4-chloro-N-(2,4-dibromophenyl)aniline")
        # Same confidence, same handler priority — decides, both orders.
        assert select_best_candidate([late, early]).name == early.name
        assert select_best_candidate([early, late]).name == early.name

    def test_confidence_still_dominates(self):
        weak_early = _cand("aaa-name", confidence=0.5)
        strong_late = _cand("zzz-name", confidence=0.9)
        assert select_best_candidate([weak_early, strong_late]).name == "zzz-name"

    def test_handler_priority_still_dominates(self):
        # HANDLER_PRIORITY['chain']=1 < ring handlers; alphanumerics must NOT
        # override the handler tier (they only break full ties).
        chain = _cand("aaa-name", handler="chain")
        ring = _cand("zzz-name", handler="complex_ring")
        assert select_best_candidate([chain, ring]).name == "zzz-name"
