"""A nested whole naming memoised for the naming call is keyed on the state of that call (``nested_memo.renaming_state``).

``name_pipeline_only``, ``name_compound`` re-entered, the peptide handler's substitutive attempt, the substitutive
name of a peptide acid fragment and ``name_substituent`` each name a structure again inside a naming call, and the
name depends on the state that call runs in: the tier flags, decision A's N-acyl float scope, and the peptide handler's
attempt (``routing.dispatch_table._PEPTIDE_SUBST_ACTIVE`` / ``_PEPTIDE_NAME_STANDS``), which their keys left out. A
value named outside the attempt was then served inside it ('(2-amino-3-hydroxypropanamido)acetic acid' where the
attempt names '2-amino-3-hydroxypropanoylglycine'), and ``ORTHONYM_MEMO=verify`` found such hits on 26 of 360 panel
rows. The keys now carry ``nested_memo.renaming_state``, so a hit is served only in the state it was named
in.
"""
import pytest

from orthonym.assembly import memo
from orthonym.namer import name_pipeline_only
from orthonym.routing import dispatch_table as dt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

SER_GLY = "NC(CO)C(=O)NCC(=O)O"


def _fresh(attempt: bool, monkeypatch) -> str:
    monkeypatch.setattr(dt, "_PEPTIDE_SUBST_ACTIVE", attempt)
    return name_pipeline_only(SER_GLY)


@pytest.mark.parametrize("attempt_first", [False, True])
def test_one_scope_serves_each_state_its_own_name(monkeypatch, attempt_first):
    outside, inside = _fresh(False, monkeypatch), _fresh(True, monkeypatch)
    assert outside != inside          # the state changes the name: the key must carry it
    tok = memo.push_scope()
    try:
        for attempt in ((True, False) if attempt_first else (False, True)):
            assert _fresh(attempt, monkeypatch) == (inside if attempt else outside)
    finally:
        memo.pop_scope(tok)
