"""WSD-06 (NUM-01) — substituted cycloalkene keeps its ring double-bond +
substituent locants (no over-elision).

Wave-0 scaffold (Phase 175). TARGET + de-collision assertions are xfail until the
WSD-06 code plan (175-07) replaces the `num_double==1` count proxy in
`should_omit_locant_one` with one real `is_only_one_substitutable_position`
topological-symmetry predicate (shared by Rules 4/5/6). The SYMMETRY CONTROLS are
non-xfail regression guards (genuinely symmetric rings must keep eliding locant 1).

Blue Book P-14.3.4.2(d)/P-14.4(e): the ene-locant is omitted ONLY for unsubstituted
cycloalkenes; the verbatim PIN for the substituted case is `3-bromocyclohex-1-ene`.
"""

import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestCycloalkeneLocant:
    @pytest.mark.xfail(reason="WSD-06 fix lands in Plan 175-07", strict=False)
    def test_bromocyclohexene_keeps_locants(self):
        assert name_compound("BrC1CCCC=C1").strip().lower() == "3-bromocyclohex-1-ene"

    @pytest.mark.xfail(reason="WSD-06 fix lands in Plan 175-07", strict=False)
    def test_distinct_isomers_do_not_collide(self):
        # Distinct molecules must NOT share a name. Today both -> 'methylcyclohexene'.
        a = name_compound("CC1CC=CCC1").strip().lower()   # 4-methylcyclohex-1-ene
        b = name_compound("CC1=CCCCC1").strip().lower()    # 1-methylcyclohex-1-ene
        assert a != b, f"distinct cycloalkenes collide on one name: {a!r}"

    def test_symmetric_saturated_ring_control(self):
        # CONTROL (NOT xfail): genuinely symmetric ring keeps eliding locant 1.
        assert name_compound("CC1CCCCC1").strip().lower() == "methylcyclohexane"

    def test_unsubstituted_cyclohexene_control(self):
        # CONTROL (NOT xfail): unsubstituted cycloalkene keeps eliding the ene-locant.
        assert name_compound("C1=CCCCC1").strip().lower() == "cyclohexene"
